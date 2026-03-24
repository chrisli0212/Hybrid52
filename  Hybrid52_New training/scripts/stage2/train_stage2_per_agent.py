#!/usr/bin/env python3
"""
Stage 2: Per-Agent Cross-Symbol Fusion (standard agents A/B/C/K/T/Q)

For each agent type, trains one CrossSymbolAgentFusion model that combines:
  - Frozen Stage 1 logits+probs from all 5 symbols (same agent type)
  - 4 cross-diffs (SPXW logit minus each peer logit)
  - 2 chain context dims (from precomputed SPXW Stage1 2D output)
  = 16-dim input → MLP → refined SPXW binary prediction

Prerequisites:
  1. Stage 1 checkpoints for all 5 symbols per agent type
  2. Chain context file from precompute_chain_context.py

Usage:
    python scripts/stage2/train_stage2_per_agent.py --target SPXW --horizon 30
    python scripts/stage2/train_stage2_per_agent.py --target SPXW --horizon 30 --agents A B K
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from hybrid51_utils import ArtifactPaths
from hybrid51_utils.artifacts import DEFAULT_TRAINING_HORIZON_MINUTES
from hybrid51_models.independent_agent import IndependentAgent
from hybrid51_models.cross_symbol_agent_fusion import CrossSymbolAgentFusion
from hybrid51_models.tlt_gated_agent_fusion import TLTGatedAgentFusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATHS = ArtifactPaths.default()
DEFAULT_PEER_SYMBOLS = ['SPY', 'QQQ', 'IWM']
STANDARD_AGENTS = ['A', 'B', 'C', 'K', 'T', 'Q']


def _standard_input_dim(peer_symbols: list[str]) -> int:
    return 2 + (2 * len(peer_symbols)) + len(peer_symbols) + 2


def _tlt_gated_input_dim(peer_symbols: list[str], tlt_ctx_dim: int = 2) -> int:
    return _standard_input_dim(peer_symbols) + tlt_ctx_dim


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.52, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * focal_weight * bce).mean()


class BinaryIndependentAgent(nn.Module):
    def __init__(self, agent_type, feat_dim=325, temporal_dim=128, dropout=0.2,
                 use_feature_subset=True, use_attention_backbone=False,
                 use_attention_pool=False, cls_input_dim=None):
        super().__init__()
        self.base = IndependentAgent(
            agent_type=agent_type, feat_dim=feat_dim, temporal_dim=temporal_dim,
            dropout=dropout, num_classes=5, use_feature_subset=use_feature_subset,
            use_attention_backbone=use_attention_backbone, use_attention_pool=use_attention_pool,
        )
        if cls_input_dim is None:
            cls_input_dim = (2 + temporal_dim) if self.base.use_backbone else (2 + 32)
        self.base.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, sequences, chain_2d=None):
        return self.base(sequences, chain_2d=chain_2d).squeeze(-1)


def _build_model_from_ckpt(ckpt, agent_type: str, device: torch.device, symbol: str):
    """Reconstruct BinaryIndependentAgent matching saved checkpoint exactly."""
    state = ckpt['model_state_dict']
    if 'feat_dim' in ckpt:
        feat_dim = int(ckpt['feat_dim'])
    elif 'base._feat_idx' in state:
        # Infer from max index: v5 standardised all symbols to 325 dims;
        # if max index < 325 then feat_dim was 325, otherwise 650.
        feat_dim = 650 if int(state['base._feat_idx'].max().item()) >= 325 else 325
    else:
        feat_dim = 325 if symbol == 'SPXW' else 650
    use_subset = bool(ckpt.get('feature_subset', True))
    use_attn_bb   = bool(ckpt.get('use_attention_backbone', False))
    use_attn_pool = bool(ckpt.get('use_attention_pool', False))
    cls_in_dim = int(state['base.classifier.0.weight'].shape[1])
    has_static_proj = 'base.static_proj.weight' in state
    model = BinaryIndependentAgent(
        agent_type=agent_type, feat_dim=feat_dim,
        use_feature_subset=use_subset,
        use_attention_backbone=use_attn_bb,
        use_attention_pool=use_attn_pool,
        cls_input_dim=cls_in_dim,
    ).to(device)
    if not has_static_proj and hasattr(model.base, 'static_proj'):
        del model.base.static_proj
    model.load_state_dict(state, strict=True)
    return model


def _load_stage1_model(symbol: str, agent: str, horizon: int, device: torch.device):
    ckpt_path = PATHS.stage1_ckpt(symbol, agent, horizon)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage1 ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = _build_model_from_ckpt(ckpt, agent, device, symbol)
    model.eval()
    feat_dim = 325 if symbol == 'SPXW' else 650
    feat_dim = int(ckpt.get('feat_dim', feat_dim))
    threshold = float(ckpt.get('optimal_threshold', 0.5))
    invert_signal = bool(ckpt.get('invert_signal', False))
    return model, feat_dim, threshold, invert_signal


@torch.no_grad()
def _infer_logits_probs(model, sequences, norm_mean, norm_std, device, batch_size=2048, invert_signal=False):
    n = len(sequences)
    logits_all = []
    norm_mean_t = None
    norm_std_t = None
    if norm_mean is not None and norm_std is not None:
        norm_mean_t = torch.from_numpy(norm_mean).float().to(device)
        norm_std_t = torch.from_numpy(norm_std).float().to(device)

    for i in range(0, n, batch_size):
        seq_np = np.asarray(sequences[i:i + batch_size], dtype=np.float32)
        if not seq_np.flags.writeable:
            seq_np = seq_np.copy()
        seq_b = torch.from_numpy(seq_np).to(device, non_blocking=True)
        if norm_mean_t is not None and norm_std_t is not None:
            seq_b = (seq_b - norm_mean_t) / norm_std_t
        logits_all.append(model(seq_b).cpu().numpy().astype(np.float32))
    logits = np.concatenate(logits_all, axis=0)
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    if invert_signal:
        logits = -logits
        probs = 1.0 - probs
    return logits, probs


def _load_split(symbol: str, horizon: int, split: str):
    data_dir = PATHS.tier3_dir(symbol, horizon)
    seq = np.load(data_dir / f'{split}_sequences.npy', mmap_mode='r')
    labels = np.load(data_dir / f'{split}_labels.npy')
    norm_mean = np.load(data_dir / 'norm_mean.npy') if (data_dir / 'norm_mean.npy').exists() else None
    norm_std = np.load(data_dir / 'norm_std.npy') if (data_dir / 'norm_std.npy').exists() else None
    ts_path = data_dir / f'{split}_timestamps.npy'
    if ts_path.exists():
        timestamps = np.load(ts_path).astype(np.int64)
    else:
        timestamps = np.arange(len(labels), dtype=np.int64)
    return seq, labels, norm_mean, norm_std, timestamps


def _align_timestamps(base_ts: np.ndarray, peer_ts_map: dict[str, np.ndarray], context_ts: np.ndarray | None = None):
    """Return base indices and per-key indices aligned on SPXW timestamps."""
    keep = np.ones(len(base_ts), dtype=bool)
    ts_maps = {}
    for key, ts in peer_ts_map.items():
        ts_maps[key] = {int(t): i for i, t in enumerate(ts)}
        keep &= np.isin(base_ts, ts)
    context_map = None
    if context_ts is not None:
        context_map = {int(t): i for i, t in enumerate(context_ts)}
        keep &= np.isin(base_ts, context_ts)

    base_idx = np.flatnonzero(keep)
    aligned_ts = base_ts[base_idx]
    peer_indices = {
        key: np.array([ts_maps[key][int(t)] for t in aligned_ts], dtype=np.int64)
        for key in peer_ts_map
    }
    ctx_idx = None
    if context_map is not None:
        ctx_idx = np.array([context_map[int(t)] for t in aligned_ts], dtype=np.int64)
    return base_idx, peer_indices, ctx_idx, aligned_ts


def _build_design_matrix(agent: str, horizon: int, split: str, device: torch.device,
                         stage1_models: dict, chain_context: dict, peer_symbols: list[str],
                         batch_size: int = 2048):
    """Build the cross-symbol input for one split."""
    logits_per_sym = {}
    probs_per_sym = {}
    split_labels = None
    ts_per_sym = {}
    all_symbols = ['SPXW'] + list(peer_symbols)

    for sym in all_symbols:
        t_sym = time.time()
        seq, labels, norm_mean, norm_std, split_ts = _load_split(sym, horizon, split)
        if split_labels is None:
            split_labels = labels

        model, invert_signal = stage1_models[sym]
        lg, pb = _infer_logits_probs(
            model, seq, norm_mean, norm_std, device, batch_size, invert_signal=invert_signal
        )
        logits_per_sym[sym] = lg
        probs_per_sym[sym] = pb
        ts_per_sym[sym] = split_ts
        logger.info(f"    [{split}] {sym}: n={len(labels):,} done in {time.time() - t_sym:.1f}s")

    if all(len(ts_per_sym[s]) > 0 for s in all_symbols) and f'{split}_timestamps' in chain_context:
        ctx_ts = np.asarray(chain_context[f'{split}_timestamps']).astype(np.int64)
        base_idx, peer_idx, ctx_idx, aligned_ts = _align_timestamps(
            ts_per_sym['SPXW'], {s: ts_per_sym[s] for s in peer_symbols}, context_ts=ctx_ts
        )
        logger.info(
            f"    [{split}] timestamp-align kept {len(base_idx):,}/{len(ts_per_sym['SPXW']):,} "
            f"({(len(base_idx) / max(1, len(ts_per_sym['SPXW']))):.1%})"
        )
        split_labels = split_labels[base_idx]
        logit_spxw = logits_per_sym['SPXW'][base_idx]
        probs_spxw = probs_per_sym['SPXW'][base_idx]
    else:
        n = min(len(logits_per_sym[s]) for s in all_symbols)
        logger.warning(f"    [{split}] timestamps unavailable -> falling back to min-length alignment n={n:,}")
        split_labels = split_labels[:n]
        logit_spxw = logits_per_sym['SPXW'][:n]
        probs_spxw = probs_per_sym['SPXW'][:n]
        peer_idx = {s: np.arange(n, dtype=np.int64) for s in peer_symbols}
        ctx_idx = np.arange(n, dtype=np.int64)
        aligned_ts = ts_per_sym['SPXW'][:n]

    parts = [logit_spxw.reshape(-1, 1), probs_spxw.reshape(-1, 1)]
    for sym in peer_symbols:
        idx = peer_idx[sym]
        parts.append(logits_per_sym[sym][idx].reshape(-1, 1))
        parts.append(probs_per_sym[sym][idx].reshape(-1, 1))

    for sym in peer_symbols:
        idx = peer_idx[sym]
        diff = logit_spxw - logits_per_sym[sym][idx]
        parts.append(diff.reshape(-1, 1))

    ctx_logits = np.asarray(chain_context[f'{split}_logits'])[ctx_idx].reshape(-1, 1)
    ctx_probs = np.asarray(chain_context[f'{split}_probs'])[ctx_idx].reshape(-1, 1)
    parts.extend([ctx_logits, ctx_probs])

    X = np.concatenate(parts, axis=1).astype(np.float32)
    y = split_labels.astype(np.float32)
    return X, y, aligned_ts


@torch.no_grad()
def _build_tlt_context(agent: str, horizon: int, split: str, device: torch.device,
                       tlt_model, batch_size: int = 2048) -> np.ndarray:
    """Run frozen TLT Stage1 model and return (n, 2) array of [logit, prob]."""
    seq, _, norm_mean, norm_std, ts = _load_split('TLT', horizon, split)
    tlt_logits, tlt_probs = _infer_logits_probs(
        tlt_model, seq, norm_mean, norm_std, device, batch_size, invert_signal=False
    )
    return np.stack([tlt_logits, tlt_probs], axis=1).astype(np.float32), ts  # (n, 2), (n,)


def _build_design_matrix_tlt(agent: str, horizon: int, split: str, device: torch.device,
                              stage1_models: dict, chain_context: dict, peer_symbols: list[str],
                              tlt_model, batch_size: int = 2048):
    """Traditional design matrix with TLT context appended as last 2 cols."""
    X, y, aligned_ts = _build_design_matrix(agent, horizon, split, device, stage1_models,
                                            chain_context, peer_symbols, batch_size)
    tlt_ctx, tlt_ts = _build_tlt_context(agent, horizon, split, device, tlt_model, batch_size)
    tlt_map = {int(t): i for i, t in enumerate(tlt_ts)}
    keep = np.array([int(t) in tlt_map for t in aligned_ts], dtype=bool)
    kept_ts = aligned_ts[keep]
    tlt_idx = np.array([tlt_map[int(t)] for t in kept_ts], dtype=np.int64)
    X = np.concatenate([X[keep], tlt_ctx[tlt_idx]], axis=1).astype(np.float32)
    y = y[keep]
    logger.info(f"    [{split}] tlt-context align kept {len(y):,}/{len(aligned_ts):,}")
    return X, y


def _sweep_threshold(probs, labels):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.30, 0.66, 0.01):
        f1 = f1_score(labels, (probs > thr).astype(int), average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1


def _metrics(labels, probs, threshold):
    preds = (probs > threshold).astype(int)
    out = {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1': float(f1_score(labels, preds, average='binary', zero_division=0)),
        'brier': float(brier_score_loss(labels, probs)),
        'threshold': float(threshold),
    }
    try:
        out['auc'] = float(roc_auc_score(labels, probs))
    except Exception:
        out['auc'] = 0.5
    try:
        out['ic'] = float(spearmanr(probs, labels).statistic)
    except Exception:
        out['ic'] = 0.0
    return out


def train_agent_fusion(agent: str, target: str, horizon: int, device: torch.device,
                       epochs: int, batch_size: int, lr: float, stage1_batch: int,
                       peer_symbols: list[str]):
    logger.info(f"\n{'='*60}")
    logger.info(f"Agent {agent}: cross-symbol fusion | target={target} h={horizon}")
    logger.info(f"{'='*60}")
    all_symbols = [target] + list(peer_symbols)
    n_standard_inputs = _standard_input_dim(peer_symbols)

    stage1_models = {}
    for sym in all_symbols:
        try:
            model, _, _, invert_signal = _load_stage1_model(sym, agent, horizon, device)
            stage1_models[sym] = (model, invert_signal)
            logger.info(f"  Loaded Stage1 {sym} Agent {agent} (invert={invert_signal})")
        except FileNotFoundError as e:
            logger.warning(f"  Missing: {e} — skipping agent {agent}")
            return None

    chain_ctx_path = PATHS.stage2_chain_context(target, horizon)
    if not chain_ctx_path.exists():
        raise FileNotFoundError(
            f"Chain context not found: {chain_ctx_path}\n"
            f"Run: python scripts/stage2/precompute_chain_context.py --symbol {target} --horizon {horizon}"
        )
    chain_context = dict(np.load(chain_ctx_path))
    logger.info(f"  Loaded chain context from {chain_ctx_path}")

    logger.info("  Building design matrices...")
    X_train, y_train, _ = _build_design_matrix(agent, horizon, 'train', device, stage1_models, chain_context, peer_symbols, stage1_batch)
    X_val,   y_val, _   = _build_design_matrix(agent, horizon, 'val',   device, stage1_models, chain_context, peer_symbols, stage1_batch)
    X_test,  y_test, _  = _build_design_matrix(agent, horizon, 'test',  device, stage1_models, chain_context, peer_symbols, stage1_batch)
    logger.info(f"  X_train={X_train.shape} X_val={X_val.shape} X_test={X_test.shape}")

    assert X_train.shape[1] == n_standard_inputs, f"Expected {n_standard_inputs} dims, got {X_train.shape[1]}"

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True,
    )

    model = CrossSymbolAgentFusion(n_inputs=n_standard_inputs, hidden_dim=32, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    up_frac = float(np.mean(y_train))
    criterion = BinaryFocalLoss(gamma=2.0, alpha=up_frac, label_smoothing=0.05)

    X_val_t = torch.from_numpy(X_val).to(device)
    best_state = None
    best_val_auc = -1.0
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy().astype(np.float32)
        val_probs = (1.0 / (1.0 + np.exp(-val_logits))).astype(np.float32)
        try:
            val_auc = float(roc_auc_score(y_val.astype(int), val_probs))
        except Exception:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(1, len(train_loader))
            logger.info(f"  Ep {epoch+1:3d}: loss={avg_loss:.4f} val_auc={val_auc:.4f} (best={best_val_auc:.4f})")

        if patience_counter >= patience:
            logger.info(f"  Early stop at epoch {epoch+1} (best val_auc={best_val_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy().astype(np.float32)
        test_logits = model(torch.from_numpy(X_test).to(device)).cpu().numpy().astype(np.float32)

    val_probs = (1.0 / (1.0 + np.exp(-val_logits))).astype(np.float32)
    test_probs = (1.0 / (1.0 + np.exp(-test_logits))).astype(np.float32)

    opt_thr, opt_val_f1 = _sweep_threshold(val_probs, y_val.astype(int))
    val_metrics = _metrics(y_val.astype(int), val_probs, opt_thr)
    test_metrics = _metrics(y_test.astype(int), test_probs, opt_thr)

    logger.info(f"  Val:  acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}")
    logger.info(f"  Test: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f} ic={test_metrics['ic']:.4f}")

    PATHS.stage2_cross_results.mkdir(parents=True, exist_ok=True)

    ckpt_path = PATHS.stage2_per_agent_ckpt(target, agent, horizon)
    torch.save({
        'model_state_dict': model.state_dict(),
        'target': target, 'agent': agent, 'horizon': horizon,
        'n_inputs': n_standard_inputs,
        'symbols': all_symbols,
        'optimal_threshold': opt_thr,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }, ckpt_path)

    probs_path = PATHS.stage2_per_agent_probs(target, agent, horizon)
    np.savez(probs_path,
             val_probs=val_probs, val_labels=y_val.astype(np.int64),
             test_probs=test_probs, test_labels=y_test.astype(np.int64))

    logger.info(f"  Saved: {ckpt_path}")
    return test_metrics


def train_agent_fusion_tlt_gated(agent: str, target: str, horizon: int, device: torch.device,
                                  epochs: int, batch_size: int, lr: float, stage1_batch: int,
                                  peer_symbols: list[str]):
    """Train TLT-gated Stage 2 per-agent fusion (learned peer-trust via TLT macro gates)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Agent {agent}: TLT-GATED cross-symbol fusion | target={target} h={horizon}")
    logger.info(f"{'='*60}")
    all_symbols = [target] + list(peer_symbols)
    n_inputs = _tlt_gated_input_dim(peer_symbols, tlt_ctx_dim=2)

    stage1_models = {}
    for sym in all_symbols:
        try:
            model, _, _, invert_signal = _load_stage1_model(sym, agent, horizon, device)
            stage1_models[sym] = (model, invert_signal)
            logger.info(f"  Loaded Stage1 {sym} Agent {agent} (invert={invert_signal})")
        except FileNotFoundError as e:
            logger.warning(f"  Missing: {e} — skipping agent {agent}")
            return None

    try:
        tlt_model, _, _, tlt_invert = _load_stage1_model('TLT', agent, horizon, device)
        if tlt_invert:
            logger.warning("  TLT Stage1 invert flag detected; tlt_gated context remains unflipped by design")
        logger.info(f"  Loaded frozen TLT Stage1 Agent {agent} for gating context")
    except FileNotFoundError as e:
        logger.warning(f"  TLT Stage1 ckpt missing: {e} — skipping tlt_gated for agent {agent}")
        return None

    chain_ctx_path = PATHS.stage2_chain_context(target, horizon)
    if not chain_ctx_path.exists():
        raise FileNotFoundError(
            f"Chain context not found: {chain_ctx_path}\n"
            f"Run: python scripts/stage2/precompute_chain_context.py --symbol {target} --horizon {horizon}"
        )
    chain_context = dict(np.load(chain_ctx_path))
    logger.info(f"  Loaded chain context from {chain_ctx_path}")

    logger.info("  Building TLT-gated design matrices...")
    X_train, y_train = _build_design_matrix_tlt(agent, horizon, 'train', device, stage1_models,
                                                  chain_context, peer_symbols, tlt_model, stage1_batch)
    X_val,   y_val   = _build_design_matrix_tlt(agent, horizon, 'val',   device, stage1_models,
                                                  chain_context, peer_symbols, tlt_model, stage1_batch)
    X_test,  y_test  = _build_design_matrix_tlt(agent, horizon, 'test',  device, stage1_models,
                                                  chain_context, peer_symbols, tlt_model, stage1_batch)
    logger.info(f"  X_train={X_train.shape} X_val={X_val.shape} X_test={X_test.shape}")
    assert X_train.shape[1] == n_inputs, f"Expected {n_inputs} dims, got {X_train.shape[1]}"

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True,
    )

    model = TLTGatedAgentFusion(
        n_peers=len(peer_symbols), tlt_ctx_dim=2, tlt_emb_dim=16,
        gate_hidden_dim=8, fusion_hidden_dim=32, dropout=0.2,
    ).to(device)
    logger.info(f"  TLTGatedAgentFusion: {model.count_parameters():,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    up_frac = float(np.mean(y_train))
    criterion = BinaryFocalLoss(gamma=2.0, alpha=up_frac, label_smoothing=0.05)

    X_val_t = torch.from_numpy(X_val).to(device)
    best_state = None
    best_val_auc = -1.0
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits, val_gates = model(X_val_t)
        val_logits = val_logits.cpu().numpy().astype(np.float32)
        val_probs = (1.0 / (1.0 + np.exp(-val_logits))).astype(np.float32)
        try:
            val_auc = float(roc_auc_score(y_val.astype(int), val_probs))
        except Exception:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(1, len(train_loader))
            gate_summary = TLTGatedAgentFusion.gate_summary(val_gates, peer_symbols)
            gate_str = '  '.join(f"{s}={v:.3f}" for s, v in gate_summary.items())
            logger.info(f"  Ep {epoch+1:3d}: loss={avg_loss:.4f} val_auc={val_auc:.4f} (best={best_val_auc:.4f})  gates: {gate_str}")

        if patience_counter >= patience:
            logger.info(f"  Early stop at epoch {epoch+1} (best val_auc={best_val_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    X_test_t = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        val_logits,  val_gates  = model(X_val_t)
        test_logits, test_gates = model(X_test_t)

    val_logits  = val_logits.cpu().numpy().astype(np.float32)
    test_logits = test_logits.cpu().numpy().astype(np.float32)
    val_probs   = (1.0 / (1.0 + np.exp(-val_logits))).astype(np.float32)
    test_probs  = (1.0 / (1.0 + np.exp(-test_logits))).astype(np.float32)

    opt_thr, _ = _sweep_threshold(val_probs, y_val.astype(int))
    val_metrics  = _metrics(y_val.astype(int),  val_probs,  opt_thr)
    test_metrics = _metrics(y_test.astype(int), test_probs, opt_thr)

    final_gate_summary = TLTGatedAgentFusion.gate_summary(test_gates, peer_symbols)
    logger.info(f"  Val:  acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}")
    logger.info(f"  Test: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f} ic={test_metrics['ic']:.4f}")
    logger.info(f"  Test gate means: {final_gate_summary}")

    PATHS.stage2_cross_results.mkdir(parents=True, exist_ok=True)

    ckpt_path = PATHS.stage2_tlt_gated_ckpt(target, agent, horizon)
    torch.save({
        'model_state_dict': model.state_dict(),
        'target': target, 'agent': agent, 'horizon': horizon,
        'n_inputs': n_inputs, 'n_peers': len(peer_symbols),
        'peer_symbols': list(peer_symbols),
        'symbols': all_symbols,
        'tlt_ctx_dim': 2,
        'optimal_threshold': opt_thr,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_gate_means': final_gate_summary,
    }, ckpt_path)

    probs_path = PATHS.stage2_tlt_gated_probs(target, agent, horizon)
    np.savez(probs_path,
             val_probs=val_probs, val_labels=y_val.astype(np.int64),
             test_probs=test_probs, test_labels=y_test.astype(np.int64))

    logger.info(f"  Saved: {ckpt_path}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='SPXW')
    parser.add_argument('--horizon', type=int, default=DEFAULT_TRAINING_HORIZON_MINUTES)
    parser.add_argument('--agents', nargs='+', default=STANDARD_AGENTS)
    parser.add_argument('--peers', nargs='+', default=DEFAULT_PEER_SYMBOLS)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--stage1-batch', type=int, default=2048)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--variant', choices=['traditional', 'tlt_gated'], default='traditional',
                        help='traditional: standard cross-symbol fusion; '
                             'tlt_gated: TLT-conditioned learned peer-trust gates')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}  variant: {args.variant}")
    peer_symbols = [sym.upper() for sym in args.peers if sym.upper() != args.target.upper()]
    logger.info(f"Peers: {peer_symbols}")

    train_fn = (train_agent_fusion_tlt_gated
                if args.variant == 'tlt_gated'
                else train_agent_fusion)

    all_results = {}
    for agent in args.agents:
        if agent == '2D':
            logger.warning("Agent 2D handled by train_stage2_agent_2d.py — skipping here")
            continue
        t0 = time.time()
        metrics = train_fn(
            agent=agent, target=args.target, horizon=args.horizon, device=device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            stage1_batch=args.stage1_batch, peer_symbols=peer_symbols,
        )
        if metrics:
            all_results[agent] = metrics
            logger.info(f"  Agent {agent} done in {time.time()-t0:.1f}s")

    logger.info(f"\n{'='*60}")
    logger.info(f"Summary  [{args.variant}]:")
    for agent, m in sorted(all_results.items()):
        logger.info(f"  {agent}: test_acc={m['accuracy']:.4f} auc={m['auc']:.4f} f1={m['f1']:.4f}")

    suffix = 'tlt_gated' if args.variant == 'tlt_gated' else 'per_agent'
    out_json = PATHS.stage2_cross_results / f"{args.target}_h{args.horizon}_stage2_{suffix}_results.json"
    PATHS.stage2_cross_results.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved: {out_json}")


if __name__ == '__main__':
    main()
