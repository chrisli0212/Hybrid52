#!/usr/bin/env python3
"""
Production Inference Script — Hybrid51 h30 VIX-Gated

Replicates the exact model loading and inference logic from the training scripts.

Usage:
    python simple_inference.py --split test --output predictions.npz
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

# Add parent to path
PROD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROD_ROOT))
from hybrid51_models.independent_agent import IndependentAgent
from hybrid51_models.cross_symbol_agent_fusion import CrossSymbolAgentFusion
from hybrid51_models.regime_gated_meta_model import RegimeGatedProbFusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Load config
CONFIG = json.load(open(PROD_ROOT / "config/production_config.json"))

HORIZON = CONFIG["model_info"]["horizon_minutes"]
ALL_AGENTS = CONFIG["architecture"]["stage1"]["agents"]
ALL_SYMBOLS = CONFIG["architecture"]["stage1"]["symbols"]

# h30 traditional: standard agents use SPXW + SPY/QQQ/IWM (no TLT)
STANDARD_PEER_SYMBOLS = ['SPY', 'QQQ', 'IWM']
# Agent 2D uses all 4 peers including TLT
AGENT_2D_PEER_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'TLT']


# ---------------------------------------------------------------------------
# BinaryIndependentAgent wrapper — matches training checkpoint format
# ---------------------------------------------------------------------------
class BinaryIndependentAgent(nn.Module):
    """Wraps IndependentAgent as self.base → state dict keys have 'base.' prefix."""
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


def _build_model_from_ckpt(ckpt: dict, agent_type: str, device: torch.device, symbol: str = 'SPXW'):
    """Reconstruct BinaryIndependentAgent matching saved checkpoint exactly."""
    state = ckpt['model_state_dict']

    # Infer feat_dim
    if 'feat_dim' in ckpt:
        feat_dim = int(ckpt['feat_dim'])
    elif 'base._feat_idx' in state and state['base._feat_idx'].numel() > 0:
        feat_dim = 650 if int(state['base._feat_idx'].max().item()) >= 325 else 325
    else:
        feat_dim = 325 if symbol == 'SPXW' else 650

    use_subset = bool(ckpt.get('feature_subset', True))
    use_attn_bb = bool(ckpt.get('use_attention_backbone', False))
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
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _load_split(data_root: Path, symbol: str, horizon: int, split: str):
    """Load sequences, labels, and normalization stats."""
    d = data_root / symbol / f"horizon_{horizon}min"
    seq = np.load(d / f'{split}_sequences.npy', mmap_mode='r')
    labels = np.load(d / f'{split}_labels.npy')
    nm = np.load(d / 'norm_mean.npy') if (d / 'norm_mean.npy').exists() else None
    ns = np.load(d / 'norm_std.npy') if (d / 'norm_std.npy').exists() else None
    return seq, labels, nm, ns


def _load_chain_split(data_root: Path, symbol: str, horizon: int, split: str):
    """Load sequences with chain_2d for 2D models."""
    d = data_root / symbol / f"horizon_{horizon}min"
    seq = np.load(d / f'{split}_sequences.npy', mmap_mode='r')
    labels = np.load(d / f'{split}_labels.npy')
    chain_path = d / f'{split}_chain_2d.npy'
    chain = np.load(chain_path) if chain_path.exists() else np.zeros((len(seq), 2), dtype=np.float32)
    nm = np.load(d / 'norm_mean.npy') if (d / 'norm_mean.npy').exists() else None
    ns = np.load(d / 'norm_std.npy') if (d / 'norm_std.npy').exists() else None
    return seq, chain, labels, nm, ns


# ---------------------------------------------------------------------------
# Stage 1 inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def _infer_logits_probs(model, sequences, norm_mean, norm_std, device, batch_size=1024):
    """Batched inference → (logits, probs) 1D arrays."""
    n = len(sequences)
    logits_all = []
    nm_t = torch.from_numpy(norm_mean).float().to(device) if norm_mean is not None else None
    ns_t = torch.from_numpy(norm_std).float().to(device) if norm_std is not None else None

    for i in range(0, n, batch_size):
        seq_np = np.asarray(sequences[i:i + batch_size], dtype=np.float32)
        if not seq_np.flags.writeable:
            seq_np = seq_np.copy()
        sb = torch.from_numpy(seq_np).to(device)
        if nm_t is not None and ns_t is not None:
            sb = (sb - nm_t) / ns_t
        logits_all.append(model(sb).cpu().numpy().astype(np.float32))

    logits = np.concatenate(logits_all, axis=0)
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return logits, probs


@torch.no_grad()
def _infer_frozen_2d(model, seq, chain, norm_mean, norm_std, device, batch_size=1024):
    """Batched 2D inference with chain context → (logits, probs)."""
    n = len(seq)
    logits_all = []
    for i in range(0, n, batch_size):
        sb = torch.from_numpy(np.asarray(seq[i:i+batch_size], dtype=np.float32))
        cb = torch.from_numpy(np.asarray(chain[i:i+batch_size], dtype=np.float32))
        if norm_mean is not None and norm_std is not None:
            nm_t = torch.from_numpy(norm_mean).float()
            ns_t = torch.from_numpy(norm_std).float()
            sb = (sb - nm_t) / ns_t
        sb, cb = sb.to(device), cb.to(device)
        logits_all.append(model(sb, chain_2d=cb).cpu().numpy().astype(np.float32))
    logits = np.concatenate(logits_all, axis=0)
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return logits, probs


def _resample_to_length(arr: np.ndarray, n_target: int) -> np.ndarray:
    if len(arr) == n_target:
        return arr
    idx = np.linspace(0, len(arr) - 1, n_target).astype(np.int64)
    return arr[idx]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid51 h30 VIX-Gated Production Inference")
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output', default='predictions.npz')
    parser.add_argument('--data-root', default=CONFIG["data_paths"]["tier3_binary_root"])
    parser.add_argument('--vix-root', default=CONFIG["data_paths"]["vix_features_root"])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    model_dir = PROD_ROOT / "models"
    split = args.split
    bs = args.batch_size

    # ==================================================================
    # STAGE 1: Per-Symbol Per-Agent Predictions
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: Per-Symbol Per-Agent Predictions")
    logger.info("=" * 60)

    # Standard agents use SPXW + SPY/QQQ/IWM (no TLT in design matrix)
    standard_symbols = ['SPXW'] + STANDARD_PEER_SYMBOLS
    # Agent 2D uses all symbols
    agent_2d_symbols = ['SPXW'] + AGENT_2D_PEER_SYMBOLS

    stage1_logits = {}  # {symbol: {agent: logits_1d}}
    stage1_probs = {}   # {symbol: {agent: probs_1d}}
    split_labels = None

    for symbol in ALL_SYMBOLS:
        stage1_logits[symbol] = {}
        stage1_probs[symbol] = {}

        for agent in ALL_AGENTS:
            ckpt_path = model_dir / f"stage1/{symbol}_agent{agent}.pt"
            if not ckpt_path.exists():
                logger.warning(f"  Missing: {ckpt_path}")
                continue

            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            if agent == '2D':
                model = _build_model_from_ckpt(ckpt, '2D', device, symbol)
                seq, chain, labels, nm, ns = _load_chain_split(data_root, symbol, HORIZON, split)
                lg, pb = _infer_frozen_2d(model, seq, chain, nm, ns, device, bs)
            else:
                model = _build_model_from_ckpt(ckpt, agent, device, symbol)
                seq, labels, nm, ns = _load_split(data_root, symbol, HORIZON, split)
                lg, pb = _infer_logits_probs(model, seq, nm, ns, device, bs)

            stage1_logits[symbol][agent] = lg
            stage1_probs[symbol][agent] = pb
            if symbol == 'SPXW' and split_labels is None:
                split_labels = labels
            logger.info(f"  {symbol} Agent {agent}: n={len(pb):,}")

            del model
            torch.cuda.empty_cache()

    # ==================================================================
    # STAGE 2: Cross-Symbol Fusion Per Agent
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Cross-Symbol Fusion Per Agent")
    logger.info("=" * 60)

    # Load chain context
    chain_ctx_data = dict(np.load(model_dir / "stage2/chain_context.npz"))

    stage2_probs = {}
    n_stage2 = None

    for agent in ALL_AGENTS:
        ckpt_path = model_dir / f"stage2/agent{agent}_fusion.pt"
        if not ckpt_path.exists():
            logger.warning(f"  Missing: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        n_inputs = ckpt['n_inputs']

        if agent == '2D':
            # Agent 2D: active_peers from checkpoint
            active_peers = ckpt.get('active_peers', AGENT_2D_PEER_SYMBOLS)
            sd_key = 'fusion_state_dict'

            # Align to min length
            n = min(len(stage1_logits['SPXW']['2D']),
                    *(len(stage1_logits.get(s, {}).get('2D', np.zeros(0)))
                      for s in active_peers if '2D' in stage1_logits.get(s, {})))

            spxw_logit = stage1_logits['SPXW']['2D'][:n].reshape(-1, 1)
            spxw_prob = stage1_probs['SPXW']['2D'][:n].reshape(-1, 1)

            peer_parts = []
            for sym in active_peers:
                if '2D' in stage1_logits.get(sym, {}):
                    peer_parts.append(stage1_logits[sym]['2D'][:n].reshape(-1, 1))
                    peer_parts.append(stage1_probs[sym]['2D'][:n].reshape(-1, 1))
            peer_feat = np.concatenate(peer_parts, axis=1)
            peer_logits_only = peer_feat[:, 0::2]
            diffs = spxw_logit - peer_logits_only

            X = np.concatenate([spxw_logit, spxw_prob, peer_feat, diffs], axis=1).astype(np.float32)
        else:
            # Standard agents: SPXW + SPY/QQQ/IWM (from checkpoint symbols)
            syms = ckpt.get('symbols', standard_symbols)
            peer_syms = [s for s in syms if s != 'SPXW']
            sd_key = 'model_state_dict'

            # Align to min length
            n = min(len(stage1_logits[s][agent]) for s in syms if agent in stage1_logits.get(s, {}))
            ctx_logits = chain_ctx_data[f'{split}_logits']
            ctx_probs = chain_ctx_data[f'{split}_probs']
            n = min(n, len(ctx_logits))

            # Design matrix: SPXW logit+prob, peer logit+prob, diffs, chain context
            spxw_logit = stage1_logits['SPXW'][agent][:n]
            parts = [spxw_logit.reshape(-1, 1), stage1_probs['SPXW'][agent][:n].reshape(-1, 1)]
            for sym in peer_syms:
                parts.append(stage1_logits[sym][agent][:n].reshape(-1, 1))
                parts.append(stage1_probs[sym][agent][:n].reshape(-1, 1))
            for sym in peer_syms:
                diff = spxw_logit - stage1_logits[sym][agent][:n]
                parts.append(diff.reshape(-1, 1))
            parts.append(ctx_logits[:n].reshape(-1, 1))
            parts.append(ctx_probs[:n].reshape(-1, 1))

            X = np.concatenate(parts, axis=1).astype(np.float32)

        assert X.shape[1] == n_inputs, \
            f"Agent {agent}: expected {n_inputs} inputs, got {X.shape[1]}"

        # Load fusion model
        fusion = CrossSymbolAgentFusion(n_inputs=n_inputs, hidden_dim=32, dropout=0.2).to(device)
        fusion.load_state_dict(ckpt[sd_key], strict=True)
        fusion.eval()

        with torch.no_grad():
            logits = fusion(torch.from_numpy(X).to(device)).cpu().numpy()
            probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

        stage2_probs[agent] = probs
        if n_stage2 is None:
            n_stage2 = n
        else:
            n_stage2 = min(n_stage2, n)
        logger.info(f"  Agent {agent}: n={len(probs):,} n_inputs={n_inputs}")

    # ==================================================================
    # STAGE 3: VIX-Gated Meta-Ensemble
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: VIX-Gated Meta-Ensemble")
    logger.info("=" * 60)

    # Align all agent probs to min length
    n_final = min(n_stage2, *(len(stage2_probs[a]) for a in ALL_AGENTS if a in stage2_probs))

    # Stack agent probs in canonical order
    agent_cols = []
    for ag in ALL_AGENTS:
        if ag in stage2_probs:
            agent_cols.append(stage2_probs[ag][:n_final].reshape(-1, 1))
        else:
            agent_cols.append(np.full((n_final, 1), 0.5, dtype=np.float32))
    agent_mat = np.concatenate(agent_cols, axis=1).astype(np.float32)

    # Load Stage 3 LogReg model
    model3 = joblib.load(model_dir / "stage3/stage3_logreg.joblib")
    threshold = 0.36

    # Build 13-dim meta features for each timestep
    _mean = agent_mat.mean(axis=1, keepdims=True)
    _std  = agent_mat.std(axis=1, keepdims=True)
    _spread = (agent_mat.max(axis=1, keepdims=True) - agent_mat.min(axis=1, keepdims=True))
    _majority = (agent_mat > 0.5).mean(axis=1, keepdims=True).astype(np.float32)
    _max = agent_mat.max(axis=1, keepdims=True)
    _min = agent_mat.min(axis=1, keepdims=True)
    meta_feat = np.concatenate([agent_mat, _mean, _std, _spread, _majority, _max, _min], axis=1)

    final_probs = model3.predict_proba(meta_feat)[:, 1].astype(np.float32)
    gates_np = np.ones((n_final, len(ALL_AGENTS)), dtype=np.float32)
    preds = (final_probs > threshold).astype(np.int64)
    final_labels = split_labels[:n_final] if split_labels is not None else np.zeros(n_final, dtype=np.int64)

    # Metrics
    acc = float((preds == final_labels).mean())
    tp = float(((preds == 1) & (final_labels == 1)).sum())
    fp = float(((preds == 1) & (final_labels == 0)).sum())
    fn = float(((preds == 0) & (final_labels == 1)).sum())
    f1 = float((2 * tp) / max(1.0, 2 * tp + fp + fn))

    logger.info(f"\n  Final: n={n_final:,}  threshold={threshold:.3f}")
    logger.info(f"  accuracy={acc:.4f}  f1={f1:.4f}")

    # Gate analysis
    for i, ag in enumerate(ALL_AGENTS):
        logger.info(f"  Gate {ag}: mean={gates_np[:, i].mean():.3f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        probs=final_probs,
        preds=preds,
        labels=final_labels,
        gates=gates_np,
        threshold=threshold,
        horizon=HORIZON,
        split=split,
    )
    logger.info(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
