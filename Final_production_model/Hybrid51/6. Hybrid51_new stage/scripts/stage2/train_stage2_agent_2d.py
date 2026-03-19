#!/usr/bin/env python3
"""
Stage 2: Agent 2D Cross-Symbol Fusion (SPXW unfrozen, peers frozen)

Trains one CrossSymbolAgentFusion model for Agent 2D by combining:
  - SPXW Agent 2D (UNFROZEN, fine-tuned jointly) → (logit, prob)
  - Peer Agent 2D (FROZEN) for SPY/QQQ/IWM/TLT → (logit, prob) each
  - 4 cross-diffs: SPXW_logit - peer_logit
  = 14-dim input → MLP → refined SPXW chain-based prediction

The SPXW Agent 2D is initialized from Stage 1 checkpoint and fine-tuned
in this Stage 2 context — it learns to be complementary to the frozen signals.

Prerequisites:
  Stage 1 Agent 2D checkpoints for all 5 symbols (from stage1_2d_chain_only)

Usage:
    python scripts/stage2/train_stage2_agent_2d.py --target SPXW --horizon 15
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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from hybrid51_utils import ArtifactPaths
from hybrid51_models.independent_agent import IndependentAgent
from hybrid51_models.cross_symbol_agent_fusion import CrossSymbolAgentFusion
from hybrid51_models.tlt_gated_agent_fusion import TLTGatedAgentFusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATHS = ArtifactPaths.default()
DEFAULT_PEER_SYMBOLS = ['SPY', 'QQQ', 'IWM']


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


def _build_model_from_ckpt(ckpt, agent_type: str, device: torch.device):
    """Reconstruct BinaryIndependentAgent matching saved checkpoint exactly."""
    state = ckpt['model_state_dict']
    feat_dim   = int(ckpt.get('feat_dim', 325))
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


def _load_2d_model(symbol: str, horizon: int, device: torch.device):
    """Load Stage 1 2D checkpoint. Returns (model, feat_dim)."""
    ckpt_path = PATHS.stage1_2d_ckpt(symbol, horizon)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage1 2D ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = _build_model_from_ckpt(ckpt, '2D', device)
    return model, int(ckpt.get('feat_dim', 325))


def _load_chain_split(symbol: str, horizon: int, split: str):
    """Load sequences, chain_2d, labels, norm stats from chain-only tier3."""
    d = PATHS.tier3_chain_dir(symbol, horizon)
    seq = np.load(d / f'{split}_sequences.npy')
    chain = np.load(d / f'{split}_chain_2d.npy')
    labels = np.load(d / f'{split}_labels.npy')
    nm = np.load(d / 'norm_mean.npy') if (d / 'norm_mean.npy').exists() else None
    ns = np.load(d / 'norm_std.npy') if (d / 'norm_std.npy').exists() else None
    return seq, chain, labels, nm, ns


@torch.no_grad()
def _infer_frozen_2d(model, seq, chain, norm_mean, norm_std, device, batch_size=1024):
    """Run batched frozen inference on 2D model → (logits, probs)."""
    n = len(seq)
    logits_all = []
    for i in range(0, n, batch_size):
        sb = torch.from_numpy(np.array(seq[i:i+batch_size])).float()
        cb = torch.from_numpy(np.array(chain[i:i+batch_size])).float()
        if norm_mean is not None and norm_std is not None:
            nm_t = torch.from_numpy(norm_mean)
            ns_t = torch.from_numpy(norm_std)
            sb = (sb - nm_t) / ns_t
        sb, cb = sb.to(device), cb.to(device)
        logits_all.append(model(sb, chain_2d=cb).cpu().numpy().astype(np.float32))
    logits = np.concatenate(logits_all, axis=0)
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return logits, probs


class Chain2DJointDataset(Dataset):
    """
    Dataset for Agent 2D Stage 2 joint training.
    Provides (spxw_seq, spxw_chain, frozen_peer_features, tlt_ctx, label) per sample.
    tlt_ctx is zeros when variant=traditional.
    """
    def __init__(self, spxw_seq, spxw_chain, peer_features, labels,
                 norm_mean=None, norm_std=None, tlt_ctx=None):
        self.spxw_seq = spxw_seq
        self.spxw_chain = torch.FloatTensor(spxw_chain)
        self.peer_features = torch.FloatTensor(peer_features)
        self.labels = torch.FloatTensor(labels)
        self.norm_mean = torch.FloatTensor(norm_mean) if norm_mean is not None else None
        self.norm_std = torch.FloatTensor(norm_std) if norm_std is not None else None
        self.tlt_ctx = torch.FloatTensor(tlt_ctx) if tlt_ctx is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.FloatTensor(np.array(self.spxw_seq[idx]))
        if self.norm_mean is not None and self.norm_std is not None:
            seq = (seq - self.norm_mean) / self.norm_std
        tlt = self.tlt_ctx[idx] if self.tlt_ctx is not None else torch.zeros(2)
        return seq, self.spxw_chain[idx], self.peer_features[idx], tlt, self.labels[idx]


def _build_peer_features(horizon: int, split: str, frozen_models: dict,
                          spxw_seq: np.ndarray, device: torch.device, batch_size: int,
                          peer_symbols: list):
    """Pre-compute frozen peer 2D logits. Returns (n, n_peers*2) array and aligned n."""
    peer_logits = {}
    peer_probs = {}
    n_spxw = len(spxw_seq)

    for sym in peer_symbols:
        seq, chain, _, nm, ns = _load_chain_split(sym, horizon, split)
        n_peer = min(len(seq), n_spxw)
        lg, pb = _infer_frozen_2d(frozen_models[sym], seq[:n_peer], chain[:n_peer], nm, ns, device, batch_size)
        peer_logits[sym] = lg
        peer_probs[sym] = pb

    n = min(n_spxw, *[len(peer_logits[s]) for s in peer_symbols])
    parts = []
    for sym in peer_symbols:
        parts.append(peer_logits[sym][:n].reshape(-1, 1))
        parts.append(peer_probs[sym][:n].reshape(-1, 1))

    peer_feat = np.concatenate(parts, axis=1).astype(np.float32)
    return peer_feat, {s: peer_logits[s][:n] for s in peer_symbols}, n


@torch.no_grad()
def _build_tlt_context_2d(horizon: int, split: str, tlt_model,
                           device: torch.device, batch_size: int) -> np.ndarray:
    """Run frozen TLT Stage1 2D model → (n, 2) array of [logit, prob]."""
    seq, chain, _, nm, ns = _load_chain_split('TLT', horizon, split)
    lg, pb = _infer_frozen_2d(tlt_model, seq, chain, nm, ns, device, batch_size)
    return np.stack([lg, pb], axis=1).astype(np.float32)


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


def _sweep_threshold(probs, labels):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.30, 0.66, 0.01):
        f1 = f1_score(labels, (probs > thr).astype(int), average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='SPXW')
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--peers', nargs='+', default=DEFAULT_PEER_SYMBOLS)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--stage1-batch', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--variant', choices=['traditional', 'tlt_gated'], default='traditional',
                        help='traditional: CrossSymbolAgentFusion; '
                             'tlt_gated: TLTGatedAgentFusion with learned TLT peer-trust gates')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    peer_symbols = [s.upper() for s in args.peers if s.upper() != args.target.upper()]
    PATHS.stage2_cross_results.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Stage 2 Agent 2D [{args.variant}]: target={args.target} h={args.horizon}")
    logger.info(f"SPXW 2D: UNFROZEN | Peers: FROZEN | peers={peer_symbols}")
    logger.info(f"{'='*60}")

    spxw_2d_model, spxw_feat_dim = _load_2d_model(args.target, args.horizon, device)
    spxw_2d_model.train()
    logger.info(f"  Loaded SPXW Stage1 2D (unfrozen), feat_dim={spxw_feat_dim}")

    frozen_peer_models = {}
    for sym in peer_symbols:
        try:
            m, _ = _load_2d_model(sym, args.horizon, device)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            frozen_peer_models[sym] = m
            logger.info(f"  Loaded {sym} Stage1 2D (frozen)")
        except FileNotFoundError as e:
            logger.warning(f"  {e} — will skip {sym} peer")

    tlt_model = None
    if args.variant == 'tlt_gated':
        try:
            tlt_model, _ = _load_2d_model('TLT', args.horizon, device)
            tlt_model.eval()
            for p in tlt_model.parameters():
                p.requires_grad_(False)
            logger.info("  Loaded TLT Stage1 2D (frozen, gate context only)")
        except FileNotFoundError as e:
            logger.error(f"  TLT 2D ckpt missing: {e}")
            return

    active_peers = [s for s in peer_symbols if s in frozen_peer_models]
    n_peer_dims = len(active_peers) * 2
    n_inputs = 2 + n_peer_dims + len(active_peers)
    if args.variant == 'tlt_gated':
        n_inputs += 2  # tlt_logit + tlt_prob
    logger.info(f"  Active peers: {active_peers}, n_inputs={n_inputs}")

    spxw_seq_train, spxw_chain_train, train_labels, nm_spxw, ns_spxw = _load_chain_split(args.target, args.horizon, 'train')
    spxw_seq_val,   spxw_chain_val,   val_labels,   _,       _        = _load_chain_split(args.target, args.horizon, 'val')
    spxw_seq_test,  spxw_chain_test,  test_labels,  _,       _        = _load_chain_split(args.target, args.horizon, 'test')

    logger.info("  Pre-computing frozen peer features...")
    peer_feat_train, _, n_train = _build_peer_features(
        args.horizon, 'train', frozen_peer_models, spxw_seq_train, device, args.stage1_batch, active_peers)
    peer_feat_val,  _, n_val   = _build_peer_features(
        args.horizon, 'val',   frozen_peer_models, spxw_seq_val,   device, args.stage1_batch, active_peers)
    peer_feat_test, _, n_test  = _build_peer_features(
        args.horizon, 'test',  frozen_peer_models, spxw_seq_test,  device, args.stage1_batch, active_peers)

    tlt_ctx_train = tlt_ctx_val = tlt_ctx_test = None
    if args.variant == 'tlt_gated':
        logger.info("  Pre-computing TLT gate context...")
        tlt_ctx_train = _build_tlt_context_2d(args.horizon, 'train', tlt_model, device, args.stage1_batch)
        tlt_ctx_val   = _build_tlt_context_2d(args.horizon, 'val',   tlt_model, device, args.stage1_batch)
        tlt_ctx_test  = _build_tlt_context_2d(args.horizon, 'test',  tlt_model, device, args.stage1_batch)
        n_train = min(n_train, len(tlt_ctx_train))
        n_val   = min(n_val,   len(tlt_ctx_val))
        n_test  = min(n_test,  len(tlt_ctx_test))

    train_labels = train_labels[:n_train].astype(np.float32)
    val_labels   = val_labels[:n_val].astype(np.float32)
    test_labels  = test_labels[:n_test].astype(np.float32)

    train_ds = Chain2DJointDataset(
        spxw_seq_train[:n_train], spxw_chain_train[:n_train],
        peer_feat_train[:n_train], train_labels, nm_spxw, ns_spxw,
        tlt_ctx=tlt_ctx_train[:n_train] if tlt_ctx_train is not None else None)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0, pin_memory=True)

    if args.variant == 'tlt_gated':
        fusion = TLTGatedAgentFusion(
            n_peers=len(active_peers), tlt_ctx_dim=2, tlt_emb_dim=16,
            gate_hidden_dim=8, fusion_hidden_dim=32, dropout=0.2,
            has_chain_ctx=False,
        ).to(device)
    else:
        fusion = CrossSymbolAgentFusion(n_inputs=n_inputs, hidden_dim=32, dropout=0.2).to(device)
    logger.info(f"  Fusion ({type(fusion).__name__}) params: {fusion.count_parameters():,} | "
                f"2D params: {sum(p.numel() for p in spxw_2d_model.parameters()):,}")

    opt = torch.optim.AdamW(
        list(spxw_2d_model.parameters()) + list(fusion.parameters()),
        lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = BinaryFocalLoss(gamma=2.0, alpha=float(np.mean(train_labels)), label_smoothing=0.05)

    best_val_auc = -1.0
    best_fusion_state = None
    best_2d_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(args.epochs):
        spxw_2d_model.train()
        fusion.train()
        total_loss = 0.0

        for spxw_seq_b, spxw_chain_b, peer_feat_b, tlt_ctx_b, y_b in train_loader:
            spxw_seq_b   = spxw_seq_b.to(device)
            spxw_chain_b = spxw_chain_b.to(device)
            peer_feat_b  = peer_feat_b.to(device)
            tlt_ctx_b    = tlt_ctx_b.to(device)
            y_b          = y_b.to(device)

            spxw_logit = spxw_2d_model(spxw_seq_b, chain_2d=spxw_chain_b)
            spxw_prob  = torch.sigmoid(spxw_logit).detach()
            peer_logits_b = peer_feat_b[:, 0::2]
            diffs = spxw_logit.unsqueeze(1) - peer_logits_b

            if args.variant == 'tlt_gated':
                x = torch.cat([spxw_logit.unsqueeze(1), spxw_prob.unsqueeze(1),
                                peer_feat_b, diffs, tlt_ctx_b], dim=1)
                opt.zero_grad()
                fusion_logit, _ = fusion(x)
            else:
                x = torch.cat([spxw_logit.unsqueeze(1), spxw_prob.unsqueeze(1),
                                peer_feat_b, diffs], dim=1)
                opt.zero_grad()
                fusion_logit = fusion(x)

            loss = criterion(fusion_logit, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(spxw_2d_model.parameters()) + list(fusion.parameters()), 1.0)
            opt.step()
            total_loss += float(loss.item())

        sched.step()

        spxw_2d_model.eval()
        fusion.eval()
        val_probs = _eval_joint(
            spxw_2d_model, fusion, spxw_seq_val[:n_val], spxw_chain_val[:n_val],
            peer_feat_val[:n_val], nm_spxw, ns_spxw, device, args.stage1_batch,
            tlt_ctx=tlt_ctx_val[:n_val] if tlt_ctx_val is not None else None,
            variant=args.variant)
        try:
            val_auc = float(roc_auc_score(val_labels.astype(int), val_probs))
        except Exception:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_fusion_state = {k: v.cpu().clone() for k, v in fusion.state_dict().items()}
            best_2d_state = {k: v.cpu().clone() for k, v in spxw_2d_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(1, len(train_loader))
            if args.variant == 'tlt_gated':
                logger.info(f"  Ep {epoch+1:3d}: loss={avg_loss:.4f} val_auc={val_auc:.4f} (best={best_val_auc:.4f})")
            else:
                logger.info(f"  Ep {epoch+1:3d}: loss={avg_loss:.4f} val_auc={val_auc:.4f} (best={best_val_auc:.4f})")

        if patience_counter >= patience:
            logger.info(f"  Early stop at epoch {epoch+1}")
            break

    if best_fusion_state:
        fusion.load_state_dict(best_fusion_state)
        spxw_2d_model.load_state_dict(best_2d_state)

    fusion.eval()
    spxw_2d_model.eval()

    val_probs  = _eval_joint(
        spxw_2d_model, fusion, spxw_seq_val[:n_val],  spxw_chain_val[:n_val],
        peer_feat_val[:n_val],  nm_spxw, ns_spxw, device, args.stage1_batch,
        tlt_ctx=tlt_ctx_val[:n_val]   if tlt_ctx_val   is not None else None,
        variant=args.variant)
    test_probs = _eval_joint(
        spxw_2d_model, fusion, spxw_seq_test[:n_test], spxw_chain_test[:n_test],
        peer_feat_test[:n_test], nm_spxw, ns_spxw, device, args.stage1_batch,
        tlt_ctx=tlt_ctx_test[:n_test] if tlt_ctx_test is not None else None,
        variant=args.variant)

    opt_thr, _ = _sweep_threshold(val_probs, val_labels.astype(int))
    val_metrics  = _metrics(val_labels.astype(int),  val_probs,  opt_thr)
    test_metrics = _metrics(test_labels.astype(int), test_probs, opt_thr)

    logger.info(f"  Val:  acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}")
    logger.info(f"  Test: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} "
                f"auc={test_metrics['auc']:.4f} ic={test_metrics['ic']:.4f}")

    if args.variant == 'tlt_gated':
        ckpt_path  = PATHS.stage2_tlt_gated_ckpt(args.target, '2D', args.horizon)
        probs_path = PATHS.stage2_tlt_gated_probs(args.target, '2D', args.horizon)
    else:
        ckpt_path  = PATHS.stage2_per_agent_ckpt(args.target, '2D', args.horizon)
        probs_path = PATHS.stage2_per_agent_probs(args.target, '2D', args.horizon)

    torch.save({
        'fusion_state_dict': fusion.state_dict(),
        'spxw_2d_state_dict': spxw_2d_model.state_dict(),
        'target': args.target, 'horizon': args.horizon, 'variant': args.variant,
        'n_inputs': n_inputs, 'active_peers': active_peers,
        'optimal_threshold': opt_thr,
        'val_metrics': val_metrics, 'test_metrics': test_metrics,
    }, ckpt_path)
    np.savez(probs_path,
             val_probs=val_probs, val_labels=val_labels.astype(np.int64),
             test_probs=test_probs, test_labels=test_labels.astype(np.int64))

    logger.info(f"  Saved: {ckpt_path}")
    logger.info("Done.")


@torch.no_grad()
def _eval_joint(spxw_2d_model, fusion, spxw_seq, spxw_chain, peer_feat,
                norm_mean, norm_std, device, batch_size,
                tlt_ctx=None, variant='traditional'):
    """Evaluate fusion model on a split."""
    n = len(spxw_seq)
    probs_all = []
    peer_feat_t = torch.from_numpy(peer_feat).float()
    tlt_t = torch.from_numpy(tlt_ctx).float() if tlt_ctx is not None else None

    for i in range(0, n, batch_size):
        sb = torch.from_numpy(np.array(spxw_seq[i:i+batch_size])).float()
        cb = torch.from_numpy(np.array(spxw_chain[i:i+batch_size])).float()
        pb = peer_feat_t[i:i+batch_size]

        if norm_mean is not None and norm_std is not None:
            nm_t = torch.from_numpy(norm_mean)
            ns_t = torch.from_numpy(norm_std)
            sb = (sb - nm_t) / ns_t

        sb, cb, pb = sb.to(device), cb.to(device), pb.to(device)
        spxw_logit = spxw_2d_model(sb, chain_2d=cb)
        spxw_prob  = torch.sigmoid(spxw_logit)
        peer_logits_b = pb[:, 0::2]
        diffs = spxw_logit.unsqueeze(1) - peer_logits_b

        if variant == 'tlt_gated' and tlt_t is not None:
            tc = tlt_t[i:i+batch_size].to(device)
            x = torch.cat([spxw_logit.unsqueeze(1), spxw_prob.unsqueeze(1), pb, diffs, tc], dim=1)
            fusion_logit, _ = fusion(x)
        else:
            x = torch.cat([spxw_logit.unsqueeze(1), spxw_prob.unsqueeze(1), pb, diffs], dim=1)
            fusion_logit = fusion(x)

        probs_all.append(torch.sigmoid(fusion_logit).cpu().numpy().astype(np.float32))

    return np.concatenate(probs_all, axis=0)


if __name__ == '__main__':
    main()
