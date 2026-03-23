#!/usr/bin/env python3
"""
Production Inference Script — Hybrid51 h30 VIX-Gated

Replicates the exact model loading and inference logic from prediction_service.py,
including:
  - Per-symbol Platt scaling on Stage 1 logits  (Fix 4)
  - Raw logits for Stage 2 design matrix
  - Stage 3 VIX regime-gated fusion for per-agent gates  (Fix 5)

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

CONFIG  = json.load(open(PROD_ROOT / "config/production_config.json"))
HORIZON     = CONFIG["model_info"]["horizon_minutes"]
ALL_AGENTS  = CONFIG["architecture"]["stage1"]["agents"]
ALL_SYMBOLS = CONFIG["architecture"]["stage1"]["symbols"]

STANDARD_PEER_SYMBOLS = ['SPY', 'QQQ', 'IWM']
AGENT_2D_PEER_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'TLT']


# ---------------------------------------------------------------------------
# BinaryIndependentAgent wrapper
# ---------------------------------------------------------------------------
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


def _build_model_from_ckpt(ckpt, agent_type, device, symbol='SPXW'):
    state = ckpt['model_state_dict']
    if 'feat_dim' in ckpt:
        feat_dim = int(ckpt['feat_dim'])
    elif 'base._feat_idx' in state and state['base._feat_idx'].numel() > 0:
        feat_dim = 650 if int(state['base._feat_idx'].max().item()) >= 325 else 325
    else:
        feat_dim = 325 if symbol == 'SPXW' else 650

    use_subset    = bool(ckpt.get('feature_subset', True))
    use_attn_bb   = bool(ckpt.get('use_attention_backbone', False))
    use_attn_pool = bool(ckpt.get('use_attention_pool', False))
    cls_in_dim    = int(state['base.classifier.0.weight'].shape[1])
    has_static_proj = 'base.static_proj.weight' in state

    model = BinaryIndependentAgent(
        agent_type=agent_type, feat_dim=feat_dim,
        use_feature_subset=use_subset, use_attention_backbone=use_attn_bb,
        use_attention_pool=use_attn_pool, cls_input_dim=cls_in_dim,
    ).to(device)
    if not has_static_proj and hasattr(model.base, 'static_proj'):
        del model.base.static_proj
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _load_split(data_root, symbol, horizon, split):
    d = data_root / symbol / f"horizon_{horizon}min"
    seq    = np.load(d / f'{split}_sequences.npy', mmap_mode='r')
    labels = np.load(d / f'{split}_labels.npy')
    nm = np.load(d / 'norm_mean.npy') if (d / 'norm_mean.npy').exists() else None
    ns = np.load(d / 'norm_std.npy')  if (d / 'norm_std.npy').exists()  else None
    return seq, labels, nm, ns


def _load_chain_split(data_root, symbol, horizon, split):
    d = data_root / symbol / f"horizon_{horizon}min"
    seq    = np.load(d / f'{split}_sequences.npy', mmap_mode='r')
    labels = np.load(d / f'{split}_labels.npy')
    chain_path = d / f'{split}_chain_2d.npy'
    chain = np.load(chain_path) if chain_path.exists() else np.zeros((len(seq), 2), dtype=np.float32)
    nm = np.load(d / 'norm_mean.npy') if (d / 'norm_mean.npy').exists() else None
    ns = np.load(d / 'norm_std.npy')  if (d / 'norm_std.npy').exists()  else None
    return seq, chain, labels, nm, ns


def _load_vix_features(vix_root, split, n_target):
    """Load pre-computed VIX features; zeros fallback if missing."""
    vix_path = Path(vix_root) / f'{split}_vix_features.npy'
    if vix_path.exists():
        arr = np.load(vix_path).astype(np.float32)
        logger.info(f"  VIX features loaded: {arr.shape}")
    else:
        logger.warning(f"  VIX features not found at {vix_path} — using zeros (uniform gates)")
        arr = np.zeros((n_target, 10), dtype=np.float32)
    if len(arr) > n_target:
        arr = arr[:n_target]
    elif len(arr) < n_target:
        arr = np.concatenate([arr, np.zeros((n_target - len(arr), arr.shape[1]), dtype=np.float32)], axis=0)
    return arr


# ---------------------------------------------------------------------------
# Stage 1 inference  (with Platt scaling — Fix 4)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _infer_logits_probs(model, sequences, norm_mean, norm_std, device,
                        platt_coef=1.0, platt_intercept=0.0, batch_size=1024):
    """Returns (raw_logits, calibrated_logits, probs).
    Stage 2 uses raw_logits; Stage 3 uses probs."""
    n, raw_all = len(sequences), []
    nm_t = torch.from_numpy(norm_mean).float().to(device) if norm_mean is not None else None
    ns_t = torch.from_numpy(norm_std).float().to(device)  if norm_std  is not None else None
    for i in range(0, n, batch_size):
        seq_np = np.asarray(sequences[i:i+batch_size], dtype=np.float32)
        if not seq_np.flags.writeable:
            seq_np = seq_np.copy()
        sb = torch.from_numpy(seq_np).to(device)
        if nm_t is not None and ns_t is not None:
            sb = (sb - nm_t) / torch.clamp(ns_t, min=1e-6)
            sb = torch.clamp(sb, -10.0, 10.0)
        raw_all.append(model(sb).cpu().numpy().astype(np.float32))
    raw   = np.clip(np.concatenate(raw_all, axis=0), -88.0, 88.0)
    cal   = np.clip(platt_coef * raw + platt_intercept, -88.0, 88.0)
    probs = (1.0 / (1.0 + np.exp(-cal))).astype(np.float32)
    return raw, cal, probs


@torch.no_grad()
def _infer_frozen_2d(model, seq, chain, norm_mean, norm_std, device,
                     platt_coef=1.0, platt_intercept=0.0, batch_size=1024):
    """Batched 2D inference → (raw_logits, calibrated_logits, probs)."""
    n, raw_all = len(seq), []
    for i in range(0, n, batch_size):
        sb = torch.from_numpy(np.asarray(seq[i:i+batch_size],   dtype=np.float32))
        cb = torch.from_numpy(np.asarray(chain[i:i+batch_size], dtype=np.float32))
        if norm_mean is not None and norm_std is not None:
            nm_t = torch.from_numpy(norm_mean).float()
            ns_t = torch.clamp(torch.from_numpy(norm_std).float(), min=1e-6)
            sb   = torch.clamp((sb - nm_t) / ns_t, -10.0, 10.0)
        sb, cb = sb.to(device), cb.to(device)
        raw_all.append(model(sb, chain_2d=cb).cpu().numpy().astype(np.float32))
    raw   = np.clip(np.concatenate(raw_all, axis=0), -88.0, 88.0)
    cal   = np.clip(platt_coef * raw + platt_intercept, -88.0, 88.0)
    probs = (1.0 / (1.0 + np.exp(-cal))).astype(np.float32)
    return raw, cal, probs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',      choices=['val', 'test'], default='test')
    parser.add_argument('--output',     default='predictions.npz')
    parser.add_argument('--data-root',  default=CONFIG["data_paths"]["tier3_binary_root"])
    parser.add_argument('--vix-root',   default=CONFIG["data_paths"]["vix_features_root"])
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()

    device    = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    model_dir = PROD_ROOT / "models"
    split, bs = args.split, args.batch_size

    # ── STAGE 1 ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1: Per-Symbol Per-Agent Predictions")
    logger.info("=" * 60)

    standard_symbols = ['SPXW'] + STANDARD_PEER_SYMBOLS

    stage1_raw_logits = {s: {} for s in ALL_SYMBOLS}   # → Stage 2 design matrix
    stage1_probs      = {s: {} for s in ALL_SYMBOLS}   # → Stage 3 meta features
    split_labels      = None

    for symbol in ALL_SYMBOLS:
        for agent in ALL_AGENTS:
            ckpt_path = model_dir / f"stage1/{symbol}_agent{agent}.pt"
            if not ckpt_path.exists():
                logger.warning(f"  Missing: {ckpt_path}")
                continue
            ckpt    = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            platt_c = float(np.array(ckpt.get('platt_scaler_coef',      [[1.0]])).flatten()[0])
            platt_i = float(np.array(ckpt.get('platt_scaler_intercept', [0.0])).flatten()[0])

            if agent == '2D':
                model = _build_model_from_ckpt(ckpt, '2D', device, symbol)
                seq, chain, labels, nm, ns = _load_chain_split(data_root, symbol, HORIZON, split)
                raw_lg, _, pb = _infer_frozen_2d(model, seq, chain, nm, ns, device, platt_c, platt_i, bs)
            else:
                model = _build_model_from_ckpt(ckpt, agent, device, symbol)
                seq, labels, nm, ns = _load_split(data_root, symbol, HORIZON, split)
                raw_lg, _, pb = _infer_logits_probs(model, seq, nm, ns, device, platt_c, platt_i, bs)

            stage1_raw_logits[symbol][agent] = raw_lg
            stage1_probs[symbol][agent]      = pb
            if symbol == 'SPXW' and split_labels is None:
                split_labels = labels
            logger.info(f"  {symbol} Agent {agent}: n={len(pb):,}  platt=[{platt_c:.3f},{platt_i:.3f}]")
            del model;  torch.cuda.empty_cache()

    # ── STAGE 2 ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Cross-Symbol Fusion Per Agent")
    logger.info("=" * 60)

    chain_ctx_data = dict(np.load(model_dir / "stage2/chain_context.npz"))
    stage2_probs   = {}
    n_stage2       = None

    for agent in ALL_AGENTS:
        ckpt_path = model_dir / f"stage2/agent{agent}_fusion.pt"
        if not ckpt_path.exists():
            logger.warning(f"  Missing: {ckpt_path}");  continue

        ckpt     = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        n_inputs = ckpt['n_inputs']

        if agent == '2D':
            active_peers = ckpt.get('active_peers', AGENT_2D_PEER_SYMBOLS)
            sd_key = 'fusion_state_dict'
            n = min(len(stage1_raw_logits['SPXW']['2D']),
                    *(len(stage1_raw_logits.get(s, {}).get('2D', np.zeros(0)))
                      for s in active_peers if '2D' in stage1_raw_logits.get(s, {})))
            spxw_logit = stage1_raw_logits['SPXW']['2D'][:n].reshape(-1, 1)
            spxw_prob  = stage1_probs['SPXW']['2D'][:n].reshape(-1, 1)
            peer_parts = []
            for sym in active_peers:
                if '2D' in stage1_raw_logits.get(sym, {}):
                    peer_parts.append(stage1_raw_logits[sym]['2D'][:n].reshape(-1, 1))
                    peer_parts.append(stage1_probs[sym]['2D'][:n].reshape(-1, 1))
            peer_feat        = np.concatenate(peer_parts, axis=1)
            peer_logits_only = peer_feat[:, 0::2]
            diffs            = spxw_logit - peer_logits_only
            X = np.concatenate([spxw_logit, spxw_prob, peer_feat, diffs], axis=1).astype(np.float32)
        else:
            syms      = ckpt.get('symbols', standard_symbols)
            peer_syms = [s for s in syms if s != 'SPXW']
            sd_key    = 'model_state_dict'
            n = min(len(stage1_raw_logits[s][agent])
                    for s in syms if agent in stage1_raw_logits.get(s, {}))
            ctx_logits = chain_ctx_data[f'{split}_logits']
            ctx_probs  = chain_ctx_data[f'{split}_probs']
            n = min(n, len(ctx_logits))
            spxw_logit = stage1_raw_logits['SPXW'][agent][:n]
            parts = [spxw_logit.reshape(-1, 1), stage1_probs['SPXW'][agent][:n].reshape(-1, 1)]
            for sym in peer_syms:
                parts.append(stage1_raw_logits[sym][agent][:n].reshape(-1, 1))
                parts.append(stage1_probs[sym][agent][:n].reshape(-1, 1))
            for sym in peer_syms:
                parts.append((spxw_logit - stage1_raw_logits[sym][agent][:n]).reshape(-1, 1))
            parts += [ctx_logits[:n].reshape(-1, 1), ctx_probs[:n].reshape(-1, 1)]
            X = np.concatenate(parts, axis=1).astype(np.float32)

        assert X.shape[1] == n_inputs, f"Agent {agent}: expected {n_inputs}, got {X.shape[1]}"
        fusion = CrossSymbolAgentFusion(n_inputs=n_inputs, hidden_dim=32, dropout=0.2).to(device)
        fusion.load_state_dict(ckpt[sd_key], strict=True);  fusion.eval()
        with torch.no_grad():
            logits = fusion(torch.from_numpy(X).to(device)).cpu().numpy()
            probs  = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        stage2_probs[agent] = probs
        n_stage2 = n if n_stage2 is None else min(n_stage2, n)
        logger.info(f"  Agent {agent}: n={len(probs):,} n_inputs={n_inputs}")

    # ── STAGE 3 ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: VIX-Gated Meta-Ensemble")
    logger.info("=" * 60)

    n_final = min(n_stage2, *(len(stage2_probs[a]) for a in ALL_AGENTS if a in stage2_probs))

    # Load VG checkpoint — get agent order from checkpoint (may differ from ALL_AGENTS)
    stage3_agent_order  = list(ALL_AGENTS)
    stage3_vg           = None
    path3_vg = model_dir / "stage3/stage3_vix_gated.pt"
    if path3_vg.exists():
        try:
            vg_ckpt   = torch.load(path3_vg, map_location=device, weights_only=False)
            stage3_vg = RegimeGatedProbFusion(
                agent_names      = vg_ckpt['agent_names'],
                vix_feat_dim     = int(vg_ckpt['vix_feat_dim']),
                regime_emb_dim   = int(vg_ckpt['regime_emb_dim']),
                fusion_hidden_dim= int(vg_ckpt['fusion_hidden_dim']),
                dropout          = float(vg_ckpt['dropout']),
            ).to(device)
            stage3_vg.load_state_dict(vg_ckpt['model_state_dict'], strict=True)
            stage3_vg.eval()
            if 'agent_names' in vg_ckpt:
                stage3_agent_order = list(vg_ckpt['agent_names'])
            logger.info(f"  Stage3-VG loaded  threshold={vg_ckpt.get('threshold', 0.47):.3f}")
        except Exception as e:
            logger.warning(f"  Stage3-VG load failed — uniform gates: {e}")
    else:
        logger.warning(f"  stage3_vix_gated.pt not found — uniform gates")

    # Agent prob matrix in VG-checkpoint order
    agent_mat = np.stack(
        [stage2_probs.get(a, np.full(n_final, 0.5, dtype=np.float32))[:n_final]
         for a in stage3_agent_order], axis=1).astype(np.float32)

    # LogReg — primary prob/pred
    model3    = joblib.load(model_dir / "stage3/stage3_logreg.joblib")
    threshold = 0.36
    _mean     = agent_mat.mean(axis=1, keepdims=True)
    _std      = agent_mat.std(axis=1,  keepdims=True)
    _spread   = agent_mat.max(axis=1,  keepdims=True) - agent_mat.min(axis=1, keepdims=True)
    _majority = (agent_mat > 0.5).mean(axis=1, keepdims=True).astype(np.float32)
    _max      = agent_mat.max(axis=1, keepdims=True)
    _min      = agent_mat.min(axis=1, keepdims=True)
    meta_feat = np.concatenate([agent_mat, _mean, _std, _spread, _majority, _max, _min], axis=1)
    final_probs = model3.predict_proba(meta_feat)[:, 1].astype(np.float32)
    preds       = (final_probs > threshold).astype(np.int64)

    # VIX regime-gated gates (Fix 5)
    vix_features = _load_vix_features(args.vix_root, split, n_final)
    gates_np     = np.ones((n_final, len(stage3_agent_order)), dtype=np.float32)
    if stage3_vg is not None:
        logger.info("  Running VIX-gated model for per-agent gates...")
        gate_list = []
        with torch.no_grad():
            for i in range(0, n_final, bs):
                ap_t  = torch.from_numpy(agent_mat[i:i+bs]).to(device)
                vix_t = torch.from_numpy(vix_features[i:i+bs]).to(device)
                try:
                    _, vg_gates, _ = stage3_vg(ap_t, vix_t)
                    gate_list.append(vg_gates.detach().cpu().numpy())
                except Exception as e:
                    logger.debug(f"  VG batch {i} failed: {e}")
                    gate_list.append(np.ones((len(ap_t), len(stage3_agent_order)), dtype=np.float32))
        gates_np = np.concatenate(gate_list, axis=0)
        logger.info(f"  Gates: shape={gates_np.shape}")

    # Metrics
    final_labels = split_labels[:n_final] if split_labels is not None else np.zeros(n_final, dtype=np.int64)
    acc = float((preds == final_labels).mean())
    tp  = float(((preds == 1) & (final_labels == 1)).sum())
    fp  = float(((preds == 1) & (final_labels == 0)).sum())
    fn  = float(((preds == 0) & (final_labels == 1)).sum())
    f1  = float((2 * tp) / max(1.0, 2 * tp + fp + fn))
    logger.info(f"\n  n={n_final:,}  threshold={threshold}  accuracy={acc:.4f}  f1={f1:.4f}")
    for i, ag in enumerate(stage3_agent_order):
        logger.info(f"  Gate {ag}: mean={gates_np[:, i].mean():.3f}  std={gates_np[:, i].std():.3f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        probs       = final_probs,
        preds       = preds,
        labels      = final_labels,
        gates       = gates_np,
        agent_order = np.array(stage3_agent_order),
        threshold   = threshold,
        horizon     = HORIZON,
        split       = split,
    )
    logger.info(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
