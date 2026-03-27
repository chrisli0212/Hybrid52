#!/usr/bin/env python3
"""
Stage 3 Cross-Agent Meta-Learner

Combines 7 per-agent Stage 2 cross-symbol fusion probabilities
(A, B, C, K, T, Q, 2D) into a final SPXW directional signal.

Input features (13 dims):
  [prob_A, prob_B, prob_C, prob_K, prob_T, prob_Q, prob_2D]   (7 agent probs)
  [mean, std, spread, majority_vote, max_prob, min_prob]       (6 summary stats)

Meta-learners tested: LogReg (multiple C) + optional MLP.

Prerequisites:
  Stage 2 per-agent probs from:
    train_stage2_per_agent.py   (agents A/B/C/K/T/Q)
    train_stage2_agent_2d.py    (agent 2D)

Usage:
    python scripts/stage3/train_stage3_cross_agent_meta.py --target SPXW --horizon 30
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from hybrid51_utils import ArtifactPaths
from hybrid51_utils.artifacts import DEFAULT_TRAINING_HORIZON_MINUTES
from hybrid51_models.regime_gated_meta_model import RegimeGatedProbFusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATHS = ArtifactPaths.default()
ALL_AGENTS = ['A', 'B', 'C', 'K', 'T', 'Q', '2D']
N_AGENT_PROBS = len(ALL_AGENTS)   # 7
N_SUMMARY     = 6                  # mean, std, spread, majority_vote, max, min
N_META_INPUTS = N_AGENT_PROBS + N_SUMMARY  # 13


def _align_min(*arrays):
    n = min(len(a) for a in arrays)
    return tuple(a[:n] for a in arrays)


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


def _prob_score(labels, probs):
    """Threshold-free validation objective for early stopping."""
    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = 0.5
    brier = float(brier_score_loss(labels, probs))
    return auc - brier


def _split_indices(n: int, train_frac: float, mode: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    train_frac = min(max(train_frac, 0.5), 0.95)
    split = int(n * train_frac)
    split = min(max(split, 1), n - 1)
    if mode == 'time':
        idx = np.arange(n)
        return idx[:split], idx[split:]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return perm[:split], perm[split:]


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def _stack_agent_probs(agent_probs: dict[str, np.ndarray]) -> np.ndarray:
    """Stack probs in fixed ALL_AGENTS order; fill missing with neutral 0.5."""
    cols = []
    n = next(iter(agent_probs.values())).shape[0]
    for ag in ALL_AGENTS:
        if ag in agent_probs:
            cols.append(agent_probs[ag].reshape(-1, 1).astype(np.float32))
        else:
            cols.append(np.full((n, 1), 0.5, dtype=np.float32))
    return np.concatenate(cols, axis=1)


def _resample_to_length(arr: np.ndarray, n_target: int) -> np.ndarray:
    if len(arr) == n_target:
        return arr
    if len(arr) <= 1:
        raise ValueError(f"Cannot resample from length {len(arr)} to {n_target}")
    idx = np.linspace(0, len(arr) - 1, n_target).astype(np.int64)
    return arr[idx]


def _load_vix_features(vix_dir: Path, split: str, n_target: int) -> np.ndarray:
    p = vix_dir / f'{split}_vix_features.npy'
    if not p.exists():
        raise FileNotFoundError(f"Missing VIX features: {p}")
    x = np.load(p)
    if x.ndim == 3:
        x = x[:, -1, :]
    if x.ndim != 2:
        raise RuntimeError(f"Unexpected VIX feature shape: {x.shape}")
    x = _resample_to_length(x.astype(np.float32), n_target)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def build_meta_features(agent_probs: dict) -> np.ndarray:
    """
    Build 13-dim meta feature vector from 7 per-agent probs.
    Missing agents filled with 0.5 (neutral).
    """
    probs_list = []
    for ag in ALL_AGENTS:
        if ag in agent_probs:
            probs_list.append(agent_probs[ag].reshape(-1, 1).astype(np.float32))
        else:
            n = next(iter(agent_probs.values())).shape[0]
            probs_list.append(np.full((n, 1), 0.5, dtype=np.float32))
            logger.warning(f"  Agent {ag} missing — filled with 0.5")

    raw = np.concatenate(probs_list, axis=1)  # (N, 7)

    mean_p   = raw.mean(axis=1, keepdims=True)
    std_p    = raw.std(axis=1, keepdims=True)
    spread   = raw.max(axis=1, keepdims=True) - raw.min(axis=1, keepdims=True)
    majority = (raw > 0.5).mean(axis=1, keepdims=True).astype(np.float32)
    max_p    = raw.max(axis=1, keepdims=True)
    min_p    = raw.min(axis=1, keepdims=True)

    X = np.concatenate([raw, mean_p, std_p, spread, majority, max_p, min_p], axis=1)
    assert X.shape[1] == N_META_INPUTS, f"Expected {N_META_INPUTS} meta dims, got {X.shape[1]}"
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class _Stage3MLP(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='SPXW')
    parser.add_argument('--horizon', type=int, default=DEFAULT_TRAINING_HORIZON_MINUTES)
    parser.add_argument('--C', type=float, default=0.01)
    parser.add_argument('--agents', nargs='+', default=ALL_AGENTS)
    parser.add_argument('--probs-variant', choices=['traditional', 'tlt_gated'], default='traditional',
                        help='Which Stage 2 probs to load: traditional (*_cross_probs.npz) or '
                             'tlt_gated (*_tlt_gated_probs.npz). Applies to all agents including 2D.')
    parser.add_argument('--meta', choices=['logreg', 'mlp', 'auto'], default='auto')
    parser.add_argument('--meta-epochs', type=int, default=100)
    parser.add_argument('--meta-batch-size', type=int, default=2048)
    parser.add_argument('--meta-lr', type=float, default=1e-3)
    parser.add_argument('--meta-hidden-dim', type=int, default=32)
    parser.add_argument('--meta-dropout', type=float, default=0.3)
    parser.add_argument('--meta-patience', type=int, default=15)
    parser.add_argument('--enable-vix-gated', action='store_true',
                        help='Enable Agent VIX regime-gated fusion candidate')
    parser.add_argument('--vix-data-root', type=str, default='/workspace/data/tier3_vix_v4/VIXW',
                        help='Directory containing val_vix_features.npy/test_vix_features.npy')
    parser.add_argument('--vix-warmstart', type=str, default='',
                        help='Optional stage1_vix warm-start checkpoint for AgentVIX')
    parser.add_argument('--vix-epochs', type=int, default=40)
    parser.add_argument('--vix-batch-size', type=int, default=2048)
    parser.add_argument('--vix-lr', type=float, default=1e-3)
    parser.add_argument('--vix-weight-decay', type=float, default=1e-4)
    parser.add_argument('--vix-patience', type=int, default=10)
    parser.add_argument('--vix-regime-emb-dim', type=int, default=32)
    parser.add_argument('--vix-fusion-hidden-dim', type=int, default=64)
    parser.add_argument('--vix-dropout', type=float, default=0.2)
    parser.add_argument('--vix-train-frac', type=float, default=0.75,
                        help='Train fraction inside val split for VIX-gated fit/monitor')
    parser.add_argument('--vix-split-mode', choices=['time', 'random'], default='time',
                        help='Split mode for VIX-gated fit/monitor on val data')
    parser.add_argument('--vix-freeze-warmstart-epochs', type=int, default=3,
                        help='Freeze AgentVIX warm-start for initial epochs')
    parser.add_argument('--vix-input-noise-std', type=float, default=0.01,
                        help='Gaussian noise std added to train inputs for regularization')
    parser.add_argument('--vix-label-smoothing', type=float, default=0.02,
                        help='Binary label smoothing factor for VIX-gated training')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = PATHS.stage3_results
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Stage 3 Cross-Agent Meta | target={args.target} h={args.horizon}")
    logger.info(f"{'='*60}")

    val_probs_by_agent: dict[str, np.ndarray] = {}
    test_probs_by_agent: dict[str, np.ndarray] = {}
    val_labels_list = []
    test_labels_list = []

    variant_suffix = f'_{args.probs_variant}' if args.probs_variant == 'tlt_gated' else ''
    logger.info(f"  Probs variant: {args.probs_variant}")

    for ag in args.agents:
        if args.probs_variant == 'tlt_gated':
            probs_path = PATHS.stage2_tlt_gated_probs(args.target, ag, args.horizon)
        else:
            probs_path = PATHS.stage2_per_agent_probs(args.target, ag, args.horizon)
        if not probs_path.exists():
            logger.warning(f"  Agent {ag}: probs not found at {probs_path} — skipping")
            continue
        data = np.load(probs_path)
        val_probs_by_agent[ag]  = data['val_probs'].astype(np.float32)
        test_probs_by_agent[ag] = data['test_probs'].astype(np.float32)
        val_labels_list.append(data['val_labels'].astype(np.int64))
        test_labels_list.append(data['test_labels'].astype(np.int64))
        logger.info(f"  Agent {ag}: val={len(val_probs_by_agent[ag]):,} test={len(test_probs_by_agent[ag]):,}")

    if not val_probs_by_agent:
        raise RuntimeError("No Stage 2 per-agent probs found. Run Stage 2 scripts first.")

    n_val  = min(len(v) for v in val_probs_by_agent.values())
    n_test = min(len(v) for v in test_probs_by_agent.values())
    for ag in val_probs_by_agent:
        val_probs_by_agent[ag]  = val_probs_by_agent[ag][:n_val]
        test_probs_by_agent[ag] = test_probs_by_agent[ag][:n_test]
    val_labels  = val_labels_list[0][:n_val]
    test_labels = test_labels_list[0][:n_test]

    X_val  = build_meta_features(val_probs_by_agent)
    X_test = build_meta_features(test_probs_by_agent)
    logger.info(f"  Meta features: val={X_val.shape} test={X_test.shape}")

    all_results: dict = {}
    val_selection_scores: dict = {}
    saved_logregs: dict = {}
    vix_artifact = None

    val_avg = np.stack(list(val_probs_by_agent.values()), axis=1).mean(axis=1)
    test_avg = np.stack(list(test_probs_by_agent.values()), axis=1).mean(axis=1)
    avg_thr, _ = _sweep_threshold(val_avg, val_labels)
    all_results['avg_agents'] = {
        'name': f'Simple Avg ({len(val_probs_by_agent)} agents)',
        'val': _metrics(val_labels, val_avg, avg_thr),
        'test': _metrics(test_labels, test_avg, avg_thr),
    }
    val_selection_scores['avg_agents'] = (
        all_results['avg_agents']['val']['accuracy'] + all_results['avg_agents']['val']['f1'])

    for c in [args.C, 0.1, 0.01]:
        key = f'logreg_C={c}'
        m = LogisticRegression(C=c, max_iter=1000, solver='lbfgs', class_weight='balanced')
        m.fit(X_val, val_labels)
        val_p  = m.predict_proba(X_val)[:, 1]
        test_p = m.predict_proba(X_test)[:, 1]
        thr, _ = _sweep_threshold(val_p, val_labels)
        saved_logregs[key] = m
        all_results[key] = {
            'name': f'LogReg (C={c})',
            'val':  _metrics(val_labels,  val_p,  thr),
            'test': _metrics(test_labels, test_p, thr),
        }
        val_selection_scores[key] = (
            all_results[key]['val']['accuracy'] + all_results[key]['val']['f1'])

    best_method = max(val_selection_scores, key=val_selection_scores.get)
    best_score  = val_selection_scores[best_method]

    if args.meta in ('mlp', 'auto'):
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_val)
        split = int(n_val * 0.8)
        tr_idx, vm_idx = perm[:split], perm[split:]

        X_tr = torch.from_numpy(X_val[tr_idx]).float().to(device)
        y_tr = torch.from_numpy(val_labels[tr_idx].astype(np.float32)).to(device)
        X_vm = torch.from_numpy(X_val[vm_idx]).float().to(device)
        y_vm_np = val_labels[vm_idx].astype(np.int64)

        mlp = _Stage3MLP(input_dim=N_META_INPUTS,
                         hidden_dim=args.meta_hidden_dim,
                         dropout=args.meta_dropout).to(device)
        up_frac = float(val_labels[tr_idx].mean())
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - up_frac) / max(up_frac, 1e-6)]).to(device))
        opt  = torch.optim.Adam(mlp.parameters(), lr=args.meta_lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.meta_epochs, eta_min=1e-6)

        best_state, best_vm = None, -1.0
        patience_ctr = 0

        for epoch in range(args.meta_epochs):
            mlp.train()
            for i in range(0, len(X_tr), args.meta_batch_size):
                xb = X_tr[i:i + args.meta_batch_size]
                yb = y_tr[i:i + args.meta_batch_size]
                opt.zero_grad()
                crit(mlp(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 5.0)
                opt.step()
            sched.step()

            mlp.eval()
            with torch.no_grad():
                vm_prob = torch.sigmoid(mlp(X_vm)).cpu().numpy().astype(np.float32)
            vm_score = float(accuracy_score(y_vm_np, (vm_prob > 0.5).astype(int)))
            vm_score += float(f1_score(y_vm_np, (vm_prob > 0.5).astype(int), average='binary', zero_division=0))

            if vm_score > best_vm:
                best_vm = vm_score
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= args.meta_patience:
                break

        if best_state:
            mlp.load_state_dict(best_state)

        mlp.eval()
        with torch.no_grad():
            val_p  = torch.sigmoid(mlp(torch.from_numpy(X_val).float().to(device))).cpu().numpy().astype(np.float32)
            test_p = torch.sigmoid(mlp(torch.from_numpy(X_test).float().to(device))).cpu().numpy().astype(np.float32)

        thr, _ = _sweep_threshold(val_p, val_labels)
        all_results['mlp'] = {
            'name': 'MLP',
            'val':  _metrics(val_labels,  val_p,  thr),
            'test': _metrics(test_labels, test_p, thr),
        }
        mlp_score = all_results['mlp']['val']['accuracy'] + all_results['mlp']['val']['f1']
        val_selection_scores['mlp'] = mlp_score
        if mlp_score > best_score:
            best_score = mlp_score
            best_method = 'mlp'

    if args.enable_vix_gated:
        try:
            val_agent = _stack_agent_probs(val_probs_by_agent)
            test_agent = _stack_agent_probs(test_probs_by_agent)
            vix_dir = Path(args.vix_data_root)
            Xv_val = _load_vix_features(vix_dir, 'val', n_val)
            Xv_test = _load_vix_features(vix_dir, 'test', n_test)

            tr_idx, vm_idx = _split_indices(
                n=n_val,
                train_frac=args.vix_train_frac,
                mode=args.vix_split_mode,
                seed=42,
            )

            Xp_tr = torch.from_numpy(val_agent[tr_idx]).float().to(device)
            Xv_tr = torch.from_numpy(Xv_val[tr_idx]).float().to(device)
            y_tr = torch.from_numpy(val_labels[tr_idx].astype(np.float32)).to(device)
            Xp_vm = torch.from_numpy(val_agent[vm_idx]).float().to(device)
            Xv_vm = torch.from_numpy(Xv_val[vm_idx]).float().to(device)
            y_vm_np = val_labels[vm_idx].astype(np.int64)

            vix_model = RegimeGatedProbFusion(
                agent_names=ALL_AGENTS,
                vix_feat_dim=Xv_val.shape[1],
                regime_emb_dim=args.vix_regime_emb_dim,
                fusion_hidden_dim=args.vix_fusion_hidden_dim,
                dropout=args.vix_dropout,
            ).to(device)

            warmstart = Path(args.vix_warmstart) if args.vix_warmstart else PATHS.stage1_vix_ckpt()
            if warmstart.exists():
                try:
                    state = torch.load(warmstart, map_location='cpu', weights_only=True)
                    vix_model.vix_agent.load_state_dict(state, strict=False)
                    logger.info(f"  Loaded VIX warm-start: {warmstart}")
                except Exception as e:
                    logger.warning(f"  VIX warm-start load failed ({warmstart}): {e}")

            up_frac = float(val_labels[tr_idx].mean())
            crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - up_frac) / max(up_frac, 1e-6)]).to(device))
            opt = torch.optim.AdamW(vix_model.parameters(), lr=args.vix_lr, weight_decay=args.vix_weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.vix_epochs, eta_min=1e-6)

            best_vm = -1.0
            best_state = None
            patience_ctr = 0

            freeze_epochs = max(0, int(args.vix_freeze_warmstart_epochs))
            if freeze_epochs > 0:
                _set_requires_grad(vix_model.vix_agent, enabled=False)

            for epoch in range(args.vix_epochs):
                if freeze_epochs > 0 and epoch == freeze_epochs:
                    _set_requires_grad(vix_model.vix_agent, enabled=True)
                vix_model.train()
                for i in range(0, len(Xp_tr), args.vix_batch_size):
                    xb_p = Xp_tr[i:i + args.vix_batch_size]
                    xb_v = Xv_tr[i:i + args.vix_batch_size]
                    yb = y_tr[i:i + args.vix_batch_size]
                    if args.vix_input_noise_std > 0:
                        noise_p = torch.randn_like(xb_p) * float(args.vix_input_noise_std)
                        noise_v = torch.randn_like(xb_v) * float(args.vix_input_noise_std)
                        xb_p = torch.clamp(xb_p + noise_p, 0.0, 1.0)
                        xb_v = xb_v + noise_v
                    if args.vix_label_smoothing > 0:
                        eps = float(min(max(args.vix_label_smoothing, 0.0), 0.45))
                        yb = yb * (1.0 - 2.0 * eps) + eps
                    opt.zero_grad()
                    logits, _, _ = vix_model(xb_p, xb_v)
                    crit(logits, yb).backward()
                    torch.nn.utils.clip_grad_norm_(vix_model.parameters(), 5.0)
                    opt.step()
                sched.step()

                vix_model.eval()
                with torch.no_grad():
                    vm_logits, _, _ = vix_model(Xp_vm, Xv_vm)
                    vm_prob = torch.sigmoid(vm_logits).cpu().numpy().astype(np.float32)
                vm_score = _prob_score(y_vm_np, vm_prob)

                if vm_score > best_vm:
                    best_vm = vm_score
                    best_state = {k: v.detach().cpu().clone() for k, v in vix_model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= args.vix_patience:
                    break

            if best_state is not None:
                vix_model.load_state_dict(best_state)

            vix_model.eval()
            with torch.no_grad():
                vm_logits, _, _ = vix_model(Xp_vm, Xv_vm)
                val_logits, _, _ = vix_model(
                    torch.from_numpy(val_agent).float().to(device),
                    torch.from_numpy(Xv_val).float().to(device),
                )
                test_logits, test_gates, _ = vix_model(
                    torch.from_numpy(test_agent).float().to(device),
                    torch.from_numpy(Xv_test).float().to(device),
                )
                vm_p = torch.sigmoid(vm_logits).cpu().numpy().astype(np.float32)
                val_p = torch.sigmoid(val_logits).cpu().numpy().astype(np.float32)
                test_p = torch.sigmoid(test_logits).cpu().numpy().astype(np.float32)

            vm_thr, _ = _sweep_threshold(vm_p, y_vm_np)
            gate_means = RegimeGatedProbFusion.gate_summary(test_gates, ALL_AGENTS)
            all_results['vix_gated'] = {
                'name': 'VIX Regime-Gated Fusion',
                'val': _metrics(val_labels, val_p, vm_thr),
                'test': _metrics(test_labels, test_p, vm_thr),
                'val_monitor': _metrics(y_vm_np, vm_p, vm_thr),
                'gate_means_test': gate_means,
            }
            vix_artifact = {
                'state_dict': {k: v.cpu().clone() for k, v in vix_model.state_dict().items()},
                'vix_feat_dim': int(Xv_val.shape[1]),
                'regime_emb_dim': int(args.vix_regime_emb_dim),
                'fusion_hidden_dim': int(args.vix_fusion_hidden_dim),
                'dropout': float(args.vix_dropout),
            }
            vix_score = (
                all_results['vix_gated']['val_monitor']['accuracy']
                + all_results['vix_gated']['val_monitor']['f1']
            )
            val_selection_scores['vix_gated'] = vix_score

            if vix_score > best_score:
                best_score = vix_score
                best_method = 'vix_gated'
        except Exception as e:
            logger.warning(f"VIX-gated method skipped due to error: {e}")

    logger.info("\nResults:")
    for k, v in all_results.items():
        vm = v['val']
        tm = v['test']
        logger.info(
            f"  {k:25s}  val(acc={vm['accuracy']:.4f} f1={vm['f1']:.4f} auc={vm['auc']:.4f})  "
            f"test(acc={tm['accuracy']:.4f} f1={tm['f1']:.4f} auc={tm['auc']:.4f})"
        )
    logger.info(f"\n  Best: {best_method} (val score={best_score:.4f})")

    metrics_path = out_dir / f'{args.target}_h{args.horizon}_stage3_cross_agent_metrics{variant_suffix}.json'
    metrics_path.write_text(json.dumps({
        'best_method': best_method,
        'agents_used': list(val_probs_by_agent.keys()),
        'n_val': int(n_val), 'n_test': int(n_test),
        'meta_input_dim': N_META_INPUTS,
        'enable_vix_gated': bool(args.enable_vix_gated),
        'vix_data_root': str(args.vix_data_root),
        'val_selection_scores': val_selection_scores,
        'all_results': all_results,
    }, indent=2))
    logger.info(f"  Saved metrics: {metrics_path}")

    if best_method.startswith('logreg_'):
        model_path = out_dir / f'{args.target}_h{args.horizon}_stage3_cross_agent_meta{variant_suffix}.joblib'
        joblib.dump(saved_logregs[best_method], model_path)
        logger.info(f"  Saved model: {model_path}")
    elif best_method == 'mlp' and best_state is not None:
        model_path = out_dir / f'{args.target}_h{args.horizon}_stage3_cross_agent_meta_mlp{variant_suffix}.pt'
        torch.save({'model_state_dict': best_state, 'input_dim': N_META_INPUTS}, model_path)
        logger.info(f"  Saved model: {model_path}")
    elif best_method == 'vix_gated' and vix_artifact is not None:
        model_path = PATHS.stage3_cross_agent_vix_model(args.target, args.horizon)
        torch.save({
            'model_state_dict': vix_artifact['state_dict'],
            'agent_names': ALL_AGENTS,
            'vix_feat_dim': vix_artifact['vix_feat_dim'],
            'regime_emb_dim': vix_artifact['regime_emb_dim'],
            'fusion_hidden_dim': vix_artifact['fusion_hidden_dim'],
            'dropout': vix_artifact['dropout'],
            'threshold': all_results['vix_gated']['val']['threshold'],
        }, model_path)
        logger.info(f"  Saved model: {model_path}")

    if args.enable_vix_gated and vix_artifact is not None and best_method != 'vix_gated':
        model_path = PATHS.stage3_cross_agent_vix_model(args.target, args.horizon)
        torch.save({
            'model_state_dict': vix_artifact['state_dict'],
            'agent_names': ALL_AGENTS,
            'vix_feat_dim': vix_artifact['vix_feat_dim'],
            'regime_emb_dim': vix_artifact['regime_emb_dim'],
            'fusion_hidden_dim': vix_artifact['fusion_hidden_dim'],
            'dropout': vix_artifact['dropout'],
            'threshold': all_results['vix_gated']['val']['threshold'],
        }, model_path)
        logger.info(f"  Saved vix-gated candidate model: {model_path}")


if __name__ == '__main__':
    main()
