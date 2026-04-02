#!/usr/bin/env python3
"""
Stage 3 v2: Meta-Learner — combines Stage 2 pair predictions into final signal.

Key changes from v1:
- Replaces MLP meta-learner with Logistic Regression (C=0.01)
- Fewer params = less overfitting (MLP had lowest accuracy despite highest F1)
- Trains on val set only (no data leakage)
- VIXW pair already dropped in Stage 2

Usage:
    python scripts/stage3/train_stage3_meta.py --target SPXW --horizon 30
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score, precision_score, recall_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from hybrid55_utils import ArtifactPaths
from hybrid55_utils.artifacts import DEFAULT_TRAINING_HORIZON_MINUTES

PATHS = ArtifactPaths.default()
STAGE2_RESULTS = PATHS.stage2_results
OUTPUT_ROOT = PATHS.stage3_results

PAIR_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'TLT']  # No VIXW
MAIN_PAIRS = ['SPY', 'QQQ', 'IWM', 'TLT']
OPTIONAL_PAIR = 'VIXW'


class Stage3MetaLearner:
    """
    Logistic Regression meta-learner for Stage 3.
    Takes Stage 2 pair probabilities + summary statistics as input.
    """
    def __init__(self, C: float = 0.01, max_iter: int = 1000):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver='lbfgs',
            class_weight='balanced',
        )
        self.C = C

    def fit(self, X_val, y_val):
        """Train on validation set only (prevents data leakage)."""
        self.model.fit(X_val, y_val)
        train_preds = self.model.predict(X_val)
        train_acc = accuracy_score(y_val, train_preds)
        train_f1 = f1_score(y_val, train_preds, average='binary')
        logger.info(f"  Meta fit: val_acc={train_acc:.4f} val_f1={train_f1:.4f}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


def build_meta_features(pair_probs: dict, pair_stats: dict = None):
    """
    Build meta features from Stage 2 pair predictions.
    Features: [pair_prob_spy, pair_prob_qqq, pair_prob_iwm, pair_prob_tlt,
               mean_prob, std_prob, max_prob, min_prob, agreement_ratio]
    """
    probs = np.column_stack([pair_probs[sym] for sym in PAIR_SYMBOLS if sym in pair_probs])

    # Summary statistics
    mean_prob = probs.mean(axis=1, keepdims=True)
    std_prob = probs.std(axis=1, keepdims=True)
    max_prob = probs.max(axis=1, keepdims=True)
    min_prob = probs.min(axis=1, keepdims=True)

    # Agreement ratio (how many pairs agree on direction)
    binary_preds = (probs > 0.5).astype(float)
    agreement = binary_preds.mean(axis=1, keepdims=True)

    features = np.hstack([probs, mean_prob, std_prob, max_prob, min_prob, agreement])
    return features


def build_enriched_meta_features(
    pair_probs: dict,
    core_logits: np.ndarray,
    use_pairs: list[str],
    optional_pair: str = OPTIONAL_PAIR,
) -> np.ndarray:
    probs_list = []
    for sym in use_pairs:
        probs_list.append(pair_probs[sym].reshape(-1))

    if optional_pair in pair_probs:
        probs_list.append(pair_probs[optional_pair].reshape(-1))
    else:
        probs_list.append(np.full(len(probs_list[0]), 0.5, dtype=np.float32))

    raw_probs = np.stack(probs_list, axis=1).astype(np.float32)

    mean_prob = raw_probs.mean(axis=1, keepdims=True)
    std_prob = raw_probs.std(axis=1, keepdims=True)
    spread = (raw_probs.max(axis=1, keepdims=True) - raw_probs.min(axis=1, keepdims=True))

    d_spy_qqq = np.abs(raw_probs[:, 0:1] - raw_probs[:, 1:2])
    d_spy_tlt = np.abs(raw_probs[:, 0:1] - raw_probs[:, 3:4])
    d_qqq_iwm = np.abs(raw_probs[:, 1:2] - raw_probs[:, 2:3])
    d_iwm_tlt = np.abs(raw_probs[:, 2:3] - raw_probs[:, 3:4])

    agree_up = (raw_probs > 0.5).sum(axis=1, keepdims=True).astype(np.float32) / float(raw_probs.shape[1])

    core_mean = core_logits.mean(axis=1, keepdims=True).astype(np.float32)
    core_std = core_logits.std(axis=1, keepdims=True).astype(np.float32)

    features = np.concatenate(
        [
            raw_probs,
            mean_prob,
            std_prob,
            spread,
            d_spy_qqq,
            d_spy_tlt,
            d_qqq_iwm,
            d_iwm_tlt,
            agree_up,
            core_mean,
            core_std,
        ],
        axis=1,
    ).astype(np.float32)

    if features.shape[1] != 15:
        raise RuntimeError(f"Expected 15 enriched meta features, got {features.shape[1]}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _align_list_to_min_length(arrs: list[np.ndarray]) -> list[np.ndarray]:
    if not arrs:
        return arrs
    n = min(len(a) for a in arrs)
    return [a[:n] for a in arrs]


def _sweep_threshold_for_f1(probs: np.ndarray, labels: np.ndarray):
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.30, 0.66, 0.01):
        f1 = f1_score(labels, (probs > thr).astype(np.int64), average='binary')
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


def _compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float):
    preds = (probs > threshold).astype(np.int64)
    out = {
        'accuracy': float(accuracy_score(labels, preds)),
        'brier': float(brier_score_loss(labels, probs)),
        'f1': float(f1_score(labels, preds, average='binary')),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'threshold': float(threshold),
    }
    try:
        out['auc'] = float(roc_auc_score(labels, probs))
    except Exception:
        out['auc'] = 0.5
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='SPXW')
    parser.add_argument('--horizon', type=int, default=DEFAULT_TRAINING_HORIZON_MINUTES)
    parser.add_argument('--C', type=float, default=0.01, help='Regularization strength')
    parser.add_argument('--pairs', nargs='+', default=None, help='Pair symbols (default: built-in list)')
    parser.add_argument('--all-pairs', action='store_true', help='Use all default pair symbols')
    parser.add_argument('--meta', choices=['logreg', 'mlp', 'auto'], default='auto')
    parser.add_argument('--meta-epochs', type=int, default=100)
    parser.add_argument('--meta-batch-size', type=int, default=2048)
    parser.add_argument('--meta-lr', type=float, default=1e-3)
    parser.add_argument('--meta-hidden-dim', type=int, default=32)
    parser.add_argument('--meta-dropout', type=float, default=0.3)
    parser.add_argument('--meta-patience', type=int, default=15)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*70}")
    logger.info(f"STAGE 3 v2: Meta-Learner (LogReg C={args.C})")
    logger.info(f"Target={args.target} | Horizon={args.horizon}min")
    logger.info(f"Pairs: {PAIR_SYMBOLS} (VIXW dropped)")
    logger.info(f"{'='*70}")

    if args.all_pairs and args.pairs is not None:
        raise SystemExit("Use only one of --all-pairs or --pairs")

    if args.all_pairs:
        pairs = PAIR_SYMBOLS
    elif args.pairs is not None and len(args.pairs) > 0:
        pairs = args.pairs
    else:
        pairs = PAIR_SYMBOLS

    val_probs_by_pair = {}
    test_probs_by_pair = {}
    val_core_logits_by_pair = {}
    test_core_logits_by_pair = {}
    val_labels_list = []
    test_labels_list = []

    for sym in pairs:
        npz_path = STAGE2_RESULTS / f'{args.target}_{sym}_h{args.horizon}_pair_probs.npz'
        if not npz_path.exists():
            raise FileNotFoundError(f"Stage2 probs not found: {npz_path}")

        data = np.load(npz_path)
        val_probs_by_pair[sym] = data['val_probs'].astype(np.float32)
        test_probs_by_pair[sym] = data['test_probs'].astype(np.float32)
        if 'val_core_logits' in data and 'test_core_logits' in data:
            val_core_logits_by_pair[sym] = data['val_core_logits'].astype(np.float32)
            test_core_logits_by_pair[sym] = data['test_core_logits'].astype(np.float32)
        else:
            raise RuntimeError(f"Missing core logits arrays in {npz_path}. Re-run Stage2 to generate val_core_logits/test_core_logits.")
        val_labels_list.append(data['val_labels'].astype(np.int64))
        test_labels_list.append(data['test_labels'].astype(np.int64))

    val_probs_list = _align_list_to_min_length([val_probs_by_pair[s] for s in pairs])
    test_probs_list = _align_list_to_min_length([test_probs_by_pair[s] for s in pairs])
    val_labels_aligned = _align_list_to_min_length(val_labels_list)[0]
    test_labels_aligned = _align_list_to_min_length(test_labels_list)[0]

    val_probs_by_pair = {s: a for s, a in zip(pairs, val_probs_list)}
    test_probs_by_pair = {s: a for s, a in zip(pairs, test_probs_list)}

    val_core_logits_list = _align_list_to_min_length([val_core_logits_by_pair[s] for s in pairs])
    test_core_logits_list = _align_list_to_min_length([test_core_logits_by_pair[s] for s in pairs])
    val_core_logits_by_pair = {s: a for s, a in zip(pairs, val_core_logits_list)}
    test_core_logits_by_pair = {s: a for s, a in zip(pairs, test_core_logits_list)}

    val_core_logits = val_core_logits_by_pair[pairs[0]]
    test_core_logits = test_core_logits_by_pair[pairs[0]]

    val_pair_probs_for_enriched = {k: val_probs_by_pair[k] for k in val_probs_by_pair}
    test_pair_probs_for_enriched = {k: test_probs_by_pair[k] for k in test_probs_by_pair}

    use_pairs = [s for s in MAIN_PAIRS if s in val_pair_probs_for_enriched]
    missing_main = [s for s in MAIN_PAIRS if s not in use_pairs]
    if missing_main:
        raise RuntimeError(f"Missing required main pairs for Stage3 enriched features: {missing_main}")

    X_val = build_enriched_meta_features(val_pair_probs_for_enriched, val_core_logits, use_pairs=use_pairs)
    X_test = build_enriched_meta_features(test_pair_probs_for_enriched, test_core_logits, use_pairs=use_pairs)

    logger.info(f"Meta features: val={X_val.shape} test={X_test.shape}")

    all_results = {}
    val_selection_scores = {}
    trained_logregs = {}

    val_avg4 = np.stack([val_pair_probs_for_enriched[sym].reshape(-1) for sym in MAIN_PAIRS], axis=1).mean(axis=1)
    test_avg4 = np.stack([test_pair_probs_for_enriched[sym].reshape(-1) for sym in MAIN_PAIRS], axis=1).mean(axis=1)
    avg4_thr, _ = _sweep_threshold_for_f1(val_avg4, val_labels_aligned)
    avg4_val = _compute_binary_metrics(val_labels_aligned, val_avg4, avg4_thr)
    avg4_test = _compute_binary_metrics(test_labels_aligned, test_avg4, avg4_thr)
    all_results['avg_4_pairs'] = {
        'name': 'Simple Avg (4 pairs)',
        'val': avg4_val,
        'test': avg4_test,
    }
    val_selection_scores['avg_4_pairs'] = avg4_val['accuracy'] + avg4_val['f1']

    def run_logreg(C: float, name: str):
        m = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', class_weight='balanced')
        m.fit(X_val, val_labels_aligned)
        val_p = m.predict_proba(X_val)[:, 1]
        test_p = m.predict_proba(X_test)[:, 1]
        thr, _ = _sweep_threshold_for_f1(val_p, val_labels_aligned)
        val_m = _compute_binary_metrics(val_labels_aligned, val_p, thr)
        test_m = _compute_binary_metrics(test_labels_aligned, test_p, thr)
        return m, val_p, test_p, thr, val_m, test_m

    for c in [args.C, 0.1, 0.01]:
        key = f'logreg_C={c}'
        m, val_p, test_p, thr, val_m, test_m = run_logreg(c, key)
        trained_logregs[key] = m
        all_results[key] = {
            'name': f'LogReg (C={c})',
            'val': val_m,
            'test': test_m,
        }
        val_selection_scores[key] = val_m['accuracy'] + val_m['f1']

    best_method = max(val_selection_scores, key=val_selection_scores.get)
    best_score = val_selection_scores[best_method]

    mlp_state = None
    mlp_test_p = None
    if args.meta in ('mlp', 'auto'):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except Exception as e:
            if args.meta == 'mlp':
                raise
            logger.warning(f"Torch not available for MLP meta ({e}). Skipping MLP.")
            torch = None

        if torch is not None:
            class _BinaryFocalLoss(nn.Module):
                def __init__(self, gamma=2.0, alpha=0.5):
                    super().__init__()
                    self.gamma = gamma
                    self.alpha = alpha

                def forward(self, logits, targets):
                    p = torch.sigmoid(logits)
                    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                    p_t = p * targets + (1 - p) * (1 - targets)
                    focal_weight = (1 - p_t) ** self.gamma
                    alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                    return (alpha_t * focal_weight * ce_loss).mean()

            class _Stage3MetaMLP(nn.Module):
                def __init__(self, input_dim=15, hidden_dim=32, dropout=0.3):
                    super().__init__()
                    self.meta = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.GELU(),
                        nn.Dropout(dropout * 0.25),
                        nn.Linear(hidden_dim // 2, 1),
                    )

                def forward(self, x):
                    return self.meta(x).squeeze(-1)

            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
            rng = np.random.RandomState(42)
            n_val = len(val_labels_aligned)
            perm = rng.permutation(n_val)
            split = int(n_val * 0.8)
            tr_idx = perm[:split]
            vm_idx = perm[split:]

            X_tr = torch.from_numpy(X_val[tr_idx]).float().to(device)
            y_tr = torch.from_numpy(val_labels_aligned[tr_idx].astype(np.float32)).float().to(device)
            X_vm = torch.from_numpy(X_val[vm_idx]).float().to(device)
            y_vm = torch.from_numpy(val_labels_aligned[vm_idx].astype(np.float32)).float().to(device)

            model = _Stage3MetaMLP(input_dim=15, hidden_dim=args.meta_hidden_dim, dropout=args.meta_dropout).to(device)
            alpha = float(val_labels_aligned[tr_idx].mean())
            crit = _BinaryFocalLoss(gamma=2.0, alpha=alpha)
            opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.meta_epochs, eta_min=1e-6)

            best_state = None
            best_vm_score = -1.0
            patience = 0

            for epoch in range(args.meta_epochs):
                model.train()
                perm_t = torch.randperm(len(X_tr), device=device)
                total_loss = 0.0
                n_batches = 0
                bs = args.meta_batch_size
                for i in range(0, len(X_tr), bs):
                    idx = perm_t[i:i + bs]
                    opt.zero_grad()
                    pred = model(X_tr[idx])
                    loss = crit(pred, y_tr[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total_loss += float(loss.item())
                    n_batches += 1
                sched.step()

                model.eval()
                with torch.no_grad():
                    vm_prob = torch.sigmoid(model(X_vm)).detach().cpu().numpy().astype(np.float32)
                vm_true = y_vm.detach().cpu().numpy().astype(np.int64)
                vm_pred = (vm_prob > 0.5).astype(np.int64)
                vm_acc = float(accuracy_score(vm_true, vm_pred))
                vm_f1 = float(f1_score(vm_true, vm_pred, average='binary'))
                vm_score = vm_acc + vm_f1

                if vm_score > best_vm_score:
                    best_vm_score = vm_score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

                if patience >= args.meta_patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            with torch.no_grad():
                test_p = torch.sigmoid(model(torch.from_numpy(X_test).float().to(device))).detach().cpu().numpy().astype(np.float32)
                val_p = torch.sigmoid(model(torch.from_numpy(X_val).float().to(device))).detach().cpu().numpy().astype(np.float32)

            thr, _ = _sweep_threshold_for_f1(val_p, val_labels_aligned)
            val_m = _compute_binary_metrics(val_labels_aligned, val_p, thr)
            test_m = _compute_binary_metrics(test_labels_aligned, test_p, thr)
            all_results['mlp_v2'] = {
                'name': 'MLP v2',
                'val': val_m,
                'test': test_m,
            }
            mlp_state = best_state
            mlp_test_p = test_p

            mlp_score = val_m['accuracy'] + val_m['f1']
            val_selection_scores['mlp_v2'] = mlp_score
            if mlp_score > best_score:
                best_score = mlp_score
                best_method = 'mlp_v2'

    logger.info("Results:")
    for k, v in all_results.items():
        vm = v['val']
        tm = v['test']
        logger.info(
            f"  {k}: val(acc={vm['accuracy']:.4f}, f1={vm['f1']:.4f}, auc={vm['auc']:.4f}, thr={vm['threshold']:.2f}) "
            f"test(acc={tm['accuracy']:.4f}, f1={tm['f1']:.4f}, auc={tm['auc']:.4f}, thr={tm['threshold']:.2f})"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_ROOT / f'{args.target}_h{args.horizon}_stage3_meta_metrics.json'
    save = {
        'best_method': best_method,
        'feat_dim': int(X_val.shape[1]),
        'n_val_train': int(len(val_labels_aligned) * 0.8),
        'n_val_meta': int(len(val_labels_aligned) - int(len(val_labels_aligned) * 0.8)),
        'n_test': int(len(test_labels_aligned)),
        'val_selection_scores': val_selection_scores,
        'all_results': all_results,
    }
    metrics_path.write_text(json.dumps(save, indent=2))
    logger.info(f"Saved Stage3 metrics: {metrics_path}")

    if best_method.startswith('logreg_C='):
        model_path = OUTPUT_ROOT / f'{args.target}_h{args.horizon}_stage3_meta.joblib'
        joblib.dump(trained_logregs[best_method], model_path)
        logger.info(f"Saved Stage3 model: {model_path}")
    elif best_method == 'mlp_v2':
        if mlp_state is not None:
            import torch
            model_path = OUTPUT_ROOT / f'{args.target}_h{args.horizon}_stage3_meta_mlp.pt'
            torch.save({'model_state_dict': mlp_state, 'feat_dim': int(X_val.shape[1])}, model_path)
            logger.info(f"Saved Stage3 model: {model_path}")


if __name__ == '__main__':
    main()
