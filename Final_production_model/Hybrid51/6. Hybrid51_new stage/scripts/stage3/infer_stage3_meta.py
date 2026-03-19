#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))
from hybrid51_utils import ArtifactPaths
from hybrid51_models.regime_gated_meta_model import RegimeGatedProbFusion

PATHS = ArtifactPaths.default()

MAIN_PAIRS = ['SPY', 'QQQ', 'IWM', 'TLT']
ALL_AGENTS = ['A', 'B', 'C', 'K', 'T', 'Q', '2D']


def _align_list_to_min_length(arrs: list[np.ndarray]) -> list[np.ndarray]:
    if not arrs:
        return arrs
    n = min(len(a) for a in arrs)
    return [a[:n] for a in arrs]


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


def build_enriched_meta_features(pair_probs: dict, core_logits: np.ndarray, use_pairs: list[str]) -> np.ndarray:
    raw_probs = np.stack([pair_probs[s].reshape(-1) for s in use_pairs], axis=1).astype(np.float32)

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
        raise RuntimeError(f"Expected 15 features, got {features.shape[1]}")

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['legacy', 'cross-agent-vix'], default='legacy')
    ap.add_argument('--target', default='SPXW')
    ap.add_argument('--horizon', type=int, default=15)
    ap.add_argument('--pairs', nargs='+', default=None)
    ap.add_argument('--agents', nargs='+', default=ALL_AGENTS)
    ap.add_argument('--probs-variant', choices=['traditional', 'tlt_gated'], default='traditional',
                    help='Which Stage 2 probs to load for cross-agent-vix mode: '
                         'traditional (*_cross_probs.npz) or tlt_gated (*_tlt_gated_probs.npz).')
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--threshold', type=float, default=None, help='Override threshold; otherwise read from saved metrics if present')
    ap.add_argument('--vix-data-root', default='/workspace/data/tier3_vix_v4/VIXW')
    ap.add_argument('--vix-model', default=None,
                    help='Optional cross-agent VIX-gated model path. Defaults to ArtifactPaths helper.')
    ap.add_argument('--save-npz', default=None)
    args = ap.parse_args()

    if args.mode == 'cross-agent-vix':
        ckpt_path = Path(args.vix_model) if args.vix_model else PATHS.stage3_cross_agent_vix_model(args.target, args.horizon)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"VIX-gated Stage3 model not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        threshold = float(args.threshold) if args.threshold is not None else float(ckpt.get('threshold', 0.5))

        probs_by_agent = {}
        labels_list = []
        for ag in args.agents:
            if args.probs_variant == 'tlt_gated':
                npz_path = PATHS.stage2_tlt_gated_probs(args.target, ag, args.horizon)
            else:
                npz_path = PATHS.stage2_per_agent_probs(args.target, ag, args.horizon)
            if not npz_path.exists():
                logger.warning(f"Stage2 cross probs missing for agent {ag}: {npz_path}")
                continue
            data = np.load(npz_path)
            probs_by_agent[ag] = data[f'{args.split}_probs'].astype(np.float32).reshape(-1)
            labels_list.append(data[f'{args.split}_labels'].astype(np.int64).reshape(-1))

        if not probs_by_agent:
            raise RuntimeError("No Stage2 per-agent probabilities found for cross-agent-vix mode")

        n = min(len(v) for v in probs_by_agent.values())
        if labels_list:
            n = min(n, min(len(x) for x in labels_list))
        labels = labels_list[0][:n] if labels_list else np.zeros(n, dtype=np.int64)

        cols = []
        for ag in ALL_AGENTS:
            if ag in probs_by_agent:
                cols.append(probs_by_agent[ag][:n].reshape(-1, 1))
            else:
                cols.append(np.full((n, 1), 0.5, dtype=np.float32))
        agent_mat = np.concatenate(cols, axis=1).astype(np.float32)
        vix_feat = _load_vix_features(Path(args.vix_data_root), args.split, n)

        model = RegimeGatedProbFusion(
            agent_names=ckpt.get('agent_names', ALL_AGENTS),
            vix_feat_dim=int(ckpt.get('vix_feat_dim', vix_feat.shape[1])),
            regime_emb_dim=int(ckpt.get('regime_emb_dim', 32)),
            fusion_hidden_dim=int(ckpt.get('fusion_hidden_dim', 64)),
            dropout=float(ckpt.get('dropout', 0.2)),
        )
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        model.eval()

        with torch.no_grad():
            logits, gates, _ = model(torch.from_numpy(agent_mat), torch.from_numpy(vix_feat))
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            gates_np = gates.cpu().numpy().astype(np.float32)

        preds = (probs > threshold).astype(np.int64)
        acc = float((preds == labels).mean())
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        f1 = float((2 * tp) / max(1.0, (2 * tp + fp + fn)))

        logger.info(
            f"Stage3 cross-agent-vix split={args.split} n={len(labels)} "
            f"thr={threshold:.2f} acc={acc:.4f} f1={f1:.4f}"
        )

        if args.save_npz:
            out_p = Path(args.save_npz)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_p,
                probs=probs,
                preds=preds,
                labels=labels,
                gates=gates_np,
                threshold=float(threshold),
                target=args.target,
                horizon=int(args.horizon),
                split=args.split,
                mode=args.mode,
            )
            logger.info(f"Saved: {out_p}")
        return

    raise NotImplementedError(
        "Legacy mode uses old pair-based Stage 2 paths that are incompatible with "
        "the current cross-symbol pipeline. Use --mode cross-agent-vix instead."
    )

    pairs = args.pairs if args.pairs else MAIN_PAIRS

    model_path = PATHS.stage3_logreg_model(args.target, args.horizon)
    metrics_path = PATHS.stage3_metrics(args.target, args.horizon)

    if not model_path.exists():
        raise FileNotFoundError(f"Stage3 model not found: {model_path}")

    threshold = args.threshold
    if threshold is None and metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text())
            best_method = m.get('best_method')
            if best_method and best_method in m.get('all_results', {}):
                threshold = float(m['all_results'][best_method]['val']['threshold'])
        except Exception:
            threshold = None

    if threshold is None:
        threshold = 0.5

    pair_probs = {}
    labels_list = []
    core_logits_list = []

    for sym in pairs:
        npz_path = PATHS.stage2_probs(args.target, sym, args.horizon)
        if not npz_path.exists():
            raise FileNotFoundError(f"Stage2 probs not found: {npz_path}")

        data = np.load(npz_path)
        probs = data[f'{args.split}_probs'].astype(np.float32).reshape(-1)
        labels = data[f'{args.split}_labels'].astype(np.int64).reshape(-1)
        core = data[f'{args.split}_core_logits'].astype(np.float32)

        pair_probs[sym] = probs
        labels_list.append(labels)
        core_logits_list.append(core)

    labels = _align_list_to_min_length(labels_list)[0]
    core_logits = _align_list_to_min_length(core_logits_list)[0]
    for sym in pairs:
        pair_probs[sym] = pair_probs[sym][: len(labels)]

    missing = [p for p in MAIN_PAIRS if p not in pair_probs]
    if missing:
        raise RuntimeError(f"Missing required pairs for enriched features: {missing}")

    X = build_enriched_meta_features(pair_probs, core_logits, use_pairs=MAIN_PAIRS)

    model = joblib.load(model_path)
    probs = model.predict_proba(X)[:, 1].astype(np.float32)
    preds = (probs > threshold).astype(np.int64)

    acc = float((preds == labels).mean())
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    f1 = float((2 * tp) / max(1.0, (2 * tp + fp + fn)))

    logger.info(f"Stage3 inference split={args.split} n={len(labels)} thr={threshold:.2f} acc={acc:.4f} f1={f1:.4f}")

    if args.save_npz:
        out_p = Path(args.save_npz)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_p,
            probs=probs,
            preds=preds,
            labels=labels,
            threshold=float(threshold),
            target=args.target,
            horizon=int(args.horizon),
            split=args.split,
        )
        logger.info(f"Saved: {out_p}")


if __name__ == '__main__':
    main()
