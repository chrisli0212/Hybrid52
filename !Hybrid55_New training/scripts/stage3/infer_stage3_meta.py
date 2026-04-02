#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))
from hybrid55_utils import ArtifactPaths
from hybrid55_utils.artifacts import DEFAULT_TRAINING_HORIZON_MINUTES
from hybrid55_models.regime_gated_meta_model import RegimeGatedProbFusion

PATHS = ArtifactPaths.default()

ALL_AGENTS = ['A', 'B', 'C', 'K', 'TQ', 'H', 'M', '2D']


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cross-agent-vix'], default='cross-agent-vix')
    ap.add_argument('--target', default='SPXW')
    ap.add_argument('--horizon', type=int, default=DEFAULT_TRAINING_HORIZON_MINUTES)
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
        "Only cross-agent-vix mode is supported in Hybrid55."
    )


if __name__ == '__main__':
    main()
