#!/usr/bin/env python3
"""
Lightning training entry point.
Wraps existing BinaryIndependentAgent + all data/metric logic.
Uses: fp16 AMP, gradient clipping, early stop on val_mcc, progress bar.
"""
import argparse, json, logging, sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.stage1.train_binary_agents_v2 import (
    BinaryIndependentAgent, SequenceWithOptionalChainDataset,
    verify_data_before_training, optimize_threshold, evaluate_model,
    ALL_AGENTS, ALL_SYMBOLS, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT,
    RETURN_SCALE
)
from scripts.stage1.lit_agent_module import LitAgent
from config.feature_subsets import AGENT_FEATURE_SUBSETS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',      default='SPXW')
    parser.add_argument('--horizon',     type=int, default=30)
    parser.add_argument('--agents',      nargs='+', default=['A','B','K','C','T','Q'])
    parser.add_argument('--epochs',      type=int, default=80)
    parser.add_argument('--batch-size',  type=int, default=512)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--patience',    type=int, default=15)
    parser.add_argument('--accum-steps', type=int, default=4)
    parser.add_argument('--f1-weight',   type=float, default=0.3)
    parser.add_argument('--noise-sigma', type=float, default=0.0)
    parser.add_argument('--use-mixup',   action='store_true')
    parser.add_argument('--threshold-objective', default='balanced_acc')
    parser.add_argument('--data-root',   default=str(DEFAULT_DATA_ROOT))
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--precision',   default='16-mixed',
                        help='Lightning precision: 16-mixed | bf16-mixed | 32')
    parser.add_argument('--no-feature-subset', action='store_true')
    args = parser.parse_args()

    device_str = 'gpu' if torch.cuda.is_available() else 'cpu'
    data_root   = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    symbol   = args.symbol
    horizon  = args.horizon
    data_dir = data_root / symbol / f"horizon_{horizon}min"

    # ── Load data ────────────────────────────────────────────
    train_seq     = np.load(data_dir / 'train_sequences.npy')
    train_labels  = np.load(data_dir / 'train_labels.npy')
    train_returns = np.load(data_dir / 'train_returns.npy')
    val_seq       = np.load(data_dir / 'val_sequences.npy')
    val_labels    = np.load(data_dir / 'val_labels.npy')
    val_returns   = np.load(data_dir / 'val_returns.npy')
    test_seq      = np.load(data_dir / 'test_sequences.npy')
    test_labels   = np.load(data_dir / 'test_labels.npy')
    test_returns  = np.load(data_dir / 'test_returns.npy')

    feat_dim = train_seq.shape[2]

    # Dead-feature mask
    X    = train_seq.reshape(-1, feat_dim)
    std  = X.std(axis=0)
    nzr  = (np.abs(X) > 0).mean(axis=0)
    mask = ((std > 1e-8) & (nzr > 1e-4)).astype(np.float32)
    m3   = mask[None, None, :]
    train_seq *= m3; val_seq *= m3; test_seq *= m3

    # Norm stats
    nm_path = data_dir / 'norm_mean.npy'
    ns_path = data_dir / 'norm_std.npy'
    norm_mean = np.load(nm_path) if nm_path.exists() else None
    norm_std  = np.load(ns_path) if ns_path.exists() else None

    verify_data_before_training(data_dir, symbol, horizon)

    all_results = {}

    for agent_type in args.agents:
        logger.info(f"\n{'='*60}\nAgent {agent_type}\n{'='*60}")

        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = BinaryIndependentAgent(
            agent_type=agent_type, feat_dim=feat_dim,
            temporal_dim=128, dropout=0.2, mode='classifier',
            use_feature_subset=not args.no_feature_subset,
        ).to(torch_device)

        positive_prior = float(np.mean(train_labels))
        focal_gamma    = 1.0 if agent_type in ('T', 'Q') else 2.0

        lit_model = LitAgent(
            model=model,
            lr=args.lr,
            f1_weight=args.f1_weight,
            focal_gamma=focal_gamma,
            positive_class_prior=positive_prior,
            noise_sigma=args.noise_sigma,
            use_mixup=args.use_mixup,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        # DataLoaders — Dataset does z-score norm in __getitem__
        train_ds = SequenceWithOptionalChainDataset(
            train_seq, train_labels.astype(np.float32),
            norm_mean=norm_mean, norm_std=norm_std,
        )
        val_ds = SequenceWithOptionalChainDataset(
            val_seq, val_labels.astype(np.float32),
            # norm applied manually in validation_step via _normalize()
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=2048,
                                  shuffle=False, num_workers=4, pin_memory=True)

        # ── Callbacks ────────────────────────────────────────
        ckpt_cb = ModelCheckpoint(
            dirpath=str(output_root),
            filename=f'{symbol}_agent{agent_type}_h{horizon}',
            monitor='val_mcc', mode='max', save_top_k=1,
        )
        early_cb = EarlyStopping(
            monitor='val_mcc', mode='max',
            patience=args.patience, min_delta=0.005,
        )
        lr_cb = LearningRateMonitor(logging_interval='epoch')

        csv_log = CSVLogger(str(output_root / 'logs'), name=f'agent_{agent_type}')

        # ── Trainer ──────────────────────────────────────────
        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator=device_str,
            devices=1,
            precision=args.precision,          # fp16 AMP by default
            accumulate_grad_batches=args.accum_steps,
            gradient_clip_val=1.0,
            callbacks=[ckpt_cb, early_cb, lr_cb],
            logger=csv_log,
            log_every_n_steps=50,
            enable_progress_bar=True,
        )

        trainer.fit(lit_model, train_loader, val_loader)

        logger.info(f"  Best val_mcc={lit_model.best_mcc:.4f}  best_auc={lit_model.best_auc:.4f}")

        # ── Threshold + test eval (reuse existing logic) ─────
        best_model = lit_model.model
        best_model.eval()

        opt_threshold, opt_val_f1, platt_scaler, invert_signal = optimize_threshold(
            best_model, val_seq, val_labels, torch_device,
            norm_mean=norm_mean, norm_std=norm_std,
            threshold_objective=args.threshold_objective,
        )

        test_metrics = evaluate_model(
            best_model, test_seq, test_labels, test_returns, torch_device,
            threshold=opt_threshold, platt_scaler=platt_scaler,
            norm_mean=norm_mean, norm_std=norm_std,
            invert_signal=invert_signal,
        )

        logger.info(
            f"  Test: acc={test_metrics['accuracy']:.4f}  f1={test_metrics['f1']:.4f}  "
            f"auc={test_metrics['auc']:.4f}  ic={test_metrics['ic']:.4f}  "
            f"thr={test_metrics['threshold']:.3f}"
        )

        all_results[f"agent_{agent_type}_classifier"] = test_metrics
        torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY: {symbol} | Horizon={horizon}min")
    logger.info(f"{'='*80}")
    logger.info(f"{'Agent':>6} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'IC':>8} {'Thr':>6}")
    logger.info("-" * 50)
    for a in args.agents:
        r = all_results.get(f"agent_{a}_classifier", {})
        if 'accuracy' in r:
            logger.info(f"{a:>6} {r['accuracy']:>10.4f} {r['f1']:>8.4f} "
                        f"{r['auc']:>8.4f} {r['ic']:>8.4f} {r.get('threshold',0.5):>6.3f}")

    with open(output_root / f'{symbol}_h{horizon}_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
