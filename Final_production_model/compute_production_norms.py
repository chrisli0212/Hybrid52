#!/usr/bin/env python3
"""
Compute Normalization Statistics from Production Snapshots

Generates norm_mean.npy and norm_std.npy for each symbol from recent snapshot data.
This enables proper z-score normalization for Stage 1 models.

Usage:
    python compute_production_norms.py
    python compute_production_norms.py --snapshots 1000  # Use more snapshots
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
HYBRID51_DIR = SCRIPT_DIR.parent / "Hybrid51" / "6. Hybrid51_new stage"
sys.path.insert(1, str(HYBRID51_DIR))

from prediction_service import FeatureBridge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=500,
                        help="Number of recent snapshots to use for statistics")
    parser.add_argument("--output-dir", type=str, default="/workspace/data/tier3_binary_v5",
                        help="Output directory for normalization stats")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("COMPUTING PRODUCTION NORMALIZATION STATISTICS")
    logger.info("=" * 70)
    
    # Find snapshot files
    snapshot_dir = SCRIPT_DIR / "daily_data" / "snapshots"
    if not snapshot_dir.exists():
        logger.error(f"Snapshot directory not found: {snapshot_dir}")
        return 1
    
    snapshot_files = sorted(glob(str(snapshot_dir / "snapshot_*.csv")))
    if not snapshot_files:
        logger.error("No snapshot files found")
        return 1
    
    # Use most recent N snapshots
    snapshot_files = snapshot_files[-args.snapshots:]
    logger.info(f"Using {len(snapshot_files)} most recent snapshots")
    logger.info(f"  First: {Path(snapshot_files[0]).name}")
    logger.info(f"  Last: {Path(snapshot_files[-1]).name}")
    
    # Initialize feature bridge
    logger.info("\nInitializing feature extractor...")
    bridge = FeatureBridge()
    
    # Extract features from all snapshots
    logger.info("\nExtracting features from snapshots...")
    all_features = []
    failed_count = 0
    
    for i, snap_file in enumerate(snapshot_files):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(snapshot_files)} snapshots...")
        
        try:
            df = pd.read_csv(snap_file)
            vec, quality = bridge.extract_325_features(df)
            all_features.append(vec)
        except Exception as e:
            logger.warning(f"  Failed to extract from {Path(snap_file).name}: {e}")
            failed_count += 1
    
    if not all_features:
        logger.error("No features extracted successfully")
        return 1
    
    logger.info(f"Successfully extracted features from {len(all_features)} snapshots")
    if failed_count > 0:
        logger.warning(f"  {failed_count} snapshots failed")
    
    # Stack features into matrix
    logger.info("\nComputing statistics...")
    features_matrix = np.stack(all_features, axis=0)  # (N, 325)
    logger.info(f"  Feature matrix shape: {features_matrix.shape}")
    
    # Compute mean and std
    mean = features_matrix.mean(axis=0)
    std = features_matrix.std(axis=0, ddof=1)  # Use sample std (ddof=1)
    
    # Identify zero-variance features
    zero_var_mask = (std < 1e-6)
    n_zero = zero_var_mask.sum()
    
    # Replace zero-std with 1.0 to avoid division by zero
    std[zero_var_mask] = 1.0
    
    logger.info(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    logger.info(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    logger.info(f"  Zero-variance features: {n_zero}/{len(mean)}")
    logger.info(f"  Non-zero features: {np.count_nonzero(mean)}/{len(mean)}")
    
    # Save for each symbol (using same stats for all symbols)
    # Note: In full training, each symbol would have separate stats
    # For production with limited data, shared stats are acceptable
    
    logger.info("\nSaving normalization statistics...")
    symbols = ["SPXW", "SPY", "QQQ", "IWM", "TLT"]
    
    for symbol in symbols:
        out_dir = Path(args.output_dir) / symbol / "horizon_30min"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(out_dir / 'norm_mean.npy', mean.astype(np.float32))
        np.save(out_dir / 'norm_std.npy', std.astype(np.float32))
        np.save(out_dir / 'zero_variance_mask.npy', zero_var_mask)
        
        logger.info(f"  ✓ {symbol}: Saved to {out_dir}")
    
    # Save metadata
    metadata = {
        'n_samples': len(all_features),
        'n_snapshots_attempted': len(snapshot_files),
        'n_failed': failed_count,
        'feat_dim': 325,
        'n_zero_variance': int(n_zero),
        'computation_date': pd.Timestamp.now().isoformat(),
    }
    
    metadata_path = Path(args.output_dir) / "normalization_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\n✓ Metadata saved to {metadata_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ NORMALIZATION STATISTICS COMPUTED AND SAVED")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("  1. Restart prediction service")
    logger.info("  2. Verify 'Normalization loaded' appears in logs")
    logger.info("  3. Check agent probabilities vary more (std >0.05)")
    logger.info("  4. Monitor predictions for 1 hour")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
