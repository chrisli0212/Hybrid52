"""
Hybrid51 Data Quality Checks (Task 13)
Validates preprocessed data and generates quality metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json

from .feature_config_v2 import TOTAL_FEATURES, FEATURE_GROUPS, FeatureGroup, get_feature_names


@dataclass
class FeatureQualityMetrics:
    name: str
    missing_pct: float = 0.0
    zero_pct: float = 0.0
    inf_pct: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    is_constant: bool = False
    quality_score: float = 1.0


@dataclass
class GroupQualityMetrics:
    name: str
    n_features: int
    avg_missing_pct: float = 0.0
    avg_zero_pct: float = 0.0
    n_constant: int = 0
    quality_score: float = 1.0


@dataclass
class DataQualityReport:
    n_samples: int = 0
    n_features: int = TOTAL_FEATURES
    overall_quality: float = 1.0
    missing_pct: float = 0.0
    zero_pct: float = 0.0
    inf_pct: float = 0.0
    feature_metrics: List[FeatureQualityMetrics] = field(default_factory=list)
    group_metrics: List[GroupQualityMetrics] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DataQualityChecker:
    def __init__(
        self,
        missing_threshold: float = 0.05,
        zero_threshold: float = 0.95,
        constant_threshold: float = 1e-10
    ):
        self.missing_threshold = missing_threshold
        self.zero_threshold = zero_threshold
        self.constant_threshold = constant_threshold
        self.feature_names = get_feature_names()
    
    def check_feature(self, values: np.ndarray, name: str) -> FeatureQualityMetrics:
        n = len(values)
        
        missing_count = np.isnan(values).sum()
        zero_count = (values == 0).sum()
        inf_count = np.isinf(values).sum()
        
        valid_values = values[~np.isnan(values) & ~np.isinf(values)]
        
        metrics = FeatureQualityMetrics(
            name=name,
            missing_pct=missing_count / n if n > 0 else 0,
            zero_pct=zero_count / n if n > 0 else 0,
            inf_pct=inf_count / n if n > 0 else 0,
        )
        
        if len(valid_values) > 0:
            metrics.mean = float(valid_values.mean())
            metrics.std = float(valid_values.std())
            metrics.min_val = float(valid_values.min())
            metrics.max_val = float(valid_values.max())
            metrics.is_constant = metrics.std < self.constant_threshold
        else:
            metrics.is_constant = True
        
        quality = 1.0
        quality -= metrics.missing_pct * 2
        quality -= metrics.inf_pct * 2
        if metrics.is_constant:
            quality -= 0.5
        metrics.quality_score = max(0, quality)
        
        return metrics
    
    def check_features(self, features: np.ndarray) -> DataQualityReport:
        n_samples, n_features = features.shape
        
        report = DataQualityReport(
            n_samples=n_samples,
            n_features=n_features
        )
        
        feature_metrics = []
        for i in range(n_features):
            name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            metrics = self.check_feature(features[:, i], name)
            feature_metrics.append(metrics)
        
        report.feature_metrics = feature_metrics
        
        for group in FeatureGroup:
            group_config = FEATURE_GROUPS[group]
            group_features = feature_metrics[group_config.start_idx:group_config.end_idx]
            
            n_group = len(group_features)
            avg_missing = np.mean([f.missing_pct for f in group_features])
            avg_zero = np.mean([f.zero_pct for f in group_features])
            n_constant = sum(1 for f in group_features if f.is_constant)
            avg_quality = np.mean([f.quality_score for f in group_features])
            
            group_metrics = GroupQualityMetrics(
                name=group_config.name,
                n_features=n_group,
                avg_missing_pct=avg_missing,
                avg_zero_pct=avg_zero,
                n_constant=n_constant,
                quality_score=avg_quality
            )
            report.group_metrics.append(group_metrics)
        
        report.missing_pct = np.mean([f.missing_pct for f in feature_metrics])
        report.zero_pct = np.mean([f.zero_pct for f in feature_metrics])
        report.inf_pct = np.mean([f.inf_pct for f in feature_metrics])
        report.overall_quality = np.mean([f.quality_score for f in feature_metrics])
        
        self._generate_warnings(report)
        
        return report
    
    def _generate_warnings(self, report: DataQualityReport):
        for fm in report.feature_metrics:
            if fm.missing_pct > self.missing_threshold:
                report.warnings.append(
                    f"Feature '{fm.name}' has {fm.missing_pct:.1%} missing values"
                )
            if fm.zero_pct > self.zero_threshold:
                report.warnings.append(
                    f"Feature '{fm.name}' is {fm.zero_pct:.1%} zeros"
                )
            if fm.is_constant:
                report.warnings.append(
                    f"Feature '{fm.name}' appears to be constant (std={fm.std:.2e})"
                )
            if fm.inf_pct > 0:
                report.errors.append(
                    f"Feature '{fm.name}' has {fm.inf_pct:.1%} infinite values"
                )
        
        if report.overall_quality < 0.8:
            report.errors.append(
                f"Overall data quality ({report.overall_quality:.2f}) is below threshold"
            )
    
    def check_sequence(
        self,
        X: np.ndarray,
        chain_2d: Optional[np.ndarray] = None
    ) -> Dict:
        n_samples, seq_len, n_features = X.shape
        
        X_flat = X.reshape(-1, n_features)
        flat_report = self.check_features(X_flat)
        
        result = {
            'flat_features': asdict(flat_report),
            'sequence_info': {
                'n_samples': n_samples,
                'seq_len': seq_len,
                'n_features': n_features
            }
        }
        
        if chain_2d is not None:
            n_s, n_g, n_strikes, n_t = chain_2d.shape
            result['chain_2d_info'] = {
                'shape': chain_2d.shape,
                'missing_pct': float(np.isnan(chain_2d).sum() / chain_2d.size),
                'zero_pct': float((chain_2d == 0).sum() / chain_2d.size),
                'mean': float(np.nanmean(chain_2d)),
                'std': float(np.nanstd(chain_2d))
            }
        
        return result


def validate_preprocessed_data(
    features: np.ndarray,
    chain_2d: Optional[np.ndarray] = None,
    output_path: Optional[str] = None
) -> DataQualityReport:
    checker = DataQualityChecker()
    
    if features.ndim == 3:
        result = checker.check_sequence(features, chain_2d)
        report = DataQualityReport(**result['flat_features'])
    else:
        report = checker.check_features(features)
    
    if output_path:
        save_quality_report(report, output_path)
    
    return report


def save_quality_report(report: DataQualityReport, output_path: str):
    report_dict = {
        'n_samples': report.n_samples,
        'n_features': report.n_features,
        'overall_quality': report.overall_quality,
        'missing_pct': report.missing_pct,
        'zero_pct': report.zero_pct,
        'inf_pct': report.inf_pct,
        'warnings': report.warnings,
        'errors': report.errors,
        'group_summary': [asdict(g) for g in report.group_metrics],
        'feature_summary': {
            fm.name: {
                'quality': fm.quality_score,
                'missing': fm.missing_pct,
                'constant': fm.is_constant
            }
            for fm in report.feature_metrics
            if fm.quality_score < 0.9 or fm.missing_pct > 0.01
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)


def print_quality_summary(report: DataQualityReport):
    print(f"\n{'='*60}")
    print("DATA QUALITY REPORT")
    print(f"{'='*60}")
    print(f"Samples: {report.n_samples}")
    print(f"Features: {report.n_features}")
    print(f"Overall Quality: {report.overall_quality:.2%}")
    print(f"Missing: {report.missing_pct:.2%}")
    print(f"Zeros: {report.zero_pct:.2%}")
    print(f"Infinite: {report.inf_pct:.2%}")
    
    print(f"\n{'Group Quality':30} {'Score':10} {'Missing':10} {'Constant'}")
    print("-" * 60)
    for gm in report.group_metrics:
        print(f"{gm.name:30} {gm.quality_score:10.2%} {gm.avg_missing_pct:10.2%} {gm.n_constant}")
    
    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings[:10]:
            print(f"  - {w}")
        if len(report.warnings) > 10:
            print(f"  ... and {len(report.warnings) - 10} more")
    
    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for e in report.errors:
            print(f"  - {e}")
    
    print(f"{'='*60}\n")
