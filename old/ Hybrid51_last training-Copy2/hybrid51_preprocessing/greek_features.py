"""
Hybrid51 Greek Feature Extraction (Task 2)
Extracts 75 Greek features from R2 data using delta bucketing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .data_validation import get_usable_greek_columns, get_excluded_columns, METADATA_COLUMNS
from .feature_config import (
    DELTA_BUCKETS, GREEKS_FOR_BUCKETING, ATM_GREEKS,
    FeatureGroup, FEATURE_GROUPS
)


def calculate_absolute_delta(df: pd.DataFrame) -> pd.Series:
    return df['delta'].abs()


def assign_delta_bucket(abs_delta: float) -> str:
    for bucket_name, low, high in DELTA_BUCKETS:
        if low <= abs_delta < high:
            return bucket_name
    return 'deep_itm' if abs_delta >= 1.0 else 'deep_otm'


def get_atm_options(df: pd.DataFrame, target_delta: float = 0.5, top_n: int = 5) -> pd.DataFrame:
    df = df.copy()
    df['delta_dist'] = (df['delta'].abs() - target_delta).abs()
    return df.nsmallest(top_n, 'delta_dist')


def aggregate_greeks_by_bucket(
    df: pd.DataFrame,
    greeks: List[str] = None,
    agg_func: str = 'mean'
) -> Dict[str, Dict[str, float]]:
    if greeks is None:
        greeks = GREEKS_FOR_BUCKETING
    
    df = df.copy()
    df['abs_delta'] = calculate_absolute_delta(df)
    df['bucket'] = df['abs_delta'].apply(assign_delta_bucket)
    
    valid_greeks = [g for g in greeks if g in df.columns]
    
    result = {}
    for bucket_name, _, _ in DELTA_BUCKETS:
        bucket_df = df[df['bucket'] == bucket_name]
        result[bucket_name] = {}
        
        for greek in valid_greeks:
            if len(bucket_df) > 0:
                if agg_func == 'mean':
                    result[bucket_name][greek] = bucket_df[greek].mean()
                elif agg_func == 'sum':
                    result[bucket_name][greek] = bucket_df[greek].sum()
                elif agg_func == 'std':
                    result[bucket_name][greek] = bucket_df[greek].std()
            else:
                result[bucket_name][greek] = 0.0
    
    return result


def extract_atm_greeks(df: pd.DataFrame) -> Dict[str, float]:
    atm_df = get_atm_options(df, target_delta=0.5, top_n=5)
    
    result = {}
    for greek in ATM_GREEKS:
        if greek in atm_df.columns and len(atm_df) > 0:
            result[f"atm_{greek}"] = atm_df[greek].mean()
        else:
            result[f"atm_{greek}"] = 0.0
    
    return result


def calculate_skew_metrics(df: pd.DataFrame) -> Dict[str, float]:
    calls = df[df['right'] == 'C'] if 'right' in df.columns else df[df['delta'] > 0]
    puts = df[df['right'] == 'P'] if 'right' in df.columns else df[df['delta'] < 0]
    
    call_iv = calls['implied_vol'].mean() if len(calls) > 0 and 'implied_vol' in calls.columns else 0.0
    put_iv = puts['implied_vol'].mean() if len(puts) > 0 and 'implied_vol' in puts.columns else 0.0
    
    call_put_iv_diff = call_iv - put_iv
    
    if 'implied_vol' in df.columns:
        iv_series = df.groupby('strike')['implied_vol'].mean()
        if len(iv_series) >= 3:
            vol_term_slope = np.polyfit(range(len(iv_series)), iv_series.values, 1)[0]
        else:
            vol_term_slope = 0.0
    else:
        vol_term_slope = 0.0
    
    otm_puts = puts[puts['delta'].abs() < 0.3] if 'delta' in puts.columns else puts
    put_skew_intensity = otm_puts['implied_vol'].mean() - put_iv if len(otm_puts) > 0 and 'implied_vol' in otm_puts.columns else 0.0
    
    return {
        "call_put_iv_diff": call_put_iv_diff,
        "vol_term_slope": vol_term_slope,
        "put_skew_intensity": put_skew_intensity,
    }


def extract_greek_features(df: pd.DataFrame) -> np.ndarray:
    features = np.zeros(75, dtype=np.float32)
    
    excluded = set(get_excluded_columns())
    df = df.drop(columns=[c for c in excluded if c in df.columns], errors='ignore')
    
    idx = 0
    
    bucket_greeks = aggregate_greeks_by_bucket(df, GREEKS_FOR_BUCKETING, 'mean')
    for bucket_name, _, _ in DELTA_BUCKETS:
        for greek in GREEKS_FOR_BUCKETING:
            val = bucket_greeks.get(bucket_name, {}).get(greek, 0.0)
            features[idx] = val if not np.isnan(val) else 0.0
            idx += 1
    
    atm_greeks = extract_atm_greeks(df)
    for greek in ATM_GREEKS:
        val = atm_greeks.get(f"atm_{greek}", 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    skew = calculate_skew_metrics(df)
    for metric in ["call_put_iv_diff", "vol_term_slope", "put_skew_intensity"]:
        val = skew.get(metric, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    assert idx == 75, f"Expected 75 features, got {idx}"
    
    return features


def extract_greek_features_batch(
    dfs: List[pd.DataFrame],
    timestamps: Optional[List[str]] = None
) -> np.ndarray:
    n_samples = len(dfs)
    features = np.zeros((n_samples, 75), dtype=np.float32)
    
    for i, df in enumerate(dfs):
        features[i] = extract_greek_features(df)
    
    return features


class GreekFeatureExtractor:
    def __init__(self):
        self.feature_group = FEATURE_GROUPS[FeatureGroup.GREEK_BY_STRIKE]
        self.n_features = self.feature_group.num_features
        self.excluded_columns = set(get_excluded_columns())
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_greek_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        return extract_greek_features_batch(dfs)
    
    def get_feature_names(self) -> List[str]:
        names = []
        for bucket_name, _, _ in DELTA_BUCKETS:
            for greek in GREEKS_FOR_BUCKETING:
                names.append(f"{bucket_name}_{greek}")
        for greek in ATM_GREEKS:
            names.append(f"atm_{greek}")
        names.extend(["call_put_iv_diff", "vol_term_slope", "put_skew_intensity"])
        return names
    
    @property
    def feature_indices(self) -> range:
        return self.feature_group.indices
