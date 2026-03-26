"""
Hybrid51 IV Surface & Skew Features (Task 4)
Extracts 25 implied volatility surface features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .feature_config_v2 import MONEYNESS_LEVELS, DTE_BUCKETS, FeatureGroup, FEATURE_GROUPS


def calculate_moneyness(df: pd.DataFrame) -> pd.Series:
    underlying = df['underlying_price'].iloc[0] if 'underlying_price' in df.columns else df['strike'].median()
    return (df['strike'] - underlying) / underlying


def extract_iv_by_moneyness(df: pd.DataFrame) -> Dict[str, float]:
    features = {}
    
    df = df.copy()
    df['moneyness'] = calculate_moneyness(df)
    
    for level in MONEYNESS_LEVELS:
        level_name = f"iv_moneyness_{level:.0%}".replace("-", "neg").replace("%", "pct")
        
        nearby = df[(df['moneyness'] >= level - 0.05) & (df['moneyness'] < level + 0.05)]
        
        if len(nearby) > 0 and 'implied_vol' in nearby.columns:
            features[level_name] = nearby['implied_vol'].mean()
        else:
            features[level_name] = 0.0
    
    return features


def calculate_dte(df: pd.DataFrame) -> pd.Series:
    if 'expiration' not in df.columns:
        return pd.Series([30] * len(df), index=df.index)
    
    if 'trade_date' in df.columns:
        try:
            exp = pd.to_datetime(df['expiration'].astype(str))
            trade = pd.to_datetime(df['trade_date'])
            return (exp - trade).dt.days
        except:
            pass
    
    return pd.Series([30] * len(df), index=df.index)


def extract_iv_term_structure(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "iv_1w": 0.0,
        "iv_1m": 0.0,
        "iv_2m": 0.0,
        "iv_3m": 0.0,
        "iv_6m": 0.0,
    }
    
    df = df.copy()
    df['dte'] = calculate_dte(df)
    
    if 'implied_vol' not in df.columns:
        return features
    
    dte_ranges = [(0, 7, "iv_1w"), (8, 30, "iv_1m"), (31, 60, "iv_2m"),
                  (61, 90, "iv_3m"), (91, 180, "iv_6m")]
    
    for low, high, name in dte_ranges:
        bucket_df = df[(df['dte'] >= low) & (df['dte'] <= high)]
        if len(bucket_df) > 0:
            features[name] = bucket_df['implied_vol'].mean()
    
    return features


def calculate_vol_skew_metrics(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "put_skew": 0.0,
        "call_skew": 0.0,
        "term_skew": 0.0,
        "smile_curvature": 0.0,
        "skew_asymmetry": 0.0,
    }
    
    if 'implied_vol' not in df.columns:
        return features
    
    df = df.copy()
    df['moneyness'] = calculate_moneyness(df)
    
    if 'right' in df.columns:
        calls = df[df['right'] == 'C']
        puts = df[df['right'] == 'P']
    else:
        calls = df[df['delta'] > 0]
        puts = df[df['delta'] <= 0]
    
    atm_iv = df[df['moneyness'].abs() < 0.05]['implied_vol'].mean()
    atm_iv = atm_iv if not np.isnan(atm_iv) else df['implied_vol'].mean()
    
    if len(puts) > 0:
        otm_puts = puts[puts['moneyness'] < -0.1]
        if len(otm_puts) > 0:
            features["put_skew"] = otm_puts['implied_vol'].mean() - atm_iv
    
    if len(calls) > 0:
        otm_calls = calls[calls['moneyness'] > 0.1]
        if len(otm_calls) > 0:
            features["call_skew"] = otm_calls['implied_vol'].mean() - atm_iv
    
    df['dte'] = calculate_dte(df)
    near_iv = df[df['dte'] < 30]['implied_vol'].mean()
    far_iv = df[df['dte'] >= 30]['implied_vol'].mean()
    if not np.isnan(near_iv) and not np.isnan(far_iv):
        features["term_skew"] = far_iv - near_iv
    
    if 'moneyness' in df.columns:
        moneyness_clean = df['moneyness'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(moneyness_clean) > 0:
            iv_by_money = df[df['moneyness'].isin(moneyness_clean)].groupby(pd.cut(moneyness_clean, bins=5))['implied_vol'].mean()
            if len(iv_by_money) >= 3:
                features["smile_curvature"] = iv_by_money.iloc[0] + iv_by_money.iloc[-1] - 2 * iv_by_money.iloc[len(iv_by_money)//2]
    
    features["skew_asymmetry"] = features["put_skew"] - features["call_skew"]
    
    return features


def calculate_iv_percentiles(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "iv_p25": 0.0,
        "iv_p50": 0.0,
        "iv_p75": 0.0,
    }
    
    if 'implied_vol' not in df.columns:
        return features
    
    iv = df['implied_vol'].dropna()
    if len(iv) > 0:
        features["iv_p25"] = iv.quantile(0.25)
        features["iv_p50"] = iv.quantile(0.50)
        features["iv_p75"] = iv.quantile(0.75)
    
    return features


def calculate_put_call_iv_diff(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "pc_iv_diff_1w": 0.0,
        "pc_iv_diff_1m": 0.0,
        "pc_iv_diff_2m": 0.0,
        "pc_iv_diff_3m": 0.0,
        "pc_iv_diff_6m": 0.0,
    }
    
    if 'implied_vol' not in df.columns or 'right' not in df.columns:
        return features
    
    df = df.copy()
    df['dte'] = calculate_dte(df)
    
    calls = df[df['right'] == 'C']
    puts = df[df['right'] == 'P']
    
    dte_ranges = [(0, 7, "pc_iv_diff_1w"), (8, 30, "pc_iv_diff_1m"), 
                  (31, 60, "pc_iv_diff_2m"), (61, 90, "pc_iv_diff_3m"), 
                  (91, 180, "pc_iv_diff_6m")]
    
    for low, high, name in dte_ranges:
        c_iv = calls[(calls['dte'] >= low) & (calls['dte'] <= high)]['implied_vol'].mean()
        p_iv = puts[(puts['dte'] >= low) & (puts['dte'] <= high)]['implied_vol'].mean()
        
        if not np.isnan(c_iv) and not np.isnan(p_iv):
            features[name] = p_iv - c_iv
    
    return features


def extract_iv_surface_features(df: pd.DataFrame) -> np.ndarray:
    features = np.zeros(25, dtype=np.float32)
    
    idx = 0
    
    iv_money = extract_iv_by_moneyness(df)
    for level in MONEYNESS_LEVELS:
        level_name = f"iv_moneyness_{level:.0%}".replace("-", "neg").replace("%", "pct")
        val = iv_money.get(level_name, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    iv_term = extract_iv_term_structure(df)
    for key in ["iv_1w", "iv_1m", "iv_2m", "iv_3m", "iv_6m"]:
        val = iv_term.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    skew = calculate_vol_skew_metrics(df)
    for key in ["put_skew", "call_skew", "term_skew", "smile_curvature", "skew_asymmetry"]:
        val = skew.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    percentiles = calculate_iv_percentiles(df)
    for key in ["iv_p25", "iv_p50", "iv_p75"]:
        val = percentiles.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    pc_diff = calculate_put_call_iv_diff(df)
    for key in ["pc_iv_diff_1w", "pc_iv_diff_1m", "pc_iv_diff_2m", "pc_iv_diff_3m", "pc_iv_diff_6m"]:
        val = pc_diff.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    assert idx == 25, f"Expected 25 IV surface features, got {idx}"
    
    return features


class IVSurfaceExtractor:
    def __init__(self):
        self.feature_group = FEATURE_GROUPS[FeatureGroup.IV_SURFACE]
        self.n_features = self.feature_group.num_features
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_iv_surface_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        names = []
        for level in MONEYNESS_LEVELS:
            names.append(f"iv_moneyness_{level:.0%}".replace("-", "neg").replace("%", "pct"))
        names.extend(["iv_1w", "iv_1m", "iv_2m", "iv_3m", "iv_6m"])
        names.extend(["put_skew", "call_skew", "term_skew", "smile_curvature", "skew_asymmetry"])
        names.extend(["iv_p25", "iv_p50", "iv_p75"])
        names.extend(["pc_iv_diff_1w", "pc_iv_diff_1m", "pc_iv_diff_2m", 
                      "pc_iv_diff_3m", "pc_iv_diff_6m"])
        return names
