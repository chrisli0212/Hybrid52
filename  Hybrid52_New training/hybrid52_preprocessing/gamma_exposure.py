"""
Hybrid51 Gamma/Vanna/Charm Exposure Features (Task 3)
Extracts 50 features: Gamma (30) + Vanna/Charm (20)
SpotGamma-inspired gamma exposure analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .data_validation import get_excluded_columns
from .feature_config_v2 import DELTA_BUCKETS, FeatureGroup, FEATURE_GROUPS


def calculate_gamma_by_strike(
    df: pd.DataFrame,
    underlying_price: float,
    n_strikes_above: int = 10,
    n_strikes_below: int = 10
) -> Dict[str, float]:
    features = {}
    
    strikes = df['strike'].unique()
    strikes = np.sort(strikes)
    
    atm_idx = np.argmin(np.abs(strikes - underlying_price))
    
    above_strikes = strikes[atm_idx:atm_idx + n_strikes_above]
    for i in range(n_strikes_above):
        if i < len(above_strikes):
            strike = above_strikes[i]
            strike_df = df[df['strike'] == strike]
            features[f"gamma_strike_above_{i}"] = strike_df['gamma'].sum() if len(strike_df) > 0 else 0.0
        else:
            features[f"gamma_strike_above_{i}"] = 0.0
    
    below_strikes = strikes[max(0, atm_idx - n_strikes_below):atm_idx][::-1]
    for i in range(n_strikes_below):
        if i < len(below_strikes):
            strike = below_strikes[i]
            strike_df = df[df['strike'] == strike]
            features[f"gamma_strike_below_{i}"] = strike_df['gamma'].sum() if len(strike_df) > 0 else 0.0
        else:
            features[f"gamma_strike_below_{i}"] = 0.0
    
    return features


def calculate_net_gamma(df: pd.DataFrame) -> Dict[str, float]:
    total_gamma = df['gamma'].sum() if 'gamma' in df.columns else 0.0
    
    if 'right' in df.columns:
        calls = df[df['right'] == 'C']
        puts = df[df['right'] == 'P']
    else:
        calls = df[df['delta'] > 0]
        puts = df[df['delta'] <= 0]
    
    call_gamma = calls['gamma'].sum() if len(calls) > 0 else 0.0
    put_gamma = puts['gamma'].sum() if len(puts) > 0 else 0.0
    net_gamma = call_gamma - put_gamma
    
    return {
        "total_gamma": total_gamma,
        "call_gamma": call_gamma,
        "put_gamma": put_gamma,
        "net_gamma": net_gamma,
    }


def estimate_dealer_positioning(df: pd.DataFrame, underlying_price: float) -> Dict[str, float]:
    if 'gamma' not in df.columns:
        return {
            "dealer_gamma_estimate": 0.0,
            "gamma_flip_level": underlying_price,
            "dist_to_gamma_flip": 0.0,
        }
    
    dealer_gamma = -df['gamma'].sum()
    
    strikes = df['strike'].unique()
    strikes = np.sort(strikes)
    
    gamma_flip_level = underlying_price
    for strike in strikes:
        strike_df = df[df['strike'] <= strike]
        net = strike_df['gamma'].sum()
        if net < 0:
            gamma_flip_level = strike
            break
    
    dist_to_gamma_flip = (underlying_price - gamma_flip_level) / underlying_price
    
    return {
        "dealer_gamma_estimate": dealer_gamma,
        "gamma_flip_level": gamma_flip_level,
        "dist_to_gamma_flip": dist_to_gamma_flip,
    }


def calculate_gamma_zones(df: pd.DataFrame, underlying_price: float) -> Dict[str, float]:
    if 'gamma' not in df.columns or 'strike' not in df.columns:
        return {
            "below_gamma_flip": 0.0,
            "above_gamma_flip": 0.0,
            "gamma_zone_strength": 0.0,
        }
    
    below_atm = df[df['strike'] < underlying_price]
    above_atm = df[df['strike'] >= underlying_price]
    
    below_gamma = below_atm['gamma'].sum() if len(below_atm) > 0 else 0.0
    above_gamma = above_atm['gamma'].sum() if len(above_atm) > 0 else 0.0
    
    total = abs(below_gamma) + abs(above_gamma)
    if total > 0:
        gamma_zone_strength = (above_gamma - below_gamma) / total
    else:
        gamma_zone_strength = 0.0
    
    return {
        "below_gamma_flip": below_gamma,
        "above_gamma_flip": above_gamma,
        "gamma_zone_strength": gamma_zone_strength,
    }


def extract_gamma_features(df: pd.DataFrame) -> np.ndarray:
    features = np.zeros(30, dtype=np.float32)
    
    if 'underlying_price' in df.columns:
        underlying_price = df['underlying_price'].iloc[0]
    else:
        underlying_price = df['strike'].median()
    
    idx = 0
    
    gamma_by_strike = calculate_gamma_by_strike(df, underlying_price)
    for i in range(10):
        val = gamma_by_strike.get(f"gamma_strike_above_{i}", 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    for i in range(10):
        val = gamma_by_strike.get(f"gamma_strike_below_{i}", 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    net_gamma = calculate_net_gamma(df)
    for key in ["total_gamma", "call_gamma", "put_gamma", "net_gamma"]:
        val = net_gamma.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    dealer = estimate_dealer_positioning(df, underlying_price)
    for key in ["dealer_gamma_estimate", "gamma_flip_level", "dist_to_gamma_flip"]:
        val = dealer.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    zones = calculate_gamma_zones(df, underlying_price)
    for key in ["below_gamma_flip", "above_gamma_flip", "gamma_zone_strength"]:
        val = zones.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    assert idx == 30, f"Expected 30 gamma features, got {idx}"
    
    return features


def calculate_vanna_by_bucket(df: pd.DataFrame) -> Dict[str, float]:
    features = {}
    
    df = df.copy()
    df['abs_delta'] = df['delta'].abs()
    
    for bucket_name, low, high in DELTA_BUCKETS:
        bucket_df = df[(df['abs_delta'] >= low) & (df['abs_delta'] < high)]
        if len(bucket_df) > 0 and 'vanna' in bucket_df.columns:
            features[f"{bucket_name}_vanna"] = bucket_df['vanna'].mean()
        else:
            features[f"{bucket_name}_vanna"] = 0.0
    
    return features


def calculate_charm_by_bucket(df: pd.DataFrame) -> Dict[str, float]:
    features = {}
    
    df = df.copy()
    df['abs_delta'] = df['delta'].abs()
    
    for bucket_name, low, high in DELTA_BUCKETS:
        bucket_df = df[(df['abs_delta'] >= low) & (df['abs_delta'] < high)]
        if len(bucket_df) > 0 and 'charm' in bucket_df.columns:
            features[f"{bucket_name}_charm"] = bucket_df['charm'].mean()
        else:
            features[f"{bucket_name}_charm"] = 0.0
    
    return features


def calculate_vanna_charm_net(df: pd.DataFrame) -> Dict[str, float]:
    if 'right' in df.columns:
        calls = df[df['right'] == 'C']
        puts = df[df['right'] == 'P']
    else:
        calls = df[df['delta'] > 0]
        puts = df[df['delta'] <= 0]
    
    total_vanna = df['vanna'].sum() if 'vanna' in df.columns else 0.0
    call_vanna = calls['vanna'].sum() if len(calls) > 0 and 'vanna' in calls.columns else 0.0
    put_vanna = puts['vanna'].sum() if len(puts) > 0 and 'vanna' in puts.columns else 0.0
    
    total_charm = df['charm'].sum() if 'charm' in df.columns else 0.0
    call_charm = calls['charm'].sum() if len(calls) > 0 and 'charm' in calls.columns else 0.0
    put_charm = puts['charm'].sum() if len(puts) > 0 and 'charm' in puts.columns else 0.0
    
    return {
        "total_vanna": total_vanna,
        "call_vanna": call_vanna,
        "put_vanna": put_vanna,
        "total_charm": total_charm,
        "call_charm": call_charm,
        "put_charm": put_charm,
    }


def calculate_cross_greek_ratios(df: pd.DataFrame) -> Dict[str, float]:
    gamma_sum = df['gamma'].sum() if 'gamma' in df.columns else 1.0
    vanna_sum = df['vanna'].sum() if 'vanna' in df.columns else 0.0
    charm_sum = df['charm'].sum() if 'charm' in df.columns else 0.0
    theta_sum = df['theta'].sum() if 'theta' in df.columns else 1.0
    vega_sum = df['vega'].sum() if 'vega' in df.columns else 1.0
    
    vanna_gamma_ratio = vanna_sum / gamma_sum if abs(gamma_sum) > 1e-10 else 0.0
    charm_theta_ratio = charm_sum / theta_sum if abs(theta_sum) > 1e-10 else 0.0
    vanna_vega_ratio = vanna_sum / vega_sum if abs(vega_sum) > 1e-10 else 0.0
    
    net_vanna_strength = abs(vanna_sum) / (abs(vanna_sum) + abs(charm_sum) + 1e-10)
    
    return {
        "vanna_gamma_ratio": vanna_gamma_ratio,
        "charm_theta_ratio": charm_theta_ratio,
        "vanna_vega_ratio": vanna_vega_ratio,
        "net_vanna_strength": net_vanna_strength,
    }


def extract_vanna_charm_features(df: pd.DataFrame) -> np.ndarray:
    features = np.zeros(20, dtype=np.float32)
    
    idx = 0
    
    vanna_bucket = calculate_vanna_by_bucket(df)
    for bucket_name, _, _ in DELTA_BUCKETS:
        val = vanna_bucket.get(f"{bucket_name}_vanna", 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    charm_bucket = calculate_charm_by_bucket(df)
    for bucket_name, _, _ in DELTA_BUCKETS:
        val = charm_bucket.get(f"{bucket_name}_charm", 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    net = calculate_vanna_charm_net(df)
    for key in ["total_vanna", "call_vanna", "put_vanna", "total_charm", "call_charm", "put_charm"]:
        val = net.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    ratios = calculate_cross_greek_ratios(df)
    for key in ["vanna_gamma_ratio", "charm_theta_ratio", "vanna_vega_ratio", "net_vanna_strength"]:
        val = ratios.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    assert idx == 20, f"Expected 20 vanna/charm features, got {idx}"
    
    return features


def extract_gamma_exposure_all(df: pd.DataFrame) -> np.ndarray:
    gamma_features = extract_gamma_features(df)
    vanna_charm_features = extract_vanna_charm_features(df)
    
    return np.concatenate([gamma_features, vanna_charm_features])


class GammaExposureExtractor:
    def __init__(self):
        self.gamma_group = FEATURE_GROUPS[FeatureGroup.GAMMA_EXPOSURE]
        self.vanna_charm_group = FEATURE_GROUPS[FeatureGroup.VANNA_CHARM]
        self.n_features = self.gamma_group.num_features + self.vanna_charm_group.num_features
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_gamma_exposure_all(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        names = []
        for i in range(10):
            names.append(f"gamma_strike_above_{i}")
        for i in range(10):
            names.append(f"gamma_strike_below_{i}")
        names.extend(["total_gamma", "call_gamma", "put_gamma", "net_gamma"])
        names.extend(["dealer_gamma_estimate", "gamma_flip_level", "dist_to_gamma_flip"])
        names.extend(["below_gamma_flip", "above_gamma_flip", "gamma_zone_strength"])
        for bucket_name, _, _ in DELTA_BUCKETS:
            names.append(f"{bucket_name}_vanna")
        for bucket_name, _, _ in DELTA_BUCKETS:
            names.append(f"{bucket_name}_charm")
        names.extend(["total_vanna", "call_vanna", "put_vanna",
                      "total_charm", "call_charm", "put_charm"])
        names.extend(["vanna_gamma_ratio", "charm_theta_ratio",
                      "vanna_vega_ratio", "net_vanna_strength"])
        return names
