"""
Hybrid51 Put/Call Walls & Positioning Features (Task 7)
Extracts 20 features for walls and dealer positioning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .feature_config_v2 import FeatureGroup, FEATURE_GROUPS


def _resolve_oi_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("oi", "open_interest", "size"):
        if col in df.columns:
            return col
    return None


def calculate_max_gamma_strikes(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "call_max_gamma_strike": 0.0,
        "put_max_gamma_strike": 0.0,
    }
    
    if 'gamma' not in df.columns or 'strike' not in df.columns:
        return features
    
    if 'right' in df.columns:
        calls = df[df['right'] == 'C']
        puts = df[df['right'] == 'P']
    else:
        calls = df[df['delta'] > 0]
        puts = df[df['delta'] <= 0]
    
    if len(calls) > 0:
        call_gamma_by_strike = calls.groupby('strike')['gamma'].sum()
        if len(call_gamma_by_strike) > 0:
            features["call_max_gamma_strike"] = call_gamma_by_strike.idxmax()
    
    if len(puts) > 0:
        put_gamma_by_strike = puts.groupby('strike')['gamma'].sum()
        if len(put_gamma_by_strike) > 0:
            features["put_max_gamma_strike"] = put_gamma_by_strike.idxmax()
    
    return features


def calculate_max_oi_strikes(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "call_max_oi_strike": 0.0,
        "put_max_oi_strike": 0.0,
        "call_oi_at_max": 0.0,
        "put_oi_at_max": 0.0,
    }
    
    oi_col = _resolve_oi_col(df)
    if oi_col is None or 'strike' not in df.columns:
        return features
    
    if 'right' in df.columns:
        calls = df[df['right'] == 'C']
        puts = df[df['right'] == 'P']
    else:
        calls = df[df['delta'] > 0]
        puts = df[df['delta'] <= 0]
    
    if len(calls) > 0:
        call_oi_by_strike = calls.groupby('strike')[oi_col].sum()
        if len(call_oi_by_strike) > 0:
            features["call_max_oi_strike"] = call_oi_by_strike.idxmax()
            features["call_oi_at_max"] = call_oi_by_strike.max()
    
    if len(puts) > 0:
        put_oi_by_strike = puts.groupby('strike')[oi_col].sum()
        if len(put_oi_by_strike) > 0:
            features["put_max_oi_strike"] = put_oi_by_strike.idxmax()
            features["put_oi_at_max"] = put_oi_by_strike.max()
    
    return features


def calculate_wall_distances(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "dist_to_call_wall": 0.0,
        "dist_to_put_wall": 0.0,
        "call_wall_strength": 0.0,
        "put_wall_strength": 0.0,
        "combined_wall_strength": 0.0,
        "pinning_prob": 0.0,
        "breakout_prob": 0.0,
        "wall_asymmetry": 0.0,
    }
    
    if 'underlying_price' in df.columns:
        underlying = df['underlying_price'].iloc[0]
    elif 'strike' in df.columns:
        underlying = df['strike'].median()
    else:
        return features
    
    oi_col = _resolve_oi_col(df)
    if oi_col is None or 'strike' not in df.columns:
        return features
    
    if 'right' in df.columns:
        calls = df[df['right'] == 'C']
        puts = df[df['right'] == 'P']
    else:
        calls = df[df['delta'] > 0]
        puts = df[df['delta'] <= 0]
    
    if len(calls) > 0:
        call_oi_by_strike = calls.groupby('strike')[oi_col].sum()
        if len(call_oi_by_strike) > 0:
            call_wall = call_oi_by_strike.idxmax()
            features["dist_to_call_wall"] = (call_wall - underlying) / underlying
            features["call_wall_strength"] = call_oi_by_strike.max() / call_oi_by_strike.sum()
    
    if len(puts) > 0:
        put_oi_by_strike = puts.groupby('strike')[oi_col].sum()
        if len(put_oi_by_strike) > 0:
            put_wall = put_oi_by_strike.idxmax()
            features["dist_to_put_wall"] = (underlying - put_wall) / underlying
            features["put_wall_strength"] = put_oi_by_strike.max() / put_oi_by_strike.sum()
    
    features["combined_wall_strength"] = features["call_wall_strength"] + features["put_wall_strength"]
    
    if features["combined_wall_strength"] > 0:
        near_atm = df[(df['strike'] >= underlying * 0.98) & (df['strike'] <= underlying * 1.02)]
        atm_oi = near_atm[oi_col].sum() if len(near_atm) > 0 else 0
        total_oi = df[oi_col].sum()
        features["pinning_prob"] = atm_oi / total_oi if total_oi > 0 else 0
    
    features["breakout_prob"] = 1 - features["combined_wall_strength"]
    
    if features["put_wall_strength"] > 0:
        features["wall_asymmetry"] = features["call_wall_strength"] / features["put_wall_strength"] - 1
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def calculate_dealer_positioning(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "dealer_net_delta": 0.0,
        "dealer_net_gamma": 0.0,
        "dealer_long_short_ratio": 0.0,
        "dealer_hedging_pressure": 0.0,
        "dealer_gamma_demand": 0.0,
        "dealer_vega_exposure": 0.0,
    }
    
    if 'delta' not in df.columns:
        return features
    
    oi_col = _resolve_oi_col(df)
    
    if oi_col is not None:
        features["dealer_net_delta"] = -(df['delta'] * df[oi_col]).sum()
        
        if 'gamma' in df.columns:
            features["dealer_net_gamma"] = -(df['gamma'] * df[oi_col]).sum()
            features["dealer_gamma_demand"] = abs(features["dealer_net_gamma"])
        
        if 'vega' in df.columns:
            features["dealer_vega_exposure"] = -(df['vega'] * df[oi_col]).sum()
    else:
        features["dealer_net_delta"] = -df['delta'].sum()
        if 'gamma' in df.columns:
            features["dealer_net_gamma"] = -df['gamma'].sum()
            features["dealer_gamma_demand"] = abs(features["dealer_net_gamma"])
        if 'vega' in df.columns:
            features["dealer_vega_exposure"] = -df['vega'].sum()
    
    if 'right' in df.columns:
        calls = df[df['right'] == 'C']
        puts = df[df['right'] == 'P']
        
        if oi_col is not None:
            long_est = calls[oi_col].sum()
            short_est = puts[oi_col].sum()
        else:
            long_est = len(calls)
            short_est = len(puts)
        
        if short_est > 0:
            features["dealer_long_short_ratio"] = long_est / short_est
    
    features["dealer_hedging_pressure"] = abs(features["dealer_net_delta"]) / (df['delta'].abs().sum() + 1e-10)
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def extract_walls_positioning_features(df: pd.DataFrame) -> np.ndarray:
    features = np.zeros(20, dtype=np.float32)
    
    idx = 0
    
    max_gamma = calculate_max_gamma_strikes(df)
    for key in ["call_max_gamma_strike", "put_max_gamma_strike"]:
        features[idx] = max_gamma.get(key, 0.0)
        idx += 1
    
    max_oi = calculate_max_oi_strikes(df)
    for key in ["call_max_oi_strike", "put_max_oi_strike", "call_oi_at_max", "put_oi_at_max"]:
        features[idx] = max_oi.get(key, 0.0)
        idx += 1
    
    walls = calculate_wall_distances(df)
    for key in ["dist_to_call_wall", "dist_to_put_wall", "call_wall_strength", "put_wall_strength",
                "combined_wall_strength", "pinning_prob", "breakout_prob", "wall_asymmetry"]:
        features[idx] = walls.get(key, 0.0)
        idx += 1
    
    dealer = calculate_dealer_positioning(df)
    for key in ["dealer_net_delta", "dealer_net_gamma", "dealer_long_short_ratio",
                "dealer_hedging_pressure", "dealer_gamma_demand", "dealer_vega_exposure"]:
        features[idx] = dealer.get(key, 0.0)
        idx += 1
    
    assert idx == 20, f"Expected 20 walls/positioning features, got {idx}"
    
    return features


class WallsPositioningExtractor:
    def __init__(self):
        self.feature_group = FEATURE_GROUPS[FeatureGroup.WALLS_POSITIONING]
        self.n_features = self.feature_group.num_features
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_walls_positioning_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            "call_max_gamma_strike", "put_max_gamma_strike",
            "call_max_oi_strike", "put_max_oi_strike", "call_oi_at_max", "put_oi_at_max",
            "dist_to_call_wall", "dist_to_put_wall", "call_wall_strength", "put_wall_strength",
            "combined_wall_strength", "pinning_prob", "breakout_prob", "wall_asymmetry",
            "dealer_net_delta", "dealer_net_gamma", "dealer_long_short_ratio",
            "dealer_hedging_pressure", "dealer_gamma_demand", "dealer_vega_exposure",
        ]
