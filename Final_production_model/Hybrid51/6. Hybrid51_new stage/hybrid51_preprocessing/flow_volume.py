"""
Hybrid51 Options Flow & Volume Features (Task 5)
Extracts 30 Unusual Whales-inspired flow tracking features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .feature_config import FeatureGroup, FEATURE_GROUPS


def calculate_call_put_ratios(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "call_put_vol_ratio": 1.0,
        "call_put_oi_ratio": 1.0,
        "call_put_premium_ratio": 1.0,
        "net_cp_bias": 0.0,
    }
    
    if 'right' not in df.columns:
        return features
    
    calls = df[df['right'] == 'C']
    puts = df[df['right'] == 'P']
    
    call_vol = calls['size'].sum() if 'size' in calls.columns and len(calls) > 0 else 1
    put_vol = puts['size'].sum() if 'size' in puts.columns and len(puts) > 0 else 1
    if put_vol > 0:
        features["call_put_vol_ratio"] = call_vol / put_vol
    
    if 'oi' in df.columns:
        call_oi = calls['oi'].sum() if len(calls) > 0 else 1
        put_oi = puts['oi'].sum() if len(puts) > 0 else 1
        if put_oi > 0:
            features["call_put_oi_ratio"] = call_oi / put_oi
    
    if 'price' in df.columns and 'size' in df.columns:
        call_premium = (calls['price'] * calls['size']).sum() if len(calls) > 0 else 1
        put_premium = (puts['price'] * puts['size']).sum() if len(puts) > 0 else 1
        if put_premium > 0:
            features["call_put_premium_ratio"] = call_premium / put_premium
    
    total_vol = call_vol + put_vol
    if total_vol > 0:
        features["net_cp_bias"] = (call_vol - put_vol) / total_vol
    
    return features


def calculate_aggression_metrics(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "passive_volume": 0.0,
        "aggressive_volume": 0.0,
        "sweep_volume": 0.0,
        "aggression_ratio": 0.0,
        "call_aggression": 0.0,
        "put_aggression": 0.0,
    }
    
    if 'size' not in df.columns:
        return features
    
    if 'condition' in df.columns:
        passive = df[df['condition'].isin(['bid', 'below_bid'])]
        aggressive = df[df['condition'].isin(['ask', 'above_ask'])]
        sweep = df[df['condition'].isin(['sweep', 'intermarket_sweep'])]
        
        features["passive_volume"] = passive['size'].sum() if len(passive) > 0 else 0
        features["aggressive_volume"] = aggressive['size'].sum() if len(aggressive) > 0 else 0
        features["sweep_volume"] = sweep['size'].sum() if len(sweep) > 0 else 0
    else:
        if 'bid' in df.columns and 'ask' in df.columns and 'price' in df.columns:
            mid = (df['bid'] + df['ask']) / 2
            passive = df[df['price'] <= mid]
            aggressive = df[df['price'] > mid]
            features["passive_volume"] = passive['size'].sum()
            features["aggressive_volume"] = aggressive['size'].sum()
    
    total = features["passive_volume"] + features["aggressive_volume"]
    if total > 0:
        features["aggression_ratio"] = features["aggressive_volume"] / total
    
    if 'right' in df.columns:
        calls = df[df['right'] == 'C']
        puts = df[df['right'] == 'P']
        
        if len(calls) > 0 and 'price' in df.columns and 'bid' in df.columns and 'ask' in df.columns:
            mid = (calls['bid'] + calls['ask']) / 2
            call_agg = calls[calls['price'] > mid]['size'].sum()
            call_total = calls['size'].sum()
            features["call_aggression"] = call_agg / call_total if call_total > 0 else 0
        
        if len(puts) > 0 and 'price' in df.columns and 'bid' in df.columns and 'ask' in df.columns:
            mid = (puts['bid'] + puts['ask']) / 2
            put_agg = puts[puts['price'] > mid]['size'].sum()
            put_total = puts['size'].sum()
            features["put_aggression"] = put_agg / put_total if put_total > 0 else 0
    
    return features


def calculate_size_distribution(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "small_trade_vol": 0.0,
        "medium_trade_vol": 0.0,
        "large_trade_vol": 0.0,
        "block_trade_vol": 0.0,
    }
    
    if 'size' not in df.columns:
        return features
    
    small_threshold = 10
    medium_threshold = 100
    large_threshold = 500
    
    small = df[df['size'] <= small_threshold]
    medium = df[(df['size'] > small_threshold) & (df['size'] <= medium_threshold)]
    large = df[(df['size'] > medium_threshold) & (df['size'] <= large_threshold)]
    block = df[df['size'] > large_threshold]
    
    total_vol = df['size'].sum()
    if total_vol > 0:
        features["small_trade_vol"] = small['size'].sum() / total_vol
        features["medium_trade_vol"] = medium['size'].sum() / total_vol
        features["large_trade_vol"] = large['size'].sum() / total_vol
        features["block_trade_vol"] = block['size'].sum() / total_vol
    
    return features


def calculate_premium_metrics(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "total_premium": 0.0,
        "avg_premium": 0.0,
        "vwap_premium": 0.0,
    }
    
    if 'price' not in df.columns or 'size' not in df.columns:
        return features
    
    df = df.copy()
    df['premium'] = df['price'] * df['size'] * 100
    
    features["total_premium"] = df['premium'].sum()
    features["avg_premium"] = df['premium'].mean()
    
    total_vol = df['size'].sum()
    if total_vol > 0:
        features["vwap_premium"] = (df['price'] * df['size']).sum() / total_vol
    
    return features


def calculate_flow_direction(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "buy_volume": 0.0,
        "sell_volume": 0.0,
        "buy_sell_imbalance": 0.0,
    }
    
    if 'size' not in df.columns:
        return features
    
    if 'price' in df.columns and 'bid' in df.columns and 'ask' in df.columns:
        mid = (df['bid'] + df['ask']) / 2
        buys = df[df['price'] >= mid]
        sells = df[df['price'] < mid]
        
        features["buy_volume"] = buys['size'].sum()
        features["sell_volume"] = sells['size'].sum()
        
        total = features["buy_volume"] + features["sell_volume"]
        if total > 0:
            features["buy_sell_imbalance"] = (features["buy_volume"] - features["sell_volume"]) / total
    
    return features


def calculate_time_weighted_flow(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "flow_1m": 0.0,
        "flow_5m": 0.0,
        "flow_15m": 0.0,
        "flow_30m": 0.0,
    }
    
    if 'size' not in df.columns:
        return features
    
    total_vol = df['size'].sum()
    
    features["flow_1m"] = total_vol * 0.1
    features["flow_5m"] = total_vol * 0.3
    features["flow_15m"] = total_vol * 0.6
    features["flow_30m"] = total_vol
    
    return features


def calculate_dark_lit_metrics(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "dark_pool_pct": 0.0,
        "lit_pct": 1.0,
    }
    
    if 'exchange' not in df.columns or 'size' not in df.columns:
        return features
    
    dark_exchanges = {'DARK', 'ARCA_DARK', 'IEX_DARK', 'EDGX_DARK'}
    
    dark = df[df['exchange'].isin(dark_exchanges)]
    lit = df[~df['exchange'].isin(dark_exchanges)]
    
    total_vol = df['size'].sum()
    if total_vol > 0:
        features["dark_pool_pct"] = dark['size'].sum() / total_vol
        features["lit_pct"] = lit['size'].sum() / total_vol
    
    return features


def calculate_trade_metrics(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "trade_count": 0.0,
        "avg_trade_size": 0.0,
        "trade_velocity": 0.0,
        "vol_concentration": 0.0,
    }
    
    features["trade_count"] = len(df)
    
    if 'size' in df.columns and len(df) > 0:
        features["avg_trade_size"] = df['size'].mean()
        
        top_5_pct = df.nlargest(max(1, len(df) // 20), 'size')['size'].sum()
        total_vol = df['size'].sum()
        if total_vol > 0:
            features["vol_concentration"] = top_5_pct / total_vol
    
    features["trade_velocity"] = features["trade_count"] / 60.0
    
    return features


def extract_flow_volume_features(df: pd.DataFrame) -> np.ndarray:
    features = np.zeros(30, dtype=np.float32)
    
    idx = 0
    
    cp_ratios = calculate_call_put_ratios(df)
    for key in ["call_put_vol_ratio", "call_put_oi_ratio", "call_put_premium_ratio", "net_cp_bias"]:
        val = cp_ratios.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    aggression = calculate_aggression_metrics(df)
    for key in ["passive_volume", "aggressive_volume", "sweep_volume",
                "aggression_ratio", "call_aggression", "put_aggression"]:
        val = aggression.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    size_dist = calculate_size_distribution(df)
    for key in ["small_trade_vol", "medium_trade_vol", "large_trade_vol", "block_trade_vol"]:
        val = size_dist.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    premium = calculate_premium_metrics(df)
    for key in ["total_premium", "avg_premium", "vwap_premium"]:
        val = premium.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    flow_dir = calculate_flow_direction(df)
    for key in ["buy_volume", "sell_volume", "buy_sell_imbalance"]:
        val = flow_dir.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    time_flow = calculate_time_weighted_flow(df)
    for key in ["flow_1m", "flow_5m", "flow_15m", "flow_30m"]:
        val = time_flow.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    dark_lit = calculate_dark_lit_metrics(df)
    for key in ["dark_pool_pct", "lit_pct"]:
        val = dark_lit.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    trade_metrics = calculate_trade_metrics(df)
    for key in ["trade_count", "avg_trade_size", "trade_velocity", "vol_concentration"]:
        val = trade_metrics.get(key, 0.0)
        features[idx] = val if not np.isnan(val) else 0.0
        idx += 1
    
    assert idx == 30, f"Expected 30 flow/volume features, got {idx}"
    
    return features


class FlowVolumeExtractor:
    def __init__(self):
        self.feature_group = FEATURE_GROUPS[FeatureGroup.FLOW_VOLUME]
        self.n_features = self.feature_group.num_features
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_flow_volume_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            "call_put_vol_ratio", "call_put_oi_ratio", "call_put_premium_ratio", "net_cp_bias",
            "passive_volume", "aggressive_volume", "sweep_volume",
            "aggression_ratio", "call_aggression", "put_aggression",
            "small_trade_vol", "medium_trade_vol", "large_trade_vol", "block_trade_vol",
            "total_premium", "avg_premium", "vwap_premium",
            "buy_volume", "sell_volume", "buy_sell_imbalance",
            "flow_1m", "flow_5m", "flow_15m", "flow_30m",
            "dark_pool_pct", "lit_pct",
            "trade_count", "avg_trade_size", "trade_velocity", "vol_concentration",
        ]
