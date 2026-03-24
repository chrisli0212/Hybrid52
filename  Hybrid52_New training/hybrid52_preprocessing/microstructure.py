"""
Hybrid51 Microstructure Features (Task 6)
Extracts 20 market microstructure features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .feature_config import FeatureGroup, FEATURE_GROUPS


def calculate_bid_ask_spread(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "spread_mean": 0.0,
        "spread_pct": 0.0,
        "spread_rolling": 0.0,
        "spread_std": 0.0,
    }
    
    if 'bid' not in df.columns or 'ask' not in df.columns:
        return features
    
    df = df.copy()
    df['spread'] = df['ask'] - df['bid']
    df['mid'] = (df['ask'] + df['bid']) / 2
    df['spread_pct_val'] = df['spread'] / df['mid'].replace(0, np.nan)
    
    features["spread_mean"] = df['spread'].mean()
    features["spread_pct"] = df['spread_pct_val'].mean()
    features["spread_rolling"] = df['spread'].rolling(10, min_periods=1).mean().iloc[-1] if len(df) > 0 else 0.0
    features["spread_std"] = df['spread'].std()
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def calculate_order_book_imbalance(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "bid_ask_imbalance": 0.0,
        "tob_imbalance": 0.0,
        "depth_imbalance": 0.0,
        "imbalance_vol": 0.0,
        "sustained_imbalance": 0.0,
    }
    
    if 'bid_size' not in df.columns or 'ask_size' not in df.columns:
        return features
    
    bid_vol = df['bid_size'].sum()
    ask_vol = df['ask_size'].sum()
    total = bid_vol + ask_vol
    
    if total > 0:
        features["bid_ask_imbalance"] = (bid_vol - ask_vol) / total
    
    if len(df) > 0:
        features["tob_imbalance"] = (df['bid_size'].iloc[-1] - df['ask_size'].iloc[-1]) / \
                                    (df['bid_size'].iloc[-1] + df['ask_size'].iloc[-1] + 1e-10)
    
    if len(df) >= 5:
        top_n = 5
        bid_depth = df.nlargest(top_n, 'bid_size')['bid_size'].sum()
        ask_depth = df.nlargest(top_n, 'ask_size')['ask_size'].sum()
        features["depth_imbalance"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
    
    features["imbalance_vol"] = features["bid_ask_imbalance"] * total if total > 0 else 0.0
    
    if len(df) >= 10:
        rolling_imb = df.apply(
            lambda row: (row['bid_size'] - row['ask_size']) / (row['bid_size'] + row['ask_size'] + 1e-10),
            axis=1
        ).rolling(10).mean()
        features["sustained_imbalance"] = rolling_imb.iloc[-1] if len(rolling_imb) > 0 else 0.0
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def calculate_quote_intensity(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "quote_frequency": 0.0,
        "cancel_rate": 0.0,
        "improvement_rate": 0.0,
    }
    
    features["quote_frequency"] = len(df) / 60.0
    
    if 'bid' in df.columns and len(df) > 1:
        bid_changes = (df['bid'].diff() != 0).sum()
        features["cancel_rate"] = bid_changes / len(df) if len(df) > 0 else 0.0
        
        bid_improvements = ((df['bid'].diff() > 0) | (df['ask'].diff() < 0)).sum()
        features["improvement_rate"] = bid_improvements / len(df) if len(df) > 0 else 0.0
    
    return features


def calculate_trade_velocity(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "trades_per_min": 0.0,
        "volume_per_min": 0.0,
    }
    
    features["trades_per_min"] = len(df)
    
    if 'size' in df.columns:
        features["volume_per_min"] = df['size'].sum()
    
    return features


def calculate_effective_spreads(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "effective_spread": 0.0,
        "realized_spread": 0.0,
    }
    
    if 'price' not in df.columns or 'bid' not in df.columns or 'ask' not in df.columns:
        return features
    
    df = df.copy()
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['effective'] = 2 * abs(df['price'] - df['mid'])
    features["effective_spread"] = df['effective'].mean()
    
    if len(df) > 5:
        df['future_mid'] = df['mid'].shift(-5)
        df['realized'] = 2 * (df['price'] - df['mid']) * np.sign(df['price'] - df['mid']) - \
                         (df['future_mid'] - df['mid'])
        features["realized_spread"] = df['realized'].mean()
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def calculate_price_impact(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "temp_impact": 0.0,
        "short_impact": 0.0,
        "size_impact_corr": 0.0,
        "impact_asymmetry": 0.0,
    }
    
    if 'price' not in df.columns or 'size' not in df.columns or len(df) < 10:
        return features
    
    df = df.copy()
    df['ret'] = df['price'].pct_change()
    
    df['future_ret_1'] = df['price'].shift(-1).pct_change()
    features["temp_impact"] = df['ret'].mean()
    
    df['future_ret_5'] = df['price'].shift(-5) / df['price'] - 1
    features["short_impact"] = df['future_ret_5'].mean() if len(df) > 5 else 0.0
    
    if df['size'].std() > 0 and df['ret'].std() > 0:
        features["size_impact_corr"] = df['size'].corr(df['ret'].abs())
    
    buys = df[df['ret'] > 0]
    sells = df[df['ret'] < 0]
    buy_impact = buys['ret'].mean() if len(buys) > 0 else 0.0
    sell_impact = abs(sells['ret'].mean()) if len(sells) > 0 else 0.0
    
    if sell_impact > 0:
        features["impact_asymmetry"] = buy_impact / sell_impact - 1
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def extract_microstructure_features(df: pd.DataFrame) -> np.ndarray:
    features = np.zeros(20, dtype=np.float32)
    
    idx = 0
    
    spread = calculate_bid_ask_spread(df)
    for key in ["spread_mean", "spread_pct", "spread_rolling", "spread_std"]:
        features[idx] = spread.get(key, 0.0)
        idx += 1
    
    imbalance = calculate_order_book_imbalance(df)
    for key in ["bid_ask_imbalance", "tob_imbalance", "depth_imbalance",
                "imbalance_vol", "sustained_imbalance"]:
        features[idx] = imbalance.get(key, 0.0)
        idx += 1
    
    intensity = calculate_quote_intensity(df)
    for key in ["quote_frequency", "cancel_rate", "improvement_rate"]:
        features[idx] = intensity.get(key, 0.0)
        idx += 1
    
    velocity = calculate_trade_velocity(df)
    for key in ["trades_per_min", "volume_per_min"]:
        features[idx] = velocity.get(key, 0.0)
        idx += 1
    
    effective = calculate_effective_spreads(df)
    for key in ["effective_spread", "realized_spread"]:
        features[idx] = effective.get(key, 0.0)
        idx += 1
    
    impact = calculate_price_impact(df)
    for key in ["temp_impact", "short_impact", "size_impact_corr", "impact_asymmetry"]:
        features[idx] = impact.get(key, 0.0)
        idx += 1
    
    assert idx == 20, f"Expected 20 microstructure features, got {idx}"
    
    return features


class MicrostructureExtractor:
    def __init__(self):
        self.feature_group = FEATURE_GROUPS[FeatureGroup.MICROSTRUCTURE]
        self.n_features = self.feature_group.num_features
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_microstructure_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            "spread_mean", "spread_pct", "spread_rolling", "spread_std",
            "bid_ask_imbalance", "tob_imbalance", "depth_imbalance",
            "imbalance_vol", "sustained_imbalance",
            "quote_frequency", "cancel_rate", "improvement_rate",
            "trades_per_min", "volume_per_min",
            "effective_spread", "realized_spread",
            "temp_impact", "short_impact", "size_impact_corr", "impact_asymmetry",
        ]
