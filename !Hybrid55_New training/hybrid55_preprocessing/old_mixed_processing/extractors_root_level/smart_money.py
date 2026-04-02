"""
Hybrid51 Smart Money Detection Features
Extracts 15 features identifying institutional/informed trading activity.

Based on professional platforms: Unusual Whales, FlowAlgo, Cheddar Flow
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import timedelta


OPRA_TRADE_CONDITIONS = {
    18: 'regular_sale',
    125: 'opening_trade',
    130: 'intermarket_sweep',
    131: 'sweep_extended_hours',
    95: 'single_leg_auction',
    133: 'complex_trade',
    138: 'stock_option_trade',
    120: 'cancel',
    121: 'cancel_last',
}


def detect_sweep_orders(df: pd.DataFrame, window_seconds: float = 1.0) -> Dict[str, float]:
    """
    Detect intermarket sweep orders (ISO) - aggressive institutional execution.
    
    Golden Sweep: Large premium (>$100K typical) + aggressive + multi-exchange
    """
    features = {
        'is_sweep': 0.0,
        'sweep_score': 0.0,
        'sweep_premium_pct': 0.0,
        'multi_exchange_count': 0.0,
    }
    
    if 'condition' not in df.columns:
        return features
    
    sweep_conditions = {130, 131}
    sweeps = df[df['condition'].isin(sweep_conditions)]
    
    features['is_sweep'] = 1.0 if len(sweeps) > 0 else 0.0
    
    if len(sweeps) == 0:
        return features
    
    if 'exchange' in sweeps.columns:
        features['multi_exchange_count'] = sweeps['exchange'].nunique()
    
    if 'size' in sweeps.columns and 'trade_timestamp' in sweeps.columns:
        sweeps = sweeps.sort_values('trade_timestamp')
        sweeps['time_delta'] = sweeps['trade_timestamp'].diff().dt.total_seconds()
        
        rapid_sweeps = sweeps[sweeps['time_delta'] < window_seconds]
        if len(rapid_sweeps) > 0:
            total_size = rapid_sweeps['size'].sum()
            n_exchanges = rapid_sweeps['exchange'].nunique() if 'exchange' in rapid_sweeps.columns else 1
            features['sweep_score'] = n_exchanges * total_size / window_seconds
    
    if 'price' in sweeps.columns and 'bid' in sweeps.columns and 'ask' in sweeps.columns:
        sweeps = sweeps.copy()
        sweeps['mid'] = (sweeps['bid'] + sweeps['ask']) / 2
        premium_pct = ((sweeps['price'] - sweeps['mid']) / sweeps['mid'].replace(0, np.nan)).mean()
        features['sweep_premium_pct'] = premium_pct if not np.isnan(premium_pct) else 0.0
    
    return features


def detect_block_trades(df: pd.DataFrame, zscore_threshold: float = 3.0) -> Dict[str, float]:
    """
    Detect unusually large block trades indicating institutional activity.
    """
    features = {
        'is_block': 0.0,
        'block_premium': 0.0,
        'block_to_avg_ratio': 0.0,
        'block_count': 0.0,
    }
    
    if 'size' not in df.columns or len(df) < 5:
        return features
    
    mean_size = df['size'].mean()
    std_size = df['size'].std()
    
    if std_size < 1e-10:
        return features
    
    df = df.copy()
    df['size_zscore'] = (df['size'] - mean_size) / std_size
    
    blocks = df[df['size_zscore'] > zscore_threshold]
    
    features['block_count'] = len(blocks)
    features['is_block'] = 1.0 if len(blocks) > 0 else 0.0
    
    if len(blocks) > 0:
        features['block_to_avg_ratio'] = blocks['size'].mean() / mean_size
        
        if 'price' in blocks.columns:
            features['block_premium'] = (blocks['price'] * blocks['size'] * 100).sum()
    
    return features


def classify_aggression(df: pd.DataFrame) -> Dict[str, float]:
    """
    Classify trade aggression: buyer-initiated (bullish) vs seller-initiated (bearish).
    """
    features = {
        'near_ask_pct': 0.0,
        'near_bid_pct': 0.0,
        'mid_execution_pct': 0.0,
        'price_improvement_pct': 0.0,
    }
    
    if 'price' not in df.columns or 'bid' not in df.columns or 'ask' not in df.columns:
        return features
    
    df = df.copy()
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['spread'] = df['ask'] - df['bid']
    
    df['distance_to_ask'] = (df['price'] - df['ask']).abs()
    df['distance_to_bid'] = (df['price'] - df['bid']).abs()
    df['distance_to_mid'] = (df['price'] - df['mid']).abs()
    
    threshold = df['spread'] * 0.1
    
    near_ask = df[df['distance_to_ask'] <= threshold]
    near_bid = df[df['distance_to_bid'] <= threshold]
    near_mid = df[df['distance_to_mid'] <= threshold]
    
    total = len(df)
    if total > 0:
        features['near_ask_pct'] = len(near_ask) / total
        features['near_bid_pct'] = len(near_bid) / total
        features['mid_execution_pct'] = len(near_mid) / total
    
    inside_nbbo = df[(df['price'] > df['bid']) & (df['price'] < df['ask'])]
    features['price_improvement_pct'] = len(inside_nbbo) / total if total > 0 else 0.0
    
    return features


def detect_unusual_size(df: pd.DataFrame) -> Dict[str, float]:
    """
    Statistical detection of unusual trade sizes.
    """
    features = {
        'size_zscore': 0.0,
        'size_percentile': 0.5,
        'large_trade_cluster': 0.0,
    }
    
    if 'size' not in df.columns or len(df) < 5:
        return features
    
    mean_size = df['size'].mean()
    std_size = df['size'].std()
    
    if len(df) > 0:
        current_size = df['size'].iloc[-1] if len(df) > 0 else mean_size
        
        if std_size > 1e-10:
            features['size_zscore'] = (current_size - mean_size) / std_size
        
        features['size_percentile'] = (df['size'] < current_size).sum() / len(df)
    
    if len(df) >= 10 and 'trade_timestamp' in df.columns:
        df = df.sort_values('trade_timestamp')
        df['is_large'] = df['size'] > (mean_size + std_size)
        
        df['time_delta'] = df['trade_timestamp'].diff().dt.total_seconds()
        
        large_trades = df[df['is_large']]
        if len(large_trades) >= 2:
            large_trades = large_trades.copy()
            large_trades['time_delta'] = large_trades['trade_timestamp'].diff().dt.total_seconds()
            
            rapid_large = large_trades[large_trades['time_delta'] < 60]
            features['large_trade_cluster'] = len(rapid_large)
    
    return features


def extract_smart_money_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract all 15 smart money detection features.
    """
    features = np.zeros(15, dtype=np.float32)
    
    idx = 0
    
    sweep = detect_sweep_orders(df)
    for key in ['is_sweep', 'sweep_score', 'sweep_premium_pct', 'multi_exchange_count']:
        val = sweep.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    block = detect_block_trades(df)
    for key in ['is_block', 'block_premium', 'block_to_avg_ratio', 'block_count']:
        val = block.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    aggression = classify_aggression(df)
    for key in ['near_ask_pct', 'near_bid_pct', 'mid_execution_pct', 'price_improvement_pct']:
        val = aggression.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    size = detect_unusual_size(df)
    for key in ['size_zscore', 'size_percentile', 'large_trade_cluster']:
        val = size.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    assert idx == 15, f"Expected 15 smart money features, got {idx}"
    
    return features


class SmartMoneyDetector:
    def __init__(self):
        self.n_features = 15
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_smart_money_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            'is_sweep', 'sweep_score', 'sweep_premium_pct', 'multi_exchange_count',
            'is_block', 'block_premium', 'block_to_avg_ratio', 'block_count',
            'near_ask_pct', 'near_bid_pct', 'mid_execution_pct', 'price_improvement_pct',
            'size_zscore', 'size_percentile', 'large_trade_cluster',
        ]
    
    @staticmethod
    def decode_condition(code: int) -> str:
        """Decode OPRA trade condition code."""
        return OPRA_TRADE_CONDITIONS.get(code, f'unknown_{code}')
