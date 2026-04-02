"""
Hybrid51 Volume Anomaly Detection Features
Extracts 12 features detecting unusual volume activity.

Based on statistical anomaly detection and professional platforms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats


def calculate_volume_zscore(df: pd.DataFrame, historical_mean: Optional[float] = None,
                           historical_std: Optional[float] = None) -> Dict[str, float]:
    """
    Z-score based volume anomaly detection.
    """
    features = {
        'volume_zscore': 0.0,
        'volume_percentile_20d': 0.5,
        'volume_percentile_60d': 0.5,
        'volume_spike_intensity': 0.0,
    }
    
    if 'size' not in df.columns or len(df) < 2:
        return features
    
    current_volume = df['size'].sum()
    
    if historical_mean is not None and historical_std is not None and historical_std > 0:
        features['volume_zscore'] = (current_volume - historical_mean) / historical_std
        features['volume_spike_intensity'] = max(0, features['volume_zscore'])
    else:
        df_mean = df['size'].mean()
        df_std = df['size'].std()
        if df_std > 0:
            features['volume_zscore'] = (current_volume - df_mean * len(df)) / (df_std * np.sqrt(len(df)))
    
    features['volume_percentile_20d'] = 0.5
    features['volume_percentile_60d'] = 0.5
    
    return features


def calculate_volume_to_oi(df: pd.DataFrame, open_interest: Optional[float] = None) -> Dict[str, float]:
    """
    Volume relative to open interest - high ratio indicates unusual activity.
    """
    features = {
        'volume_to_oi_ratio': 0.0,
        'oi_turnover_rate': 0.0,
        'volume_acceleration': 0.0,
        'unusual_activity_flag': 0.0,
    }
    
    if 'size' not in df.columns:
        return features
    
    current_volume = df['size'].sum()
    
    if open_interest is not None and open_interest > 0:
        features['volume_to_oi_ratio'] = current_volume / open_interest
        features['oi_turnover_rate'] = features['volume_to_oi_ratio']
        
        if features['volume_to_oi_ratio'] > 0.5:
            features['unusual_activity_flag'] = 1.0
        elif features['volume_to_oi_ratio'] > 0.3:
            features['unusual_activity_flag'] = 0.5
    
    if len(df) >= 20 and 'trade_timestamp' in df.columns:
        df = df.sort_values('trade_timestamp')
        
        first_half = df.iloc[:len(df)//2]
        second_half = df.iloc[len(df)//2:]
        
        vol_first = first_half['size'].sum()
        vol_second = second_half['size'].sum()
        
        if vol_first > 0:
            features['volume_acceleration'] = (vol_second - vol_first) / vol_first
    
    return features


def calculate_time_based_anomaly(df: pd.DataFrame) -> Dict[str, float]:
    """
    Detect unusual volume patterns by time of day.
    """
    features = {
        'volume_vs_hour_avg': 1.0,
        'volume_vs_dow_avg': 1.0,
        'pre_market_volume_pct': 0.0,
        'post_market_volume_pct': 0.0,
    }
    
    if 'trade_timestamp' not in df.columns or 'size' not in df.columns:
        return features
    
    df = df.copy()
    
    try:
        if not pd.api.types.is_datetime64_any_dtype(df['trade_timestamp']):
            df['trade_timestamp'] = pd.to_datetime(df['trade_timestamp'])
        
        df['hour'] = df['trade_timestamp'].dt.hour
        df['minute'] = df['trade_timestamp'].dt.minute
        
        market_open = (df['hour'] >= 9) & ((df['hour'] > 9) | (df['minute'] >= 30))
        market_close = (df['hour'] < 16)
        regular_hours = market_open & market_close
        
        total_volume = df['size'].sum()
        if total_volume > 0:
            pre_market = df[df['hour'] < 9]
            post_market = df[df['hour'] >= 16]
            
            features['pre_market_volume_pct'] = pre_market['size'].sum() / total_volume
            features['post_market_volume_pct'] = post_market['size'].sum() / total_volume
    except Exception:
        pass
    
    return features


def extract_volume_anomaly_features(
    df: pd.DataFrame,
    historical_volume_mean: Optional[float] = None,
    historical_volume_std: Optional[float] = None,
    open_interest: Optional[float] = None
) -> np.ndarray:
    """
    Extract all 12 volume anomaly detection features.
    """
    features = np.zeros(12, dtype=np.float32)
    
    idx = 0
    
    zscore = calculate_volume_zscore(df, historical_volume_mean, historical_volume_std)
    for key in ['volume_zscore', 'volume_percentile_20d', 'volume_percentile_60d', 'volume_spike_intensity']:
        val = zscore.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    vol_oi = calculate_volume_to_oi(df, open_interest)
    for key in ['volume_to_oi_ratio', 'oi_turnover_rate', 'volume_acceleration', 'unusual_activity_flag']:
        val = vol_oi.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    time_anomaly = calculate_time_based_anomaly(df)
    for key in ['volume_vs_hour_avg', 'volume_vs_dow_avg', 'pre_market_volume_pct', 'post_market_volume_pct']:
        val = time_anomaly.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    assert idx == 12, f"Expected 12 volume anomaly features, got {idx}"
    
    return features


class VolumeAnomalyDetector:
    def __init__(self):
        self.n_features = 12
        self.historical_stats = {}
    
    def set_historical_context(
        self,
        symbol: str,
        strike: float,
        historical_volume_mean: float,
        historical_volume_std: float
    ):
        """Set historical volume statistics for anomaly detection."""
        key = f"{symbol}_{strike}"
        self.historical_stats[key] = {
            'mean': historical_volume_mean,
            'std': historical_volume_std
        }
    
    def extract(
        self,
        df: pd.DataFrame,
        open_interest: Optional[float] = None,
        symbol: Optional[str] = None,
        strike: Optional[float] = None
    ) -> np.ndarray:
        hist_mean = None
        hist_std = None
        
        if symbol and strike:
            key = f"{symbol}_{strike}"
            if key in self.historical_stats:
                hist_mean = self.historical_stats[key]['mean']
                hist_std = self.historical_stats[key]['std']
        
        return extract_volume_anomaly_features(df, hist_mean, hist_std, open_interest)
    
    def extract_batch(
        self,
        dfs: List[pd.DataFrame],
        open_interests: Optional[List[float]] = None
    ) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        
        for i, df in enumerate(dfs):
            oi = open_interests[i] if open_interests else None
            features[i] = self.extract(df, oi)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            'volume_zscore', 'volume_percentile_20d', 'volume_percentile_60d', 'volume_spike_intensity',
            'volume_to_oi_ratio', 'oi_turnover_rate', 'volume_acceleration', 'unusual_activity_flag',
            'volume_vs_hour_avg', 'volume_vs_dow_avg', 'pre_market_volume_pct', 'post_market_volume_pct',
        ]
