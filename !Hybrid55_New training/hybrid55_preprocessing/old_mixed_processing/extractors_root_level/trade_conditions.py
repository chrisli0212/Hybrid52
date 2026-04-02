"""
Hybrid51 Trade Condition Analysis Features
Extracts 10 features from OPRA trade condition codes.

Decodes institutional order types: ISOs, complex orders, auctions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set


OPRA_CONDITION_CODES = {
    18: {'name': 'regular_sale', 'type': 'regular', 'urgency': 'normal'},
    125: {'name': 'opening_trade', 'type': 'opening', 'urgency': 'normal'},
    130: {'name': 'intermarket_sweep', 'type': 'iso', 'urgency': 'high'},
    131: {'name': 'sweep_extended_hours', 'type': 'iso', 'urgency': 'high'},
    95: {'name': 'single_leg_auction', 'type': 'auction', 'urgency': 'low'},
    133: {'name': 'complex_trade', 'type': 'complex', 'urgency': 'normal'},
    138: {'name': 'stock_option_trade', 'type': 'complex', 'urgency': 'normal'},
    120: {'name': 'cancel', 'type': 'cancel', 'urgency': 'none'},
    121: {'name': 'cancel_last', 'type': 'cancel', 'urgency': 'none'},
}

ISO_CONDITIONS = {130, 131}
COMPLEX_CONDITIONS = {133, 138}
AUCTION_CONDITIONS = {95}
OPENING_CONDITIONS = {125}
CANCEL_CONDITIONS = {120, 121}


def parse_trade_conditions(df: pd.DataFrame) -> Dict[str, float]:
    """
    Parse trade condition codes to identify order types.
    """
    features = {
        'is_iso': 0.0,
        'is_complex': 0.0,
        'is_opening': 0.0,
        'is_closing': 0.0,
        'is_auction': 0.0,
        'is_contingent': 0.0,
    }
    
    if 'condition' not in df.columns:
        return features
    
    conditions = set(df['condition'].unique())
    
    features['is_iso'] = 1.0 if ISO_CONDITIONS & conditions else 0.0
    features['is_complex'] = 1.0 if COMPLEX_CONDITIONS & conditions else 0.0
    features['is_opening'] = 1.0 if OPENING_CONDITIONS & conditions else 0.0
    features['is_auction'] = 1.0 if AUCTION_CONDITIONS & conditions else 0.0
    
    features['is_contingent'] = features['is_complex']
    
    return features


def calculate_condition_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate volume distribution by condition type.
    """
    features = {
        'iso_volume_pct': 0.0,
        'complex_order_pct': 0.0,
        'auction_participation_pct': 0.0,
        'condition_diversity': 0.0,
    }
    
    if 'condition' not in df.columns or 'size' not in df.columns:
        return features
    
    total_volume = df['size'].sum()
    if total_volume == 0:
        return features
    
    iso_trades = df[df['condition'].isin(ISO_CONDITIONS)]
    complex_trades = df[df['condition'].isin(COMPLEX_CONDITIONS)]
    auction_trades = df[df['condition'].isin(AUCTION_CONDITIONS)]
    
    features['iso_volume_pct'] = iso_trades['size'].sum() / total_volume
    features['complex_order_pct'] = complex_trades['size'].sum() / total_volume
    features['auction_participation_pct'] = auction_trades['size'].sum() / total_volume
    
    unique_conditions = df['condition'].nunique()
    features['condition_diversity'] = unique_conditions
    
    return features


def extract_trade_condition_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract all 10 trade condition features.
    """
    features = np.zeros(10, dtype=np.float32)
    
    idx = 0
    
    parsed = parse_trade_conditions(df)
    for key in ['is_iso', 'is_complex', 'is_opening', 'is_closing', 'is_auction', 'is_contingent']:
        val = parsed.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    stats = calculate_condition_statistics(df)
    for key in ['iso_volume_pct', 'complex_order_pct', 'auction_participation_pct', 'condition_diversity']:
        val = stats.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    assert idx == 10, f"Expected 10 trade condition features, got {idx}"
    
    return features


class TradeConditionAnalyzer:
    def __init__(self):
        self.n_features = 10
        self.condition_codes = OPRA_CONDITION_CODES
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_trade_condition_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            'is_iso', 'is_complex', 'is_opening', 'is_closing', 'is_auction', 'is_contingent',
            'iso_volume_pct', 'complex_order_pct', 'auction_participation_pct', 'condition_diversity',
        ]
    
    def decode_condition(self, code: int) -> Dict:
        """Decode OPRA condition code to readable info."""
        return self.condition_codes.get(code, {
            'name': f'unknown_{code}',
            'type': 'unknown',
            'urgency': 'unknown'
        })
    
    def get_urgency_score(self, df: pd.DataFrame) -> float:
        """Calculate composite urgency score (0-1, higher = more urgent)."""
        if 'condition' not in df.columns or 'size' not in df.columns:
            return 0.0
        
        urgency_weights = {
            'high': 1.0,
            'normal': 0.5,
            'low': 0.2,
            'none': 0.0,
        }
        
        total_size = df['size'].sum()
        if total_size == 0:
            return 0.0
        
        weighted_urgency = 0.0
        for _, row in df.iterrows():
            condition_info = self.decode_condition(row['condition'])
            urgency = condition_info.get('urgency', 'normal')
            weight = urgency_weights.get(urgency, 0.5)
            weighted_urgency += weight * row['size']
        
        return weighted_urgency / total_size
