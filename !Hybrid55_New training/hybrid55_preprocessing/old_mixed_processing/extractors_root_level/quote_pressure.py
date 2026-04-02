"""
Hybrid51 Quote Pressure & Exchange Routing Features
Extracts 20 features: 10 quote pressure + 10 exchange routing.

Tape reading and order book dynamics analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def calculate_cumulative_volume_delta(df: pd.DataFrame) -> Dict[str, float]:
    """
    Cumulative Volume Delta (CVD) - running sum of buy vol - sell vol.
    Professional tape reading metric.
    """
    features = {
        'cvd_total': 0.0,
        'cvd_momentum': 0.0,
        'cvd_divergence': 0.0,
    }
    
    if 'price' not in df.columns or 'bid' not in df.columns or 'ask' not in df.columns:
        return features
    
    if 'size' not in df.columns or len(df) < 2:
        return features
    
    df = df.copy()
    df['mid'] = (df['bid'] + df['ask']) / 2
    
    df['side'] = 0
    df.loc[df['price'] >= df['mid'], 'side'] = 1
    df.loc[df['price'] < df['mid'], 'side'] = -1
    
    df['signed_volume'] = df['side'] * df['size']
    
    features['cvd_total'] = df['signed_volume'].sum()
    
    if len(df) >= 10:
        recent = df.iloc[-10:]
        older = df.iloc[:-10]
        
        recent_cvd = recent['signed_volume'].sum()
        older_cvd = older['signed_volume'].sum() if len(older) > 0 else 0
        
        features['cvd_momentum'] = recent_cvd - older_cvd
    
    if 'underlying_price' in df.columns and len(df) > 1:
        price_change = df['underlying_price'].iloc[-1] - df['underlying_price'].iloc[0]
        cvd = features['cvd_total']
        
        if price_change > 0 and cvd < 0:
            features['cvd_divergence'] = -1.0
        elif price_change < 0 and cvd > 0:
            features['cvd_divergence'] = 1.0
    
    return features


def calculate_quote_pressure(df: pd.DataFrame) -> Dict[str, float]:
    """
    Bid/ask pressure and order book dynamics.
    """
    features = {
        'bid_pressure': 0.0,
        'quote_update_frequency': 0.0,
        'quote_improvement_rate': 0.0,
    }
    
    if 'bid_size' not in df.columns or 'ask_size' not in df.columns:
        return features
    
    total_bid = df['bid_size'].sum()
    total_ask = df['ask_size'].sum()
    total = total_bid + total_ask
    
    if total > 0:
        features['bid_pressure'] = (total_bid - total_ask) / total
    
    if 'quote_timestamp' in df.columns:
        try:
            df_sorted = df.sort_values('quote_timestamp')
            if 'bid' in df_sorted.columns and 'ask' in df_sorted.columns:
                ba = df_sorted[['bid', 'ask']].copy()
                ba = ba.fillna(method='ffill').fillna(method='bfill')
                changes = (ba.diff().abs().sum(axis=1) > 0).sum()
                features['quote_update_frequency'] = float(changes)
            else:
                features['quote_update_frequency'] = float(len(df))
        except Exception:
            features['quote_update_frequency'] = float(len(df))
    
    if 'bid' in df.columns and 'ask' in df.columns and len(df) > 1:
        df = df.copy()
        df['bid_improved'] = df['bid'].diff() > 0
        df['ask_improved'] = df['ask'].diff() < 0
        df['improved'] = df['bid_improved'] | df['ask_improved']
        
        features['quote_improvement_rate'] = df['improved'].sum() / len(df)
    
    return features


def calculate_tape_reading_signals(df: pd.DataFrame) -> Dict[str, float]:
    """
    Tape reading: print clustering, trade sequences, absorption.
    """
    features = {
        'print_clustering_score': 0.0,
        'trade_sequence_momentum': 0.0,
        'absorption_quality': 0.0,
        'tape_reading_signal': 0.0,
    }
    
    if 'trade_timestamp' not in df.columns or len(df) < 5:
        return features
    
    if 'sequence' in df.columns:
        df = df.sort_values(['trade_timestamp', 'sequence'])
    else:
        df = df.sort_values('trade_timestamp')
    df = df.copy()
    df['time_delta'] = df['trade_timestamp'].diff().dt.total_seconds()
    
    clustered = df[df['time_delta'] < 1.0]
    features['print_clustering_score'] = len(clustered) / len(df)
    
    if 'price' in df.columns and 'bid' in df.columns and 'ask' in df.columns:
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['above_mid'] = (df['price'] >= df['mid']).astype(int) * 2 - 1
        
        if len(df) >= 10:
            recent = df.iloc[-10:]
            features['trade_sequence_momentum'] = recent['above_mid'].sum() / 10
    
    if 'size' in df.columns and 'price' in df.columns and len(df) >= 5:
        df['price_change'] = df['price'].diff().abs()
        df['size_norm'] = df['size'] / df['size'].mean()
        
        large_trades = df[df['size_norm'] > 2.0]
        if len(large_trades) > 0:
            avg_impact = large_trades['price_change'].mean()
            overall_impact = df['price_change'].mean()
            
            if overall_impact > 0:
                features['absorption_quality'] = 1.0 - min(1.0, avg_impact / overall_impact)
    
    features['tape_reading_signal'] = (
        features['trade_sequence_momentum'] * 0.4 +
        features['print_clustering_score'] * 0.3 +
        features['absorption_quality'] * 0.3
    )
    
    return features


def calculate_order_book_dynamics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Order book depth and liquidity imbalance.
    """
    features = {
        'depth_ratio': 1.0,
        'liquidity_imbalance_score': 0.0,
    }
    
    if 'bid_size' not in df.columns or 'ask_size' not in df.columns:
        return features
    
    total_bid_depth = df['bid_size'].sum()
    total_ask_depth = df['ask_size'].sum()
    
    if total_ask_depth > 0:
        features['depth_ratio'] = total_bid_depth / total_ask_depth
    
    if 'bid' in df.columns and 'ask' in df.columns:
        total_liquidity = total_bid_depth + total_ask_depth
        if total_liquidity > 0:
            features['liquidity_imbalance_score'] = (total_bid_depth - total_ask_depth) / total_liquidity
    
    return features


def calculate_exchange_routing(df: pd.DataFrame) -> Dict[str, float]:
    """
    Exchange routing patterns and diversity.
    """
    features = {
        'cboe_pct': 0.0,
        'phlx_pct': 0.0,
        'ise_pct': 0.0,
        'exchange_diversity': 0.0,
        'multi_exchange_trades': 0.0,
        'exchange_concentration': 1.0,
    }
    
    if ('bid_exchange' in df.columns or 'ask_exchange' in df.columns):
        be = df['bid_exchange'] if 'bid_exchange' in df.columns else None
        ae = df['ask_exchange'] if 'ask_exchange' in df.columns else None
        frames = []
        if be is not None:
            frames.append(be.dropna().rename('ex'))
        if ae is not None:
            frames.append(ae.dropna().rename('ex'))
        if not frames:
            return features

        all_ex = pd.concat(frames, axis=0)
        total = len(all_ex)
        if total == 0:
            return features

        exchange_volume = all_ex.value_counts()
        total_volume = float(total)
    else:
        if 'exchange' not in df.columns or 'size' not in df.columns:
            return features
        
        total_volume = df['size'].sum()
        if total_volume == 0:
            return features
        
        exchange_volume = df.groupby('exchange')['size'].sum()
    
    # Map exchange codes (can be int or string)
    # CBOE=C=67, PHLX=P=80, ISE=I=73, AMEX=A=65
    exchange_map = {
        'CBOE': 'cboe_pct',
        'C': 'cboe_pct',
        67: 'cboe_pct',
        'PHLX': 'phlx_pct',
        'P': 'phlx_pct',
        80: 'phlx_pct',
        'ISE': 'ise_pct',
        'I': 'ise_pct',
        73: 'ise_pct',
    }
    
    for exchange, vol in exchange_volume.items():
        pct = float(vol) / float(total_volume)
        
        # Direct match for int codes or exact string match
        if exchange in exchange_map:
            features[exchange_map[exchange]] += pct
        elif isinstance(exchange, str):
            # Check if string starts with exchange code
            for ex_key, feature_name in exchange_map.items():
                if isinstance(ex_key, str) and exchange.startswith(ex_key):
                    features[feature_name] += pct
                    break
    
    features['exchange_diversity'] = float(len(exchange_volume))

    if 'bid_exchange' in df.columns and 'ask_exchange' in df.columns:
        try:
            split = (df['bid_exchange'].notna() & df['ask_exchange'].notna() & (df['bid_exchange'] != df['ask_exchange'])).any()
            features['multi_exchange_trades'] = 1.0 if split else 0.0
        except Exception:
            features['multi_exchange_trades'] = 0.0
    else:
        if len(exchange_volume) > 1:
            features['multi_exchange_trades'] = 1.0
    
    if len(exchange_volume) > 0:
        max_vol = exchange_volume.max()
        features['exchange_concentration'] = float(max_vol) / float(total_volume)
    
    return features


def extract_quote_pressure_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract all 18 quote pressure & exchange routing features.
    """
    features = np.zeros(18, dtype=np.float32)
    
    idx = 0
    
    cvd = calculate_cumulative_volume_delta(df)
    for key in ['cvd_total', 'cvd_momentum', 'cvd_divergence']:
        val = cvd.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    pressure = calculate_quote_pressure(df)
    for key in ['bid_pressure', 'quote_update_frequency', 'quote_improvement_rate']:
        val = pressure.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    tape = calculate_tape_reading_signals(df)
    for key in ['print_clustering_score', 'trade_sequence_momentum', 'absorption_quality', 'tape_reading_signal']:
        val = tape.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    book = calculate_order_book_dynamics(df)
    for key in ['depth_ratio', 'liquidity_imbalance_score']:
        val = book.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    routing = calculate_exchange_routing(df)
    for key in ['cboe_pct', 'phlx_pct', 'ise_pct', 'exchange_diversity', 'multi_exchange_trades', 'exchange_concentration']:
        val = routing.get(key, 0.0)
        features[idx] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        idx += 1
    
    assert idx == 18, f"Expected 18 quote/routing features, got {idx}"
    
    return features


class QuotePressureAnalyzer:
    def __init__(self):
        self.n_features = 18
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_quote_pressure_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            'cvd_total', 'cvd_momentum', 'cvd_divergence',
            'bid_pressure', 'quote_update_frequency', 'quote_improvement_rate',
            'print_clustering_score', 'trade_sequence_momentum', 'absorption_quality', 'tape_reading_signal',
            'depth_ratio', 'liquidity_imbalance_score',
            'cboe_pct', 'phlx_pct', 'ise_pct', 'exchange_diversity', 'multi_exchange_trades', 'exchange_concentration',
        ]
