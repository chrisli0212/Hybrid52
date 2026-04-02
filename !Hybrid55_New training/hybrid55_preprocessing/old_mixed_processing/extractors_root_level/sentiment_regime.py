"""
Hybrid51 Sentiment & Regime Features (Task 9)
Extracts 20 market sentiment and regime indicator features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .feature_config_v2 import FeatureGroup, FEATURE_GROUPS


def calculate_sentiment_scores(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "cp_sentiment": 0.0,
        "premium_sentiment": 0.0,
        "flow_sentiment": 0.0,
    }
    
    if 'right' not in df.columns:
        return features
    
    calls = df[df['right'] == 'C']
    puts = df[df['right'] == 'P']
    
    vol_col = 'size' if 'size' in df.columns else None
    
    if vol_col:
        call_vol = calls[vol_col].sum() if len(calls) > 0 else 0
        put_vol = puts[vol_col].sum() if len(puts) > 0 else 0
        total_vol = call_vol + put_vol
        
        if total_vol > 0:
            features["cp_sentiment"] = (call_vol - put_vol) / total_vol
    
    if 'price' in df.columns and vol_col:
        call_premium = (calls['price'] * calls[vol_col]).sum() if len(calls) > 0 else 0
        put_premium = (puts['price'] * puts[vol_col]).sum() if len(puts) > 0 else 0
        total_premium = call_premium + put_premium
        
        if total_premium > 0:
            features["premium_sentiment"] = (call_premium - put_premium) / total_premium
    
    if 'price' in df.columns and 'bid' in df.columns and 'ask' in df.columns and vol_col:
        df = df.copy()
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['is_buy'] = df['price'] >= df['mid']
        
        call_buys = calls[calls.index.isin(df[df['is_buy']].index)][vol_col].sum() if len(calls) > 0 else 0
        put_buys = puts[puts.index.isin(df[df['is_buy']].index)][vol_col].sum() if len(puts) > 0 else 0
        
        total_buys = call_buys + put_buys
        if total_buys > 0:
            features["flow_sentiment"] = (call_buys - put_buys) / total_buys
    
    return features


def calculate_volatility_regime(df: pd.DataFrame, historical_iv: Optional[pd.Series] = None) -> Dict[str, float]:
    features = {
        "iv_percentile": 50.0,
        "iv_expansion": 0.0,
        "iv_contraction": 0.0,
        "vol_regime": 0.0,
    }
    
    if 'implied_vol' not in df.columns:
        return features
    
    current_iv = df['implied_vol'].mean()
    
    if historical_iv is not None and len(historical_iv) > 0:
        features["iv_percentile"] = (historical_iv < current_iv).sum() / len(historical_iv) * 100
    else:
        features["iv_percentile"] = 50.0
    
    if len(df) > 10:
        recent_iv = df['implied_vol'].iloc[-10:].mean()
        older_iv = df['implied_vol'].iloc[:10].mean() if len(df) >= 20 else df['implied_vol'].mean()
        
        if older_iv > 0:
            iv_change = (recent_iv - older_iv) / older_iv
            features["iv_expansion"] = max(0, iv_change)
            features["iv_contraction"] = max(0, -iv_change)
    
    if features["iv_percentile"] > 70:
        features["vol_regime"] = 1.0
    elif features["iv_percentile"] < 30:
        features["vol_regime"] = -1.0
    else:
        features["vol_regime"] = 0.0
    
    return features


def calculate_trend_stress(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "momentum_1d": 0.0,
        "momentum_5d": 0.0,
        "momentum_20d": 0.0,
        "trend_strength": 0.0,
        "stress_indicator": 0.0,
        "fear_indicator": 0.0,
    }
    
    if 'underlying_price' not in df.columns:
        return features
    
    prices = df['underlying_price']
    if len(prices) > 0:
        current_price = prices.iloc[-1]
        
        if len(prices) > 1:
            features["momentum_1d"] = (current_price / prices.iloc[0] - 1) * 100
        
        if len(prices) > 5:
            features["momentum_5d"] = (current_price / prices.iloc[-5] - 1) * 100
        
        if len(prices) > 20:
            features["momentum_20d"] = (current_price / prices.iloc[-20] - 1) * 100
    
    if abs(features["momentum_20d"]) > 0.01:
        features["trend_strength"] = np.sign(features["momentum_20d"]) * min(1.0, abs(features["momentum_20d"]) / 10)
    
    if 'implied_vol' in df.columns and 'right' in df.columns:
        puts = df[df['right'] == 'P']
        calls = df[df['right'] == 'C']
        
        put_iv = puts['implied_vol'].mean() if len(puts) > 0 else 0
        call_iv = calls['implied_vol'].mean() if len(calls) > 0 else 1
        
        features["stress_indicator"] = put_iv - call_iv if call_iv > 0 else 0
        
        if put_iv > 0 and call_iv > 0:
            features["fear_indicator"] = put_iv / call_iv - 1
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def calculate_correlation_metrics(df: pd.DataFrame, market_data: Optional[Dict] = None) -> Dict[str, float]:
    features = {
        "spx_corr": 0.0,
        "vix_corr": 0.0,
        "sector_corr": 0.0,
        "beta_to_spx": 1.0,
    }
    
    if 'underlying_price' not in df.columns or len(df) < 5:
        return features
    
    prices = df['underlying_price']
    returns = prices.pct_change().dropna()
    
    if market_data is not None:
        if 'spx_returns' in market_data and len(market_data['spx_returns']) >= len(returns):
            spx_ret = market_data['spx_returns'][:len(returns)]
            if len(returns) > 2:
                features["spx_corr"] = returns.corr(pd.Series(spx_ret))
                if spx_ret.std() > 0:
                    features["beta_to_spx"] = returns.cov(pd.Series(spx_ret)) / spx_ret.var()
        
        if 'vix_returns' in market_data and len(market_data['vix_returns']) >= len(returns):
            vix_ret = market_data['vix_returns'][:len(returns)]
            if len(returns) > 2:
                features["vix_corr"] = returns.corr(pd.Series(vix_ret))
    else:
        features["spx_corr"] = 0.5
        features["beta_to_spx"] = 1.0
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def calculate_vix_relative(df: pd.DataFrame, vix_level: Optional[float] = None) -> Dict[str, float]:
    features = {
        "iv_vix_ratio": 1.0,
        "relative_iv": 0.0,
        "vix_term_impact": 0.0,
    }
    
    if 'implied_vol' not in df.columns:
        return features
    
    avg_iv = df['implied_vol'].mean()
    
    if vix_level is None:
        vix_level = 20.0
    
    if vix_level > 0:
        features["iv_vix_ratio"] = avg_iv / vix_level
        features["relative_iv"] = avg_iv - vix_level
    
    if 'expiration' in df.columns:
        near = df[df['expiration'] == df['expiration'].min()]
        far = df[df['expiration'] == df['expiration'].max()]
        
        near_iv = near['implied_vol'].mean() if len(near) > 0 else 0
        far_iv = far['implied_vol'].mean() if len(far) > 0 else 0
        
        if vix_level > 0:
            features["vix_term_impact"] = (far_iv - near_iv) / vix_level
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def extract_sentiment_regime_features(
    df: pd.DataFrame,
    historical_iv: Optional[pd.Series] = None,
    market_data: Optional[Dict] = None,
    vix_level: Optional[float] = None
) -> np.ndarray:
    features = np.zeros(20, dtype=np.float32)
    
    idx = 0
    
    sentiment = calculate_sentiment_scores(df)
    for key in ["cp_sentiment", "premium_sentiment", "flow_sentiment"]:
        features[idx] = sentiment.get(key, 0.0)
        idx += 1
    
    vol_regime = calculate_volatility_regime(df, historical_iv)
    for key in ["iv_percentile", "iv_expansion", "iv_contraction", "vol_regime"]:
        features[idx] = vol_regime.get(key, 0.0)
        idx += 1
    
    trend = calculate_trend_stress(df)
    for key in ["momentum_1d", "momentum_5d", "momentum_20d",
                "trend_strength", "stress_indicator", "fear_indicator"]:
        features[idx] = trend.get(key, 0.0)
        idx += 1
    
    corr = calculate_correlation_metrics(df, market_data)
    for key in ["spx_corr", "vix_corr", "sector_corr", "beta_to_spx"]:
        features[idx] = corr.get(key, 0.0)
        idx += 1
    
    vix_rel = calculate_vix_relative(df, vix_level)
    for key in ["iv_vix_ratio", "relative_iv", "vix_term_impact"]:
        features[idx] = vix_rel.get(key, 0.0)
        idx += 1
    
    assert idx == 20, f"Expected 20 sentiment/regime features, got {idx}"
    
    return features


class SentimentRegimeExtractor:
    def __init__(self):
        self.feature_group = FEATURE_GROUPS[FeatureGroup.SENTIMENT_REGIME]
        self.n_features = self.feature_group.num_features
        self.historical_iv = None
        self.market_data = None
        self.vix_level = None
    
    def set_context(
        self,
        historical_iv: Optional[pd.Series] = None,
        market_data: Optional[Dict] = None,
        vix_level: Optional[float] = None
    ):
        self.historical_iv = historical_iv
        self.market_data = market_data
        self.vix_level = vix_level
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_sentiment_regime_features(
            df,
            self.historical_iv,
            self.market_data,
            self.vix_level
        )
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            "cp_sentiment", "premium_sentiment", "flow_sentiment",
            "iv_percentile", "iv_expansion", "iv_contraction", "vol_regime",
            "momentum_1d", "momentum_5d", "momentum_20d",
            "trend_strength", "stress_indicator", "fear_indicator",
            "spx_corr", "vix_corr", "sector_corr", "beta_to_spx",
            "iv_vix_ratio", "relative_iv", "vix_term_impact",
        ]
