"""
Hybrid51 Cross-Strike & Time Decay Features (Task 8)
Extracts 30 features: Cross-Strike (15) + Time Decay (15)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .feature_config_v2 import DTE_BUCKETS, FeatureGroup, FEATURE_GROUPS


def _resolve_oi_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("oi", "open_interest", "size"):
        if col in df.columns:
            return col
    return None


def calculate_oi_volume_distribution(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "oi_concentration": 0.0,
        "vol_concentration_gini": 0.0,
        "oi_skewness": 0.0,
        "vol_skewness": 0.0,
        "oi_vol_corr": 0.0,
        "oi_vol_divergence": 0.0,
    }
    
    oi_col = _resolve_oi_col(df)
    vol_col = 'size' if 'size' in df.columns else None
    
    if 'strike' not in df.columns:
        return features
    
    if oi_col:
        oi_by_strike = df.groupby('strike')[oi_col].sum()
        if len(oi_by_strike) > 0:
            total_oi = oi_by_strike.sum()
            if total_oi > 0:
                top_5 = oi_by_strike.nlargest(5).sum()
                features["oi_concentration"] = top_5 / total_oi
                
                features["oi_skewness"] = oi_by_strike.skew() if len(oi_by_strike) > 2 else 0.0
                
                sorted_oi = oi_by_strike.sort_values()
                n = len(sorted_oi)
                if n > 1:
                    cumsum = sorted_oi.cumsum() / sorted_oi.sum()
                    gini = 1 - 2 * cumsum.mean()
                    features["vol_concentration_gini"] = gini
    
    if vol_col:
        vol_by_strike = df.groupby('strike')[vol_col].sum()
        if len(vol_by_strike) > 2:
            features["vol_skewness"] = vol_by_strike.skew()
    
    if oi_col and vol_col:
        oi_by_strike = df.groupby('strike')[oi_col].sum()
        vol_by_strike = df.groupby('strike')[vol_col].sum()
        
        if len(oi_by_strike) > 2 and len(vol_by_strike) > 2:
            common_strikes = oi_by_strike.index.intersection(vol_by_strike.index)
            if len(common_strikes) > 2:
                features["oi_vol_corr"] = oi_by_strike[common_strikes].corr(vol_by_strike[common_strikes])
                
                oi_norm = oi_by_strike[common_strikes] / oi_by_strike[common_strikes].sum()
                vol_norm = vol_by_strike[common_strikes] / vol_by_strike[common_strikes].sum()
                features["oi_vol_divergence"] = (oi_norm - vol_norm).abs().sum()
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def calculate_greek_concentrations(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "gamma_concentration": 0.0,
        "vanna_concentration": 0.0,
        "charm_concentration": 0.0,
    }
    
    if 'strike' not in df.columns:
        return features
    
    for greek in ['gamma', 'vanna', 'charm']:
        if greek in df.columns:
            greek_by_strike = df.groupby('strike')[greek].sum().abs()
            if len(greek_by_strike) > 0:
                total = greek_by_strike.sum()
                if total > 0:
                    top_5 = greek_by_strike.nlargest(5).sum()
                    features[f"{greek}_concentration"] = top_5 / total
    
    return features


def calculate_strike_clustering(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "active_strikes": 0.0,
        "strike_uniformity": 0.0,
        "atm_clustering": 0.0,
    }
    
    if 'strike' not in df.columns:
        return features
    
    vol_col = 'size' if 'size' in df.columns else None
    
    features["active_strikes"] = df['strike'].nunique()
    
    if vol_col:
        vol_by_strike = df.groupby('strike')[vol_col].sum()
        if len(vol_by_strike) > 1:
            cv = vol_by_strike.std() / vol_by_strike.mean() if vol_by_strike.mean() > 0 else 0
            features["strike_uniformity"] = 1 / (1 + cv)
    
    if 'underlying_price' in df.columns:
        underlying = df['underlying_price'].iloc[0]
    else:
        underlying = df['strike'].median()
    
    atm_range = 0.05
    atm_strikes = df[(df['strike'] >= underlying * (1 - atm_range)) & 
                     (df['strike'] <= underlying * (1 + atm_range))]
    
    if vol_col and len(atm_strikes) > 0:
        atm_vol = atm_strikes[vol_col].sum()
        total_vol = df[vol_col].sum()
        features["atm_clustering"] = atm_vol / total_vol if total_vol > 0 else 0
    else:
        features["atm_clustering"] = len(atm_strikes) / len(df) if len(df) > 0 else 0
    
    return features


def calculate_liquidity_gradient(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "liquidity_decay": 0.0,
        "otm_put_liquidity": 0.0,
        "otm_call_liquidity": 0.0,
    }
    
    if 'strike' not in df.columns:
        return features
    
    if 'underlying_price' in df.columns:
        underlying = df['underlying_price'].iloc[0]
    else:
        underlying = df['strike'].median()
    
    vol_col = 'size' if 'size' in df.columns else None
    
    if vol_col:
        df = df.copy()
        df['dist_from_atm'] = abs(df['strike'] - underlying) / underlying
        
        if len(df) > 5:
            near = df[df['dist_from_atm'] <= 0.05][vol_col].sum()
            far = df[df['dist_from_atm'] > 0.15][vol_col].sum()
            features["liquidity_decay"] = far / near if near > 0 else 0
        
        if 'right' in df.columns:
            otm_puts = df[(df['right'] == 'P') & (df['strike'] < underlying)]
            otm_calls = df[(df['right'] == 'C') & (df['strike'] > underlying)]
        else:
            otm_puts = df[(df['delta'] < 0) & (df['strike'] < underlying)]
            otm_calls = df[(df['delta'] > 0) & (df['strike'] > underlying)]
        
        total_vol = df[vol_col].sum()
        if total_vol > 0:
            features["otm_put_liquidity"] = otm_puts[vol_col].sum() / total_vol if len(otm_puts) > 0 else 0
            features["otm_call_liquidity"] = otm_calls[vol_col].sum() / total_vol if len(otm_calls) > 0 else 0
    
    return features


def extract_cross_strike_features(df: pd.DataFrame) -> np.ndarray:
    features = np.zeros(15, dtype=np.float32)
    
    idx = 0
    
    oi_vol = calculate_oi_volume_distribution(df)
    for key in ["oi_concentration", "vol_concentration_gini", "oi_skewness",
                "vol_skewness", "oi_vol_corr", "oi_vol_divergence"]:
        features[idx] = oi_vol.get(key, 0.0)
        idx += 1
    
    greek_conc = calculate_greek_concentrations(df)
    for key in ["gamma_concentration", "vanna_concentration", "charm_concentration"]:
        features[idx] = greek_conc.get(key, 0.0)
        idx += 1
    
    clustering = calculate_strike_clustering(df)
    for key in ["active_strikes", "strike_uniformity", "atm_clustering"]:
        features[idx] = clustering.get(key, 0.0)
        idx += 1
    
    liquidity = calculate_liquidity_gradient(df)
    for key in ["liquidity_decay", "otm_put_liquidity", "otm_call_liquidity"]:
        features[idx] = liquidity.get(key, 0.0)
        idx += 1
    
    assert idx == 15, f"Expected 15 cross-strike features, got {idx}"
    
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


def calculate_dte_bucket_oi(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "oi_0_7d": 0.0,
        "oi_8_30d": 0.0,
        "oi_31_60d": 0.0,
        "oi_61_90d": 0.0,
        "oi_90plus": 0.0,
    }
    
    df = df.copy()
    df['dte'] = calculate_dte(df)
    
    oi_col = _resolve_oi_col(df)
    if oi_col is None:
        return features
    
    total_oi = df[oi_col].sum()
    if total_oi == 0:
        return features
    
    for bucket_name, low, high in DTE_BUCKETS:
        bucket_df = df[(df['dte'] >= low) & (df['dte'] <= high)]
        key = f"oi_{bucket_name}"
        features[key] = bucket_df[oi_col].sum() / total_oi if len(bucket_df) > 0 else 0.0
    
    return features


def calculate_decay_accelerations(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "theta_accel": 0.0,
        "gamma_accel": 0.0,
        "charm_accel": 0.0,
        "weighted_theta": 0.0,
        "weighted_charm": 0.0,
    }
    
    df = df.copy()
    df['dte'] = calculate_dte(df)
    
    if 'theta' in df.columns:
        near_term = df[df['dte'] <= 7]
        mid_term = df[(df['dte'] > 7) & (df['dte'] <= 30)]
        
        near_theta = near_term['theta'].mean() if len(near_term) > 0 else 0
        mid_theta = mid_term['theta'].mean() if len(mid_term) > 0 else 1
        
        features["theta_accel"] = near_theta / mid_theta if abs(mid_theta) > 1e-10 else 0
        
        oi_col = _resolve_oi_col(df)
        if oi_col:
            total_oi = df[oi_col].sum()
            if total_oi > 0:
                features["weighted_theta"] = (df['theta'] * df[oi_col]).sum() / total_oi
    
    if 'gamma' in df.columns:
        near_term = df[df['dte'] <= 7]
        mid_term = df[(df['dte'] > 7) & (df['dte'] <= 30)]
        
        near_gamma = near_term['gamma'].mean() if len(near_term) > 0 else 0
        mid_gamma = mid_term['gamma'].mean() if len(mid_term) > 0 else 1
        
        features["gamma_accel"] = near_gamma / mid_gamma if abs(mid_gamma) > 1e-10 else 0
    
    if 'charm' in df.columns:
        near_term = df[df['dte'] <= 7]
        mid_term = df[(df['dte'] > 7) & (df['dte'] <= 30)]
        
        near_charm = near_term['charm'].mean() if len(near_term) > 0 else 0
        mid_charm = mid_term['charm'].mean() if len(mid_term) > 0 else 1
        
        features["charm_accel"] = near_charm / mid_charm if abs(mid_charm) > 1e-10 else 0
        
        oi_col = _resolve_oi_col(df)
        if oi_col:
            total_oi = df[oi_col].sum()
            if total_oi > 0:
                features["weighted_charm"] = (df['charm'] * df[oi_col]).sum() / total_oi
    
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0
    
    return features


def calculate_time_concentrations(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "near_term_oi_conc": 0.0,
        "near_term_vol_conc": 0.0,
        "near_term_gamma_conc": 0.0,
    }
    
    df = df.copy()
    df['dte'] = calculate_dte(df)
    
    near_term = df[df['dte'] <= 7]
    
    oi_col = _resolve_oi_col(df)
    
    if oi_col:
        total_oi = df[oi_col].sum()
        if total_oi > 0 and len(near_term) > 0:
            features["near_term_oi_conc"] = near_term[oi_col].sum() / total_oi
    
    if 'size' in df.columns:
        total_vol = df['size'].sum()
        if total_vol > 0 and len(near_term) > 0:
            features["near_term_vol_conc"] = near_term['size'].sum() / total_vol
    
    if 'gamma' in df.columns:
        total_gamma = df['gamma'].abs().sum()
        if total_gamma > 0 and len(near_term) > 0:
            features["near_term_gamma_conc"] = near_term['gamma'].abs().sum() / total_gamma
    
    return features


def calculate_calendar_proximity(df: pd.DataFrame) -> Dict[str, float]:
    features = {
        "days_to_major_exp": 30.0,
        "days_to_opex": 30.0,
    }
    
    df = df.copy()
    df['dte'] = calculate_dte(df)
    
    if len(df) > 0:
        features["days_to_major_exp"] = df['dte'].min()
        
        monthly_dte = df[df['dte'] % 7 == 0]['dte'].min()
        features["days_to_opex"] = monthly_dte if not np.isnan(monthly_dte) else df['dte'].min()
    
    return features


def extract_time_decay_features(df: pd.DataFrame) -> np.ndarray:
    features = np.zeros(15, dtype=np.float32)
    
    idx = 0
    
    dte_oi = calculate_dte_bucket_oi(df)
    for key in ["oi_0_7d", "oi_8_30d", "oi_31_60d", "oi_61_90d", "oi_90plus"]:
        features[idx] = dte_oi.get(key, 0.0)
        idx += 1
    
    decay = calculate_decay_accelerations(df)
    for key in ["theta_accel", "gamma_accel", "charm_accel", "weighted_theta", "weighted_charm"]:
        features[idx] = decay.get(key, 0.0)
        idx += 1
    
    conc = calculate_time_concentrations(df)
    for key in ["near_term_oi_conc", "near_term_vol_conc", "near_term_gamma_conc"]:
        features[idx] = conc.get(key, 0.0)
        idx += 1
    
    calendar = calculate_calendar_proximity(df)
    for key in ["days_to_major_exp", "days_to_opex"]:
        features[idx] = calendar.get(key, 0.0)
        idx += 1
    
    assert idx == 15, f"Expected 15 time decay features, got {idx}"
    
    return features


class CrossStrikeExtractor:
    def __init__(self):
        self.feature_group = FEATURE_GROUPS[FeatureGroup.CROSS_STRIKE]
        self.n_features = self.feature_group.num_features
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_cross_strike_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            "oi_concentration", "vol_concentration_gini", "oi_skewness",
            "vol_skewness", "oi_vol_corr", "oi_vol_divergence",
            "gamma_concentration", "vanna_concentration", "charm_concentration",
            "active_strikes", "strike_uniformity", "atm_clustering",
            "liquidity_decay", "otm_put_liquidity", "otm_call_liquidity",
        ]


class TimeDecayExtractor:
    def __init__(self):
        self.feature_group = FEATURE_GROUPS[FeatureGroup.TIME_DECAY]
        self.n_features = self.feature_group.num_features
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        return extract_time_decay_features(df)
    
    def extract_batch(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        n_samples = len(dfs)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)
        for i, df in enumerate(dfs):
            features[i] = self.extract(df)
        return features
    
    def get_feature_names(self) -> List[str]:
        return [
            "oi_0_7d", "oi_8_30d", "oi_31_60d", "oi_61_90d", "oi_90plus",
            "theta_accel", "gamma_accel", "charm_accel", "weighted_theta", "weighted_charm",
            "near_term_oi_conc", "near_term_vol_conc", "near_term_gamma_conc",
            "days_to_major_exp", "days_to_opex",
        ]
