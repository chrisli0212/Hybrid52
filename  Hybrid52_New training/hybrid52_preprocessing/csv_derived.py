"""
CsvDerivedExtractor — dims 270-285 (16 dims)
Extracts pre-computed CSV columns not captured by any other extractor:
  270: lambda_mean        - mean leverage ratio across active chain
  271: lambda_atm         - leverage ratio at ATM strike
  272: lambda_skew        - (call_lambda - put_lambda) / mean
  273: dist_atm_mean      - mean |dist_atm_pct| across active chain
  274: dist_atm_weighted  - volume-weighted dist_atm_pct
  275: spread_pct_mean    - mean spread_pct (liquidity proxy)
  276: spread_pct_atm     - spread_pct at ATM strike
  277: spread_pct_skew    - OTM_put_spread / OTM_call_spread
  278: dual_delta_mean    - mean dual_delta across active chain
  279: dual_gamma_mean    - mean dual_gamma across active chain
  280: d1_atm             - d1 at ATM strike
  281: d2_atm             - d2 at ATM strike
  282: iv_error_mean      - mean iv fitting error
  283: ultima_mean        - mean ultima across chain
  284: oi_mean            - mean open interest across active chain
  285: oi_put_call_ratio  - put OI / call OI ratio
"""
import numpy as np
import pandas as pd

NUM_FEATURES = 16
START_DIM = 270


class CsvDerivedExtractor:
    """Extracts 16 CSV-native features (dims 270-285)."""

    def extract(self, greek_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(NUM_FEATURES, dtype=np.float32)
        if greek_df is None or len(greek_df) == 0:
            return out

        df = greek_df.copy()
        has_lambda = ('lambda_ratio' in df.columns) or ('lambda' in df.columns)
        has_dist   = 'dist_atm_pct' in df.columns
        has_spct   = 'spread_pct'   in df.columns
        right_col  = ('right' if 'right' in df.columns
                      else 'cp_sign' if 'cp_sign' in df.columns else None)

        # lambda features (270-272)
        if has_lambda:
            lam_col = 'lambda_ratio' if 'lambda_ratio' in df.columns else 'lambda'
            lam = pd.to_numeric(df[lam_col], errors='coerce').fillna(0.0)
            out[0] = float(lam.mean())
            if 'moneyness' in df.columns:
                mn = pd.to_numeric(df['moneyness'], errors='coerce').fillna(1.0)
                atm_idx = (mn - 1.0).abs().idxmin()
            elif 'delta' in df.columns:
                dl = pd.to_numeric(df['delta'], errors='coerce').abs().fillna(0.0)
                atm_idx = (dl - 0.5).abs().idxmin()
            else:
                atm_idx = df.index[len(df) // 2]
            out[1] = float(lam.loc[atm_idx]) if atm_idx in lam.index else out[0]
            if right_col:
                calls = df[right_col].astype(str).str.upper().isin(['C', 'CALL', '1'])
                puts  = df[right_col].astype(str).str.upper().isin(['P', 'PUT', '-1'])
                c_mean = float(lam[calls].mean()) if calls.any() else 0.0
                p_mean = float(lam[puts].mean())  if puts.any()  else 0.0
                denom  = abs(out[0]) if abs(out[0]) > 1e-6 else 1.0
                out[2] = (c_mean - p_mean) / denom

        # dist_atm features (273-274)
        if has_dist:
            dist = pd.to_numeric(df['dist_atm_pct'], errors='coerce').fillna(0.0).abs()
            out[3] = float(dist.mean())
            if 'volume' in df.columns:
                vol = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0).clip(lower=0)
                total_vol = vol.sum()
                out[4] = float((dist * vol).sum() / total_vol) if total_vol > 0 else out[3]
            else:
                out[4] = out[3]

        # spread_pct features (275-277)
        if has_spct:
            sp = pd.to_numeric(df['spread_pct'], errors='coerce').fillna(0.0).clip(lower=0)
            out[5] = float(sp.mean())
            if 'moneyness' in df.columns:
                mn = pd.to_numeric(df['moneyness'], errors='coerce').fillna(1.0)
                atm_idx2 = (mn - 1.0).abs().idxmin()
                out[6] = float(sp.loc[atm_idx2]) if atm_idx2 in sp.index else out[5]
            else:
                out[6] = out[5]
            if right_col and 'moneyness' in df.columns:
                mn = pd.to_numeric(df['moneyness'], errors='coerce').fillna(1.0)
                calls_otm = df[right_col].astype(str).str.upper().isin(['C','CALL','1']) & (mn > 1.02)
                puts_otm  = df[right_col].astype(str).str.upper().isin(['P','PUT','-1']) & (mn < 0.98)
                c_sp = float(sp[calls_otm].mean()) if calls_otm.any() else out[5]
                p_sp = float(sp[puts_otm].mean())  if puts_otm.any()  else out[5]
                out[7] = p_sp / c_sp if c_sp > 1e-6 else 1.0

        # greek aux features (278-283)
        if 'dual_delta' in df.columns:
            out[8] = float(pd.to_numeric(df['dual_delta'], errors='coerce').fillna(0.0).mean())
        if 'dual_gamma' in df.columns:
            out[9] = float(pd.to_numeric(df['dual_gamma'], errors='coerce').fillna(0.0).mean())

        if 'moneyness' in df.columns:
            mn = pd.to_numeric(df['moneyness'], errors='coerce').fillna(1.0)
            atm_idx3 = (mn - 1.0).abs().idxmin()
        elif 'delta' in df.columns:
            dl = pd.to_numeric(df['delta'], errors='coerce').abs().fillna(0.0)
            atm_idx3 = (dl - 0.5).abs().idxmin()
        else:
            atm_idx3 = df.index[len(df) // 2]

        if 'd1' in df.columns:
            d1 = pd.to_numeric(df['d1'], errors='coerce').fillna(0.0)
            out[10] = float(d1.loc[atm_idx3]) if atm_idx3 in d1.index else float(d1.mean())
        if 'd2' in df.columns:
            d2 = pd.to_numeric(df['d2'], errors='coerce').fillna(0.0)
            out[11] = float(d2.loc[atm_idx3]) if atm_idx3 in d2.index else float(d2.mean())
        if 'iv_error' in df.columns:
            out[12] = float(pd.to_numeric(df['iv_error'], errors='coerce').fillna(0.0).mean())
        if 'ultima' in df.columns:
            out[13] = float(pd.to_numeric(df['ultima'], errors='coerce').fillna(0.0).mean())

        # OI enrichments (284-285)
        oi_col = 'open_interest' if 'open_interest' in df.columns else ('oi' if 'oi' in df.columns else None)
        if oi_col is not None:
            oi = pd.to_numeric(df[oi_col], errors='coerce').fillna(0.0).clip(lower=0.0)
            out[14] = float(oi.mean())
            if right_col:
                calls = df[right_col].astype(str).str.upper().isin(['C', 'CALL', '1'])
                puts = df[right_col].astype(str).str.upper().isin(['P', 'PUT', '-1'])
                c_oi = float(oi[calls].sum()) if calls.any() else 0.0
                p_oi = float(oi[puts].sum()) if puts.any() else 0.0
                out[15] = p_oi / c_oi if c_oi > 1e-6 else 1.0

        return out
