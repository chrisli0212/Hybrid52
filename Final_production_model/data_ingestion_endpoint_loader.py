from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None


class EndpointBatchLoader:
    """
    Load latest endpoint batch from theta_archive using DuckDB.

    Endpoints remain logically separate:
      - greeks stream
      - quote stream
      - trade stream
      - ohlc stream
      - oi stream
    """

    # Written each fetch batch by theta_fetching_v5.py; prediction reads these first (live batch, not archive-only).
    MODEL_GREEKS_NAME = "theta_model_greeks.csv"
    MODEL_TRADE_QUOTE_NAME = "theta_model_trade_quote.csv"

    def __init__(self, data_dir: Path, max_files_per_endpoint: int = 8):
        self.data_dir = Path(data_dir)
        self.archive_dir = self.data_dir / "theta_archive"
        self.max_files_per_endpoint = max_files_per_endpoint
        self._conn = duckdb.connect(database=":memory:") if duckdb is not None else None

    def _latest_files(self, endpoint: str) -> List[Path]:
        if not self.archive_dir.exists():
            return []
        files = sorted(
            self.archive_dir.glob(f"theta_{endpoint}_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files[: self.max_files_per_endpoint]

    def _read_latest_batch(self, endpoint: str) -> pd.DataFrame:
        files = self._latest_files(endpoint)
        if not files:
            return pd.DataFrame()
        if self._conn is None:
            parts = [pd.read_csv(p, on_bad_lines="skip") for p in files]
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            if df.empty or "batch_id" not in df.columns:
                return pd.DataFrame()
            b = pd.to_numeric(df["batch_id"], errors="coerce").dropna()
            if b.empty:
                return pd.DataFrame()
            latest = int(b.max())
            return df[pd.to_numeric(df["batch_id"], errors="coerce") == latest].copy()

        q_files = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
        base_sql = (
            "SELECT * FROM read_csv_auto(["
            + q_files
            + "], union_by_name=true, ignore_errors=true, sample_size=-1)"
        )
        max_batch_row = self._conn.execute(
            f"SELECT max(CAST(batch_id AS BIGINT)) AS b FROM ({base_sql}) t WHERE batch_id IS NOT NULL"
        ).fetchone()
        if not max_batch_row or max_batch_row[0] is None:
            return pd.DataFrame()
        latest_batch = int(max_batch_row[0])
        return self._conn.execute(
            f"SELECT * FROM ({base_sql}) t WHERE CAST(batch_id AS BIGINT) = {latest_batch}"
        ).fetchdf()

    @staticmethod
    def _ensure_symbol(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "symbol" in out.columns:
            out["symbol"] = out["symbol"].astype(str)
        return out

    def _read_csv_safe(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _try_load_model_rolling_files(
        self,
    ) -> Optional[Tuple[Optional[int], Dict[str, pd.DataFrame], pd.DataFrame]]:
        gp = self.data_dir / self.MODEL_GREEKS_NAME
        greeks = self._ensure_symbol(self._read_csv_safe(gp))
        if greeks.empty or "batch_id" not in greeks.columns:
            return None
        tqp = self.data_dir / self.MODEL_TRADE_QUOTE_NAME
        trade_quote = self._ensure_symbol(self._read_csv_safe(tqp))
        quotes = pd.DataFrame()
        trades = pd.DataFrame()
        ohlc = self._ensure_symbol(self._read_latest_batch("ohlc"))
        oi = self._ensure_symbol(self._read_latest_batch("oi"))
        return self._bundle_endpoint_frames(greeks, quotes, trades, trade_quote, ohlc, oi)

    def _bundle_endpoint_frames(
        self,
        greeks: pd.DataFrame,
        quotes: pd.DataFrame,
        trades: pd.DataFrame,
        trade_quote: pd.DataFrame,
        ohlc: pd.DataFrame,
        oi: pd.DataFrame,
    ) -> Tuple[Optional[int], Dict[str, pd.DataFrame], pd.DataFrame]:
        batch_candidates: List[int] = []
        for df in (greeks, quotes, trades, ohlc, oi):
            if df is not None and not df.empty and "batch_id" in df.columns:
                vals = pd.to_numeric(df["batch_id"], errors="coerce").dropna()
                if not vals.empty:
                    batch_candidates.append(int(vals.max()))
        if trade_quote is not None and not trade_quote.empty and "batch_id" in trade_quote.columns:
            vals = pd.to_numeric(trade_quote["batch_id"], errors="coerce").dropna()
            if not vals.empty:
                batch_candidates.append(int(vals.max()))

        batch_id: Optional[int] = max(batch_candidates) if batch_candidates else None

        if trade_quote is None or trade_quote.empty:
            trade_quote = pd.concat([quotes, trades], ignore_index=True, sort=False)
        trade_quote = self._ensure_symbol(trade_quote)

        agg_rows = []
        if not greeks.empty:
            now_ts = str(greeks["ts"].iloc[-1]) if "ts" in greeks.columns and not greeks["ts"].empty else ""
            for sym, sdf in greeks.groupby("symbol"):
                spot_col = "underlying_price" if "underlying_price" in sdf.columns else ("spot" if "spot" in sdf.columns else None)
                spot = float(pd.to_numeric(sdf[spot_col], errors="coerce").dropna().iloc[-1]) if spot_col and not pd.to_numeric(sdf[spot_col], errors="coerce").dropna().empty else 0.0
                iv_col = "implied_vol" if "implied_vol" in sdf.columns else None
                right_col = "right" if "right" in sdf.columns else None
                call_iv = None
                put_iv = None
                if iv_col and right_col:
                    rr = sdf[right_col].astype(str).str.upper()
                    c = pd.to_numeric(sdf.loc[rr.str.startswith("C"), iv_col], errors="coerce")
                    p = pd.to_numeric(sdf.loc[rr.str.startswith("P"), iv_col], errors="coerce")
                    call_iv = float(c.mean()) if not c.empty else None
                    put_iv = float(p.mean()) if not p.empty else None
                agg_rows.append(
                    {
                        "batch_id": batch_id if batch_id is not None else 0,
                        "ts": now_ts,
                        "dte_group": "all",
                        "symbol": sym,
                        "spot": spot,
                        "call_vol": None,
                        "put_vol": None,
                        "pc_ratio": None,
                        "call_iv": call_iv,
                        "put_iv": put_iv,
                        "iv_skew": (put_iv - call_iv) if (put_iv is not None and call_iv is not None) else None,
                    }
                )
        agg_df = pd.DataFrame(agg_rows)

        return batch_id, {
            "greeks": greeks,
            "quotes": quotes,
            "trades": trades,
            "trade_quote": trade_quote,
            "ohlc": ohlc,
            "oi": oi,
        }, agg_df

    def load_latest(self) -> Tuple[Optional[int], Dict[str, pd.DataFrame], pd.DataFrame]:
        rolled = self._try_load_model_rolling_files()
        if rolled is not None:
            return rolled

        greeks = self._ensure_symbol(self._read_latest_batch("greeks"))
        quotes = self._ensure_symbol(self._read_latest_batch("quotes"))
        trades = self._ensure_symbol(self._read_latest_batch("trades"))
        ohlc = self._ensure_symbol(self._read_latest_batch("ohlc"))
        oi = self._ensure_symbol(self._read_latest_batch("oi"))
        trade_quote = pd.concat([quotes, trades], ignore_index=True, sort=False)
        return self._bundle_endpoint_frames(greeks, quotes, trades, trade_quote, ohlc, oi)
