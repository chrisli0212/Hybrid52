# Pipeline Data Audit (Smoke15)

- Symbol: `SPXW`
- Overall status: **FAIL**

## fetch_output: FAIL
- Notes: No raw fetch artifacts found in source symbol directory; fetch->duckdb handoff cannot be independently verified from this workspace snapshot.

| Check | Value | Pass |
|---|---:|:---:|
| `raw_fetch_files_present` | `0` | N |

## duckdb_tables: PASS

| Check | Value | Pass |
|---|---:|:---:|
| `greeks_all.duckdb_exists` | `/workspace/data/theta_data_3year/SPXW/greeks_all.duckdb` | Y |
| `greeks_all.duckdb:greeks_all_table_exists` | `True` | Y |
| `greeks_all.duckdb:greeks_all_row_count_gt0` | `153510120` | Y |
| `trade_quote.duckdb_exists` | `/workspace/data/theta_data_3year/SPXW/trade_quote.duckdb` | Y |
| `trade_quote.duckdb:trades_table_exists` | `True` | Y |
| `trade_quote.duckdb:trades_row_count_gt0` | `198799486` | Y |
| `trade_quote.duckdb:quotes_table_exists` | `True` | Y |
| `trade_quote.duckdb:quotes_row_count_gt0` | `51473633` | Y |
| `ohlcv.duckdb_exists` | `/workspace/data/theta_data_3year/SPXW/ohlcv.duckdb` | Y |
| `ohlcv.duckdb:ohlcv_table_exists` | `True` | Y |
| `ohlcv.duckdb:ohlcv_row_count_gt0` | `148848226` | Y |

## tier1_parquet: PASS

| Check | Value | Pass |
|---|---:|:---:|
| `tier1_greek_file_count` | `15` | Y |
| `tier1_tq_file_count` | `15` | Y |
| `tier1_ohlc_file_count` | `15` | Y |
| `tier1_common_dates_count` | `15` | Y |
| `2024-09-25_greek.parquet_rows_gt0` | `36788` | Y |
| `2024-09-26_greek.parquet_rows_gt0` | `20938` | Y |
| `2024-09-27_greek.parquet_rows_gt0` | `22053` | Y |
| `2024-09-30_greek.parquet_rows_gt0` | `69354` | Y |
| `2024-10-01_greek.parquet_rows_gt0` | `95663` | Y |
| `2024-10-02_greek.parquet_rows_gt0` | `57603` | Y |
| `2024-10-03_greek.parquet_rows_gt0` | `33699` | Y |
| `2024-10-04_greek.parquet_rows_gt0` | `29484` | Y |
| `2024-10-07_greek.parquet_rows_gt0` | `80120` | Y |
| `2024-10-08_greek.parquet_rows_gt0` | `79555` | Y |
| `2024-10-09_greek.parquet_rows_gt0` | `52936` | Y |
| `2024-10-10_greek.parquet_rows_gt0` | `24426` | Y |
| `2024-10-11_greek.parquet_rows_gt0` | `22838` | Y |
| `2024-10-14_greek.parquet_rows_gt0` | `57208` | Y |
| `2024-10-15_greek.parquet_rows_gt0` | `67080` | Y |
| `2024-09-25_tq.parquet_rows_gt0` | `525529` | Y |
| `2024-09-26_tq.parquet_rows_gt0` | `537095` | Y |
| `2024-09-27_tq.parquet_rows_gt0` | `562613` | Y |
| `2024-09-30_tq.parquet_rows_gt0` | `638899` | Y |
| `2024-10-01_tq.parquet_rows_gt0` | `768003` | Y |
| `2024-10-02_tq.parquet_rows_gt0` | `588670` | Y |
| `2024-10-03_tq.parquet_rows_gt0` | `585731` | Y |
| `2024-10-04_tq.parquet_rows_gt0` | `621640` | Y |
| `2024-10-07_tq.parquet_rows_gt0` | `671576` | Y |
| `2024-10-08_tq.parquet_rows_gt0` | `639989` | Y |
| `2024-10-09_tq.parquet_rows_gt0` | `632388` | Y |
| `2024-10-10_tq.parquet_rows_gt0` | `583549` | Y |
| `2024-10-11_tq.parquet_rows_gt0` | `563288` | Y |
| `2024-10-14_tq.parquet_rows_gt0` | `566438` | Y |
| `2024-10-15_tq.parquet_rows_gt0` | `773860` | Y |
| `2024-09-25_ohlc.parquet_rows_gt0` | `296769` | Y |
| `2024-09-26_ohlc.parquet_rows_gt0` | `205275` | Y |
| `2024-09-27_ohlc.parquet_rows_gt0` | `283866` | Y |
| `2024-09-30_ohlc.parquet_rows_gt0` | `403512` | Y |
| `2024-10-01_ohlc.parquet_rows_gt0` | `412896` | Y |
| `2024-10-02_ohlc.parquet_rows_gt0` | `310063` | Y |
| `2024-10-03_ohlc.parquet_rows_gt0` | `216223` | Y |
| `2024-10-04_ohlc.parquet_rows_gt0` | `227171` | Y |
| `2024-10-07_ohlc.parquet_rows_gt0` | `369495` | Y |
| `2024-10-08_ohlc.parquet_rows_gt0` | `369886` | Y |
| `2024-10-09_ohlc.parquet_rows_gt0` | `310454` | Y |
| `2024-10-10_ohlc.parquet_rows_gt0` | `208794` | Y |
| `2024-10-11_ohlc.parquet_rows_gt0` | `226389` | Y |
| `2024-10-14_ohlc.parquet_rows_gt0` | `389436` | Y |
| `2024-10-15_ohlc.parquet_rows_gt0` | `428536` | Y |

## tier2_nonzero_map: PASS

| Check | Value | Pass |
|---|---:|:---:|
| `tier2_minutes_file_exists` | `/workspace/data/tier2_hybrid55_smoke15/SPXW_minutes.parquet` | Y |
| `tier2_feature_dim_311` | `311` | Y |
| `tier2_no_nan` | `False` | Y |
| `tier2_live_features_gt0` | `169` | Y |
| `tier2_live_feature_rate_ge_0.30` | `0.5434` | Y |
| `tier2_ohlc_live_features_gt0` | `11` | Y |
| `tier2_tq_live_features_gt0` | `62` | Y |

## tier3_live_mask_continuity: PASS

| Check | Value | Pass |
|---|---:|:---:|
| `tier3_dir_exists` | `/workspace/data/tier3_hybrid55_smoke15_nostrip/SPXW/horizon_30min` | Y |
| `tier3_zero_mask_dim_311` | `311` | Y |
| `tier3_live_indices_consistent_with_mask` | `1` | Y |
| `tier2_tier3_live_jaccard_ge_0.95` | `1.0` | Y |
| `tier3_live_indices_greek_exists` | `True` | Y |
| `tier3_greek_indices_subset_of_live` | `True` | Y |
| `tier3_live_indices_tq_exists` | `True` | Y |
| `tier3_tq_indices_subset_of_live` | `True` | Y |
| `tier3_live_indices_ohlc_exists` | `True` | Y |
| `tier3_ohlc_indices_subset_of_live` | `True` | Y |
| `tier3_chain_norm_mean_exists` | `True` | Y |
| `tier3_chain_norm_std_exists` | `True` | Y |
