# Hybrid52 — Cursor AI Context Note
## What Was Done in This Session (2026-03-24)

---

## Background

The project is **Hybrid52** — a PyTorch ML ensemble for intraday directional prediction on SPXW/SPX options.
It runs 7+ specialist neural agents, each processing a different slice of Theta Data option chain features, whose outputs are combined by an ensemble voter into a final UP/DOWN probability.

The predecessor was **Hybrid51**, which was trained on a **158-dimensional feature vector** built from Theta Data option chain snapshots. The problem discovered: ~40–125 of those 158 dimensions were **dead** — always zero or constant (e.g. `open`, `high`, `low`, `close`, `count`, `volume`, `vwap`, `bid_exchange`, `ask_exchange`) because Theta Data does not populate these fields in EOD chain snapshots. Agents were feeding pure noise into their weights on every forward pass.

The repo lives at:
```
/workspace/ Hybrid52_New training/
├── hybrid52_models/agents/        ← agent source files
├── hybrid52_preprocessing/        ← feature config + extractors
└── data/tier3_binary_v5/          ← training data
```
Note: there is a **leading space** in the folder name `" Hybrid52_New training"`. All paths must include it.

---

## What Was Amended

### 1. `agent_2d.py` — Silent synthetic fallback removed
**Before:** If `chain_2d=None`, the agent silently trained on Gaussian noise (synthetic data).  
**After:** Patched to raise `RuntimeWarning` immediately when `chain_2d=None`, so bad data never reaches training silently.  
**Status:** Patched and pushed. Real `chain_2d.npy` files still need to be built from raw Theta Data — this is the **next blocker**.

---

### 2. `agent_a.py` — Architecture improvements + input_dim fix
**Before:** `input_dim=158` (included ~40 zero/constant columns). Pool was `AdaptiveMaxPool1d(1)` → 48-dim. No temporal gate. `fusion_dim=272`.  
**After:**
- `input_dim=53` (only real populated Theta Data features)
- `AdaptiveMaxPool1d` replaced with `max_pool + avg_pool` concatenated → 96-dim (`cnn_out`)
- Learned temporal gate: `Linear(96→1) + Sigmoid` scales `cnn_out` before fusion
- `fusion_dim` updated to `96 + temporal_dim + 96 = 320`
- Params: ~219k (was ~195k)

**Forward pass verified:** `score=(4,1), conf=(4,1), signal=(4,5)` ✅

---

### 3. `agent_t.py` — Recency-weighted pool replaces AdaptiveMaxPool1d
**Problem:** `AdaptiveMaxPool1d(1)` only picks the strongest spike in the sequence. When markets trend up (low VIX), every spike is bullish → agent becomes a degenerate UP-only predictor. F1 collapsed to **0.016** at confidence ≥0.6 threshold.  
**Fix:** Replaced with dynamically computed recency-weighted average pool:
```python
w = torch.linspace(0.5, 1.5, T, device=cnn_out.device).view(1, 1, T)
w = w / w.sum()
flow_temporal = (cnn_out * w).sum(dim=2)
```
Weights are computed per-forward-pass for any sequence length T — no fixed buffer needed.  
**Verified:** `score=(4,1), conf=(4,1), params=71,031` ✅

---

### 4. `agent_q.py` — Gated residual encoder + LSTM dropout warning fix
**Problem:** Classifier only had 34-dim input (vs Agent A's 130-dim), due to LSTM bottleneck. F1 collapsed at high confidence (UP-only bias).  
**Fix 1:** Added `quote_residual = Linear(quote_feat_dim, 96)` and a learned gate that blends the encoder output with the raw residual:
```python
enc = self.quote_encoder(static)
res = self.quote_residual(static)
gate = self.encoder_gate(enc)
quote_encoded = gate * enc + (1 - gate) * res
```
**Fix 2:** Removed `dropout=0.1` from single-layer BiLSTM (PyTorch silently ignores it but logs a warning on every forward pass).  
**Verified:** `score=(4,1), conf=(4,1), params=87,444` ✅

---

### 5. `agent_c.py` — Full embedding + backbone gate fix
**Problem 1:** `self.embedding = nn.Linear(min(input_dim, 32), embed_dim)` — truncated input to 32 dims, discarding 126 of 158 features.  
**Problem 2:** `use_backbone: True` means a TemporalBackbone already compressed the signal into 128-dim before Agent C's own CNN+Attention ran on the same input — double extraction, attention learned nothing extra.  
**Problem 3:** `backbone_gate` was slicing `pooled[:, :embed_dim]` (only first 96 of 192-dim BiLSTM output) before the gate decision.  
**Fix:**
- `embedding` now takes full `input_dim` (not truncated to 32)
- `backbone_gate` input changed from `embed_dim=96` → `192` (full BiLSTM output)
- Gate now suppresses attention output when backbone has already captured the signal

**Verified:** `score=(4,1), conf=(4,1), params=267,952` ✅

---

### 6. `agent_b.py` — BiLSTM rewrite: input_dim 158→34 + new architecture
**Before:** `input_dim=158` (massive dead weight). Single-stage BiLSTM. `static` tensor passed as `seq` — shape bug. No temporal fusion.  
**After:** Major rewrite:
- `input_dim=34` (19 raw per-contract features + 15 cross-strike aggregations)
- Separate `static_dim=53` parameter (static and seq are different tensors with different dims)
- Stacked BiLSTM (2 layers) with LayerNorm between layers
- Attention pooling with exponential recency bias (recent bars weighted higher)
- Momentum deltas: 1-bar diff appended to seq → seq becomes `(B, T, 68)`
- Time-of-day embedding: `Embedding(390, 8)` per bar → seq becomes `(B, T, 76)`
- Parallel TCN branch: 3-layer dilated conv (dilation 1→2→4) + global avg pool → `(B, 64)`
- Wider fusion: `hidden(128) + TCN(64) + temporal(128) = 320 → 96 → 48`
- LSTM dropout warning fixed

**Verified:** `score=(4,1), conf=(4,1)` ✅ (with and without temporal)

---

### 7. `hybrid52_preprocessing/feature_config_agent_a.py` — New file
Documents the 53 real features used by Agent A, grouped into 8 categories:
- `atm_greeks` (9): delta, gamma, vega, theta, vanna, charm, iv, spread, spread_pct
- `gex_vanna_charm` (9): GEX call/put/net, GEX flip distance, vanna/charm exposures
- `oi_structure` (10): OI totals, P/C ratio, call/put walls, wall distances
- `iv_surface` (7): ATM IV, 25d skew, 10d put, term slope, smile curvature
- `liquidity` (3): mean spread %, spread std, % liquid strikes
- `quote_imbalance` (5): bid/ask size totals, quote imbalance call/put
- `delta_bucketed_oi` (6): OI by moneyness bucket, net delta exposure
- `dte_structure` (4): min DTE, weighted avg DTE, frac 0DTE, frac 7d

Explicitly documents **what to drop** from raw Theta Data CSVs:
```python
THETA_HIST_DROP = ['open', 'high', 'low', 'close', 'count', 'volume', 'vwap',
                   'bid_exchange', 'ask_exchange']
```

---

### 8. `hybrid52_preprocessing/extract_agent_a_features.py` — New file
Full feature extractor function `extract_agent_a_snapshot(hist_df, oi_df, underlying_price)` that:
- Drops dead columns from raw Theta CSVs
- Clips `lambda` to `[-500, 500]`
- Joins OI data on `strike + right`
- Computes all 53 features including GEX, vanna/charm exposures, IV surface, OI walls

---

## Missing Data — Impact on Prediction

Always-zero structural features (`speed`, `vera`, `ext_condition1-4`) are **already excluded** and have zero impact. The real risk is **silent pipeline failures** making normally real features appear as zeros at inference time. Protection mechanisms already in place:
- `feature_completeness < 0.01` → prediction suppressed
- `warmup_fraction < 0.35` → prediction suppressed
- `_chain_has_real_data()` guard → skips Agent 2D if chain tensor is all zeros

---

## What Still Needs to Be Done (Retrain Blockers)

1. **Build real `chain_2d.npy` files** — Agent 2D cannot train without them. Raw Theta Data option chain snapshots must be located on RunPod (`find /workspace -name "*.parquet"`) and converted using the chain_2d build pipeline.
2. **Retrain all 4 patched agents** — weight shapes changed so old `.pt` checkpoints are incompatible:
   ```bash
   cd "/workspace/ Hybrid52_New training/"
   for agent in a b c t q; do
       python3 train_single_agent.py --agent ${agent} --symbol SPXW --horizon 30min
   done
   ```
3. **Verify Agent T confidence buckets after retraining** — the primary success metric is F1 at confidence ≥0.6 recovering from `0.016`. If the recency-weighted pool fix worked, F1 should be `>0.4`.
4. **Apply same `input_dim` fix to remaining agents** — Agent C, K, VIX still use `input_dim=158` with dead columns. Same zero-column audit needs to be applied.
5. **Remove `.bak` files from repo** — `git rm hybrid52_models/agents/*.bak`

---

*Session date: 2026-03-24 | Written for Cursor AI context*
