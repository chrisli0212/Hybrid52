python /workspace/csv_to_tier1.py \
  --csv-dir /workspace/historical_data_1yr \
  --out-root /workspace/data/tier1_2026_v1 \
  --symbols SPXW SPY QQQ IWM TLT

# Step 2: Tier1 -> Tier2 (use build_tier2.py, not build_tier2_fast.py)
python "/workspace/Final_production_model/Hybrid51/6. Hybrid51_new stage/scripts/phase0/build_tier2.py" \
  --all-symbols \
  --tier1-root /workspace/data/tier1_2026_v1 \
  --output-root /workspace/data/tier2_minutes_2026_v1 \
  --workers 4

# Step 3: Tier2 -> Tier3 + fresh norms
python "/workspace/Final_production_model/Hybrid51/6. Hybrid51_new stage/scripts/phase0/build_tier3_binary.py" \
  --all-symbols \
  --tier2-root /workspace/data/tier2_minutes_2026_v1 \
  --output-root /workspace/data/tier3_binary_2026_v1 \
  --horizons 5 15 30 \
  --seq-len 20 \
  --return-threshold 0.0003

# Step 4: put 30m norms where production loader expects them
python /workspace/Final_production_model/compute_production_norms.py \
  --source-root /workspace/data/tier3_binary_2026_v1 \
  --output-dir /workspace/data/tier3_binary_v5 \
  --horizon 30 \
  --symbols SPXW SPY QQQ IWM TLT