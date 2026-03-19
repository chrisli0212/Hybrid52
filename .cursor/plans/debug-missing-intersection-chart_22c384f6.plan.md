---
name: debug-missing-intersection-chart
overview: Investigate why the signal divergence chart (intersection of model prediction and rule-based signals) is not appearing in the dashboard and outline steps to restore it.
todos:
  - id: locate-divergence-call
    content: Re‑locate and summarize the block in update_dashboard that calls _create_signal_divergence_chart and appends the figure.
    status: pending
  - id: analyze-divergence-helper
    content: Read _create_signal_divergence_chart and document its input requirements and failure modes.
    status: pending
  - id: trace-data-sources
    content: Trace how df_agg and pred_df are produced, and when they can be empty or misaligned after a reset.
    status: pending
  - id: draft-ux-fixes
    content: List specific code tweaks to make the divergence chart’s appearance and absence clearer and more robust.
    status: pending
isProject: false
---

## Goal

Understand why the **signal divergence / intersection chart** (model prediction vs rule-based, turns purple on intersection) is not rendering between the top panel columns, and define concrete steps to make it show again.

## What we know

- The dashboard layout in `[Final_production_model/theta_dashboard_v4_modern.py](Final_production_model/theta_dashboard_v4_modern.py)` builds a large content list in `update_dashboard`.
- Around the middle of that function, there is a call like `_create_signal_divergence_chart(df_agg, symbol, pred_df)` and then a block that wraps the returned figure in a `dcc.Graph`.
- That block is only executed when:
  - `symbol != 'ALL'`, and
  - The helper returns a non-`None` figure without raising an exception.
- If an exception is raised in `_create_signal_divergence_chart`, the code catches it, logs a warning, and **skips adding the chart**, which could explain why the user sees an empty space.

## Plan

- **Step 1 – Confirm when the chart should render**
  - Re‑inspect `update_dashboard` to confirm the exact conditions (`symbol`, `dte`, and data availability) under which `_create_signal_divergence_chart` is called and the resulting figure is appended.
  - Note where in the layout this chart is inserted relative to the top signal panel and Pipeline Monitor, to confirm the user’s expectation about its position.
- **Step 2 – Inspect `_create_signal_divergence_chart`**
  - Open its definition in `[theta_dashboard_v4_modern.py](Final_production_model/theta_dashboard_v4_modern.py)`.
  - Identify its inputs (`df_agg`, `symbol`, `pred_df`) and what conditions cause it to:
    - Return `None` (no chart), or
    - Raise an exception that gets caught in `update_dashboard`.
  - Look for dependencies on specific columns (e.g. `batch_id`, `signal_rule_based`, `signal_model`, timestamps) or minimum history length that might not be satisfied after a reset.
- **Step 3 – Trace the data feeding the chart**
  - From `update_dashboard`, follow how `df_agg` and `pred_df` are constructed:
    - `df_agg` from `load_data(dte_filter=dte)`.
    - `pred_df` from `_load_prediction_csv()` which reads `daily_data/prediction.csv`.
  - Check for code paths where these could be `None`/empty (e.g. shortly after `DELETE ALL`, before new batches and predictions are written), causing `_create_signal_divergence_chart` to bail out.
- **Step 4 – Identify practical preconditions for the chart**
  - From the implementation, write down simple rules like:
    - “Requires at least N prediction rows for the chosen symbol.”
    - “Requires df_agg with non-empty time series for that symbol and selected DTE.”
    - “Only shows for `symbol != 'ALL'`.”
  - Translate those into user-facing guidance (e.g. "after pressing START, wait until prediction service has produced at least X rows before expecting the purple intersection chart").
- **Step 5 – Propose fixes or safeguards (no code changes yet)**
  - Based on what we find, outline minimal code changes that would make the UX clearer, for example:
    - If data is insufficient, show a small placeholder card explaining why the divergence chart is hidden.
    - Relax overly strict requirements (e.g. reduce required history length) if appropriate.
  - Keep this as an actionable list; we will only implement them after you approve.
- **Step 6 – Verify with your workflow**
  - Map the theoretical preconditions against how you actually run the system:
    - After using `DELETE ALL`, in what order you start fetcher, prediction service, and dashboard.
  - Highlight any mismatch (e.g. prediction.csv empty for a while) that explains why the intersection chart is missing even though the rest of the dashboard is running.

## Todos

- **locate-divergence-call**: Re‑locate and summarize the block in `update_dashboard` that calls `_create_signal_divergence_chart` and appends the figure.
- **analyze-divergence-helper**: Read `_create_signal_divergence_chart` and document its input requirements and failure modes.
- **trace-data-sources**: Trace how `df_agg` and `pred_df` are produced, and when they can be empty or misaligned after a reset.
- **draft-ux-fixes**: List specific code tweaks (placeholders, relaxed conditions, or clearer status text) that would make the chart behavior predictable and understandable for your trading workflow.

