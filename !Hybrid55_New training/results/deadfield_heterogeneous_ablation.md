# Dead-Field + Heterogeneous Ablation (Structural)

- Feature schema: `hybrid55_v1_live_raw_guarded_311`
- Total flat features: `311`
- Dead raw tier1 fields locked: `['batch_id', 'dual_gamma', 'endpoint', 'iv_error', 'speed', 'ts', 'vera', 'zomma']`
- Derived feature names matching dead tokens: `0`

## Agent Summary

| Agent | Subset Dim | Backbone | Total Params | Backbone Params | Dead-name overlap |
|---|---:|:---:|---:|---:|---:|
| A | 130 | True | 586657 | 365056 | 0 |
| B | 75 | False | 969918 | 0 | 0 |
| C | 69 | True | 685678 | 357248 | 0 |
| K | 75 | False | 239550 | 0 | 0 |
| TQ | 70 | False | 172022 | 0 | 0 |
| H | 11 | False | 91006 | 0 | 0 |
| M | 146 | True | 676328 | 502488 | 0 |
| 2D | 0 | False | 670793 | 0 | 0 |

## Heterogeneous Mechanism Check

- Synthetic logit correlation between `Agent A` and `Agent M`: `-0.0259` (lower usually means more architectural diversity).

- JSON details: `/workspace/!Hybrid55_New training/results/deadfield_heterogeneous_ablation.json`
