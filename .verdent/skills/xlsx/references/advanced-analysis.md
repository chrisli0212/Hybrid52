# Advanced Spreadsheet Analysis Reference

Supplementary patterns for complex analysis tasks. Load only when SKILL.md guidance is insufficient.

## Table of Contents

- [Multi-Sheet Cross Analysis](#multi-sheet-cross-analysis)
- [Data Quality Audit Pipeline](#data-quality-audit-pipeline)
- [Merge and Join Strategies](#merge-and-join-strategies)
- [Fuzzy Matching](#fuzzy-matching)
- [Window Functions & Rolling Calculations](#window-functions--rolling-calculations)
- [Performance Tuning for Large Files](#performance-tuning-for-large-files)
- [Writing Analysis Results Back to Excel](#writing-analysis-results-back-to-excel)
- [Troubleshooting](#troubleshooting)

## Multi-Sheet Cross Analysis

Combine data from multiple sheets or workbooks for unified analysis.

```python
import pandas as pd

all_sheets = pd.read_excel("workbook.xlsx", sheet_name=None)
combined = pd.concat(all_sheets.values(), keys=all_sheets.keys(), names=["sheet"])
combined = combined.reset_index(level="sheet")

# Cross-file merge
df_a = pd.read_excel("sales.xlsx")
df_b = pd.read_excel("targets.xlsx")
merged = df_a.merge(df_b, on="region", how="left", suffixes=("_actual", "_target"))
merged["variance"] = merged["revenue_actual"] - merged["revenue_target"]
merged["pct_var"] = merged["variance"] / merged["revenue_target"]
```

### Compare Two Versions of a Sheet

```python
old = pd.read_excel("report_v1.xlsx")
new = pd.read_excel("report_v2.xlsx")

# Cell-by-cell diff
diff = old.compare(new, keep_shape=True, keep_equal=False)
print(diff.dropna(how="all"))
```

## Data Quality Audit Pipeline

Run a structured quality check to surface problems before analysis.

```python
def audit(df):
    report = {}
    report["rows"] = len(df)
    report["columns"] = len(df.columns)
    report["duplicated_rows"] = df.duplicated().sum()

    missing = df.isnull().sum()
    report["missing"] = missing[missing > 0].to_dict()

    report["constant_cols"] = [c for c in df.columns if df[c].nunique() <= 1]

    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        q1, q3 = numeric[col].quantile(0.25), numeric[col].quantile(0.75)
        iqr = q3 - q1
        n_out = ((numeric[col] < q1 - 1.5 * iqr) | (numeric[col] > q3 + 1.5 * iqr)).sum()
        if n_out > 0:
            report.setdefault("outliers", {})[col] = int(n_out)

    for col in df.select_dtypes(include="object").columns:
        mixed = df[col].dropna().apply(type).nunique()
        if mixed > 1:
            report.setdefault("mixed_types", []).append(col)

    return report
```

### Automated Checks Checklist

| Check                     | Expression                             |
| ------------------------- | -------------------------------------- |
| Missing values per column | `df.isnull().sum()`                    |
| Duplicate rows            | `df.duplicated().sum()`                |
| Unique counts             | `df.nunique()`                         |
| Negative where unexpected | `(df[col] < 0).sum()`                  |
| Date range validity       | `df["date"].between(start, end).all()` |
| Format consistency        | `df[col].str.match(pattern).all()`     |

## Merge and Join Strategies

```python
# Inner join — only matching keys
merged = left.merge(right, on="key")

# Left join — keep all left rows
merged = left.merge(right, on="key", how="left")

# Multi-key join
merged = left.merge(right, on=["region", "date"])

# Join on different column names
merged = left.merge(right, left_on="emp_id", right_on="employee_id")

# Validate uniqueness
merged = left.merge(right, on="key", validate="one_to_one")

# Indicator column shows match source
merged = left.merge(right, on="key", how="outer", indicator=True)
unmatched = merged[merged["_merge"] != "both"]
```

## Fuzzy Matching

When keys don't align exactly (typos, casing, abbreviations).

```python
from difflib import get_close_matches

def fuzzy_map(source_keys, target_keys, cutoff=0.8):
    mapping = {}
    for key in source_keys:
        matches = get_close_matches(key, target_keys, n=1, cutoff=cutoff)
        if matches:
            mapping[key] = matches[0]
    return mapping

key_map = fuzzy_map(df_a["name"].unique(), df_b["name"].unique())
df_a["matched_name"] = df_a["name"].map(key_map)
merged = df_a.merge(df_b, left_on="matched_name", right_on="name", suffixes=("_a", "_b"))
```

## Window Functions & Rolling Calculations

```python
# Rolling average
df["ma_7d"] = df["value"].rolling(7).mean()

# Expanding cumulative sum
df["cum_total"] = df["value"].expanding().sum()

# Rank within group
df["rank"] = df.groupby("category")["score"].rank(ascending=False)

# Lag / lead
df["prev_value"] = df.groupby("id")["value"].shift(1)
df["change"] = df["value"] - df["prev_value"]

# Percent change
df["pct_change"] = df.groupby("id")["value"].pct_change()
```

## Performance Tuning for Large Files

### Read Optimization

```python
# Column selection — read only what you need
df = pd.read_excel("big.xlsx", usecols=["A", "C", "F"])

# Row limit for sampling
df = pd.read_excel("big.xlsx", nrows=10000)

# openpyxl read-only mode — avoids full parse
from openpyxl import load_workbook
wb = load_workbook("big.xlsx", read_only=True, data_only=True)
ws = wb.active
for row in ws.iter_rows(min_row=1, max_row=100, values_only=True):
    print(row)
wb.close()
```

### Chunked CSV Processing

```python
chunks = pd.read_csv("huge.csv", chunksize=50000)
results = []
for chunk in chunks:
    agg = chunk.groupby("category")["amount"].sum()
    results.append(agg)
final = pd.concat(results).groupby(level=0).sum()
```

### Memory Reduction

```python
# Downcast numeric types
for col in df.select_dtypes(include="integer").columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")
for col in df.select_dtypes(include="float").columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

# Convert low-cardinality strings to category
for col in df.select_dtypes(include="object").columns:
    if df[col].nunique() / len(df) < 0.5:
        df[col] = df[col].astype("category")
```

## Writing Analysis Results Back to Excel

Export DataFrame results with formatting using openpyxl writer.

```python
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

wb = Workbook()
ws = wb.active
ws.title = "Analysis"

for r_idx, row in enumerate(dataframe_to_rows(summary, index=True, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        ws.cell(row=r_idx, column=c_idx, value=val)

# Style header row
for cell in ws[1]:
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="2F5496")
    cell.alignment = Alignment(horizontal="center")

# Auto-fit column widths (approximate)
for col in ws.columns:
    max_len = max(len(str(c.value or "")) for c in col)
    ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

wb.save("analysis_output.xlsx")
```

## Troubleshooting

### read_excel returns empty DataFrame

1. Check `sheet_name` — default reads only the first sheet.
2. Check `header` — if there is no header row, use `header=None`.
3. Check `skiprows` — data may start below row 1.

### Mixed types in a column

pandas reads mixed columns as `object`. Force a type:

```python
df["col"] = pd.to_numeric(df["col"], errors="coerce")  # non-numeric -> NaN
```

### Date parsing failures

```python
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
```

### openpyxl data_only returns None

`data_only=True` reads cached values. If the file was saved without opening in Excel, cached values may be missing. Run `scripts/recalc.py` first.

### Memory issues with large files

- Use `read_only=True` in openpyxl
- Use `usecols` and `nrows` in pandas
- Process CSV in chunks with `chunksize`
- Downcast numeric types

### Formula errors after edit

Run `python scripts/recalc.py file.xlsx` and inspect the JSON output. Common errors:

| Error     | Cause                          | Fix                             |
| --------- | ------------------------------ | ------------------------------- |
| `#REF!`   | Deleted cells still referenced | Update references               |
| `#DIV/0!` | Zero denominator               | Add `IF(denom=0, 0, ...)` guard |
| `#VALUE!` | Type mismatch in formula       | Check cell types                |
| `#NAME?`  | Typo in function name          | Correct spelling                |
| `#N/A`    | VLOOKUP/MATCH miss             | Verify lookup range             |
