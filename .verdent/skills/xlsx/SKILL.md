---
name: xlsx
description: 'Spreadsheet reading, data analysis, visualization, and smart editing toolkit. Use when the agent needs to: (1) Read and inspect .xlsx/.xlsm/.csv/.tsv files — structure, sheets, cell ranges, (2) Profile and analyze data — statistics, distributions, anomalies, correlations, (3) Query and filter data with pandas, (4) Generate charts and visual summaries from spreadsheet data, (5) Create or edit Excel files with formulas and formatting, or (6) Recalculate and verify formula integrity.'
metadata:
  version: '2.0.0'
---

# Spreadsheet Analyst Toolkit

## Reading & Inspection

### Quick Peek — Structure and Shape

Understand what a file contains before diving into analysis.

```python
import openpyxl, pandas as pd

wb = openpyxl.load_workbook("file.xlsx", read_only=True, data_only=True)
for name in wb.sheetnames:
    ws = wb[name]
    print(f"[{name}] rows={ws.max_row} cols={ws.max_column}")
wb.close()

sheets = pd.read_excel("file.xlsx", sheet_name=None, nrows=0)
for name, df in sheets.items():
    print(f"[{name}] columns: {list(df.columns)}")
```

### Load Data

```python
import pandas as pd

df = pd.read_excel("file.xlsx")
df = pd.read_excel("file.xlsx", sheet_name="Sales")
all_sheets = pd.read_excel("file.xlsx", sheet_name=None)

df = pd.read_csv("data.csv")
df = pd.read_csv("data.tsv", sep="\t")

# Large files — limit columns and rows
df = pd.read_excel("big.xlsx", usecols=["A", "B", "D"], nrows=5000)

# Type control
df = pd.read_excel("file.xlsx", dtype={"id": str, "amount": float}, parse_dates=["date"])
```

### Inspect Raw Cells (openpyxl)

```python
from openpyxl import load_workbook

wb = load_workbook("file.xlsx", data_only=True)
ws = wb.active

for row in ws.iter_rows(min_row=1, max_row=5, values_only=False):
    for cell in row:
        print(f"{cell.coordinate}: {cell.value} (type={type(cell.value).__name__})")

# Read formulas (not computed values)
wb_f = load_workbook("file.xlsx", data_only=False)
ws_f = wb_f.active
print(ws_f["B2"].value)  # e.g. "=SUM(B3:B10)"
```

## Data Analysis

### Profiling — Overview of Each Column

```python
df.info()
df.describe(include="all")
df.dtypes
df.isnull().sum()
df.nunique()
```

### Distribution & Outliers

```python
col = "revenue"
print(df[col].quantile([0.01, 0.25, 0.5, 0.75, 0.99]))

iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
lower, upper = df[col].quantile(0.25) - 1.5 * iqr, df[col].quantile(0.75) + 1.5 * iqr
outliers = df[(df[col] < lower) | (df[col] > upper)]
print(f"Outliers: {len(outliers)} / {len(df)}")
```

### Correlation

```python
numeric_cols = df.select_dtypes(include="number")
corr = numeric_cols.corr()
print(corr.to_string())
```

### Grouping & Aggregation

```python
summary = df.groupby("category").agg(
    count=("id", "count"),
    total=("amount", "sum"),
    avg=("amount", "mean"),
).sort_values("total", ascending=False)
```

### Pivot Tables

```python
pivot = df.pivot_table(
    values="amount",
    index="region",
    columns="quarter",
    aggfunc=["sum", "mean"],
    margins=True,
)
```

### Time Series

```python
df["date"] = pd.to_datetime(df["date"])
monthly = df.set_index("date").resample("M")["value"].agg(["sum", "mean", "count"])
yoy = monthly["sum"].pct_change(12)
```

### Deduplication & Cleaning

```python
dupes = df[df.duplicated(subset=["email"], keep=False)]
df_clean = df.drop_duplicates(subset=["email"], keep="last")
df_clean["name"] = df_clean["name"].str.strip().str.title()
```

## Visualization

Generate charts and save to file. Prefer matplotlib for static charts.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
summary.plot.bar(y="total", ax=ax)
ax.set_title("Total by Category")
ax.set_ylabel("Amount")
fig.tight_layout()
fig.savefig("chart.png", dpi=150)
plt.close(fig)
```

Common chart patterns:

```python
# Line — trend
df.plot(x="date", y="revenue", ax=ax)

# Scatter — correlation
df.plot.scatter(x="spend", y="revenue", ax=ax, alpha=0.5)

# Histogram — distribution
df["score"].plot.hist(bins=30, ax=ax)

# Heatmap — correlation matrix
import seaborn as sns
sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)

# Box — outlier overview
df.boxplot(column="amount", by="category", ax=ax)
```

## Creating & Editing Excel Files

### New Workbook

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, numbers

wb = Workbook()
ws = wb.active
ws.title = "Report"

ws["A1"] = "Category"
ws["B1"] = "Amount"
for cell in ws[1]:
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="4472C4")
    cell.alignment = Alignment(horizontal="center")

for i, (cat, amt) in enumerate([("A", 100), ("B", 200)], start=2):
    ws[f"A{i}"] = cat
    ws[f"B{i}"] = amt

ws[f"B{i+1}"] = f"=SUM(B2:B{i})"
ws.column_dimensions["A"].width = 18
ws.column_dimensions["B"].width = 14
ws[f"B2"].number_format = '#,##0'

wb.save("report.xlsx")
```

### Edit Existing File

```python
from openpyxl import load_workbook

wb = load_workbook("existing.xlsx")
ws = wb["Sheet1"]
ws["A1"] = "Updated"
ws.insert_rows(2)
wb.save("modified.xlsx")
```

### Formula Rules

- Use Excel formulas, not Python-computed values
- Place assumptions in dedicated cells with references
- Verify no #REF!, #DIV/0!, #VALUE!, #NAME? after saving

```python
# Correct
ws["C2"] = "=A2*B2"
ws["C10"] = "=SUM(C2:C9)"

# Wrong — hardcoded
total = sum(values)
ws["C10"] = total
```

### Recalculate Formulas

After writing formulas, use the bundled formula engine:

```bash
# Recalculate + audit (default)
python scripts/formula_engine.py output.xlsx

# Audit only — scan errors without recalculating
python scripts/formula_engine.py output.xlsx --audit-only

# Show formula dependency map
python scripts/formula_engine.py output.xlsx --deps

# Custom LibreOffice timeout
python scripts/formula_engine.py output.xlsx --timeout 60
```

Returns per-sheet breakdown:

```json
{
  "status": "clean",
  "sheets": {
    "Sheet1": { "cells": 500, "formulas": 42, "errors": {} }
  },
  "totals": { "formulas": 42, "errors": 0 }
}
```

If `has_errors`, check each sheet's `errors` dict for error types and cell locations, fix source formulas, re-save, and recalculate again.

## Financial Model Standards

Apply these only when building financial models.

### Color Conventions

| Cell Content      | Color                         |
| ----------------- | ----------------------------- |
| Hardcoded inputs  | Blue text (0,0,255)           |
| Formulas          | Black text (0,0,0)            |
| Cross-sheet links | Green text (0,128,0)          |
| External links    | Red text (255,0,0)            |
| Key assumptions   | Yellow background (255,255,0) |

### Number Formats

| Data Type   | Format                                  |
| ----------- | --------------------------------------- |
| Currency    | `$#,##0;($#,##0);"-"` (units in header) |
| Percentage  | `0.0%`                                  |
| Multiples   | `0.0x`                                  |
| Year labels | Text, not number                        |
| Negatives   | Parentheses `(123)` not `-123`          |

## Quick Lookup

| Goal                            | Tool                      | Key API                                      |
| ------------------------------- | ------------------------- | -------------------------------------------- |
| Read data into DataFrame        | pandas                    | `pd.read_excel()`                            |
| Column stats & profiling        | pandas                    | `df.describe()`, `df.info()`                 |
| Group / pivot / aggregate       | pandas                    | `df.groupby()`, `df.pivot_table()`           |
| Inspect raw cells / formulas    | openpyxl                  | `load_workbook(data_only=False)`             |
| Charts                          | matplotlib                | `df.plot()`, `plt.savefig()`                 |
| Heatmaps                        | seaborn                   | `sns.heatmap()`                              |
| Create workbook with formatting | openpyxl                  | `Workbook()` + styles                        |
| Recalculate + audit formulas    | scripts/formula_engine.py | `python scripts/formula_engine.py file.xlsx` |
| Audit formulas only             | scripts/formula_engine.py | `--audit-only`                               |
| Formula dependency map          | scripts/formula_engine.py | `--deps`                                     |
| Large file reading              | pandas                    | `usecols=`, `nrows=`, `read_only=True`       |
| CSV / TSV                       | pandas                    | `pd.read_csv()`                              |

## Advanced

For advanced analysis workflows, specialized techniques, and troubleshooting:

- **Multi-sheet cross analysis**: See [references/advanced-analysis.md](references/advanced-analysis.md)
- **Data quality audit pipeline**: See [references/advanced-analysis.md](references/advanced-analysis.md)
- **Merge and join strategies**: See [references/advanced-analysis.md](references/advanced-analysis.md)
- **Performance tuning for large files**: See [references/advanced-analysis.md](references/advanced-analysis.md)
- **Troubleshooting common issues**: See [references/advanced-analysis.md](references/advanced-analysis.md)
