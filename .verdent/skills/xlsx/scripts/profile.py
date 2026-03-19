#!/usr/bin/env python3
"""Quick data profiling for spreadsheet files. Outputs a JSON summary."""

import json, sys
from pathlib import Path

import pandas as pd


def profile(filepath, sheet=None, max_rows=None):
    ext = Path(filepath).suffix.lower()
    kw = {}
    if max_rows:
        kw["nrows"] = max_rows

    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(filepath, sep=sep, **kw)
        sheets = {"data": df}
    else:
        target = sheet if sheet else None
        raw = pd.read_excel(filepath, sheet_name=target or None, **kw)
        sheets = raw if isinstance(raw, dict) else {target or "Sheet1": raw}

    result = {}
    for name, df in sheets.items():
        cols = []
        for c in df.columns:
            info = {
                "name": str(c),
                "dtype": str(df[c].dtype),
                "non_null": int(df[c].count()),
                "null": int(df[c].isnull().sum()),
                "unique": int(df[c].nunique()),
            }
            if pd.api.types.is_numeric_dtype(df[c]):
                desc = df[c].describe()
                info["min"] = float(desc.get("min", 0))
                info["max"] = float(desc.get("max", 0))
                info["mean"] = round(float(desc.get("mean", 0)), 4)
                info["std"] = round(float(desc.get("std", 0)), 4)
            elif pd.api.types.is_string_dtype(df[c]):
                top = df[c].value_counts().head(5)
                info["top_values"] = {str(k): int(v) for k, v in top.items()}
            cols.append(info)
        result[name] = {"rows": len(df), "columns": len(df.columns), "column_profiles": cols}

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile.py <file> [sheet_name] [max_rows]")
        sys.exit(1)
    fp = sys.argv[1]
    sh = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else None
    mr = None
    for a in sys.argv[2:]:
        if a.isdigit():
            mr = int(a)
            break
    print(json.dumps(profile(fp, sh, mr), indent=2, default=str))
