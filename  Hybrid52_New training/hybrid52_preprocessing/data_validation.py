"""
Hybrid51 Data Validation & Column Filtering
Task 1: Identify and exclude zero/constant columns from R2 data.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class ColumnStatus(Enum):
    ACTIVE = "active"
    ALWAYS_ZERO = "always_zero"
    CONSTANT = "constant"
    HIGH_MISSING = "high_missing"
    METADATA = "metadata"


@dataclass
class ColumnValidation:
    name: str
    status: ColumnStatus
    reason: str
    dtype: str = "float64"
    group: str = "unknown"
    zero_pct: float = 0.0
    missing_pct: float = 0.0
    unique_count: int = 0


@dataclass
class ValidationReport:
    total_columns: int = 0
    usable_columns: int = 0
    excluded_columns: int = 0
    columns: List[ColumnValidation] = field(default_factory=list)
    excluded_names: List[str] = field(default_factory=list)
    usable_names: List[str] = field(default_factory=list)


KNOWN_ZERO_COLUMNS = {
    'speed', 'vera',
    'rho', 'epsilon',
    'vomma', 'veta', 'zomma', 'color',
    'ultima',
    'dual_delta', 'dual_gamma',
    'd1', 'd2',
    'iv_error',
}

METADATA_COLUMNS = {
    'symbol', 'expiration', 'strike', 'right', 
    'timestamp', 'trade_date', 'underlying_timestamp'
}

GREEK_COLUMN_GROUPS = {
    'greeks_1st': ['delta', 'gamma', 'vega', 'theta', 'rho', 'epsilon', 'lambda'],
    'greeks_2nd': ['vanna', 'charm', 'vomma', 'veta', 'zomma', 'color'],
    'greeks_3rd': ['speed', 'ultima'],
    'greeks_other': ['vera', 'dual_delta', 'dual_gamma'],
    'pricing': ['bid', 'ask', 'underlying_price'],
    'volatility': ['implied_vol', 'iv_error'],
    'technical': ['d1', 'd2'],
}


def get_column_group(col_name: str) -> str:
    for group, columns in GREEK_COLUMN_GROUPS.items():
        if col_name in columns:
            return group
    if col_name in METADATA_COLUMNS:
        return 'metadata'
    return 'unknown'


def validate_greek_columns() -> ValidationReport:
    report = ValidationReport()
    
    all_greek_columns = [
        ('symbol', 'str', 'metadata'),
        ('expiration', 'int64', 'metadata'),
        ('strike', 'float64', 'metadata'),
        ('right', 'str', 'metadata'),
        ('timestamp', 'str', 'metadata'),
        ('trade_date', 'str', 'metadata'),
        ('underlying_timestamp', 'str', 'metadata'),
        ('bid', 'float64', 'pricing'),
        ('ask', 'float64', 'pricing'),
        ('underlying_price', 'float64', 'pricing'),
        ('delta', 'float64', 'greeks_1st'),
        ('gamma', 'float64', 'greeks_1st'),
        ('vega', 'float64', 'greeks_1st'),
        ('theta', 'float64', 'greeks_1st'),
        ('rho', 'float64', 'greeks_1st'),
        ('epsilon', 'float64', 'greeks_1st'),
        ('lambda', 'float64', 'greeks_1st'),
        ('vanna', 'float64', 'greeks_2nd'),
        ('charm', 'float64', 'greeks_2nd'),
        ('vomma', 'float64', 'greeks_2nd'),
        ('veta', 'float64', 'greeks_2nd'),
        ('zomma', 'float64', 'greeks_2nd'),
        ('color', 'float64', 'greeks_2nd'),
        ('speed', 'float64', 'greeks_3rd'),
        ('ultima', 'float64', 'greeks_3rd'),
        ('vera', 'float64', 'greeks_other'),
        ('dual_delta', 'float64', 'greeks_other'),
        ('dual_gamma', 'float64', 'greeks_other'),
        ('d1', 'float64', 'technical'),
        ('d2', 'float64', 'technical'),
        ('implied_vol', 'float64', 'volatility'),
        ('iv_error', 'float64', 'volatility'),
    ]
    
    report.total_columns = len(all_greek_columns)
    
    for col_name, dtype, group in all_greek_columns:
        if col_name in KNOWN_ZERO_COLUMNS:
            validation = ColumnValidation(
                name=col_name,
                status=ColumnStatus.ALWAYS_ZERO,
                reason=f"Always 0.0 across all data samples - EXCLUDE",
                dtype=dtype,
                group=group,
                zero_pct=100.0
            )
            report.excluded_names.append(col_name)
            report.excluded_columns += 1
        elif col_name in METADATA_COLUMNS:
            validation = ColumnValidation(
                name=col_name,
                status=ColumnStatus.METADATA,
                reason="Metadata column - used for joining, not as feature",
                dtype=dtype,
                group=group
            )
        else:
            validation = ColumnValidation(
                name=col_name,
                status=ColumnStatus.ACTIVE,
                reason="Active feature column",
                dtype=dtype,
                group=group
            )
            report.usable_names.append(col_name)
            report.usable_columns += 1
        
        report.columns.append(validation)
    
    return report


def get_usable_greek_columns() -> List[str]:
    return [
        'delta', 'gamma', 'vega', 'theta', 'lambda',
        'vanna', 'charm',
        'implied_vol',
        'bid', 'ask', 'underlying_price',
        'open_interest', 'moneyness', 'dist_atm_pct',
        'mid', 'spread', 'spread_pct', 'lambda_ratio',
        'dte_int', 'cp_sign',
    ]


def get_excluded_columns() -> List[str]:
    return list(KNOWN_ZERO_COLUMNS)


def get_metadata_columns() -> List[str]:
    return list(METADATA_COLUMNS)


def get_feature_columns_by_group() -> Dict[str, List[str]]:
    return {
        'greeks_1st_order': ['delta', 'gamma', 'vega', 'theta', 'rho', 'epsilon', 'lambda'],
        'greeks_2nd_order': ['vanna', 'charm', 'vomma', 'veta', 'zomma', 'color'],
        'greeks_3rd_order': ['ultima'],
        'greeks_other': ['dual_delta', 'dual_gamma'],
        'pricing': ['bid', 'ask', 'underlying_price'],
        'volatility': ['implied_vol', 'iv_error'],
        'technical': ['d1', 'd2'],
    }


TRADE_QUOTE_ALWAYS_CONSTANT = {
    'ext_condition1': 255,
    'ext_condition2': 255,
    'ext_condition3': 255,
    'ext_condition4': 255,
    'bid_condition': 50,
    'ask_condition': 50,
}

TRADE_QUOTE_COLUMNS = {
    'metadata': ['symbol', 'expiration', 'strike', 'right', 'trade_date'],
    'timestamps': ['trade_timestamp', 'quote_timestamp'],
    'trade': ['sequence', 'condition', 'size', 'exchange', 'price'],
    'quote': ['bid_size', 'bid_exchange', 'bid', 'ask_size', 'ask_exchange', 'ask'],
}


def get_trade_quote_feature_columns() -> List[str]:
    """Get usable trade/quote columns (excluding constants)."""
    return (
        TRADE_QUOTE_COLUMNS['trade'] + 
        TRADE_QUOTE_COLUMNS['quote']
    )


def get_trade_quote_excluded_columns() -> List[str]:
    """Get trade/quote columns that are always constant."""
    return list(TRADE_QUOTE_ALWAYS_CONSTANT.keys())


def save_validation_report(report: ValidationReport, output_path: Path) -> None:
    report_dict = {
        'total_columns': report.total_columns,
        'usable_columns': report.usable_columns,
        'excluded_columns': report.excluded_columns,
        'excluded_names': report.excluded_names,
        'usable_names': report.usable_names,
        'columns': [
            {
                'name': col.name,
                'status': col.status.value,
                'reason': col.reason,
                'dtype': col.dtype,
                'group': col.group,
                'zero_pct': col.zero_pct,
                'missing_pct': col.missing_pct,
            }
            for col in report.columns
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)


def save_usable_columns(output_path: Path) -> None:
    usable = {
        'greek_columns': get_usable_greek_columns(),
        'excluded_columns': get_excluded_columns(),
        'metadata_columns': get_metadata_columns(),
        'feature_groups': get_feature_columns_by_group(),
        'trade_quote_columns': get_trade_quote_feature_columns(),
        'total_greek_features': len(get_usable_greek_columns()),
    }
    
    with open(output_path, 'w') as f:
        json.dump(usable, f, indent=2)


def print_validation_summary():
    report = validate_greek_columns()
    
    print("=" * 80)
    print("HYBRID51 DATA VALIDATION REPORT")
    print("=" * 80)
    print(f"\nTotal columns in Greek CSV: {report.total_columns}")
    print(f"Usable feature columns: {report.usable_columns}")
    print(f"Excluded columns: {report.excluded_columns}")
    
    print("\n" + "-" * 40)
    print("EXCLUDED COLUMNS (Always Zero):")
    print("-" * 40)
    for col in report.columns:
        if col.status == ColumnStatus.ALWAYS_ZERO:
            print(f"  X {col.name:20s} - {col.reason}")
    
    print("\n" + "-" * 40)
    print("USABLE FEATURE COLUMNS:")
    print("-" * 40)
    
    groups = get_feature_columns_by_group()
    for group_name, columns in groups.items():
        print(f"\n  {group_name} ({len(columns)} features):")
        for col in columns:
            print(f"    - {col}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL USABLE GREEK FEATURES: {len(get_usable_greek_columns())}")
    print("=" * 80)


if __name__ == "__main__":
    print_validation_summary()
    
    output_dir = Path("/workspace/hybrid51/hybrid51_preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = validate_greek_columns()
    save_validation_report(report, output_dir / "data_validation_report.json")
    save_usable_columns(output_dir / "usable_columns.json")
    
    print(f"\nSaved reports to {output_dir}")
