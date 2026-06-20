"""
Data Validator Module

Validates data quality and structure for panel data analysis.
Checks for missing values, outliers, data types, and panel structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class DataValidator:
    """
    Validate data quality and panel structure

    Features:
    - Panel structure validation
    - Missing value analysis
    - Outlier detection
    - Variable range checks
    - Data type validation

    Usage:
        validator = DataValidator()
        report = validator.validate_panel_structure(data, 'company_id', 'year')
        missing_report = validator.check_missing_values(data)
    """

    def __init__(self):
        """Initialize DataValidator"""
        pass

    def validate_panel_structure(
        self,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str
    ) -> Dict[str, Any]:
        """
        Validate panel data structure

        Checks:
        - Multiple entities exist
        - Multiple time periods exist
        - Whether panel is balanced or unbalanced
        - Entity and time coverage

        Args:
            data: DataFrame to validate
            entity_col: Name of entity identifier column
            time_col: Name of time identifier column

        Returns:
            Dictionary with validation results
        """
        print("\n" + "=" * 60)
        print("PANEL DATA STRUCTURE VALIDATION")
        print("=" * 60)

        report = {}

        # Check if columns exist
        if entity_col not in data.columns:
            raise ValueError(f"Entity column '{entity_col}' not found in DataFrame")
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")

        # Entity information
        n_entities = data[entity_col].nunique()
        entities = data[entity_col].unique()
        report['n_entities'] = n_entities
        report['entities'] = list(entities)

        # Time information
        n_periods = data[time_col].nunique()
        periods = sorted(data[time_col].unique())
        report['n_periods'] = n_periods
        report['time_periods'] = list(periods)

        # Total observations
        n_obs = len(data)
        report['n_observations'] = n_obs

        # Check if balanced
        expected_obs = n_entities * n_periods
        is_balanced = (n_obs == expected_obs)
        report['is_balanced'] = is_balanced

        # Entity-time coverage
        entity_time_counts = data.groupby(entity_col)[time_col].count()
        report['min_periods_per_entity'] = int(entity_time_counts.min())
        report['max_periods_per_entity'] = int(entity_time_counts.max())
        report['mean_periods_per_entity'] = float(entity_time_counts.mean())

        # Print report
        print(f"\nEntities: {n_entities}")
        print(f"Time Periods: {n_periods} {periods}")
        print(f"Total Observations: {n_obs}")
        print(f"Expected (balanced): {expected_obs}")
        print(f"Panel Type: {'Balanced' if is_balanced else 'Unbalanced'}")

        if not is_balanced:
            print(f"\nObservations per entity:")
            print(f"  Min: {report['min_periods_per_entity']}")
            print(f"  Max: {report['max_periods_per_entity']}")
            print(f"  Mean: {report['mean_periods_per_entity']:.1f}")

            # Show entities with incomplete data
            incomplete = entity_time_counts[entity_time_counts < n_periods]
            if len(incomplete) > 0:
                print(f"\nEntities with incomplete data ({len(incomplete)}):")
                for entity, count in incomplete.items():
                    print(f"  {entity}: {count}/{n_periods} periods")

        # Validation status
        if n_entities < 2:
            report['status'] = 'INVALID'
            report['message'] = 'Panel data requires at least 2 entities'
            print(f"\n✗ VALIDATION FAILED: {report['message']}")
        elif n_periods < 2:
            report['status'] = 'INVALID'
            report['message'] = 'Panel data requires at least 2 time periods'
            print(f"\n✗ VALIDATION FAILED: {report['message']}")
        else:
            report['status'] = 'VALID'
            report['message'] = 'Panel structure is valid'
            print(f"\n✓ Panel structure is valid")

        return report

    def check_missing_values(
        self,
        data: pd.DataFrame,
        threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Analyze missing values in dataset

        Args:
            data: DataFrame to analyze
            threshold: Threshold for flagging columns (default: 0.05 = 5%)

        Returns:
            DataFrame with missing value report
        """
        print("\n" + "=" * 60)
        print("MISSING VALUE ANALYSIS")
        print("=" * 60)

        missing_stats = []

        for col in data.columns:
            n_missing = data[col].isna().sum()
            pct_missing = n_missing / len(data) * 100

            missing_stats.append({
                'column': col,
                'n_missing': n_missing,
                'pct_missing': pct_missing,
                'flag': '⚠' if pct_missing > threshold * 100 else '✓'
            })

        missing_df = pd.DataFrame(missing_stats)
        missing_df = missing_df.sort_values('pct_missing', ascending=False)

        # Print summary
        total_missing = missing_df['n_missing'].sum()
        cols_with_missing = (missing_df['n_missing'] > 0).sum()

        print(f"\nTotal missing values: {total_missing:,}")
        print(f"Columns with missing values: {cols_with_missing}/{len(data.columns)}")

        if cols_with_missing > 0:
            print(f"\nMissing value breakdown:")
            print(missing_df[missing_df['n_missing'] > 0].to_string(index=False))
        else:
            print("\n✓ No missing values detected")

        return missing_df

    def detect_outliers(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in specified columns

        Args:
            data: DataFrame to analyze
            columns: List of column names to check
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
                      IQR: 1.5 (default) or 3.0 (extreme outliers)
                      Z-score: 3.0 (default)

        Returns:
            Dictionary mapping column names to DataFrames of outliers
        """
        print("\n" + "=" * 60)
        print(f"OUTLIER DETECTION ({method.upper()} method)")
        print("=" * 60)

        outliers = {}

        for col in columns:
            if col not in data.columns:
                print(f"Warning: Column '{col}' not found")
                continue

            if not pd.api.types.is_numeric_dtype(data[col]):
                print(f"Skipping non-numeric column: {col}")
                continue

            col_data = data[col].dropna()

            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)

            elif method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                z_scores = np.abs((data[col] - mean) / std)
                outlier_mask = z_scores > threshold

            else:
                raise ValueError(f"Unknown method: {method}")

            outlier_data = data[outlier_mask].copy()
            outliers[col] = outlier_data

            # Print summary
            n_outliers = len(outlier_data)
            pct_outliers = n_outliers / len(data) * 100

            print(f"\n{col}:")
            print(f"  Outliers: {n_outliers} ({pct_outliers:.2f}%)")

            if method == 'iqr' and n_outliers > 0:
                print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"  Outlier values: min={outlier_data[col].min():.2f}, "
                      f"max={outlier_data[col].max():.2f}")

        return outliers

    def validate_variable_ranges(
        self,
        data: pd.DataFrame,
        variable_specs: Dict[str, Tuple[Optional[float], Optional[float]]]
    ) -> Dict[str, List]:
        """
        Check if variables are within expected ranges

        Args:
            data: DataFrame to validate
            variable_specs: Dictionary mapping variable names to (min, max) tuples
                          Use None for unbounded

        Returns:
            Dictionary of variables with out-of-range values
        """
        print("\n" + "=" * 60)
        print("VARIABLE RANGE VALIDATION")
        print("=" * 60)

        violations = {}

        for var, (min_val, max_val) in variable_specs.items():
            if var not in data.columns:
                print(f"Warning: Variable '{var}' not found")
                continue

            var_data = data[var].dropna()

            out_of_range = []

            if min_val is not None:
                below_min = var_data < min_val
                if below_min.any():
                    out_of_range.extend(
                        data[below_min][[var]].to_dict('records')
                    )

            if max_val is not None:
                above_max = var_data > max_val
                if above_max.any():
                    out_of_range.extend(
                        data[above_max][[var]].to_dict('records')
                    )

            if out_of_range:
                violations[var] = out_of_range
                print(f"\n✗ {var}: {len(out_of_range)} values out of range "
                      f"[{min_val}, {max_val}]")
            else:
                print(f"✓ {var}: All values within range [{min_val}, {max_val}]")

        return violations


__all__ = ['DataValidator']
