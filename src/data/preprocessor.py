"""
Data Preprocessor Module

Handles data cleaning and transformation for panel data analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional


class DataPreprocessor:
    """
    Clean and transform data for analysis

    Features:
    - Missing value handling
    - Variable transformations (log, standardization)
    - Panel structure creation
    - Data type conversions

    Usage:
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.handle_missing_values(data)
        panel_data = preprocessor.create_panel_structure(clean_data, 'company_id', 'year')
    """

    def __init__(self):
        """Initialize DataPreprocessor"""
        pass

    def handle_missing_values(
        self,
        data: pd.DataFrame,
        strategy: str = 'drop',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values

        Args:
            data: DataFrame with potential missing values
            strategy: Strategy for handling missing values
                     'drop': Drop rows with missing values
                     'mean': Fill with mean
                     'median': Fill with median
                     'forward_fill': Forward fill
                     'backward_fill': Backward fill
            columns: Specific columns to handle (default: all)

        Returns:
            DataFrame with missing values handled
        """
        data = data.copy()

        if columns is None:
            columns = data.columns.tolist()

        print(f"\nHandling missing values using strategy: '{strategy}'")
        before_count = data.isnull().sum().sum()

        if strategy == 'drop':
            data = data.dropna(subset=columns)
            print(f"Dropped rows with missing values")

        elif strategy == 'mean':
            for col in columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].fillna(data[col].mean())
            print(f"Filled missing values with mean")

        elif strategy == 'median':
            for col in columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].fillna(data[col].median())
            print(f"Filled missing values with median")

        elif strategy == 'forward_fill':
            data[columns] = data[columns].fillna(method='ffill')
            print(f"Forward filled missing values")

        elif strategy == 'backward_fill':
            data[columns] = data[columns].fillna(method='bfill')
            print(f"Backward filled missing values")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        after_count = data.isnull().sum().sum()
        print(f"Missing values: {before_count} → {after_count}")
        print(f"Rows: {len(data)}")

        return data

    def create_log_variables(
        self,
        data: pd.DataFrame,
        columns: List[str],
        suffix: str = '_log'
    ) -> pd.DataFrame:
        """
        Create log-transformed variables

        Args:
            data: DataFrame
            columns: Columns to log-transform
            suffix: Suffix for new column names (default: '_log')

        Returns:
            DataFrame with log-transformed columns added
        """
        data = data.copy()

        print(f"\nCreating log-transformed variables:")

        for col in columns:
            if col not in data.columns:
                print(f"Warning: Column '{col}' not found")
                continue

            new_col = f"{col}{suffix}"

            # Check for non-positive values
            if (data[col] <= 0).any():
                print(f"Warning: '{col}' contains non-positive values. "
                      f"Adding constant before log transform.")
                min_val = data[col].min()
                shift = abs(min_val) + 1 if min_val <= 0 else 0
                data[new_col] = np.log(data[col] + shift)
            else:
                data[new_col] = np.log(data[col])

            print(f"  {col} → {new_col}")

        return data

    def standardize_variables(
        self,
        data: pd.DataFrame,
        columns: List[str],
        suffix: str = '_std'
    ) -> pd.DataFrame:
        """
        Standardize variables (mean=0, std=1)

        Args:
            data: DataFrame
            columns: Columns to standardize
            suffix: Suffix for new column names (default: '_std')

        Returns:
            DataFrame with standardized columns added
        """
        data = data.copy()

        print(f"\nStandardizing variables:")

        for col in columns:
            if col not in data.columns:
                print(f"Warning: Column '{col}' not found")
                continue

            new_col = f"{col}{suffix}"
            mean = data[col].mean()
            std = data[col].std()

            if std == 0:
                print(f"Warning: '{col}' has zero standard deviation. Skipping.")
                continue

            data[new_col] = (data[col] - mean) / std
            print(f"  {col} → {new_col} (μ={mean:.2f}, σ={std:.2f})")

        return data

    def create_panel_structure(
        self,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        sort: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for panel analysis

        Args:
            data: DataFrame
            entity_col: Name of entity identifier column
            time_col: Name of time identifier column
            sort: Sort by entity and time (default: True)

        Returns:
            DataFrame with proper panel structure
        """
        data = data.copy()

        print(f"\nCreating panel structure:")
        print(f"  Entity column: {entity_col}")
        print(f"  Time column: {time_col}")

        # Ensure columns exist
        if entity_col not in data.columns:
            raise ValueError(f"Entity column '{entity_col}' not found")
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found")

        # Sort by entity and time
        if sort:
            data = data.sort_values([entity_col, time_col])
            print(f"  Sorted by {entity_col}, {time_col}")

        # Reset index
        data = data.reset_index(drop=True)

        print(f"  Entities: {data[entity_col].nunique()}")
        print(f"  Time periods: {data[time_col].nunique()}")
        print(f"  Total observations: {len(data)}")

        return data

    def remove_outliers(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from dataset

        Args:
            data: DataFrame
            columns: Columns to check for outliers
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        data = data.copy()
        original_len = len(data)

        print(f"\nRemoving outliers using {method.upper()} method:")

        for col in columns:
            if col not in data.columns:
                continue

            col_data = data[col].dropna()

            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                mask = (data[col] >= lower_bound) & (data[col] <= upper_bound)

            elif method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                z_scores = np.abs((data[col] - mean) / std)
                mask = z_scores <= threshold

            else:
                raise ValueError(f"Unknown method: {method}")

            before_len = len(data)
            data = data[mask | data[col].isna()]  # Keep NaN values
            removed = before_len - len(data)

            if removed > 0:
                print(f"  {col}: Removed {removed} outliers")

        total_removed = original_len - len(data)
        print(f"\nTotal rows removed: {total_removed} ({total_removed/original_len*100:.1f}%)")
        print(f"Remaining rows: {len(data)}")

        return data


__all__ = ['DataPreprocessor']
