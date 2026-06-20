"""
Data Loader Module

Handles loading data from various formats (Excel, JSON, CSV).
Provides a unified interface for data loading operations.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Union


class DataLoader:
    """
    Load data from multiple file formats

    Supports:
    - Excel files (.xlsx, .xls)
    - JSON files (.json)
    - CSV files (.csv)

    Usage:
        loader = DataLoader()
        data = loader.load_excel('data/raw/audit_data.xlsx')
        data = loader.load_json('data/processed/audit_data.json')
    """

    def __init__(self):
        """Initialize DataLoader"""
        pass

    def load_excel(
        self,
        filepath: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from Excel file

        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index (default: 0, first sheet)
            **kwargs: Additional arguments passed to pandas.read_excel

        Returns:
            DataFrame containing the data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.suffix not in ['.xlsx', '.xls']:
            raise ValueError(f"Invalid Excel file format: {filepath.suffix}")

        try:
            print(f"Loading Excel file: {filepath}")
            df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading Excel file: {str(e)}")

    def load_json(
        self,
        filepath: Union[str, Path],
        orient: str = 'records'
    ) -> pd.DataFrame:
        """
        Load data from JSON file

        Args:
            filepath: Path to JSON file
            orient: JSON orientation (default: 'records')
                   Options: 'records', 'index', 'columns', 'values', 'table'

        Returns:
            DataFrame containing the data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.suffix != '.json':
            raise ValueError(f"Invalid JSON file format: {filepath.suffix}")

        try:
            print(f"Loading JSON file: {filepath}")
            df = pd.read_json(filepath, orient=orient)
            print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {str(e)}")

    def load_csv(
        self,
        filepath: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments passed to pandas.read_csv

        Returns:
            DataFrame containing the data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV format is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.suffix != '.csv':
            raise ValueError(f"Invalid CSV file format: {filepath.suffix}")

        try:
            print(f"Loading CSV file: {filepath}")
            df = pd.read_csv(filepath, **kwargs)
            print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    def load_auto(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Automatically detect file format and load data

        Args:
            filepath: Path to data file
            **kwargs: Additional arguments passed to specific loader

        Returns:
            DataFrame containing the data
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = filepath.suffix.lower()

        if suffix in ['.xlsx', '.xls']:
            return self.load_excel(filepath, **kwargs)
        elif suffix == '.json':
            return self.load_json(filepath, **kwargs)
        elif suffix == '.csv':
            return self.load_csv(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def get_file_info(self, filepath: Union[str, Path]) -> dict:
        """
        Get information about a data file without loading it

        Args:
            filepath: Path to data file

        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        return {
            'filename': filepath.name,
            'format': filepath.suffix,
            'size_bytes': filepath.stat().st_size,
            'size_mb': filepath.stat().st_size / (1024 * 1024),
            'exists': True
        }


__all__ = ['DataLoader']
