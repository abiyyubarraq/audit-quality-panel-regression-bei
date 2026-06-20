"""
Data Converter Module

Handles conversion between different data formats.
Primary use: Converting Excel data to JSON format.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Union, Optional
from datetime import datetime


class ExcelToJSONConverter:
    """
    Convert Excel data to JSON format with validation

    Features:
    - Handles date/datetime conversions
    - Preserves data types
    - Validates conversion accuracy
    - Adds metadata

    Usage:
        converter = ExcelToJSONConverter()
        converter.convert(
            'data/raw/audit_data.xlsx',
            'data/processed/audit_data.json'
        )
    """

    def __init__(self):
        """Initialize converter"""
        pass

    def convert(
        self,
        excel_path: Union[str, Path],
        json_path: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        date_format: str = 'iso',
        include_metadata: bool = True,
        validate: bool = True
    ) -> None:
        """
        Convert Excel file to JSON

        Args:
            excel_path: Path to source Excel file
            json_path: Path to destination JSON file
            sheet_name: Excel sheet to convert (default: 0, first sheet)
            date_format: Format for dates ('iso' or 'epoch')
            include_metadata: Include conversion metadata in JSON
            validate: Validate conversion accuracy

        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If conversion fails or validation fails
        """
        excel_path = Path(excel_path)
        json_path = Path(json_path)

        # Ensure output directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # Load Excel data
        print(f"Loading Excel file: {excel_path}")
        excel_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"Loaded {len(excel_df)} rows, {len(excel_df.columns)} columns")

        # Convert to JSON-friendly format
        json_df = self._prepare_for_json(excel_df, date_format)

        # Create JSON structure
        if include_metadata:
            json_data = {
                'metadata': {
                    'source_file': str(excel_path),
                    'conversion_date': datetime.now().isoformat(),
                    'n_rows': len(json_df),
                    'n_columns': len(json_df.columns),
                    'columns': list(json_df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in json_df.dtypes.items()}
                },
                'data': json_df.to_dict(orient='records')
            }
        else:
            json_data = json_df.to_dict(orient='records')

        # Save to JSON
        print(f"Saving JSON file: {json_path}")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Conversion complete: {json_path}")

        # Validate if requested
        if validate:
            self.validate_conversion(excel_df, json_df)

    def _prepare_for_json(
        self,
        df: pd.DataFrame,
        date_format: str = 'iso'
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for JSON conversion

        Handles:
        - Date/datetime conversion
        - NaN replacement
        - Type conversions

        Args:
            df: Input DataFrame
            date_format: Format for dates ('iso' or 'epoch')

        Returns:
            DataFrame ready for JSON conversion
        """
        df = df.copy()

        # Convert datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if date_format == 'iso':
                    df[col] = df[col].dt.strftime('%Y-%m-%d')
                elif date_format == 'epoch':
                    df[col] = df[col].astype(int) / 10**9  # Convert to seconds

        # Replace NaN with None for JSON
        df = df.where(pd.notnull(df), None)

        return df

    def validate_conversion(
        self,
        excel_df: pd.DataFrame,
        json_df: pd.DataFrame
    ) -> bool:
        """
        Validate that conversion preserved data integrity

        Args:
            excel_df: Original DataFrame from Excel
            json_df: Converted DataFrame

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        print("\nValidating conversion...")

        # Check shape
        if excel_df.shape != json_df.shape:
            raise ValueError(
                f"Shape mismatch: Excel {excel_df.shape} vs JSON {json_df.shape}"
            )

        # Check columns
        if not all(excel_df.columns == json_df.columns):
            raise ValueError("Column names don't match")

        # Check for data loss (non-null counts)
        for col in excel_df.columns:
            excel_count = excel_df[col].notna().sum()
            json_count = json_df[col].notna().sum()
            if excel_count != json_count:
                print(f"Warning: Non-null count mismatch in column '{col}': "
                      f"Excel={excel_count}, JSON={json_count}")

        print("✓ Validation passed")
        return True

    def convert_with_data_dict(
        self,
        excel_path: Union[str, Path],
        json_path: Union[str, Path],
        data_dict_path: Union[str, Path],
        sheet_name: Union[str, int] = 0
    ) -> None:
        """
        Convert Excel to JSON with data dictionary metadata

        Args:
            excel_path: Path to Excel file
            json_path: Path to output JSON file
            data_dict_path: Path to data dictionary (YAML/JSON)
            sheet_name: Excel sheet name/index
        """
        # Load data dictionary
        import yaml

        data_dict_path = Path(data_dict_path)
        if data_dict_path.suffix == '.yaml':
            with open(data_dict_path, 'r') as f:
                data_dict = yaml.safe_load(f)
        elif data_dict_path.suffix == '.json':
            with open(data_dict_path, 'r') as f:
                data_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported data dictionary format: {data_dict_path.suffix}")

        # Convert
        self.convert(excel_path, json_path, sheet_name, include_metadata=True)

        # Add data dictionary to JSON
        json_path = Path(json_path)
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        json_data['data_dictionary'] = data_dict

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Data dictionary added to {json_path}")


__all__ = ['ExcelToJSONConverter']
