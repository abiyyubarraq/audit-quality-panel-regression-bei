"""
Statistical Tests Package

Base classes and utilities for statistical testing in panel data analysis.
Provides a standardized interface for all diagnostic tests.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd


@dataclass
class TestResult:
    """
    Standardized test result structure

    Attributes:
        test_name: Name of the statistical test
        statistic: Test statistic value
        p_value: P-value of the test (None if not applicable)
        reject_null: Boolean indicating whether to reject null hypothesis
        interpretation: Human-readable interpretation of the result
        additional_info: Dictionary of additional test-specific information
    """
    test_name: str
    statistic: float
    p_value: Optional[float]
    reject_null: bool
    interpretation: str
    additional_info: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format test result for console output"""
        output = []
        output.append(f"\n{self.test_name}")
        output.append("=" * 60)
        output.append(f"Statistic: {self.statistic:.4f}")

        if self.p_value is not None:
            output.append(f"P-value: {self.p_value:.4f}")

        output.append(f"Reject Null Hypothesis: {'Yes' if self.reject_null else 'No'}")
        output.append(f"\nInterpretation: {self.interpretation}")

        if self.additional_info:
            output.append("\nAdditional Information:")
            for key, value in self.additional_info.items():
                if isinstance(value, float):
                    output.append(f"  {key}: {value:.4f}")
                else:
                    output.append(f"  {key}: {value}")

        return "\n".join(output)

    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary"""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'reject_null': self.reject_null,
            'interpretation': self.interpretation,
            'additional_info': self.additional_info
        }


class StatisticalTest(ABC):
    """
    Abstract base class for all statistical tests

    Provides a common interface for running statistical tests with
    standardized output format.

    Attributes:
        alpha: Significance level (default: 0.05)
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical test

        Args:
            alpha: Significance level for hypothesis testing (default: 0.05)
        """
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha

    @abstractmethod
    def run(self, data, **kwargs) -> TestResult:
        """
        Execute the statistical test

        Args:
            data: Input data (format depends on specific test)
            **kwargs: Additional test-specific parameters

        Returns:
            TestResult object containing test results
        """
        pass

    def run_multiple(self, data: pd.DataFrame, columns: list, **kwargs) -> Dict[str, TestResult]:
        """
        Run test on multiple variables

        Args:
            data: DataFrame containing variables to test
            columns: List of column names to test
            **kwargs: Additional test-specific parameters

        Returns:
            Dictionary mapping column names to TestResult objects
        """
        results = {}
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            try:
                results[col] = self.run(data[col], **kwargs)
            except Exception as e:
                print(f"Warning: Test failed for column '{col}': {str(e)}")
                results[col] = None
        return results


# Export classes
__all__ = ['TestResult', 'StatisticalTest']
