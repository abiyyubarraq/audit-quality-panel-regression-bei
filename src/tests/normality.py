"""
Normality Tests

Tests for checking whether data follows a normal distribution.
Commonly used for testing normality of regression residuals.

Available tests:
- Shapiro-Wilk Test (recommended for small-medium samples)
- Kolmogorov-Smirnov Test (general purpose)
- Jarque-Bera Test (tests skewness and kurtosis)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional
from . import StatisticalTest, TestResult


class ShapiroWilkTest(StatisticalTest):
    """
    Shapiro-Wilk Test for Normality

    Null Hypothesis: Data follows a normal distribution

    Characteristics:
    - Best for small to medium samples (n < 2000)
    - Most powerful normality test for moderate sample sizes
    - Recommended as default for econometric research

    Usage:
        test = ShapiroWilkTest(alpha=0.05)
        result = test.run(residuals)
        print(result)
    """

    def run(self, data: pd.Series, **kwargs) -> TestResult:
        """
        Execute Shapiro-Wilk normality test

        Args:
            data: Series of values to test for normality
            **kwargs: Additional parameters (not used)

        Returns:
            TestResult object with test results

        Raises:
            ValueError: If data has fewer than 3 observations
        """
        # Remove NaN values
        clean_data = data.dropna()

        if len(clean_data) < 3:
            raise ValueError("Shapiro-Wilk test requires at least 3 observations")

        if len(clean_data) > 5000:
            print(f"Warning: Large sample size (n={len(clean_data)}). "
                  "Consider using Kolmogorov-Smirnov or Jarque-Bera test instead.")

        # Perform Shapiro-Wilk test
        statistic, p_value = stats.shapiro(clean_data)
        reject_null = p_value < self.alpha

        # Interpretation
        interpretation = (
            f"At α = {self.alpha}, the data "
            f"{'does NOT follow' if reject_null else 'follows'} "
            f"a normal distribution (p = {p_value:.4f})"
        )

        return TestResult(
            test_name="Shapiro-Wilk Normality Test",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            interpretation=interpretation,
            additional_info={
                "n_observations": len(clean_data),
                "n_missing": len(data) - len(clean_data),
                "alpha": self.alpha
            }
        )


class KolmogorovSmirnovTest(StatisticalTest):
    """
    Kolmogorov-Smirnov Test for Normality

    Null Hypothesis: Data follows a normal distribution

    Characteristics:
    - General-purpose goodness-of-fit test
    - Compares empirical distribution to normal distribution
    - Less powerful than Shapiro-Wilk for normality testing

    Usage:
        test = KolmogorovSmirnovTest(alpha=0.05)
        result = test.run(residuals)
        print(result)
    """

    def run(self, data: pd.Series, **kwargs) -> TestResult:
        """
        Execute Kolmogorov-Smirnov normality test

        Args:
            data: Series of values to test for normality
            **kwargs: Additional parameters (not used)

        Returns:
            TestResult object with test results

        Raises:
            ValueError: If data is empty
        """
        # Remove NaN values
        clean_data = data.dropna()

        if len(clean_data) == 0:
            raise ValueError("K-S test requires at least 1 observation")

        # Standardize data (subtract mean, divide by std)
        mean = clean_data.mean()
        std = clean_data.std()

        if std == 0:
            raise ValueError("Data has zero standard deviation")

        standardized_data = (clean_data - mean) / std

        # Perform K-S test against standard normal distribution
        statistic, p_value = stats.kstest(standardized_data, 'norm')
        reject_null = p_value < self.alpha

        # Interpretation
        interpretation = (
            f"At α = {self.alpha}, the data "
            f"{'does NOT follow' if reject_null else 'follows'} "
            f"a normal distribution (p = {p_value:.4f})"
        )

        return TestResult(
            test_name="Kolmogorov-Smirnov Normality Test",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            interpretation=interpretation,
            additional_info={
                "n_observations": len(clean_data),
                "n_missing": len(data) - len(clean_data),
                "mean": mean,
                "std": std,
                "alpha": self.alpha
            }
        )


class JarqueBeraTest(StatisticalTest):
    """
    Jarque-Bera Test for Normality

    Null Hypothesis: Data follows a normal distribution

    Characteristics:
    - Tests whether skewness and kurtosis match normal distribution
    - Asymptotic test (best for large samples, n > 2000)
    - Provides information about distribution shape

    Formula:
        JB = (n/6) * (S^2 + (K-3)^2/4)
        where S = skewness, K = kurtosis

    Usage:
        test = JarqueBeraTest(alpha=0.05)
        result = test.run(residuals)
        print(result)
    """

    def run(self, data: pd.Series, **kwargs) -> TestResult:
        """
        Execute Jarque-Bera normality test

        Args:
            data: Series of values to test for normality
            **kwargs: Additional parameters (not used)

        Returns:
            TestResult object with test results

        Raises:
            ValueError: If data has fewer than 2 observations
        """
        # Remove NaN values
        clean_data = data.dropna()

        if len(clean_data) < 2:
            raise ValueError("Jarque-Bera test requires at least 2 observations")

        # Perform Jarque-Bera test
        statistic, p_value = stats.jarque_bera(clean_data)
        reject_null = p_value < self.alpha

        # Calculate skewness and kurtosis
        skewness = stats.skew(clean_data)
        kurtosis = stats.kurtosis(clean_data)

        # Interpretation with skewness and kurtosis information
        skew_interpretation = (
            "positively skewed (right tail)" if skewness > 0
            else "negatively skewed (left tail)" if skewness < 0
            else "symmetric"
        )

        kurt_interpretation = (
            "heavy-tailed (leptokurtic)" if kurtosis > 0
            else "light-tailed (platykurtic)" if kurtosis < 0
            else "mesokurtic (normal tails)"
        )

        interpretation = (
            f"At α = {self.alpha}, the data "
            f"{'does NOT follow' if reject_null else 'follows'} "
            f"a normal distribution (p = {p_value:.4f}). "
            f"Distribution is {skew_interpretation} and {kurt_interpretation}."
        )

        return TestResult(
            test_name="Jarque-Bera Normality Test",
            statistic=statistic,
            p_value=p_value,
            reject_null=reject_null,
            interpretation=interpretation,
            additional_info={
                "n_observations": len(clean_data),
                "n_missing": len(data) - len(clean_data),
                "skewness": skewness,
                "kurtosis": kurtosis,
                "skew_interpretation": skew_interpretation,
                "kurt_interpretation": kurt_interpretation,
                "alpha": self.alpha
            }
        )


class NormalityTestSuite:
    """
    Run multiple normality tests and compare results

    Provides a convenient way to run all normality tests at once
    and get a comprehensive assessment.

    Usage:
        suite = NormalityTestSuite(alpha=0.05)
        results = suite.run_all(residuals)
        summary = suite.summarize(results)
        print(summary)
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize test suite

        Args:
            alpha: Significance level for all tests
        """
        self.alpha = alpha
        self.tests = {
            'shapiro': ShapiroWilkTest(alpha),
            'ks': KolmogorovSmirnovTest(alpha),
            'jarque_bera': JarqueBeraTest(alpha)
        }

    def run_all(
        self,
        data: pd.Series,
        tests: Optional[list] = None
    ) -> Dict[str, TestResult]:
        """
        Run all normality tests

        Args:
            data: Series of values to test for normality
            tests: List of test names to run (default: all)
                  Options: ['shapiro', 'ks', 'jarque_bera']

        Returns:
            Dictionary mapping test names to TestResult objects
        """
        if tests is None:
            tests = list(self.tests.keys())

        results = {}
        for test_name in tests:
            if test_name not in self.tests:
                print(f"Warning: Unknown test '{test_name}'. Skipping.")
                continue

            try:
                results[test_name] = self.tests[test_name].run(data)
            except Exception as e:
                print(f"Error running {test_name} test: {str(e)}")
                results[test_name] = None

        return results

    def summarize(self, results: Dict[str, TestResult]) -> str:
        """
        Create summary of all test results

        Args:
            results: Dictionary of test results from run_all()

        Returns:
            Formatted string summarizing all test results
        """
        output = []
        output.append("\n" + "=" * 60)
        output.append("NORMALITY TEST SUITE SUMMARY")
        output.append("=" * 60)

        # Count rejections
        valid_results = [r for r in results.values() if r is not None]
        if not valid_results:
            return "No valid test results to summarize"

        n_reject = sum(1 for r in valid_results if r.reject_null)
        n_total = len(valid_results)

        output.append(f"\nTests rejecting normality: {n_reject}/{n_total}")

        # Individual test results
        for test_name, result in results.items():
            if result is not None:
                output.append(str(result))
            else:
                output.append(f"\n{test_name}: FAILED")

        # Overall conclusion
        output.append("\n" + "-" * 60)
        output.append("OVERALL ASSESSMENT:")

        if n_reject == 0:
            conclusion = "All tests indicate data follows normal distribution"
        elif n_reject == n_total:
            conclusion = "All tests reject normality - data is NOT normally distributed"
        else:
            conclusion = f"Mixed results - {n_reject}/{n_total} tests reject normality"

        output.append(conclusion)
        output.append("=" * 60)

        return "\n".join(output)

    def recommend_test(self, sample_size: int) -> str:
        """
        Recommend most appropriate normality test based on sample size

        Args:
            sample_size: Number of observations

        Returns:
            Recommended test name
        """
        if sample_size < 50:
            return "shapiro"
        elif sample_size < 2000:
            return "shapiro"
        else:
            return "jarque_bera"


# Export classes
__all__ = [
    'ShapiroWilkTest',
    'KolmogorovSmirnovTest',
    'JarqueBeraTest',
    'NormalityTestSuite'
]
