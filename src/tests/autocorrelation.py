"""
Autocorrelation Tests

Tests for checking independence of errors (no autocorrelation).
Autocorrelation can occur in time-series or panel data and leads to
inefficient estimates and biased standard errors.

Available tests:
- Durbin-Watson Test (quick screening for first-order autocorrelation)
- Breusch-Godfrey Test (tests higher-order autocorrelation)
"""

import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from typing import Dict, Optional, Tuple
from . import StatisticalTest, TestResult


class DurbinWatsonTest(StatisticalTest):
    """
    Durbin-Watson Test for First-Order Autocorrelation

    Test statistic ranges from 0 to 4:
    - DW ≈ 2: No autocorrelation
    - DW < 2: Positive autocorrelation
    - DW > 2: Negative autocorrelation

    Rule of thumb:
    - 1.5 < DW < 2.5: Acceptable (no strong autocorrelation)
    - DW < 1.5: Positive autocorrelation
    - DW > 2.5: Negative autocorrelation

    Characteristics:
    - Simple and widely used
    - Only tests first-order autocorrelation
    - Standard in audit research literature
    - Recommended as initial screening tool

    Usage:
        test = DurbinWatsonTest(alpha=0.05)
        result = test.run(model_results)
        print(result)
    """

    def __init__(self, alpha: float = 0.05, acceptable_range: Tuple[float, float] = (1.5, 2.5)):
        """
        Initialize Durbin-Watson test

        Args:
            alpha: Significance level (not directly used in DW test)
            acceptable_range: Tuple of (lower, upper) bounds for acceptable DW statistic
        """
        super().__init__(alpha)
        self.acceptable_range = acceptable_range

    def run(self, model_results, **kwargs) -> TestResult:
        """
        Execute Durbin-Watson test

        Args:
            model_results: Fitted regression model results (from statsmodels)
            **kwargs: Additional parameters (not used)

        Returns:
            TestResult object with test results

        Raises:
            ValueError: If model_results is invalid
        """
        try:
            # Calculate Durbin-Watson statistic
            dw_stat = durbin_watson(model_results.resid)

            # Determine autocorrelation type
            lower_bound, upper_bound = self.acceptable_range

            if dw_stat < lower_bound:
                interpretation = (
                    f"Positive autocorrelation detected (DW = {dw_stat:.4f} < {lower_bound}). "
                    "Residuals are positively correlated over time."
                )
                reject_null = True
                autocorr_type = "positive"

            elif dw_stat > upper_bound:
                interpretation = (
                    f"Negative autocorrelation detected (DW = {dw_stat:.4f} > {upper_bound}). "
                    "Residuals are negatively correlated over time."
                )
                reject_null = True
                autocorr_type = "negative"

            else:
                interpretation = (
                    f"No significant autocorrelation (DW = {dw_stat:.4f} is within acceptable range "
                    f"[{lower_bound}, {upper_bound}])."
                )
                reject_null = False
                autocorr_type = "none"

            return TestResult(
                test_name="Durbin-Watson Test for Autocorrelation",
                statistic=dw_stat,
                p_value=None,  # DW test doesn't provide p-value directly
                reject_null=reject_null,
                interpretation=interpretation,
                additional_info={
                    "acceptable_range": f"{lower_bound} - {upper_bound}",
                    "autocorrelation_type": autocorr_type,
                    "ideal_value": 2.0,
                    "note": "DW ≈ 2 indicates no autocorrelation"
                }
            )

        except Exception as e:
            raise ValueError(f"Error running Durbin-Watson test: {str(e)}")


class BreuschGodfreyTest(StatisticalTest):
    """
    Breusch-Godfrey LM Test for Autocorrelation

    Null Hypothesis: No autocorrelation of any order up to lag p
    Alternative: Autocorrelation exists

    Characteristics:
    - More general than Durbin-Watson
    - Can test for higher-order autocorrelation
    - Provides formal hypothesis test with p-value
    - Important for panel data (within-entity correlation)
    - Robust to various model specifications

    Usage:
        test = BreuschGodfreyTest(alpha=0.05)
        result = test.run(model_results, nlags=1)
        print(result)
    """

    def run(self, model_results, nlags: int = 1, **kwargs) -> TestResult:
        """
        Execute Breusch-Godfrey test

        Args:
            model_results: Fitted regression model results (from statsmodels)
            nlags: Number of lags to test (default: 1)
            **kwargs: Additional parameters (not used)

        Returns:
            TestResult object with test results

        Raises:
            ValueError: If model_results is invalid or nlags is invalid
        """
        if nlags < 1:
            raise ValueError("nlags must be at least 1")

        try:
            # Perform Breusch-Godfrey test
            lm_stat, lm_pvalue, f_stat, f_pvalue = acorr_breusch_godfrey(
                model_results,
                nlags=nlags
            )

            reject_null = lm_pvalue < self.alpha

            # Interpretation
            if reject_null:
                interpretation = (
                    f"At α = {self.alpha}, autocorrelation is detected "
                    f"up to lag {nlags} (p = {lm_pvalue:.4f}). "
                    "Consider using robust standard errors clustered by entity."
                )
            else:
                interpretation = (
                    f"At α = {self.alpha}, no autocorrelation detected "
                    f"up to lag {nlags} (p = {lm_pvalue:.4f})."
                )

            return TestResult(
                test_name=f"Breusch-Godfrey Test for Autocorrelation (lag={nlags})",
                statistic=lm_stat,
                p_value=lm_pvalue,
                reject_null=reject_null,
                interpretation=interpretation,
                additional_info={
                    "f_statistic": f_stat,
                    "f_pvalue": f_pvalue,
                    "lags_tested": nlags,
                    "alpha": self.alpha,
                    "test_type": "LM test"
                }
            )

        except Exception as e:
            raise ValueError(f"Error running Breusch-Godfrey test: {str(e)}")


class AutocorrelationTestSuite:
    """
    Run multiple autocorrelation tests and compare results

    Provides comprehensive assessment of the independence assumption.

    Usage:
        suite = AutocorrelationTestSuite(alpha=0.05)
        results = suite.run_all(model_results)
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
        self.dw_test = DurbinWatsonTest(alpha)
        self.bg_test = BreuschGodfreyTest(alpha)

    def run_all(
        self,
        model_results,
        bg_lags: int = 1,
        tests: Optional[list] = None
    ) -> Dict[str, TestResult]:
        """
        Run all autocorrelation tests

        Args:
            model_results: Fitted regression model results
            bg_lags: Number of lags for Breusch-Godfrey test (default: 1)
            tests: List of test names to run (default: all)
                  Options: ['durbin_watson', 'breusch_godfrey']

        Returns:
            Dictionary mapping test names to TestResult objects
        """
        if tests is None:
            tests = ['durbin_watson', 'breusch_godfrey']

        results = {}

        if 'durbin_watson' in tests:
            try:
                results['durbin_watson'] = self.dw_test.run(model_results)
            except Exception as e:
                print(f"Error running Durbin-Watson test: {str(e)}")
                results['durbin_watson'] = None

        if 'breusch_godfrey' in tests:
            try:
                results['breusch_godfrey'] = self.bg_test.run(model_results, nlags=bg_lags)
            except Exception as e:
                print(f"Error running Breusch-Godfrey test: {str(e)}")
                results['breusch_godfrey'] = None

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
        output.append("AUTOCORRELATION TEST SUITE SUMMARY")
        output.append("=" * 60)

        # Count rejections
        valid_results = [r for r in results.values() if r is not None]
        if not valid_results:
            return "No valid test results to summarize"

        n_reject = sum(1 for r in valid_results if r.reject_null)
        n_total = len(valid_results)

        output.append(f"\nTests detecting autocorrelation: {n_reject}/{n_total}")

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
            conclusion = "No autocorrelation detected - independence assumption holds"
        elif n_reject == n_total:
            conclusion = ("Autocorrelation detected by all tests - "
                         "Consider using robust standard errors clustered by entity")
        else:
            conclusion = (f"Mixed results - {n_reject}/{n_total} tests detect autocorrelation. "
                         "Consider clustered standard errors as precaution.")

        output.append(conclusion)
        output.append("=" * 60)

        return "\n".join(output)


# Export classes
__all__ = [
    'DurbinWatsonTest',
    'BreuschGodfreyTest',
    'AutocorrelationTestSuite'
]
