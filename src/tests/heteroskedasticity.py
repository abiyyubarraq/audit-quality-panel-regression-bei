"""
Heteroskedasticity Tests

Tests for checking homoskedasticity assumption (constant variance of errors).
Heteroskedasticity can lead to inefficient estimates and invalid standard errors.

Available tests:
- Breusch-Pagan Test (recommended for panel data)
- White's Test (more general, no functional form assumption)
- Glejser Test (regresses absolute residuals)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from typing import Dict, Optional
from . import StatisticalTest, TestResult


class BreuschPaganTest(StatisticalTest):
    """
    Breusch-Pagan Test for Heteroskedasticity

    Null Hypothesis: Homoskedasticity (constant variance)
    Alternative: Heteroskedasticity (non-constant variance)

    Characteristics:
    - Standard test in econometrics and panel data analysis
    - Assumes linear relationship between variance and predictors
    - Recommended as default for thesis research

    Remedy if heteroskedasticity detected:
    - Use robust standard errors (White's correction)
    - Transform variables (e.g., log transformation)
    - Use weighted least squares

    Usage:
        test = BreuschPaganTest(alpha=0.05)
        result = test.run(model_results)
        print(result)
    """

    def run(self, model_results, **kwargs) -> TestResult:
        """
        Execute Breusch-Pagan test

        Args:
            model_results: Fitted regression model results (from statsmodels)
            **kwargs: Additional parameters (not used)

        Returns:
            TestResult object with test results

        Raises:
            ValueError: If model_results is invalid
        """
        try:
            # Extract residuals and design matrix
            resid = model_results.resid
            exog = model_results.model.exog

            # Perform Breusch-Pagan test
            lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, exog)

            reject_null = lm_pvalue < self.alpha

            # Interpretation
            interpretation = (
                f"At α = {self.alpha}, "
                f"{'heteroskedasticity is detected' if reject_null else 'homoskedasticity holds'} "
                f"(p = {lm_pvalue:.4f}). "
            )

            if reject_null:
                interpretation += "Consider using robust standard errors in your regression."
            else:
                interpretation += "No evidence of heteroskedasticity."

            return TestResult(
                test_name="Breusch-Pagan Test for Heteroskedasticity",
                statistic=lm_stat,
                p_value=lm_pvalue,
                reject_null=reject_null,
                interpretation=interpretation,
                additional_info={
                    "f_statistic": f_stat,
                    "f_pvalue": f_pvalue,
                    "alpha": self.alpha,
                    "test_type": "LM test"
                }
            )

        except Exception as e:
            raise ValueError(f"Error running Breusch-Pagan test: {str(e)}")


class WhiteTest(StatisticalTest):
    """
    White's Test for Heteroskedasticity

    Null Hypothesis: Homoskedasticity (constant variance)
    Alternative: Heteroskedasticity (non-constant variance)

    Characteristics:
    - More general than Breusch-Pagan
    - Includes cross-products and squares of predictors
    - Does not assume specific functional form
    - Can detect more forms of heteroskedasticity

    Usage:
        test = WhiteTest(alpha=0.05)
        result = test.run(model_results)
        print(result)
    """

    def run(self, model_results, **kwargs) -> TestResult:
        """
        Execute White's test

        Args:
            model_results: Fitted regression model results (from statsmodels)
            **kwargs: Additional parameters (not used)

        Returns:
            TestResult object with test results

        Raises:
            ValueError: If model_results is invalid
        """
        try:
            # Extract residuals and design matrix
            resid = model_results.resid
            exog = model_results.model.exog

            # Perform White's test
            lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(resid, exog)

            reject_null = lm_pvalue < self.alpha

            # Interpretation
            interpretation = (
                f"At α = {self.alpha}, "
                f"{'heteroskedasticity is detected' if reject_null else 'homoskedasticity holds'} "
                f"(p = {lm_pvalue:.4f}). "
            )

            if reject_null:
                interpretation += "Consider using heteroskedasticity-robust standard errors."
            else:
                interpretation += "No evidence of heteroskedasticity."

            return TestResult(
                test_name="White's Test for Heteroskedasticity",
                statistic=lm_stat,
                p_value=lm_pvalue,
                reject_null=reject_null,
                interpretation=interpretation,
                additional_info={
                    "f_statistic": f_stat,
                    "f_pvalue": f_pvalue,
                    "alpha": self.alpha,
                    "test_type": "LM test"
                }
            )

        except Exception as e:
            raise ValueError(f"Error running White's test: {str(e)}")


class GlejserTest(StatisticalTest):
    """
    Glejser Test for Heteroskedasticity

    Null Hypothesis: Homoskedasticity (constant variance)
    Alternative: Heteroskedasticity (non-constant variance)

    Characteristics:
    - Regresses absolute residuals on independent variables
    - Simple and intuitive approach
    - Tests if variance systematically relates to predictors

    Method:
    1. Fit original regression model
    2. Calculate absolute residuals
    3. Regress |residuals| on independent variables
    4. F-test for overall significance

    Usage:
        test = GlejserTest(alpha=0.05)
        result = test.run(model_results)
        print(result)
    """

    def run(self, model_results, **kwargs) -> TestResult:
        """
        Execute Glejser test

        Args:
            model_results: Fitted regression model results (from statsmodels)
            **kwargs: Additional parameters (not used)

        Returns:
            TestResult object with test results

        Raises:
            ValueError: If model_results is invalid
        """
        try:
            # Get absolute residuals
            abs_resid = np.abs(model_results.resid)
            exog = model_results.model.exog

            # Remove constant column if exists (first column is usually constant)
            if np.all(exog[:, 0] == exog[0, 0]):
                exog_no_const = exog[:, 1:]
            else:
                exog_no_const = exog

            # Add constant for Glejser regression
            exog_with_const = sm.add_constant(exog_no_const)

            # Regress absolute residuals on predictors
            glejser_model = sm.OLS(abs_resid, exog_with_const).fit()

            # F-test for overall significance
            f_stat = glejser_model.fvalue
            f_pvalue = glejser_model.f_pvalue

            reject_null = f_pvalue < self.alpha

            # Interpretation
            interpretation = (
                f"At α = {self.alpha}, "
                f"{'heteroskedasticity is detected' if reject_null else 'homoskedasticity holds'} "
                f"(p = {f_pvalue:.4f}). "
            )

            if reject_null:
                interpretation += "Variance appears to be related to predictor variables."
            else:
                interpretation += "No systematic relationship between variance and predictors."

            return TestResult(
                test_name="Glejser Test for Heteroskedasticity",
                statistic=f_stat,
                p_value=f_pvalue,
                reject_null=reject_null,
                interpretation=interpretation,
                additional_info={
                    "alpha": self.alpha,
                    "r_squared": glejser_model.rsquared,
                    "glejser_model_df": glejser_model.df_model
                }
            )

        except Exception as e:
            raise ValueError(f"Error running Glejser test: {str(e)}")


class HeteroskedasticityTestSuite:
    """
    Run multiple heteroskedasticity tests and compare results

    Provides a comprehensive assessment of the homoskedasticity assumption.

    Usage:
        suite = HeteroskedasticityTestSuite(alpha=0.05)
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
        self.tests = {
            'breusch_pagan': BreuschPaganTest(alpha),
            'white': WhiteTest(alpha),
            'glejser': GlejserTest(alpha)
        }

    def run_all(
        self,
        model_results,
        tests: Optional[list] = None
    ) -> Dict[str, TestResult]:
        """
        Run all heteroskedasticity tests

        Args:
            model_results: Fitted regression model results
            tests: List of test names to run (default: all)
                  Options: ['breusch_pagan', 'white', 'glejser']

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
                results[test_name] = self.tests[test_name].run(model_results)
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
        output.append("HETEROSKEDASTICITY TEST SUITE SUMMARY")
        output.append("=" * 60)

        # Count rejections
        valid_results = [r for r in results.values() if r is not None]
        if not valid_results:
            return "No valid test results to summarize"

        n_reject = sum(1 for r in valid_results if r.reject_null)
        n_total = len(valid_results)

        output.append(f"\nTests detecting heteroskedasticity: {n_reject}/{n_total}")

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
            conclusion = "All tests support homoskedasticity assumption"
        elif n_reject == n_total:
            conclusion = ("All tests detect heteroskedasticity - "
                         "USE ROBUST STANDARD ERRORS in final model")
        else:
            conclusion = (f"Mixed results - {n_reject}/{n_total} tests detect heteroskedasticity. "
                         "Consider using robust standard errors as precaution.")

        output.append(conclusion)
        output.append("=" * 60)

        return "\n".join(output)


# Export classes
__all__ = [
    'BreuschPaganTest',
    'WhiteTest',
    'GlejserTest',
    'HeteroskedasticityTestSuite'
]
