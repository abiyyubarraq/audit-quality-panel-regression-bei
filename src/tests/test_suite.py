"""
Comprehensive Test Suite

Orchestrates all diagnostic tests for panel data regression analysis.
Provides a one-stop interface for running all classical assumption tests.

Usage:
    from src.tests.test_suite import ComprehensiveTestSuite

    # Initialize suite
    suite = ComprehensiveTestSuite(alpha=0.05)

    # Run all diagnostic tests
    results = suite.run_all_diagnostics(
        data=data,
        model_results=model_results,
        dependent_var='AQMS',
        independent_vars=['ARL', 'FEE', 'FO', 'SIZE', 'ROA']
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from .normality import NormalityTestSuite
from .heteroskedasticity import HeteroskedasticityTestSuite
from .autocorrelation import AutocorrelationTestSuite
from .multicollinearity import VIFTest


class ComprehensiveTestSuite:
    """
    Run all diagnostic tests with sensible defaults

    Default configuration:
    - Normality: Shapiro-Wilk (primary), K-S, Jarque-Bera
    - Heteroskedasticity: Breusch-Pagan (primary), White, Glejser
    - Autocorrelation: Durbin-Watson + Breusch-Godfrey
    - Multicollinearity: VIF with threshold=10

    This class provides a unified interface for running all tests needed
    for validating regression assumptions in panel data analysis.
    """

    def __init__(self, alpha: float = 0.05, vif_threshold: float = 10.0):
        """
        Initialize comprehensive test suite

        Args:
            alpha: Significance level for hypothesis tests (default: 0.05)
            vif_threshold: VIF threshold for multicollinearity (default: 10.0)
        """
        self.alpha = alpha
        self.vif_threshold = vif_threshold

        # Initialize test suites
        self.normality_suite = NormalityTestSuite(alpha)
        self.heteroskedasticity_suite = HeteroskedasticityTestSuite(alpha)
        self.autocorrelation_suite = AutocorrelationTestSuite(alpha)
        self.vif_test = VIFTest(threshold=vif_threshold, alpha=alpha)

    def test_normality(
        self,
        data: pd.Series,
        method: str = 'shapiro'
    ) -> Dict[str, Any]:
        """
        Run a specific normality test

        Args:
            data: Series of values to test for normality (e.g., residuals)
            method: Test method to use. Options:
                   - 'shapiro': Shapiro-Wilk test (default)
                   - 'ks': Kolmogorov-Smirnov test
                   - 'jarque_bera': Jarque-Bera test

        Returns:
            Dictionary containing:
            - 'result': TestResult object
            - 'formatted_output': Formatted string output

        Example:
            >>> result = suite.test_normality(residuals, method='shapiro')
            >>> print(result['formatted_output'])
        """
        if method not in self.normality_suite.tests:
            raise ValueError(
                f"Unknown normality test method '{method}'. "
                f"Valid options: {list(self.normality_suite.tests.keys())}"
            )

        # Run the specific test
        test_result = self.normality_suite.tests[method].run(data)

        # Format output
        formatted_output = str(test_result)

        return {
            'result': test_result,
            'formatted_output': formatted_output
        }

    def test_heteroskedasticity(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        method: str = 'breusch_pagan'
    ) -> Dict[str, Any]:
        """
        Run a specific heteroskedasticity test

        Args:
            y: Dependent variable (Series)
            X: Independent variables (DataFrame, should include constant)
            method: Test method to use. Options:
                   - 'breusch_pagan': Breusch-Pagan test (default)
                   - 'white': White's test
                   - 'glejser': Glejser test

        Returns:
            Dictionary containing:
            - 'statistic': Test statistic value
            - 'p_value': P-value
            - 'reject_null': Boolean indicating if null hypothesis is rejected
            - 'interpretation': Text interpretation
            - 'formatted_output': Formatted string output

        Example:
            >>> result = suite.test_heteroskedasticity(y, X, method='breusch_pagan')
            >>> print(result['formatted_output'])
        """
        if method not in self.heteroskedasticity_suite.tests:
            raise ValueError(
                f"Unknown heteroskedasticity test method '{method}'. "
                f"Valid options: {list(self.heteroskedasticity_suite.tests.keys())}"
            )

        # Fit OLS model to get model_results
        import statsmodels.api as sm
        model = sm.OLS(y, X)
        model_results = model.fit()

        # Run the specific test
        test_result = self.heteroskedasticity_suite.tests[method].run(model_results)

        # Format output
        formatted_output = str(test_result)

        return {
            'statistic': test_result.statistic,
            'p_value': test_result.p_value,
            'reject_null': test_result.reject_null,
            'interpretation': test_result.interpretation,
            'result': test_result,
            'formatted_output': formatted_output
        }

    def test_autocorrelation(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        method: str = 'durbin_watson',
        nlags: int = 1
    ) -> Dict[str, Any]:
        """
        Run a specific autocorrelation test

        Args:
            y: Dependent variable (Series)
            X: Independent variables (DataFrame, should include constant)
            method: Test method to use. Options:
                   - 'durbin_watson': Durbin-Watson test (default)
                   - 'breusch_godfrey': Breusch-Godfrey test
            nlags: Number of lags for Breusch-Godfrey test (default: 1)

        Returns:
            Dictionary containing:
            - 'statistic': Test statistic value
            - 'p_value': P-value (None for Durbin-Watson)
            - 'reject_null': Boolean indicating if null hypothesis is rejected
            - 'interpretation': Text interpretation
            - 'formatted_output': Formatted string output

        Example:
            >>> result = suite.test_autocorrelation(y, X, method='durbin_watson')
            >>> print(result['formatted_output'])
        """
        valid_methods = ['durbin_watson', 'breusch_godfrey']
        if method not in valid_methods:
            raise ValueError(
                f"Unknown autocorrelation test method '{method}'. "
                f"Valid options: {valid_methods}"
            )

        # Fit OLS model to get model_results
        import statsmodels.api as sm
        model = sm.OLS(y, X)
        model_results = model.fit()

        # Run the specific test
        if method == 'durbin_watson':
            test_result = self.autocorrelation_suite.dw_test.run(model_results)
        elif method == 'breusch_godfrey':
            test_result = self.autocorrelation_suite.bg_test.run(model_results, nlags=nlags)

        # Format output
        formatted_output = str(test_result)

        return {
            'statistic': test_result.statistic,
            'p_value': test_result.p_value,
            'reject_null': test_result.reject_null,
            'interpretation': test_result.interpretation,
            'result': test_result,
            'formatted_output': formatted_output
        }

    def test_multicollinearity(
        self,
        data: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run VIF test for multicollinearity

        Args:
            data: DataFrame containing independent variables (without constant)
            threshold: VIF threshold for detecting problems (default: uses self.vif_threshold)

        Returns:
            Dictionary containing:
            - 'vif_values': List of dictionaries with variable names and VIF values
            - 'max_vif': Maximum VIF value
            - 'mean_vif': Mean VIF value
            - 'has_issues': Boolean indicating if any variable exceeds threshold
            - 'formatted_output': Formatted string output

        Example:
            >>> result = suite.test_multicollinearity(data[['ARL', 'FEE', 'FO']])
            >>> print(result['formatted_output'])
        """
        if threshold is None:
            threshold = self.vif_threshold

        # Get variable names from DataFrame columns
        exog_vars = list(data.columns)

        # Run VIF test
        vif_results = self.vif_test.run(data, exog_vars)

        # Extract VIF values for summary
        vif_values = []
        for var, result in vif_results.items():
            vif_values.append({
                'Variable': var,
                'VIF': result.statistic if not np.isinf(result.statistic) else float('inf'),
                'Status': 'OK' if result.statistic < 5 else ('Monitor' if result.statistic < 10 else 'Problematic')
            })

        # Calculate statistics
        finite_vifs = [v['VIF'] for v in vif_values if not np.isinf(v['VIF'])]
        max_vif = max(finite_vifs) if finite_vifs else float('inf')
        mean_vif = np.mean(finite_vifs) if finite_vifs else float('inf')
        has_issues = any(v['VIF'] >= threshold for v in vif_values)

        # Format output
        summary_df = self.vif_test.summarize(vif_results)
        overall_assessment = self.vif_test.check_overall_multicollinearity(vif_results)
        formatted_output = f"{summary_df.to_string(index=False)}\n{overall_assessment}"

        return {
            'vif_values': vif_values,
            'vif_results': vif_results,
            'max_vif': max_vif,
            'mean_vif': mean_vif,
            'has_issues': has_issues,
            'threshold': threshold,
            'formatted_output': formatted_output
        }

    def run_all_diagnostics(
        self,
        data: pd.DataFrame,
        model_results,
        dependent_var: str,
        independent_vars: List[str],
        print_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete diagnostic test suite

        Args:
            data: DataFrame containing all variables
            model_results: Fitted regression model results (from statsmodels/linearmodels)
            dependent_var: Name of dependent variable (e.g., 'AQMS')
            independent_vars: List of independent variable names
            print_results: Whether to print results to console (default: True)

        Returns:
            Dictionary containing all test results:
            {
                'normality': {...},
                'heteroskedasticity': {...},
                'autocorrelation': {...},
                'multicollinearity': {...}
            }
        """
        results = {
            'normality': {},
            'heteroskedasticity': {},
            'autocorrelation': {},
            'multicollinearity': {}
        }

        # 1. Normality Tests (on residuals)
        if print_results:
            print("\n" + "=" * 60)
            print("1. NORMALITY TESTS (Residuals)")
            print("=" * 60)

        try:
            residuals = pd.Series(model_results.resid, name='residuals')
            results['normality'] = self.normality_suite.run_all(residuals)

            if print_results:
                summary = self.normality_suite.summarize(results['normality'])
                print(summary)

        except Exception as e:
            print(f"Error running normality tests: {str(e)}")
            results['normality'] = {'error': str(e)}

        # 2. Heteroskedasticity Tests
        if print_results:
            print("\n" + "=" * 60)
            print("2. HETEROSKEDASTICITY TESTS")
            print("=" * 60)

        try:
            results['heteroskedasticity'] = self.heteroskedasticity_suite.run_all(model_results)

            if print_results:
                summary = self.heteroskedasticity_suite.summarize(results['heteroskedasticity'])
                print(summary)

        except Exception as e:
            print(f"Error running heteroskedasticity tests: {str(e)}")
            results['heteroskedasticity'] = {'error': str(e)}

        # 3. Autocorrelation Tests
        if print_results:
            print("\n" + "=" * 60)
            print("3. AUTOCORRELATION TESTS")
            print("=" * 60)

        try:
            results['autocorrelation'] = self.autocorrelation_suite.run_all(model_results)

            if print_results:
                summary = self.autocorrelation_suite.summarize(results['autocorrelation'])
                print(summary)

        except Exception as e:
            print(f"Error running autocorrelation tests: {str(e)}")
            results['autocorrelation'] = {'error': str(e)}

        # 4. Multicollinearity Tests (VIF)
        if print_results:
            print("\n" + "=" * 60)
            print("4. MULTICOLLINEARITY TESTS (VIF)")
            print("=" * 60)

        try:
            results['multicollinearity'] = self.vif_test.run(data, independent_vars)

            if print_results:
                vif_summary_df = self.vif_test.summarize(results['multicollinearity'])
                print("\n" + vif_summary_df.to_string(index=False))
                print(self.vif_test.check_overall_multicollinearity(results['multicollinearity']))

        except Exception as e:
            print(f"Error running multicollinearity tests: {str(e)}")
            results['multicollinearity'] = {'error': str(e)}

        # Overall Summary
        if print_results:
            print("\n" + "=" * 60)
            print("DIAGNOSTIC TEST SUITE COMPLETED")
            print("=" * 60)
            print(self._generate_overall_summary(results))

        return results

    def _generate_overall_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate overall summary of all diagnostic tests

        Args:
            results: Dictionary of all test results

        Returns:
            Formatted string with overall assessment
        """
        output = []
        output.append("\nOVERALL DIAGNOSTIC SUMMARY:")
        output.append("-" * 60)

        # Check each test category
        issues = []

        # Normality
        if 'error' not in results['normality']:
            norm_results = [r for r in results['normality'].values() if r is not None]
            n_reject_norm = sum(1 for r in norm_results if r.reject_null)
            if n_reject_norm > len(norm_results) / 2:
                issues.append("⚠ Residuals may not be normally distributed")
            else:
                output.append("✓ Normality: Residuals appear normally distributed")

        # Heteroskedasticity
        if 'error' not in results['heteroskedasticity']:
            hetero_results = [r for r in results['heteroskedasticity'].values() if r is not None]
            n_reject_hetero = sum(1 for r in hetero_results if r.reject_null)
            if n_reject_hetero > 0:
                issues.append("✗ Heteroskedasticity detected - USE ROBUST STANDARD ERRORS")
            else:
                output.append("✓ Heteroskedasticity: Homoskedasticity assumption holds")

        # Autocorrelation
        if 'error' not in results['autocorrelation']:
            autocorr_results = [r for r in results['autocorrelation'].values() if r is not None]
            n_reject_autocorr = sum(1 for r in autocorr_results if r.reject_null)
            if n_reject_autocorr > 0:
                issues.append("✗ Autocorrelation detected - Consider clustered standard errors")
            else:
                output.append("✓ Autocorrelation: No autocorrelation detected")

        # Multicollinearity
        if 'error' not in results['multicollinearity']:
            vif_values = [r.statistic for r in results['multicollinearity'].values()
                         if r is not None and r.statistic != float('inf')]
            if vif_values:
                max_vif = max(vif_values)
                if max_vif >= 10:
                    issues.append(f"✗ High multicollinearity detected (max VIF = {max_vif:.2f})")
                elif max_vif >= 5:
                    issues.append(f"⚠ Moderate multicollinearity detected (max VIF = {max_vif:.2f})")
                else:
                    output.append("✓ Multicollinearity: No multicollinearity issues")

        # Print issues
        if issues:
            output.append("\nISSUES REQUIRING ATTENTION:")
            for issue in issues:
                output.append(f"  {issue}")

        if not issues:
            output.append("\n✓ All diagnostic tests passed. Regression assumptions appear valid.")
        else:
            output.append("\n⚠ Some diagnostic tests indicate violations. Review results above.")

        output.append("-" * 60)

        return "\n".join(output)

    def export_results(self, results: Dict[str, Any], output_dir: str = 'results/tables'):
        """
        Export test results to files

        Args:
            results: Dictionary of test results
            output_dir: Directory to save results (default: 'results/tables')
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Export VIF results to CSV
        if 'multicollinearity' in results and 'error' not in results['multicollinearity']:
            vif_df = self.vif_test.summarize(results['multicollinearity'])
            vif_df.to_csv(f"{output_dir}/vif_results.csv", index=False)
            print(f"VIF results exported to {output_dir}/vif_results.csv")

        # Export all results to JSON
        # Convert TestResult objects to dictionaries
        exportable_results = {}
        for category, category_results in results.items():
            if isinstance(category_results, dict) and 'error' not in category_results:
                exportable_results[category] = {}
                for test_name, test_result in category_results.items():
                    if test_result is not None and hasattr(test_result, 'to_dict'):
                        exportable_results[category][test_name] = test_result.to_dict()
                    else:
                        exportable_results[category][test_name] = str(test_result)

        with open(f"{output_dir}/diagnostic_tests_summary.json", 'w') as f:
            json.dump(exportable_results, f, indent=2)
        print(f"All results exported to {output_dir}/diagnostic_tests_summary.json")


# Export classes
__all__ = ['ComprehensiveTestSuite']
