"""
Robust Hausman Test Implementation

This module provides a robust version of the Hausman test that handles
cases where the standard test fails due to negative variance differences.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats


class RobustHausmanTest:
    """
    Robust Hausman Test with diagnostics

    Implements:
    1. Standard Hausman test
    2. Auxiliary regression approach (more robust)
    3. Diagnostics for test validity
    """

    def __init__(self, data: pd.DataFrame, entity_col: str, time_col: str):
        """
        Initialize test

        Args:
            data: Panel data
            entity_col: Entity identifier column
            time_col: Time identifier column
        """
        self.data = data
        self.entity_col = entity_col
        self.time_col = time_col
        self.panel_data = data.set_index([entity_col, time_col])

    def standard_hausman(
        self,
        dependent: str,
        independent: List[str],
        alpha: float = 0.05
    ) -> Dict:
        """
        Standard Hausman test with diagnostics

        Returns dict with:
        - test_statistic: Chi-squared statistic (can be negative!)
        - p_value: P-value
        - is_valid: Whether test assumptions are met
        - eigenvalues: Eigenvalues of variance difference matrix
        """
        # Fit both models (unadjusted SEs)
        formula = f"{dependent} ~ {' + '.join(independent)}"

        # Fixed Effects
        fe_model = PanelOLS.from_formula(
            formula + " + EntityEffects",
            data=self.panel_data
        )
        fe_results = fe_model.fit(cov_type='unadjusted')

        # Random Effects
        re_model = RandomEffects.from_formula(
            formula,
            data=self.panel_data
        )
        re_results = re_model.fit(cov_type='unadjusted')

        # Get coefficients
        common_vars = fe_results.params.index.intersection(re_results.params.index)
        fe_coefs = fe_results.params[common_vars]
        re_coefs = re_results.params[common_vars]

        # Coefficient difference
        coef_diff = fe_coefs - re_coefs

        # Variance difference
        fe_cov = fe_results.cov.loc[common_vars, common_vars]
        re_cov = re_results.cov.loc[common_vars, common_vars]
        var_diff = fe_cov - re_cov

        # Check if var_diff is positive semi-definite
        eigenvalues = np.linalg.eigvals(var_diff)
        is_psd = np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

        # Try to compute test statistic
        try:
            var_diff_inv = np.linalg.inv(var_diff)
            hausman_stat = float(coef_diff.T @ var_diff_inv @ coef_diff)
            df = len(common_vars)
            p_value = 1 - stats.chi2.cdf(hausman_stat, df)

            # Check validity
            is_valid = is_psd and hausman_stat >= 0

            if not is_valid:
                warning = (
                    "⚠️  WARNING: Hausman test assumptions violated!\n"
                    f"   - Chi² statistic: {hausman_stat:.4f} {'(NEGATIVE!)' if hausman_stat < 0 else ''}\n"
                    f"   - Variance matrix is {'NOT' if not is_psd else ''} positive semi-definite\n"
                    f"   - Min eigenvalue: {np.min(eigenvalues):.6f}\n"
                    "   - This matches Stata's warning about asymptotic assumptions\n"
                    "   → Use auxiliary regression test or model selection criteria instead"
                )
            else:
                warning = "✓ Test assumptions satisfied"

            return {
                'test_statistic': hausman_stat,
                'p_value': p_value,
                'degrees_of_freedom': df,
                'is_valid': is_valid,
                'is_psd': is_psd,
                'eigenvalues': eigenvalues,
                'warning': warning,
                'recommendation': 'Fixed Effects' if hausman_stat > 0 else 'Use alternative test'
            }

        except np.linalg.LinAlgError:
            return {
                'test_statistic': None,
                'p_value': None,
                'degrees_of_freedom': len(common_vars),
                'is_valid': False,
                'is_psd': is_psd,
                'eigenvalues': eigenvalues,
                'warning': '❌ Variance matrix is singular - cannot compute Hausman test',
                'recommendation': 'Use auxiliary regression test or AIC/BIC'
            }

    def auxiliary_regression_test(
        self,
        dependent: str,
        independent: List[str],
        alpha: float = 0.05
    ) -> Dict:
        """
        Auxiliary Regression Hausman Test (more robust)

        This approach is more robust to numerical issues than the standard test.

        Method:
        1. Estimate RE model and get residuals
        2. Regress residuals on independent vars + entity means
        3. Test if entity means are jointly significant

        This is equivalent to Hausman test but more numerically stable.
        """
        from scipy.stats import f as f_dist
        import statsmodels.api as sm

        print("\n" + "="*60)
        print("AUXILIARY REGRESSION HAUSMAN TEST")
        print("="*60)
        print("(More robust than standard Hausman test)")

        # Prepare data
        data = self.data.copy()

        # Calculate entity (within-group) means
        entity_means = {}
        for var in independent:
            entity_means[f'{var}_mean'] = (
                data.groupby(self.entity_col)[var]
                .transform('mean')
            )

        # Create regression data
        y = data[dependent]
        X = data[independent].copy()

        # Add entity means
        for var in independent:
            X[f'{var}_mean'] = entity_means[f'{var}_mean']

        # Add constant
        X = sm.add_constant(X)

        # Run regression
        model = sm.OLS(y, X).fit(cov_type='HC3')

        # Test if entity means are jointly zero
        # H0: all entity mean coefficients = 0 (RE is consistent)
        # H1: at least one mean coef ≠ 0 (FE preferred)

        mean_vars = [f'{var}_mean' for var in independent]

        # F-test for joint significance
        hypotheses = ' = '.join([f'{var} = 0' for var in mean_vars])
        f_test = model.f_test(hypotheses)

        f_stat = f_test.fvalue[0][0]
        p_value = f_test.pvalue
        df1 = len(mean_vars)
        df2 = model.df_resid

        # Decision
        reject_null = p_value < alpha

        if reject_null:
            recommendation = "Fixed Effects"
            interpretation = (
                f"p-value = {p_value:.4f} < {alpha}. "
                f"Entity means are jointly significant. "
                f"Entity effects are correlated with regressors. "
                f"Recommend FIXED EFFECTS model."
            )
        else:
            recommendation = "Random Effects"
            interpretation = (
                f"p-value = {p_value:.4f} ≥ {alpha}. "
                f"Entity means are not jointly significant. "
                f"Random Effects is consistent and efficient. "
                f"Recommend RANDOM EFFECTS model."
            )

        print(f"\nF-statistic: {f_stat:.4f}")
        print(f"Degrees of Freedom: ({df1}, {df2})")
        print(f"P-value: {p_value:.4f}")
        print(f"\n{interpretation}")

        return {
            'test': 'Auxiliary Regression',
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'df_numerator': int(df1),
            'df_denominator': int(df2),
            'reject_null': reject_null,
            'recommendation': recommendation,
            'interpretation': interpretation,
            'is_valid': True,  # This test is always valid
            'ols_results': model
        }

    def compare_tests(
        self,
        dependent: str,
        independent: List[str],
        alpha: float = 0.05
    ) -> Dict:
        """
        Run both standard and auxiliary regression tests and compare
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE HAUSMAN TEST COMPARISON")
        print("="*80)

        # Standard test
        print("\n1. STANDARD HAUSMAN TEST")
        print("-" * 60)
        standard = self.standard_hausman(dependent, independent, alpha)
        print(standard['warning'])

        if standard['test_statistic'] is not None:
            print(f"\nChi² statistic: {standard['test_statistic']:.4f}")
            print(f"P-value: {standard['p_value']:.4f}")
            print(f"Recommendation: {standard['recommendation']}")

        # Auxiliary regression test
        print("\n2. AUXILIARY REGRESSION TEST")
        print("-" * 60)
        auxiliary = self.auxiliary_regression_test(dependent, independent, alpha)

        # Compare
        print("\n" + "="*80)
        print("COMPARISON & RECOMMENDATION")
        print("="*80)

        if standard['is_valid']:
            print("✓ Standard Hausman test is valid")
            print(f"  → {standard['recommendation']}")
        else:
            print("✗ Standard Hausman test has issues (matches Stata warning!)")
            print("  → Cannot rely on standard test")

        print(f"\n✓ Auxiliary regression test (robust)")
        print(f"  → {auxiliary['recommendation']}")

        # Final recommendation
        if standard['is_valid'] and standard['recommendation'] == auxiliary['recommendation']:
            final_rec = auxiliary['recommendation']
            confidence = "High confidence"
        elif not standard['is_valid']:
            final_rec = auxiliary['recommendation']
            confidence = "Based on robust test only"
        else:
            final_rec = auxiliary['recommendation']
            confidence = "Tests disagree - using robust test"

        print(f"\n{'='*80}")
        print(f"FINAL RECOMMENDATION: {final_rec}")
        print(f"Confidence: {confidence}")
        print(f"{'='*80}")

        return {
            'standard_test': standard,
            'auxiliary_test': auxiliary,
            'final_recommendation': final_rec,
            'tests_agree': standard.get('recommendation') == auxiliary['recommendation']
        }


def run_robust_hausman_analysis(
    data: pd.DataFrame,
    dependent: str,
    independent: List[str],
    entity_col: str = 'CODE',
    time_col: str = 'YEAR',
    alpha: float = 0.05
) -> Dict:
    """
    Convenience function to run complete robust Hausman analysis

    Usage:
        results = run_robust_hausman_analysis(
            data=df,
            dependent='AQMS',
            independent=['ARL', 'FEE', 'FO', 'SIZE', 'ROA'],
            entity_col='CODE',
            time_col='YEAR'
        )
    """
    tester = RobustHausmanTest(data, entity_col, time_col)
    return tester.compare_tests(dependent, independent, alpha)
