"""
Panel Data Model Selection Tests

Implements statistical tests for choosing between panel data models:
- F-test (Chow test): Tests Fixed Effects vs Pooled OLS
- Breusch-Pagan LM test: Tests Random Effects vs Pooled OLS
- Hausman test: Tests Fixed Effects vs Random Effects (in panel_regression.py)

These tests help determine which panel model specification is appropriate.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects


class FTestPoolability:
    """
    F-test (Chow test) for Fixed Effects vs Pooled OLS

    Null Hypothesis (H0): All entity fixed effects are zero (use Pooled OLS)
    Alternative (H1): At least one entity effect is non-zero (use Fixed Effects)

    Test Statistic:
        F = [(SSR_pooled - SSR_fe) / (N - 1)] / [SSR_fe / (NT - N - K)]

    Where:
        - SSR_pooled = Sum of squared residuals from Pooled OLS
        - SSR_fe = Sum of squared residuals from Fixed Effects
        - N = number of entities
        - T = number of time periods
        - K = number of independent variables
        - NT = total observations

    Decision Rule:
        - If p-value < α: Reject H0 → Use Fixed Effects
        - If p-value ≥ α: Fail to reject H0 → Use Pooled OLS
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize F-test for poolability

        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        self.name = "F-test for Fixed Effects vs Pooled OLS"

    def run(
        self,
        data: pd.DataFrame,
        dependent: str,
        independent: list,
        entity_col: str,
        time_col: str
    ) -> Dict[str, Any]:
        """
        Perform F-test for poolability

        Args:
            data: Panel DataFrame
            dependent: Name of dependent variable
            independent: List of independent variable names
            entity_col: Name of entity identifier column
            time_col: Name of time identifier column

        Returns:
            Dictionary containing test results
        """
        # Set panel index
        panel_data = data.set_index([entity_col, time_col])

        # 1. Fit Pooled OLS
        y = panel_data[dependent]
        X = panel_data[independent]
        X = sm.add_constant(X)
        pooled_model = sm.OLS(y, X).fit()
        ssr_pooled = np.sum(pooled_model.resid ** 2)

        # 2. Fit Fixed Effects
        formula = f"{dependent} ~ {' + '.join(independent)} + EntityEffects"
        fe_model = PanelOLS.from_formula(formula, data=panel_data)
        fe_results = fe_model.fit(cov_type='unadjusted')
        ssr_fe = np.sum(fe_results.resids ** 2)

        # 3. Calculate F-statistic
        n_entities = data[entity_col].nunique()
        n_time = data[time_col].nunique()
        n_obs = len(data)
        k = len(independent)  # Number of independent variables

        # Degrees of freedom
        df1 = n_entities - 1  # Entity fixed effects
        df2 = n_obs - n_entities - k  # Residual df for FE model

        # F-statistic
        numerator = (ssr_pooled - ssr_fe) / df1
        denominator = ssr_fe / df2
        f_stat = numerator / denominator

        # P-value
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        # Decision
        reject_null = p_value < self.alpha

        if reject_null:
            recommendation = "Fixed Effects"
            interpretation = (
                f"p-value = {p_value:.4f} < {self.alpha}. "
                f"Reject null hypothesis. "
                f"Entity fixed effects are jointly significant. "
                f"Use FIXED EFFECTS model instead of Pooled OLS."
            )
        else:
            recommendation = "Pooled OLS"
            interpretation = (
                f"p-value = {p_value:.4f} ≥ {self.alpha}. "
                f"Fail to reject null hypothesis. "
                f"Entity fixed effects are not significant. "
                f"POOLED OLS is sufficient."
            )

        result = {
            'test_name': self.name,
            'statistic': float(f_stat),
            'p_value': float(p_value),
            'df1': int(df1),
            'df2': int(df2),
            'alpha': self.alpha,
            'reject_null': reject_null,
            'recommendation': recommendation,
            'interpretation': interpretation,
            'ssr_pooled': float(ssr_pooled),
            'ssr_fe': float(ssr_fe),
            'n_entities': int(n_entities),
            'n_obs': int(n_obs)
        }

        return result


class BreuschPaganLMTest:
    """
    Breusch-Pagan Lagrange Multiplier (LM) Test for Random Effects vs Pooled OLS

    Null Hypothesis (H0): Variance of random effects = 0 (use Pooled OLS)
    Alternative (H1): Variance of random effects ≠ 0 (use Random Effects)

    This is DIFFERENT from the Breusch-Pagan test for heteroskedasticity.

    Test Statistic:
        LM = [NT / (2(T-1))] * [(Σᵢ(Σₜεᵢₜ)² / Σᵢₜεᵢₜ²) - 1]²

    Where:
        - εᵢₜ = residuals from Pooled OLS
        - N = number of entities
        - T = number of time periods

    Under H0, LM ~ χ²(1)

    Decision Rule:
        - If p-value < α: Reject H0 → Use Random Effects
        - If p-value ≥ α: Fail to reject H0 → Use Pooled OLS
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize Breusch-Pagan LM test

        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        self.name = "Breusch-Pagan LM Test for Random Effects vs Pooled OLS"

    def run(
        self,
        data: pd.DataFrame,
        dependent: str,
        independent: list,
        entity_col: str,
        time_col: str
    ) -> Dict[str, Any]:
        """
        Perform Breusch-Pagan LM test for random effects

        Args:
            data: Panel DataFrame
            dependent: Name of dependent variable
            independent: List of independent variable names
            entity_col: Name of entity identifier column
            time_col: Name of time identifier column

        Returns:
            Dictionary containing test results
        """
        # 1. Fit Pooled OLS to get residuals
        y = data[dependent]
        X = data[independent]
        X = sm.add_constant(X)
        pooled_model = sm.OLS(y, X).fit()
        residuals = pooled_model.resid

        # Add residuals to data
        data_with_resid = data.copy()
        data_with_resid['resid'] = residuals

        # 2. Calculate LM statistic
        n_entities = data[entity_col].nunique()
        n_time_avg = len(data) / n_entities  # Average time periods per entity

        # Group residuals by entity
        entity_resid_sum = data_with_resid.groupby(entity_col)['resid'].sum()
        entity_resid_sum_sq = (entity_resid_sum ** 2).sum()

        total_resid_sq = (residuals ** 2).sum()

        # LM statistic
        # LM = [NT / 2(T-1)] * [(Σᵢ(Σₜεᵢₜ)² / Σᵢₜεᵢₜ²) - 1]²
        ratio = entity_resid_sum_sq / total_resid_sq
        lm_stat = (len(data) / (2 * (n_time_avg - 1))) * ((ratio - 1) ** 2)

        # P-value from chi-square distribution with df=1
        p_value = 1 - stats.chi2.cdf(lm_stat, df=1)

        # Decision
        reject_null = p_value < self.alpha

        if reject_null:
            recommendation = "Random Effects"
            interpretation = (
                f"p-value = {p_value:.4f} < {self.alpha}. "
                f"Reject null hypothesis. "
                f"Significant random effects detected. "
                f"Use RANDOM EFFECTS model instead of Pooled OLS."
            )
        else:
            recommendation = "Pooled OLS"
            interpretation = (
                f"p-value = {p_value:.4f} ≥ {self.alpha}. "
                f"Fail to reject null hypothesis. "
                f"No significant random effects. "
                f"POOLED OLS is sufficient."
            )

        result = {
            'test_name': self.name,
            'statistic': float(lm_stat),
            'p_value': float(p_value),
            'df': 1,
            'alpha': self.alpha,
            'reject_null': reject_null,
            'recommendation': recommendation,
            'interpretation': interpretation,
            'n_entities': int(n_entities),
            'n_obs': len(data),
            'avg_time_periods': float(n_time_avg)
        }

        return result


class PanelModelSelectionTests:
    """
    Comprehensive panel model selection test suite

    Runs all three key tests for panel data model selection:
    1. F-test: Fixed Effects vs Pooled OLS
    2. Breusch-Pagan LM: Random Effects vs Pooled OLS
    3. Hausman: Fixed Effects vs Random Effects

    Usage:
        panel_tests = PanelModelSelectionTests(alpha=0.05)

        results = panel_tests.run_all(
            data=df,
            dependent='AQMS',
            independent=['ARL', 'FEE', 'FO', 'SIZE', 'ROA'],
            entity_col='CODE',
            time_col='YEAR'
        )

        # Print recommendations
        print(results['summary'])
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize panel model selection tests

        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        self.f_test = FTestPoolability(alpha)
        self.bp_lm_test = BreuschPaganLMTest(alpha)

    def run_all(
        self,
        data: pd.DataFrame,
        dependent: str,
        independent: list,
        entity_col: str,
        time_col: str
    ) -> Dict[str, Any]:
        """
        Run all panel model selection tests

        Args:
            data: Panel DataFrame
            dependent: Name of dependent variable
            independent: List of independent variable names
            entity_col: Name of entity identifier column
            time_col: Name of time identifier column

        Returns:
            Dictionary with all test results and summary
        """
        print("\n" + "=" * 80)
        print("PANEL MODEL SELECTION TESTS")
        print("=" * 80)

        # Run tests
        results = {}

        # 1. F-test (FE vs Pooled)
        print("\n1. F-TEST: Fixed Effects vs Pooled OLS")
        print("-" * 80)
        f_result = self.f_test.run(data, dependent, independent, entity_col, time_col)
        results['f_test'] = f_result
        self._print_test_result(f_result)

        # 2. BP-LM test (RE vs Pooled)
        print("\n2. BREUSCH-PAGAN LM TEST: Random Effects vs Pooled OLS")
        print("-" * 80)
        bp_result = self.bp_lm_test.run(data, dependent, independent, entity_col, time_col)
        results['bp_lm_test'] = bp_result
        self._print_test_result(bp_result)

        # 3. Summary and recommendation
        print("\n" + "=" * 80)
        print("SUMMARY OF MODEL SELECTION TESTS")
        print("=" * 80)

        summary = self._generate_summary(results)
        results['summary'] = summary
        print(summary)

        return results

    def _print_test_result(self, result: Dict[str, Any]):
        """Print formatted test result"""
        print(f"\nTest: {result['test_name']}")
        print(f"Statistic: {result['statistic']:.4f}")
        print(f"P-value: {result['p_value']:.4f}")
        if 'df1' in result:
            print(f"Degrees of Freedom: ({result['df1']}, {result['df2']})")
        else:
            print(f"Degrees of Freedom: {result['df']}")
        print(f"\nDecision: {'Reject H0' if result['reject_null'] else 'Fail to reject H0'}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"\n{result['interpretation']}")

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate overall summary and recommendation"""
        f_rec = results['f_test']['recommendation']
        bp_rec = results['bp_lm_test']['recommendation']

        output = []
        output.append("\n📊 Test Results Summary:")
        output.append(f"   • F-test (FE vs Pooled):      {f_rec}")
        output.append(f"   • BP-LM test (RE vs Pooled):  {bp_rec}")

        output.append("\n🔍 Interpretation:")

        # Determine overall recommendation
        if f_rec == "Fixed Effects" and bp_rec == "Random Effects":
            output.append("   Both FE and RE are preferred over Pooled OLS.")
            output.append("   ➜ Panel structure matters! Use panel methods.")
            output.append("   ➜ Next: Run HAUSMAN TEST to choose between FE and RE.")
            overall_rec = "Run Hausman test to choose FE or RE"
        elif f_rec == "Fixed Effects" and bp_rec == "Pooled OLS":
            output.append("   Fixed effects significant, but random effects not.")
            output.append("   ➜ Use FIXED EFFECTS model.")
            overall_rec = "Fixed Effects"
        elif f_rec == "Pooled OLS" and bp_rec == "Random Effects":
            output.append("   Random effects significant, but fixed effects not.")
            output.append("   ➜ Use RANDOM EFFECTS model.")
            overall_rec = "Random Effects"
        else:  # Both recommend Pooled
            output.append("   Neither FE nor RE are significant.")
            output.append("   ➜ Panel structure may not matter. POOLED OLS is sufficient.")
            overall_rec = "Pooled OLS"

        output.append(f"\n✅ OVERALL RECOMMENDATION: {overall_rec}")
        output.append("=" * 80)

        return "\n".join(output)


__all__ = ['FTestPoolability', 'BreuschPaganLMTest', 'PanelModelSelectionTests']
