"""
Panel Data Regression Models

Implements Fixed Effects, Random Effects, and Hausman test for panel data analysis.
Uses linearmodels library for robust panel data estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel.results import PanelResults, PanelEffectsResults
import statsmodels.api as sm
from scipy import stats


class PanelDataAnalysis:
    """
    Panel Data Regression Analysis

    Supports:
    - Fixed Effects (FE) Model
    - Random Effects (RE) Model
    - Hausman Test for model selection
    - Model comparison

    Usage:
        panel_model = PanelDataAnalysis(data, entity_col='company_id', time_col='year')

        # Fixed Effects
        fe_results = panel_model.fixed_effects(
            dependent='AQMS',
            independent=['ARL', 'FEE', 'FO', 'SIZE', 'ROA']
        )

        # Random Effects
        re_results = panel_model.random_effects(
            dependent='AQMS',
            independent=['ARL', 'FEE', 'FO', 'SIZE', 'ROA']
        )

        # Hausman Test
        hausman = panel_model.hausman_test(
            dependent='AQMS',
            independent=['ARL', 'FEE', 'FO', 'SIZE', 'ROA']
        )
    """

    def __init__(
        self,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str
    ):
        """
        Initialize panel data analysis

        Args:
            data: Panel dataset with entity and time identifiers
            entity_col: Name of entity identifier column (e.g., 'company_id')
            time_col: Name of time identifier column (e.g., 'year')
        """
        self.data = data.copy()
        self.entity_col = entity_col
        self.time_col = time_col

        # Validate panel structure
        if entity_col not in data.columns:
            raise ValueError(f"Entity column '{entity_col}' not found")
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found")

        # Set multi-index for panel data
        self.panel_data = data.set_index([entity_col, time_col])

        print(f"Panel data initialized:")
        print(f"  Entities: {data[entity_col].nunique()}")
        print(f"  Time periods: {data[time_col].nunique()}")
        print(f"  Total observations: {len(data)}")

    def fixed_effects(
        self,
        dependent: str,
        independent: List[str],
        entity_effects: bool = True,
        time_effects: bool = False,
        robust: bool = True,
        cluster_entity: bool = False
    ) -> PanelEffectsResults:
        """
        Fixed Effects (FE) Model

        Controls for time-invariant unobserved heterogeneity at entity level.

        Advantages:
        - Controls for unobserved entity-specific effects
        - No need to assume effects are uncorrelated with predictors
        - Most conservative approach for panel data

        Args:
            dependent: Name of dependent variable
            independent: List of independent variable names
            entity_effects: Include entity (company) fixed effects (default: True)
            time_effects: Include time (year) fixed effects (default: False)
            robust: Use heteroskedasticity-robust standard errors (default: True)
            cluster_entity: Cluster standard errors by entity (default: False)

        Returns:
            PanelEffectsResults object from linearmodels
        """
        print("\n" + "=" * 60)
        print("FIXED EFFECTS MODEL")
        print("=" * 60)

        # Create formula
        formula = f"{dependent} ~ {' + '.join(independent)}"
        print(f"\nFormula: {formula}")
        print(f"Entity effects: {entity_effects}")
        print(f"Time effects: {time_effects}")

        # Fit model
        try:
            # PanelOLS.from_formula() doesn't accept entity_effects/time_effects as parameters
            # Instead, use EntityEffects and TimeEffects keywords in the formula string

            formula_parts = [formula]
            if entity_effects:
                formula_parts.append("EntityEffects")
            if time_effects:
                formula_parts.append("TimeEffects")

            full_formula = " + ".join(formula_parts)
            print(f"Full formula with effects: {full_formula}")

            model = PanelOLS.from_formula(
                full_formula,
                data=self.panel_data
            )

            # Choose covariance type
            if cluster_entity:
                cov_type = 'clustered'
                # Clusters must have same MultiIndex as panel data
                # Use entity level from the MultiIndex
                clusters = pd.Series(
                    self.panel_data.index.get_level_values(0),
                    index=self.panel_data.index,
                    name='clusters'
                )
                cov_kwds = {'clusters': clusters}
                print(f"Standard errors: Clustered by entity")
            elif robust:
                cov_type = 'robust'
                cov_kwds = {}
                print(f"Standard errors: Heteroskedasticity-robust")
            else:
                cov_type = 'unadjusted'
                cov_kwds = {}
                print(f"Standard errors: Standard (homoskedastic)")

            results = model.fit(cov_type=cov_type, **cov_kwds)

            print("\n" + str(results))

            return results

        except Exception as e:
            raise ValueError(f"Error fitting Fixed Effects model: {str(e)}")

    def random_effects(
        self,
        dependent: str,
        independent: List[str],
        robust: bool = True
    ) -> PanelResults:
        """
        Random Effects (RE) Model

        Assumes unobserved entity effects are uncorrelated with predictors.
        More efficient than FE if assumption holds.

        Advantages:
        - More efficient estimates (smaller standard errors)
        - Can estimate time-invariant variables
        - Better for random samples from large population

        Args:
            dependent: Name of dependent variable
            independent: List of independent variable names
            robust: Use robust standard errors (default: True)

        Returns:
            PanelResults object from linearmodels
        """
        print("\n" + "=" * 60)
        print("RANDOM EFFECTS MODEL")
        print("=" * 60)

        # Create formula
        formula = f"{dependent} ~ {' + '.join(independent)}"
        print(f"\nFormula: {formula}")

        # Fit model
        try:
            model = RandomEffects.from_formula(
                formula,
                data=self.panel_data
            )

            cov_type = 'robust' if robust else 'unadjusted'
            print(f"Standard errors: {'Robust' if robust else 'Standard'}")

            results = model.fit(cov_type=cov_type)

            print("\n" + str(results))

            return results

        except Exception as e:
            raise ValueError(f"Error fitting Random Effects model: {str(e)}")

    def hausman_test(
        self,
        dependent: str,
        independent: List[str],
        alpha: float = 0.05
    ) -> Dict:
        """
        Hausman Test for FE vs RE model selection

        Null Hypothesis: RE is consistent and efficient (prefer RE)
        Alternative: RE is inconsistent, FE is consistent (prefer FE)

        Decision Rule:
        - If p-value < α: Reject H0 → Use Fixed Effects
        - If p-value > α: Fail to reject H0 → Use Random Effects

        Args:
            dependent: Name of dependent variable
            independent: List of independent variable names
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results and recommendation
        """
        print("\n" + "=" * 60)
        print("HAUSMAN TEST (FE vs RE)")
        print("=" * 60)

        try:
            # Fit both models without robust SE for Hausman test
            fe_results = self.fixed_effects(
                dependent, independent,
                entity_effects=True, time_effects=False,
                robust=False
            )

            re_results = self.random_effects(
                dependent, independent,
                robust=False
            )

            # Extract coefficients
            fe_coefs = fe_results.params
            re_coefs = re_results.params

            # Ensure same ordering
            common_vars = fe_coefs.index.intersection(re_coefs.index)
            fe_coefs = fe_coefs[common_vars]
            re_coefs = re_coefs[common_vars]

            # Calculate difference
            coef_diff = fe_coefs - re_coefs

            # Calculate variance difference
            fe_cov = fe_results.cov[common_vars].loc[common_vars]
            re_cov = re_results.cov[common_vars].loc[common_vars]
            var_diff = fe_cov - re_cov

            # Hausman statistic
            try:
                var_diff_inv = np.linalg.inv(var_diff)
                hausman_stat = coef_diff.T @ var_diff_inv @ coef_diff
                df = len(common_vars)
                p_value = 1 - stats.chi2.cdf(hausman_stat, df)

                # Decision
                reject_null = p_value < alpha

                if reject_null:
                    recommendation = "Fixed Effects"
                    interpretation = (
                        f"p-value = {p_value:.4f} < {alpha}. "
                        f"Reject null hypothesis. "
                        f"Random Effects is inconsistent. "
                        f"Recommend FIXED EFFECTS model."
                    )
                else:
                    recommendation = "Random Effects"
                    interpretation = (
                        f"p-value = {p_value:.4f} > {alpha}. "
                        f"Fail to reject null hypothesis. "
                        f"Random Effects is consistent and efficient. "
                        f"Recommend RANDOM EFFECTS model."
                    )

                result = {
                    'test_statistic': float(hausman_stat),
                    'p_value': float(p_value),
                    'degrees_of_freedom': int(df),
                    'alpha': alpha,
                    'reject_null': reject_null,
                    'recommendation': recommendation,
                    'interpretation': interpretation
                }

                # Print results
                print(f"\nHausman Test Statistic: {hausman_stat:.4f}")
                print(f"Degrees of Freedom: {df}")
                print(f"P-value: {p_value:.4f}")
                print(f"\n{interpretation}")
                print("\n" + "=" * 60)

                return result

            except np.linalg.LinAlgError:
                print("\nWarning: Variance matrix is singular. Cannot compute Hausman test.")
                print("This may indicate numerical issues or perfect multicollinearity.")
                print("Recommendation: Use Fixed Effects as a conservative choice.")

                return {
                    'test_statistic': None,
                    'p_value': None,
                    'degrees_of_freedom': None,
                    'alpha': alpha,
                    'reject_null': None,
                    'recommendation': 'Fixed Effects (by default)',
                    'interpretation': 'Hausman test failed. Use FE as conservative choice.'
                }

        except Exception as e:
            raise ValueError(f"Error running Hausman test: {str(e)}")

    def pooled_ols(
        self,
        dependent: str,
        independent: List[str],
        robust: bool = True
    ):
        """
        Pooled OLS (for comparison)

        Ignores panel structure. Useful for comparison to show why
        panel methods are superior.

        Args:
            dependent: Name of dependent variable
            independent: List of independent variable names
            robust: Use robust standard errors (default: True)

        Returns:
            OLS results from statsmodels
        """
        print("\n" + "=" * 60)
        print("POOLED OLS (Ignores Panel Structure)")
        print("=" * 60)

        # Prepare data
        data = self.panel_data.reset_index()
        y = data[dependent]
        X = data[independent]
        X = sm.add_constant(X)

        # Fit model
        model = sm.OLS(y, X)

        if robust:
            results = model.fit(cov_type='HC3')  # Robust standard errors
            print("Standard errors: Heteroskedasticity-robust (HC3)")
        else:
            results = model.fit()
            print("Standard errors: Standard")

        print("\n" + str(results.summary()))

        return results

    def compare_models(
        self,
        dependent: str,
        independent: List[str]
    ) -> pd.DataFrame:
        """
        Compare FE, RE, and Pooled OLS side-by-side

        Args:
            dependent: Name of dependent variable
            independent: List of independent variable names

        Returns:
            DataFrame with model comparison
        """
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        # Fit all models
        fe_results = self.fixed_effects(dependent, independent, robust=True)
        re_results = self.random_effects(dependent, independent, robust=True)
        ols_results = self.pooled_ols(dependent, independent, robust=True)

        # Extract coefficients and standard errors
        comparison_data = []

        for var in independent:
            row = {'Variable': var}

            # Fixed Effects
            if var in fe_results.params.index:
                row['FE_Coef'] = fe_results.params[var]
                row['FE_SE'] = fe_results.std_errors[var]
                row['FE_PValue'] = fe_results.pvalues[var]

            # Random Effects
            if var in re_results.params.index:
                row['RE_Coef'] = re_results.params[var]
                row['RE_SE'] = re_results.std_errors[var]
                row['RE_PValue'] = re_results.pvalues[var]

            # Pooled OLS
            if var in ols_results.params.index:
                row['OLS_Coef'] = ols_results.params[var]
                row['OLS_SE'] = ols_results.bse[var]
                row['OLS_PValue'] = ols_results.pvalues[var]

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Add model statistics
        stats_row = {
            'Variable': 'R-squared',
            'FE_Coef': fe_results.rsquared,
            'RE_Coef': re_results.rsquared,
            'OLS_Coef': ols_results.rsquared
        }
        comparison_df = pd.concat([comparison_df, pd.DataFrame([stats_row])], ignore_index=True)

        print("\n" + comparison_df.to_string(index=False))

        return comparison_df


class PanelRegression:
    """
    Simplified Panel Regression wrapper with clustered standard errors

    This class provides an easy-to-use interface for panel regression with
    clustered standard errors by default to handle autocorrelation in panel data.

    Usage:
        panel_reg = PanelRegression(entity_col='CODE', time_col='YEAR')

        # Fixed Effects with clustered SEs (recommended for autocorrelation)
        results = panel_reg.fit(
            data=df,
            dependent_var='AQMS',
            independent_vars=['ARL', 'FEE', 'FO', 'SIZE', 'ROA'],
            model_type='fixed_effects'
        )
    """

    def __init__(
        self,
        entity_col: str = 'CODE',
        time_col: str = 'YEAR',
        use_robust: bool = True
    ):
        """
        Initialize Panel Regression

        Args:
            entity_col: Column name for entity identifier
            time_col: Column name for time identifier
            use_robust: Use clustered standard errors (default: True)
        """
        self.entity_col = entity_col
        self.time_col = time_col
        self.use_robust = use_robust

    def fit(
        self,
        data: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str],
        model_type: str = 'fixed_effects',
        cluster: bool = True,
        entity_effects: bool = True,
        time_effects: bool = False
    ) -> Dict:
        """
        Fit panel regression model

        Args:
            data: DataFrame with panel data
            dependent_var: Name of dependent variable
            independent_vars: List of independent variable names
            model_type: 'fixed_effects' or 'random_effects'
            cluster: Use clustered standard errors by entity (default: True)
            entity_effects: Include entity fixed effects (default: True)
            time_effects: Include time fixed effects (default: False)

        Returns:
            Dictionary with model results and summary
        """
        # Initialize analysis
        panel_analysis = PanelDataAnalysis(
            data=data,
            entity_col=self.entity_col,
            time_col=self.time_col
        )

        # Fit model based on type
        if model_type == 'fixed_effects':
            results = panel_analysis.fixed_effects(
                dependent=dependent_var,
                independent=independent_vars,
                entity_effects=entity_effects,
                time_effects=time_effects,
                robust=self.use_robust and not cluster,
                cluster_entity=cluster
            )
        elif model_type == 'random_effects':
            results = panel_analysis.random_effects(
                dependent=dependent_var,
                independent=independent_vars,
                robust=self.use_robust
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'fixed_effects' or 'random_effects'")

        # Return results dictionary
        return {
            'model': results,
            'summary': str(results),
            'params': results.params,
            'std_errors': results.std_errors,
            'pvalues': results.pvalues,
            'rsquared': results.rsquared,
            'nobs': results.nobs,
            'model_type': model_type,
            'clustered': cluster
        }

    def hausman_test(
        self,
        data: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str],
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform Hausman test for FE vs RE selection

        Args:
            data: DataFrame with panel data
            dependent_var: Name of dependent variable
            independent_vars: List of independent variable names
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results
        """
        panel_analysis = PanelDataAnalysis(
            data=data,
            entity_col=self.entity_col,
            time_col=self.time_col
        )

        results = panel_analysis.hausman_test(
            dependent=dependent_var,
            independent=independent_vars,
            alpha=alpha
        )

        # Format output
        formatted_output = f"""
Hausman Test Results
{'='*60}
Test Statistic: {results['test_statistic']:.4f}
Degrees of Freedom: {results['degrees_of_freedom']}
P-value: {results['p_value']:.4f}
Alpha: {results['alpha']}

Decision: {'Reject H0' if results['reject_null'] else 'Fail to reject H0'}
Recommendation: {results['recommendation']}

Interpretation:
{results['interpretation']}
{'='*60}
        """

        results['formatted_output'] = formatted_output
        return results


__all__ = ['PanelDataAnalysis', 'PanelRegression']
