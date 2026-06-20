"""
Multicollinearity Tests

Tests for checking multicollinearity among independent variables.
High multicollinearity inflates standard errors and makes coefficient
estimates unstable.

Available tests:
- VIF (Variance Inflation Factor) - Standard measure of multicollinearity
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List
from . import StatisticalTest, TestResult


class VIFTest(StatisticalTest):
    """
    Variance Inflation Factor (VIF) Test for Multicollinearity

    VIF measures how much the variance of an estimated coefficient is inflated
    due to multicollinearity with other predictors.

    Interpretation Guidelines:
    - VIF < 5:  Low multicollinearity (generally acceptable)
    - 5 ≤ VIF < 10: Moderate multicollinearity (concerning, monitor)
    - VIF ≥ 10: High multicollinearity (problematic, action needed)

    Formula:
        VIF_j = 1 / (1 - R²_j)
        where R²_j is from regressing X_j on all other predictors

    Remedies for high VIF:
    - Remove highly correlated variables
    - Combine correlated variables (e.g., create index)
    - Use ridge regression or other regularization
    - Collect more data
    - Center variables

    Usage:
        test = VIFTest(threshold=10.0)
        results = test.run(data, ['ARL', 'FEE', 'FO', 'SIZE', 'ROA'])
        summary = test.summarize(results)
        print(summary)
    """

    def __init__(self, threshold: float = 10.0, alpha: float = 0.05):
        """
        Initialize VIF test

        Args:
            threshold: VIF threshold for detecting problematic multicollinearity
                      Standard: 10.0, Conservative: 5.0
            alpha: Significance level (not directly used in VIF)
        """
        super().__init__(alpha)
        self.threshold = threshold

    def run(self, data: pd.DataFrame, exog_vars: List[str], **kwargs) -> Dict[str, TestResult]:
        """
        Calculate VIF for each independent variable

        Args:
            data: DataFrame containing all variables
            exog_vars: List of independent variable names to test
            **kwargs: Additional parameters (not used)

        Returns:
            Dictionary mapping variable names to TestResult objects

        Raises:
            ValueError: If data is invalid or variables not found
        """
        # Validate inputs
        missing_vars = [var for var in exog_vars if var not in data.columns]
        if missing_vars:
            raise ValueError(f"Variables not found in DataFrame: {missing_vars}")

        # Extract independent variables
        X = data[exog_vars].copy()

        # Check for missing values
        if X.isnull().any().any():
            print("Warning: Missing values detected. Dropping rows with NaN.")
            X = X.dropna()

        # Check for constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            raise ValueError(f"Constant columns detected: {constant_cols}. Remove them before VIF calculation.")

        # Add constant for VIF calculation
        X_with_const = sm.add_constant(X, has_constant='add')

        vif_results = {}

        for i, var in enumerate(exog_vars):
            try:
                # Calculate VIF (skip constant column at index 0)
                vif_value = variance_inflation_factor(X_with_const.values, i + 1)

                # Check if VIF is problematic
                problematic = vif_value >= self.threshold

                # Determine severity level
                if vif_value < 5:
                    severity = "low"
                    interpretation = f"Low multicollinearity for {var} (VIF = {vif_value:.2f}). Acceptable."
                elif vif_value < 10:
                    severity = "moderate"
                    interpretation = f"Moderate multicollinearity for {var} (VIF = {vif_value:.2f}). Monitor this variable."
                else:
                    severity = "high"
                    interpretation = (f"High multicollinearity for {var} (VIF = {vif_value:.2f}). "
                                     "Consider removing or combining with related variables.")

                vif_results[var] = TestResult(
                    test_name=f"VIF Test - {var}",
                    statistic=vif_value,
                    p_value=None,  # VIF doesn't have p-value
                    reject_null=problematic,
                    interpretation=interpretation,
                    additional_info={
                        "threshold": self.threshold,
                        "severity": severity,
                        "r_squared": 1 - (1 / vif_value) if vif_value > 0 else None
                    }
                )

            except Exception as e:
                # Handle calculation errors (e.g., perfect multicollinearity)
                vif_results[var] = TestResult(
                    test_name=f"VIF Test - {var}",
                    statistic=float('inf'),
                    p_value=None,
                    reject_null=True,
                    interpretation=f"Error calculating VIF for {var}: {str(e)}. Likely perfect multicollinearity.",
                    additional_info={"error": str(e)}
                )

        return vif_results

    def summarize(self, vif_results: Dict[str, TestResult]) -> pd.DataFrame:
        """
        Create summary table of VIF results

        Args:
            vif_results: Dictionary of VIF test results

        Returns:
            DataFrame with VIF summary sorted by VIF value
        """
        summary_data = []

        for var, result in vif_results.items():
            vif_value = result.statistic
            severity = result.additional_info.get('severity', 'error')

            # Status indicator
            if vif_value < 5:
                status = "✓ OK"
            elif vif_value < 10:
                status = "⚠ Monitor"
            else:
                status = "✗ Problematic"

            summary_data.append({
                'Variable': var,
                'VIF': round(vif_value, 2) if not np.isinf(vif_value) else 'Inf',
                'Severity': severity.capitalize(),
                'Status': status,
                'Interpretation': result.interpretation
            })

        summary_df = pd.DataFrame(summary_data)

        # Sort by VIF value (descending)
        summary_df['VIF_numeric'] = summary_df['VIF'].apply(
            lambda x: float('inf') if x == 'Inf' else float(x)
        )
        summary_df = summary_df.sort_values('VIF_numeric', ascending=False)
        summary_df = summary_df.drop('VIF_numeric', axis=1)

        return summary_df

    def check_overall_multicollinearity(self, vif_results: Dict[str, TestResult]) -> str:
        """
        Provide overall assessment of multicollinearity

        Args:
            vif_results: Dictionary of VIF test results

        Returns:
            String with overall assessment and recommendations
        """
        vif_values = [r.statistic for r in vif_results.values() if not np.isinf(r.statistic)]

        if not vif_values:
            return "Unable to assess multicollinearity (all VIF calculations failed)"

        max_vif = max(vif_values)
        mean_vif = np.mean(vif_values)

        n_high = sum(1 for v in vif_values if v >= 10)
        n_moderate = sum(1 for v in vif_values if 5 <= v < 10)
        n_low = sum(1 for v in vif_values if v < 5)

        output = []
        output.append("\n" + "=" * 60)
        output.append("MULTICOLLINEARITY ASSESSMENT SUMMARY")
        output.append("=" * 60)
        output.append(f"\nVariables tested: {len(vif_results)}")
        output.append(f"Maximum VIF: {max_vif:.2f}")
        output.append(f"Mean VIF: {mean_vif:.2f}")
        output.append(f"\nSeverity Distribution:")
        output.append(f"  Low (VIF < 5):        {n_low} variables")
        output.append(f"  Moderate (5 ≤ VIF < 10): {n_moderate} variables")
        output.append(f"  High (VIF ≥ 10):      {n_high} variables")

        # Overall conclusion
        output.append("\n" + "-" * 60)
        output.append("OVERALL ASSESSMENT:")

        if n_high == 0 and n_moderate == 0:
            conclusion = "✓ No multicollinearity issues detected. All VIF values are acceptable."
        elif n_high == 0:
            conclusion = f"⚠ Moderate multicollinearity detected in {n_moderate} variable(s). Monitor these variables."
        else:
            conclusion = (f"✗ High multicollinearity detected in {n_high} variable(s). "
                         "Consider remedial action (remove variables, combine, or use regularization).")

        output.append(conclusion)

        # Recommendations
        if n_high > 0 or n_moderate > 0:
            output.append("\nRECOMMENDATIONS:")
            output.append("1. Examine correlation matrix to identify highly correlated pairs")
            output.append("2. Consider removing or combining highly correlated variables")
            output.append("3. Use ridge regression if theoretical reasons require keeping all variables")
            output.append("4. Interpret coefficients cautiously for variables with high VIF")

        output.append("=" * 60)

        return "\n".join(output)


# Export classes
__all__ = ['VIFTest']
