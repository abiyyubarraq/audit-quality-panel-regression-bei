"""
Results Formatter Module

Formats statistical results for console output and export.
"""

import pandas as pd
from typing import Dict, Any
from tabulate import tabulate


class ResultsFormatter:
    """
    Format statistical results for display

    Usage:
        formatter = ResultsFormatter()
        print(formatter.format_test_results(test_results))
        print(formatter.format_vif_results(vif_df))
    """

    @staticmethod
    def format_regression_summary(results, title: str = "Regression Results") -> str:
        """
        Format regression results

        Args:
            results: Regression results object
            title: Title for the output

        Returns:
            Formatted string
        """
        output = []
        output.append("\n" + "=" * 60)
        output.append(title)
        output.append("=" * 60)
        output.append(str(results.summary))
        return "\n".join(output)

    @staticmethod
    def format_test_results(test_results: Dict[str, Any]) -> str:
        """
        Format multiple test results as a table

        Args:
            test_results: Dictionary of test results

        Returns:
            Formatted table string
        """
        table_data = []

        for test_name, result in test_results.items():
            if result is None:
                continue

            if hasattr(result, 'test_name'):
                table_data.append([
                    result.test_name,
                    f"{result.statistic:.4f}",
                    f"{result.p_value:.4f}" if result.p_value is not None else "N/A",
                    "Yes" if result.reject_null else "No",
                    result.interpretation[:50] + "..." if len(result.interpretation) > 50 else result.interpretation
                ])

        if not table_data:
            return "No valid test results to display"

        headers = ["Test", "Statistic", "P-value", "Reject H0", "Interpretation"]
        return tabulate(table_data, headers=headers, tablefmt="grid")

    @staticmethod
    def format_vif_results(vif_df: pd.DataFrame) -> str:
        """
        Format VIF results as a table

        Args:
            vif_df: DataFrame with VIF results

        Returns:
            Formatted table string
        """
        return tabulate(vif_df, headers='keys', tablefmt='grid', showindex=False)

    @staticmethod
    def format_coefficient_table(
        coefficients: pd.Series,
        std_errors: pd.Series,
        pvalues: pd.Series,
        conf_int: pd.DataFrame = None
    ) -> str:
        """
        Format coefficient table with significance stars

        Args:
            coefficients: Series of coefficient values
            std_errors: Series of standard errors
            pvalues: Series of p-values
            conf_int: DataFrame with confidence intervals (optional)

        Returns:
            Formatted table string
        """
        table_data = []

        for var in coefficients.index:
            coef = coefficients[var]
            se = std_errors[var]
            pval = pvalues[var]

            # Significance stars
            if pval < 0.01:
                sig = "***"
            elif pval < 0.05:
                sig = "**"
            elif pval < 0.1:
                sig = "*"
            else:
                sig = ""

            row = [
                var,
                f"{coef:.4f}{sig}",
                f"({se:.4f})",
                f"{pval:.4f}"
            ]

            if conf_int is not None and var in conf_int.index:
                row.append(f"[{conf_int.loc[var, 0]:.4f}, {conf_int.loc[var, 1]:.4f}]")

            table_data.append(row)

        headers = ["Variable", "Coefficient", "Std. Error", "P-value"]
        if conf_int is not None:
            headers.append("95% CI")

        output = []
        output.append(tabulate(table_data, headers=headers, tablefmt="grid"))
        output.append("\nSignificance: *** p<0.01, ** p<0.05, * p<0.1")

        return "\n".join(output)


__all__ = ['ResultsFormatter']
