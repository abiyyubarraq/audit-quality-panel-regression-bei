"""
Regression Visualization Utilities

Provides plotting functions for panel data regression analysis,
including coefficient plots, diagnostic plots, and comparison visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


class RegressionPlotter:
    """
    Visualization utilities for panel data regression analysis

    Usage:
        plotter = RegressionPlotter(style='seaborn-v0_8-whitegrid')

        # Coefficient plot with confidence intervals
        plotter.plot_coefficients(
            coefficients=model.params,
            std_errors=model.std_errors,
            title='Fixed Effects Model'
        )

        # Diagnostic plots
        plotter.plot_diagnostics(
            residuals=model.resids,
            fitted=model.fitted_values
        )
    """

    def __init__(
        self,
        style: str = 'seaborn-v0_8-whitegrid',
        palette: str = 'Set2',
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Initialize the plotter with style settings

        Args:
            style: Matplotlib style to use
            palette: Seaborn color palette
            figsize: Default figure size (width, height)
        """
        plt.style.use(style)
        sns.set_palette(palette)
        self.default_figsize = figsize

    def plot_coefficients(
        self,
        coefficients: Union[pd.Series, Dict],
        std_errors: Union[pd.Series, Dict],
        title: str = 'Coefficient Plot with 95% Confidence Intervals',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        alpha: float = 0.05
    ) -> plt.Figure:
        """
        Plot coefficients with confidence intervals

        Args:
            coefficients: Coefficient estimates
            std_errors: Standard errors
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Path to save figure
            alpha: Significance level for CI (default: 0.05 for 95% CI)

        Returns:
            matplotlib Figure object
        """
        if isinstance(coefficients, dict):
            coefficients = pd.Series(coefficients)
        if isinstance(std_errors, dict):
            std_errors = pd.Series(std_errors)

        # Calculate confidence intervals
        z_score = stats.norm.ppf(1 - alpha / 2)
        ci_lower = coefficients - z_score * std_errors
        ci_upper = coefficients + z_score * std_errors

        # Create plot
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)

        var_names = coefficients.index.tolist()
        y_pos = np.arange(len(var_names))

        # Plot with error bars
        ax.errorbar(
            coefficients.values,
            y_pos,
            xerr=[
                coefficients.values - ci_lower.values,
                ci_upper.values - coefficients.values
            ],
            fmt='o',
            markersize=8,
            capsize=5,
            linewidth=2,
            color='steelblue',
            ecolor='steelblue'
        )

        # Add zero line
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No effect')

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_names)
        ax.set_xlabel('Coefficient Estimate', fontweight='bold', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to: {save_path}")

        return fig

    def plot_diagnostics(
        self,
        residuals: Union[pd.Series, np.ndarray],
        fitted: Union[pd.Series, np.ndarray],
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create diagnostic plots for regression models

        Args:
            residuals: Model residuals
            fitted: Fitted values
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Convert to numpy arrays if needed
        if isinstance(residuals, pd.Series):
            residuals = residuals.values
        if isinstance(fitted, pd.Series):
            fitted = fitted.values

        # 1. Residuals vs Fitted
        axes[0, 0].scatter(fitted, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fitted Values', fontweight='bold')
        axes[0, 0].set_ylabel('Residuals', fontweight='bold')
        axes[0, 0].set_title('Residuals vs Fitted', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].set_xlabel('Residuals', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Distribution of Residuals', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Scale-Location (sqrt of standardized residuals)
        sqrt_std_resid = np.sqrt(np.abs(residuals / np.std(residuals)))
        axes[1, 1].scatter(fitted, sqrt_std_resid, alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Fitted Values', fontweight='bold')
        axes[1, 1].set_ylabel('√|Standardized Residuals|', fontweight='bold')
        axes[1, 1].set_title('Scale-Location Plot', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Diagnostics saved to: {save_path}")

        return fig

    def plot_model_comparison(
        self,
        models: Dict[str, Dict],
        variables: List[str],
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare coefficients across multiple models

        Args:
            models: Dictionary of model results
                   e.g., {'FE': {'params': ..., 'std_errors': ...}, 'RE': {...}}
            variables: List of variables to compare
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        n_models = len(models)
        n_vars = len(variables)
        width = 0.8 / n_models
        x_pos = np.arange(n_vars)

        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for i, (model_name, model_results) in enumerate(models.items()):
            coefs = [model_results['params'].get(var, 0) for var in variables]
            ses = [model_results['std_errors'].get(var, 0) for var in variables]

            positions = x_pos + (i - n_models/2 + 0.5) * width

            ax.bar(
                positions,
                coefs,
                width,
                label=model_name,
                alpha=0.8,
                color=colors[i],
                yerr=[1.96 * se for se in ses],
                capsize=3
            )

        ax.set_xlabel('Variables', fontweight='bold', fontsize=11)
        ax.set_ylabel('Coefficient Estimate', fontweight='bold', fontsize=11)
        ax.set_title('Model Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to: {save_path}")

        return fig

    def plot_standard_error_comparison(
        self,
        se_dict: Dict[str, pd.Series],
        variables: List[str],
        title: str = 'Standard Error Comparison',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare standard errors across different estimation methods

        Args:
            se_dict: Dictionary of standard errors
                    e.g., {'Robust': pd.Series, 'Clustered': pd.Series}
            variables: List of variables to plot
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        n_methods = len(se_dict)
        width = 0.8 / n_methods
        x_pos = np.arange(len(variables))

        colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

        for i, (method_name, se_values) in enumerate(se_dict.items()):
            ses = [se_values.get(var, 0) for var in variables]
            positions = x_pos + (i - n_methods/2 + 0.5) * width

            ax.bar(
                positions,
                ses,
                width,
                label=method_name,
                alpha=0.8,
                color=colors[i]
            )

        ax.set_xlabel('Variables', fontweight='bold', fontsize=11)
        ax.set_ylabel('Standard Error', fontweight='bold', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ SE comparison saved to: {save_path}")

        return fig

    def plot_panel_structure(
        self,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize panel data structure

        Args:
            data: Panel dataset
            entity_col: Entity identifier column
            time_col: Time identifier column
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Count observations by entity
        entity_counts = data[entity_col].value_counts().sort_values()

        # Count observations by time
        time_counts = data[time_col].value_counts().sort_index()

        # Plot 1: Distribution of observations per entity
        axes[0].hist(entity_counts.values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Observations per Entity', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Distribution of Observations per Entity', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Plot 2: Observations by time period
        axes[1].bar(range(len(time_counts)), time_counts.values, alpha=0.7, color='coral')
        axes[1].set_xlabel('Time Period', fontweight='bold')
        axes[1].set_ylabel('Number of Observations', fontweight='bold')
        axes[1].set_title('Observations by Time Period', fontweight='bold')
        axes[1].set_xticks(range(len(time_counts)))
        axes[1].set_xticklabels(time_counts.index, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Panel structure plot saved to: {save_path}")

        return fig

    def plot_hypothesis_results(
        self,
        hypothesis_df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize hypothesis testing results

        Args:
            hypothesis_df: DataFrame with hypothesis results
                          Expected columns: 'Variable', 'Coefficient', 'Std_Error', 'p_value'
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        variables = hypothesis_df['Variable'].tolist()
        coefs = hypothesis_df['Coefficient'].values
        ses = hypothesis_df['Std_Error'].values if 'Std_Error' in hypothesis_df.columns else hypothesis_df['Std_Error_Clustered'].values
        pvals = hypothesis_df['p_value'].values

        # Color by significance
        colors = []
        for pval in pvals:
            if pval < 0.01:
                colors.append('darkgreen')
            elif pval < 0.05:
                colors.append('green')
            elif pval < 0.10:
                colors.append('orange')
            else:
                colors.append('lightgray')

        y_pos = np.arange(len(variables))

        # Plot coefficients with error bars
        for i, (coef, se, color) in enumerate(zip(coefs, ses, colors)):
            ax.errorbar(
                coef,
                y_pos[i],
                xerr=1.96 * se,
                fmt='o',
                markersize=10,
                capsize=5,
                linewidth=2,
                color=color
            )

        # Add zero line
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No effect')

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables)
        ax.set_xlabel('Coefficient Estimate (95% CI)', fontweight='bold', fontsize=11)
        ax.set_title('Hypothesis Testing Results', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend for significance levels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='p < 0.01 (***)'),
            Patch(facecolor='green', label='p < 0.05 (**)'),
            Patch(facecolor='orange', label='p < 0.10 (*)'),
            Patch(facecolor='lightgray', label='Not significant')
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Hypothesis results plot saved to: {save_path}")

        return fig


__all__ = ['RegressionPlotter']
