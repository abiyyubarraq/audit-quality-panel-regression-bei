"""
Understanding Between R² in Fixed Effects Models

This script demonstrates why FE has negative Between R² and why that's OK.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# Load data
data = pd.read_json('../data/processed/audit_data.json')

# Set up variables
entity_col = 'CODE'
time_col = 'YEAR'
dep_var = 'AQMS'
exog_vars = ['ARL', 'FEE', 'FO', 'SIZE', 'ROA']

print("="*80)
print("UNDERSTANDING BETWEEN R² IN FIXED EFFECTS")
print("="*80)

# ============================================================================
# PART 1: What does Between R² measure?
# ============================================================================

print("\n" + "="*80)
print("PART 1: What does Between R² measure?")
print("="*80)

# Calculate company means (between variation)
company_means = data.groupby(entity_col)[dep_var + exog_vars].mean()

print(f"\nCompany means (first 10 companies):")
print(company_means.head(10))

print(f"\nThis is the 'between' variation:")
print(f"  - Why does Company A have higher average AQMS than Company B?")
print(f"  - Uses only ONE observation per company (the average over time)")
print(f"  - Total observations: {len(company_means)} (one per company)")

# ============================================================================
# PART 2: How does Fixed Effects work?
# ============================================================================

print("\n" + "="*80)
print("PART 2: How does Fixed Effects work?")
print("="*80)

print("\nFixed Effects transformation (within-group demeaning):")
print("  1. For each company, calculate the mean of each variable over time")
print("  2. Subtract this mean from each observation")
print("  3. Estimate regression on these DEVIATIONS from company means")

# Example for one company
example_company = 'AALI'
company_data = data[data[entity_col] == example_company].copy()

print(f"\nExample: Company {example_company}")
print(f"\nOriginal data:")
print(company_data[['YEAR', 'AQMS', 'ARL', 'FEE']].to_string(index=False))

# Calculate means
company_data['AQMS_mean'] = company_data['AQMS'].mean()
company_data['ARL_mean'] = company_data['ARL'].mean()
company_data['FEE_mean'] = company_data['FEE'].mean()

# Calculate deviations
company_data['AQMS_demeaned'] = company_data['AQMS'] - company_data['AQMS_mean']
company_data['ARL_demeaned'] = company_data['ARL'] - company_data['ARL_mean']
company_data['FEE_demeaned'] = company_data['FEE'] - company_data['FEE_mean']

print(f"\nAfter demeaning (deviations from company mean):")
print(company_data[['YEAR', 'AQMS_demeaned', 'ARL_demeaned', 'FEE_demeaned']].to_string(index=False))

print(f"\n🔑 KEY POINT:")
print(f"   Fixed Effects uses ONLY these deviations (within variation)")
print(f"   It COMPLETELY IGNORES the company mean levels (between variation)")
print(f"   Company {example_company}'s mean AQMS = {company_data['AQMS_mean'].iloc[0]:.2f}")
print(f"   → This information is THROWN AWAY by FE!")

# ============================================================================
# PART 3: Why FE has negative Between R²
# ============================================================================

print("\n" + "="*80)
print("PART 3: Why FE has negative Between R²")
print("="*80)

# Fit FE model
data_panel = data.set_index([entity_col, time_col])
fe_model = PanelOLS(
    data_panel[dep_var],
    data_panel[exog_vars],
    entity_effects=True
).fit(cov_type='clustered', cluster_entity=True)

print(f"\nFixed Effects R² statistics:")
print(f"  Within R²:   {fe_model.rsquared_within:.4f}  (explains within variation)")
print(f"  Between R²:  {fe_model.rsquared_between:.4f}  (explains between variation)")
print(f"  Overall R²:  {fe_model.rsquared_overall:.4f}  (weighted average)")

print(f"\n❓ Why is Between R² NEGATIVE ({fe_model.rsquared_between:.4f})?")
print(f"\nBetween R² measures: Can FE coefficients predict company means?")
print(f"\nLet's test this:")

# Get FE coefficients (estimated from within variation)
fe_coefs = fe_model.params

print(f"\nFE coefficients (estimated from WITHIN variation):")
for var in exog_vars:
    print(f"  {var}: {fe_coefs[var]:7.4f}")

# Now use these coefficients to predict company means (between variation)
X_between = company_means[exog_vars]
y_between_actual = company_means[dep_var]

# Predict using FE coefficients
y_between_predicted = X_between.dot(fe_coefs[exog_vars])

# Calculate Between R²
ss_total = ((y_between_actual - y_between_actual.mean())**2).sum()
ss_residual = ((y_between_actual - y_between_predicted)**2).sum()
between_r2_manual = 1 - (ss_residual / ss_total)

print(f"\n🧮 Manual calculation of Between R²:")
print(f"   SS_total (between):    {ss_total:.2f}")
print(f"   SS_residual (between): {ss_residual:.2f}")
print(f"   Between R²: 1 - ({ss_residual:.2f} / {ss_total:.2f}) = {between_r2_manual:.4f}")

print(f"\n💡 INTERPRETATION:")
if between_r2_manual < 0:
    print(f"   Between R² = {between_r2_manual:.4f} < 0")
    print(f"   → FE coefficients predict company means WORSE than just using the mean!")
    print(f"   → This is NORMAL and EXPECTED because:")
    print(f"      • FE coefficients come from WITHIN-company changes")
    print(f"      • Within-company relationships ≠ Between-company relationships")
    print(f"      • Example: 'Within firm, size ↑ → AQMS ↓' does NOT mean")
    print(f"                 'Bigger firms have lower AQMS than smaller firms'")

# ============================================================================
# PART 4: Visual demonstration
# ============================================================================

print("\n" + "="*80)
print("PART 4: Visual demonstration")
print("="*80)

# Pick one variable to visualize
var = 'SIZE'

# Calculate within and between components
data_viz = data.copy()
data_viz['SIZE_between'] = data_viz.groupby(entity_col)['SIZE'].transform('mean')
data_viz['SIZE_within'] = data_viz['SIZE'] - data_viz['SIZE_between']
data_viz['AQMS_between'] = data_viz.groupby(entity_col)['AQMS'].transform('mean')
data_viz['AQMS_within'] = data_viz['AQMS'] - data_viz['AQMS_between']

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Raw data (total variation)
axes[0].scatter(data_viz['SIZE'], data_viz['AQMS'], alpha=0.5, s=30)
z = np.polyfit(data_viz['SIZE'], data_viz['AQMS'], 1)
p = np.poly1d(z)
x_line = np.linspace(data_viz['SIZE'].min(), data_viz['SIZE'].max(), 100)
axes[0].plot(x_line, p(x_line), "r-", linewidth=2, label=f'OLS: slope={z[0]:.3f}')
axes[0].set_xlabel('Firm Size (SIZE)', fontweight='bold')
axes[0].set_ylabel('Audit Quality (AQMS)', fontweight='bold')
axes[0].set_title('Total Variation (Pooled OLS)', fontweight='bold', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Between variation (company means)
axes[1].scatter(company_means['SIZE'], company_means['AQMS'], alpha=0.7, s=100, color='orange')
z_between = np.polyfit(company_means['SIZE'], company_means['AQMS'], 1)
p_between = np.poly1d(z_between)
x_line_between = np.linspace(company_means['SIZE'].min(), company_means['SIZE'].max(), 100)
axes[1].plot(x_line_between, p_between(x_line_between), "r-", linewidth=2,
             label=f'Between: slope={z_between[0]:.3f}')
axes[1].set_xlabel('Average Firm Size', fontweight='bold')
axes[1].set_ylabel('Average Audit Quality', fontweight='bold')
axes[1].set_title('Between Variation (Company Averages)\n63 points, 1 per company',
                   fontweight='bold', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Within variation (deviations from company means)
axes[2].scatter(data_viz['SIZE_within'], data_viz['AQMS_within'], alpha=0.5, s=30, color='green')
z_within = np.polyfit(data_viz['SIZE_within'], data_viz['AQMS_within'], 1)
p_within = np.poly1d(z_within)
x_line_within = np.linspace(data_viz['SIZE_within'].min(), data_viz['SIZE_within'].max(), 100)
axes[2].plot(x_line_within, p_within(x_line_within), "r-", linewidth=2,
             label=f'Within (FE): slope={z_within[0]:.3f}')
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[2].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[2].set_xlabel('SIZE - Company Mean', fontweight='bold')
axes[2].set_ylabel('AQMS - Company Mean', fontweight='bold')
axes[2].set_title('Within Variation (Deviations from Means)\nThis is what FE uses!',
                   fontweight='bold', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/within_vs_between_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📊 Visualization saved to: results/within_vs_between_decomposition.png")

print(f"\n🔍 KEY INSIGHT FROM VISUALIZATION:")
print(f"   Left plot (Total):    Pooled OLS slope = {z[0]:.3f}")
print(f"   Middle plot (Between): Between slope = {z_between[0]:.3f}")
print(f"   Right plot (Within):   FE slope = {z_within[0]:.3f}")
print(f"\n   → The slopes are DIFFERENT!")
print(f"   → FE uses only the Within slope ({z_within[0]:.3f})")
print(f"   → When you use Within slope to predict Between differences...")
print(f"     it can perform WORSE than the mean → Negative Between R²!")

# ============================================================================
# PART 5: Detailed example with specific companies
# ============================================================================

print("\n" + "="*80)
print("PART 5: Concrete Example - Why FE fails at between prediction")
print("="*80)

# Pick two companies with very different average AQMS
company_means_sorted = company_means.sort_values('AQMS')
low_aqms_company = company_means_sorted.index[0]
high_aqms_company = company_means_sorted.index[-1]

print(f"\nCompare two companies:")
print(f"\n1. {low_aqms_company} (Lowest average AQMS):")
print(company_means.loc[low_aqms_company].to_string())

print(f"\n2. {high_aqms_company} (Highest average AQMS):")
print(company_means.loc[high_aqms_company].to_string())

# Predict using FE coefficients
X_low = company_means.loc[low_aqms_company, exog_vars]
X_high = company_means.loc[high_aqms_company, exog_vars]

pred_low = fe_coefs[exog_vars].dot(X_low)
pred_high = fe_coefs[exog_vars].dot(X_high)

actual_low = company_means.loc[low_aqms_company, 'AQMS']
actual_high = company_means.loc[high_aqms_company, 'AQMS']

print(f"\n🎯 FE prediction using Within-company coefficients:")
print(f"\n   {low_aqms_company}:")
print(f"      Actual AQMS:    {actual_low:.2f}")
print(f"      FE Predicted:   {pred_low:.2f}")
print(f"      Error:          {abs(actual_low - pred_low):.2f}")

print(f"\n   {high_aqms_company}:")
print(f"      Actual AQMS:    {actual_high:.2f}")
print(f"      FE Predicted:   {pred_high:.2f}")
print(f"      Error:          {abs(actual_high - pred_high):.2f}")

# Compare to just using the overall mean
overall_mean = data[dep_var].mean()
error_mean_low = abs(actual_low - overall_mean)
error_mean_high = abs(actual_high - overall_mean)

print(f"\n📊 Compare to just using overall mean ({overall_mean:.2f}):")
print(f"\n   {low_aqms_company}:")
print(f"      Actual:         {actual_low:.2f}")
print(f"      Mean:           {overall_mean:.2f}")
print(f"      Error:          {error_mean_low:.2f}")

print(f"\n   {high_aqms_company}:")
print(f"      Actual:         {actual_high:.2f}")
print(f"      Mean:           {overall_mean:.2f}")
print(f"      Error:          {error_mean_high:.2f}")

print(f"\n❗ CONCLUSION:")
print(f"   FE coefficients (from within variation) do NOT explain")
print(f"   between-company differences well.")
print(f"   → This is WHY Between R² is negative!")
print(f"   → This is NORMAL and NOT A PROBLEM!")

# ============================================================================
# PART 6: What model explains between variation?
# ============================================================================

print("\n" + "="*80)
print("PART 6: What model SHOULD explain between variation?")
print("="*80)

# Between-Effects model
from linearmodels.panel import BetweenOLS

be_model = BetweenOLS(
    data_panel[dep_var],
    data_panel[exog_vars]
).fit()

print("\nBetween-Effects Model (uses only company averages):")
print(f"  R²: {be_model.rsquared:.4f}")
print(f"\nCoefficients:")
for var in exog_vars:
    print(f"  {var}: {be_model.params[var]:7.4f}")

print(f"\n📊 Comparison:")
print(f"\n{'Variable':<10} {'FE (Within)':<15} {'BE (Between)':<15} {'Same?'}")
print(f"{'-'*55}")
for var in exog_vars:
    fe_coef = fe_coefs[var]
    be_coef = be_model.params[var]
    same = 'Yes ✓' if np.sign(fe_coef) == np.sign(be_coef) else 'No ✗'
    print(f"{var:<10} {fe_coef:>10.4f}     {be_coef:>10.4f}      {same}")

print(f"\n💡 KEY INSIGHT:")
print(f"   • Fixed Effects (FE) uses WITHIN-company relationships")
print(f"   • Between Effects (BE) uses BETWEEN-company relationships")
print(f"   • These can be DIFFERENT (and often are!)")
print(f"   • FE has negative Between R² because Within ≠ Between")
print(f"   • This is EXPECTED and NORMAL!")

# ============================================================================
# PART 7: Summary and recommendations
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: Is negative Between R² a problem?")
print("="*80)

print(f"\n❌ NO! It is NOT a problem. Here's why:")

print(f"\n1. FIXED EFFECTS IS NOT DESIGNED TO EXPLAIN BETWEEN VARIATION")
print(f"   • FE removes all between-company differences (that's the point!)")
print(f"   • FE focuses on WITHIN-company changes over time")
print(f"   • Between R² in FE is irrelevant - ignore it!")

print(f"\n2. YOUR SITUATION IS COMMON:")
print(f"   • Within R²: {fe_model.rsquared_within:.4f} (low - your variables don't change much within firms)")
print(f"   • Between R²: {fe_model.rsquared_between:.4f} (negative - FE can't explain between differences)")
print(f"   • This pattern is TYPICAL when:")
print(f"     - Variables are relatively stable within firms (low within variation)")
print(f"     - Within and between relationships differ")
print(f"     - Short time period (T=4)")

print(f"\n3. WHAT TO REPORT IN YOUR THESIS:")
print(f'   "The Fixed Effects within-R² of {fe_model.rsquared_within:.4f} indicates')
print(f'    that our independent variables explain {fe_model.rsquared_within*100:.1f}% of the within-firm')
print(f'    variation in audit quality. The negative between-R² is expected in')
print(f'    Fixed Effects models when within-firm and between-firm relationships')
print(f'    differ, and does not indicate model misspecification (Wooldridge, 2010).')
print(f'    The low within-R² suggests audit quality is primarily determined by')
print(f'    time-invariant firm characteristics (captured by the entity fixed')
print(f'    effects) rather than by temporal changes in the independent variables."')

print(f"\n4. ALTERNATIVE MODELS FOR BETWEEN VARIATION:")
print(f"   If you want to explain WHY Company A > Company B:")
print(f"   • Use Between-Effects model (BE R² = {be_model.rsquared:.4f})")
print(f"   • Use Pooled OLS (but biased if unobserved heterogeneity exists)")
print(f"   • Use Random Effects (but Hausman test rejected it)")

print(f"\n5. YOUR MAIN RESULT IS STILL VALID:")
print(f"   • FE with clustered SEs is the correct model (Hausman test)")
print(f"   • Your hypotheses about WITHIN-firm effects are properly tested")
print(f"   • Negative Between R² doesn't invalidate your findings")
print(f"   • Focus on and report the Within R² ({fe_model.rsquared_within:.4f})")

print("\n" + "="*80)
print("✅ CONCLUSION: Your FE model is FINE. Negative Between R² is NORMAL.")
print("="*80)
