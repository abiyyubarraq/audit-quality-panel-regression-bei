# Thesis Methodology Assessment & Breakdown

**Date**: 2025-12-12
**Analysis**: Audit Quality Panel Data Regression (Indonesian Stock Exchange, 2021-2024)

---

## EXECUTIVE SUMMARY

### ✅ **Overall Verdict**: SCIENTIFICALLY VALID with MODERATE CONCERNS

Your methodology is **fundamentally sound** and follows proper econometric practices for panel data analysis. However, there are **important limitations** you must acknowledge and address in your thesis.

**Key Strengths**:
- Proper panel data methods (Fixed Effects with clustered SEs)
- Comprehensive diagnostic testing
- Correct model selection procedures
- Transparent reporting of results

**Critical Issues**:
1. **Extremely low within-R² (2.89%)** - Model explains almost no within-entity variation
2. **Very small sample** - Only 3.7 observations per parameter (below recommended 10-15)
3. **Low within-variation in key variables** - FO has only 18.6% within variation
4. **All hypotheses not supported** - None of your main variables are significant
5. **ROA outliers** - Two extreme values (LAPD company: -1391%, +3612%)

---

## I. DETAILED STATISTICAL ISSUES & SOLUTIONS

### 🔴 **ISSUE 1: Extremely Low Within-R² (CRITICAL)**

**Problem**:
```
Fixed Effects R² within:  0.0289 (2.89%)
Fixed Effects R² between: -2.1555 (NEGATIVE!)
```

**What this means**:
- Your independent variables explain only **2.89%** of the within-company variation in audit quality
- Negative between-R² suggests model performs worse than the mean for cross-sectional variation
- This is a **red flag** indicating poor model fit

**Implications**:
- Fixed Effects may not be appropriate for this data
- Your theoretical model may be misspecified (missing important variables)
- The relationship between your IVs and DV may be primarily cross-sectional, not longitudinal

**Solutions**:
1. **Acknowledge this limitation prominently in your thesis**:
   > "The low within-R² (2.89%) suggests that our independent variables primarily explain cross-sectional differences between companies rather than changes within companies over time. This is consistent with audit quality being relatively stable within firms across the short 4-year period."

2. **Consider reporting Random Effects results as primary**:
   - RE R² overall: 0.7276 (72.76%) - much better fit!
   - RE uses both between and within variation
   - But Hausman test rejected RE... (see Issue 3 below)

3. **Add discussion**:
   > "The limited time dimension (T=4) and high stability of audit quality within firms over this period may explain why our model captures between-entity differences better than within-entity changes. Future research with longer panels may better identify dynamic effects."

---

### 🟡 **ISSUE 2: Low Within-Variation in Key Variables**

**Problem**:
```
Variable   Within Variation (% of total)   Between/Within Ratio
--------   ------------------------------   --------------------
FO         18.6%                           5.32 (VERY HIGH)
SIZE       26.9%                           3.60
FEE        28.7%                           3.35
ARL        59.5%                           1.36
```

**What this means**:
- **Foreign Ownership (FO)** barely changes within companies (only 18.6% of variation is over time)
- **SIZE** and **FEE** are also relatively stable within firms
- Fixed Effects **cannot** effectively estimate effects of time-invariant variables

**Implications**:
- Your Fixed Effects estimates for FO, SIZE, FEE are based on **minimal** within-company changes
- This explains why these variables are not significant in FE model
- FE "throws away" 81.4% of FO variation (the between-company part)

**Solutions**:
1. **Report Between/Within decomposition in thesis** (you already have this in notebook 02 - good!)

2. **Discuss this explicitly**:
   > "Foreign ownership shows limited within-firm variation over the 4-year period (18.6%), with most variation occurring between firms (Between/Within ratio = 5.32). Fixed Effects estimation, which relies solely on within-firm variation, may lack statistical power to detect foreign ownership effects. This suggests foreign ownership's influence on audit quality may be structural rather than dynamic."

3. **Consider Between-Effects estimator** as robustness check:
   ```python
   # Add this to notebook 04
   from linearmodels.panel import BetweenOLS

   be_model = BetweenOLS(data_panel[dep_var],
                          data_panel[exog_vars]).fit()
   ```

   This uses only cross-sectional variation (opposite of FE).

---

### 🟡 **ISSUE 3: Hausman Test Rejection but Low FE R²**

**Problem**:
- Hausman test: **p < 0.0001** → Strongly recommends Fixed Effects
- BUT Fixed Effects R²: **0.0289** (terrible fit)
- Random Effects R²: **0.7276** (excellent fit)

**This is unusual and needs explanation!**

**Possible reasons**:
1. **Entity effects correlate with regressors** (Hausman test detects this)
2. **But relationship is primarily cross-sectional** (explains low FE R² but high RE R²)
3. **Short time period** (T=4) limits FE's ability to estimate within effects

**Solutions**:
1. **Report BOTH models in main text** (not just appendix):
   - FE as primary (Hausman recommendation)
   - RE as alternative specification
   - Discuss the trade-off

2. **Add this discussion**:
   > "The Hausman test (χ² = 39.65, p < 0.001) suggests entity-specific effects correlate with our independent variables, recommending Fixed Effects. However, the very low within-R² (2.89%) compared to the Random Effects overall-R² (72.76%) indicates our variables primarily capture between-firm differences rather than within-firm changes. This discrepancy reflects the short time dimension (T=4) and high stability of key variables. We report Fixed Effects as our primary specification following the Hausman test, but acknowledge Random Effects may better capture the cross-sectional relationships in our data."

---

### 🔴 **ISSUE 4: ROA Extreme Outliers (CRITICAL DATA ERROR)**

**Problem**:
```
Company: LAPD
  2021: ROA = -1391.15%  (IMPOSSIBLE!)
  2022: ROA = +3612.44%  (IMPOSSIBLE!)

Normal ROA range: -30% to +30%
Your data: -1391% to +3612%
```

**This is almost certainly a data error or unit mismatch!**

**Implications**:
- ROA coefficient is driven by these 2 outliers
- ROA is significant in your model (p < 0.001) but may be spurious
- This affects model diagnostics and interpretation

**Solutions** (MUST DO ONE):

**Option A: Winsorize ROA** (Recommended):
```python
# Add this to notebook 01 or 02
from scipy.stats import mstats

# Winsorize ROA at 1% and 99% percentiles
data['ROA_winsorized'] = mstats.winsorize(data['ROA'], limits=[0.01, 0.01])

# Use ROA_winsorized in regressions
```

**Option B: Remove LAPD company**:
```python
# In notebook 01
data_clean = data[data['CODE'] != 'LAPD']
```

**Option C: Investigate and correct the data**:
- Check original Excel file for LAPD 2021-2022
- ROA should be Net Income / Total Assets (usually -30% to +30%)
- May be a decimal point error (e.g., 36.1244 entered as 3612.44)

**CRITICAL**: You **MUST** address this before submitting your thesis!

---

### 🟡 **ISSUE 5: Small Sample & Degrees of Freedom**

**Problem**:
```
Total observations: 252
Entity fixed effects: 63
Independent variables: 5
Total parameters: 68
Residual DF: 184

Observations per parameter: 3.7 (Rule of thumb: need 10-15)
```

**Implications**:
- Limited statistical power to detect effects
- FE estimates may be unstable with 63 entity dummies
- Large standard errors likely

**Solutions**:
1. **Acknowledge in limitations section**:
   > "Our sample consists of 63 companies over 4 years (N=252), which is modest for panel data analysis with Fixed Effects. The ratio of observations to parameters (3.7:1) is below the recommended 10-15:1, which may limit statistical power and contribute to the lack of significant findings."

2. **Justify sample size**:
   > "This sample represents all consumer goods sector companies listed on the Indonesian Stock Exchange with complete data over 2021-2024, providing a comprehensive sector-level analysis despite the limited sample size."

3. **Consider pooling with Fama-French sectors** if possible (future research).

---

### 🟢 **ISSUE 6: All Hypotheses Not Supported (This is OK!)**

**Problem**:
```
H1 (ARL → AQMS): p = 0.1326 (Not significant)
H2 (FEE → AQMS): p = 0.6058 (Not significant)
H3 (FO → AQMS):  p = 0.7757 (Not significant)
```

**Good news**: This is **scientifically valid**! Negative results are still results.

**Why this happened**:
1. Low within-variation in IVs (see Issue 2)
2. Small sample size (Issue 5)
3. Short time period (T=4)
4. Audit quality may be more influenced by time-invariant factors (firm reputation, industry)

**Solutions for thesis**:
1. **Do NOT spin negative results as positive** - that's unethical
2. **Frame as contribution**:
   > "Contrary to prior literature based on cross-sectional designs, our within-firm analysis finds no significant effect of audit report lag, audit fees, or foreign ownership on audit quality when controlling for time-invariant firm characteristics. This suggests that audit quality is primarily determined by structural firm attributes rather than these annually varying factors."

3. **Discuss thoroughly in Discussion section**:
   - Compare to prior studies (which may have found significance)
   - Explain methodological differences (FE vs cross-sectional OLS)
   - Discuss contextual factors (Indonesia, short time period, COVID-19 period 2021-2024)

---

## II. WHAT YOUR RESULTS ACTUALLY SHOW

### Interpretation Guide:

**Fixed Effects Results** (Primary model):
- **ARL**: β = 0.26, p = 0.13 → Positive but not significant
  - *Interpretation*: "Within companies, increases in audit report lag are not significantly associated with changes in audit quality."

- **FEE**: β = -0.03, p = 0.61 → Negative but not significant
  - *Interpretation*: "Changes in audit fees within the same company do not significantly predict changes in audit quality."

- **FO**: β = -0.08, p = 0.78 → Negative but not significant
  - *Interpretation*: "Changes in foreign ownership within companies are not significantly related to changes in audit quality."

- **SIZE**: β = -0.07, p = 0.02 → **SIGNIFICANT!** (Negative)
  - *Interpretation*: "Within companies, growth in firm size is associated with decreased audit quality scores (β = -0.07, p = 0.02)."

- **ROA**: β = -0.0002, p < 0.001 → **SIGNIFICANT!** (But see outlier issue)
  - *Interpretation*: "Higher profitability within firms is associated with lower audit quality, though this result should be interpreted cautiously due to ROA outliers."

**Key insight**: SIZE and ROA are significant, but your **main hypotheses (ARL, FEE, FO) are not**.

---

## III. THESIS STRUCTURE BREAKDOWN

### Chapter 3: Research Methodology

#### 3.1 Research Design
- Quantitative approach
- Panel data methodology
- Longitudinal design (2021-2024)

#### 3.2 Sample and Data Collection
**Based on your notebook 01**:

```
Sample: 63 companies from Indonesian Stock Exchange
Sector: Consumer Goods
Time period: 2021-2024 (4 years)
Total observations: 252 (balanced panel)
Data source: [Specify source - annual reports? Database?]
```

**Sample justification**:
> "We focus on the consumer goods sector to control for industry-specific factors that may affect audit quality. This sector represents a significant portion of the Indonesian economy and provides a homogeneous context for analysis."

#### 3.3 Variable Measurement
**Based on your notebook 01 & config**:

**Dependent Variable**:
- **AQMS** (Audit Quality Measurement Score): [Need to explain how this is measured - is it 0-4 scale? What do values mean?]

**Independent Variables**:
- **ARL** (Audit Report Lag): [Log of days? Need to specify]
- **FEE** (Audit Fee): [Log of fee amount? Currency?]
- **FO** (Foreign Ownership): [Proportion? 0-1 scale?]

**Control Variables**:
- **SIZE** (Firm Size): [Log of total assets?]
- **ROA** (Return on Assets): [Net Income / Total Assets × 100]

**Critical**: You need to add a detailed measurement table in your thesis!

#### 3.4 Statistical Analysis Methods
**Based on your notebooks 03-04**:

##### 3.4.1 Data Exploration (Notebook 01-02)
```
1. Descriptive statistics (overall, by year, by company)
2. Panel structure validation (balanced vs unbalanced)
3. Between-within variation decomposition
4. Correlation analysis
5. Outlier detection
```

##### 3.4.2 Diagnostic Tests (Notebook 03)
```
1. Normality tests:
   - Shapiro-Wilk test
   - Kolmogorov-Smirnov test
   - Jarque-Bera test

2. Heteroskedasticity tests:
   - Breusch-Pagan test
   - White test

3. Autocorrelation tests:
   - Durbin-Watson test
   - Breusch-Godfrey test

4. Multicollinearity test:
   - Variance Inflation Factor (VIF)
```

**Results to report**:
- Normality: [Report which tests passed/failed]
- Heteroskedasticity: Detected (White test p < 0.01) → Use robust SEs
- Autocorrelation: Detected (DW = 0.89, BG p < 0.001) → Use clustered SEs
- Multicollinearity: Not detected (Max VIF = 1.98 < 10)

##### 3.4.3 Panel Data Models (Notebook 04)
```
1. Model estimation:
   a. Pooled OLS (baseline)
   b. Fixed Effects (FE)
   c. Random Effects (RE)

2. Model selection tests:
   a. F-test (FE vs Pooled): F = 14.19, p < 0.001 → Prefer FE
   b. BP-LM test (RE vs Pooled): χ² = 190.89, p < 0.001 → Prefer RE
   c. Hausman test (FE vs RE): χ² = 39.65, p < 0.001 → Prefer FE

3. Final model: Fixed Effects with entity-clustered standard errors
```

**Regression equation**:
```
AQMS_it = β₀ + β₁(ARL)_it + β₂(FEE)_it + β₃(FO)_it + β₄(SIZE)_it + β₅(ROA)_it + α_i + ε_it

Where:
  i = company (1 to 63)
  t = year (2021 to 2024)
  α_i = company fixed effects
  ε_it = error term

Standard errors clustered by company to account for within-firm autocorrelation.
```

---

### Chapter 4: Results and Analysis

#### 4.1 Descriptive Statistics
**Based on notebook 02**:

**Table 4.1: Descriptive Statistics (Overall)**
```
Variable   Mean    SD      Min      Max     Skewness  Kurtosis
--------   -----   -----   ------   ------  --------  --------
AQMS       1.55    1.07    0.00     4.00    0.21      -1.13
ARL        4.45    0.26    3.81     6.04    1.25      7.52
FEE        22.65   1.65    17.45    26.50   0.05      -0.25
FO         0.19    0.28    0.00     0.92    1.12      -0.30
SIZE       28.42   2.05    17.98    32.94   -1.03     4.06
ROA        8.87    244.18  -1391.15 3612.44 12.15     194.94
```

**Table 4.2: Panel Variation Decomposition**
```
Variable   Overall SD   Between SD   Within SD   B/W Ratio   Interpretation
--------   ----------   ----------   ---------   ---------   --------------
AQMS       1.07         1.01         0.37        2.70        Mostly between-firm
ARL        0.26         0.21         0.15        1.36        Mixed
FEE        1.65         1.59         0.48        3.35        Mostly between-firm
FO         0.28         0.28         0.05        5.32        Highly between-firm
SIZE       2.05         1.99         0.55        3.60        Mostly between-firm
ROA        244.18       69.95        234.07      0.30        Mostly within-firm
```

**Key finding to highlight**:
> "Most independent variables show greater between-firm than within-firm variation (B/W ratios > 1), suggesting audit quality differences are more attributable to firm characteristics than to temporal changes within firms. Foreign ownership displays the highest between-firm variation (ratio = 5.32), indicating it is relatively stable within companies over the 4-year period."

#### 4.2 Correlation Analysis
**Based on notebook 02**:

[Include correlation matrix - check for high correlations]

**Report findings**:
- No high correlations (|r| < 0.7) detected
- [Mention any moderate correlations]

#### 4.3 Diagnostic Test Results
**Based on notebook 03**:

**Table 4.3: Classical Assumption Tests**
```
Test                        Statistic   P-value   Decision
-------------------------   ---------   -------   ------------------
Shapiro-Wilk (Normality)    0.9883      0.0389    Reject H0 (non-normal)
Jarque-Bera (Normality)     3.9617      0.1380    Fail to reject (normal)
BP Test (Heterosked.)       7.6964      0.1738    Fail to reject (homo)
White Test (Heterosked.)    39.1352     0.0064    Reject H0 (hetero)
Durbin-Watson (Autocorr.)   0.8874      -         Positive autocorr.
Breusch-Godfrey (Autocorr.) 78.4872     <0.0001   Reject H0 (autocorr.)
Max VIF (Multicollinearity) 1.98        -         No multicollinearity
```

**Remedial actions**:
> "To address detected heteroskedasticity and autocorrelation, we employ robust standard errors clustered by company in all panel regression models. This approach accounts for arbitrary within-firm correlation over time and heteroskedastic errors across firms."

#### 4.4 Model Selection Results
**Based on notebook 04**:

**Table 4.4: Model Selection Tests**
```
Test                     Statistic   DF      P-value   Decision
----------------------   ---------   -----   -------   -------------
F-test (FE vs Pooled)    14.19       62,184  <0.001    Choose FE
BP-LM (RE vs Pooled)     190.89      1       <0.001    Choose RE
Hausman (FE vs RE)       39.65       5       <0.001    Choose FE
```

**Conclusion**:
> "All model selection tests reject the pooled OLS specification, confirming that panel methods are necessary. The Hausman test (χ² = 39.65, p < 0.001) indicates systematic differences between Fixed and Random Effects estimates, recommending Fixed Effects as the consistent estimator."

#### 4.5 Regression Results
**Based on notebook 04**:

**Table 4.5: Panel Regression Results**
```
                     Pooled OLS          Random Effects      Fixed Effects*
Variable             Coef    (SE)        Coef    (SE)        Coef    (SE)
----------------     -----   ------      -----   ------      -----   ------
ARL                  -0.500** (0.228)   -0.153   (0.158)     0.261   (0.173)
FEE                   0.226*** (0.050)   0.076   (0.043)    -0.031   (0.060)
FO                    0.280   (0.206)    0.068   (0.311)    -0.081   (0.283)
SIZE                  0.107*  (0.044)    0.018   (0.037)    -0.069** (0.029)
ROA                  -0.000   (0.001)   -0.000** (0.000)    -0.000***(0.000)

Constant             -4.451** (1.555)    -               Not reported (absorbed)

N                     252                 252              252
Entities              63                  63               63
R²                    0.314               0.727 (overall)  0.029 (within)
R² Between            -                   0.759            -2.156
R² Within             -                  -0.029            0.029
F-statistic           22.17***            43.16***         1.10
Standard Errors       Robust (HC3)        Robust           Clustered (entity)

Notes:
* Primary specification based on Hausman test.
Standard errors in parentheses.
*** p<0.01, ** p<0.05, * p<0.10
Fixed Effects model includes 63 company fixed effects (not reported).
```

**Key results paragraph**:
> "The Fixed Effects model (Table 4.5, column 3) reveals that none of the hypothesized relationships are statistically significant at conventional levels. Audit report lag (β = 0.261, p = 0.133), audit fees (β = -0.031, p = 0.606), and foreign ownership (β = -0.081, p = 0.776) show no significant within-firm associations with audit quality. However, firm size exhibits a significant negative relationship (β = -0.069, p = 0.018), suggesting that within-firm growth corresponds to decreased audit quality scores. ROA is also significant (β = -0.0002, p < 0.001), though this result requires cautious interpretation due to identified outliers."

#### 4.6 Hypothesis Testing Results
**Based on notebook 04**:

**Table 4.6: Hypothesis Testing Summary**
```
Hypothesis  Description                           Coef    SE      t       p       95% CI            Decision
----------  ------------------------------------  ------  ------  ------  ------  ----------------  ----------------
H1          ARL → AQMS (Audit Report Lag)         0.261   0.173   1.51    0.133   [-0.078, 0.600]   Not Supported
H2          FEE → AQMS (Audit Fee)               -0.031   0.060  -0.52    0.606   [-0.149, 0.087]   Not Supported
H3          FO → AQMS (Foreign Ownership)        -0.081   0.283  -0.29    0.776   [-0.636, 0.474]   Not Supported
```

**Results summary**:
> "None of the three primary hypotheses receive empirical support in the Fixed Effects specification with clustered standard errors. This suggests that within-company temporal variation in audit report lag, audit fees, and foreign ownership does not significantly predict changes in audit quality over the 2021-2024 period."

---

### Chapter 5: Discussion

#### 5.1 Interpretation of Findings

**5.1.1 Why hypotheses were not supported:**

1. **Methodological explanation**:
   > "The lack of significant findings in the Fixed Effects model contrasts with some prior cross-sectional studies that found significant relationships. This discrepancy can be attributed to several methodological factors. First, Fixed Effects estimation isolates within-firm variation, controlling for all time-invariant firm characteristics including reputation, organizational culture, and industry factors. Prior cross-sectional studies may have detected spurious relationships driven by omitted time-invariant variables. Second, our variables exhibit low within-firm variation (e.g., foreign ownership's within-component is only 18.6% of total variation), providing limited identifying variation for Fixed Effects. Third, the short time dimension (T=4) may be insufficient to capture dynamic adjustments in audit quality following changes in audit characteristics."

2. **Substantive explanation**:
   > "Our null findings suggest that audit quality is primarily determined by structural, time-invariant firm characteristics rather than annually varying factors such as audit fees or report lags. This interpretation is consistent with the high between-firm variation in audit quality (B/W ratio = 2.70) and the extremely low within-R² (2.89%). Audit quality appears to be a stable firm attribute in the Indonesian context, possibly reflecting persistent auditor-client relationships and firm-specific governance structures."

3. **Contextual factors**:
   > "The 2021-2024 period encompasses unique economic conditions, including COVID-19 recovery and heightened regulatory scrutiny in Indonesia. These macro factors may have dampened the sensitivity of audit quality to firm-level variations in audit characteristics, as all firms faced similar external pressures."

**5.1.2 Significant control variables:**

**SIZE** (negative, p = 0.018):
> "The significant negative relationship between firm size and audit quality is counterintuitive but may reflect measurement artifacts in the audit quality score or scale effects. Larger firms may receive more conservative audit quality ratings due to heightened scrutiny, or the audit quality measure may not fully capture the complexity of auditing larger entities."

**ROA** (negative, p < 0.001):
> "The significant negative ROA coefficient should be interpreted cautiously due to extreme outliers in the data (range: -1391% to +3612%). After addressing these outliers through winsorization [if you do this], this relationship warrants further investigation."

#### 5.2 Comparison with Prior Literature

[Compare your null findings with existing studies]

**Table 5.1: Comparison with Prior Studies**
```
Study               Sample              Method          ARL Effect   FEE Effect   FO Effect
-----------------   -----------------   -------------   ----------   ----------   ---------
Current Study       Indonesia 2021-24   FE (clustered)  ns           ns           ns
[Author1, Year]     [Country, Years]    Cross-section   Positive***  -            -
[Author2, Year]     [Country, Years]    Pooled OLS      -            Positive**   -
[Author3, Year]     [Country, Years]    RE              -            -            Positive*
```

**Key discussion point**:
> "Our findings diverge from [Author1] and [Author2], who reported significant effects using cross-sectional designs. We attribute this difference to methodological rigor: our Fixed Effects approach controls for unobserved time-invariant heterogeneity that may confound cross-sectional estimates. Our results suggest that apparent relationships in cross-sectional studies may reflect between-firm differences rather than causal effects."

#### 5.3 Limitations

**Be transparent about limitations**:

1. **Small sample size**:
   > "Our sample of 63 companies over 4 years (N=252) provides limited statistical power, with only 3.7 observations per estimated parameter. This falls below the recommended 10-15:1 ratio, potentially contributing to wide standard errors and non-significant findings."

2. **Short time period**:
   > "The 4-year panel is at the minimum threshold for meaningful Fixed Effects estimation. Longer panels would provide greater within-firm variation and more precise estimates of dynamic relationships."

3. **Low within-variation**:
   > "Key independent variables, particularly foreign ownership (18.6% within-variation) and firm size (26.9% within-variation), show limited temporal variation within firms. Fixed Effects estimation relies exclusively on within-variation, which may be insufficient to detect true relationships in our data."

4. **Audit quality measurement**:
   > "[Discuss limitations of your AQMS measure - is it subjective? Does it capture all dimensions of audit quality? Is the 0-4 scale appropriate?]"

5. **Sector-specific focus**:
   > "Our exclusive focus on the consumer goods sector enhances internal validity by controlling for industry effects but limits generalizability to other sectors with different audit environments."

6. **ROA outliers**:
   > "Two extreme ROA observations (LAPD company: -1391%, +3612%) suggest potential data errors. While we [winsorized/excluded] these outliers in sensitivity analyses, they highlight data quality concerns that may affect other variables."

7. **Low model fit**:
   > "The extremely low within-R² (2.89%) indicates our model explains minimal within-firm variation in audit quality. This suggests important omitted variables or that audit quality is primarily determined by time-invariant factors not captured by our model."

#### 5.4 Contributions

**Even with null findings, you contribute**:

1. **Methodological contribution**:
   > "This study is among the first to apply rigorous panel data methods with Fixed Effects and clustered standard errors to audit quality research in Indonesia. Our methodological rigor provides more credible causal inference than cross-sectional designs prevalent in prior literature."

2. **Empirical contribution**:
   > "Our null findings challenge the generalizability of cross-sectional findings in prior studies. We demonstrate that relationships observed between firms may not hold when examining changes within firms over time, highlighting the importance of panel data methods in audit research."

3. **Contextual contribution**:
   > "This study provides updated evidence from the Indonesian context during the post-COVID period (2021-2024), contributing to the limited literature on emerging market audit quality."

#### 5.5 Implications

**Practical implications**:
> "For regulators and policymakers, our findings suggest that audit quality in Indonesia is primarily driven by structural firm characteristics rather than annually varying factors. Interventions targeting audit report timeliness or fee structures may have limited impact on audit quality. Instead, policies should focus on improving firm-level governance structures and auditor selection processes."

**Theoretical implications**:
> "Our results suggest that audit quality theories emphasizing dynamic relationships may require refinement in contexts where audit quality is stable within firms. Future theoretical work should distinguish between cross-sectional determinants (why some firms have higher quality audits) and longitudinal determinants (what causes audit quality to change within firms)."

---

### Chapter 6: Conclusion

**6.1 Summary of Findings**:
> "This study examined the determinants of audit quality in Indonesian consumer goods companies using panel data from 2021-2024. Employing Fixed Effects regression with entity-clustered standard errors, we tested three hypotheses regarding the effects of audit report lag, audit fees, and foreign ownership on audit quality. Contrary to expectations, none of these variables showed significant within-firm relationships with audit quality. Control variables firm size and ROA were significant, though the latter requires cautious interpretation due to outliers. These null findings suggest audit quality in Indonesia is primarily determined by time-invariant firm characteristics rather than annually varying audit attributes."

**6.2 Limitations and Future Research**:
> "Future research should address several limitations of this study. First, longer panel data (T≥10) would provide greater statistical power and allow examination of lagged effects. Second, expanding beyond the consumer goods sector would enhance generalizability. Third, alternative audit quality measures beyond AQMS could be examined. Fourth, researchers should investigate which time-invariant factors (e.g., auditor identity, firm reputation) explain the substantial between-firm variation in audit quality observed in our data."

---

## IV. RECOMMENDED ACTIONS BEFORE THESIS SUBMISSION

### 🔴 **CRITICAL** (Must do):

1. **Fix ROA outliers**:
   - [ ] Investigate LAPD company data for 2021-2022
   - [ ] Apply winsorization OR remove outliers
   - [ ] Re-run all analyses with corrected data
   - [ ] Document the correction in methods section

2. **Add variable measurement details**:
   - [ ] Explain how AQMS is calculated (what do scores 0-4 mean?)
   - [ ] Clarify if ARL is logged
   - [ ] Clarify if FEE is logged and in what currency
   - [ ] Document all variable transformations

3. **Address low within-R² explicitly**:
   - [ ] Add discussion in limitations
   - [ ] Report Random Effects as alternative specification
   - [ ] Explain the FE vs RE trade-off

### 🟡 **STRONGLY RECOMMENDED**:

4. **Add robustness checks**:
   - [ ] Between-Effects estimator (uses cross-sectional variation)
   - [ ] Pooled OLS with year fixed effects
   - [ ] Sensitivity analysis excluding outliers
   - [ ] Test alternative clustering (by year, two-way)

5. **Enhance descriptive statistics**:
   - [ ] Add table showing which companies/years have missing changes
   - [ ] Plot time trends for each variable
   - [ ] Show distribution of AQMS (is it ordinal? Continuous?)

6. **Improve visualizations**:
   - [ ] Scatter plots of DV vs each IV
   - [ ] Panel-specific trend lines (spaghetti plots)
   - [ ] Residual diagnostics plots

### 🟢 **OPTIONAL** (But helpful):

7. **Additional analyses**:
   - [ ] Test for non-linear relationships (quadratic terms)
   - [ ] Interaction effects (e.g., FO × SIZE)
   - [ ] Subgroup analysis (large vs small firms)
   - [ ] Dynamic panel model (include lagged DV)

---

## V. SAMPLE THESIS SECTIONS (Ready to Copy)

### Methods Section - Statistical Analysis

```
3.4.4 Panel Regression Model

We employ Fixed Effects (FE) regression to examine within-firm determinants
of audit quality. The FE model is specified as:

AQMS_it = β₁(ARL)_it + β₂(FEE)_it + β₃(FO)_it + β₄(SIZE)_it + β₅(ROA)_it + α_i + ε_it

where i indexes companies (i = 1, ..., 63), t indexes years (t = 2021, ..., 2024),
α_i represents time-invariant company-specific effects, and ε_it is the idiosyncratic
error term. The FE estimator eliminates α_i through within-group transformation,
providing consistent estimates even if unobserved firm characteristics correlate
with the independent variables (Wooldridge, 2010).

We use entity-clustered standard errors to account for potential autocorrelation
within firms over time (Petersen, 2009). This approach allows for arbitrary
within-firm correlation in the error structure while maintaining efficiency.

Model selection follows a systematic procedure. First, we test Fixed Effects
against Pooled OLS using the F-test for joint significance of entity effects.
Second, we test Random Effects against Pooled OLS using the Breusch-Pagan
Lagrange Multiplier test. Third, if both tests reject Pooled OLS, we apply
the Hausman test to choose between Fixed and Random Effects based on the
consistency of the Random Effects estimator.

All analyses are conducted using Python 3.11 with the linearmodels package
(Sheppard, 2021) for panel regression and statsmodels (Seabold & Perktold, 2010)
for diagnostic tests. Statistical significance is evaluated at the 1%, 5%, and
10% levels.
```

### Results Section - Model Selection

```
4.4 Model Selection

Table 4.4 presents results from the panel model selection tests. The F-test
strongly rejects the null hypothesis that all entity fixed effects are zero
(F = 14.19, p < 0.001), indicating significant heterogeneity across companies.
Similarly, the Breusch-Pagan LM test rejects the null of zero random effects
variance (χ² = 190.89, p < 0.001). Both tests conclusively favor panel methods
over Pooled OLS.

To choose between Fixed and Random Effects, we conduct the Hausman test, which
examines whether entity-specific effects correlate with the regressors. The test
strongly rejects the null hypothesis of no correlation (χ² = 39.65, df = 5,
p < 0.001), indicating systematic differences between FE and RE coefficients.
This result suggests that Random Effects estimates would be inconsistent, leading
us to prefer the Fixed Effects specification as our primary model.

[Insert Table 4.4 here]
```

### Results Section - Main Findings

```
4.5 Panel Regression Results

Table 4.5 reports coefficient estimates from Pooled OLS (column 1), Random
Effects (column 2), and Fixed Effects (column 3) models. We focus interpretation
on the Fixed Effects model following the Hausman test recommendation.

Contrary to our hypotheses, none of the primary independent variables demonstrate
statistical significance in the Fixed Effects specification. Hypothesis 1 posited
that audit report lag would negatively affect audit quality. However, the FE
coefficient is positive and non-significant (β = 0.261, SE = 0.173, p = 0.133),
providing no support for H1. Similarly, Hypothesis 2, predicting a positive
effect of audit fees, is not supported (β = -0.031, SE = 0.060, p = 0.606).
Hypothesis 3, proposing a positive foreign ownership effect, also receives no
empirical support (β = -0.081, SE = 0.283, p = 0.776).

Among the control variables, firm size exhibits a significant negative relationship
with audit quality (β = -0.069, SE = 0.029, p = 0.018), suggesting that within-firm
growth corresponds to decreased audit quality scores. Return on assets is also
significantly negative (β = -0.0002, SE = 0.00003, p < 0.001), though this result
warrants cautious interpretation due to outliers in the ROA distribution.

The Fixed Effects model's within-R² is notably low at 0.029, indicating that the
independent variables explain only 2.9% of the within-firm variation in audit
quality. This contrasts sharply with the Random Effects overall-R² of 0.727,
suggesting that audit quality variation is predominantly between firms rather
than within firms over time.

[Insert Table 4.5 here]
```

### Discussion - Interpreting Null Results

```
5.1.1 Interpretation of Non-Significant Findings

The lack of significant effects for our three primary hypotheses requires careful
interpretation. Several complementary explanations emerge from our analysis.

First, methodological factors play a crucial role. The Fixed Effects estimator
relies exclusively on within-firm variation, which is limited in our data.
Foreign ownership, for instance, exhibits a between-to-within variance ratio
of 5.32, meaning 84% of its variation is between firms rather than over time.
With most companies maintaining stable ownership structures over the 4-year
period, the Fixed Effects estimator has minimal identifying variation to detect
FO effects. This methodological constraint affects audit fees (71% between-firm)
and firm size (73% between-firm) as well, though to a lesser degree.

Second, the extremely low within-R² (2.9%) suggests a substantive interpretation:
audit quality in Indonesian consumer goods companies is primarily determined by
time-invariant firm characteristics rather than annually varying factors. These
time-invariant factors—absorbed by the entity fixed effects in our model—might
include auditor identity, long-term auditor-client relationships, firm reputation,
organizational culture, and persistent governance structures. Our findings imply
that these structural elements dominate short-term variations in audit report
lags, fees, or ownership composition.

Third, contextual factors specific to Indonesia during 2021-2024 may dampen the
sensitivity of audit quality to firm-level changes. This period encompasses
COVID-19 recovery and heightened regulatory scrutiny following high-profile
corporate governance failures. In such an environment, audit quality may have
become more standardized across firms and less responsive to individual firm
characteristics.

Finally, we note that the apparent discrepancy between our null findings and
some prior positive findings in the literature may stem from methodological
differences. Cross-sectional studies that report significant relationships may
be detecting spurious correlations driven by omitted time-invariant variables
rather than true causal effects. Our Fixed Effects approach, by controlling for
all time-invariant heterogeneity, provides more credible causal inference at
the cost of requiring temporal variation in both independent and dependent
variables.
```

---

## VI. FINAL CHECKLIST

### Before Thesis Defense:

- [ ] ROA outliers addressed (critical!)
- [ ] Variable measurement fully documented
- [ ] Low within-R² discussed in limitations
- [ ] All 4 notebooks run without errors
- [ ] All results files generated and reviewed
- [ ] Tables formatted for thesis
- [ ] Figures have proper captions
- [ ] Methods section fully written
- [ ] Results section fully written
- [ ] Discussion addresses null findings
- [ ] Limitations section complete
- [ ] References cited properly
- [ ] Advisor reviewed and approved methodology
- [ ] Data source properly cited
- [ ] Ethics approval obtained (if required)

### Defense Preparation:

**Anticipated Questions**:

1. **"Why are all your hypotheses not supported?"**
   - Answer: Explain low within-variation, methodological rigor of FE, time-invariant nature of audit quality

2. **"Why is your R² so low?"**
   - Answer: FE within-R² isolates temporal variation; audit quality is primarily structural (high between-R²)

3. **"Should you use Random Effects instead since it has better fit?"**
   - Answer: Hausman test indicates RE is inconsistent; FE is the consistent estimator despite lower fit

4. **"What about those ROA outliers?"**
   - Answer: Data error identified and corrected through [winsorization/exclusion]; results robust to this correction

5. **"Your sample is small - is it valid?"**
   - Answer: Represents full population of consumer goods sector with complete data; acknowledge power limitations

6. **"Can you compare to prior studies?"**
   - Answer: [Prepare comparison table with prior literature showing methodological differences]

---

## VII. CONCLUSION: IS YOUR METHODOLOGY SCIENTIFICALLY VALID?

### ✅ **YES, with caveats**:

**Your methodology is fundamentally sound**:
- Proper panel data methods
- Correct diagnostic tests
- Appropriate model selection
- Clustered SEs for autocorrelation
- Transparent reporting

**But you must address**:
1. ROA outliers (critical data error)
2. Low within-R² (discuss extensively)
3. Small sample limitations
4. Variable measurement documentation
5. Null findings interpretation

**Bottom line**: This is **publishable research** if you:
- Fix the ROA issue
- Thoroughly discuss limitations
- Frame null results as contribution
- Provide complete methodological transparency

**Grade potential** (assuming corrections):
- With improvements: **A- to A** (excellent methodology despite null findings)
- Without corrections: **B+ to B-** (valid but incomplete)

---

## VIII. REFERENCES TO ADD TO YOUR THESIS

```
Allison, P. D. (2009). Fixed effects regression models. Sage publications.

Baltagi, B. H. (2013). Econometric analysis of panel data (5th ed.). John Wiley & Sons.

Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference.
  Journal of Human Resources, 50(2), 317-372.

Hsiao, C. (2014). Analysis of panel data (3rd ed.). Cambridge University Press.

Petersen, M. A. (2009). Estimating standard errors in finance panel data sets: Comparing
  approaches. Review of Financial Studies, 22(1), 435-480.

Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data (2nd ed.).
  MIT Press.

[Add your domain-specific audit quality references here]
```

---

**END OF METHODOLOGY ASSESSMENT**

**Date**: 2025-12-12
**Reviewer**: Claude Code AI Assistant
**Recommendation**: APPROVE WITH REVISIONS (Focus on ROA outliers and limitations discussion)
