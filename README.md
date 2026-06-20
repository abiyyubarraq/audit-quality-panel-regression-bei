# The Effect of Audit Report Lag, Audit Fee, and Foreign Ownership on Audit Quality

**A panel data regression study of consumer non-cyclicals companies listed on the Indonesia Stock Exchange (BEI), 2021–2024.**

> **Resuming this project?** Jump to [Restore & Resume From Scratch](#restore--resume-from-scratch) below — it has the exact steps to get a fresh machine running, including where the data lives and what each document covers.

## Overview
Panel data regression analysis examining the impact of Audit Report Lag (ARL), Audit Fee (FEE), and Foreign Ownership (FO) on Audit Quality (AQMS) for consumer non-cyclicals companies listed on Indonesian Stock Exchange (BEI) from 2021-2024.

## Research Questions
1. Analyzing the effect of audit report lag on audit quality
2. Analyzing the effect of audit fee on audit quality
3. Analyzing the effect of foreign ownership on audit quality

## Variables
- **Dependent Variable**: AQMS (Audit Quality Measurement Score)
- **Independent Variables**:
  - ARL (Audit Report Lag) - days between fiscal year-end and audit report date
  - FEE (Audit Fee) - total audit fees paid
  - FO (Foreign Ownership) - percentage of foreign ownership
- **Control Variables**:
  - SIZE (Company Size) - natural log of total assets
  - ROA (Return on Assets) - profitability measure

## Project Structure
```
audit-quality-panel-regression-bei/
├── data/                   # Data files
│   ├── raw/               # Original Excel files
│   ├── processed/         # Processed JSON/CSV
│   └── interim/           # Intermediate files
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── models/           # Regression models
│   ├── tests/            # Statistical tests
│   ├── visualization/    # Plotting functions
│   └── utils/            # Utilities
├── tests/                 # Unit tests
├── config/                # Configuration files
├── results/               # Analysis outputs
└── logs/                  # Log files
```

## Installation

### 1. Clone or Download the Repository
```bash
git clone https://github.com/abiyyubarraq/audit-quality-panel-regression-bei.git
cd audit-quality-panel-regression-bei
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# For development (includes Jupyter, testing tools)
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

## Restore & Resume From Scratch

This repository is **self-contained** — everything needed to rerun the full analysis is committed, including the source data. Follow these steps on a fresh machine to pick the project back up.

### Prerequisites
- **Python 3.9–3.11** recommended. (Note: some scientific packages may lag the very newest Python; if `pip install` fails on a brand-new Python, install 3.11.)
- Git, and optionally JupyterLab for the notebooks.

### Steps
```bash
# 1. Clone the repo
git clone https://github.com/abiyyubarraq/audit-quality-panel-regression-bei.git
cd audit-quality-panel-regression-bei

# 2. Create + activate a virtual environment
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (Git Bash) / macOS / Linux:
source venv/bin/activate    # Windows bash: source venv/Scripts/activate

# 3. Install dependencies (dev set includes Jupyter + testing tools)
pip install -r requirements-dev.txt
pip install -e .            # install the `src` package in editable mode

# 4. Verify the install
pytest tests/               # run unit tests
jupyter lab                 # launch notebooks
```

### What's already in the repo (no setup needed)
| Path | Contents |
|------|----------|
| `data/raw/audit_data.xlsx` | **Source data** — the original dataset (committed; this is the only copy). |
| `data/processed/audit_data.json` | Cleaned/processed data used by the analysis. |
| `notebooks/01..06_*.ipynb` | The analysis workflow, run in order (see [Quick Start](#quick-start)). |
| `src/` | Reusable package: data loaders, panel models, statistical tests, plotting. |
| `config/default_config.yaml`, `config/variables.yaml` | All tunable settings and variable definitions. |
| `results/` | Committed outputs — figures (`.png`), tables (`.csv`), and model results (`.json`). |

### Key documents (read these to reload context)
| Document | What it explains |
|----------|------------------|
| `README.md` | This file — overview, methods, how to run. |
| `THESIS_METHODOLOGY_ASSESSMENT.md` | Full assessment of the research methodology. |
| `IMMEDIATE_ACTION_PLAN.md` | Outstanding tasks / next steps for the thesis. |
| `WHY_NEGATIVE_BETWEEN_R2_IS_OK.md` | Explanation of a specific statistical result (negative between-R²). |
| `docs/FE_vs_RE_explanation.md` | Fixed Effects vs. Random Effects rationale. |

### Reproduce the analysis end-to-end
Run the notebooks in `notebooks/` in numeric order (01 → 06). They read from `data/`, write fresh outputs to `results/`, and reproduce every figure and table from the committed source data.

## Quick Start

### 1. Prepare Your Data
The source Excel file is already committed at `data/raw/audit_data.xlsx`. To use a different dataset, replace that file (keep the same column structure defined in `config/variables.yaml`).

### 2. Run Analysis Notebooks
Open Jupyter Lab and run the notebooks in sequence:
```bash
jupyter lab
```

**Notebook Sequence:**
1. `01_data_exploration.ipynb` - Load and explore data
2. `02_descriptive_stats.ipynb` - Descriptive statistics
3. `03_diagnostic_tests.ipynb` - Classical assumption tests
4. `04_panel_data_analysis.ipynb` - Main regression analysis
5. `05_pooled_ols_comparison.ipynb` - Compare with pooled OLS
6. `06_visualization.ipynb` - Generate plots

### 3. Example: Run Statistical Tests
```python
from src.data.loader import DataLoader
from src.tests.test_suite import ComprehensiveTestSuite

# Load data
loader = DataLoader()
data = loader.load_json('data/processed/audit_data.json')

# Run diagnostic tests
test_suite = ComprehensiveTestSuite(alpha=0.05)
# ... (see notebooks for complete examples)
```

## Statistical Methods

### Classical Assumption Tests
- **Normality**: Shapiro-Wilk (default), Kolmogorov-Smirnov, Jarque-Bera
- **Heteroskedasticity**: Breusch-Pagan (default), White, Glejser
- **Autocorrelation**: Durbin-Watson (default), Breusch-Godfrey
- **Multicollinearity**: VIF (Variance Inflation Factor, threshold=10)

### Panel Data Methods
- **Fixed Effects (FE)** - Controls for time-invariant company characteristics
- **Random Effects (RE)** - More efficient if assumptions hold
- **Hausman Test** - Formal test to choose between FE and RE
- **Pooled OLS** - For comparison (ignores panel structure)

## Configuration
Edit `config/default_config.yaml` to customize:
- Significance levels (default: α = 0.05)
- Test preferences (which tests to run by default)
- VIF thresholds
- Robust standard errors

Edit `config/variables.yaml` to update:
- Variable names and descriptions
- Panel structure (entity and time variables)
- Data validation rules

## Why Panel Data Methods?

Panel data methods are appropriate for this research because:

1. **Data Structure**: Multiple companies observed over multiple years (2021-2024)
2. **Unobserved Heterogeneity**: Companies have time-invariant characteristics (management quality, audit complexity) that affect audit quality but aren't measured
3. **Within-Company Variation**: Leverages how changes in independent variables affect changes in audit quality for the same company over time
4. **Reduced Omitted Variable Bias**: Controls for unobserved entity-specific effects

## Expected Output

### Console Output Example
```
==============================================================
NORMALITY TESTS (Residuals)
==============================================================
Shapiro-Wilk Normality Test
Statistic: 0.9823
P-value: 0.1234
Reject H0: No
Interpretation: Data follows normal distribution at 0.05 level

==============================================================
HETEROSKEDASTICITY TESTS
==============================================================
Breusch-Pagan Test
Statistic: 12.45
P-value: 0.0145
Reject H0: Yes
Interpretation: Heteroskedasticity detected at 0.05 level

==============================================================
PANEL REGRESSION RESULTS
==============================================================
Fixed Effects Model
Coefficients:
  ARL: -0.0123 (p=0.032) *
  FEE:  0.0456 (p=0.001) ***
  FO:   0.0234 (p=0.156)

Hausman Test: p=0.0234
Recommendation: Use Fixed Effects model
```

## Project Features
- Modular, reusable code
- Multiple test variations with configurable defaults
- Comprehensive logging
- Clear console output for thesis documentation
- Visualization for residual diagnostics
- Reproducible research structure
- Unit tests for code validation

## Testing
Run unit tests:
```bash
pytest tests/

# With coverage
pytest --cov=src tests/
```

## Troubleshooting

### Import Errors
Make sure you've installed the package in editable mode:
```bash
pip install -e .
```

### Missing Dependencies
Reinstall requirements:
```bash
pip install -r requirements.txt --upgrade
```

## License
This project is for academic research purposes.

## Citation
[Add your thesis citation information here]

## Contact
[Your contact information]

## Acknowledgments
Research conducted for thesis at [Your University]
