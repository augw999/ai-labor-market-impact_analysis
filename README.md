# How AI Affects the U.S. Job Market

Authors: Aug Wu, Bouba Katompa, and Bridgid Cutright
CSE 163 AC

## Overview

This project investigates how AI Occupational Exposure (AIOE) relates to employment growth and wage dispersion changes across U.S. occupations from 2021 to 2024. It combines Bureau of Labor Statistics (BLS) Occupational Employment and Wage Statistics (OEWS) data with the Felten, Raj, and Seamans AIOE dataset to answer three research questions about the relationship between
AI exposure and labor market outcomes.

## Setup and Installation

### Required Software

- Python 3.11 or higher

### Required Libraries

Install the following libraries before running:

```bash
pip install pandas numpy scipy statsmodels seaborn matplotlib openpyxl
```

- `pandas` — data loading, merging, and manipulation
- `numpy` — weighted percentile calculations
- `scipy` — unweighted Welch t-tests
- `statsmodels` — employment-weighted Welch t-tests
- `seaborn` — scatter plots, box plots, and heatmaps
- `matplotlib` — plot rendering and saving
- `openpyxl` — reading `.xlsx` data files

### Required Data Files

The following data files must be placed in the *same directory* as the
Python scripts:

| File | Source |
|------|--------|
| `AIOE_DataAppendix.xlsx` | [GitHub — AIOE-Data/AIOE](https://github.com/AIOE-Data/AIOE) |
| `national_M2021_dl.xlsx` | [BLS OEWS Tables](https://www.bls.gov/oes/tables.htm) (2021 national data) |
| `national_M2022_dl.xlsx` | [BLS OEWS Tables](https://www.bls.gov/oes/tables.htm) (2022 national data) |
| `national_M2023_dl.xlsx` | [BLS OEWS Tables](https://www.bls.gov/oes/tables.htm) (2023 national data) |
| `national_M2024_dl.xlsx` | [BLS OEWS Tables](https://www.bls.gov/oes/tables.htm) (2024 national data) |

## File Descriptions

### Python Modules

| File | Description |
|------|-------------|
| `EDA_260304.py` | **EDA pipeline.** Loads the raw AIOE and OEWS Excel files, checks for missing data, drops empty columns, left-joins OEWS with AIOE by occupation code, filters unmatched rows and non-numeric placeholders, prints descriptive statistics, generates EDA visualizations (AIOE vs. wage scatter and employment-by-AIOE-bin bar charts for each year), and saves cleaned per-year CSV files for downstream analysis. |
| `analysis_260309.py` | **Analysis pipeline.** Loads the cleaned CSV files produced by the EDA step, merges them into a single wide-format panel, computes growth rates and wage dispersion ratios, assigns employment-weighted AIOE tiers, runs weighted and unweighted one-tailed Welch t-tests for RQ1 (employment growth) and RQ2 (wage dispersion change), assigns performance grades for RQ3, and generates all results visualizations (scatter plot, box plot, and heatmaps). |
| `testing.py` | **Testing file.** Contains unit tests for core EDA functions (missing-value handling, merge correctness, column filtering, numeric coercion) using small synthetic DataFrames, plus an optional integration test that runs the full EDA pipeline on the real datasets if the Excel files are present. |

### Data Files

| File | Description |
|------|-------------|
| `AIOE_DataAppendix.xlsx` | AI Occupational Exposure scores by SOC code (774 occupations). |
| `national_M2021_dl.xlsx` | BLS OEWS national employment and wage estimates for 2021. |
| `national_M2022_dl.xlsx` | BLS OEWS national employment and wage estimates for 2022. |
| `national_M2023_dl.xlsx` | BLS OEWS national employment and wage estimates for 2023. |
| `national_M2024_dl.xlsx` | BLS OEWS national employment and wage estimates for 2024. |
| `cleaned_data_2021.csv` | Cleaned and merged OEWS+AIOE data for 2021 (produced by EDA). |
| `cleaned_data_2022.csv` | Cleaned and merged OEWS+AIOE data for 2022 (produced by EDA). |
| `cleaned_data_2023.csv` | Cleaned and merged OEWS+AIOE data for 2023 (produced by EDA). |
| `cleaned_data_2024.csv` | Cleaned and merged OEWS+AIOE data for 2024 (produced by EDA). |

### Other Files

| File | Description |
|------|-------------|
| `report.pdf` | Final project report. |
| `README.md` | This file. |

## How to Reproduce Results

All commands should be run from the project directory where the Python scripts and data files are located.

### Step 1: Run the EDA pipeline

```bash
python EDA_260304.py
```

This will:
- Print data quality summaries, merge diagnostics, and descriptive statistics to the console.
- Save EDA visualizations as PNG files: `AIOE_Annual_Income_{year}.png` and `Total_Emp_per_AIOE_Range_{year}.png` for each year 2021–2024.
- Save cleaned CSV files: from `cleaned_data_2021.csv` through `cleaned_data_2024.csv`.

### Step 2: Run the analysis pipeline

```bash
python analysis_260309.py
```

This requires the `cleaned_data_{year}.csv` files from Step 1 to be in the same directory. It will:
- Print aggregated growth metrics, t-test results (weighted and unweighted means, p-values), and RQ3 performance grade distributions to the console.
- Save results visualizations as PNG files: `rq1.png` (scatter plot), `rq2.png` (box plot), `rq3_emp.png` and `rq3_cnt.png` (heatmaps).

### Step 3: Run the tests

```bash
python testing.py
```

This will:
- Run all unit tests using synthetic DataFrames (no external data needed).
- If the raw Excel files are present, also run an optional integration test on the full datasets.
- Print "All tests passed" upon success.
