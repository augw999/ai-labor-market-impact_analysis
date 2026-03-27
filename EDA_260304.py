"""
Aug Wu, Bouba Bubanji Katompa, and Bridgid Cutright
CSE 163 AC
Exploratory Data Analysis (EDA) pipeline for the final project.

This module:
- Loads AIOE and OEWS datasets from Excel files
- Performs basic missingness checks and drops known empty OEWS columns
- Left-joins OEWS (by OCC_CODE) with AIOE (by SOC Code)
- Summarizes key numeric and categorical features
- Produces and saves visualizations for AIOE vs. wage and employment by AIOE
  bin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Global variables
AIOE_FILE = 'AIOE_DataAppendix.xlsx'
AIOE_SHEET = 'Appendix A'

OEWS_FILES = {
    2024: 'national_M2024_dl.xlsx',
    2023: 'national_M2023_dl.xlsx',
    2022: 'national_M2022_dl.xlsx',
    2021: 'national_M2021_dl.xlsx'
}

OEWS_SHEETS = {
    2024: 'national_M2024_dl',
    2023: 'national_M2023_dl',
    2022: 'national_M2022_dl',
    2021: 'national_M2021_dl'
}


def load_data() -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Loads the AIOE table and OEWS tables by year from Excel files and then
    returns a tuple of aioe_df and oews_dfs where:
        - aioe_df is the AIOE DataFrame loaded from AIOE_FILE/AIOE_SHEET
        - oews_dfs maps each year (int) to that year's OEWS DataFrame
    """
    aioe_df = pd.read_excel(AIOE_FILE, sheet_name=AIOE_SHEET)

    oews_dfs = {}
    for i in range(2021, 2025):
        oews_dfs[i] = pd.read_excel(OEWS_FILES[i], sheet_name=OEWS_SHEETS[i])

    return aioe_df, oews_dfs


def preliminary_data_check(aioe_df: pd.DataFrame,
                           oews_dfs: dict[int, pd.DataFrame],
                           drop_columns: list[str]) -> tuple[
                               pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Given the AIOE DataFrame, a dictionary of OEWS DataFrames by year,
    and a list of column names to be dropped, prints basic missing-value
    information before and after dropping known empty OEWS columns, returns:
    - the AIOE DataFrame after dropping rows with missing values, and
    - the OEWS DataFrames after removing the specified empty columns.
    """
    aioe_shape_original = aioe_df.shape
    print(f"The AIOE dataset has {aioe_shape_original[0]} rows " +
          f"and {aioe_shape_original[1]} columns.")
    aioe_df = aioe_df.dropna()
    aioe_shape_dropped = aioe_df.shape
    aioe_change_rows = aioe_shape_original[0] - aioe_shape_dropped[0]
    print(f"{aioe_change_rows} rows has been dropped due to missing values.\n")

    print("Before dropping any columns:")
    for i in range(2021, 2025):
        oews_curr = oews_dfs[i]
        curr_rows = oews_curr.shape[0]
        oews_remaining = oews_curr.dropna()
        oews_rows_remaining = oews_remaining.shape[0]
        print(f"Year {i}, OEWS file has {curr_rows} rows and " +
              f"{oews_curr.shape[1]} columns. " +
              f"And {oews_rows_remaining} out of {curr_rows} " +
              "row remains after dropping missing values.")

    print("\nAfter dropping 6 empty columns:")
    for i in range(2021, 2025):
        oews_dfs[i] = oews_dfs[i].drop(columns=drop_columns, errors='ignore')
        oews_shape_original = oews_dfs[i].shape
        oews_curr = oews_dfs[i].dropna()
        oews_shape_remaining = oews_curr.shape
        oews_change_rows = oews_shape_original[0] - oews_shape_remaining[0]
        print(f"In year {i}, {oews_change_rows} rows has been dropped due to "
              + "missing values.")
    print()

    return aioe_df, oews_dfs


def data_join(aioe_df: pd.DataFrame,
              oews_dfs: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """
    Given the AIOE DataFrame and a dictionary of OEWS DataFrames by year,
    performs a left join of each OEWS table with the AIOE table. Returns
    a dictionary mapping each year to its merged DataFrame.
    """
    combined = {}
    for i in range(2021, 2025):
        combined[i] = pd.merge(oews_dfs[i], aioe_df, how='left',
                               left_on='OCC_CODE', right_on='SOC Code')

    return combined


def data_check(combined: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """
    Given merged OEWS+AIOE DataFrames by year, prints summary information for
    each year, including:
    - the merged table shape,
    - 'O_GROUP' distributions for matched vs. unmatched occupation codes,
    - any matched rows with 'O_GROUP' as 'broad',
    - how many rows are removed after dropping missing values, and
    - the total employee numbers and percent of employees in unmatched
      'detailed' rows (based on TOT_EMP).
    Returns a dictionary mapping each year to the merged DataFrame after
    dropping rows with missing values.
    """
    # The numbers of rows and columns
    combined_shape_original = {}
    for i in range(2021, 2025):
        combined_shape_original[i] = combined[i].shape
        print(f"The combined table for Year {i} has " +
              f"{combined_shape_original[i][0]} rows " +
              f"and {combined_shape_original[i][1]} columns.")

    # The numbers of missing data
    c_dropped = {}
    c_dropped_shape = {}
    for i in range(2021, 2025):
        matched = combined[i][combined[i]['SOC Code'].notna()]
        unmatched = combined[i][combined[i]['SOC Code'].isna()]
        # Unmatched rows
        print(f"Year {i} it has {len(unmatched)} unmatched rows")
        print(f"\nYear {i}'s unmathced records' O_GROUP distribution is:")
        print(unmatched['O_GROUP'].value_counts())

        # Matched rows
        print(f"\nYear {i}'s mathced records' O_GROUP distribution is:")
        print(matched['O_GROUP'].value_counts())
        broad_filtered = matched['O_GROUP'] == 'broad'
        print("\nThe matched broad 'O_GROUP' is:")
        print(matched[broad_filtered][['OCC_CODE', 'O_GROUP', 'OCC_TITLE']])

        # After dropping known-empty columns and not treating suppressed
        # placeholders as missing, OEWS has no nulls in the fields used for
        # the merge and analysis. Therefore, nulls introduced in AIOE fields
        # after the left join indicate unmatched OCC_CODE values.
        c_dropped[i] = combined[i].dropna()
        c_dropped_shape[i] = c_dropped[i].shape
        difference = combined_shape_original[i][0] - \
            c_dropped_shape[i][0]
        print(f"The combined table for Year {i} has " +
              f"{difference} rows affected by unmatched OCC-code.")
        tot_emp_matched = matched.\
            loc[matched['O_GROUP'] == 'detailed', 'TOT_EMP'].sum()
        tot_emp_unmatched = unmatched.\
            loc[unmatched['O_GROUP'] == 'detailed', 'TOT_EMP'].sum()
        tot_emp = tot_emp_matched + tot_emp_unmatched
        tot_perc = round(tot_emp_unmatched / tot_emp * 100, 1)
        print("The unmateched records account for " +
              f"{tot_emp_unmatched} people, {tot_perc}% of total employees")

    return c_dropped


def narrow_down_columns(
        combined: dict[int, pd.DataFrame],
        key_features: list[str]) -> dict[int, pd.DataFrame]:
    """
    Given merged OEWS+AIOE DataFrames by year and a list of key feature column
    names, selects those columns for each year and returns a dictionary mapping
    each year to its narrowed DataFrame containing only the selected features.
    """
    narrowed = {}
    for i in range(2021, 2025):
        narrowed[i] = combined[i].loc[:, key_features].copy()

    return narrowed


def describe_key_features(
        narrowed: dict[int, pd.DataFrame],
        num_features: list[str],
        cat_features: list[str]) -> dict[int, pd.DataFrame]:
    """
    Given narrowed OEWS+AIOE DataFrames by year, a list of numeric feature
    names, and a list of categorical feature names, coerces the numeric
    features to numeric dtype when possible, drops rows with missing values in
    those numeric features, and prints descriptive statistics for both numeric
    and categorical features for each year. Returns a dictionary mapping each
    year to its cleaned DataFrame after the numeric-type conversion and
    row filtering.
    """
    for i in range(2021, 2025):
        for feature in num_features:
            narrowed[i][feature] =\
                pd.to_numeric(narrowed[i][feature], errors='coerce')

        prev_rows = narrowed[i].shape[0]
        narrowed[i] = narrowed[i].dropna(subset=num_features)
        post_rows = narrowed[i].shape[0]

        print(f"\nIn year {i},")
        print(f"{prev_rows - post_rows} rows have been removed due to " +
              "non-numeric values in the numeric features")
        print(narrowed[i][num_features].describe())
        print(narrowed[i][cat_features].describe())

    return narrowed


def visualization(df: dict[int, pd.DataFrame]) -> None:
    """
    Given (merged and narrowed) OEWS+AIOE DataFrames by year, generates and
    saves two plots for each year:
    - A scatter plot showing the relationship between AIOE and median annual
      income.
    - A bar chart showing total employment number across different AIOE ranges.
    """
    sns.set_theme()

    for i in range(2021, 2025):
        plt.figure()
        sns.scatterplot(data=df[i], x='A_MEDIAN', y='AIOE')
        plt.xlabel("Median annual income by occupation")
        plt.ylabel("AI Occupational Exposure (AIOE)")
        plt.title(f"Relationship between AIOE and Annual Income for Year {i}")
        plt.savefig(f"AIOE_Annual_Income_{i}.png")
        plt.show()

        plt.figure(figsize=(10, 10))
        plot_df = df[i].copy()
        x = 1.5
        dividers = [-2 * x, -x, 0, x, 2 * x]
        plot_df['AIOE_bin'] = pd.cut(plot_df['AIOE'], bins=dividers)
        agg = plot_df.\
            groupby('AIOE_bin', observed=True)['TOT_EMP'].sum().reset_index()
        sns.barplot(data=agg, x='AIOE_bin', y='TOT_EMP')
        plt.xlabel("AI Occupational Exposure (AIOE)")
        plt.xticks(rotation=45)
        plt.ylabel("Number of Total Employees")
        plt.title("Number of Total Employment in Different AIOE Range "
                  + f"for Year {i}")
        plt.savefig(f"Total_Emp_per_AIOE_Range_{i}.png")
        plt.show()


def save_files(df: dict[int, pd.DataFrame]) -> None:
    """
    Given (merged and narrowed) OEWS+AIOE DataFrames by year,
    saves them to csv files for further analysis.
    """
    for i in range(2021, 2025):
        df[i].to_csv(f"cleaned_data_{i}.csv", index=False)


def main() -> None:
    aioe_df, oews_dfs = load_data()

    drop_columns = ['JOBS_1000', 'LOC_QUOTIENT', 'PCT_TOTAL', 'PCT_RPT',
                    'ANNUAL', 'HOURLY']
    aioe_df, oews_dfs = preliminary_data_check(aioe_df, oews_dfs, drop_columns)

    combined_dfs = data_join(aioe_df, oews_dfs)
    combined_dfs = data_check(combined_dfs)

    key_features = ['OCC_CODE', 'OCC_TITLE', 'TOT_EMP',
                    'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'AIOE']
    narrowed_dfs = narrow_down_columns(combined_dfs, key_features)

    num_features = ['TOT_EMP', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'AIOE']
    cat_features = ['OCC_CODE', 'OCC_TITLE']
    neat_df = describe_key_features(narrowed_dfs, num_features, cat_features)

    visualization(neat_df)

    save_files(neat_df)


if __name__ == "__main__":
    main()
