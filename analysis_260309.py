"""
Aug Wu, Bouba Bubanji Katompa, and Bridgid Cutright
CSE 163 AC
Analysis pipeline for the final project.

This module:
- Loads pre-processed occupation data from cleaned CSV files produced by EDA
- Merges four yearly DataFrames into a single wide-format panel
- Computes year-over-year and 2022-2024 growth rates for employment and wages
- Computes wage dispersion ratios and their growth rates per occupation
- Assigns employment-weighted AIOE, employment growth, and dispersion growth
  tier labels (H / M / L) using quartile cutoffs
- Aggregates employment-weighted average wages by AIOE tier and year
- Runs unweighted and employment-weighted one-tailed Welch t-tests for RQ1
  (employment growth) and RQ2 (wage dispersion change)
- Assigns performance grades (1-5) by combining employment and dispersion tiers
- Computes AIOE tier distributions within each performance grade for RQ3
- Produces and saves visualizations for RQ1 (scatter), RQ2 (boxplot), and
  RQ3 (heatmap)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans

# Global variables
FILE_NM_PREFIX = 'cleaned_data_'


def load_files() -> dict[int, pd.DataFrame]:
    """
    Loads pre-processed OEWS+AIOE CSV files for years 2021-2024 and returns
    a dictionary mapping each year (int) to its DataFrame.
    """
    dfs = {}

    for i in range(2021, 2025):
        dfs[i] = pd.read_csv(FILE_NM_PREFIX + str(i) + '.csv')

    return dfs


def merge_dfs(dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Given a dictionary of yearly DataFrames, merges them into a single
    wide-format panel with one row per occupation and year-suffixed columns
    (e.g. TOT_EMP_21, A_MEDIAN_22).Also adds a AVG_TOT_EMP column
    (the mean of 2022-2024 employment). Returns the merged DataFrame.
    """
    # create a new DataFrame
    merged_df = pd.DataFrame()

    # Insert 2021 data
    merged_df[['OCC_CODE', 'OCC_TITLE', 'AIOE', 'TOT_EMP_21',
               'A_PCT25_21', 'A_MEDIAN_21', 'A_PCT75_21']] =\
        dfs[2021][['OCC_CODE', 'OCC_TITLE', 'AIOE', 'TOT_EMP',
                   'A_PCT25', 'A_MEDIAN', 'A_PCT75']]

    # Insert 2022 - 2024 data
    for i in range(2022, 2025):
        yr = i - 2000
        df = pd.DataFrame()
        df[['OCC_CODE', f'TOT_EMP_{yr}',
            f'A_PCT25_{yr}', f'A_MEDIAN_{yr}', f'A_PCT75_{yr}']] =\
            dfs[i][['OCC_CODE', 'TOT_EMP',
                    'A_PCT25', 'A_MEDIAN', 'A_PCT75']]
        merged_df = merged_df.merge(df, how='inner', on='OCC_CODE')

    merged_df['AVG_TOT_EMP'] = (merged_df['TOT_EMP_22'] +
                                merged_df['TOT_EMP_23'] +
                                merged_df['TOT_EMP_24']) // 3

    return merged_df


def calculate_weighted_percentile(merged_df: pd.DataFrame,
                                  col_name: str) -> tuple[float, float]:
    """
    Given the merged DataFrame and a column name, computes and returns the
    employment-weighted 25th and 75th percentiles of that column
    """
    col = list(merged_df[col_name])
    avg_emp_col = list(merged_df['AVG_TOT_EMP'])

    percentile_25 = np.percentile(col, 25, method='inverted_cdf',
                                  weights=avg_emp_col)
    percentile_75 = np.percentile(col, 75, method='inverted_cdf',
                                  weights=avg_emp_col)

    return percentile_25, percentile_75


def add_cat_labels(merged_df: pd.DataFrame,
                   percentile_25: float, percentile_75: float,
                   attr: str) -> None:
    """
    Given the merged DataFrame, employment-weighted percentile cutoffs, and
    an attribute name, assigns tier labels to a new column.
    Modifies the DataFrame in place and returns None.
    """
    merged_df[f"{attr}_CAT"] = 'M'

    merged_df.loc[merged_df[attr] <= percentile_25, f"{attr}_CAT"] = 'L'
    merged_df.loc[merged_df[attr] > percentile_75, f"{attr}_CAT"] = 'H'


def add_yoy_product_cols(merged_df: pd.DataFrame) -> None:
    """
    Given the merged DataFrame, computes and adds the following columns
    in place:
    - Product_{col}_{yr}: wage multiplied by employment, which will be used
      for weighted aggregation in aggregate_df
    - {col}_YoY_{yr}: year-over-year growth rate for each attribute in 22-24
    - {col}_22-24 and {col}_21-24: cumulative growth rates over each window
    - dispersion_ratio_{yr} for each year and each attribute
    - DISP_GROWTH_22-24 and DISP_GROWTH_21-24 as the relative change in
      dispersion ratio over each window
    """
    cols = ['TOT_EMP', 'A_PCT25', 'A_MEDIAN', 'A_PCT75']

    for col in cols:
        for i in range(21, 25):
            # Calculate the product of different wage and employment
            if col != 'TOT_EMP':
                merged_df[f"Product_{col}_{i}"] = merged_df[f"{col}_{i}"] *\
                    merged_df[f"TOT_EMP_{i}"]
            # Calculate yearly growth rate on all columns
            if i != 21:
                merged_df[f"{col}_YoY_{i}"] = merged_df[f"{col}_{i}"] /\
                    merged_df[f"{col}_{i - 1}"] - 1

        # Calculate 22-24 and 21-24 growth rate for each metric
        merged_df[f"{col}_22-24"] = merged_df[f"{col}_24"] /\
            merged_df[f"{col}_22"] - 1
        merged_df[f"{col}_21-24"] = merged_df[f"{col}_24"] /\
            merged_df[f"{col}_21"] - 1

    # Calculate wage dispersion ratio
    for i in range(21, 25):
        merged_df[f"dispersion_ratio_{i}"] = (merged_df[f"A_PCT75_{i}"] -
                                              merged_df[f"A_PCT25_{i}"]) /\
                                              merged_df[f"A_MEDIAN_{i}"]
    # Calculate dispersion ratio's increase rate
    merged_df['DISP_GROWTH_22-24'] =\
        merged_df['dispersion_ratio_24'] / merged_df['dispersion_ratio_22'] - 1
    merged_df['DISP_GROWTH_21-24'] =\
        merged_df['dispersion_ratio_24'] / merged_df['dispersion_ratio_21'] - 1


def aggregate_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the merged DataFrame, groups occupations by AIOE_CAT and computes
    employment-weighted average wages (incl. A_PCT25, A_MEDIAN, A_PCT75)
    and total employment for each tier and year. Appends a Total row summing
    across all tiers. And adds dispersion_ratio_{yr} for each year.
    Returns the aggregated DataFrame object.
    """
    agg_df = merged_df.groupby('AIOE_CAT')[['TOT_EMP_21', 'Product_A_PCT25_21',
                                            'Product_A_MEDIAN_21',
                                            'Product_A_PCT75_21',
                                            'TOT_EMP_22', 'Product_A_PCT25_22',
                                            'Product_A_MEDIAN_22',
                                            'Product_A_PCT75_22',
                                            'TOT_EMP_23', 'Product_A_PCT25_23',
                                            'Product_A_MEDIAN_23',
                                            'Product_A_PCT75_23',
                                            'TOT_EMP_24', 'Product_A_PCT25_24',
                                            'Product_A_MEDIAN_24',
                                            'Product_A_PCT75_24']].sum()
    agg_df.loc['Total'] = agg_df.sum(axis=0)

    cols = ['TOT_EMP', 'A_PCT25', 'A_MEDIAN', 'A_PCT75']
    for i in range(21, 25):
        for col in cols:
            if col != 'TOT_EMP':
                agg_df[f"{col}_{i}"] = agg_df[f"Product_{col}_{i}"] /\
                    agg_df[f"TOT_EMP_{i}"]
                agg_df.drop(f"Product_{col}_{i}", axis=1, inplace=True)

        agg_df[f"dispersion_ratio_{i}"] = (agg_df[f"A_PCT75_{i}"] -
                                           agg_df[f"A_PCT25_{i}"]) /\
            agg_df[f"A_MEDIAN_{i}"]

    return agg_df


def cal_agg_growth(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the aggregated DataFrame from aggregate_df, computes year-over-year
    and cumulative growth rates (2021-2024 and 2022-2024) for total employment,
    wage percentiles, and dispersion ratios at each tier level. Returns a new
    DataFrame of growth metrics.
    """
    agg_growth = pd.DataFrame()

    cols = ['TOT_EMP', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'dispersion_ratio']

    for col in cols:
        for i in range(22, 25):
            agg_growth[f"{col}_YoY_{i}"] = agg_df[f"{col}_{i}"] /\
                agg_df[f"{col}_{i - 1}"] - 1

        agg_growth[f"{col}_GROWTH_21-24"] = agg_df[f"{col}_24"] /\
            agg_df[f"{col}_21"] - 1
        agg_growth[f"{col}_GROWTH_22-24"] = agg_df[f"{col}_24"] /\
            agg_df[f"{col}_22"] - 1

    return agg_growth


def divide_merged_df(merged_df: pd.DataFrame,
                     attr: str) -> tuple[pd.DataFrame,
                                         pd.DataFrame,
                                         pd.DataFrame]:
    """
    Given the merged DataFrame and an attribute name, splits the DataFrame
    into three subsets by the attribute's tier label column.
    Returns a tuple of (merged_h, merged_m, merged_l) for H, M, and L tiers.
    """
    merged_h = merged_df[merged_df[f"{attr}_CAT"] == 'H']
    merged_m = merged_df[merged_df[f"{attr}_CAT"] == 'M']
    merged_l = merged_df[merged_df[f"{attr}_CAT"] == 'L']

    return merged_h, merged_m, merged_l


def get_weight_list(merged_h: pd.DataFrame, merged_m: pd.DataFrame,
                    merged_l: pd.DataFrame) -> tuple[pd.DataFrame,
                                                     pd.DataFrame,
                                                     pd.DataFrame]:
    """
    Given the H, M, and L tier DataFrames, computes normalized employment
    weights for each tier. Returns a tuple of (weight_h, weight_m, weight_l).
    """
    weight_h = merged_h['TOT_EMP_22'] / merged_h['TOT_EMP_22'].mean()
    weight_m = merged_m['TOT_EMP_22'] / merged_m['TOT_EMP_22'].mean()
    weight_l = merged_l['TOT_EMP_22'] / merged_l['TOT_EMP_22'].mean()

    return weight_h, weight_m, weight_l


def unweighted_t_test(merged_h: pd.DataFrame, merged_m: pd.DataFrame,
                      merged_l: pd.DataFrame, attr: str) -> None:
    """
    Given the H, M, and L tier DataFrames and an attribute column name,
    runs a one-tailed t-test comparing H vs. L tiers and prints the
    one-tailed p-value.
    """
    labels = ['H', 'M', 'L']
    mergeds = [merged_h, merged_m, merged_l]

    for categorized_df, label in zip(mergeds, labels):
        df = categorized_df[attr]
        print(f"During the period, AIOE-{label} {attr}'s unweighted mean is " +
              f"{df.mean():.4f}, median is {df.median():.4f}, " +
              f"standard deviation is {df.std():.4f}")

    t_stat, p_welch = stats.ttest_ind(merged_h[attr], merged_l[attr],
                                      equal_var=False)
    print(f"The one-tailed p-vale is: {p_welch / 2}")


def weighted_t_test(merged_h: pd.DataFrame, merged_m: pd.DataFrame,
                    merged_l: pd.DataFrame, attr: str) -> None:
    """
    Given the H, M, and L tier DataFrames and an attribute column name,
    runs a one-tailed employment-weighted t-test comparing H vs. L and
    prints the one-tailed p-value.
    """
    weight_h, weight_m, weight_l =\
        get_weight_list(merged_h, merged_m, merged_l)

    weighted_merged_h = DescrStatsW(merged_h[attr], weights=weight_h)
    weighted_merged_m = DescrStatsW(merged_m[attr], weights=weight_m)
    weighted_merged_l = DescrStatsW(merged_l[attr], weights=weight_l)

    print(f"H mean is {weighted_merged_h.mean}")
    print(f"M mean is {weighted_merged_m.mean}")
    print(f"L mean is {weighted_merged_l.mean}")

    cm = CompareMeans(weighted_merged_h, weighted_merged_l)
    t_stat, p_one, df = cm.ttest_ind(usevar='unequal', alternative='larger')
    print(f"The one-tailed p-value is: {p_one}")


def generate_distribution(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the merged DataFrame, creates an aggregated DataFrame by the AIOE,
    employment growth, and dispersion growth tiers, and adds a performance
    grade (1-5) based on the combinations of the employment growth and
    dispersion growth tiers following the rule: 1 (HH), 2 (HM/MH),
    3 (MM/HL/LH), 4 (LM/ML), 5 (LL).
    Returns the resulting DataFrame.
    """
    category_cnt_df = merged_df[['AIOE_CAT', 'TOT_EMP_22-24_CAT',
                                 'DISP_GROWTH_22-24_CAT'
                                 ]].value_counts().reset_index()

    category_emp_df = merged_df.groupby(['AIOE_CAT', 'TOT_EMP_22-24_CAT',
                                         'DISP_GROWTH_22-24_CAT'
                                         ])['AVG_TOT_EMP'].sum().reset_index()

    category_df = category_cnt_df.merge(category_emp_df, how='inner',
                                        on=['AIOE_CAT', 'TOT_EMP_22-24_CAT',
                                            'DISP_GROWTH_22-24_CAT'])

    category_df['Performance_grade'] = 3

    category_df.loc[(category_df['TOT_EMP_22-24_CAT'] == 'H') &
                    (category_df['DISP_GROWTH_22-24_CAT'] == 'H'),
                    'Performance_grade'] = 1
    category_df.loc[(category_df['TOT_EMP_22-24_CAT'] == 'L') &
                    (category_df['DISP_GROWTH_22-24_CAT'] == 'L'),
                    'Performance_grade'] = 5
    category_df.loc[((category_df['TOT_EMP_22-24_CAT'] == 'H') &
                     (category_df['DISP_GROWTH_22-24_CAT'] == 'M')) |
                    ((category_df['TOT_EMP_22-24_CAT'] == 'M') &
                     (category_df['DISP_GROWTH_22-24_CAT'] == 'H')),
                    'Performance_grade'] = 2
    category_df.loc[((category_df['TOT_EMP_22-24_CAT'] == 'L') &
                     (category_df['DISP_GROWTH_22-24_CAT'] == 'M')) |
                    ((category_df['TOT_EMP_22-24_CAT'] == 'M') &
                     (category_df['DISP_GROWTH_22-24_CAT'] == 'L')),
                    'Performance_grade'] = 4

    return category_df


def get_occupations(merged_df: pd.DataFrame, aioe_cat: str,
                    emp_growth_cat: str, disp_growth_cat: str) -> str:
    """
    Given the merged DataFrame and tier labels for AIOE, employment growth,
    and dispersion growth, filters occupations matching all three categories
    and returns their titles as a string, sorted by 2024 employment
    and median wage in descending order.
    """
    target_group = merged_df[(merged_df['AIOE_CAT'] == aioe_cat) &
                             (merged_df['TOT_EMP_22-24_CAT'] ==
                              emp_growth_cat) &
                             (merged_df['DISP_GROWTH_22-24_CAT'] ==
                              disp_growth_cat)]

    output = "; ".join(list(target_group.sort_values(['TOT_EMP_24',
                                                      'A_MEDIAN_24'
                                                      ], ascending=False
                                                     )['OCC_TITLE']))

    return output


def make_plots_rq1(merged_df: pd.DataFrame) -> None:
    """
    Given the merged DataFrame, generates and saves a scatter plot of AIOE
    score vs. employment growth rate (2022-2024).
    """
    sns.set_theme()

    cat_colors = {'H': '#DD8451',
                  'M': '#4C72B1',
                  'L': '#55A868'}

    sns.scatterplot(data=merged_df, x='AIOE', y='TOT_EMP_22-24',
                    hue='AIOE_CAT', palette=cat_colors, legend=True)

    for cat, group in merged_df.groupby('AIOE_CAT'):
        sns.regplot(data=group, x='AIOE', y='TOT_EMP_22-24',
                    scatter=False, lowess=True,
                    color=cat_colors[cat])

    plt.xlabel("AIOE Score")
    plt.ylabel("Employment growth from 2022 to 2024")
    plt.title("Employment Growth vs. AIOE Score (2022–2024)")

    plt.savefig("rq1.png")
    plt.close()


def make_plots_rq2(merged_df: pd.DataFrame) -> None:
    """
    Given the merged DataFrame, generates and saves a box plot of wage
    dispersion ratio growth rate (2022-2024) by AIOE tier.
    """
    sns.set_theme()

    cat_colors = {'H': '#DD8451',
                  'M': '#4C72B1',
                  'L': '#55A868'}

    sns.boxplot(data=merged_df, x='AIOE_CAT', y='DISP_GROWTH_22-24',
                hue='AIOE_CAT', palette=cat_colors, legend=True, fliersize=0)

    plt.axhline(0, color='lightgray', linestyle='--', linewidth=2)

    plt.xlabel("AIOE Tier")
    plt.ylabel("Dispersion Growth (2022–2024)")
    plt.title("Dispersion Growth for Each AIOE Tier(2022-2024)")
    plt.ylim(-0.6, 0.4)

    plt.savefig("rq2.png")
    plt.close()


def make_plots_rq3(performance_dist_df: pd.DataFrame,
                   attr: str, filename: str) -> None:
    """
    Given the performance distribution DataFrame from generate_distribution,
    an attribute column (either 'count' for occupation count or 'AVG_TOT_EMP'
    for employment count), and an output filename, generates and saves a png
    showing the percentage of each performance grade's total attr that falls
    within each AIOE tier (H / M / L). Saves the figure to the specified
    filename.
    """
    sns.set_theme()

    names = {
        'AVG_TOT_EMP': 'Employment',
        'count': '# of Occuapation'
    }

    cats = ['H', 'M', 'L']
    grades = {1: 'Very Good',
              2: 'Good',
              3: 'Average',
              4: 'Poor',
              5: 'Very Poor'}

    performance_attrs =\
        performance_dist_df.groupby('Performance_grade')[attr].sum()

    heat_data = {}
    for grade, name in grades.items():
        this_heat = {}
        for cat in cats:
            subtotal =\
                performance_dist_df[(performance_dist_df['Performance_grade']
                                     == grade) &
                                    (performance_dist_df['AIOE_CAT'] == cat)
                                    ][attr].sum()
            this_heat[cat] =\
                round(subtotal / performance_attrs[grade] * 100, 1)
        heat_data[grade] = this_heat

    heat_df = pd.DataFrame(heat_data, index=cats)

    sns.heatmap(heat_df, annot=True, fmt='.1f', cmap='Blues',
                linewidths=0.5,
                cbar_kws={'label': '% within Performance grade'})

    plt.xlabel("Performance Grade")
    plt.ylabel("AIOE Category")
    plt.title(f"AIOE Cat's {names[attr]} Dist " +
              "for each Performance Grade (2022-2024)")

    plt.savefig(filename)
    plt.close()


def main():
    dfs = load_files()
    merged_df = merge_dfs(dfs)

    aioe_pct25, aioe_pct75 =\
        calculate_weighted_percentile(merged_df, 'AIOE')
    add_cat_labels(merged_df, aioe_pct25, aioe_pct75, "AIOE")

    add_yoy_product_cols(merged_df)

    agg_df = aggregate_df(merged_df)
    agg_growth = cal_agg_growth(agg_df)
    print(agg_growth)

    merged_h, merged_m, merged_l = divide_merged_df(merged_df, "AIOE")

    unweighted_t_test(merged_h, merged_m, merged_l, "TOT_EMP_22-24")
    weighted_t_test(merged_h, merged_m, merged_l, "TOT_EMP_22-24")

    unweighted_t_test(merged_h, merged_m, merged_l,
                      "DISP_GROWTH_22-24")
    weighted_t_test(merged_h, merged_m, merged_l,
                    "DISP_GROWTH_22-24")

    print(merged_df.columns)
    emp_growth_pct25, emp_growth_pct75 =\
        calculate_weighted_percentile(merged_df, "TOT_EMP_22-24")
    add_cat_labels(merged_df, emp_growth_pct25, emp_growth_pct75,
                   "TOT_EMP_22-24")

    disp_growth_pct25, disp_growth_pct75 =\
        calculate_weighted_percentile(merged_df,
                                      "DISP_GROWTH_22-24")
    add_cat_labels(merged_df, disp_growth_pct25, disp_growth_pct75,
                   "DISP_GROWTH_22-24")

    performance_dist_df = generate_distribution(merged_df)
    print(performance_dist_df)
    print()
    print(get_occupations(merged_df, 'H', 'H', 'H'))
    print()
    print(get_occupations(merged_df, 'H', 'L', 'L'))

    make_plots_rq1(merged_df)
    make_plots_rq2(merged_df)
    make_plots_rq3(performance_dist_df, 'AVG_TOT_EMP', 'rq3_emp.png')
    make_plots_rq3(performance_dist_df, 'count', 'rq3_cnt.png')


if __name__ == '__main__':
    main()
