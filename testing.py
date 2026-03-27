"""
Aug Wu, Bouba Bubanji Katompa, and Bridgid Cutright

Testing file for EDA_260304.py and analysis_260309.py.

Run:
    python testing.py

This file uses small, synthetic DataFrames to test core logic without relying
on external Excel files.
"""

import numpy as np
import pandas as pd
import os

import EDA_260304 as eda
import analysis_260309 as ana


def _make_aioe_df() -> pd.DataFrame:
    """
    Create a small synthetic AIOE DataFrame for testing.
    """
    return pd.DataFrame({
        "SOC Code": ["11-1011", "13-2011", None],
        "AIOE": [0.5, -1.2, 0.1],
        "Other": [1, 2, 3],
    })


def _make_oews_df() -> pd.DataFrame:
    """
    Create a small synthetic OEWS DataFrame for testing.
    """
    return pd.DataFrame({
        "OCC_CODE": ["11-1011", "15-9999"],
        "OCC_TITLE": ["Chief Exec", "Mystery Job"],
        "O_GROUP": ["detailed", "detailed"],
        "TOT_EMP": [1000, 50],
        "A_MEDIAN": [200000, 60000],
        "JOBS_1000": [np.nan, np.nan],
        "LOC_QUOTIENT": [np.nan, np.nan],
    })


def _make_oews_dict() -> dict[int, pd.DataFrame]:
    """
    Create a dictionary of synthetic OEWS DataFrames for multiple years.
    """
    return {year: _make_oews_df().copy() for year in range(2021, 2025)}


def _make_analysis_df() -> pd.DataFrame:
    """
    Create a small synthetic merged-style DataFrame for analysis tests.
    """
    return pd.DataFrame({
        "OCC_CODE": ["11-1011", "13-2011", "15-1256"],
        "OCC_TITLE": ["Chief Exec", "Accountant", "Software Dev"],
        "AIOE": [0.8, 0.0, -0.5],
        "TOT_EMP_21": [100, 200, 300],
        "TOT_EMP_22": [110, 210, 310],
        "TOT_EMP_23": [120, 220, 320],
        "TOT_EMP_24": [130, 230, 330],
        "A_PCT25_21": [10, 20, 30],
        "A_PCT25_22": [11, 21, 31],
        "A_PCT25_23": [12, 22, 32],
        "A_PCT25_24": [13, 23, 33],
        "A_MEDIAN_21": [20, 30, 40],
        "A_MEDIAN_22": [21, 31, 41],
        "A_MEDIAN_23": [22, 32, 42],
        "A_MEDIAN_24": [23, 33, 43],
        "A_PCT75_21": [30, 40, 50],
        "A_PCT75_22": [31, 41, 51],
        "A_PCT75_23": [32, 42, 52],
        "A_PCT75_24": [33, 43, 53],
    })


def test_preliminary_data_check() -> None:
    """
    Test that preliminary_data_check drops missing SOC codes and bad columns.
    """
    aioe = _make_aioe_df()
    oews = _make_oews_dict()

    drop_cols = ["JOBS_1000", "LOC_QUOTIENT", "PCT_TOTAL"]
    aioe2, oews2 = eda.preliminary_data_check(aioe, oews, drop_cols)

    assert aioe2.shape[0] == 2
    assert aioe2["SOC Code"].isna().sum() == 0

    for year in range(2021, 2025):
        assert "JOBS_1000" not in oews2[year].columns
        assert "LOC_QUOTIENT" not in oews2[year].columns
        assert "PCT_TOTAL" not in oews2[year].columns


def test_data_join() -> None:
    """
    Test that data_join preserves left-join rows and merges matching AIOE data.
    """
    aioe = _make_aioe_df().dropna()
    oews = _make_oews_dict()
    for year in oews:
        oews[year] = oews[year].drop(columns=["JOBS_1000", "LOC_QUOTIENT"])

    combined = eda.data_join(aioe, oews)

    for year in range(2021, 2025):
        assert combined[year].shape[0] == oews[year].shape[0]

        matched = combined[year][combined[year]["OCC_CODE"]
                                 == "11-1011"].iloc[0]
        assert matched["SOC Code"] == "11-1011"
        assert np.isclose(matched["AIOE"], 0.5)

        unmatched = combined[year][combined[year]["OCC_CODE"] ==
                                   "15-9999"].iloc[0]
        assert pd.isna(unmatched["SOC Code"])
        assert pd.isna(unmatched["AIOE"])


def test_data_check() -> None:
    """
    Test that data_check removes unmatched rows with missing joined values.
    """
    aioe = _make_aioe_df().dropna()
    oews = _make_oews_dict()
    for year in oews:
        oews[year] = oews[year].drop(columns=["JOBS_1000", "LOC_QUOTIENT"])

    combined = eda.data_join(aioe, oews)
    cleaned = eda.data_check(combined)

    for year in range(2021, 2025):
        assert cleaned[year].shape[0] == 1
        assert cleaned[year]["OCC_CODE"].iloc[0] == "11-1011"


def test_narrow_down_columns() -> None:
    """
    Test that narrow_down_columns keeps only requested columns and
    returns copies.
    """
    base = {
        year: pd.DataFrame({
            "OCC_CODE": ["11-1011"],
            "OCC_TITLE": ["Chief Exec"],
            "TOT_EMP": [1000],
            "A_MEDIAN": [200000],
            "AIOE": [0.5],
            "EXTRA": [123],
        })
        for year in range(2021, 2025)
    }

    key = ["OCC_CODE", "OCC_TITLE", "TOT_EMP", "A_MEDIAN", "AIOE"]
    narrowed = eda.narrow_down_columns(base, key)

    for year in range(2021, 2025):
        assert list(narrowed[year].columns) == key

        narrowed[year].loc[0, "TOT_EMP"] = 999
        assert base[year].loc[0, "TOT_EMP"] == 1000


def test_describe_key_features() -> None:
    """
    Test that describe_key_features coerces numeric columns and drops bad rows.
    """
    narrowed = {
        year: pd.DataFrame({
            "OCC_CODE": ["11-1011", "13-2011"],
            "OCC_TITLE": ["Chief Exec", "Accountant"],
            "TOT_EMP": ["1000", "bad"],
            "A_MEDIAN": ["200000", "70000"],
            "AIOE": ["0.5", "oops"],
        })
        for year in range(2021, 2025)
    }

    num = ["TOT_EMP", "A_MEDIAN", "AIOE"]
    cat = ["OCC_CODE", "OCC_TITLE"]
    cleaned = eda.describe_key_features(narrowed, num, cat)

    for year in range(2021, 2025):
        assert cleaned[year].shape[0] == 1
        assert cleaned[year]["OCC_CODE"].iloc[0] == "11-1011"
        assert cleaned[year]["TOT_EMP"].dtype.kind in "if"


def test_add_yoy_product_cols() -> None:
    """Test that add_yoy_product_cols creates expected columns."""
    base = _make_analysis_df()

    ana.add_yoy_product_cols(base)

    assert "TOT_EMP_22-24" in base.columns
    assert "A_MEDIAN_22-24" in base.columns
    assert "Product_A_MEDIAN_22" in base.columns
    assert "dispersion_ratio_24" in base.columns


def test_add_cat_labels() -> None:
    """
    Test that add_cat_labels creates the category column.
    """
    base = _make_analysis_df()

    ana.add_cat_labels(base, 0.0, 0.5, "AIOE")

    assert "AIOE_CAT" in base.columns

    labels = set(base["AIOE_CAT"])
    assert labels.issubset({"H", "M", "L"})


def test_divide_merged_df() -> None:
    """
    Test that divide_merged_df splits rows correctly.
    """
    base = _make_analysis_df()

    ana.add_cat_labels(base, 0.0, 0.5, "AIOE")

    merged_h, merged_m, merged_l = ana.divide_merged_df(base, "AIOE")

    total = (
        merged_h.shape[0] +
        merged_m.shape[0] +
        merged_l.shape[0]
    )

    assert total == base.shape[0]


def test_generate_distribution() -> None:
    """
    Test that generate_distribution returns a valid summary DataFrame.
    """
    base = _make_analysis_df()

    base["AIOE_CAT"] = ["H", "M", "L"]
    base["TOT_EMP_22-24_CAT"] = ["H", "M", "L"]
    base["DISP_GROWTH_22-24_CAT"] = ["H", "M", "L"]
    base["AVG_TOT_EMP"] = [100, 200, 300]

    result = ana.generate_distribution(base)

    assert "Performance_grade" in result.columns
    assert result.shape[0] > 0


def test_get_occupations() -> None:
    """
    Test that get_occupations returns occupation names.
    """
    base = _make_analysis_df()

    base["AIOE_CAT"] = ["H", "H", "L"]
    base["TOT_EMP_22-24_CAT"] = ["H", "H", "L"]
    base["DISP_GROWTH_22-24_CAT"] = ["H", "M", "L"]

    result = ana.get_occupations(base, "H", "H", "H")

    assert isinstance(result, str)
    assert "Chief Exec" in result


def run_all_tests() -> None:
    """Run all unit tests for the EDA pipeline."""
    test_preliminary_data_check()
    test_data_join()
    test_data_check()
    test_narrow_down_columns()
    test_describe_key_features()

    test_add_yoy_product_cols()
    test_add_cat_labels()
    test_divide_merged_df()
    test_generate_distribution()
    test_get_occupations()

    print("All tests passed")


def optional_integration_run() -> None:
    """
    Run the full pipeline if required input files are available locally.
    """
    required = [
        eda.AIOE_FILE,
        eda.OEWS_FILES[2021],
        eda.OEWS_FILES[2022],
        eda.OEWS_FILES[2023],
        eda.OEWS_FILES[2024],
    ]
    if all(os.path.exists(path) for path in required):
        aioe_df, oews_dfs = eda.load_data()
        drop_cols = [
            "JOBS_1000", "LOC_QUOTIENT", "PCT_TOTAL",
            "PCT_RPT", "ANNUAL", "HOURLY"
        ]
        aioe_df, oews_dfs = eda.preliminary_data_check(
            aioe_df, oews_dfs, drop_cols
        )
        combined = eda.data_join(aioe_df, oews_dfs)
        cleaned = eda.data_check(combined)
        narrowed = eda.narrow_down_columns(
            cleaned, ["OCC_CODE", "OCC_TITLE", "TOT_EMP", "A_MEDIAN", "AIOE"]
        )
        eda.describe_key_features(
            narrowed, ["TOT_EMP", "A_MEDIAN", "AIOE"],
            ["OCC_CODE", "OCC_TITLE"]
        )
        print("Integration run completed (skipped visualization)")
    else:
        print("Integration run skipped (required Excel files not found).")


if __name__ == "__main__":
    run_all_tests()
    optional_integration_run()
