"""Tests for driving_profiles.validation.common."""

import numpy as np
import pandas as pd

from driving_profiles.validation import common

# --- ks_result ------------------------------------------------------------


def test_ks_result_passes_for_identical_distributions():
    rng = np.random.default_rng(0)
    values = pd.Series(rng.normal(size=200))

    result = common.ks_result("s", "m", values, values)

    assert result["passed"] is True
    assert result["p_value"] == 1.0


def test_ks_result_fails_for_clearly_different_distributions():
    rng = np.random.default_rng(0)
    source = pd.Series(rng.normal(loc=0, size=200))
    synthetic = pd.Series(rng.normal(loc=50, size=200))

    result = common.ks_result("s", "m", source, synthetic)

    assert result["passed"] is False
    assert result["p_value"] < 0.05


def test_ks_result_drops_nan_before_comparing():
    source = pd.Series([1.0, 2.0, 3.0, float("nan"), float("nan")])
    synthetic = pd.Series([1.0, 2.0, 3.0])

    result = common.ks_result("s", "m", source, synthetic)

    assert result["n_source"] == 3
    assert result["n_synthetic"] == 3


def test_ks_result_returns_none_passed_when_insufficient_data():
    result = common.ks_result("s", "m", pd.Series([1.0]), pd.Series([1.0, 2.0, 3.0]))

    assert result["passed"] is None
    assert "insufficient" in result["detail"]


def test_ks_result_row_has_expected_schema_keys():
    result = common.ks_result("section", "metric", pd.Series([1, 2, 3]), pd.Series([1, 2, 3]))

    assert set(result.keys()) == set(common.RESULT_COLUMNS)


# --- variance_ratio_result --------------------------------------------------


def test_variance_ratio_result_passes_when_variances_match():
    source = pd.Series([1, 2, 3, 4, 5] * 20)
    synthetic = pd.Series([1, 2, 3, 4, 5] * 20)

    result = common.variance_ratio_result("s", "m", source, synthetic)

    assert result["passed"] is True
    assert result["statistic"] == 0.0


def test_variance_ratio_result_fails_when_synthetic_variance_flattened():
    source = pd.Series(list(range(100)))
    synthetic = pd.Series([50] * 100)  # zero variance

    result = common.variance_ratio_result("s", "m", source, synthetic)

    assert result["passed"] is False


# --- chi_square_result -------------------------------------------------------


def test_chi_square_result_passes_for_matching_categorical_shares():
    source = pd.Series(["a"] * 50 + ["b"] * 50)
    synthetic = pd.Series(["a"] * 500 + ["b"] * 500)

    result = common.chi_square_result("s", "m", source, synthetic)

    assert result["passed"] is True


def test_chi_square_result_fails_for_shifted_categorical_shares():
    source = pd.Series(["a"] * 90 + ["b"] * 10)
    synthetic = pd.Series(["a"] * 10 + ["b"] * 90)

    result = common.chi_square_result("s", "m", source, synthetic)

    assert result["passed"] is False


def test_chi_square_result_handles_no_observations():
    empty = pd.Series([], dtype=object)
    result = common.chi_square_result("s", "m", empty, empty)

    assert result["passed"] is None


# --- proportion_result --------------------------------------------------------


def test_proportion_result_passes_within_tolerance():
    source = pd.Series([True] * 80 + [False] * 20)
    synthetic = pd.Series([True] * 81 + [False] * 19)

    result = common.proportion_result("s", "m", source, synthetic, max_diff_pp=2.0)

    assert result["passed"] is True


def test_proportion_result_fails_outside_tolerance():
    source = pd.Series([True] * 80 + [False] * 20)
    synthetic = pd.Series([True] * 50 + [False] * 50)

    result = common.proportion_result("s", "m", source, synthetic, max_diff_pp=2.0)

    assert result["passed"] is False


# --- percentile_result --------------------------------------------------------


def test_percentile_result_passes_when_tail_matches():
    source = pd.Series(range(1, 101))
    synthetic = pd.Series(range(1, 101))

    result = common.percentile_result("s", "m", source, synthetic, percentile=90)

    assert result["passed"] is True
    assert result["statistic"] == 0.0


def test_percentile_result_fails_when_tail_diverges():
    source = pd.Series(range(1, 101))
    synthetic = pd.Series(list(range(1, 91)) + [1000] * 10)

    result = common.percentile_result("s", "m", source, synthetic, percentile=95, max_rel_diff=0.10)

    assert result["passed"] is False


# --- structural_result --------------------------------------------------------


def test_structural_result_passes_with_zero_violations():
    result = common.structural_result("s", "m", n_violations=0, n_checked=100)

    assert result["passed"] is True


def test_structural_result_fails_with_any_violation():
    result = common.structural_result("s", "m", n_violations=1, n_checked=100)

    assert result["passed"] is False


# --- results_frame -------------------------------------------------------------


def test_results_frame_empty_has_expected_columns():
    df = common.results_frame([])

    assert list(df.columns) == common.RESULT_COLUMNS
    assert len(df) == 0


def test_results_frame_concatenates_rows():
    rows = [common.result_row("s", "m1"), common.result_row("s", "m2")]

    df = common.results_frame(rows)

    assert len(df) == 2
    assert list(df["metric"]) == ["m1", "m2"]
