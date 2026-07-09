"""Tests for driving_profiles.validation.population."""

import numpy as np
import pandas as pd
import pytest

from driving_profiles.generator import sample as sample_module
from driving_profiles.validation import population as pv


def _make_population(n: int, cluster_id: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "HOUSEID": [f"H{cluster_id}-{i}" for i in range(n)],
            "PERSONID": ["01"] * n,
            "age": rng.integers(20, 70, size=n),
            "age_band": rng.choice(["<25", "25-34", "35-44", "45-54", "55-64", "65+"], size=n),
            "worker_status": ["worker"] * n,
            "is_worker": pd.array([True] * n, dtype="boolean"),
            "household_income_bracket": rng.integers(1, 11, size=n).astype(float),
            "household_size": rng.integers(1, 6, size=n),
            "household_vehicle_count": rng.integers(0, 4, size=n),
            "vehicles_per_driver": rng.uniform(0.5, 2.0, size=n),
            "vehicle_per_driver_adequate": rng.random(size=n) > 0.2,
            "used_household_vehicle": rng.random(size=n) > 0.1,
            "commute_distance_survey_miles": rng.gamma(2, 5, size=n),
            "commute_duration_minutes": rng.gamma(2, 10, size=n),
            "work_arrival_time": rng.integers(600, 950, size=n).astype(float),
            "work_departure_time": rng.integers(1500, 1900, size=n).astype(float),
            "trips_per_day": rng.integers(1, 6, size=n),
            "number_of_stops": rng.integers(0, 5, size=n),
            "total_daily_miles": rng.gamma(2, 10, size=n),
            "total_driving_minutes": rng.gamma(2, 20, size=n),
            "average_trip_distance_miles": rng.gamma(2, 5, size=n),
            "cluster_id": pd.array([cluster_id] * n, dtype="Int64"),
        }
    )


@pytest.fixture
def source_and_synthetic():
    source = pd.concat(
        [_make_population(200, 0, seed=1), _make_population(80, 1, seed=2)], ignore_index=True
    )
    # Same generating distributions, different draw -> should mostly pass.
    synthetic = pd.concat(
        [_make_population(400, 0, seed=3), _make_population(160, 1, seed=4)], ignore_index=True
    )
    return source, synthetic


# --- load_* --------------------------------------------------------------------


def test_load_source_population_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        pv.load_source_population(tmp_path)


def test_load_synthetic_population_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        pv.load_synthetic_population(tmp_path)


def test_load_source_population_filters_to_clustered(tmp_path):
    df = _make_population(10, 0, seed=1)
    unclustered = _make_population(3, 0, seed=5)
    unclustered["cluster_id"] = pd.array([pd.NA] * 3, dtype="Int64")
    combined = pd.concat([df, unclustered], ignore_index=True)
    combined.to_parquet(tmp_path / sample_module.CLUSTER_TABLE_FILENAME, index=False)

    result = pv.load_source_population(tmp_path)

    assert len(result) == 10
    assert result["cluster_id"].notna().all()


def test_load_synthetic_population_reads_written_parquet(tmp_path):
    df = _make_population(5, 0, seed=1)
    df.to_parquet(tmp_path / sample_module.SYNTHETIC_EMPLOYEE_FILENAME, index=False)

    result = pv.load_synthetic_population(tmp_path)

    assert len(result) == 5


# --- iter_groups -----------------------------------------------------------------


def test_iter_groups_yields_pooled_and_per_cluster(source_and_synthetic):
    source, synthetic = source_and_synthetic

    labels = [group for group, _, _ in pv.iter_groups(source, synthetic)]

    assert labels[0] == "pooled"
    assert set(labels[1:]) == {"cluster_0", "cluster_1"}


def test_iter_groups_subsets_match_cluster_id(source_and_synthetic):
    source, synthetic = source_and_synthetic

    for group, s, y in pv.iter_groups(source, synthetic):
        if group == "pooled":
            continue
        cluster_id = int(group.split("_")[1])
        assert (s["cluster_id"] == cluster_id).all()
        assert (y["cluster_id"] == cluster_id).all()


# --- validate_* --------------------------------------------------------------------


def test_validate_demographics_produces_expected_metrics(source_and_synthetic):
    source, synthetic = source_and_synthetic

    result = pv.validate_demographics(source, synthetic)

    assert set(pv.DEMOGRAPHIC_KS_FEATURES) <= set(result["metric"])
    assert set(pv.DEMOGRAPHIC_CATEGORICAL_FEATURES) <= set(result["metric"])
    assert "is_worker" in result["metric"].tolist()


def test_validate_household_flags_unjittered_field_mismatch(source_and_synthetic):
    source, synthetic = source_and_synthetic
    synthetic = synthetic.copy()
    synthetic["vehicle_per_driver_adequate"] = False  # deliberately break it

    result = pv.validate_household(source, synthetic)

    row = result.loc[
        (result["metric"] == "vehicle_per_driver_adequate") & (result["group"] == "pooled")
    ].iloc[0]
    assert bool(row["passed"]) is False

def test_validate_commute_includes_percentile_check(source_and_synthetic):
    source, synthetic = source_and_synthetic

    result = pv.validate_commute(source, synthetic)

    assert "p90_diff" in result["test"].tolist()
    assert {"work_arrival_time_minutes", "work_departure_time_minutes"} <= set(result["metric"])


def test_validate_daily_mobility_compares_nonnull_subset(source_and_synthetic):
    source, synthetic = source_and_synthetic
    source = source.copy()
    source.loc[source.index[:5], "total_daily_miles"] = float("nan")

    result = pv.validate_daily_mobility(source, synthetic)

    row = result.loc[result["metric"] == "total_daily_miles_nonnull"].iloc[0]
    assert row["n_source"] < len(source)


def test_validate_daily_mobility_includes_variance_check(source_and_synthetic):
    source, synthetic = source_and_synthetic

    result = pv.validate_daily_mobility(source, synthetic)

    assert "variance_ratio" in result["test"].tolist()


# --- run_population_validation ----------------------------------------------------


def test_run_population_validation_combines_all_sections(source_and_synthetic):
    source, synthetic = source_and_synthetic

    result = pv.run_population_validation(source, synthetic)

    assert result["section"].eq("population").all()
    assert len(result) > 0
    assert result["passed"].notna().any()


def test_run_population_validation_can_load_from_disk(tmp_path):
    source_df = _make_population(30, 0, seed=1)
    source_df.to_parquet(tmp_path / sample_module.CLUSTER_TABLE_FILENAME, index=False)
    synthetic_df = _make_population(60, 0, seed=2)
    synthetic_df.to_parquet(tmp_path / sample_module.SYNTHETIC_EMPLOYEE_FILENAME, index=False)

    result = pv.run_population_validation(processed_dir=tmp_path)

    assert len(result) > 0
