"""Tests for driving_profiles.validation.clusters."""

import numpy as np
import pandas as pd
import pytest

from driving_profiles.validation import clusters as cv


def _cluster_population(n0: int, n1: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n0):
        rows.append(
            {
                "cluster_id": pd.array([0], dtype="Int64")[0],
                "commute_distance_survey_miles": rng.normal(5, 1),
                "commute_duration_minutes": rng.normal(15, 2),
                "work_arrival_time": rng.normal(800, 20),
                "work_departure_time": rng.normal(1630, 20),
                "trips_per_day": rng.integers(2, 4),
                "total_daily_miles": rng.normal(10, 2),
                "total_driving_minutes": rng.normal(20, 3),
                "number_of_stops": rng.integers(1, 3),
                "vehicles_per_driver": rng.normal(1.0, 0.2),
            }
        )
    for i in range(n1):
        rows.append(
            {
                "cluster_id": pd.array([1], dtype="Int64")[0],
                "commute_distance_survey_miles": rng.normal(40, 5),
                "commute_duration_minutes": rng.normal(60, 5),
                "work_arrival_time": rng.normal(900, 20),
                "work_departure_time": rng.normal(1800, 20),
                "trips_per_day": rng.integers(5, 8),
                "total_daily_miles": rng.normal(80, 10),
                "total_driving_minutes": rng.normal(100, 10),
                "number_of_stops": rng.integers(4, 7),
                "vehicles_per_driver": rng.normal(1.3, 0.2),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def populations():
    source = _cluster_population(100, 20, seed=1)
    synthetic = _cluster_population(400, 80, seed=2)
    return source, synthetic


# --- validate_cluster_proportions ------------------------------------------------


def test_validate_cluster_proportions_passes_when_matched(populations):
    source, synthetic = populations

    result = cv.validate_cluster_proportions(source, synthetic)

    assert result["passed"].all()


def test_validate_cluster_proportions_fails_on_skewed_synthetic_mix():
    source = _cluster_population(50, 50, seed=1)
    synthetic = _cluster_population(90, 10, seed=2)

    result = cv.validate_cluster_proportions(source, synthetic, max_diff_pp=2.0)

    assert not result["passed"].all()


# --- profile_cluster_centroids ----------------------------------------------------


def test_profile_cluster_centroids_has_one_row_per_cluster_feature(populations):
    source, _ = populations

    profile = cv.profile_cluster_centroids(source, "source")

    assert set(profile["cluster_id"]) == {0, 1}
    assert "commute_distance_survey_miles" in profile["feature"].tolist()
    assert (profile["dataset"] == "source").all()


# --- validate_cluster_separation --------------------------------------------------


def test_validate_cluster_separation_reports_large_effect_size_for_separated_clusters(populations):
    source, synthetic = populations

    result = cv.validate_cluster_separation(source, synthetic)

    commute = result.loc[
        (result["metric"] == "commute_distance_survey_miles_effect_size")
        & (result["group"] == "source")
    ]
    assert abs(commute.iloc[0]["statistic"]) > 1.0  # clusters are clearly separated by construction


def test_validate_cluster_separation_is_informational_only(populations):
    source, synthetic = populations

    result = cv.validate_cluster_separation(source, synthetic)

    assert result["passed"].isna().all()


def test_validate_cluster_separation_empty_when_not_two_clusters():
    df = pd.DataFrame({"cluster_id": pd.array([0, 0, 1, 1, 2, 2], dtype="Int64")})
    for col in [
        "commute_distance_survey_miles",
        "commute_duration_minutes",
        "work_arrival_time",
        "work_departure_time",
        "trips_per_day",
        "total_daily_miles",
        "total_driving_minutes",
        "number_of_stops",
        "vehicles_per_driver",
    ]:
        df[col] = [1.0] * 6

    result = cv.validate_cluster_separation(df, df)

    assert result.empty


# --- hierarchical_cross_check -----------------------------------------------------


def test_hierarchical_cross_check_returns_merge_heights(tmp_path, monkeypatch):
    from driving_profiles.features import cluster as cluster_module

    rng = np.random.default_rng(0)
    n = 60
    features = pd.DataFrame(
        {
            "HOUSEID": [str(i) for i in range(n)],
            "PERSONID": ["01"] * n,
            "is_worker": pd.array([True] * n, dtype="boolean"),
            "work_trip_count": [1] * n,
        }
    )
    for col in cluster_module.CONTINUOUS_FEATURES:
        features[col] = rng.normal(size=n)
    for col in cluster_module.BOOLEAN_FEATURES:
        features[col] = True
    for col in cluster_module.EXCLUDED_FEATURES:
        if col not in features.columns:
            features[col] = 0

    features.to_parquet(tmp_path / cluster_module.FEATURE_TABLE_FILENAME, index=False)

    result = cv.hierarchical_cross_check(processed_dir=tmp_path, sample_size=50, seed=0)

    assert not result.empty
    assert set(result["k"]) <= set(range(2, 9))
    assert (result["n_sampled"] <= 50).all()


# --- run_cluster_validation --------------------------------------------------------


def test_run_cluster_validation_combines_proportions_and_separation(populations):
    source, synthetic = populations

    result = cv.run_cluster_validation(source, synthetic)

    assert "cluster_share" in result["metric"].tolist()
    assert any(result["metric"].str.endswith("_effect_size"))
