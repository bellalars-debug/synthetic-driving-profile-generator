"""Tests for driving_profiles.features.cluster."""

import numpy as np
import pandas as pd
import pytest

from driving_profiles.features import cluster as cl

FEATURE_COLUMNS = [
    "age",
    "age_band",
    "worker_status",
    "is_worker",
    "household_income_bracket",
    "household_size",
    "household_vehicle_count",
    "work_trip_count",
    "commute_distance_survey_miles",
    "commute_distance_trip_miles",
    "commute_duration_minutes",
    "work_arrival_time",
    "work_departure_time",
    "trips_per_day",
    "total_daily_miles",
    "total_driving_minutes",
    "number_of_stops",
    "average_trip_distance_miles",
    "vehicles_per_driver",
    "vehicle_per_driver_adequate",
    "household_vehicle_trip_count",
    "used_household_vehicle",
]


def _employee_row(house_id: str, person_id: str = "01", **overrides) -> dict:
    """One clustering-eligible employee row with sensible defaults for
    every employee_features.parquet column; override individual fields per
    test."""
    row = {
        "HOUSEID": house_id,
        "PERSONID": person_id,
        "age": 40,
        "age_band": "35-44",
        "worker_status": "worker",
        "is_worker": True,
        "household_income_bracket": 7.0,
        "household_size": 3,
        "household_vehicle_count": 2,
        "work_trip_count": 1,
        "commute_distance_survey_miles": 10.0,
        "commute_distance_trip_miles": 10.0,
        "commute_duration_minutes": 20.0,
        "work_arrival_time": 830.0,
        "work_departure_time": 1700.0,
        "trips_per_day": 4,
        "total_daily_miles": 25.0,
        "total_driving_minutes": 45.0,
        "number_of_stops": 2,
        "average_trip_distance_miles": 6.25,
        "vehicles_per_driver": 1.5,
        "vehicle_per_driver_adequate": True,
        "household_vehicle_trip_count": 3,
        "used_household_vehicle": True,
    }
    row.update(overrides)
    return row


def _employee_features_df(*rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    df["HOUSEID"] = df["HOUSEID"].astype(str)
    df["PERSONID"] = df["PERSONID"].astype(str)
    df["is_worker"] = df["is_worker"].astype("boolean")
    return df


def _two_cluster_population(n_per_group: int = 15) -> pd.DataFrame:
    """A population with two well-separated commute-behavior groups (short
    vs. long commute) plus a non-worker and a no-commute worker, to
    exercise both the clustering split and the population filter."""
    rows = []
    for i in range(n_per_group):
        rows.append(
            _employee_row(
                f"S{i}",
                commute_distance_survey_miles=3.0 + 0.1 * i,
                commute_duration_minutes=10.0 + 0.1 * i,
                work_arrival_time=800.0,
                work_departure_time=1630.0,
                total_daily_miles=8.0 + 0.1 * i,
                total_driving_minutes=15.0 + 0.1 * i,
            )
        )
        rows.append(
            _employee_row(
                f"L{i}",
                commute_distance_survey_miles=40.0 + 0.1 * i,
                commute_duration_minutes=55.0 + 0.1 * i,
                work_arrival_time=900.0,
                work_departure_time=1800.0,
                total_daily_miles=85.0 + 0.1 * i,
                total_driving_minutes=100.0 + 0.1 * i,
            )
        )
    rows.append(_employee_row("NW1", is_worker=False, work_trip_count=0))
    rows.append(_employee_row("NC1", work_trip_count=0))
    return _employee_features_df(*rows)


# --- select_clustering_features ----------------------------------------------


def test_select_clustering_features_applies_population_filter():
    df = _two_cluster_population(n_per_group=3)

    result = cl.select_clustering_features(df)

    # Population filter drops the non-worker and the no-commute worker.
    assert len(result) == len(df) - 2
    assert "NW1" not in result["HOUSEID"].tolist()
    assert "NC1" not in result["HOUSEID"].tolist()


def test_select_clustering_features_drops_undefined_vehicle_ratio():
    df = _employee_features_df(
        _employee_row("H1", vehicles_per_driver=float("nan")),
        _employee_row("H2"),
    )

    result = cl.select_clustering_features(df)

    assert result["HOUSEID"].tolist() == ["H2"]


def test_select_clustering_features_returns_only_id_and_selected_columns():
    df = _two_cluster_population(n_per_group=2)

    result = cl.select_clustering_features(df)

    assert list(result.columns) == cl.PERSON_KEY + cl.CLUSTERING_FEATURES
    assert set(cl.CLUSTERING_FEATURES).isdisjoint(cl.EXCLUDED_FEATURES)


def test_excluded_features_and_clustering_features_cover_all_columns():
    all_columns = set(FEATURE_COLUMNS) | set(cl.PERSON_KEY)
    assert all_columns - set(cl.CLUSTERING_FEATURES) == set(cl.EXCLUDED_FEATURES)


# --- preprocess_features -------------------------------------------------------


def test_preprocess_features_separates_ids_from_model_matrix():
    selected = cl.select_clustering_features(_two_cluster_population(n_per_group=3))

    ids, X = cl.preprocess_features(selected)

    assert list(ids.columns) == cl.PERSON_KEY
    assert "HOUSEID" not in X.columns
    assert "PERSONID" not in X.columns
    assert len(ids) == len(X) == len(selected)


def test_preprocess_features_zero_fills_no_driving_trip_columns():
    selected = cl.select_clustering_features(
        _two_cluster_population(n_per_group=2)
    )
    selected.loc[0, "total_daily_miles"] = float("nan")
    selected.loc[0, "total_driving_minutes"] = float("nan")

    _, X = cl.preprocess_features(selected)

    # After z-score scaling, a filled 0 should be the most-negative value
    # in a column whose real values are all positive.
    assert X.loc[0, "total_daily_miles"] == X["total_daily_miles"].min()


def test_preprocess_features_median_imputes_true_missing_values():
    rows = [_employee_row(f"H{i}", commute_distance_survey_miles=10.0 + i) for i in range(5)]
    df = _employee_features_df(*rows)
    df.loc[0, "commute_distance_survey_miles"] = float("nan")
    selected = cl.select_clustering_features(df)

    _, X = cl.preprocess_features(selected)

    assert not X["commute_distance_survey_miles"].isna().any()


def test_preprocess_features_scales_continuous_and_passes_through_booleans():
    selected = cl.select_clustering_features(_two_cluster_population(n_per_group=10))

    _, X = cl.preprocess_features(selected)

    for column in cl.CONTINUOUS_FEATURES:
        assert X[column].mean() == pytest.approx(0.0, abs=1e-8)
    for column in cl.BOOLEAN_FEATURES:
        assert set(X[column].unique()).issubset({0, 1})


# --- determine_optimal_clusters -------------------------------------------------


def test_determine_optimal_clusters_returns_metrics_per_k():
    selected = cl.select_clustering_features(_two_cluster_population(n_per_group=15))
    _, X = cl.preprocess_features(selected)

    evaluation = cl.determine_optimal_clusters(X, k_range=range(2, 5), random_state=0)

    assert list(evaluation["k"]) == [2, 3, 4]
    assert (evaluation["inertia"] > 0).all()
    assert evaluation["silhouette_score"].between(-1, 1).all()


def test_determine_optimal_clusters_prefers_k2_for_two_separated_groups():
    selected = cl.select_clustering_features(_two_cluster_population(n_per_group=20))
    _, X = cl.preprocess_features(selected)

    evaluation = cl.determine_optimal_clusters(X, k_range=range(2, 6), random_state=0)

    best_k = evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"]
    assert best_k == 2


# --- run_clustering --------------------------------------------------------------


def test_run_clustering_produces_two_clusters_for_separated_groups():
    selected = cl.select_clustering_features(_two_cluster_population(n_per_group=15))
    _, X = cl.preprocess_features(selected)

    labels, model = cl.run_clustering(X, k=2, random_state=0)

    assert len(labels) == len(X)
    assert set(np.unique(labels)) == {0, 1}


def test_run_clustering_is_reproducible_with_same_seed():
    selected = cl.select_clustering_features(_two_cluster_population(n_per_group=15))
    _, X = cl.preprocess_features(selected)

    labels_a, _ = cl.run_clustering(X, k=2, random_state=7)
    labels_b, _ = cl.run_clustering(X, k=2, random_state=7)

    assert np.array_equal(labels_a, labels_b)


# --- save_clustered_profiles / save_cluster_evaluation --------------------------


def test_save_clustered_profiles_preserves_employee_count(tmp_path):
    df = _two_cluster_population(n_per_group=5)
    selected = cl.select_clustering_features(df)
    ids, X = cl.preprocess_features(selected)
    labels, _ = cl.run_clustering(X, k=2, random_state=0)

    path = cl.save_clustered_profiles(df, ids, labels, tmp_path)
    result = pd.read_parquet(path)

    assert len(result) == len(df)
    assert not result.duplicated(subset=cl.PERSON_KEY).any()


def test_save_clustered_profiles_ids_remain_strings(tmp_path):
    df = _two_cluster_population(n_per_group=3)
    selected = cl.select_clustering_features(df)
    ids, X = cl.preprocess_features(selected)
    labels, _ = cl.run_clustering(X, k=2, random_state=0)

    path = cl.save_clustered_profiles(df, ids, labels, tmp_path)
    result = pd.read_parquet(path)

    assert pd.api.types.is_string_dtype(result["HOUSEID"]) or result["HOUSEID"].dtype == object
    assert pd.api.types.is_string_dtype(result["PERSONID"]) or result["PERSONID"].dtype == object


def test_save_clustered_profiles_creates_cluster_id_and_nulls_excluded_rows(tmp_path):
    df = _two_cluster_population(n_per_group=5)
    selected = cl.select_clustering_features(df)
    ids, X = cl.preprocess_features(selected)
    labels, _ = cl.run_clustering(X, k=2, random_state=0)

    path = cl.save_clustered_profiles(df, ids, labels, tmp_path)
    result = pd.read_parquet(path)

    clustered = result.loc[result["HOUSEID"].isin(ids["HOUSEID"])]
    excluded = result.loc[result["HOUSEID"].isin(["NW1", "NC1"])]
    assert clustered["cluster_id"].notna().all()
    assert clustered["cluster_id"].isin([0, 1]).all()
    assert excluded["cluster_id"].isna().all()


def test_save_clustered_profiles_retains_all_original_feature_columns(tmp_path):
    df = _two_cluster_population(n_per_group=3)
    selected = cl.select_clustering_features(df)
    ids, X = cl.preprocess_features(selected)
    labels, _ = cl.run_clustering(X, k=2, random_state=0)

    path = cl.save_clustered_profiles(df, ids, labels, tmp_path)
    result = pd.read_parquet(path)

    assert set(df.columns).issubset(result.columns)
    assert "cluster_id" in result.columns


def test_save_cluster_evaluation_writes_csv(tmp_path):
    evaluation = pd.DataFrame({"k": [2, 3], "inertia": [10.0, 8.0], "silhouette_score": [0.5, 0.4]})

    path = cl.save_cluster_evaluation(evaluation, tmp_path)

    assert path.exists()
    roundtrip = pd.read_csv(path)
    assert roundtrip["k"].tolist() == [2, 3]


# --- load_employee_features ------------------------------------------------------


def test_load_employee_features_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        cl.load_employee_features(tmp_path)


def test_load_employee_features_reads_build_features_output(tmp_path):
    df = _employee_features_df(_employee_row("H1"))
    df.to_parquet(tmp_path / cl.FEATURE_TABLE_FILENAME, index=False)

    result = cl.load_employee_features(tmp_path)

    assert result["HOUSEID"].tolist() == ["H1"]
