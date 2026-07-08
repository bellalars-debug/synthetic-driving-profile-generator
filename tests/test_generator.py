"""Tests for driving_profiles.generator.sample."""

import pandas as pd
import pytest

from driving_profiles.generator import sample as sm

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


def _employee_row(house_id: str, person_id: str = "01", cluster_id=None, **overrides) -> dict:
    """One employee_clusters.parquet row with sensible defaults; override
    individual fields per test."""
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
        "cluster_id": cluster_id,
    }
    row.update(overrides)
    return row


def _employee_clusters_df(*rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    df["HOUSEID"] = df["HOUSEID"].astype(str)
    df["PERSONID"] = df["PERSONID"].astype(str)
    df["is_worker"] = df["is_worker"].astype("boolean")
    df["cluster_id"] = pd.array(df["cluster_id"], dtype="Int64")
    return df


def _two_cluster_population(n_per_group: int = 15) -> pd.DataFrame:
    """A clustered population with two archetypes (short vs. long commute),
    plus unclustered non-worker/no-commute rows that generation must ignore."""
    rows = []
    for i in range(n_per_group):
        rows.append(
            _employee_row(
                f"S{i}",
                cluster_id=0,
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
                cluster_id=1,
                commute_distance_survey_miles=40.0 + 0.1 * i,
                commute_duration_minutes=55.0 + 0.1 * i,
                work_arrival_time=900.0,
                work_departure_time=1800.0,
                total_daily_miles=85.0 + 0.1 * i,
                total_driving_minutes=100.0 + 0.1 * i,
            )
        )
    # Unclustered rows: generation must never draw from these.
    rows.append(_employee_row("NW1", cluster_id=None, is_worker=False, work_trip_count=0))
    rows.append(_employee_row("NC1", cluster_id=None, work_trip_count=0))
    return _employee_clusters_df(*rows)


# --- load_clustered_employees ------------------------------------------------


def test_load_clustered_employees_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        sm.load_clustered_employees(tmp_path)


def test_load_clustered_employees_filters_to_non_null_cluster_id(tmp_path):
    df = _two_cluster_population(n_per_group=3)
    df.to_parquet(tmp_path / sm.CLUSTER_TABLE_FILENAME, index=False)

    result = sm.load_clustered_employees(tmp_path)

    assert result["cluster_id"].notna().all()
    assert "NW1" not in result["HOUSEID"].tolist()
    assert "NC1" not in result["HOUSEID"].tolist()
    assert len(result) == len(df) - 2


# --- determine_cluster_sampling ----------------------------------------------


def test_determine_cluster_sampling_preserves_observed_proportions():
    clustered = _two_cluster_population(n_per_group=25)  # cluster 0: 25, cluster 1: 25

    counts = sm.determine_cluster_sampling(clustered, n=1000, seed=0)

    assert counts.sum() == 1000
    assert counts.loc[0] == pytest.approx(500, abs=5)
    assert counts.loc[1] == pytest.approx(500, abs=5)


def test_determine_cluster_sampling_matches_skewed_proportions():
    rows = [_employee_row(f"S{i}", cluster_id=0) for i in range(80)]
    rows += [_employee_row(f"L{i}", cluster_id=1) for i in range(20)]
    clustered = _employee_clusters_df(*rows)

    counts = sm.determine_cluster_sampling(clustered, n=1000, seed=0)

    assert counts.sum() == 1000
    assert counts.loc[0] == pytest.approx(800, abs=5)
    assert counts.loc[1] == pytest.approx(200, abs=5)


def test_determine_cluster_sampling_sums_exactly_to_n_for_any_n():
    clustered = _two_cluster_population(n_per_group=7)  # uneven proportions (15 vs 15 minus edge)

    for n in (1, 3, 7, 100, 1001):
        counts = sm.determine_cluster_sampling(clustered, n=n, seed=1)
        assert counts.sum() == n


def test_determine_cluster_sampling_accepts_explicit_weight_override():
    clustered = _two_cluster_population(n_per_group=10)  # observed ~50/50

    counts = sm.determine_cluster_sampling(
        clustered, n=1000, cluster_weights={0: 0.9, 1: 0.1}, seed=0
    )

    assert counts.sum() == 1000
    assert counts.loc[0] == pytest.approx(900, abs=5)
    assert counts.loc[1] == pytest.approx(100, abs=5)


def test_determine_cluster_sampling_is_reproducible_with_same_seed():
    clustered = _two_cluster_population(n_per_group=11)

    counts_a = sm.determine_cluster_sampling(clustered, n=101, seed=5)
    counts_b = sm.determine_cluster_sampling(clustered, n=101, seed=5)

    pd.testing.assert_series_equal(counts_a, counts_b)


# --- sample_employees ---------------------------------------------------------


def test_sample_employees_produces_requested_count_per_cluster():
    clustered = _two_cluster_population(n_per_group=15)
    cluster_sampling = pd.Series({0: 30, 1: 10}, name="n_synthetic")
    cluster_sampling.index.name = "cluster_id"

    sampled = sm.sample_employees(clustered, cluster_sampling, seed=0)

    assert len(sampled) == 40
    assert (sampled["cluster_id"] == 0).sum() == 30
    assert (sampled["cluster_id"] == 1).sum() == 10


def test_sample_employees_preserves_joint_relationships_via_whole_row_draw():
    # Cluster 0 rows always pair a specific age with a specific household size,
    # a relationship independent per-column sampling would be free to break.
    rows = [
        _employee_row(f"H{i}", cluster_id=0, age=25, household_size=1)
        for i in range(10)
    ]
    rows += [
        _employee_row(f"H{i + 10}", cluster_id=0, age=55, household_size=4)
        for i in range(10)
    ]
    clustered = _employee_clusters_df(*rows)
    cluster_sampling = pd.Series({0: 200}, name="n_synthetic")
    cluster_sampling.index.name = "cluster_id"

    sampled = sm.sample_employees(clustered, cluster_sampling, seed=0)

    paired = set(zip(sampled["age"], sampled["household_size"]))
    assert paired.issubset({(25, 1), (55, 4)})


def test_sample_employees_jitters_continuous_fields():
    clustered = _two_cluster_population(n_per_group=20)
    cluster_sampling = pd.Series({0: 100, 1: 0}, name="n_synthetic")
    cluster_sampling.index.name = "cluster_id"

    sampled = sm.sample_employees(clustered, cluster_sampling, seed=0)

    # Jittered values should not all be exact copies of the small set of
    # real source values.
    real_values = set(clustered.loc[clustered["cluster_id"] == 0, "commute_distance_survey_miles"])
    synthetic_values = set(sampled["commute_distance_survey_miles"])
    assert not synthetic_values.issubset(real_values)


def test_sample_employees_leaves_nan_source_values_as_nan():
    rows = [_employee_row(f"H{i}", cluster_id=0, total_daily_miles=float("nan")) for i in range(10)]
    clustered = _employee_clusters_df(*rows)
    cluster_sampling = pd.Series({0: 50}, name="n_synthetic")
    cluster_sampling.index.name = "cluster_id"

    sampled = sm.sample_employees(clustered, cluster_sampling, seed=0)

    assert sampled["total_daily_miles"].isna().all()


def test_sample_employees_keeps_counts_at_least_one_after_jitter():
    clustered = _two_cluster_population(n_per_group=20)
    cluster_sampling = pd.Series({0: 200, 1: 200}, name="n_synthetic")
    cluster_sampling.index.name = "cluster_id"

    sampled = sm.sample_employees(clustered, cluster_sampling, jitter_scale=5.0, seed=0)

    assert (sampled["trips_per_day"] >= 1).all()
    assert (sampled["number_of_stops"] >= 1).all()


def test_sample_employees_clamps_time_of_day_to_valid_range():
    clustered = _two_cluster_population(n_per_group=20)
    cluster_sampling = pd.Series({0: 200, 1: 200}, name="n_synthetic")
    cluster_sampling.index.name = "cluster_id"

    sampled = sm.sample_employees(clustered, cluster_sampling, jitter_scale=10.0, seed=0)

    assert (sampled["work_arrival_time"] >= 0).all()
    assert (sampled["work_arrival_time"] <= 2359).all()
    assert (sampled["work_departure_time"] >= 0).all()
    assert (sampled["work_departure_time"] <= 2359).all()


def test_sample_employees_is_reproducible_with_same_seed():
    clustered = _two_cluster_population(n_per_group=15)
    cluster_sampling = pd.Series({0: 50, 1: 50}, name="n_synthetic")
    cluster_sampling.index.name = "cluster_id"

    sampled_a = sm.sample_employees(clustered, cluster_sampling, seed=3)
    sampled_b = sm.sample_employees(clustered, cluster_sampling, seed=3)

    pd.testing.assert_frame_equal(sampled_a, sampled_b)


# --- assign_unique_employee_ids -----------------------------------------------


def test_assign_unique_employee_ids_generates_unique_ids():
    clustered = _two_cluster_population(n_per_group=15)
    cluster_sampling = pd.Series({0: 50, 1: 50}, name="n_synthetic")
    cluster_sampling.index.name = "cluster_id"
    sampled = sm.sample_employees(clustered, cluster_sampling, seed=0)

    result = sm.assign_unique_employee_ids(sampled)

    assert result[sm.SYNTHETIC_ID_COLUMN].is_unique
    assert len(result) == 100


def test_assign_unique_employee_ids_retains_source_ids_and_drops_real_key():
    clustered = _two_cluster_population(n_per_group=5)
    cluster_sampling = pd.Series({0: 10, 1: 10}, name="n_synthetic")
    cluster_sampling.index.name = "cluster_id"
    sampled = sm.sample_employees(clustered, cluster_sampling, seed=0)

    result = sm.assign_unique_employee_ids(sampled)

    assert "HOUSEID" not in result.columns
    assert "PERSONID" not in result.columns
    assert sm.SOURCE_HOUSEID_COLUMN in result.columns
    assert sm.SOURCE_PERSONID_COLUMN in result.columns
    assert result[sm.SOURCE_HOUSEID_COLUMN].isin(clustered["HOUSEID"]).all()
    assert result[sm.SOURCE_PERSONID_COLUMN].isin(clustered["PERSONID"]).all()


# --- create_synthetic_employee_table / save_synthetic_employees --------------


@pytest.fixture
def clustered_parquet(tmp_path):
    df = _two_cluster_population(n_per_group=25)
    path = tmp_path / sm.CLUSTER_TABLE_FILENAME
    df.to_parquet(path, index=False)
    return tmp_path


@pytest.mark.parametrize("n", [100, 1000])
def test_create_synthetic_employee_table_produces_requested_population_size(clustered_parquet, n):
    result = sm.create_synthetic_employee_table(n=n, seed=0, processed_dir=clustered_parquet)

    assert len(result) == n


def test_create_synthetic_employee_table_preserves_cluster_proportions(clustered_parquet):
    clustered = sm.load_clustered_employees(clustered_parquet)
    observed = clustered["cluster_id"].value_counts(normalize=True).sort_index()

    result = sm.create_synthetic_employee_table(n=5000, seed=0, processed_dir=clustered_parquet)
    realized = result["cluster_id"].value_counts(normalize=True).sort_index()

    for cluster_id in observed.index:
        assert realized.loc[cluster_id] == pytest.approx(observed.loc[cluster_id], abs=0.02)


def test_create_synthetic_employee_table_has_unique_synthetic_ids(clustered_parquet):
    result = sm.create_synthetic_employee_table(n=500, seed=0, processed_dir=clustered_parquet)

    assert result[sm.SYNTHETIC_ID_COLUMN].is_unique
    assert len(result) == 500


def test_create_synthetic_employee_table_is_reproducible_with_fixed_seed(clustered_parquet):
    result_a = sm.create_synthetic_employee_table(n=300, seed=7, processed_dir=clustered_parquet)
    result_b = sm.create_synthetic_employee_table(n=300, seed=7, processed_dir=clustered_parquet)

    pd.testing.assert_frame_equal(result_a, result_b)


def test_create_synthetic_employee_table_varies_with_different_seed(clustered_parquet):
    result_a = sm.create_synthetic_employee_table(n=300, seed=1, processed_dir=clustered_parquet)
    result_b = sm.create_synthetic_employee_table(n=300, seed=2, processed_dir=clustered_parquet)

    assert not result_a["commute_distance_survey_miles"].equals(
        result_b["commute_distance_survey_miles"]
    )


def test_create_synthetic_employee_table_column_shape(clustered_parquet):
    result = sm.create_synthetic_employee_table(n=100, seed=0, processed_dir=clustered_parquet)

    assert sm.SYNTHETIC_ID_COLUMN in result.columns
    assert sm.SOURCE_HOUSEID_COLUMN in result.columns
    assert sm.SOURCE_PERSONID_COLUMN in result.columns
    assert "HOUSEID" not in result.columns
    assert "PERSONID" not in result.columns
    assert set(FEATURE_COLUMNS + ["cluster_id"]).issubset(result.columns)


def test_save_synthetic_employees_writes_parquet(tmp_path, clustered_parquet):
    result = sm.create_synthetic_employee_table(n=50, seed=0, processed_dir=clustered_parquet)

    path = sm.save_synthetic_employees(result, tmp_path)
    roundtrip = pd.read_parquet(path)

    assert path.name == sm.SYNTHETIC_EMPLOYEE_FILENAME
    assert len(roundtrip) == 50
    assert roundtrip[sm.SYNTHETIC_ID_COLUMN].is_unique
