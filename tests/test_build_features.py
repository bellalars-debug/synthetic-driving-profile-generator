"""Tests for driving_profiles.features.build_features."""

import pandas as pd
import pytest

from driving_profiles.features import build_features as bf

FEATURE_COLUMNS = {
    "HOUSEID",
    "PERSONID",
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
}


def _trip(**overrides) -> dict:
    """One trip row with sensible non-sentinel defaults for every column
    build_features.py reads; override individual fields per test."""
    row = {
        "HOUSEID": "H1",
        "PERSONID": "01",
        "TRIPID": "01",
        "TRAVDAY": 2,  # Monday
        "WHYTRP1S": 1,  # home
        "LOOP_TRIP": 2,  # not a loop trip
        "TRPTRANS": 3,  # car
        "TRPHHVEH": 1,  # household vehicle
        "STRTTIME": 800.0,
        "ENDTIME": 830.0,
        "TRVLCMIN": 30.0,
        "TRPMILES": 10.0,
        "GCDWORK": 9.5,
        "R_AGE": 40,
        "WORKER": 1.0,
        "HHFAMINC": 7.0,
        "HHSIZE": 3,
        "HHVEHCNT": 2,
        "DRVRCNT": 2,
    }
    row.update(overrides)
    return row


def _trips_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --- build_demographic_features ---------------------------------------------


def test_build_demographic_features_computes_age_band_and_worker_status():
    trips = _trips_df(_trip(R_AGE=45, WORKER=1.0), _trip(HOUSEID="H2", R_AGE=70, WORKER=2.0))

    result = bf.build_demographic_features(trips)

    row1 = result.loc[result["HOUSEID"] == "H1"].iloc[0]
    row2 = result.loc[result["HOUSEID"] == "H2"].iloc[0]
    assert row1["age"] == 45
    assert row1["age_band"] == "45-54"
    assert row1["worker_status"] == "worker"
    assert bool(row1["is_worker"]) is True
    assert row2["age_band"] == "65+"
    assert row2["worker_status"] == "non_worker"
    assert bool(row2["is_worker"]) is False


def test_build_demographic_features_treats_negative_age_as_missing():
    trips = _trips_df(_trip(R_AGE=-1))

    result = bf.build_demographic_features(trips)

    assert pd.isna(result.loc[0, "age"])
    assert pd.isna(result.loc[0, "age_band"])


def test_build_demographic_features_worker_status_missing_when_worker_unascertained():
    trips = _trips_df(_trip(WORKER=float("nan")))

    result = bf.build_demographic_features(trips)

    assert pd.isna(result.loc[0, "worker_status"])
    assert pd.isna(result.loc[0, "is_worker"])


def test_build_demographic_features_one_row_per_person_despite_multiple_trips():
    trips = _trips_df(_trip(TRIPID="01"), _trip(TRIPID="02"))

    result = bf.build_demographic_features(trips)

    assert len(result) == 1


# --- build_household_features -----------------------------------------------


def test_build_household_features_passes_through_and_renames():
    trips = _trips_df(_trip(HHFAMINC=8.0, HHSIZE=4, HHVEHCNT=3))

    result = bf.build_household_features(trips)

    assert result.loc[0, "household_income_bracket"] == 8.0
    assert result.loc[0, "household_size"] == 4
    assert result.loc[0, "household_vehicle_count"] == 3


# --- build_commute_features ---------------------------------------------------


def test_build_commute_features_computes_arrival_and_departure_around_work_leg():
    trips = _trips_df(
        _trip(TRIPID="01", WHYTRP1S=1, STRTTIME=700.0, ENDTIME=730.0, TRPMILES=1.0, TRVLCMIN=30.0),
        _trip(TRIPID="02", WHYTRP1S=10, STRTTIME=745.0, ENDTIME=815.0, TRPMILES=9.0, TRVLCMIN=30.0),
        _trip(
            TRIPID="03", WHYTRP1S=1, STRTTIME=1700.0, ENDTIME=1730.0, TRPMILES=9.0, TRVLCMIN=30.0
        ),
    )

    result = bf.build_commute_features(trips)
    row = result.iloc[0]

    assert row["work_trip_count"] == 1
    assert row["commute_distance_trip_miles"] == 9.0
    assert row["commute_duration_minutes"] == 30.0
    assert row["work_arrival_time"] == 815.0
    assert row["work_departure_time"] == 1700.0
    assert row["commute_distance_survey_miles"] == 9.5  # GCDWORK, from _trip() default


def test_build_commute_features_no_departure_when_no_trailing_trip():
    trips = _trips_df(
        _trip(TRIPID="01", WHYTRP1S=10, STRTTIME=800.0, ENDTIME=830.0),
    )

    result = bf.build_commute_features(trips)

    assert pd.isna(result.loc[0, "work_departure_time"])


def test_build_commute_features_zero_work_trips_is_a_real_value_not_nan():
    trips = _trips_df(_trip(WHYTRP1S=1))

    result = bf.build_commute_features(trips)

    assert result.loc[0, "work_trip_count"] == 0
    assert pd.isna(result.loc[0, "commute_distance_trip_miles"])
    assert pd.isna(result.loc[0, "work_arrival_time"])


# --- build_daily_mobility_features -------------------------------------------


def test_build_daily_mobility_features_sums_driving_trips_and_counts_stops():
    trips = _trips_df(
        _trip(TRIPID="01", WHYTRP1S=10, TRPTRANS=3, TRPMILES=10.0, TRVLCMIN=20.0),
        _trip(TRIPID="02", WHYTRP1S=30, TRPTRANS=3, TRPMILES=5.0, TRVLCMIN=10.0),
        _trip(TRIPID="03", WHYTRP1S=1, TRPTRANS=1, TRPMILES=0.2, TRVLCMIN=5.0),  # walk home
    )

    result = bf.build_daily_mobility_features(trips)
    row = result.iloc[0]

    assert row["trips_per_day"] == 3
    assert row["total_daily_miles"] == 15.0
    assert row["total_driving_minutes"] == 30.0
    assert row["average_trip_distance_miles"] == 7.5  # 15 miles / 2 driving trips
    assert row["number_of_stops"] == 2  # excludes the final home-purpose leg


def test_build_daily_mobility_features_excludes_loop_trips():
    trips = _trips_df(
        _trip(TRIPID="01", LOOP_TRIP=1, WHYTRP1S=30),
        _trip(TRIPID="02", LOOP_TRIP=2, WHYTRP1S=30),
    )

    result = bf.build_daily_mobility_features(trips)

    assert result.loc[0, "trips_per_day"] == 1
    assert result.loc[0, "number_of_stops"] == 1


def test_build_daily_mobility_features_nan_average_when_no_driving_trips():
    trips = _trips_df(_trip(TRPTRANS=1))  # walk only

    result = bf.build_daily_mobility_features(trips)

    assert pd.isna(result.loc[0, "total_daily_miles"])
    assert pd.isna(result.loc[0, "average_trip_distance_miles"])


# --- build_vehicle_availability_features -------------------------------------


def test_build_vehicle_availability_features_computes_ratio_and_usage():
    trips = _trips_df(
        _trip(TRIPID="01", HHVEHCNT=4, DRVRCNT=2, TRPHHVEH=1),
        _trip(TRIPID="02", HHVEHCNT=4, DRVRCNT=2, TRPHHVEH=2),
    )

    result = bf.build_vehicle_availability_features(trips)
    row = result.iloc[0]

    assert row["vehicles_per_driver"] == 2.0
    assert bool(row["vehicle_per_driver_adequate"]) is True
    assert row["household_vehicle_trip_count"] == 1
    assert bool(row["used_household_vehicle"]) is True


def test_build_vehicle_availability_features_handles_zero_drivers():
    trips = _trips_df(_trip(HHVEHCNT=1, DRVRCNT=0))

    result = bf.build_vehicle_availability_features(trips)

    assert pd.isna(result.loc[0, "vehicles_per_driver"])
    assert bool(result.loc[0, "vehicle_per_driver_adequate"]) is False


# --- create_employee_feature_table (end to end) ------------------------------


def test_create_employee_feature_table_one_row_per_person():
    trips = _trips_df(
        _trip(HOUSEID="H1", PERSONID="01", TRIPID="01"),
        _trip(HOUSEID="H1", PERSONID="01", TRIPID="02"),
        _trip(HOUSEID="H1", PERSONID="02", TRIPID="01"),
        _trip(HOUSEID="H2", PERSONID="01", TRIPID="01"),
    )

    result = bf.create_employee_feature_table(trips)

    assert len(result) == 3
    assert not result.duplicated(subset=["HOUSEID", "PERSONID"]).any()


def test_create_employee_feature_table_has_expected_columns():
    trips = _trips_df(_trip())

    result = bf.create_employee_feature_table(trips)

    assert FEATURE_COLUMNS.issubset(result.columns)


def test_create_employee_feature_table_ids_remain_strings():
    trips = _trips_df(_trip(HOUSEID="9000013002", PERSONID="01"))

    result = bf.create_employee_feature_table(trips)

    assert pd.api.types.is_string_dtype(result["HOUSEID"]) or result["HOUSEID"].dtype == object
    assert pd.api.types.is_string_dtype(result["PERSONID"]) or result["PERSONID"].dtype == object
    assert result.loc[0, "HOUSEID"] == "9000013002"
    assert result.loc[0, "PERSONID"] == "01"


def test_create_employee_feature_table_filters_to_weekday_only():
    trips = _trips_df(
        _trip(HOUSEID="H1", TRAVDAY=1),  # Sunday - dropped
        _trip(HOUSEID="H2", TRAVDAY=2),  # Monday - kept
    )

    result = bf.create_employee_feature_table(trips)

    assert result["HOUSEID"].tolist() == ["H2"]


def test_create_employee_feature_table_sorted_by_person_key():
    trips = _trips_df(
        _trip(HOUSEID="H2", PERSONID="01"),
        _trip(HOUSEID="H1", PERSONID="02"),
        _trip(HOUSEID="H1", PERSONID="01"),
    )

    result = bf.create_employee_feature_table(trips)

    assert result[["HOUSEID", "PERSONID"]].values.tolist() == [
        ["H1", "01"],
        ["H1", "02"],
        ["H2", "01"],
    ]


# --- load_cleaned_trips / save_feature_table ---------------------------------


def test_load_cleaned_trips_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        bf.load_cleaned_trips(tmp_path)


def test_load_cleaned_trips_reads_clean_py_output(tmp_path):
    df = pd.DataFrame({"HOUSEID": ["H1"], "PERSONID": ["01"]})
    df.to_parquet(tmp_path / "trips_clean.parquet", index=False)

    result = bf.load_cleaned_trips(tmp_path)

    assert result["HOUSEID"].tolist() == ["H1"]


def test_save_feature_table_writes_parquet(tmp_path):
    df = pd.DataFrame({"HOUSEID": ["H1"], "PERSONID": ["01"]})

    path = bf.save_feature_table(df, tmp_path / "processed")

    assert path.exists()
    assert path.name == bf.FEATURE_TABLE_FILENAME
    roundtrip = pd.read_parquet(path)
    assert roundtrip["PERSONID"].tolist() == ["01"]
