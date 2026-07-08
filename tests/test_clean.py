"""Tests for driving_profiles.data.clean."""

from pathlib import Path

import pandas as pd
import pytest

from driving_profiles.data import clean, ingest


def _make_table(required_columns: tuple[str, ...], n: int, **overrides) -> pd.DataFrame:
    """Build a minimal DataFrame with every required column present.

    Every column defaults to "1" (a valid, non-sentinel placeholder value)
    for each of `n` rows; pass `column=[...]` overrides to control specific
    columns (e.g. ID columns for join tests, or a column under test).
    """
    df = pd.DataFrame({column: ["1"] * n for column in required_columns})
    for column, values in overrides.items():
        df[column] = values
    return df


def _household_df(n: int = 1, **overrides) -> pd.DataFrame:
    return _make_table(ingest.HOUSEHOLD_REQUIRED_COLUMNS, n, **overrides)


def _person_df(n: int = 1, **overrides) -> pd.DataFrame:
    return _make_table(ingest.PERSON_REQUIRED_COLUMNS, n, **overrides)


def _vehicle_df(n: int = 1, **overrides) -> pd.DataFrame:
    return _make_table(ingest.VEHICLE_REQUIRED_COLUMNS, n, **overrides)


def _trip_df(n: int = 1, **overrides) -> pd.DataFrame:
    return _make_table(ingest.TRIP_REQUIRED_COLUMNS, n, **overrides)


# --- replace_missing_sentinels -----------------------------------------


def test_replace_missing_sentinels_converts_standard_codes_to_nan():
    df = pd.DataFrame({"TRPMILES": ["5.2", "-1", "-7", "-8", "-9"]})

    result = clean.replace_missing_sentinels(df, ("TRPMILES",))

    assert result["TRPMILES"].iloc[0] == 5.2
    assert result["TRPMILES"].isna().tolist() == [False, True, True, True, True]


def test_replace_missing_sentinels_applies_extra_codes_only_to_named_column():
    df = pd.DataFrame({"ANNMILES": ["1000", "-77", "-88"], "OTHER": ["-77", "-77", "-77"]})

    result = clean.replace_missing_sentinels(
        df, ("ANNMILES", "OTHER"), clean.EXTRA_MISSING_CODES
    )

    assert result["ANNMILES"].isna().tolist() == [False, True, True]
    # -77 is not a standard code and OTHER has no extra-code entry, so it
    # should be left as a normal (non-sentinel) numeric value.
    assert result["OTHER"].isna().sum() == 0


def test_replace_missing_sentinels_leaves_columns_not_listed_untouched():
    df = pd.DataFrame({"TRPMILES": ["-1"], "UNLISTED": ["-1"]})

    result = clean.replace_missing_sentinels(df, ("TRPMILES",))

    assert result["UNLISTED"].tolist() == ["-1"]


# --- filter_valid_records -----------------------------------------------


def test_filter_valid_records_drops_rows_missing_or_blank_ids():
    df = pd.DataFrame(
        {
            "HOUSEID": ["1", "", None, "2", "3"],
            "PERSONID": ["01", "01", "01", "01", "  "],
        }
    )

    result = clean.filter_valid_records(df, ("HOUSEID", "PERSONID"))

    assert result["HOUSEID"].tolist() == ["1", "2"]


def test_filter_valid_records_ignores_columns_not_present():
    df = pd.DataFrame({"HOUSEID": ["1", "2"]})

    result = clean.filter_valid_records(df, ("HOUSEID", "NOT_A_COLUMN"))

    assert len(result) == 2


# --- join_nhts_tables -----------------------------------------------------


def test_join_nhts_tables_keeps_one_row_per_trip_and_merges_attributes():
    household = _household_df(HOUSEID=["H1"])
    person = _person_df(HOUSEID=["H1"], PERSONID=["01"], R_AGE=["40"])
    vehicle = _vehicle_df(HOUSEID=["H1"], VEHID=["01"], VEHTYPE=["03"])
    trip = _trip_df(
        2,
        HOUSEID=["H1", "H1"],
        PERSONID=["01", "01"],
        TRIPID=["01", "02"],
        VEHID=["01", "-1"],
        VEHCASEID=["H1_01", "-1"],
    )

    joined = clean.join_nhts_tables(
        {"household": household, "person": person, "vehicle": vehicle, "trip": trip}
    )

    assert len(joined) == 2
    assert not any(column.endswith(("_x", "_y")) for column in joined.columns)
    assert joined["R_AGE"].tolist() == ["40", "40"]


def test_join_nhts_tables_left_joins_vehicle_so_non_vehicle_trips_survive():
    household = _household_df(HOUSEID=["H1"])
    person = _person_df(HOUSEID=["H1"], PERSONID=["01"])
    vehicle = _vehicle_df(HOUSEID=["H1"], VEHID=["01"], VEHTYPE=["03"])
    trip = _trip_df(
        2,
        HOUSEID=["H1", "H1"],
        PERSONID=["01", "01"],
        TRIPID=["01", "02"],
        VEHID=["01", "-1"],
        VEHCASEID=["H1_01", "-1"],
    )

    joined = clean.join_nhts_tables(
        {"household": household, "person": person, "vehicle": vehicle, "trip": trip}
    )

    with_vehicle = joined.loc[joined["TRIPID"] == "01", "VEHTYPE"].iloc[0]
    without_vehicle = joined.loc[joined["TRIPID"] == "02", "VEHTYPE"].iloc[0]
    assert with_vehicle == "03"
    assert pd.isna(without_vehicle)


def test_join_nhts_tables_preserves_id_columns_as_strings():
    household = _household_df(HOUSEID=["9000013002"])
    person = _person_df(HOUSEID=["9000013002"], PERSONID=["01"])
    vehicle = _vehicle_df(HOUSEID=["9000013002"], VEHID=["01"])
    trip = _trip_df(
        1,
        HOUSEID=["9000013002"],
        PERSONID=["01"],
        TRIPID=["01"],
        VEHID=["01"],
        VEHCASEID=["9000013002_01"],
    )

    joined = clean.join_nhts_tables(
        {"household": household, "person": person, "vehicle": vehicle, "trip": trip}
    )

    for column in ("HOUSEID", "PERSONID", "TRIPID", "VEHID"):
        assert pd.api.types.is_string_dtype(joined[column]) or joined[column].dtype == object
        assert joined.loc[0, column] in {"9000013002", "01"}


def test_join_nhts_tables_drops_duplicate_travday_tdaydate_from_household():
    household = _household_df(HOUSEID=["H1"], TRAVDAY=["9"], TDAYDATE=["999999"])
    person = _person_df(HOUSEID=["H1"], PERSONID=["01"])
    vehicle = _vehicle_df(HOUSEID=["H1"], VEHID=["01"])
    trip = _trip_df(
        1,
        HOUSEID=["H1"],
        PERSONID=["01"],
        TRIPID=["01"],
        VEHID=["01"],
        VEHCASEID=["H1_01"],
        TRAVDAY=["2"],
        TDAYDATE=["202201"],
    )

    joined = clean.join_nhts_tables(
        {"household": household, "person": person, "vehicle": vehicle, "trip": trip}
    )

    assert joined.loc[0, "TRAVDAY"] == "2"
    assert joined.loc[0, "TDAYDATE"] == "202201"


# --- clean_trips -----------------------------------------------------------


def _minimal_trip_frame(**overrides) -> pd.DataFrame:
    base = {
        "HOUSEID": ["H1"],
        "PERSONID": ["01"],
        "TRIPID": ["01"],
        "STRTTIME": ["0800"],
        "ENDTIME": ["0830"],
        "TRVLCMIN": ["30"],
        "TRPMILES": ["5.0"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_trips_drops_duplicate_trip_records():
    df = pd.concat([_minimal_trip_frame(), _minimal_trip_frame()], ignore_index=True)

    result = clean.clean_trips(df)

    assert len(result) == 1


def test_clean_trips_drops_trips_missing_core_travel_values():
    valid = _minimal_trip_frame()
    missing_distance = _minimal_trip_frame(TRIPID=["02"], TRPMILES=["-9"])
    missing_start_time = _minimal_trip_frame(TRIPID=["03"], STRTTIME=["-9"])
    df = pd.concat([valid, missing_distance, missing_start_time], ignore_index=True)

    result = clean.clean_trips(df)

    assert result["TRIPID"].tolist() == ["01"]


def test_clean_trips_converts_sentinel_codes_across_joined_columns():
    df = _minimal_trip_frame(GCDWORK=["-1"], HHFAMINC=["-7"], ANNMILES=["-88"])

    result = clean.clean_trips(df)

    assert pd.isna(result.loc[0, "GCDWORK"])
    assert pd.isna(result.loc[0, "HHFAMINC"])
    assert pd.isna(result.loc[0, "ANNMILES"])


def test_clean_trips_resets_index():
    df = pd.concat(
        [_minimal_trip_frame(TRIPID=["01"]), _minimal_trip_frame(TRIPID=["02"])],
        ignore_index=True,
    )

    result = clean.clean_trips(df)

    assert result.index.tolist() == list(range(len(result)))


# --- create_analysis_dataset (end to end) -----------------------------------


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(columns)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def test_create_analysis_dataset_end_to_end(tmp_path):
    household_columns = list(ingest.HOUSEHOLD_REQUIRED_COLUMNS)
    person_columns = list(ingest.PERSON_REQUIRED_COLUMNS)
    vehicle_columns = list(ingest.VEHICLE_REQUIRED_COLUMNS)
    trip_columns = list(ingest.TRIP_REQUIRED_COLUMNS)

    def row(columns, overrides):
        return [overrides.get(column, "1") for column in columns]

    _write_csv(
        tmp_path / "hhv2pub.csv",
        household_columns,
        [row(household_columns, {"HOUSEID": "H1"})],
    )
    _write_csv(
        tmp_path / "perv2pub.csv",
        person_columns,
        [row(person_columns, {"HOUSEID": "H1", "PERSONID": "01"})],
    )
    _write_csv(
        tmp_path / "vehv2pub.csv",
        vehicle_columns,
        [row(vehicle_columns, {"HOUSEID": "H1", "VEHID": "01"})],
    )
    _write_csv(
        tmp_path / "tripv2pub.csv",
        trip_columns,
        [
            row(
                trip_columns,
                {
                    "HOUSEID": "H1",
                    "PERSONID": "01",
                    "TRIPID": "01",
                    "VEHID": "01",
                    "VEHCASEID": "H1_01",
                    "STRTTIME": "0800",
                    "ENDTIME": "0830",
                    "TRVLCMIN": "30",
                    "TRPMILES": "5.0",
                },
            ),
            # This trip is missing a core travel value and should be dropped.
            row(
                trip_columns,
                {
                    "HOUSEID": "H1",
                    "PERSONID": "01",
                    "TRIPID": "02",
                    "VEHID": "-1",
                    "VEHCASEID": "-1",
                    "TRPMILES": "-9",
                },
            ),
        ],
    )

    result = clean.create_analysis_dataset(tmp_path)

    assert len(result) == 1
    assert result.loc[0, "TRIPID"] == "01"
    assert pd.api.types.is_string_dtype(result["HOUSEID"]) or result["HOUSEID"].dtype == object


def test_create_analysis_dataset_raises_for_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        clean.create_analysis_dataset(tmp_path)


# --- save_analysis_dataset ---------------------------------------------


def test_save_analysis_dataset_writes_parquet(tmp_path):
    df = pd.DataFrame({"HOUSEID": ["H1"], "TRPMILES": [5.0]})

    path = clean.save_analysis_dataset(df, tmp_path / "interim")

    assert path.exists()
    assert path.name == clean.ANALYSIS_DATASET_FILENAME
    roundtrip = pd.read_parquet(path)
    assert roundtrip["HOUSEID"].tolist() == ["H1"]
