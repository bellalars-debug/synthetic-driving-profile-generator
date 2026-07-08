"""Tests for driving_profiles.data.ingest."""

from pathlib import Path

import pandas as pd
import pytest

from driving_profiles.data import ingest


def _fake_row(columns: list[str], id_columns: tuple[str, ...]) -> list[str]:
    """Build a single dummy data row: zero-padded strings for ID columns."""
    return ["01" if column in id_columns else "1" for column in columns]


def _write_csv(path: Path, columns: list[str], row: list[str]) -> None:
    path.write_text(",".join(columns) + "\n" + ",".join(row) + "\n")


FILES = [
    (
        "hhv2pub.csv",
        "load_household",
        ingest.HOUSEHOLD_ID_COLUMNS,
        ingest.HOUSEHOLD_REQUIRED_COLUMNS,
    ),
    ("perv2pub.csv", "load_person", ingest.PERSON_ID_COLUMNS, ingest.PERSON_REQUIRED_COLUMNS),
    ("vehv2pub.csv", "load_vehicle", ingest.VEHICLE_ID_COLUMNS, ingest.VEHICLE_REQUIRED_COLUMNS),
    ("tripv2pub.csv", "load_trip", ingest.TRIP_ID_COLUMNS, ingest.TRIP_REQUIRED_COLUMNS),
]

PARAMS = "filename,loader_name,id_columns,required_columns"


@pytest.mark.parametrize(PARAMS, FILES)
def test_load_preserves_id_columns_as_strings(
    tmp_path, filename, loader_name, id_columns, required_columns
):
    columns = list(required_columns)
    _write_csv(tmp_path / filename, columns, _fake_row(columns, id_columns))
    loader = getattr(ingest, loader_name)

    df = loader(tmp_path)

    assert list(df.columns) == columns
    for id_column in id_columns:
        assert df.loc[0, id_column] == "01"
        assert pd.api.types.is_string_dtype(df[id_column])


@pytest.mark.parametrize(PARAMS, FILES)
def test_load_missing_file_raises_with_helpful_message(
    tmp_path, filename, loader_name, id_columns, required_columns
):
    loader = getattr(ingest, loader_name)

    with pytest.raises(FileNotFoundError, match="driving_profiles.data.download"):
        loader(tmp_path)


@pytest.mark.parametrize(PARAMS, FILES)
def test_load_missing_required_column_raises_value_error(
    tmp_path, filename, loader_name, id_columns, required_columns
):
    dropped_column = required_columns[-1]
    columns = [column for column in required_columns if column != dropped_column]
    _write_csv(tmp_path / filename, columns, _fake_row(columns, id_columns))
    loader = getattr(ingest, loader_name)

    with pytest.raises(ValueError, match=dropped_column):
        loader(tmp_path)


def test_load_all_returns_all_four_tables_and_ignores_long_distance_file(tmp_path):
    for filename, _loader_name, id_columns, required_columns in FILES:
        columns = list(required_columns)
        _write_csv(tmp_path / filename, columns, _fake_row(columns, id_columns))

    tables = ingest.load_all(tmp_path)

    assert set(tables) == {"household", "person", "vehicle", "trip"}
    for df in tables.values():
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
    assert not (tmp_path / "ldtv2pub.csv").exists()
