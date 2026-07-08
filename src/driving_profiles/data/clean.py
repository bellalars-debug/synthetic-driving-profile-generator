"""Filter, join, and reconstruct trip chains from ingested NHTS 2022 data.

Produces the "cleaned traveler data" pipeline artifact: one row per trip,
enriched with the person/household/vehicle attributes that describe who
took it and what it was taken in. Reads no CSVs directly - all four tables
come from ingest.py. Feature engineering, clustering, synthetic employee
generation, and EV charging modeling are out of scope here; see
features/build_features.py and beyond for those stages.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from driving_profiles.data import ingest

logger = logging.getLogger(__name__)

DEFAULT_INTERIM_DIR = Path("data/interim")
ANALYSIS_DATASET_FILENAME = "trips_clean.parquet"

# Missing-value convention shared by every NHTS 2022 file (see
# docs/data_requirements.md section 1): -1 valid skip, -7 refused,
# -8 don't know, -9 not ascertained.
STANDARD_MISSING_CODES = (-1, -7, -8, -9)

# A few derived numeric variables use extra sentinel codes on top of the
# standard four (docs/data_requirements.md section 1).
EXTRA_MISSING_CODES: dict[str, tuple[int, ...]] = {
    "ANNMILES": (-77, -88),
}

# Columns (per table) that carry survey-response sentinel codes and should
# have STANDARD_MISSING_CODES / EXTRA_MISSING_CODES converted to NaN. ID
# columns are deliberately excluded here: e.g. "-1" in VEHID or TRPHHVEH
# means "trip did not use a household vehicle", which the join already
# handles correctly (see join_nhts_tables), and converting a merge key to
# NaN would break it.
HOUSEHOLD_MISSING_VALUE_COLUMNS = ("HHFAMINC",)
PERSON_MISSING_VALUE_COLUMNS = (
    "WORKER",
    "PAYPROF",
    "PRMACT",
    "EMPLOYMENT2",
    "WRKLOC",
    "WKFMHM22",
    "WRKTRANS",
    "EMPPASS",
    "GCDWORK",
)
VEHICLE_MISSING_VALUE_COLUMNS = ("VEHFUEL", "ANNMILES")
TRIP_MISSING_VALUE_COLUMNS = (
    "WHYFROM",
    "WHYTO",
    "WHYTRP1S",
    "WHYTRP90",
    "TRIPPURP",
    "TRPTRANS",
    "NUMONTRP",
    "TRPHHVEH",
    "STRTTIME",
    "ENDTIME",
    "TRVLCMIN",
    "DWELTIME",
    "TRPMILES",
    "VMT_MILE",
    "WHODROVE",
    "WHODROVE_IMP",
)

# The natural key of one trip record, used to de-duplicate exact repeats.
# (A subset of ingest.TRIP_ID_COLUMNS, which also includes the vehicle-join
# keys VEHID/VEHCASEID and the derived SEQ_TRIPID/TDCASEID.)
TRIP_NATURAL_KEY = ("HOUSEID", "PERSONID", "TRIPID")

# Trip-level variables that must be present for a trip record to be usable
# for travel-behavior analysis (docs/data_requirements.md sections 3 and 6).
# TRPMILES/TRVLCMIN/STRTTIME/ENDTIME are converted to NaN by
# replace_missing_sentinels above when they hold a sentinel code, so this
# check also catches "not ascertained" trips, not just blank fields.
TRIP_CRITICAL_VALUE_COLUMNS = ("STRTTIME", "ENDTIME", "TRVLCMIN", "TRPMILES")


def replace_missing_sentinels(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    extra_codes: dict[str, tuple[int, ...]] | None = None,
) -> pd.DataFrame:
    """Replace NHTS sentinel codes with NaN in the given columns.

    Only touches columns that are both in `columns` and present in `df`;
    every other column (including all ID columns) is left untouched. Since
    NaN requires a float column, affected columns become float64 even if
    every real value is an integer code - feature engineering can re-cast
    as needed.
    """
    extra_codes = extra_codes or {}
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            continue
        codes = STANDARD_MISSING_CODES + extra_codes.get(column, ())
        numeric = pd.to_numeric(df[column], errors="coerce")
        df[column] = numeric.mask(numeric.isin(codes))
    return df


def filter_valid_records(df: pd.DataFrame, id_columns: tuple[str, ...]) -> pd.DataFrame:
    """Drop rows missing any of `id_columns` (NaN or blank string).

    Every downstream join and groupby depends on these keys being present,
    so a record missing one is unusable rather than merely incomplete.
    """
    mask = pd.Series(True, index=df.index)
    for column in id_columns:
        if column not in df.columns:
            continue
        mask &= df[column].notna() & (df[column].astype(str).str.strip() != "")

    dropped = len(df) - int(mask.sum())
    if dropped:
        logger.info("filter_valid_records: dropping %d row(s) missing %s", dropped, id_columns)
    return df.loc[mask].copy()


def join_nhts_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join household, person, vehicle, and trip into one trip-level table.

    Trip is the base record (one row per trip) since travel behavior is
    what this dataset is for. Person and household attributes are inner-
    joined on PERSONID/HOUSEID (every trip belongs to an existing person in
    an existing household). Vehicle attributes are left-joined on
    HOUSEID+VEHID because not every trip uses a household vehicle - walk/
    bike/transit trips carry VEHID "-1", which has no matching vehicle row
    and correctly produces NaN vehicle columns rather than dropping the
    trip.

    Each table is first restricted to its ID columns plus
    ingest.py's *_REQUIRED_COLUMNS - the variables this project's pipeline
    actually depends on (docs/data_requirements.md sections 2-4) - so the
    result carries relevant travel-behavior variables only, not every raw
    NHTS column.

    A couple of columns are dropped from one side before merging because
    they duplicate a column already coming from another table:
    - household.TRAVDAY/TDAYDATE: the survey day is fixed per household,
      and the trip file already carries both at trip granularity.
    - vehicle.VEHCASEID: identical to trip.VEHCASEID whenever the two are
      joined (both derive from HOUSEID+VEHID); trip's copy is kept since
      it is present even for non-vehicle trips.
    """
    household = tables["household"][list(ingest.HOUSEHOLD_REQUIRED_COLUMNS)].drop(
        columns=["TRAVDAY", "TDAYDATE"]
    )
    person = tables["person"][list(ingest.PERSON_REQUIRED_COLUMNS)]
    vehicle = tables["vehicle"][list(ingest.VEHICLE_REQUIRED_COLUMNS)].drop(columns=["VEHCASEID"])
    trip = tables["trip"][list(ingest.TRIP_REQUIRED_COLUMNS)]

    joined = trip.merge(person, on=["HOUSEID", "PERSONID"], how="inner", validate="m:1")
    joined = joined.merge(household, on="HOUSEID", how="inner", validate="m:1")
    joined = joined.merge(vehicle, on=["HOUSEID", "VEHID"], how="left", validate="m:1")
    return joined


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a joined trip-level table: sentinel values, dupes, invalid trips.

    - Converts NHTS sentinel codes to NaN across the household/person/
      vehicle/trip columns known to use them.
    - Drops exact duplicate trip records (same HOUSEID+PERSONID+TRIPID).
    - Drops trips missing a core travel value (start/end time, travel
      time, or distance) after sentinel conversion, since a trip without
      these can't be placed on a timeline or contribute to distance/
      duration features.
    """
    df = replace_missing_sentinels(df, HOUSEHOLD_MISSING_VALUE_COLUMNS)
    df = replace_missing_sentinels(df, PERSON_MISSING_VALUE_COLUMNS)
    df = replace_missing_sentinels(df, VEHICLE_MISSING_VALUE_COLUMNS, EXTRA_MISSING_CODES)
    df = replace_missing_sentinels(df, TRIP_MISSING_VALUE_COLUMNS)

    before = len(df)
    df = df.drop_duplicates(subset=list(TRIP_NATURAL_KEY))
    if len(df) != before:
        logger.info("clean_trips: dropped %d duplicate trip record(s)", before - len(df))

    before = len(df)
    valid = pd.Series(True, index=df.index)
    for column in TRIP_CRITICAL_VALUE_COLUMNS:
        valid &= df[column].notna()
    df = df.loc[valid].copy()
    if len(df) != before:
        logger.info(
            "clean_trips: dropped %d invalid trip record(s) missing a core travel value",
            before - len(df),
        )

    return df.reset_index(drop=True)


def create_analysis_dataset(raw_dir: Path = ingest.DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Build the cleaned, joined NHTS 2022 travel dataset (one row per trip).

    This is the clean.py stage of download.py -> ingest.py -> clean.py:
    load the four core tables via ingest.py, drop records missing their
    critical ID columns, join them into one trip-level table, and clean
    the result. Feature engineering and everything downstream of it is
    intentionally out of scope - see features/build_features.py.
    """
    tables = ingest.load_all(raw_dir)

    tables["household"] = filter_valid_records(tables["household"], ingest.HOUSEHOLD_ID_COLUMNS)
    tables["person"] = filter_valid_records(tables["person"], ingest.PERSON_ID_COLUMNS)
    tables["vehicle"] = filter_valid_records(tables["vehicle"], ingest.VEHICLE_ID_COLUMNS)
    tables["trip"] = filter_valid_records(tables["trip"], ingest.TRIP_ID_COLUMNS)

    joined = join_nhts_tables(tables)
    return clean_trips(joined)


def save_analysis_dataset(
    df: pd.DataFrame, interim_dir: Path = DEFAULT_INTERIM_DIR
) -> Path:
    """Write the cleaned dataset to data/interim/ as Parquet, returning its path."""
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / ANALYSIS_DATASET_FILENAME
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = create_analysis_dataset()
    output_path = save_analysis_dataset(dataset)
    logger.info(
        "Wrote %d cleaned trip record(s), %d columns, to %s",
        len(dataset),
        len(dataset.columns),
        output_path,
    )
