"""Load raw NHTS 2022 CSV files into typed DataFrames.

Ingests the four core NHTS 2022 daily-travel files (household, person,
vehicle, trip) as-is. This module only reads and validates; it does not
join, clean, filter, or engineer features - see clean.py and
features/build_features.py for those stages.

ldtv2pub.csv (long-distance trips) is intentionally not loaded here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")

# Codebook type "C" (character) key columns for each file. These are
# fixed-width, zero-padded IDs (e.g. PERSONID "01") - reading them as
# anything but strings silently drops leading zeros and breaks joins
# in clean.py. See docs/data_requirements.md section 5.
HOUSEHOLD_ID_COLUMNS = ("HOUSEID",)
PERSON_ID_COLUMNS = ("HOUSEID", "PERSONID")
VEHICLE_ID_COLUMNS = ("HOUSEID", "VEHID", "VEHCASEID")
TRIP_ID_COLUMNS = ("HOUSEID", "PERSONID", "TRIPID", "SEQ_TRIPID", "TDCASEID", "VEHID", "VEHCASEID")

# Columns this project's pipeline depends on downstream (see
# docs/data_requirements.md sections 2-3). Missing any of these means the
# NHTS release layout has changed and clean.py/build_features.py
# assumptions need re-checking before trusting the file.
HOUSEHOLD_REQUIRED_COLUMNS = HOUSEHOLD_ID_COLUMNS + (
    "HHSIZE",
    "HHFAMINC",
    "HOMEOWN",
    "HHVEHCNT",
    "DRVRCNT",
    "LIF_CYC",
    "HH_RACE",
    "HH_HISP",
    "URBAN",
    "URBRUR",
    "URBANSIZE",
    "MSACAT",
    "MSASIZE",
    "CENSUS_R",
    "CENSUS_D",
    "TRAVDAY",
    "TDAYDATE",
    "WRKCOUNT",
)

PERSON_REQUIRED_COLUMNS = PERSON_ID_COLUMNS + (
    "R_AGE",
    "R_SEX",
    "R_RACE",
    "R_HISP",
    "EDUC",
    "R_RELAT",
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

VEHICLE_REQUIRED_COLUMNS = VEHICLE_ID_COLUMNS + (
    "VEHTYPE",
    "VEHFUEL",
    "ANNMILES",
)

TRIP_REQUIRED_COLUMNS = TRIP_ID_COLUMNS + (
    "TRPHHVEH",
    "WHYFROM",
    "WHYTO",
    "WHYTRP1S",
    "WHYTRP90",
    "TRIPPURP",
    "TRPTRANS",
    "NUMONTRP",
    "LOOP_TRIP",
    "STRTTIME",
    "ENDTIME",
    "TRVLCMIN",
    "DWELTIME",
    "TRAVDAY",
    "TDAYDATE",
    "TRPMILES",
    "VMT_MILE",
    "WHODROVE",
    "WHODROVE_IMP",
)


def _read_nhts_csv(
    path: Path, id_columns: tuple[str, ...], required_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Read one NHTS CSV, validating it exists and has the expected columns.

    ID/key columns are forced to string dtype so zero-padded values like
    PERSONID "01" survive intact; every other column is left to pandas'
    default type inference.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required NHTS file not found: {path}. Run "
            "`python -m driving_profiles.data.download` to fetch and "
            "extract the NHTS 2022 public-use CSVs into data/raw/."
        )

    header = pd.read_csv(path, nrows=0).columns
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(
            f"{path.name} is missing expected column(s): {missing}. "
            "The NHTS release layout may have changed since "
            "docs/data_requirements.md was written - re-verify before "
            "trusting this file."
        )

    dtype = {column: str for column in id_columns if column in header}
    return pd.read_csv(path, dtype=dtype)


def load_household(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the household file (hhv2pub.csv). One row per household."""
    return _read_nhts_csv(
        Path(raw_dir) / "hhv2pub.csv", HOUSEHOLD_ID_COLUMNS, HOUSEHOLD_REQUIRED_COLUMNS
    )


def load_person(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the person file (perv2pub.csv). One row per household member."""
    return _read_nhts_csv(
        Path(raw_dir) / "perv2pub.csv", PERSON_ID_COLUMNS, PERSON_REQUIRED_COLUMNS
    )


def load_vehicle(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the vehicle file (vehv2pub.csv). One row per household vehicle."""
    return _read_nhts_csv(
        Path(raw_dir) / "vehv2pub.csv", VEHICLE_ID_COLUMNS, VEHICLE_REQUIRED_COLUMNS
    )


def load_trip(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the trip file (tripv2pub.csv). One row per trip on the travel day."""
    return _read_nhts_csv(Path(raw_dir) / "tripv2pub.csv", TRIP_ID_COLUMNS, TRIP_REQUIRED_COLUMNS)


def load_all(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load the four core NHTS 2022 files into a dict keyed by table name.

    Does not join, clean, filter, or otherwise transform them - see
    clean.py for that.
    """
    return {
        "household": load_household(raw_dir),
        "person": load_person(raw_dir),
        "vehicle": load_vehicle(raw_dir),
        "trip": load_trip(raw_dir),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name, table in load_all().items():
        logger.info("%s: %d rows, %d columns", name, len(table), len(table.columns))
