"""Aggregate cleaned NHTS 2022 trip records into an employee-level driving
behavior feature table.

Produces the "daily travel features" pipeline artifact: one row per
HOUSEID+PERSONID (a person's single surveyed weekday), carrying the
demographic, household, commute, daily-mobility, and vehicle-availability
features specified in `docs/feature_engineering_plan.md`. Reads only
`data/interim/trips_clean.parquet` (the `clean.py` output) - no raw CSVs are
loaded here. Clustering, synthetic employee sampling, EV penetration, and
charging demand are out of scope; see `cluster.py`, `generator/`, and
`scenarios/` for those stages.

NHTS variable dtypes after `clean.py`: columns clean.py's
`replace_missing_sentinels` touches (WORKER, HHFAMINC, WHYTRP1S, TRPTRANS,
TRPHHVEH, STRTTIME, ENDTIME, TRVLCMIN, TRPMILES, GCDWORK, WRKLOC, ...) become
float64 *only if* a sentinel value actually occurs in the data - pandas
does not upcast an all-valid int64 column when `Series.mask` has nothing to
replace. Columns clean.py never lists (TRAVDAY, LOOP_TRIP, R_AGE, HHSIZE,
HHVEHCNT, DRVRCNT) stay whatever dtype `ingest.py`'s `pd.read_csv` inferred
(int64 for all of these in the 2022 public-use files). Every code comparison
below is written to work identically whether the column ended up int64 or
float64 (e.g. `df["WHYTRP1S"] == 10` matches both `10` and `10.0`), so this
mixed-dtype reality does not need to be normalized before use.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from driving_profiles.data import clean

logger = logging.getLogger(__name__)

DEFAULT_INTERIM_DIR = Path("data/interim")
DEFAULT_PROCESSED_DIR = Path("data/processed")
FEATURE_TABLE_FILENAME = "employee_features.parquet"

PERSON_KEY = ["HOUSEID", "PERSONID"]

# --- NHTS code assumptions -------------------------------------------------
# These are grounded in the standard NHTS purpose/mode/day coding convention
# that has held across survey waves, and were cross-checked against the
# actual value distributions in data/raw/*.csv for this project (see
# docs/feature_engineering_plan.md and docs/data_requirements.md). None of
# them were independently re-verified against the 2022 codebook PDF text
# itself, so they are flagged here as assumptions to revisit if downstream
# numbers look inconsistent.

# TRAVDAY: 01=Sunday ... 07=Saturday. Weekday = Monday-Friday. Per plan §1,
# profiles are built from weekday records only (workplace charging is a
# weekday phenomenon); TRAVDAY is fixed per household, so this is really a
# household-level filter - either every member of a household appears, or
# none do.
WEEKDAY_TRAVDAY_CODES = (2, 3, 4, 5, 6)

# WHYTRP1S: collapsed trip-purpose-of-destination summary. 1=Home (by far
# the most common code, as expected), 10=Work. Confirmed against
# data/raw/tripv2pub.csv's value_counts for this project.
HOME_PURPOSE_WHYTRP1S_CODE = 1
WORK_PURPOSE_WHYTRP1S_CODE = 10

# TRPTRANS: personal-vehicle-type modes, i.e. trips this project treats as
# "driving" for VMT/driving-time purposes. 1=Walk and 2=Bicycle are
# explicitly excluded; codes 10+ are transit/rail/air/taxi/other and are
# also excluded. 3=Car and 4=SUV dominate the codes actually used in this
# dataset (10376 and 3297 trips respectively); 5-9 (Van, Pickup truck, and
# a handful of minor personal-vehicle-type codes) are included for
# completeness but are individually rare.
DRIVING_MODE_TRPTRANS_CODES = (3, 4, 5, 6, 7, 8, 9)

# LOOP_TRIP: 1=Yes (trip starts and ends at the same location - no location
# change), 2=No. Loop trips are excluded from trip counts and stop counts
# per plan §2D, since they don't represent a distinct destination.
LOOP_TRIP_YES_CODE = 1

# WORKER: 1=Worker, 2=Not a worker.
WORKER_YES_CODE = 1
WORKER_NO_CODE = 2

# TRPHHVEH: 1=Yes, trip was made in a household vehicle; 2=No.
HOUSEHOLD_VEHICLE_TRPHHVEH_CODE = 1

AGE_BAND_BINS = [-np.inf, 24, 34, 44, 54, 64, np.inf]
AGE_BAND_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]


def _one_row_per_person(trips: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reduce a trip-level frame to one row per person, keeping `columns`.

    Person/household attributes are broadcast identically across every trip
    row for a given person by clean.py's join, so the first occurrence is
    sufficient - this is not sampling or guessing, just de-duplicating a
    constant.
    """
    return trips.drop_duplicates(subset=PERSON_KEY)[PERSON_KEY + columns].copy()


def build_demographic_features(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per person: age, age band, worker status.

    NHTS variables: `R_AGE` (person), `WORKER` (person).

    - `age`: `R_AGE` pass-through. clean.py does not include R_AGE in its
      missing-value columns (no sentinel values were observed for it in the
      2022 public-use file), but it is defensively re-checked here against
      the standard sentinel codes and negative values, since age drives
      `age_band` downstream and a silently-wrong age would corrupt the
      bucket.
    - `age_band`: `age` bucketed into <25, 25-34, 35-44, 45-54, 55-64, 65+
      per plan §2A. NaN when age is missing.
    - `worker_status`/`is_worker`: `WORKER` mapped to "worker"/"non_worker"
      (1/2); NaN/`pd.NA` when WORKER is missing (refused/not ascertained),
      distinct from an explicit "not a worker" answer.
    """
    base = _one_row_per_person(trips, ["R_AGE", "WORKER"])

    age = pd.to_numeric(base["R_AGE"], errors="coerce")
    age = age.mask(age.isin(clean.STANDARD_MISSING_CODES) | (age < 0))
    base["age"] = age
    base["age_band"] = pd.cut(age, bins=AGE_BAND_BINS, labels=AGE_BAND_LABELS)

    worker = pd.to_numeric(base["WORKER"], errors="coerce")
    base["worker_status"] = worker.map({WORKER_YES_CODE: "worker", WORKER_NO_CODE: "non_worker"})
    is_worker = pd.array(worker == WORKER_YES_CODE, dtype="boolean")
    is_worker[worker.isna().to_numpy()] = pd.NA
    base["is_worker"] = is_worker

    return base[PERSON_KEY + ["age", "age_band", "worker_status", "is_worker"]]


def build_household_features(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per person: household income, household size, vehicle count.

    NHTS variables: `HHFAMINC`, `HHSIZE`, `HHVEHCNT` (household, broadcast
    to every member). All three are pass-throughs per plan §2B - HHFAMINC's
    missing-value sentinels are already converted to NaN by clean.py;
    HHSIZE/HHVEHCNT have no sentinel codes in the 2022 file (they are plain
    counts) and are carried through as-is.
    """
    base = _one_row_per_person(trips, ["HHFAMINC", "HHSIZE", "HHVEHCNT"])
    return base.rename(
        columns={
            "HHFAMINC": "household_income_bracket",
            "HHSIZE": "household_size",
            "HHVEHCNT": "household_vehicle_count",
        }
    )


def build_commute_features(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per person: work trip count, commute distance/duration,
    work arrival/departure time.

    NHTS variables: `WHYTRP1S` (marks the trip leg arriving at a
    work-purpose destination, code 10), `GCDWORK` (person-level survey
    great-circle commute-distance estimate), `TRPMILES`/`TRVLCMIN` (trip-
    based distance/duration, summed over the person's work-purpose leg(s)
    as a cross-check against GCDWORK per plan §2C), `STRTTIME`/`ENDTIME`.

    - `work_trip_count`: number of WHYTRP1S=10 legs for the person that day
      (usually 0 or 1; 0 is a real, known value - "didn't go to work" - not
      a missing value, so it is filled rather than left NaN).
    - `commute_distance_survey_miles`: `GCDWORK` pass-through.
    - `commute_distance_trip_miles`: sum of `TRPMILES` over work-purpose
      legs; NaN if the person has none.
    - `commute_duration_minutes`: sum of `TRVLCMIN` over work-purpose legs.
    - `work_arrival_time`: `ENDTIME` (HHMM, e.g. 830.0 = 8:30am) of the
      earliest work-purpose leg.
    - `work_departure_time`: `STRTTIME` of the trip immediately following
      that leg (the next leg in the person's day, identified by trip
      sequence rather than decoding WHYFROM's detailed code space, which
      this project's cleaned dataset does not resolve to a "work" code).
      NaN if no later trip exists that day (e.g. the person's diary ends
      while still at work).

    All commute columns other than `work_trip_count` are NaN for persons
    with no work-purpose leg that day - "not applicable," not "unknown."
    """
    ordered = trips.assign(_seq=trips["TRIPID"].astype(int)).sort_values(
        PERSON_KEY + ["_seq"]
    )
    is_work_leg = ordered["WHYTRP1S"] == WORK_PURPOSE_WHYTRP1S_CODE
    work_legs = ordered.loc[is_work_leg]

    work_trip_count = work_legs.groupby(PERSON_KEY).size().rename("work_trip_count")
    commute_distance_survey = (
        ordered.groupby(PERSON_KEY)["GCDWORK"].first().rename("commute_distance_survey_miles")
    )
    commute_distance_trip = (
        work_legs.groupby(PERSON_KEY)["TRPMILES"].sum(min_count=1).rename("commute_distance_trip_miles")
    )
    commute_duration = (
        work_legs.groupby(PERSON_KEY)["TRVLCMIN"].sum(min_count=1).rename("commute_duration_minutes")
    )

    first_work_leg = work_legs.groupby(PERSON_KEY, as_index=False).first()
    work_arrival_time = (
        first_work_leg.set_index(PERSON_KEY)["ENDTIME"].rename("work_arrival_time")
    )

    next_leg_lookup = ordered[PERSON_KEY + ["_seq", "STRTTIME"]].rename(
        columns={"_seq": "_next_seq", "STRTTIME": "work_departure_time"}
    )
    departure = (
        first_work_leg.assign(_next_seq=first_work_leg["_seq"] + 1)
        .merge(next_leg_lookup, on=PERSON_KEY + ["_next_seq"], how="left")
        .set_index(PERSON_KEY)["work_departure_time"]
    )

    persons_index = pd.MultiIndex.from_frame(trips[PERSON_KEY].drop_duplicates())
    result = pd.DataFrame(index=persons_index).join(
        [
            work_trip_count,
            commute_distance_survey,
            commute_distance_trip,
            commute_duration,
            work_arrival_time,
            departure,
        ]
    )
    result["work_trip_count"] = result["work_trip_count"].fillna(0).astype(int)
    return result.reset_index()


def build_daily_mobility_features(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per person: trips per day, total daily miles/driving
    minutes, average trip distance, number of stops.

    NHTS variables: `TRIPID` (count), `TRPMILES`, `TRVLCMIN`, `TRPTRANS`
    (driving-mode filter for VMT/minutes), `WHYTRP1S` (home-purpose filter
    for stop counting), `LOOP_TRIP` (excluded throughout - a loop trip
    starts and ends at the same location, so it isn't a distinct stop or
    location change per plan §2D).

    - `trips_per_day`: count of non-loop trip legs. 0 is a real value (a
      person whose only recorded trip that day was a loop trip).
    - `total_daily_miles`/`total_driving_minutes`: `TRPMILES`/`TRVLCMIN`
      summed over non-loop trips in a driving mode
      (`DRIVING_MODE_TRPTRANS_CODES`). NaN if the person took no driving
      trips that day (distinct from 0 miles driven).
    - `average_trip_distance_miles`: `total_daily_miles` divided by the
      count of driving trips (not `trips_per_day`, which includes
      walk/bike/transit legs with near-zero distance that would understate
      average driving-trip length). NaN if there are no driving trips.
    - `number_of_stops`: count of non-loop trips whose destination purpose
      is not home (`WHYTRP1S != 1`) - i.e. distinct stops made away from
      home, as opposed to `trips_per_day`'s count of every leg including
      the final return-home leg.
    """
    non_loop = trips.loc[trips["LOOP_TRIP"] != LOOP_TRIP_YES_CODE]

    trips_per_day = non_loop.groupby(PERSON_KEY).size().rename("trips_per_day")

    driving = non_loop.loc[non_loop["TRPTRANS"].isin(DRIVING_MODE_TRPTRANS_CODES)]
    driving_grouped = driving.groupby(PERSON_KEY)
    total_daily_miles = driving_grouped["TRPMILES"].sum(min_count=1).rename("total_daily_miles")
    total_driving_minutes = (
        driving_grouped["TRVLCMIN"].sum(min_count=1).rename("total_driving_minutes")
    )
    driving_trip_count = driving_grouped.size().rename("_driving_trip_count")

    non_home = non_loop.loc[non_loop["WHYTRP1S"] != HOME_PURPOSE_WHYTRP1S_CODE]
    number_of_stops = non_home.groupby(PERSON_KEY).size().rename("number_of_stops")

    persons_index = pd.MultiIndex.from_frame(trips[PERSON_KEY].drop_duplicates())
    result = pd.DataFrame(index=persons_index).join(
        [
            trips_per_day,
            total_daily_miles,
            total_driving_minutes,
            driving_trip_count,
            number_of_stops,
        ]
    )
    result["trips_per_day"] = result["trips_per_day"].fillna(0).astype(int)
    result["number_of_stops"] = result["number_of_stops"].fillna(0).astype(int)
    result["average_trip_distance_miles"] = (
        result["total_daily_miles"] / result["_driving_trip_count"]
    )
    return result.drop(columns=["_driving_trip_count"]).reset_index()


def build_vehicle_availability_features(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per person: vehicle usage and availability indicators.

    NHTS variables: `HHVEHCNT`, `DRVRCNT` (household, broadcast to every
    member), `TRPHHVEH` (trip-level: was a household vehicle used).

    - `vehicles_per_driver`: `HHVEHCNT / DRVRCNT` per plan §2B/§2E - a
      household-level ratio broadcast to each member. NaN when DRVRCNT is
      0 (undefined ratio).
    - `vehicle_per_driver_adequate`: `vehicles_per_driver >= 1`, a simple
      derived flag for "this household has at least as many vehicles as
      drivers." False (not missing) when DRVRCNT is 0, since a zero-driver
      household has no adequate ratio by definition. This is a
      household-level approximation of plan §2E's fuller "vehicle
      household-day overlap" concept (which requires joining every
      household member's trip schedule against a specific vehicle ID) -
      that per-vehicle overlap computation is deferred, not implemented
      here.
    - `used_household_vehicle`/`household_vehicle_trip_count`: whether/how
      many of the person's trips that day used a household vehicle
      (`TRPHHVEH = 1`).
    """
    base = _one_row_per_person(trips, ["HHVEHCNT", "DRVRCNT"])
    drvrcnt = pd.to_numeric(base["DRVRCNT"], errors="coerce")
    hhvehcnt = pd.to_numeric(base["HHVEHCNT"], errors="coerce")
    vehicles_per_driver = hhvehcnt / drvrcnt.replace(0, np.nan)
    base["vehicles_per_driver"] = vehicles_per_driver
    base["vehicle_per_driver_adequate"] = (vehicles_per_driver >= 1).fillna(False)
    base = base.drop(columns=["HHVEHCNT", "DRVRCNT"])

    hh_vehicle_trips = trips.loc[trips["TRPHHVEH"] == HOUSEHOLD_VEHICLE_TRPHHVEH_CODE]
    counts = hh_vehicle_trips.groupby(PERSON_KEY).size().rename("household_vehicle_trip_count")

    result = base.set_index(PERSON_KEY).join(counts)
    trip_count = result["household_vehicle_trip_count"].fillna(0).astype(int)
    result["household_vehicle_trip_count"] = trip_count
    result["used_household_vehicle"] = trip_count > 0
    return result.reset_index()


def create_employee_feature_table(trips_clean: pd.DataFrame) -> pd.DataFrame:
    """Build the employee-level driving behavior feature table.

    Filters `trips_clean` (the clean.py output: one row per trip) to
    weekday records per plan §1, then aggregates up to one row per
    HOUSEID+PERSONID by calling each `build_*_features` function and
    joining the results on that key. Feature engineering stops here -
    clustering, synthetic employee sampling, EV penetration, and charging
    demand are all out of scope (see module docstring).
    """
    weekday = trips_clean.loc[trips_clean["TRAVDAY"].isin(WEEKDAY_TRAVDAY_CODES)].copy()

    feature_table = build_demographic_features(weekday)
    for part in (
        build_household_features(weekday),
        build_commute_features(weekday),
        build_daily_mobility_features(weekday),
        build_vehicle_availability_features(weekday),
    ):
        feature_table = feature_table.merge(part, on=PERSON_KEY, how="left", validate="1:1")

    feature_table["HOUSEID"] = feature_table["HOUSEID"].astype(str)
    feature_table["PERSONID"] = feature_table["PERSONID"].astype(str)

    return feature_table.sort_values(PERSON_KEY).reset_index(drop=True)


def load_cleaned_trips(interim_dir: Path = DEFAULT_INTERIM_DIR) -> pd.DataFrame:
    """Read the clean.py output (`trips_clean.parquet`) from `interim_dir`."""
    path = Path(interim_dir) / clean.ANALYSIS_DATASET_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}. Run "
            "`python -m driving_profiles.data.clean` first."
        )
    return pd.read_parquet(path)


def save_feature_table(df: pd.DataFrame, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    """Write the feature table to data/processed/ as Parquet, returning its path."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / FEATURE_TABLE_FILENAME
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleaned = load_cleaned_trips()
    features = create_employee_feature_table(cleaned)
    output_path = save_feature_table(features)
    logger.info(
        "Wrote %d employee feature row(s), %d columns, to %s",
        len(features),
        len(features.columns),
        output_path,
    )
