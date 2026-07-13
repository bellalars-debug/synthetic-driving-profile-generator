"""Expand synthetic employee profiles into full daily driving activity
profiles (trip-by-trip) by borrowing and rescaling real NHTS trip chains
from the same behavioral cluster.

Produces the "synthetic driving activity profiles" pipeline artifact
(`data/processed/synthetic_activity.parquet`), one row per synthetic-
employee-leg - the trip-level analog of `data/interim/trips_clean.parquet`
for the synthetic population. Reads `data/processed/synthetic_employees.parquet`
(`sample.py`'s output - the target values each chain is rescaled to),
`data/processed/employee_clusters.parquet` (the bridge from a real
HOUSEID+PERSONID to a `cluster_id`), and `data/interim/trips_clean.parquet`
(the donor pool of real, internally-consistent trip chains). No EV
ownership, charging behavior, or energy calculation is introduced here -
that is deferred to `scenarios/charging_demand.py`. Methodology:
`docs/activity_generation_plan.md`.

## Donor selection and rescaling (plan §3)

Every synthetic employee's chain is a real NHTS respondent's chain,
restricted to donors sharing that employee's `cluster_id` and whose day's
driving status (`has_driving_leg`, derived from `TRPTRANS`) matches whether
the employee's own `total_daily_miles` is non-null - a driver only borrows
from a donor who also drove that day, and a non-driver only borrows from a
donor who had no driving-mode legs, so a donor's raw (unscaled, mode-blind)
`TRPMILES`/`TRVLCMIN` can never be mistaken for driving distance/duration
(`docs/validation_results.md` §6's "donor mode-blindness" finding). Within
that restriction, donors are matched on trip/stop count (exact match first,
widening to +-1 - `MATCH_TOLERANCES`; if that still finds nothing, a final
relaxed tier drops the trip/stop-count restriction entirely rather than
give up a same-cluster/same-driving-status donor - see `select_donor`),
then rescaled (never copied
verbatim): times are shifted so the chain's
first work-purpose leg and the leg immediately following it land on the
employee's own drawn `work_arrival_time`/`work_departure_time`
(`rescale_chain_times`), and distances/durations are scaled so the chain's
total matches `total_daily_miles`/`total_driving_minutes`, with the
work-purpose leg anchored specifically to a commute-distance value picked by
`select_commute_anchor_miles` (`commute_distance_trip_miles` when it
represents a single uncontaminated commute leg, else
`commute_distance_survey_miles`) and passed to `rescale_chain_distances`.
Only when the employee's cluster/driving-status pool has *no* donor at all
(a real risk for sparse clusters - the donor pool is bounded by real NHTS
respondents per cluster, not by synthetic population size) is a minimal
synthesized home->work->home chain used instead
(`build_fallback_chain`, `chain_source == "fallback"`).

## ID handling: a deliberate deviation from plan §4

Plan §4 specifies `donor_houseid`/`donor_personid` columns, mirroring
`sample.py`'s `source_houseid`/`source_personid`, for dev-traceability back
to the donor chain. Unlike `sample.py`, this task's requirements are
explicit that this stage's output must not carry real NHTS IDs at all -
"do not copy real IDs into outputs" - so those two columns are intentionally
omitted here; `chain_source` and the donor's `vehicle_type`/`vehicle_fuel`
(non-identifying, descriptive-only - see plan §4) are kept instead. A
future dev-traceability need should revisit this the same way `sample.py`'s
own docstring flags its opposite-direction deviation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from driving_profiles.data import clean
from driving_profiles.features import cluster as cluster_module
from driving_profiles.features.build_features import (
    DRIVING_MODE_TRPTRANS_CODES,
    HOME_PURPOSE_WHYTRP1S_CODE,
    LOOP_TRIP_YES_CODE,
    WORK_PURPOSE_WHYTRP1S_CODE,
)
from driving_profiles.generator import sample as sample_module
from driving_profiles.generator.time_utils import MINUTES_PER_DAY, hhmm_to_minutes, minutes_to_hhmm
from driving_profiles.utils import random_seed

logger = logging.getLogger(__name__)

DEFAULT_INTERIM_DIR = Path("data/interim")
DEFAULT_PROCESSED_DIR = Path("data/processed")
ACTIVITY_TABLE_FILENAME = "synthetic_activity.parquet"

PERSON_KEY = ["HOUSEID", "PERSONID"]

TRIP_PURPOSE_HOME = "home"
TRIP_PURPOSE_WORK = "work"
TRIP_PURPOSE_OTHER = "other"

DONOR_CHAIN_SOURCE = "donor"
FALLBACK_CHAIN_SOURCE = "fallback"

# plan §3 step 2/3: exact trip/stop-count match first, then widen to +-1
# before select_donor falls through to its own relaxed (no trip/stop-count
# restriction) tier, and only then to a synthesized fallback chain.
MATCH_TOLERANCES = (0, 1)

# select_donor: within a given trip/stop-count tolerance tier, a donor whose
# own workplace arrival/departure falls within this many minutes of the
# synthetic employee's target arrival/departure is preferred over one that
# doesn't - reduces excess variance in non-anchor activity times without
# ever causing a fallback on time-incompatibility alone (time preference is
# applied only as a within-tier ranking, never as a filter that can empty a
# tier - see select_donor).
TIME_MATCH_TOLERANCE_MINUTES = 60.0

# Fallback minimal-chain constants (plan §3 step 3) - only reached when a
# synthetic employee has no donor at all in its cluster/driving-status pool
# (select_donor's relaxed tier is empty too) *and* one of their own drawn
# values needed to build even a minimal chain is itself missing.
# Not a typical code path; see build_fallback_chain.
FALLBACK_COMMUTE_DURATION_MINUTES = 20.0
FALLBACK_WORKDAY_MINUTES = 480.0

# rescale_chain_distances: a rescaled leg's implied speed (distance/duration)
# must fall within this range or its duration is re-derived from
# ASSUMED_AVERAGE_SPEED_MPH instead - guards against a near-zero donor
# distance/duration (a GPS-rounding artifact) blowing up the distance-scale
# factor into an implausible duration. See that function's docstring.
MIN_PLAUSIBLE_SPEED_MPH = 5.0
MAX_PLAUSIBLE_SPEED_MPH = 70.0
ASSUMED_AVERAGE_SPEED_MPH = 30.0

# build_donor_legs: this project models local daily/commute driving (see
# clean.py's docstring - the NHTS long-distance trip file is deliberately
# not loaded), so a donor whose day includes a leg this long is not a
# representative local-commute template even though such legs do appear in
# the regular trip file (~1.4% of trips_clean.parquet).
MAX_PLAUSIBLE_LEG_MILES = 150.0

DONOR_LEG_COLUMNS = [
    "HOUSEID",
    "PERSONID",
    "TRIPID",
    "LOOP_TRIP",
    "STRTTIME",
    "ENDTIME",
    "TRVLCMIN",
    "TRPMILES",
    "WHYTRP1S",
    "TRPTRANS",
    "VEHTYPE",
    "VEHFUEL",
]

OUTPUT_COLUMNS = [
    "synthetic_employee_id",
    "trip_number",
    "departure_time",
    "arrival_time",
    "trip_purpose",
    "distance",
    "duration",
    "dwell_time_after",
    "is_workplace_arrival",
    "is_workplace_departure",
    "workplace_dwell_minutes",
    "chain_source",
    "vehicle_type",
    "vehicle_fuel",
]


# --- Time-of-day helpers -----------------------------------------------------
#
# hhmm_to_minutes/minutes_to_hhmm/MINUTES_PER_DAY now live in
# generator/time_utils.py (imported above) so generator/sample.py can also
# convert to/from true minutes-since-midnight when jittering time-of-day
# fields, without an import cycle (this module already imports sample.py).
# Re-imported into this module's namespace so existing call sites
# (`ac.hhmm_to_minutes`, etc., including in tests and validation/) keep
# working unchanged.


# --- Donor pool construction (plan §3 step 1) --------------------------------


def classify_trip_purpose(whytrp1s: pd.Series) -> pd.Series:
    """Collapse NHTS `WHYTRP1S` to home/work/other (plan §4), reusing the
    same code assumptions `build_features.py` already established and
    cross-checked (`HOME_PURPOSE_WHYTRP1S_CODE`, `WORK_PURPOSE_WHYTRP1S_CODE`)
    rather than re-deriving them here.
    """
    purpose = pd.Series(TRIP_PURPOSE_OTHER, index=whytrp1s.index, dtype=object)
    purpose[whytrp1s == HOME_PURPOSE_WHYTRP1S_CODE] = TRIP_PURPOSE_HOME
    purpose[whytrp1s == WORK_PURPOSE_WHYTRP1S_CODE] = TRIP_PURPOSE_WORK
    return purpose


def build_donor_legs(trips_clean: pd.DataFrame, employee_clusters: pd.DataFrame) -> pd.DataFrame:
    """Restrict `trips_clean` to real respondents with a `cluster_id` (plan
    §3 step 1) and order each respondent's legs chronologically.

    Joins on `PERSON_KEY` (inner - a respondent with no `cluster_id` has no
    archetype to match donors on and is dropped). Loop trips (`LOOP_TRIP`)
    are excluded, mirroring `build_features.py`'s own
    `trips_per_day`/`number_of_stops` convention - donor `trip_count`/
    `stop_count` (`summarize_donor_chains`) must be computed the same way
    those target values were, or donor matching would be comparing
    incompatible counts. Each respondent's remaining legs are sorted by
    `TRIPID` cast to int, matching `build_features.py`'s own trip-sequence
    convention, so the resulting chain shape is that respondent's real,
    ordered day.

    A small fraction of real respondents (observed ~1.5% of this project's
    donor pool) have trip times that don't actually describe a single
    forward-moving day - either `STRTTIME` doesn't increase in `TRIPID`
    order across legs, or an individual leg's own `ENDTIME` is before its
    `STRTTIME` (both are data artifacts of a diary that crosses midnight:
    NHTS records a leg starting at, say, 23:30 and ending at 00:10 as
    `STRTTIME=2330`/`ENDTIME=10` - numerically backwards even though the
    trip itself was a normal ~40-minute drive). The whole premise of
    borrowing a donor's chain (plan §3) is that "every real respondent
    already has a physically coherent day"; a respondent whose own recorded
    times aren't chronological doesn't meet that premise and is dropped
    here entirely (not just the offending leg, since the rest of that
    respondent's chain shape is only meaningful together) rather than
    passed through to `rescale_chain_times`, which assumes chronological
    input throughout.

    A respondent with any single leg longer than `MAX_PLAUSIBLE_LEG_MILES`
    is dropped for the same "not a representative local-commute template"
    reason (see that constant) - this project models local daily/commute
    driving, and an occasional long-distance leg in the regular trip file
    (its dedicated long-distance file is deliberately not loaded, per
    `clean.py`) would otherwise get carried through
    `rescale_chain_distances` unscaled whenever a synthetic employee's own
    `total_daily_miles` is NaN (no target to scale it down to).

    Also adds a per-leg `is_driving_leg` flag (`TRPTRANS` in
    `DRIVING_MODE_TRPTRANS_CODES`, reusing `build_features.py`'s own
    driving-mode definition rather than re-deriving it) - `summarize_donor_chains`
    reduces this to a per-donor `has_driving_leg`, which `select_donor` uses
    to keep a donor whose day had no real driving-mode trips from donating
    its raw (mode-blind) `TRPMILES`/`TRVLCMIN` into a driving-activity chain
    (validation finding: `docs/validation_results.md` §6 "donor
    mode-blindness").
    """
    clusters = employee_clusters.loc[
        employee_clusters["cluster_id"].notna(), PERSON_KEY + ["cluster_id"]
    ]
    legs = trips_clean[DONOR_LEG_COLUMNS].merge(clusters, on=PERSON_KEY, how="inner")
    legs = legs.loc[legs["LOOP_TRIP"] != LOOP_TRIP_YES_CODE].drop(columns="LOOP_TRIP")

    legs["_seq"] = legs["TRIPID"].astype(int)
    legs = legs.sort_values(PERSON_KEY + ["_seq"]).drop(columns="_seq").reset_index(drop=True)
    legs["trip_purpose"] = classify_trip_purpose(legs["WHYTRP1S"])
    legs["is_driving_leg"] = legs["TRPTRANS"].isin(DRIVING_MODE_TRPTRANS_CODES)

    dep_min = legs["STRTTIME"].apply(hhmm_to_minutes)
    arr_min = legs["ENDTIME"].apply(hhmm_to_minutes)
    within_leg_ok = arr_min >= dep_min
    person_groups = [legs["HOUSEID"], legs["PERSONID"]]
    all_legs_within_leg_ok = within_leg_ok.groupby(person_groups).transform("all")
    across_leg_ok = dep_min.groupby(person_groups).transform(
        lambda s: s.is_monotonic_increasing
    )
    is_chronological = all_legs_within_leg_ok & across_leg_ok

    is_plausible_distance = legs["TRPMILES"] <= MAX_PLAUSIBLE_LEG_MILES
    all_legs_plausible = is_plausible_distance.groupby(person_groups).transform("all")

    is_valid_donor = is_chronological & all_legs_plausible
    n_dropped = int((~is_valid_donor).sum())
    if n_dropped:
        logger.info(
            "build_donor_legs: dropping %d leg(s) from donor(s) with a non-chronological "
            "trip-time sequence or an implausibly long leg",
            n_dropped,
        )
    return legs.loc[is_valid_donor].reset_index(drop=True)


def _own_workplace_times(group: pd.DataFrame) -> pd.Series:
    """Per-donor `own_arrival_min`/`own_departure_min` for
    `summarize_donor_chains`: the arrival time (minutes since midnight) of
    the donor's first work-purpose leg, and the departure time of the leg
    immediately following it (NaN if that arrival leg is the donor's last).

    `group` is already in the donor's own chronological order
    (`build_donor_legs`), mirroring `rescale_chain_times`'s own arrival-leg/
    departure-leg indexing. `summarize_donor_chains` calls this over every
    donor group (including one with no work-purpose leg at all, dropped
    afterward via `has_work_leg`), so a group with no work leg returns NaN
    for both rather than raising.
    """
    work_mask = (group["trip_purpose"] == TRIP_PURPOSE_WORK).to_numpy()
    if not work_mask.any():
        return pd.Series({"own_arrival_min": float("nan"), "own_departure_min": float("nan")})
    arrival_idx = int(np.flatnonzero(work_mask)[0])
    own_arrival_min = hhmm_to_minutes(group["ENDTIME"].iloc[arrival_idx])

    departure_idx = arrival_idx + 1
    if departure_idx < len(group):
        own_departure_min = hhmm_to_minutes(group["STRTTIME"].iloc[departure_idx])
    else:
        own_departure_min = float("nan")

    return pd.Series({"own_arrival_min": own_arrival_min, "own_departure_min": own_departure_min})


def summarize_donor_chains(donor_legs: pd.DataFrame) -> pd.DataFrame:
    """One row per donor respondent: `cluster_id`, chain shape
    (`trip_count`, `stop_count`), `has_driving_leg`, and own workplace
    arrival/departure (`own_arrival_min`/`own_departure_min`), restricted to
    donors who actually reached a work-purpose destination that day.

    A donor without a work leg has no anchor for `rescale_chain_times`'s
    arrival/departure rescaling and would be unusable regardless of how
    well its trip/stop count matches (plan §3) - `has_work_leg` is only a
    filter, so (unlike `has_driving_leg`) it's dropped from the returned
    frame once applied, since every remaining row is guaranteed True.

    `has_driving_leg` (`build_donor_legs`'s per-leg `is_driving_leg`,
    reduced with "any") is *not* filtered here - whether a donor needs to
    have driven that day depends on the requesting synthetic employee's own
    `total_daily_miles` (null vs. not), which this function doesn't see.
    It's kept in the output so `select_donor` can filter on it per-call.

    `own_arrival_min`/`own_departure_min` (`_own_workplace_times`) are the
    donor's own workplace arrival/departure, in minutes since midnight,
    computed the same way `rescale_chain_times` locates its arrival/
    departure legs - `select_donor` uses these to prefer a donor whose own
    schedule is already close to the requesting employee's target schedule
    (reduces excess variance in non-anchor activity times) without ever
    filtering a donor out on time grounds alone.
    """
    grouped = donor_legs.groupby(PERSON_KEY)
    summary = grouped.agg(
        cluster_id=("cluster_id", "first"),
        trip_count=("trip_purpose", "size"),
        stop_count=("trip_purpose", lambda s: int((s != TRIP_PURPOSE_HOME).sum())),
        has_work_leg=("trip_purpose", lambda s: bool((s == TRIP_PURPOSE_WORK).any())),
        has_driving_leg=("is_driving_leg", "any"),
    ).reset_index()
    summary = (
        summary.loc[summary["has_work_leg"]].drop(columns="has_work_leg").reset_index(drop=True)
    )

    own_times = grouped.apply(_own_workplace_times).reset_index()
    return summary.merge(own_times, on=PERSON_KEY, how="left")


def select_donor(
    cluster_id,
    trips_per_day: int,
    number_of_stops: int,
    donor_summary: pd.DataFrame,
    rng: np.random.Generator,
    require_driving_leg: bool,
    target_arrival_min: float = float("nan"),
    target_departure_min: float = float("nan"),
) -> tuple[str, str] | None:
    """Pick one donor `(HOUSEID, PERSONID)` from `donor_summary` sharing
    `cluster_id` and matching `require_driving_leg` against the donor's own
    `has_driving_leg`, then matching `trips_per_day`/`number_of_stops` as
    closely as possible (plan §3 step 2: exact match first, widening to +-1
    - `MATCH_TOLERANCES`).

    `require_driving_leg` fixes the donor mode-blindness validation finding
    (`docs/validation_results.md` §6): a synthetic employee with a non-null
    `total_daily_miles` (a real driving day) should only ever be matched to
    a donor whose own day included at least one driving-mode leg, and a
    synthetic employee with a null `total_daily_miles` should only ever be
    matched to a donor whose day had *no* driving-mode legs at all, so no
    real (unscaled) vehicle mileage can leak into that employee's chain -
    see `generate_chain_for_employee`, which derives this flag from
    `total_daily_miles`.

    Within each trip/stop-count tolerance tier (the widening loop below), and
    again within the relaxed tier described below, `target_arrival_min`/
    `target_departure_min` (both minutes since midnight, NaN if not being
    applied) rank that tier's candidates by donor time-schedule compatibility
    before falling back to the tier's full, unrestricted candidate set - a
    *preference* applied strictly within an already-nonempty tier, never a
    filter that can empty one, so time incompatibility alone can never reach
    the fallback chain (`_select_by_time_preference`):

    - Tier A (combined): `abs(own_arrival_min - target_arrival_min) <=
      TIME_MATCH_TOLERANCE_MINUTES`, and either `own_departure_min` or
      `target_departure_min` is NaN, or their difference is also within
      `TIME_MATCH_TOLERANCE_MINUTES`.
    - Tier B (arrival-only): Tier A empty, so just the arrival condition
      above.
    - Tier C (unrestricted): Tier B also empty (including whenever
      `target_arrival_min` itself is NaN, since no donor can then satisfy
      Tier A/B) - the tier's full candidate set, exactly as before this
      preference existed.

    Ties are broken by a seeded random draw against a `PERSON_KEY`-sorted
    candidate list, so donor selection is reproducible given the same `rng`
    sequence rather than depending on incidental row order.

    If no donor matches `trips_per_day`/`number_of_stops` at either
    `MATCH_TOLERANCES` widening step, a final **relaxed tier** is tried
    before giving up: every donor sharing `cluster_id` and `require_driving_leg`
    (i.e. `pool` itself, already guaranteed non-empty at this point), with the
    trip/stop-count restriction dropped entirely but the same Tier A/B/C time
    preference re-applied. This is what keeps a same-cluster/same-driving-status
    donor from being discarded in favor of a handcrafted fallback chain purely
    because its shape didn't happen to fall within +-1 trip/stop. A donor
    returned from this relaxed tier is still `DONOR_CHAIN_SOURCE`, not
    `FALLBACK_CHAIN_SOURCE` (see `generate_chain_for_employee`).

    Returns `None` only when the same-cluster/same-driving-status pool itself
    is empty - the true last resort that reaches `build_fallback_chain`.
    """
    pool = donor_summary.loc[
        (donor_summary["cluster_id"] == cluster_id)
        & (donor_summary["has_driving_leg"] == require_driving_leg)
    ]
    if pool.empty:
        return None

    trip_diff = (pool["trip_count"] - trips_per_day).abs()
    stop_diff = (pool["stop_count"] - number_of_stops).abs()
    for tolerance in MATCH_TOLERANCES:
        candidates = pool.loc[(trip_diff <= tolerance) & (stop_diff <= tolerance)]
        if candidates.empty:
            continue
        return _select_by_time_preference(
            candidates, rng, target_arrival_min, target_departure_min
        )

    # Relaxed tier (plan §3 step 3, Option B): every existing trip/stop-count
    # tolerance tier came up empty, but `pool` (checked non-empty above) still
    # has a same-cluster/same-driving-status donor - use it, still preferring
    # a time-compatible one, rather than reaching for the fallback chain.
    return _select_by_time_preference(pool, rng, target_arrival_min, target_departure_min)


def _select_by_time_preference(
    candidates: pd.DataFrame,
    rng: np.random.Generator,
    target_arrival_min: float,
    target_departure_min: float,
) -> tuple[str, str]:
    """Shared donor time-compatibility preference (`select_donor`'s Tier
    A/B/C, see its docstring) applied to an already trip/stop-count-filtered
    (or, in the relaxed tier, unfiltered) non-empty `candidates` pool.

    Ties within the chosen tier are broken by a seeded random draw against a
    `PERSON_KEY`-sorted candidate list, so donor selection stays reproducible
    given the same `rng` sequence rather than depending on incidental row
    order.
    """
    candidates = candidates.sort_values(PERSON_KEY)

    arrival_compatible = (
        candidates["own_arrival_min"] - target_arrival_min
    ).abs() <= TIME_MATCH_TOLERANCE_MINUTES
    departure_compatible = (
        candidates["own_departure_min"].isna()
        | pd.isna(target_departure_min)
        | (
            (candidates["own_departure_min"] - target_departure_min).abs()
            <= TIME_MATCH_TOLERANCE_MINUTES
        )
    )

    tier_a = candidates.loc[arrival_compatible & departure_compatible]
    tier_b = candidates.loc[arrival_compatible]
    if not tier_a.empty:
        chosen_pool = tier_a
    elif not tier_b.empty:
        chosen_pool = tier_b
    else:
        chosen_pool = candidates

    choice = chosen_pool.iloc[int(rng.integers(len(chosen_pool)))]
    return (choice["HOUSEID"], choice["PERSONID"])


# --- Rescaling (plan §3 "Rescaling the selected donor chain") ---------------


def rescale_chain_times(
    donor_legs: pd.DataFrame, target_arrival_hhmm: float, target_departure_hhmm: float
) -> pd.DataFrame:
    """Shift a donor's leg times so the chain's first work-purpose leg lands
    on `target_arrival_hhmm` and the leg immediately following it lands on
    `target_departure_hhmm`.

    `donor_legs` must already be sorted into the donor's own chronological
    order (`build_donor_legs`) and contain at least one leg with
    `trip_purpose == TRIP_PURPOSE_WORK` (guaranteed by
    `summarize_donor_chains`'s `has_work_leg` filter). Legs up to and
    including that first work-purpose leg are shifted by one offset
    (anchored on arrival); legs from the following leg onward are shifted
    by a second, independent offset (anchored on departure) - deliberately
    not a single uniform shift for the whole chain, since a synthetic
    employee's own `work_departure_time` need not fall the same number of
    minutes after `work_arrival_time` as the donor's did. When there is no
    leg after the arrival leg, or `target_departure_hhmm` is missing or
    would land at or before the (already-shifted) arrival time - both real
    in this project's synthetic population, see module docstring - the
    arrival-anchored offset is applied to the rest of the chain instead,
    since an independent departure anchor isn't usable in that case.

    Adds `departure_time`/`arrival_time` (HHMM) plus internal
    `_departure_minutes`/`_arrival_minutes` (used by
    `_finalize_chain`/`rescale_chain_distances`'s dwell-time computation).
    """
    legs = donor_legs.reset_index(drop=True).copy()
    dep_min = legs["STRTTIME"].apply(hhmm_to_minutes).to_numpy(dtype=float, copy=True)
    arr_min = legs["ENDTIME"].apply(hhmm_to_minutes).to_numpy(dtype=float, copy=True)

    work_mask = (legs["trip_purpose"] == TRIP_PURPOSE_WORK).to_numpy()
    arrival_idx = int(np.flatnonzero(work_mask)[0])

    target_arrival_min = hhmm_to_minutes(target_arrival_hhmm)
    offset_pre = target_arrival_min - arr_min[arrival_idx]
    dep_min[: arrival_idx + 1] += offset_pre
    arr_min[: arrival_idx + 1] += offset_pre

    departure_idx = arrival_idx + 1
    target_departure_min = hhmm_to_minutes(target_departure_hhmm)
    use_departure_anchor = (
        departure_idx < len(legs)
        and not pd.isna(target_departure_min)
        and target_departure_min > target_arrival_min
    )
    if use_departure_anchor:
        offset_post = target_departure_min - dep_min[departure_idx]
        dep_min[departure_idx:] += offset_post
        arr_min[departure_idx:] += offset_post
    else:
        dep_min[departure_idx:] += offset_pre
        arr_min[departure_idx:] += offset_pre

    legs["_departure_minutes"] = np.clip(dep_min, 0, MINUTES_PER_DAY - 1)
    legs["_arrival_minutes"] = np.clip(arr_min, 0, MINUTES_PER_DAY - 1)
    legs["departure_time"] = [minutes_to_hhmm(m) for m in dep_min]
    legs["arrival_time"] = [minutes_to_hhmm(m) for m in arr_min]
    return legs


def rescale_chain_distances(
    legs: pd.DataFrame, total_daily_miles: float, commute_distance_survey_miles: float
) -> pd.DataFrame:
    """Scale a (time-rescaled) donor chain's per-leg `TRPMILES`/`TRVLCMIN`
    so the chain's total distance matches `total_daily_miles` and the
    *first* work-purpose leg (the commute leg `rescale_chain_times` anchors
    arrival on - `work_idx` below) specifically matches
    `commute_distance_survey_miles`.

    Only that one leg is treated as "the commute" and excluded from the
    proportional total-distance scaling; every other leg - including a
    *second* work-purpose leg later in the chain (plan §5's fragmented-
    dwell-window case: a midday departure from and return to work) - is
    part of the "remaining budget" scaled to fit `total_daily_miles`. A
    second work-purpose leg is not the commute; leaving it out of both the
    anchor and the proportional pool (an earlier version of this function's
    bug) would carry the donor's own raw, unscaled distance through
    untouched and inflate the chain's total past `total_daily_miles`.

    Falls back to the donor's own raw values, unscaled, wherever a target
    value is missing (NaN) - real in this project's population (see module
    docstring): a worker's `total_daily_miles` is legitimately NaN when
    none of their recorded trips that day were in a driving mode, and
    rescaling to a target that doesn't exist would just be inventing one.
    `commute_distance_survey_miles == 0` (a real drawn value, e.g. a
    WFH-adjacent employee) is used as-is - it is `NaN`, not `0`, that means
    "no target."

    Duration is scaled by the same factor as its leg's distance so implied
    per-leg speed doesn't drift arbitrarily from the rescaling (plan §3) -
    *unless* that would imply a speed outside
    `[MIN_PLAUSIBLE_SPEED_MPH, MAX_PLAUSIBLE_SPEED_MPH]`. A handful of real
    NHTS legs have a near-zero recorded `TRPMILES` or `TRVLCMIN` (a
    GPS-rounding artifact on a very short hop); dividing by a near-zero
    donor value to derive a scale factor can blow up to an implausible
    duration (thousands of minutes for a normal-length trip) even though
    the *distance* rescaling itself is fine. For those legs, duration is
    instead derived from `ASSUMED_AVERAGE_SPEED_MPH` directly - still "a
    leg's implied speed stays plausible" (plan §3), just via a flat
    assumption instead of the donor's own (implausible, for this one leg)
    implied speed.

    Adds `distance`/`duration` columns.
    """
    legs = legs.copy()
    work_mask = (legs["trip_purpose"] == TRIP_PURPOSE_WORK).to_numpy()
    work_idx = int(np.flatnonzero(work_mask)[0])
    anchor_mask = np.zeros(len(legs), dtype=bool)
    anchor_mask[work_idx] = True

    donor_miles = legs["TRPMILES"].to_numpy(dtype=float)
    donor_minutes = legs["TRVLCMIN"].to_numpy(dtype=float)

    new_miles = donor_miles.copy()

    if pd.notna(commute_distance_survey_miles):
        new_miles[work_idx] = commute_distance_survey_miles

    other_mask = ~anchor_mask
    donor_other_sum = donor_miles[other_mask].sum()
    if pd.notna(total_daily_miles) and donor_other_sum > 0:
        remaining_budget = max(total_daily_miles - new_miles[work_idx], 0.0)
        other_scale = remaining_budget / donor_other_sum
        new_miles[other_mask] = donor_miles[other_mask] * other_scale

    scale = np.divide(new_miles, donor_miles, out=np.ones_like(new_miles), where=donor_miles > 0)
    scaled_duration = donor_minutes * scale
    fallback_duration = (new_miles / ASSUMED_AVERAGE_SPEED_MPH) * 60.0

    implied_speed_mph = np.divide(
        new_miles,
        scaled_duration / 60.0,
        out=np.full_like(new_miles, np.inf),
        where=scaled_duration > 0,
    )
    plausible = (implied_speed_mph >= MIN_PLAUSIBLE_SPEED_MPH) & (
        implied_speed_mph <= MAX_PLAUSIBLE_SPEED_MPH
    )
    duration = np.where(plausible, scaled_duration, fallback_duration)

    legs["distance"] = new_miles
    legs["duration"] = np.clip(duration, 0, None)
    return legs


def compute_dwell_time_after(legs: pd.DataFrame) -> pd.Series:
    """Per-leg minutes spent at that leg's destination before the next leg
    departs, computed directly from the (already rescaled) `_departure_minutes`/
    `_arrival_minutes` columns rather than carried through from the donor's
    own `DWELTIME`.

    Recomputing rather than reusing the donor's `DWELTIME` is what makes
    the workplace dwell window correct after rescaling: `rescale_chain_times`
    applies a different offset before vs. after the work-arrival/departure
    boundary specifically so a synthetic employee's own (not the donor's)
    workday duration is what ends up between those two legs. NaN for the
    last leg of a chain (no next leg that day). This also naturally
    supports plan §5's fragmented-dwell-window case: every leg gets its own
    dwell value, not just a single derived workday-level pair.
    """
    dep = legs["_departure_minutes"].to_numpy(dtype=float)
    arr = legs["_arrival_minutes"].to_numpy(dtype=float)
    dwell = np.full(len(legs), np.nan)
    if len(legs) > 1:
        dwell[:-1] = np.clip(dep[1:] - arr[:-1], 0, None)
    return pd.Series(dwell, index=legs.index)


# --- Fallback minimal chain (plan §3 step 3) ---------------------------------


def build_fallback_chain(employee_row: pd.Series) -> pd.DataFrame:
    """Synthesize a minimal home->work->home chain directly from the
    employee's own drawn values, for the rare cluster/trip-count
    combination with no close-enough donor (plan §3 step 3).

    Every field used is the employee's own; nothing is borrowed from a real
    respondent, so the resulting chain carries no donor vehicle info.
    `commute_duration_minutes`/`commute_distance_survey_miles`/
    `work_departure_time` can themselves be NaN for a given employee (see
    module docstring); `FALLBACK_COMMUTE_DURATION_MINUTES`/
    `FALLBACK_WORKDAY_MINUTES` are the last-resort defaults for that case,
    not typical values.
    """
    arrival_min = hhmm_to_minutes(employee_row["work_arrival_time"])

    commute_minutes = employee_row["commute_duration_minutes"]
    if pd.isna(commute_minutes):
        commute_minutes = FALLBACK_COMMUTE_DURATION_MINUTES

    commute_miles = employee_row["commute_distance_survey_miles"]
    if pd.isna(commute_miles):
        total_miles = employee_row["total_daily_miles"]
        commute_miles = total_miles / 2 if pd.notna(total_miles) else 0.0

    departure = employee_row["work_departure_time"]
    departure_min = hhmm_to_minutes(departure)
    if pd.isna(departure_min) or departure_min <= arrival_min:
        departure_min = arrival_min + FALLBACK_WORKDAY_MINUTES

    outbound_departure_min = max(arrival_min - commute_minutes, 0.0)
    inbound_arrival_min = min(departure_min + commute_minutes, MINUTES_PER_DAY - 1)

    legs = pd.DataFrame(
        {
            "trip_purpose": [TRIP_PURPOSE_WORK, TRIP_PURPOSE_HOME],
            "_departure_minutes": [outbound_departure_min, departure_min],
            "_arrival_minutes": [arrival_min, inbound_arrival_min],
            "distance": [commute_miles, commute_miles],
            "duration": [commute_minutes, commute_minutes],
            "VEHTYPE": [np.nan, np.nan],
            "VEHFUEL": [np.nan, np.nan],
        }
    )
    legs["departure_time"] = legs["_departure_minutes"].apply(minutes_to_hhmm)
    legs["arrival_time"] = legs["_arrival_minutes"].apply(minutes_to_hhmm)
    return legs


# --- Per-employee orchestration -----------------------------------------------


def _finalize_chain(legs: pd.DataFrame, employee_id: str, chain_source: str) -> pd.DataFrame:
    """Assemble one employee's leg table into the final output schema,
    deriving the workplace-arrival/departure/dwell columns and dropping
    every donor/intermediate column not in `OUTPUT_COLUMNS` (this is what
    keeps real `HOUSEID`/`PERSONID` out of the output - see module
    docstring).
    """
    legs = legs.reset_index(drop=True).copy()
    legs["synthetic_employee_id"] = employee_id
    legs["trip_number"] = np.arange(1, len(legs) + 1)
    legs["dwell_time_after"] = compute_dwell_time_after(legs)

    is_work = legs["trip_purpose"] == TRIP_PURPOSE_WORK
    legs["is_workplace_arrival"] = is_work
    legs["is_workplace_departure"] = is_work.shift(fill_value=False)
    legs["workplace_dwell_minutes"] = legs["dwell_time_after"].where(is_work)

    legs["chain_source"] = chain_source
    legs["vehicle_type"] = legs["VEHTYPE"]
    legs["vehicle_fuel"] = legs["VEHFUEL"]
    return legs[OUTPUT_COLUMNS]


def select_commute_anchor_miles(employee_row: pd.Series) -> float:
    """Pick the work-leg rescale anchor for `rescale_chain_distances`:
    `commute_distance_trip_miles` when it represents a single, uncontaminated
    commute leg, else `commute_distance_survey_miles` (the prior, always-used
    anchor).

    `commute_distance_trip_miles` (`build_features.py`'s
    `build_commute_features`) sums `TRPMILES` over *every* work-purpose leg
    in the day, so it only matches "the one commute leg"
    `rescale_chain_distances` anchors to when there was exactly one
    (`work_trip_count == 1`) - otherwise it silently includes a second
    midday work-purpose leg (plan §5's fragmented-dwell-window case) and
    would overstate the commute. It is also, unlike
    `commute_distance_survey_miles`, not bounded by any upstream
    plausibility filter, so `MAX_PLAUSIBLE_LEG_MILES` (the same bound
    `build_donor_legs` already applies to a usable donor leg) guards against
    its own uncapped tail. `commute_distance_survey_miles` remains the
    fallback in every other case, unchanged from prior behavior.
    """
    trip_miles = employee_row["commute_distance_trip_miles"]
    if (
        employee_row["work_trip_count"] == 1
        and pd.notna(trip_miles)
        and trip_miles <= MAX_PLAUSIBLE_LEG_MILES
    ):
        return trip_miles
    return employee_row["commute_distance_survey_miles"]


def generate_chain_for_employee(
    employee_row: pd.Series,
    donor_summary: pd.DataFrame,
    donor_legs_by_person: dict[tuple[str, str], pd.DataFrame],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build one synthetic employee's full daily chain: select and rescale a
    donor (plan §3), or fall back to a minimal synthesized chain if none is
    close enough.

    Requires the selected donor's `has_driving_leg` to match whether this
    employee's own `total_daily_miles` is non-null - see `select_donor`'s
    `require_driving_leg` docstring (donor mode-blindness fix).

    Also passes this employee's own `work_arrival_time`/`work_departure_time`
    (converted to minutes since midnight) through to `select_donor` as
    `target_arrival_min`/`target_departure_min`, so donor selection can
    prefer a donor whose own workplace schedule is already close to this
    employee's target schedule (see `select_donor`'s tiered time-preference
    docstring).
    """
    donor_key = select_donor(
        employee_row["cluster_id"],
        int(employee_row["trips_per_day"]),
        int(employee_row["number_of_stops"]),
        donor_summary,
        rng,
        require_driving_leg=pd.notna(employee_row["total_daily_miles"]),
        target_arrival_min=hhmm_to_minutes(employee_row["work_arrival_time"]),
        target_departure_min=hhmm_to_minutes(employee_row["work_departure_time"]),
    )
    if donor_key is None:
        legs = build_fallback_chain(employee_row)
        chain_source = FALLBACK_CHAIN_SOURCE
    else:
        donor_legs = donor_legs_by_person[donor_key]
        legs = rescale_chain_times(
            donor_legs, employee_row["work_arrival_time"], employee_row["work_departure_time"]
        )
        legs = rescale_chain_distances(
            legs, employee_row["total_daily_miles"], select_commute_anchor_miles(employee_row)
        )
        chain_source = DONOR_CHAIN_SOURCE

    return _finalize_chain(legs, employee_row["synthetic_employee_id"], chain_source)


def generate_synthetic_activity(
    synthetic_employees: pd.DataFrame,
    employee_clusters: pd.DataFrame,
    trips_clean: pd.DataFrame,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run the full activity-generation pipeline: build the donor pool, then
    generate one rescaled (or fallback) chain per synthetic employee, in
    `synthetic_employees`'s own row order.

    Reproducible given the same `seed` against the same inputs: `rng` is
    drawn once and consumed sequentially per employee in table order, so
    identical inputs always produce identical donor picks (plan §3/§6).
    """
    rng = random_seed.get_rng(seed)
    donor_legs = build_donor_legs(trips_clean, employee_clusters)
    donor_summary = summarize_donor_chains(donor_legs)
    donor_legs_by_person = {key: group for key, group in donor_legs.groupby(PERSON_KEY)}

    chains = [
        generate_chain_for_employee(row, donor_summary, donor_legs_by_person, rng)
        for _, row in synthetic_employees.iterrows()
    ]
    activity = pd.concat(chains, ignore_index=True)

    per_employee_source = activity.drop_duplicates("synthetic_employee_id")["chain_source"]
    n_fallback = int((per_employee_source == FALLBACK_CHAIN_SOURCE).sum())
    logger.info(
        "generate_synthetic_activity: %d/%d synthetic employee(s) used a fallback chain (%.1f%%)",
        n_fallback,
        len(synthetic_employees),
        100 * n_fallback / len(synthetic_employees) if len(synthetic_employees) else 0.0,
    )
    return activity


# --- I/O ----------------------------------------------------------------------


def load_synthetic_employees(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Read `sample.py`'s output (`synthetic_employees.parquet`)."""
    path = Path(processed_dir) / sample_module.SYNTHETIC_EMPLOYEE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic employee table not found: {path}. Run "
            "`python -m driving_profiles.generator.sample` first."
        )
    return pd.read_parquet(path)


def load_employee_clusters(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Read `cluster.py`'s output (`employee_clusters.parquet`)."""
    path = Path(processed_dir) / cluster_module.CLUSTER_TABLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Clustered employee table not found: {path}. Run "
            "`python -m driving_profiles.features.cluster` first."
        )
    return pd.read_parquet(path)


def load_trips_clean(interim_dir: Path = DEFAULT_INTERIM_DIR) -> pd.DataFrame:
    """Read `clean.py`'s output (`trips_clean.parquet`)."""
    path = Path(interim_dir) / clean.ANALYSIS_DATASET_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned trip dataset not found: {path}. Run "
            "`python -m driving_profiles.data.clean` first."
        )
    return pd.read_parquet(path)


def save_synthetic_activity(
    activity: pd.DataFrame, processed_dir: Path = DEFAULT_PROCESSED_DIR
) -> Path:
    """Write the synthetic activity table to `processed_dir / ACTIVITY_TABLE_FILENAME`."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / ACTIVITY_TABLE_FILENAME
    activity.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Generate synthetic daily driving activity profiles."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: driving_profiles.utils.random_seed.DEFAULT_SEED).",
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM_DIR)
    args = parser.parse_args()

    synthetic_employees = load_synthetic_employees(args.processed_dir)
    employee_clusters = load_employee_clusters(args.processed_dir)
    trips_clean = load_trips_clean(args.interim_dir)

    activity = generate_synthetic_activity(
        synthetic_employees, employee_clusters, trips_clean, seed=args.seed
    )
    output_path = save_synthetic_activity(activity, args.processed_dir)
    logger.info(
        "Wrote %d activity leg(s) for %d synthetic employee(s) to %s",
        len(activity),
        activity["synthetic_employee_id"].nunique(),
        output_path,
    )
