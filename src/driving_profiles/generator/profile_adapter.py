"""Parse `DriverProfiles.csv` and annotate/match its driving legs against
this pipeline's own validated NHTS donor-leg pool.

Implements `docs/profile_based_generation_plan.md` §8.3 (leg annotation,
shared by both sides) and §8.4 (donor pool construction + tiered matching).
Distance/duration reconciliation (§8.5) and chronology reconciliation
(§8.6) live in `generator/profile_based.py`, which calls into this module
for the annotate/match step.

This module never writes anywhere; it only reads `DriverProfiles.csv`
(read-only, external, under `data/external/nhts_datasetanalysis`) and
reuses `generator/activity.py`'s already-validated `build_donor_legs`
unchanged - no new donor data, no new external inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from driving_profiles.generator import activity as activity_module
from driving_profiles.generator.time_utils import hhmm_to_minutes

DEFAULT_DRIVER_PROFILES_PATH = Path(
    "data/external/nhts_datasetanalysis/driver_profile_analysis/DriverProfiles.csv"
)

STATE_PARKED = "Parked"
STATE_DRIVING = "Driving"
STATE_CHARGING = "Charging"

LOCATION_HOME = "Home"
LOCATION_WORK = "Work"
LOCATION_DRIVING_SENTINEL = "-1"

TRIP_PURPOSE_HOME = activity_module.TRIP_PURPOSE_HOME
TRIP_PURPOSE_WORK = activity_module.TRIP_PURPOSE_WORK
TRIP_PURPOSE_OTHER = activity_module.TRIP_PURPOSE_OTHER

# §8.4 Tiers: (name, requires_chain_type, requires_chain_segment, time_tolerance_minutes)
# time_tolerance_minutes is None for "unrestricted". Tiers narrow left-to-right within
# 1a/1b/1c (same match keys, widening time window), then drop match keys tier by tier.
TIME_TOLERANCE_TIER_1A = 60.0
TIME_TOLERANCE_TIER_1B = 120.0

MATCH_TIER_1A = "1a"
MATCH_TIER_1B = "1b"
MATCH_TIER_1C = "1c"
MATCH_TIER_2 = "2"
MATCH_TIER_3 = "3"
MATCH_TIER_4 = "4"
MATCH_TIER_UNREPAIRED = "external_unrepaired"


# --- §8.3 leg annotation (shared by both sides) ------------------------------


def _person_key(house_id: pd.Series, person_id: pd.Series) -> pd.Series:
    """Combine HOUSEID/PERSONID into a single hashable groupby key."""
    return house_id.astype(str) + "::" + person_id.astype(str)


def annotate_chain_segments(
    legs: pd.DataFrame, group_col: str, destination_col: str
) -> pd.DataFrame:
    """§8.3 steps 1-5: tag each leg (already in that person's own
    chronological order) with `origin_purpose`, `purpose_transition`,
    `chain_segment`, `leg_index_in_segment`, `chain_type`, and
    `is_arrival_at_work`.

    `legs` must contain every one of a person's legs (not pre-filtered to
    driving-only), since work-occurrence position is defined over the whole
    day's sequence (§8.3 step 1) - callers filter to a driving-only subset
    only *after* calling this function (mirrors §8.4's "computed once from
    donor_legs' existing per-person grouping" before restricting to
    `is_driving_leg`).

    `destination_col` must already be collapsed to `home`/`work`/`other`.

    Step 2's "midday_i - every leg strictly between w_i and w_{i+1}" leaves
    the boundary leg w_{i+1} itself textually ambiguous (also compare
    against `commute_out`'s "up to and including w_1"). Resolved here by
    including each work-occurrence leg in the segment it *terminates*
    (mirroring commute_out's own "up to and including" rule and item 6's
    description of a midday sub-chain as one that "returns to Work") -
    every work-occurrence leg after the first is the last leg of its
    `midday_i` segment, not the first leg of the next one. This keeps the
    partition exhaustive (every leg assigned to exactly one segment, per
    step 2's own requirement) and keeps a work-arrival leg's segment
    membership consistent with its own arrival-anchored role in §8.6.
    """
    df = legs.reset_index(drop=True).copy()

    grouped_dest = df.groupby(group_col, sort=False)[destination_col]
    df["origin_purpose"] = grouped_dest.shift(1).fillna(TRIP_PURPOSE_HOME)
    df["is_arrival_at_work"] = df[destination_col] == TRIP_PURPOSE_WORK
    df["purpose_transition"] = df["origin_purpose"] + "->" + df[destination_col].astype(str)

    chain_segment = np.empty(len(df), dtype=object)
    for _, idx in df.groupby(group_col, sort=False).groups.items():
        idx = list(idx)
        dest = df.loc[idx, destination_col].to_numpy()
        work_positions = np.flatnonzero(dest == TRIP_PURPOSE_WORK)
        n = len(idx)
        seg = np.empty(n, dtype=object)
        if len(work_positions) == 0:
            # No work-occurrence leg at all (real for some donors, not for
            # any of the 250 external profiles - §8.1 item 1 confirms
            # k >= 1 there). Nothing to anchor segments on; the whole
            # sequence is treated as one commute_out-shaped run so it still
            # gets a deterministic label rather than crashing.
            seg[:] = "commute_out"
        else:
            w1 = work_positions[0]
            seg[: w1 + 1] = "commute_out"
            prev_work = w1
            for i in range(1, len(work_positions)):
                wi = work_positions[i]
                seg[prev_work + 1 : wi + 1] = f"midday_{i}"
                prev_work = wi
            if prev_work + 1 < n:
                seg[prev_work + 1 :] = "commute_return"
        for local_pos, global_idx in enumerate(idx):
            chain_segment[global_idx] = seg[local_pos]
    df["chain_segment"] = chain_segment

    df["leg_index_in_segment"] = (
        df.groupby([group_col, "chain_segment"], sort=False).cumcount() + 1
    )
    segment_size = df.groupby([group_col, "chain_segment"], sort=False)[destination_col].transform(
        "size"
    )
    df["chain_type"] = np.where(segment_size == 1, "direct", "chained")
    return df


# --- External profile parsing -------------------------------------------------


def collapse_location(location: pd.Series) -> pd.Series:
    """Collapse `DriverProfiles.csv`'s `Location` column to home/work/other,
    matching the same three-way collapse `classify_trip_purpose` already
    applies to NHTS `WHYTRP1S` on the donor side (§8.3 step 4). Driving
    rows (`Location == "-1"`) have no location of their own and get NaN -
    a driving leg's own purpose comes from the *surrounding* windows, not
    its own row.
    """
    loc = location.astype(str)
    out = pd.Series(TRIP_PURPOSE_OTHER, index=loc.index, dtype=object)
    out[loc == LOCATION_HOME] = TRIP_PURPOSE_HOME
    out[loc == LOCATION_WORK] = TRIP_PURPOSE_WORK
    out[loc == LOCATION_DRIVING_SENTINEL] = np.nan
    return out


def load_driver_profiles(path: Path = DEFAULT_DRIVER_PROFILES_PATH) -> pd.DataFrame:
    """Read `DriverProfiles.csv` read-only and normalize to internal column
    names. Only the fields §8's preservation contract actually uses (`User
    ID`, `State`, `Start time (hour)`, `End time (hour)`, `Location`) are
    kept, plus `Distance (mi)` under a clearly-external-only name
    (`external_distance_mi`) so it can never be mistaken for an output
    field - it is carried through only for the guard test (§8.9 required
    test a) to assert against, never read by the reconciliation logic.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"DriverProfiles.csv not found: {path}")
    raw = pd.read_csv(path)
    df = raw.rename(
        columns={
            "User ID": "user_id",
            "State": "state",
            "Start time (hour)": "start_hour",
            "End time (hour)": "end_hour",
            "Distance (mi)": "external_distance_mi",
            "Location": "location",
        }
    )[["user_id", "state", "start_hour", "end_hour", "external_distance_mi", "location"]]
    df["row_index"] = df.groupby("user_id", sort=False).cumcount()
    df["start_min"] = df["start_hour"] * 60.0
    df["end_min"] = df["end_hour"] * 60.0
    df["collapsed_location"] = collapse_location(df["location"])
    return df.reset_index(drop=True)


def select_profile_user_ids(
    driver_profiles: pd.DataFrame, n_employees: int, seed: int | None = None
) -> list[int]:
    """Pick which `DriverProfiles.csv` `user_id`s to run the reconciliation
    on. Uses every available user (in ascending `user_id` order) when
    `n_employees` is at or above the file's population (250); otherwise
    draws a seeded, reproducible, ascending-sorted subset.
    """
    from driving_profiles.utils import random_seed

    all_ids = sorted(driver_profiles["user_id"].unique().tolist())
    if n_employees >= len(all_ids):
        return all_ids
    rng = random_seed.get_rng(seed)
    chosen = rng.choice(all_ids, size=n_employees, replace=False)
    return sorted(int(x) for x in chosen)


def build_external_driving_legs(driver_profiles: pd.DataFrame) -> pd.DataFrame:
    """Extract every `Driving` row and annotate it per §8.3, using the
    *full* per-user row sequence (Parked/Driving/Charging alike) to locate
    workplace anchors - a driving leg's own `destination_purpose` is the
    collapsed `Location` of the very next row in that user's day.
    """
    driving = driver_profiles.loc[driver_profiles["state"] == STATE_DRIVING].copy()
    lookup = driver_profiles.set_index(["user_id", "row_index"])["collapsed_location"]
    next_key = list(zip(driving["user_id"], driving["row_index"] + 1))
    driving["destination_purpose"] = [lookup.get(key, np.nan) for key in next_key]
    driving = driving.reset_index(drop=True)
    return annotate_chain_segments(
        driving, group_col="user_id", destination_col="destination_purpose"
    )


# --- Donor pool construction (§8.4) -------------------------------------------


def build_donor_leg_pool(
    trips_clean: pd.DataFrame, employee_clusters: pd.DataFrame
) -> pd.DataFrame:
    """Reuse `build_donor_legs` unchanged (§8.4's "Pool" paragraph), then
    annotate every leg per §8.3 using each donor's full per-person trip
    sequence, and only *after* annotating, restrict to legs usable as a
    distance/duration donor: `is_driving_leg` and in-band implied speed
    (`[MIN_PLAUSIBLE_SPEED_MPH, MAX_PLAUSIBLE_SPEED_MPH]`).

    Pre-filtering to plausible speed here - rather than relying on a
    post-hoc fallback - is what makes "100% of substituted legs are
    speed-plausible" true by construction (§8.4).
    """
    donor_legs = activity_module.build_donor_legs(trips_clean, employee_clusters).reset_index(
        drop=True
    )
    donor_legs["_person"] = _person_key(donor_legs["HOUSEID"], donor_legs["PERSONID"])
    donor_legs["destination_purpose"] = donor_legs["trip_purpose"]
    donor_legs["start_min"] = donor_legs["STRTTIME"].apply(hhmm_to_minutes)

    annotated = annotate_chain_segments(
        donor_legs, group_col="_person", destination_col="destination_purpose"
    )

    speed_mph = annotated["TRPMILES"] / (annotated["TRVLCMIN"] / 60.0)
    plausible = (speed_mph >= activity_module.MIN_PLAUSIBLE_SPEED_MPH) & (
        speed_mph <= activity_module.MAX_PLAUSIBLE_SPEED_MPH
    )
    pool = annotated.loc[annotated["is_driving_leg"] & plausible].copy()
    pool = pool.sort_values(["HOUSEID", "PERSONID", "TRIPID"]).reset_index(drop=True)
    return pool


# --- Tiered matching (§8.4) ---------------------------------------------------


def _draw(candidates: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Seeded uniform draw from an already `(HOUSEID, PERSONID, TRIPID)`-
    sorted, non-empty candidate pool (§8.4 "Selection within a tier").
    """
    return candidates.iloc[int(rng.integers(len(candidates)))]


def match_donor_leg(
    external_leg: pd.Series, pool: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.Series | None, str]:
    """§8.4 Tier 1a-4 search: widen progressively, stop at the first
    non-empty tier. Returns `(None, MATCH_TIER_UNREPAIRED)` only if even
    Tier 4 is empty - expected never to happen (see module docstring in
    `profile_based.py`).
    """
    same_transition = pool.loc[pool["purpose_transition"] == external_leg["purpose_transition"]]
    same_transition_segment = same_transition.loc[
        same_transition["chain_segment"] == external_leg["chain_segment"]
    ]
    tier1_pool = same_transition_segment.loc[
        same_transition_segment["chain_type"] == external_leg["chain_type"]
    ]

    if not tier1_pool.empty:
        time_diff = (tier1_pool["start_min"] - external_leg["start_min"]).abs()
        for tolerance, tier_name in (
            (TIME_TOLERANCE_TIER_1A, MATCH_TIER_1A),
            (TIME_TOLERANCE_TIER_1B, MATCH_TIER_1B),
        ):
            within = tier1_pool.loc[time_diff <= tolerance]
            if not within.empty:
                return _draw(within, rng), tier_name
        return _draw(tier1_pool, rng), MATCH_TIER_1C

    if not same_transition_segment.empty:
        return _draw(same_transition_segment, rng), MATCH_TIER_2

    if not same_transition.empty:
        return _draw(same_transition, rng), MATCH_TIER_3

    tier4_pool = pool.loc[pool["destination_purpose"] == external_leg["destination_purpose"]]
    if not tier4_pool.empty:
        return _draw(tier4_pool, rng), MATCH_TIER_4

    return None, MATCH_TIER_UNREPAIRED
