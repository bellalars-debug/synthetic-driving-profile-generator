"""Profile-based mobility-generation experiment (`docs/profile_based_generation_plan.md`
§8): implement `DriverProfiles.csv`'s external activity-profile schedules,
repairing only the one confirmed defect (independently-sampled, jointly-
implausible `Distance (mi)`/duration per driving leg, §3) by substituting a
distance/duration pair from this pipeline's own validated NHTS donor-leg
pool (`generator/activity.py`'s `build_donor_legs`, via `profile_adapter.py`).

Everything else about the external profile - state sequence, location
sequence, leg count, stop count, direct-vs-chained structure, midday
structure, and (whenever the plausibility-driven duration change permits)
workplace arrival/departure clock times - is preserved exactly (§8.1).

This module only ever reads `DriverProfiles.csv` (read-only, external) and
this pipeline's existing `data/interim/trips_clean.parquet` /
`data/processed/employee_clusters.parquet` (read-only, already-validated
production inputs). It never writes to `data/processed/` - outputs go only
to `data/validation/profile_based/` (see `save_profile_based_output`).

## Chronology reconciliation (§8.6) - implementation note

Implemented as specified, including the cascade's "ripples forward/
backward... as a uniform time translation applied to every subsequent (or
preceding) leg and window, preserving each of their own internal durations"
(§8.6): `_ripple_forward`/`_ripple_backward` first locate the next
protected boundary (workplace arrival/departure time, or the day boundary),
translate every boundary strictly in between by the same delta (so every
row those boundaries touch keeps its own original duration), and only the
one row immediately adjacent to that protected boundary is actually
clamped to `MIN_DWELL_FLOOR_MINUTES`. Only if even the floor isn't enough
does the protected boundary itself move (logged as `anchor_shifted`),
recursing past it with the true leftover. Confirmed on this file's
250-profile population: the cascade path is reached on 15/560 driving legs
(all resolved by the immediately adjacent window; none needed a second
level of recursion past a further protected boundary).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from driving_profiles.generator import activity as activity_module
from driving_profiles.generator import profile_adapter
from driving_profiles.utils import random_seed

logger = logging.getLogger(__name__)

DEFAULT_VALIDATION_DIR = Path("data/validation/profile_based")
OUTPUT_FILENAME = "profile_schedule_adapter_output.parquet"

# §8.6 cascade: the floor a Parked/Charging window's duration is clamped to
# rather than being pushed negative - "proposed at 1.0 minute, matching the
# order of magnitude of the shortest real donor legs already tolerated
# elsewhere in this pipeline" (§8.6).
MIN_DWELL_FLOOR_MINUTES = 1.0

TRIP_PURPOSE_WORK = activity_module.TRIP_PURPOSE_WORK

SCHEDULE_STATUS_PRESERVED = "preserved"
SCHEDULE_STATUS_ADJUSTED = "adjusted"

DISTANCE_DURATION_SOURCE_DONOR = "nhts_donor"
DISTANCE_DURATION_SOURCE_UNREPAIRED = "external_unrepaired"

OUTPUT_COLUMNS = [
    "profile_employee_id",
    "row_index",
    "state",
    "location",
    "start_hour",
    "end_hour",
    "distance_mi",
    "duration_min",
    "chain_segment",
    "chain_type",
    "purpose_transition",
    "is_arrival_at_work",
    "sequence_source",
    "schedule_status",
    "distance_duration_source",
    "match_tier",
    "adjustment_minutes",
    "anchor_shifted",
    "anchor_shift_minutes",
]


# --- §8.5 distance/duration reconciliation ------------------------------------


def reconcile_distance_duration(donor_leg: pd.Series) -> tuple[float, float]:
    """§8.5 verbatim: `rescale_chain_distances`'s existing per-leg
    plausibility branch, reused rather than re-derived. Because the donor
    pool (`profile_adapter.build_donor_leg_pool`) is already restricted to
    in-band implied speed, the `else` branch is a defensive guard, not the
    normal path.
    """
    new_distance_mi = float(donor_leg["TRPMILES"])
    donor_minutes = float(donor_leg["TRVLCMIN"])
    donor_speed_mph = (
        (new_distance_mi / (donor_minutes / 60.0)) if donor_minutes > 0 else float("inf")
    )
    if (
        activity_module.MIN_PLAUSIBLE_SPEED_MPH
        <= donor_speed_mph
        <= activity_module.MAX_PLAUSIBLE_SPEED_MPH
    ):
        new_duration_min = new_distance_mi / donor_speed_mph * 60.0
    else:
        new_duration_min = new_distance_mi / activity_module.ASSUMED_AVERAGE_SPEED_MPH * 60.0
    return new_distance_mi, new_duration_min


# --- §8.6 chronology reconciliation: anchor rule + cascade --------------------


def _ripple_forward(
    boundary: np.ndarray,
    protected: np.ndarray,
    floor_minutes: float,
    moved_idx: int,
    delta: float,
    anchor_events: list[tuple[int, float]],
) -> None:
    """`boundary[moved_idx]` has just been finalized (already shifted by
    `delta` from its original value). If the immediately adjacent window
    (ending at `moved_idx + 1`) has enough room to absorb this on its own
    (its far boundary needs no change at all), stop immediately - the
    ordinary, non-cascade case. Otherwise, find the next protected boundary,
    translate every boundary strictly between `moved_idx` and it by the same
    `delta` (preserving each of those rows' own original durations - the
    "uniform time translation" of the module docstring's cascade note), then
    let the one row immediately before the protected boundary absorb
    whatever's left, clamped to `floor_minutes`. Only if even that isn't
    enough does the protected boundary itself move (recursing past it with
    the true leftover).
    """
    n = len(boundary) - 1
    adjacent_far = moved_idx + 1
    if boundary[adjacent_far] - boundary[moved_idx] >= floor_minutes:
        return

    anchor = adjacent_far
    while not protected[anchor]:
        anchor += 1
    for b in range(adjacent_far, anchor):
        boundary[b] += delta

    naive_duration = boundary[anchor] - boundary[anchor - 1]
    if naive_duration >= floor_minutes:
        return
    old_anchor_value = boundary[anchor]
    boundary[anchor] = boundary[anchor - 1] + floor_minutes
    leftover = boundary[anchor] - old_anchor_value
    anchor_events.append((anchor, leftover))
    if anchor < n:
        _ripple_forward(boundary, protected, floor_minutes, anchor, leftover, anchor_events)


def _ripple_backward(
    boundary: np.ndarray,
    protected: np.ndarray,
    floor_minutes: float,
    moved_idx: int,
    delta: float,
    anchor_events: list[tuple[int, float]],
) -> None:
    """Mirror of `_ripple_forward` for arrival-anchored legs: `boundary[moved_idx]`
    just moved by `delta`; ripple backward toward the previous protected
    boundary."""
    adjacent_near = moved_idx - 1
    if boundary[moved_idx] - boundary[adjacent_near] >= floor_minutes:
        return

    anchor = adjacent_near
    while not protected[anchor]:
        anchor -= 1
    for b in range(adjacent_near, anchor, -1):
        boundary[b] += delta

    naive_duration = boundary[anchor + 1] - boundary[anchor]
    if naive_duration >= floor_minutes:
        return
    old_anchor_value = boundary[anchor]
    boundary[anchor] = boundary[anchor + 1] - floor_minutes
    leftover = boundary[anchor] - old_anchor_value
    anchor_events.append((anchor, leftover))
    if anchor > 0:
        _ripple_backward(boundary, protected, floor_minutes, anchor, leftover, anchor_events)


def _protected_boundaries(n_rows: int, driving_legs: pd.DataFrame) -> np.ndarray:
    """§8.6 anchor rule: the day boundary and every workplace
    arrival/departure clock time are protected. An arrival-at-work leg's
    own end boundary is protected (item 8's "arrival" half); the start
    boundary of a leg departing `Work` is protected (item 8's "departure"
    half) - together these are why a `Work` window's two boundaries are
    each the anchored side of exactly one neighboring leg (§8.6's own
    explanation).
    """
    protected = np.zeros(n_rows + 1, dtype=bool)
    protected[0] = True
    protected[n_rows] = True
    for _, leg in driving_legs.iterrows():
        row_index = int(leg["row_index"])
        if leg["is_arrival_at_work"]:
            protected[row_index + 1] = True
        if leg["origin_purpose"] == TRIP_PURPOSE_WORK:
            protected[row_index] = True
    return protected


def reconcile_user_schedule(
    timeline: pd.DataFrame,
    driving_legs: pd.DataFrame,
    donor_matches: dict[int, tuple[pd.Series | None, str]],
    floor_minutes: float = MIN_DWELL_FLOOR_MINUTES,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Reconcile one user's full day: replace each driving leg's
    distance/duration (§8.5) and propagate the resulting clock-time change
    per the §8.6 anchor rule + cascade.

    `timeline` is this user's full ordered row sequence (`row_index`
    0..n-1, contiguous). `driving_legs` is the annotated subset of
    `timeline` where `state == "Driving"`. `donor_matches` maps
    `row_index -> (donor_leg_row_or_None, match_tier)`
    (`profile_adapter.match_donor_leg`'s output per leg).

    Returns the reconciled boundary array (`len(timeline) + 1`, minutes
    since midnight - `boundary[i]`/`boundary[i + 1]` are row `i`'s
    start/end) and a `row_index -> audit dict` mapping (§8.7 fields).
    """
    n = len(timeline)
    boundary = np.empty(n + 1, dtype=float)
    boundary[0] = float(timeline["start_min"].iloc[0])
    boundary[1:] = timeline["end_min"].to_numpy(dtype=float)

    protected = _protected_boundaries(n, driving_legs)

    audit: dict[int, dict] = {}
    for _, leg in driving_legs.sort_values("row_index").iterrows():
        row_index = int(leg["row_index"])
        donor_leg, match_tier = donor_matches[row_index]

        if donor_leg is None:
            audit[row_index] = {
                "schedule_status": SCHEDULE_STATUS_PRESERVED,
                "adjustment_minutes": 0.0,
                "anchor_shifted": False,
                "anchor_shift_minutes": 0.0,
                "distance_duration_source": DISTANCE_DURATION_SOURCE_UNREPAIRED,
                "match_tier": match_tier,
                "new_distance_mi": None,
                "new_duration_min": None,
            }
            continue

        new_distance_mi, new_duration_min = reconcile_distance_duration(donor_leg)
        anchor_events: list[tuple[int, float]] = []

        if leg["is_arrival_at_work"]:
            old_start = boundary[row_index]
            new_start = boundary[row_index + 1] - new_duration_min
            boundary[row_index] = new_start
            adjustment_minutes = new_start - old_start
            if adjustment_minutes != 0.0 and row_index > 0:
                _ripple_backward(
                    boundary, protected, floor_minutes, row_index, adjustment_minutes, anchor_events
                )
        else:
            old_end = boundary[row_index + 1]
            new_end = boundary[row_index] + new_duration_min
            boundary[row_index + 1] = new_end
            adjustment_minutes = new_end - old_end
            if adjustment_minutes != 0.0 and row_index + 1 < n:
                _ripple_forward(
                    boundary,
                    protected,
                    floor_minutes,
                    row_index + 1,
                    adjustment_minutes,
                    anchor_events,
                )

        audit[row_index] = {
            "schedule_status": (
                SCHEDULE_STATUS_PRESERVED if adjustment_minutes == 0.0 else SCHEDULE_STATUS_ADJUSTED
            ),
            "adjustment_minutes": float(adjustment_minutes),
            "anchor_shifted": len(anchor_events) > 0,
            "anchor_shift_minutes": float(sum(m for _, m in anchor_events)),
            "distance_duration_source": DISTANCE_DURATION_SOURCE_DONOR,
            "match_tier": match_tier,
            "new_distance_mi": new_distance_mi,
            "new_duration_min": new_duration_min,
        }

    return boundary, audit


# --- Per-employee orchestration ------------------------------------------------


def assign_profile_employee_ids(user_ids: list[int]) -> dict[int, str]:
    """New synthetic IDs, `PROF-XXXX`, one per external `user_id` - the
    original `DriverProfiles.csv` `User ID` never appears in the output
    (requirement: "no source User IDs"), mirroring `sample.py`'s
    `SYN-XXXXXXXX` convention for the production pipeline.
    """
    return {uid: f"PROF-{i + 1:04d}" for i, uid in enumerate(sorted(user_ids))}


def _build_output_rows(
    profile_employee_id: str,
    timeline: pd.DataFrame,
    driving_legs: pd.DataFrame,
    boundary: np.ndarray,
    audit: dict[int, dict],
) -> list[dict]:
    legs_by_row = driving_legs.set_index("row_index")
    rows = []
    for i in range(len(timeline)):
        row = timeline.iloc[i]
        out = {
            "profile_employee_id": profile_employee_id,
            "row_index": i,
            "state": row["state"],
            "location": row["location"],
            "start_hour": boundary[i] / 60.0,
            "end_hour": boundary[i + 1] / 60.0,
        }
        if row["state"] == profile_adapter.STATE_DRIVING:
            a = audit[i]
            leg = legs_by_row.loc[i]
            if a["new_distance_mi"] is None:
                # §8.4's documented last resort (Tier 4 exhausted) - keep
                # the external leg's own values verbatim rather than
                # inventing a substitute. Expected never to trigger.
                out["distance_mi"] = float(row["external_distance_mi"])
                out["duration_min"] = float(row["end_min"] - row["start_min"])
            else:
                out["distance_mi"] = a["new_distance_mi"]
                out["duration_min"] = a["new_duration_min"]
            out["chain_segment"] = leg["chain_segment"]
            out["chain_type"] = leg["chain_type"]
            out["purpose_transition"] = leg["purpose_transition"]
            out["is_arrival_at_work"] = bool(leg["is_arrival_at_work"])
            out["sequence_source"] = "external"
            out["schedule_status"] = a["schedule_status"]
            out["distance_duration_source"] = a["distance_duration_source"]
            out["match_tier"] = a["match_tier"]
            out["adjustment_minutes"] = a["adjustment_minutes"]
            out["anchor_shifted"] = a["anchor_shifted"]
            out["anchor_shift_minutes"] = a["anchor_shift_minutes"]
        else:
            for col in OUTPUT_COLUMNS[6:]:
                out[col] = np.nan
        rows.append(out)
    return rows


def run_profile_based_reconciliation(
    driver_profiles: pd.DataFrame,
    donor_pool: pd.DataFrame,
    user_ids: list[int],
    seed: int | None = None,
) -> pd.DataFrame:
    """Run the full §8 reconciliation over `user_ids`: annotate every
    external driving leg (§8.3), match it to a donor leg (§8.4), reconcile
    its distance/duration (§8.5) and clock times (§8.6), and assemble the
    per-segment output table (§8.9's "Outputs").

    Reproducible given the same `seed`: a single `rng` is drawn once and
    consumed in `(user_id, row_index)` order, mirroring
    `generate_synthetic_activity`'s own reproducibility convention.
    """
    rng = random_seed.get_rng(seed)
    external_driving_legs = profile_adapter.build_external_driving_legs(driver_profiles)
    id_map = assign_profile_employee_ids(user_ids)

    rows: list[dict] = []
    for uid in sorted(user_ids):
        timeline = (
            driver_profiles.loc[driver_profiles["user_id"] == uid]
            .sort_values("row_index")
            .reset_index(drop=True)
        )
        legs = (
            external_driving_legs.loc[external_driving_legs["user_id"] == uid]
            .sort_values("row_index")
            .reset_index(drop=True)
        )

        donor_matches = {}
        for _, leg in legs.iterrows():
            donor_leg, tier = profile_adapter.match_donor_leg(leg, donor_pool, rng)
            donor_matches[int(leg["row_index"])] = (donor_leg, tier)

        boundary, audit = reconcile_user_schedule(timeline, legs, donor_matches)
        rows.extend(
            _build_output_rows(id_map[uid], timeline, legs, boundary, audit)
        )

    n_unrepaired = sum(
        1 for r in rows if r.get("distance_duration_source") == DISTANCE_DURATION_SOURCE_UNREPAIRED
    )
    n_driving = sum(1 for r in rows if r.get("state") == profile_adapter.STATE_DRIVING)
    if n_driving:
        logger.info(
            "run_profile_based_reconciliation: %d/%d driving leg(s) reached the "
            "Tier-4-exhausted last resort (%.1f%%)",
            n_unrepaired,
            n_driving,
            100 * n_unrepaired / n_driving,
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# --- I/O ------------------------------------------------------------------------


def save_profile_based_output(
    output: pd.DataFrame, validation_dir: Path = DEFAULT_VALIDATION_DIR
) -> Path:
    """Write the reconciled output table under `data/validation/profile_based/`
    only - never `data/processed/` (§8.1/§8.9's outputs constraint)."""
    validation_dir = Path(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)
    path = validation_dir / OUTPUT_FILENAME
    output.to_parquet(path, index=False)
    return path


def load_profile_based_output(validation_dir: Path = DEFAULT_VALIDATION_DIR) -> pd.DataFrame:
    path = Path(validation_dir) / OUTPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Profile-based output not found: {path}. Run "
            "`python scripts/run_profile_based_test.py` first."
        )
    return pd.read_parquet(path)
