"""Validation metrics for the profile-based experiment (`docs/profile_based_generation_plan.md`
§8.8): sequence preservation, workplace arrival/departure preservation,
schedule-adjustment magnitude, chronological validity, and implied-speed
plausibility.

Read-only: takes the external `DriverProfiles.csv` frame
(`profile_adapter.load_driver_profiles`'s output) and the reconciled output
table (`profile_based.run_profile_based_reconciliation`'s output) and
compares them - never regenerates or mutates either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from driving_profiles.generator import activity as activity_module
from driving_profiles.generator import profile_adapter
from driving_profiles.generator import profile_based as profile_based_module
from driving_profiles.validation import common

SECTION = "profile_based"

DEFAULT_ARRIVAL_DEPARTURE_TOLERANCES_MINUTES = (5, 15, 30)


def _id_map(user_ids: list[int]) -> dict[int, str]:
    return profile_based_module.assign_profile_employee_ids(user_ids)


def _original_sequence(driver_profiles: pd.DataFrame, user_id: int) -> list[tuple[str, str]]:
    rows = driver_profiles.loc[driver_profiles["user_id"] == user_id].sort_values("row_index")
    return list(zip(rows["state"], rows["location"]))


def _reconciled_sequence(output: pd.DataFrame, profile_employee_id: str) -> list[tuple[str, str]]:
    rows = output.loc[output["profile_employee_id"] == profile_employee_id].sort_values("row_index")
    return list(zip(rows["state"], rows["location"]))


def validate_sequence_preserved(
    driver_profiles: pd.DataFrame, output: pd.DataFrame, user_ids: list[int]
) -> pd.DataFrame:
    """§8.9 required test (c) / §8.8 "location sequences preserved exactly":
    byte-identical (state, location) sequence, per profile, between input
    and output. Also implies leg count and stop count are unchanged (items
    1-4 of the §8.1 preservation contract), since a changed count would
    change the sequence length.
    """
    id_map = _id_map(user_ids)
    n_violations = 0
    for uid in user_ids:
        original = _original_sequence(driver_profiles, uid)
        reconciled = _reconciled_sequence(output, id_map[uid])
        if original != reconciled:
            n_violations += 1
    return common.results_frame(
        [
            common.structural_result(
                SECTION,
                "location_sequence_preserved",
                n_violations=n_violations,
                n_checked=len(user_ids),
                threshold="100% byte-identical (state, location) sequence vs. DriverProfiles.csv",
            )
        ]
    )


def _workplace_boundary_preservation(
    driver_profiles: pd.DataFrame,
    output: pd.DataFrame,
    user_ids: list[int],
    boundary: str,
    tolerances_minutes: tuple[int, ...],
) -> pd.DataFrame:
    """Shared implementation for arrival (`boundary="end_hour"`, the arrival
    leg's own end) and departure (`boundary="start_hour"`, the departure
    leg's own start) preservation metrics (§8.8).
    """
    id_map = _id_map(user_ids)
    diffs_minutes = []
    for uid in user_ids:
        peid = id_map[uid]
        out_legs = output.loc[
            (output["profile_employee_id"] == peid)
            & (output["state"] == profile_adapter.STATE_DRIVING)
            & (output["is_arrival_at_work"] == (boundary == "end_hour"))
        ]
        orig_rows = driver_profiles.loc[driver_profiles["user_id"] == uid].set_index("row_index")
        for _, leg in out_legs.iterrows():
            row_index = int(leg["row_index"])
            original_value = orig_rows.loc[row_index, boundary] * 60.0
            reconciled_value = leg[boundary] * 60.0
            diffs_minutes.append(abs(reconciled_value - original_value))

    rows = []
    n_checked = len(diffs_minutes)
    diffs = np.array(diffs_minutes, dtype=float)
    boundary_label = "arrival" if boundary == "end_hour" else "departure"
    for tolerance in tolerances_minutes:
        n_within = int((diffs <= tolerance).sum()) if n_checked else 0
        pct = (n_within / n_checked * 100.0) if n_checked else float("nan")
        rows.append(
            common.result_row(
                SECTION,
                f"workplace_{boundary_label}_preserved_within_{tolerance}min",
                test="tolerance_share",
                statistic=pct,
                n_synthetic=n_checked,
                threshold=f"reported share within {tolerance} min",
                passed=None,
                detail=f"{n_within}/{n_checked} within {tolerance} min",
            )
        )
    return common.results_frame(rows)


def validate_workplace_arrival_preserved(
    driver_profiles: pd.DataFrame,
    output: pd.DataFrame,
    user_ids: list[int],
    tolerances_minutes: tuple[int, ...] = DEFAULT_ARRIVAL_DEPARTURE_TOLERANCES_MINUTES,
) -> pd.DataFrame:
    """§8.8: "% of workplace arrival times preserved within 5/15/30 minutes",
    computed across every work-occurrence leg (`w_1...w_k`) of every
    profile, comparing reconstructed vs. original `End time (hour)`.
    """
    return _workplace_boundary_preservation(
        driver_profiles, output, user_ids, "end_hour", tolerances_minutes
    )


def validate_workplace_departure_preserved(
    driver_profiles: pd.DataFrame,
    output: pd.DataFrame,
    user_ids: list[int],
    tolerances_minutes: tuple[int, ...] = DEFAULT_ARRIVAL_DEPARTURE_TOLERANCES_MINUTES,
) -> pd.DataFrame:
    """§8.8: "% of workplace departure times preserved within 5/15/30
    minutes", on the `Start time (hour)` of each `Work` window's departure
    leg (the leg whose `origin_purpose == "work"`).
    """
    id_map = _id_map(user_ids)
    diffs_minutes = []
    for uid in user_ids:
        peid = id_map[uid]
        out_legs = output.loc[
            (output["profile_employee_id"] == peid)
            & (output["state"] == profile_adapter.STATE_DRIVING)
            & (output["purpose_transition"].str.startswith("work->", na=False))
        ]
        orig_rows = driver_profiles.loc[driver_profiles["user_id"] == uid].set_index("row_index")
        for _, leg in out_legs.iterrows():
            row_index = int(leg["row_index"])
            original_value = orig_rows.loc[row_index, "start_hour"] * 60.0
            reconciled_value = leg["start_hour"] * 60.0
            diffs_minutes.append(abs(reconciled_value - original_value))

    rows = []
    n_checked = len(diffs_minutes)
    diffs = np.array(diffs_minutes, dtype=float)
    for tolerance in tolerances_minutes:
        n_within = int((diffs <= tolerance).sum()) if n_checked else 0
        pct = (n_within / n_checked * 100.0) if n_checked else float("nan")
        rows.append(
            common.result_row(
                SECTION,
                f"workplace_departure_preserved_within_{tolerance}min",
                test="tolerance_share",
                statistic=pct,
                n_synthetic=n_checked,
                threshold=f"reported share within {tolerance} min",
                passed=None,
                detail=f"{n_within}/{n_checked} within {tolerance} min",
            )
        )
    return common.results_frame(rows)


def validate_schedule_adjustment(output: pd.DataFrame) -> pd.DataFrame:
    """§8.8: mean/max `abs(adjustment_minutes)` across every reconstructed
    leg's one adjustable side, and the share of driving legs whose schedule
    required adjustment (`schedule_status == "adjusted"`).
    """
    driving = output.loc[output["state"] == profile_adapter.STATE_DRIVING]
    adjustment = driving["adjustment_minutes"].abs()
    rows = [
        common.result_row(
            SECTION,
            "mean_schedule_adjustment_minutes",
            test="descriptive",
            statistic=float(adjustment.mean()) if len(adjustment) else float("nan"),
            n_synthetic=len(adjustment),
            threshold="informational",
            passed=None,
        ),
        common.result_row(
            SECTION,
            "max_schedule_adjustment_minutes",
            test="descriptive",
            statistic=float(adjustment.max()) if len(adjustment) else float("nan"),
            n_synthetic=len(adjustment),
            threshold="informational",
            passed=None,
        ),
    ]
    is_adjusted = driving["schedule_status"] == profile_based_module.SCHEDULE_STATUS_ADJUSTED
    n_adjusted = int(is_adjusted.sum())
    rows.append(
        common.result_row(
            SECTION,
            "share_driving_legs_adjusted",
            test="proportion",
            statistic=(n_adjusted / len(driving) * 100.0) if len(driving) else float("nan"),
            n_synthetic=len(driving),
            threshold="informational",
            passed=None,
            detail=f"{n_adjusted}/{len(driving)} driving leg(s) required a schedule adjustment",
        )
    )
    return common.results_frame(rows)


def validate_chronological_validity(output: pd.DataFrame, user_ids: list[int]) -> pd.DataFrame:
    """§8.8: 100% of reconstructed timelines remain monotonic,
    non-overlapping, and span 0-24h - should be 100% by construction (§8.6
    only ever changes existing segment durations), measured anyway to catch
    a cascade-logic defect."""
    id_map = _id_map(user_ids)
    n_violations = 0
    for uid in user_ids:
        peid = id_map[uid]
        rows = output.loc[output["profile_employee_id"] == peid].sort_values("row_index")
        starts = rows["start_hour"].to_numpy()
        ends = rows["end_hour"].to_numpy()
        contiguous = np.allclose(ends[:-1], starts[1:], atol=1e-6) if len(rows) > 1 else True
        spans_full_day = np.isclose(starts[0], 0.0, atol=1e-6) and np.isclose(
            ends[-1], 24.0, atol=1e-6
        )
        monotonic = bool(np.all(ends >= starts))
        if not (contiguous and spans_full_day and monotonic):
            n_violations += 1
    return common.results_frame(
        [
            common.structural_result(
                SECTION,
                "chronological_validity",
                n_violations=n_violations,
                n_checked=len(user_ids),
                threshold="100% monotonic, non-overlapping, 0-24h span",
            )
        ]
    )


def validate_speed_plausibility(output: pd.DataFrame) -> pd.DataFrame:
    """§8.8: 100% of substituted legs' implied speed
    (`distance_mi / (duration_min / 60)`) falls in
    `[MIN_PLAUSIBLE_SPEED_MPH, MAX_PLAUSIBLE_SPEED_MPH]` - should be 100%
    by construction (the donor pool is pre-filtered to this band), measured
    rather than assumed.
    """
    is_donor_sourced = (
        output["distance_duration_source"] == profile_based_module.DISTANCE_DURATION_SOURCE_DONOR
    )
    substituted = output.loc[(output["state"] == profile_adapter.STATE_DRIVING) & is_donor_sourced]
    implied_speed = substituted["distance_mi"] / (substituted["duration_min"] / 60.0)
    violations = (
        (implied_speed < activity_module.MIN_PLAUSIBLE_SPEED_MPH)
        | (implied_speed > activity_module.MAX_PLAUSIBLE_SPEED_MPH)
    ).sum()
    detail = (
        f"{int(violations)}/{len(substituted)} substituted leg(s) outside the plausible-speed band"
    )
    return common.results_frame(
        [
            common.structural_result(
                SECTION,
                "implied_speed_plausible",
                n_violations=int(violations),
                n_checked=len(substituted),
                threshold=(
                    f"100% within [{activity_module.MIN_PLAUSIBLE_SPEED_MPH}, "
                    f"{activity_module.MAX_PLAUSIBLE_SPEED_MPH}] mph"
                ),
                detail=detail,
            )
        ]
    )


def validate_no_source_user_ids(output: pd.DataFrame, user_ids: list[int]) -> pd.DataFrame:
    """§8's requirement 8: no source `User ID` from `DriverProfiles.csv`
    appears in the output - `profile_employee_id` must always be the
    synthesized `PROF-XXXX` form, never a raw integer user id.
    """
    leaked = output["profile_employee_id"].astype(str).isin({str(uid) for uid in user_ids})
    return common.results_frame(
        [
            common.structural_result(
                SECTION,
                "no_source_user_ids_in_output",
                n_violations=int(leaked.sum()),
                n_checked=len(output),
                threshold="0 rows carrying a raw DriverProfiles.csv User ID",
            )
        ]
    )


def run_profile_based_validation(
    driver_profiles: pd.DataFrame, output: pd.DataFrame, user_ids: list[int]
) -> pd.DataFrame:
    """Run every §8.8 check and return one combined result table."""
    return pd.concat(
        [
            validate_sequence_preserved(driver_profiles, output, user_ids),
            validate_workplace_arrival_preserved(driver_profiles, output, user_ids),
            validate_workplace_departure_preserved(driver_profiles, output, user_ids),
            validate_schedule_adjustment(output),
            validate_chronological_validity(output, user_ids),
            validate_speed_plausibility(output),
            validate_no_source_user_ids(output, user_ids),
        ],
        ignore_index=True,
    )


def save_validation_report(
    results: pd.DataFrame, validation_dir: Path = profile_based_module.DEFAULT_VALIDATION_DIR
) -> Path:
    """Write the validation result table under `data/validation/profile_based/`."""
    validation_dir = Path(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)
    path = validation_dir / "profile_based_validation_report.csv"
    results.to_csv(path, index=False)
    return path
