"""Estimate workplace EV charging demand from synthetic driving activity
profiles.

Layers a *scenario* model (EV adoption, vehicle efficiency, charger power,
unmanaged-immediate charging) on top of the already-finalized, NHTS-validated
mobility outputs (`data/processed/synthetic_employees.parquet`,
`data/processed/synthetic_activity.parquet`). This module only reads those
two files - `generator/sample.py` and `generator/activity.py` are not
imported or modified here, and nothing in this module ever writes back to
either input path. Methodology: `docs/charging_demand_plan.md`.

## Unit note (the same gotcha `generator/time_utils.py` documents)

`arrival_time`/`departure_time` on `synthetic_activity.parquet` are
HHMM-encoded, not minutes-since-midnight, and must be converted with
`hhmm_to_minutes` before any interval-bucketing or duration arithmetic.
`workplace_dwell_minutes`, `duration`, and `dwell_time_after` are already
true minutes and must NOT be passed through that conversion - doing so would
silently corrupt every dwell/duration value in this module (e.g. treating a
250-minute dwell as "2:50" and decoding it as 170 minutes).

## Driving eligibility vs. mileage availability

A synthetic employee's `used_household_vehicle` (whether they represent
driving behavior at all) is independent of whether `total_daily_miles` is
populated (whether we can quantify how far they drove today) - the plan's
§5 finding is that `total_daily_miles`'s ~52% null rate is unrelated to
non-driving status, so conflating the two would wrongly treat over half the
driving population as non-drivers. `assign_evs` determines eligibility from
`used_household_vehicle`/`is_worker`/workplace-visit presence only;
`create_charging_sessions` separately determines mileage usability.

## EV ownership is a scenario draw, not a data fact

`vehicle_type`/`vehicle_fuel` on `synthetic_activity.parquet` are per-leg
NHTS donor codes, not stable per employee (plan §2) - they describe whichever
real donor a leg was rescaled from, not "this employee's vehicle". EV
ownership is instead drawn independently per scenario, reproducibly, from
`ev_adoption_rate` and `random_seed` (`assign_evs`).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from driving_profiles.generator import sample as sample_module
from driving_profiles.generator.activity import ACTIVITY_TABLE_FILENAME
from driving_profiles.generator.time_utils import MINUTES_PER_DAY, hhmm_to_minutes
from driving_profiles.utils import random_seed

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")
SESSIONS_FILENAME = "ev_charging_sessions.parquet"
LOAD_PROFILE_FILENAME = "ev_charging_load_profile.parquet"
SUMMARY_FILENAME = "ev_charging_summary.csv"

INTERVALS_PER_DAY = MINUTES_PER_DAY // 15  # 96, fixed per plan §8


@dataclass(frozen=True)
class ChargingScenarioConfig:
    """A named bundle of scenario assumptions (plan §4). Deliberately not
    hard-coded as scattered module constants - every value a caller might
    reasonably want to vary (adoption rate, vehicle efficiency, charger
    power, charging efficiency) lives here so a new scenario is a new
    instance of this dataclass, not a code change.

    `ev_adoption_rate` is a single project-wide float for this MVP;
    cluster-specific rates (a `dict[cluster_id, rate]`) are a documented
    future extension (plan §5), not implemented here.
    """

    scenario_name: str = "baseline_unmanaged"
    ev_adoption_rate: float = 0.20
    vehicle_efficiency_kwh_per_mile: float = 0.30
    charging_efficiency: float = 0.90
    charger_power_kw: float = 7.2
    interval_minutes: int = 15
    random_seed: int | None = None
    # Baseline MVP assumption (plan §3): a workplace-arrival leg with no
    # closing leg (the day's chain simply ends at work) is treated as
    # parked through end-of-day, clamped one minute short of midnight so it
    # never spills into a 97th interval.
    open_ended_workplace_departure_minutes: float = 1439.0


# --- Inputs -------------------------------------------------------------------


def load_charging_inputs(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two finalized mobility outputs this stage depends on,
    failing clearly (not with a raw pandas traceback) if either is missing.

    Read-only: nothing in this module ever writes to either path.
    """
    processed_dir = Path(processed_dir)
    employees_path = processed_dir / sample_module.SYNTHETIC_EMPLOYEE_FILENAME
    activity_path = processed_dir / ACTIVITY_TABLE_FILENAME

    if not employees_path.exists():
        raise FileNotFoundError(
            f"Synthetic employee table not found: {employees_path}. Run "
            "`python -m driving_profiles.generator.sample` first."
        )
    if not activity_path.exists():
        raise FileNotFoundError(
            f"Synthetic activity table not found: {activity_path}. Run "
            "`python -m driving_profiles.generator.activity` first."
        )

    employees = pd.read_parquet(employees_path)
    activity = pd.read_parquet(activity_path)
    return employees, activity


# --- Workplace windows (plan §3) ----------------------------------------------


def build_workplace_windows(activity: pd.DataFrame, config: ChargingScenarioConfig) -> pd.DataFrame:
    """Extract one row per workplace visit from `is_workplace_arrival == True`
    activity legs, converting `arrival_time` from HHMM to true minutes
    (`hhmm_to_minutes`) before any arithmetic - `workplace_dwell_minutes` is
    already true minutes and is used as-is, never HHMM-converted.

    `departure_time_minutes` is `arrival_time_minutes + workplace_dwell_minutes`
    when the dwell is known, or `config.open_ended_workplace_departure_minutes`
    directly (flagged `open_ended_window`) when `workplace_dwell_minutes` is
    null because the workplace-arrival leg is that employee's last leg of the
    day (plan §3). Every departure is then clamped to
    `open_ended_workplace_departure_minutes` so no window spills past the
    96-interval day.

    Windows with an invalid (non-finite) arrival time, a non-positive dwell,
    or `departure <= arrival` are dropped. Remaining windows are sorted by
    employee then arrival time and numbered `workplace_visit_number` (1-indexed,
    per employee).
    """
    visits = activity.loc[activity["is_workplace_arrival"]].copy()

    visits["arrival_time_minutes"] = visits["arrival_time"].apply(hhmm_to_minutes)
    valid_arrival = np.isfinite(visits["arrival_time_minutes"].to_numpy(dtype=float))

    dwell = visits["workplace_dwell_minutes"]
    is_open_ended = dwell.isna()
    departure_time_minutes = visits["arrival_time_minutes"] + dwell
    departure_time_minutes = departure_time_minutes.where(
        ~is_open_ended, config.open_ended_workplace_departure_minutes
    )
    departure_time_minutes = departure_time_minutes.clip(
        upper=config.open_ended_workplace_departure_minutes
    )

    visits["departure_time_minutes"] = departure_time_minutes
    visits["open_ended_window"] = is_open_ended
    visits["available_dwell_minutes"] = (
        visits["departure_time_minutes"] - visits["arrival_time_minutes"]
    )

    valid = (
        valid_arrival
        & (visits["departure_time_minutes"] > visits["arrival_time_minutes"])
        & (visits["available_dwell_minutes"] > 0)
    )
    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.info(
            "build_workplace_windows: dropping %d invalid/zero-dwell workplace window(s)",
            n_dropped,
        )
    windows = visits.loc[valid].sort_values(
        ["synthetic_employee_id", "arrival_time_minutes"], kind="mergesort"
    )
    windows["workplace_visit_number"] = windows.groupby("synthetic_employee_id").cumcount() + 1

    columns = [
        "synthetic_employee_id",
        "workplace_visit_number",
        "arrival_time_minutes",
        "departure_time_minutes",
        "available_dwell_minutes",
        "open_ended_window",
    ]
    return windows[columns].reset_index(drop=True)


# --- EV eligibility and assignment (plan §5) -----------------------------------


def assign_evs(
    employees: pd.DataFrame, windows: pd.DataFrame, config: ChargingScenarioConfig
) -> pd.DataFrame:
    """Classify every synthetic employee's charging eligibility and draw EV
    ownership, reproducibly, among the charging-eligible pool.

    `driving_eligible` is `is_worker == True and used_household_vehicle ==
    True` - not `total_daily_miles.notna()` (see module docstring).
    `has_workplace_window` is whether the employee has >=1 row in `windows`
    (already-validated visits, `build_workplace_windows`). `charging_eligible`
    is the AND of both - the pool EV ownership is drawn from, since no
    vehicle presence at the workplace means no charging opportunity
    regardless of EV ownership (plan §5).

    EV ownership is assigned as an exact, deterministic count -
    `round(len(charging_eligible) * ev_adoption_rate)` - drawn without
    replacement from the charging-eligible pool sorted by
    `synthetic_employee_id` (so the draw doesn't depend on incidental row
    order), using `random_seed.get_rng(config.random_seed)`. The same seed
    always reproduces the same EV set; a different seed produces a different
    one. Donor `vehicle_type`/`vehicle_fuel` are never consulted.

    Returns one row per employee in `employees` (not just the eligible ones,
    so every category - not-eligible, eligible-no-window, eligible-EV,
    eligible-non-EV - is explicitly represented) with `cluster_id` and
    `total_daily_miles` carried through for downstream use.
    """
    driving_eligible = employees["is_worker"].fillna(False) & employees[
        "used_household_vehicle"
    ].fillna(False)
    has_window = employees["synthetic_employee_id"].isin(
        set(windows["synthetic_employee_id"])
    )
    charging_eligible = driving_eligible & has_window

    eligibility = pd.DataFrame(
        {
            "synthetic_employee_id": employees["synthetic_employee_id"],
            "cluster_id": employees["cluster_id"],
            "total_daily_miles": employees["total_daily_miles"],
            "driving_eligible": driving_eligible.to_numpy(dtype=bool),
            "has_workplace_window": has_window.to_numpy(dtype=bool),
            "charging_eligible": charging_eligible.to_numpy(dtype=bool),
        }
    )

    eligible_ids = np.sort(
        eligibility.loc[eligibility["charging_eligible"], "synthetic_employee_id"].to_numpy()
    )
    n_ev = int(round(len(eligible_ids) * config.ev_adoption_rate))
    rng = random_seed.get_rng(config.random_seed)
    ev_ids = (
        set(rng.choice(eligible_ids, size=n_ev, replace=False)) if n_ev > 0 else set()
    )
    eligibility["ev_assigned"] = eligibility["synthetic_employee_id"].isin(ev_ids)

    logger.info(
        "assign_evs: %d/%d driving-eligible, %d/%d charging-eligible, %d EV(s) assigned "
        "(target rate %.3f)",
        int(driving_eligible.sum()),
        len(employees),
        len(eligible_ids),
        len(employees),
        len(ev_ids),
        config.ev_adoption_rate,
    )
    return eligibility


# --- Mileage/energy + charging sessions (plan §6-7) ----------------------------


def create_charging_sessions(
    windows: pd.DataFrame, eligibility: pd.DataFrame, config: ChargingScenarioConfig
) -> pd.DataFrame:
    """Compute each EV employee's daily requested energy and allocate it
    across their workplace visits in chronological order, one output row per
    workplace visit used for charging allocation.

    Only `ev_assigned` employees with usable mileage (`total_daily_miles`
    finite and `>= 0`) produce rows; employees with missing or unusable
    mileage are excluded from charging demand entirely (plan §6) and are not
    silently imputed. A `total_daily_miles == 0` employee IS usable (a real,
    informative "EV owner who didn't drive today") and always gets exactly
    one zero-valued row for their earliest visit, per plan §6 - not zero
    rows - even though its own allocation loop would otherwise never find
    positive remaining energy to allocate.

    `employee_requested_energy_kwh` is the whole-day total (constant across
    an employee's rows); `visit_requested_energy_kwh` is the balance owed
    *going into* that specific visit (starts at the employee total, then
    decreases visit over visit); `remaining_energy_after_visit_kwh` is the
    balance carried to the next visit; `employee_unmet_energy_kwh` is the
    final balance after the employee's last *processed* visit, broadcast
    onto every row for that employee so summing it per unique employee
    (never per row) gives the correct total (plan's double-counting
    warning).

    Allocation loop, per employee, in `workplace_visit_number` order: the
    first visit is always processed (so a zero-mileage employee still gets
    one row); every subsequent visit is only processed while the running
    balance is still positive going in (plan: "continue to later visits only
    while remaining energy is positive") - once it reaches zero, later
    visits that day are not part of the allocation and get no row.
    """
    ev = eligibility.loc[eligibility["ev_assigned"]].copy()
    finite_miles = np.isfinite(ev["total_daily_miles"].to_numpy(dtype=float))
    ev["mileage_usable"] = finite_miles & (ev["total_daily_miles"] >= 0)

    traction = ev["total_daily_miles"] * config.vehicle_efficiency_kwh_per_mile
    ev["employee_requested_energy_kwh"] = (traction / config.charging_efficiency).where(
        ev["mileage_usable"]
    )

    usable_columns = [
        "synthetic_employee_id",
        "cluster_id",
        "total_daily_miles",
        "employee_requested_energy_kwh",
    ]
    usable = ev.loc[ev["mileage_usable"], usable_columns]
    n_excluded = int((~ev["mileage_usable"]).sum())
    if n_excluded:
        logger.info(
            "create_charging_sessions: excluding %d EV-assigned employee(s) with missing/"
            "unusable mileage from charging demand",
            n_excluded,
        )
    if usable.empty:
        return _empty_sessions_frame()

    employee_windows = windows.merge(usable, on="synthetic_employee_id", how="inner")
    employee_windows = employee_windows.sort_values(
        ["synthetic_employee_id", "workplace_visit_number"], kind="mergesort"
    )

    rows: list[dict] = []
    for employee_id, group in employee_windows.groupby("synthetic_employee_id", sort=False):
        employee_requested = float(group["employee_requested_energy_kwh"].iloc[0])
        cluster_id = group["cluster_id"].iloc[0]
        miles_to_replenish = float(group["total_daily_miles"].iloc[0])

        remaining = employee_requested
        emitted: list[dict] = []
        for i, (_, visit) in enumerate(group.iterrows()):
            if i > 0 and remaining <= 0:
                break

            dwell_minutes = float(visit["available_dwell_minutes"])
            maximum_deliverable_kwh = config.charger_power_kw * dwell_minutes / 60.0
            visit_requested = remaining
            delivered = min(remaining, maximum_deliverable_kwh)
            duration = (
                delivered / config.charger_power_kw * 60.0
                if config.charger_power_kw > 0
                else 0.0
            )
            arrival = float(visit["arrival_time_minutes"])
            remaining_after = remaining - delivered

            emitted.append(
                {
                    "synthetic_employee_id": employee_id,
                    "cluster_id": cluster_id,
                    "workplace_visit_number": int(visit["workplace_visit_number"]),
                    "arrival_time_minutes": arrival,
                    "departure_time_minutes": float(visit["departure_time_minutes"]),
                    "available_dwell_minutes": dwell_minutes,
                    "open_ended_window": bool(visit["open_ended_window"]),
                    "miles_to_replenish": miles_to_replenish,
                    "employee_requested_energy_kwh": employee_requested,
                    "visit_requested_energy_kwh": visit_requested,
                    "delivered_energy_kwh": delivered,
                    "remaining_energy_after_visit_kwh": remaining_after,
                    "charger_power_kw": config.charger_power_kw,
                    "charging_start_minutes": arrival,
                    "charging_end_minutes": arrival + duration,
                    "charging_duration_minutes": duration,
                    "scenario_name": config.scenario_name,
                    "ev_adoption_rate": config.ev_adoption_rate,
                }
            )
            remaining = remaining_after

        final_unmet = remaining
        for row in emitted:
            row["employee_unmet_energy_kwh"] = final_unmet
        rows.extend(emitted)

    sessions = pd.DataFrame(rows)
    return sessions[_SESSION_COLUMNS].reset_index(drop=True)


_SESSION_COLUMNS = [
    "synthetic_employee_id",
    "cluster_id",
    "workplace_visit_number",
    "arrival_time_minutes",
    "departure_time_minutes",
    "available_dwell_minutes",
    "open_ended_window",
    "miles_to_replenish",
    "employee_requested_energy_kwh",
    "visit_requested_energy_kwh",
    "delivered_energy_kwh",
    "remaining_energy_after_visit_kwh",
    "employee_unmet_energy_kwh",
    "charger_power_kw",
    "charging_start_minutes",
    "charging_end_minutes",
    "charging_duration_minutes",
    "scenario_name",
    "ev_adoption_rate",
]


def _empty_sessions_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_SESSION_COLUMNS)


# --- 15-minute load profile (plan §8) ------------------------------------------


def build_load_profile(sessions: pd.DataFrame, config: ChargingScenarioConfig) -> pd.DataFrame:
    """Build the fixed 96-row (24h / `interval_minutes`) aggregate workplace
    load profile from per-session dwell and charging windows.

    `connected_ev_count` counts sessions whose *dwell* window
    (`arrival_time_minutes` -> `departure_time_minutes`) overlaps an
    interval - present at the workplace, whether or not still drawing power.
    `charging_ev_count` counts sessions whose *charging* window
    (`charging_start_minutes` -> `charging_end_minutes`) overlaps - actively
    drawing power. A zero-duration charging window (a zero-mileage EV
    employee) never overlaps any interval, so it contributes to
    `connected_ev_count` but never `charging_ev_count`/energy - correct,
    since it never actually charges.

    `interval_energy_kwh` sums `charger_power_kw * overlap_minutes / 60`
    across sessions per interval (so summing it across all 96 intervals
    reconstructs `sum(delivered_energy_kwh)` exactly, within floating-point
    tolerance); `charging_power_kw` sums
    `charger_power_kw * overlap_minutes / interval_minutes` - the fraction of
    the interval actively charged, consistent with `interval_energy_kwh` via
    energy = power * time.
    """
    n_intervals = MINUTES_PER_DAY // config.interval_minutes
    interval_start = np.arange(n_intervals) * config.interval_minutes
    interval_end = interval_start + config.interval_minutes

    if sessions.empty:
        connected = np.zeros(n_intervals, dtype=int)
        charging = np.zeros(n_intervals, dtype=int)
        interval_energy = np.zeros(n_intervals, dtype=float)
        charging_power = np.zeros(n_intervals, dtype=float)
    else:
        arr = sessions["arrival_time_minutes"].to_numpy(dtype=float)[:, None]
        dep = sessions["departure_time_minutes"].to_numpy(dtype=float)[:, None]
        cs = sessions["charging_start_minutes"].to_numpy(dtype=float)[:, None]
        ce = sessions["charging_end_minutes"].to_numpy(dtype=float)[:, None]
        power = sessions["charger_power_kw"].to_numpy(dtype=float)[:, None]

        i_start = interval_start[None, :]
        i_end = interval_end[None, :]

        dwell_overlap = np.clip(np.minimum(dep, i_end) - np.maximum(arr, i_start), 0, None)
        charge_overlap = np.clip(np.minimum(ce, i_end) - np.maximum(cs, i_start), 0, None)

        connected = (dwell_overlap > 0).sum(axis=0)
        charging = (charge_overlap > 0).sum(axis=0)
        interval_energy = (power * charge_overlap / 60.0).sum(axis=0)
        charging_power = (power * charge_overlap / config.interval_minutes).sum(axis=0)

    profile = pd.DataFrame(
        {
            "interval_index": np.arange(n_intervals),
            "interval_start_minutes": interval_start,
            "interval_end_minutes": interval_end,
            "connected_ev_count": connected,
            "charging_ev_count": charging,
            "charging_power_kw": charging_power,
            "interval_energy_kwh": interval_energy,
        }
    )
    profile["cumulative_energy_kwh"] = profile["interval_energy_kwh"].cumsum()
    profile["scenario_name"] = config.scenario_name
    return profile


# --- Summary (plan §9) ---------------------------------------------------------


def summarize_charging_scenario(
    employees: pd.DataFrame,
    eligibility: pd.DataFrame,
    windows: pd.DataFrame,
    sessions: pd.DataFrame,
    load_profile: pd.DataFrame,
    config: ChargingScenarioConfig,
) -> pd.DataFrame:
    """One-row scenario summary. Requested/unmet energy are aggregated by
    first taking one value per unique employee (both are constant across an
    employee's session rows) before summing, so a multi-visit employee's
    energy isn't double-counted (plan's explicit warning); delivered energy
    sums directly since it's disjoint per visit by construction.
    """
    total_employees = len(employees)
    driving_eligible = int(eligibility["driving_eligible"].sum())
    ev_assigned = int(eligibility["ev_assigned"].sum())

    finite_miles = np.isfinite(eligibility["total_daily_miles"].to_numpy(dtype=float))
    mileage_usable = finite_miles & (eligibility["total_daily_miles"] >= 0)
    ev_usable_mileage = int((eligibility["ev_assigned"] & mileage_usable).sum())
    ev_excluded_mileage = ev_assigned - ev_usable_mileage

    if sessions.empty:
        per_employee = sessions
        total_requested = 0.0
        total_unmet = 0.0
        avg_session_energy = 0.0
        avg_duration = 0.0
    else:
        per_employee = sessions.drop_duplicates("synthetic_employee_id")
        total_requested = float(per_employee["employee_requested_energy_kwh"].sum())
        total_unmet = float(per_employee["employee_unmet_energy_kwh"].sum())
        avg_session_energy = float(sessions["delivered_energy_kwh"].mean())
        avg_duration = float(sessions["charging_duration_minutes"].mean())

    total_delivered = float(sessions["delivered_energy_kwh"].sum()) if len(sessions) else 0.0
    percent_delivered = (
        (100.0 * total_delivered / total_requested) if total_requested > 0 else 0.0
    )

    has_profile = len(load_profile) > 0
    peak_idx = int(load_profile["charging_power_kw"].to_numpy().argmax()) if has_profile else 0
    peak_power = float(load_profile["charging_power_kw"].iloc[peak_idx]) if has_profile else 0.0
    peak_start = (
        int(load_profile["interval_start_minutes"].iloc[peak_idx]) if has_profile else 0
    )

    summary = pd.DataFrame(
        [
            {
                "scenario_name": config.scenario_name,
                "total_employees": total_employees,
                "employees_with_workplace_window": int(
                    windows["synthetic_employee_id"].nunique()
                ),
                "driving_eligible_employees": driving_eligible,
                "not_driving_eligible_employees": total_employees - driving_eligible,
                "ev_assigned_employees": ev_assigned,
                "ev_adoption_rate": config.ev_adoption_rate,
                "ev_employees_with_usable_mileage": ev_usable_mileage,
                "employees_excluded_missing_or_unusable_mileage": ev_excluded_mileage,
                "open_ended_workplace_windows": int(windows["open_ended_window"].sum()),
                "total_requested_energy_kwh": total_requested,
                "total_delivered_energy_kwh": total_delivered,
                "total_unmet_energy_kwh": total_unmet,
                "percent_energy_delivered": percent_delivered,
                "peak_charging_power_kw": peak_power,
                "peak_interval_start_minutes": peak_start,
                "max_connected_evs": int(load_profile["connected_ev_count"].max())
                if len(load_profile)
                else 0,
                "max_simultaneous_charging_evs": int(load_profile["charging_ev_count"].max())
                if len(load_profile)
                else 0,
                "average_session_energy_kwh": avg_session_energy,
                "average_charging_duration_minutes": avg_duration,
            }
        ]
    )
    return summary


# --- Orchestration + I/O --------------------------------------------------------


def run_charging_scenario(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    config: ChargingScenarioConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full charging-demand pipeline end to end: load inputs, build
    workplace windows, assign EVs, allocate charging sessions, build the
    load profile, and summarize. Returns `(sessions, load_profile, summary)`.
    """
    config = config or ChargingScenarioConfig()
    employees, activity = load_charging_inputs(processed_dir)

    windows = build_workplace_windows(activity, config)
    eligibility = assign_evs(employees, windows, config)
    sessions = create_charging_sessions(windows, eligibility, config)
    load_profile = build_load_profile(sessions, config)
    summary = summarize_charging_scenario(
        employees, eligibility, windows, sessions, load_profile, config
    )
    return sessions, load_profile, summary


def save_charging_outputs(
    sessions: pd.DataFrame,
    load_profile: pd.DataFrame,
    summary: pd.DataFrame,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> tuple[Path, Path, Path]:
    """Write the three scenario outputs to `processed_dir`."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sessions_path = processed_dir / SESSIONS_FILENAME
    load_profile_path = processed_dir / LOAD_PROFILE_FILENAME
    summary_path = processed_dir / SUMMARY_FILENAME

    sessions.to_parquet(sessions_path, index=False)
    load_profile.to_parquet(load_profile_path, index=False)
    summary.to_csv(summary_path, index=False)
    return sessions_path, load_profile_path, summary_path


def _config_from_args(args: argparse.Namespace) -> ChargingScenarioConfig:
    config = ChargingScenarioConfig(random_seed=args.seed)
    overrides = {}
    if args.ev_adoption_rate is not None:
        overrides["ev_adoption_rate"] = args.ev_adoption_rate
    if args.charger_power_kw is not None:
        overrides["charger_power_kw"] = args.charger_power_kw
    if args.vehicle_efficiency_kwh_per_mile is not None:
        overrides["vehicle_efficiency_kwh_per_mile"] = args.vehicle_efficiency_kwh_per_mile
    if args.charging_efficiency is not None:
        overrides["charging_efficiency"] = args.charging_efficiency
    return replace(config, **overrides) if overrides else config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Estimate workplace EV charging demand from synthetic activity profiles."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ev-adoption-rate", type=float, default=None)
    parser.add_argument("--charger-power-kw", type=float, default=None)
    parser.add_argument("--vehicle-efficiency-kwh-per-mile", type=float, default=None)
    parser.add_argument("--charging-efficiency", type=float, default=None)
    args = parser.parse_args()

    scenario_config = _config_from_args(args)
    sessions, load_profile, summary = run_charging_scenario(args.processed_dir, scenario_config)
    paths = save_charging_outputs(sessions, load_profile, summary, args.processed_dir)

    logger.info(
        "Wrote %d charging session row(s), %d load-profile interval(s), and a summary to %s",
        len(sessions),
        len(load_profile),
        ", ".join(str(p) for p in paths),
    )
