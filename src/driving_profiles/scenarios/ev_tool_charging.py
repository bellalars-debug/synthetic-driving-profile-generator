"""Workplace charging estimation via the ev-infrastructure-tool station/queue
simulator (the final pipeline stage).

This is an *alternative* charging backend to `scenarios/charging_demand.py`.
Where `charging_demand.py` applies a scenario energy model (adoption, efficiency,
unmanaged-immediate delivery) directly on the activity output, this module hands
the same synthetic activity profiles to an external, physics-of-queueing
simulator — Rongxin Yin's ev-infrastructure-tool (MIT, vendored under
`third_party/ev_infrastructure_tool/`) — which models discrete L2/L3 charging
stations, a first-come queue, station contention, and waiting time. Use it when
you need station counts / utilization / waiting behavior rather than an
idealized energy total.

Data flow (the "add the ev-tool to the end of our pipeline" step):

    data/processed/synthetic_activity.parquet   (this pipeline's output)
    data/processed/synthetic_employees.parquet
        -> reconstruct each employee's day as an ev-tool activity timeline
        -> pov_driving_pattern.json  (the ev-tool's exact input schema)
        -> ev-tool run_charging_management(...)          [unchanged, vendored]
        -> charging-estimate CSV + summary

Reads only the two finalized parquet artifacts; never writes back to them.

Schema note (mirrors `generator/activity.py` OUTPUT_COLUMNS and
`generator/time_utils.py`): `departure_time`/`arrival_time` are HHMM-encoded
and are converted with `hhmm_to_minutes` before any timeline arithmetic;
`duration`, `dwell_time_after`, `workplace_dwell_minutes` are already true
minutes and must not pass through that conversion.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# time helper is shared with the generator (HHMM <-> minutes)
try:
    from driving_profiles.generator.time_utils import hhmm_to_minutes
except Exception:                                            # pragma: no cover
    def hhmm_to_minutes(hhmm: float) -> float:
        if pd.isna(hhmm):
            return float("nan")
        h, m = divmod(float(hhmm), 100)
        return h * 60 + m

REPO_ROOT = Path(__file__).resolve().parents[3]
EV_TOOL_BACKEND = REPO_ROOT / "third_party" / "ev_infrastructure_tool" / "charging_backend"

KWH_PER_GAL = 33.7          # gasoline-gallon-equivalent, matches the ev-tool
DEFAULT_BATTERY_KWH = 75    # candidate-EV battery; the sim picks EVs by adoption rate
LIGHT_DUTY_MPGE = (117.0, 3.2)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORKDAYS = DAYS[:5]

# activity `trip_purpose` (destination) -> ev-tool parking location vocabulary
PURPOSE_TO_LOCATION = {"home": "Home", "work": "On-Site", "other": "Off-Site"}


def _hhmm_str(minute: float) -> str:
    m = int(round(min(max(float(minute), 0.0), 24 * 60 - 1)))
    return f"{m // 60:02d}:{m % 60:02d}"


def _day_activities(legs: pd.DataFrame) -> tuple[list, float, float]:
    """Reconstruct one synthetic employee's day (ordered legs) into contiguous
    ev-tool activities covering 00:00->23:59. Returns (activities, total_miles,
    total_drive_minutes)."""
    legs = legs.sort_values("trip_number")
    acts, aid, cursor = [], 1, 0
    total_miles = 0.0
    total_drive_min = 0.0

    def add(state, start, end, location, dist):
        nonlocal aid, cursor
        end = min(end, 24 * 60 - 1)
        if end <= start:
            return
        acts.append({"activity_id": aid, "start_time": _hhmm_str(start),
                     "end_time": _hhmm_str(end), "activity_type": state,
                     "location": location,
                     "driving_distance": round(float(dist), 4) if state == "Driving" else 0})
        aid += 1
        cursor = end

    rows = list(legs.itertuples(index=False))
    if not rows:
        return ([{"activity_id": 1, "start_time": "00:00", "end_time": "23:59",
                  "activity_type": "Parked", "location": "Home", "driving_distance": 0}],
                0.0, 0.0)

    first_dep = hhmm_to_minutes(rows[0].departure_time)
    add("Parked", 0, first_dep, "Home", 0)
    for i, leg in enumerate(rows):
        dep = hhmm_to_minutes(leg.departure_time)
        arr = hhmm_to_minutes(leg.arrival_time)
        dist = float(leg.distance) if pd.notna(leg.distance) else 0.0
        if arr <= dep:                                       # guard degenerate leg
            arr = dep + max(float(getattr(leg, "duration", 1) or 1), 1)
        add("Driving", max(dep, cursor), arr, "Off-Site", dist)
        total_miles += dist
        total_drive_min += (arr - dep)
        # parked at destination until the next leg departs (or end of day)
        dest_loc = PURPOSE_TO_LOCATION.get(str(leg.trip_purpose), "Off-Site")
        park_end = hhmm_to_minutes(rows[i + 1].departure_time) if i + 1 < len(rows) else 24 * 60 - 1
        add("Parked", cursor, park_end, dest_loc, 0)
    if acts:
        acts[-1]["end_time"] = "23:59"
    return acts, total_miles, total_drive_min


def build_patterns(activity: pd.DataFrame, employees: pd.DataFrame | None,
                   site_id: str = "bldg-90", days_drive: int = 5,
                   seed: int = 42) -> list[dict]:
    """Convert the pipeline's activity output into ev-tool pov_driving_pattern
    entries (one per synthetic employee)."""
    rng = np.random.default_rng(seed)
    emp_lookup = {}
    if employees is not None and "synthetic_employee_id" in employees.columns:
        emp_lookup = employees.set_index("synthetic_employee_id").to_dict("index")

    patterns = []
    for eid, legs in activity.groupby("synthetic_employee_id", sort=True):
        day_acts, total_miles, drive_min = _day_activities(legs)
        # workplace commute distance/time for the ranking + filter fields
        work_leg = legs[legs.get("is_workplace_arrival", False) == True]  # noqa: E712
        one_way_mi = float(work_leg["distance"].iloc[0]) if len(work_leg) else \
            (total_miles / 2 if total_miles else 1.0)
        emp = emp_lookup.get(eid, {})
        first = legs.sort_values("trip_number").iloc[0]
        depart_min = hhmm_to_minutes(first["departure_time"])
        return_min = hhmm_to_minutes(legs.sort_values("trip_number")["departure_time"].iloc[-1])

        mpge = float(rng.normal(*LIGHT_DUTY_MPGE))
        speed_mph = float(np.clip((total_miles / (drive_min / 60.0)) if drive_min > 0 else 25.0, 5, 80))
        rate = speed_mph * KWH_PER_GAL / mpge
        kwh = total_miles * KWH_PER_GAL / mpge
        drive_days = set(rng.choice(WORKDAYS, size=min(days_drive, 5), replace=False))
        weekly = []
        for d in DAYS:
            if d in drive_days and total_miles > 0:
                weekly.append({"day of week": d, "driving distance": total_miles,
                               "equivalent efficiency mpge": mpge, "average speed mph": speed_mph,
                               "equivalent electricity consumption rate": rate,
                               "equivalent electricity kWh": kwh, "activities": day_acts})
            else:
                weekly.append({"day of week": d, "driving distance": 0,
                               "equivalent efficiency mpge": 0, "average speed mph": 0,
                               "equivalent electricity consumption rate": 0,
                               "equivalent electricity kWh": 0,
                               "activities": [{"activity_id": 1, "start_time": "00:00",
                                               "end_time": "23:59", "activity_type": "Parked",
                                               "location": "Home", "driving_distance": 0}]})

        patterns.append({
            "employee_id": str(eid), "commute_mode": "car",
            "onsite_bldg": site_id, "parking_lot": site_id, "parking lot": site_id,
            "zev site": site_id, "vehicle id": str(eid),
            "leave_to_work_time": _hhmm_str(depart_min),
            "return_home_time": _hhmm_str(return_min),
            "distance": {"text": f"{one_way_mi:.1f} mi", "value": int(round(one_way_mi * 1609))},
            "duration": {"text": "", "value": int(round(float(getattr(first, "duration", 0) or 0) * 60))},
            "status": "OK",
            "home_charging": "Level 2" if rng.random() < 0.7 else "Level 1",
            "days_drive": days_drive, "category": "Light-Duty",
            "battery capacity": DEFAULT_BATTERY_KWH,
            "cluster_id": emp.get("cluster_id"),
            "vehicle_fuel": str(first.get("vehicle_fuel")) if hasattr(first, "get") else None,
            "weekly driving patterns": weekly,
        })
    return patterns


def _load_ev_tool():
    """Import the vendored ev-tool charging simulator (hyphenated filename)."""
    sys.path.insert(0, str(EV_TOOL_BACKEND))                 # so utilities.* / models.* resolve
    path = EV_TOOL_BACKEND / "standalone-pov-charging-management.py"
    spec = importlib.util.spec_from_file_location("ev_charging_mgmt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def estimate_charging(
    activity_path: str | Path,
    employees_path: str | Path | None = None,
    out_dir: str | Path = "reports/xlsx/charging_demand/ev_tool",
    site_id: str = "bldg-90",
    adoption_rate: float = 0.36,
    run_period_days: int = 5,
    l2_max_rate_kw: float = 7.0,
    l3_max_rate_kw: float = 50.0,
    start_date: datetime.datetime | None = None,
    seed: int = 42,
) -> dict:
    """End stage: synthetic activity profiles -> ev-tool charging estimates.

    Returns a summary dict and writes `pov_driving_pattern.json`,
    `ev_tool_vehicle_status_{rate}.csv`, and `ev_tool_summary.json` to `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    activity = pd.read_parquet(activity_path)
    employees = pd.read_parquet(employees_path) if employees_path else None

    patterns = build_patterns(activity, employees, site_id=site_id,
                              days_drive=min(run_period_days, 5), seed=seed)
    pattern_path = out_dir / "pov_driving_pattern.json"
    pattern_path.write_text(json.dumps(patterns))

    ev_tool = _load_ev_tool()
    results = ev_tool.run_charging_management(
        site_id, str(pattern_path),
        start_time=start_date or datetime.datetime(2024, 2, 1),
        run_period=run_period_days, l2_max_rate=l2_max_rate_kw,
        l3_max_rate=l3_max_rate_kw, adoption_rate=adoption_rate)
    status_csv = out_dir / f"ev_tool_vehicle_status_{adoption_rate}.csv"
    results.to_csv(status_csv, index=False)

    charging = results[results["status"] == "Charging"]
    peak_by_stations = {}
    if len(charging):
        counts = charging.groupby(["L2", "time"]).size()
        peak_by_stations = {int(k): int(v) for k, v in counts.groupby("L2").max().items()}
    summary = {
        "site_id": site_id, "n_synthetic_vehicles": len(patterns),
        "adoption_rate": adoption_rate,
        "n_evs_selected": int(results["vehicle_id"].nunique()) if len(results) else 0,
        "l2_station_scenarios": sorted(int(x) for x in results["L2"].unique()) if len(results) else [],
        "peak_simultaneous_charging_by_L2_count": peak_by_stations,
        "run_period_days": run_period_days,
        "pattern_file": str(pattern_path), "status_file": str(status_csv),
        "backend": "ev-infrastructure-tool (Rongxin Yin, MIT) station/queue simulator",
    }
    (out_dir / "ev_tool_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
