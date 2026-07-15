"""Smoke demo for the ev-tool charging stage.

Builds a small fixture in the EXACT schema `generator/activity.py` and
`generator/sample.py` emit (`synthetic_activity.parquet` /
`synthetic_employees.parquet`), then runs the end stage
(`scenarios/ev_tool_charging.estimate_charging`) — proving the ev-tool
station/queue simulator consumes this pipeline's output and produces charging
estimates. Real runs use the pipeline's actual parquet output instead of this
fixture.

    python scripts/demo_ev_tool_charging.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from driving_profiles.scenarios import ev_tool_charging as ev  # noqa: E402

ACTIVITY_COLUMNS = [
    "synthetic_employee_id", "trip_number", "departure_time", "arrival_time",
    "trip_purpose", "distance", "duration", "dwell_time_after",
    "is_workplace_arrival", "is_workplace_departure", "workplace_dwell_minutes",
    "chain_source", "vehicle_type", "vehicle_fuel",
]


def _leg(eid, n, dep, arr, purpose, dist, dur, wa=False, wd=False, wdwell=0.0):
    return {"synthetic_employee_id": eid, "trip_number": n, "departure_time": float(dep),
            "arrival_time": float(arr), "trip_purpose": purpose, "distance": float(dist),
            "duration": float(dur), "dwell_time_after": 0.0, "is_workplace_arrival": wa,
            "is_workplace_departure": wd, "workplace_dwell_minutes": float(wdwell),
            "chain_source": "donor", "vehicle_type": "Car", "vehicle_fuel": "Gas"}


def make_fixture(n_employees: int = 60, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows, emps = [], []
    for i in range(1, n_employees + 1):
        eid = f"SE{i:04d}"
        dep = 700 + int(rng.integers(0, 90))          # 07:00-08:29 HHMM
        dep_h, dep_m = divmod(dep, 100)
        arr_m = dep_m + int(rng.integers(10, 35))     # commute minutes
        arrh, arrmin = divmod(dep_h * 60 + arr_m, 60)
        arr = arrh * 100 + arrmin
        D = float(round(rng.uniform(3, 30), 1))
        wdep = 1600 + int(rng.integers(0, 120))       # 16:00-17:59
        wdep_h, wdep_m = divmod(wdep, 100)
        chained = rng.random() < 0.35                 # ~a third chain a stop
        wdwell = (wdep_h * 60 + wdep_m) - (arrh * 60 + arrmin)
        if not chained:
            rows.append(_leg(eid, 1, dep, arr, "work", D, arr_m, wa=True, wdwell=wdwell))
            rh, rm = divmod(wdep_h * 60 + wdep_m + int(rng.integers(15, 35)), 60)
            rows.append(_leg(eid, 2, wdep, rh * 100 + rm, "home", D, 25, wd=True))
            nstops = 0
        else:
            # school drop -> work ; shopping -> home
            s_arr_m = dep_m + 12
            sh, sm = divmod(dep_h * 60 + s_arr_m, 60)
            rows.append(_leg(eid, 1, dep, sh * 100 + sm, "other", round(D * 0.4, 1), 12))
            w_dep = sh * 100 + (sm + 5 if sm + 5 < 60 else sm)
            wh, wm = divmod(sh * 60 + sm + 5 + 18, 60)
            rows.append(_leg(eid, 2, w_dep, wh * 100 + wm, "work", round(D * 0.6, 1), 18, wa=True, wdwell=wdwell))
            rows.append(_leg(eid, 3, wdep, wdep + 20, "other", 5, 20, wd=True))
            eh, em = divmod(wdep_h * 60 + wdep_m + 20 + 15, 60)
            rows.append(_leg(eid, 4, eh * 100 + em, eh * 100 + em + 15, "home", round(D * 0.5, 1), 15))
            nstops = 2
        emps.append({"synthetic_employee_id": eid, "cluster_id": int(rng.integers(0, 6)),
                     "trips_per_day": 2 + nstops, "number_of_stops": nstops,
                     "work_arrival_time": float(arr), "work_departure_time": float(wdep)})
    activity = pd.DataFrame(rows)[ACTIVITY_COLUMNS]
    employees = pd.DataFrame(emps)
    return activity, employees


def main():
    activity, employees = make_fixture()
    tmp = Path(tempfile.mkdtemp(prefix="ev_tool_demo_"))
    ap, ep = tmp / "synthetic_activity.parquet", tmp / "synthetic_employees.parquet"
    activity.to_parquet(ap, index=False)
    employees.to_parquet(ep, index=False)
    print(f"Fixture: {employees['synthetic_employee_id'].nunique()} employees, "
          f"{len(activity)} legs -> {tmp}")

    summary = ev.estimate_charging(ap, ep, out_dir=tmp / "out", site_id="bldg-90",
                                   adoption_rate=0.36, run_period_days=3)
    print("\n=== ev-tool charging estimate (summary) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
