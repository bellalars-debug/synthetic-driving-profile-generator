"""Smoke test for the ev-tool charging stage: a schema-accurate fixture must
flow through `estimate_charging` and yield a well-formed charging summary."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from driving_profiles.scenarios import ev_tool_charging as ev  # noqa: E402

ACTIVITY_COLUMNS = [
    "synthetic_employee_id", "trip_number", "departure_time", "arrival_time",
    "trip_purpose", "distance", "duration", "dwell_time_after",
    "is_workplace_arrival", "is_workplace_departure", "workplace_dwell_minutes",
    "chain_source", "vehicle_type", "vehicle_fuel",
]


def _fixture(n=40, seed=1):
    rng = np.random.default_rng(seed)
    rows, emps = [], []
    for i in range(1, n + 1):
        eid = f"SE{i:03d}"
        D = float(round(rng.uniform(4, 25), 1))
        rows.append({"synthetic_employee_id": eid, "trip_number": 1,
                     "departure_time": 730.0, "arrival_time": 800.0, "trip_purpose": "work",
                     "distance": D, "duration": 30.0, "dwell_time_after": 0.0,
                     "is_workplace_arrival": True, "is_workplace_departure": False,
                     "workplace_dwell_minutes": 510.0, "chain_source": "donor",
                     "vehicle_type": "Car", "vehicle_fuel": "Gas"})
        rows.append({"synthetic_employee_id": eid, "trip_number": 2,
                     "departure_time": 1700.0, "arrival_time": 1730.0, "trip_purpose": "home",
                     "distance": D, "duration": 30.0, "dwell_time_after": 0.0,
                     "is_workplace_arrival": False, "is_workplace_departure": True,
                     "workplace_dwell_minutes": 0.0, "chain_source": "donor",
                     "vehicle_type": "Car", "vehicle_fuel": "Gas"})
        emps.append({"synthetic_employee_id": eid, "cluster_id": 0, "trips_per_day": 2,
                     "number_of_stops": 0, "work_arrival_time": 800.0,
                     "work_departure_time": 1700.0})
    return pd.DataFrame(rows)[ACTIVITY_COLUMNS], pd.DataFrame(emps)


def test_build_patterns_are_contiguous_0_to_24():
    activity, employees = _fixture()
    patterns = ev.build_patterns(activity, employees, site_id="bldg-90", seed=3)
    assert len(patterns) == 40
    for v in patterns:
        assert v["parking_lot"] == "bldg-90"
        for day in v["weekly driving patterns"]:
            acts = day["activities"]
            assert acts[0]["start_time"] == "00:00"
            assert acts[-1]["end_time"] == "23:59"
            for a, b in zip(acts, acts[1:]):        # no gaps / overlaps
                assert a["end_time"] == b["start_time"]


def test_estimate_charging_end_to_end(tmp_path):
    activity, employees = _fixture()
    ap, epth = tmp_path / "a.parquet", tmp_path / "e.parquet"
    activity.to_parquet(ap, index=False)
    employees.to_parquet(epth, index=False)
    summary = ev.estimate_charging(ap, epth, out_dir=tmp_path / "out",
                                   adoption_rate=0.36, run_period_days=2)
    assert summary["n_synthetic_vehicles"] == 40
    assert summary["n_evs_selected"] >= 1
    assert (tmp_path / "out" / "pov_driving_pattern.json").exists()
    assert Path(summary["status_file"]).exists()
