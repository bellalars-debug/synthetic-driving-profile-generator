"""Employee count -> predicted workplace EV charging demand.

Drives the whole thing behind the website:

  headcount
    -> NHTS synthetic drivers               (lbnl_model/lbnl_sim.py, fast)
    -> ev-tool pov_driving_pattern schema   (in-process, no file needed)
    -> ev-tool station/queue charging sim   (vendored, MIT, unmodified)
    -> daily load curve + peak kW + energy + recommended chargers

The charging simulator runs with an ample station count so the PEAK simultaneous
workplace charging it produces is the number of Level-2 chargers needed to serve
demand without queueing — that peak is the recommended charger count.
"""
from __future__ import annotations

import os
import sys
import datetime
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NHTS = os.path.join(_HERE, "nhts_generator")        # bundled NHTS generator + data
sys.path.insert(0, _NHTS)                            # for lbnl_model.lbnl_sim
sys.path.insert(0, os.path.join(_HERE, "ev_charging_sim"))   # for models.* / utilities.*

import lbnl_model.lbnl_sim as S
from models.vehicle import Vehicle
from models.charging_station import ChargingStation
from utilities.helpers import rank_and_select_vehicles, initialize_vehicles
from utilities.queue import ChargingQueue, assign_charging_stations, update_charging_queue

KWH_PER_GAL = 33.7
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORKDAYS = DAYS[:5]
L2_RATE_KW = 7.0


def _hhmm(minute: float) -> str:
    m = int(round(min(max(float(minute), 0.0), 24 * 60 - 1)))
    return f"{m // 60:02d}:{m % 60:02d}"


def _activities(emp) -> list:
    """lbnl_sim employee 0-24h timeline -> ev-tool activities (00:00->23:59)."""
    acts, aid, cur = [], 1, 0
    for a in emp.activities:
        end = min(int(round(a.end_min)), 24 * 60 - 1)
        if end <= cur:
            continue
        if a.state == "Driving":
            state, loc = "Driving", "Off-Site"
        else:
            state = "Parked"
            loc = {"Home": "Home", "Work": "On-Site"}.get(a.location, "Off-Site")
        acts.append({"activity_id": aid, "start_time": _hhmm(cur), "end_time": _hhmm(end),
                     "activity_type": state, "location": loc,
                     "driving_distance": round(a.distance_miles, 4) if state == "Driving" else 0})
        aid += 1
        cur = end
    if not acts:
        acts = [{"activity_id": 1, "start_time": "00:00", "end_time": "23:59",
                 "activity_type": "Parked", "location": "Home", "driving_distance": 0}]
    acts[-1]["end_time"] = "23:59"
    return acts


def _to_pattern(emp, uid, site_id, rng) -> dict:
    day_acts = _activities(emp)
    total_miles = float(sum(t.distance_miles for t in emp.trips))
    drive_min = sum((a.end_min - a.start_min) for a in emp.activities if a.state == "Driving")
    speed = float(np.clip((total_miles / (drive_min / 60.0)) if drive_min else 25.0, 5, 80))
    mpge = float(rng.normal(117, 3.2))
    rate = speed * KWH_PER_GAL / mpge
    kwh = total_miles * KWH_PER_GAL / mpge
    days_drive = 5 if emp.telework_status == "onsite" else 3
    drive_days = set(rng.choice(WORKDAYS, size=days_drive, replace=False))
    weekly = []
    for d in DAYS:
        if d in drive_days and total_miles > 0:
            weekly.append({"day of week": d, "driving distance": total_miles,
                           "equivalent efficiency mpge": mpge, "average speed mph": speed,
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
    return {"employee_id": str(uid), "vehicle id": str(uid), "parking_lot": site_id,
            "parking lot": site_id, "zev site": site_id, "onsite_bldg": site_id,
            "category": "Light-Duty", "battery capacity": 75, "home_charging":
            "Level 2" if rng.random() < 0.7 else "Level 1", "commute_mode": "car",
            "leave_to_work_time": _hhmm(emp.depart_home_min),
            "return_home_time": _hhmm(emp.depart_work_min),
            "distance": {"text": "", "value": int(round(emp.commute_distance_mi * 1609))},
            "duration": {"text": "", "value": int(round(emp.commute_duration_min * 60))},
            "status": "OK", "days_drive": days_drive,
            "weekly driving patterns": weekly}


def estimate(employees: int, adoption_rate: float = 0.36, seed: int = 42,
             site_id: str = "site", run_period_days: int = 2) -> dict:
    """Run the full estimate. Returns a JSON-serializable summary + load curve."""
    employees = int(max(1, min(employees, 5000)))
    adoption_rate = float(min(max(adoption_rate, 0.01), 1.0))

    # 1) synthetic drivers -> ev-tool patterns
    sim = S.LBNLSimulation(employees, "Company", site_id, S.DEFAULT_TABLES, seed=seed)
    sim.run()
    rng = np.random.default_rng(seed)
    patterns = [_to_pattern(p.employee, i + 1, site_id, rng)
                for i, p in enumerate(sim.profiles)]

    # 2) select EV adopters (ev-tool ranks by commute distance)
    site_vehicles = [v for v in patterns if v["distance"]["value"] < 100 * 1609]
    selected = rank_and_select_vehicles(site_vehicles, adoption_rate)
    ev_vehicles = [Vehicle(info, 80) for info in selected]
    n_evs = len(ev_vehicles)

    # 3) run one design day with ample L2 stations (peak = chargers needed, no queue).
    # Vehicles start the day partially depleted (initialize_vehicles -> SoC 80) and
    # charge at work as they arrive — the demand basis for sizing workplace chargers.
    n_stations = max(n_evs, 1)
    start = datetime.datetime(2024, 2, 5)               # a Monday (a drive day)
    end = start + datetime.timedelta(days=1)
    initialize_vehicles(ev_vehicles)
    # Realistic morning-arrival state of charge: drivers with home Level-2 arrive
    # well-charged; those on Level-1 / limited home charging arrive lower and top
    # up at work. This heterogeneity (not a fixed 80%) is what drives real
    # workplace daytime charging demand.
    soc_rng = np.random.default_rng(seed + 7)
    for v in ev_vehicles:
        if getattr(v, "home_charging", "Level 1") == "Level 2":
            v.soc = float(soc_rng.uniform(70, 90))
        else:
            v.soc = float(soc_rng.uniform(35, 60))
    stations = [ChargingStation("L2", maximum_power=L2_RATE_KW) for _ in range(n_stations)]
    queue = ChargingQueue()

    import pandas as pd
    # record instantaneous load per timestep, tagged by day, so the reported
    # peak and the displayed curve come from the SAME steady-state day.
    per_step = []                                      # (date, 'HH:MM', work_kw, home_kw)
    for t in pd.date_range(start=start, end=end, freq="15min"):
        if t.hour == 0 and t.minute == 0:
            for v in ev_vehicles:
                v.update_activities(t)
            queue.queue.clear()
            for stn in stations:
                stn.release()
        for v in ev_vehicles:
            v.update_location(t)
        assign_charging_stations(ev_vehicles, stations, queue, t)
        update_charging_queue(queue, stations)
        for v in ev_vehicles:
            v.update_status(t, queue, stations, 240)
            v.update_soc(15)
        work_kw = home_kw = 0.0
        for v in ev_vehicles:
            rate = v.charging_rate if isinstance(v.charging_rate, (int, float)) else 0.0
            if v.location == "On-Site" and rate:
                work_kw += rate
            elif v.location == "Home" and rate:
                home_kw += rate
        per_step.append((t.date(), t.strftime("%H:%M"), work_kw, home_kw))

    # 4) build the 24h curve from the design day (peak/curve/chargers all consistent)
    grid = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
    design_day = start.date()
    day_work = {tod: kw for d, tod, kw, _ in per_step if d == design_day}
    day_home = {tod: kw for d, tod, _, kw in per_step if d == design_day}
    work_curve = [round(day_work.get(tod, 0.0), 2) for tod in grid]
    home_curve = [round(day_home.get(tod, 0.0), 2) for tod in grid]

    peak_work_kw = max(work_curve) if work_curve else 0.0
    peak_work_vehicles = int(round(peak_work_kw / L2_RATE_KW))   # all workplace charging is L2
    total_work_kwh = round(sum(work_curve) * 0.25, 1)     # 15-min intervals -> hours
    total_home_kwh = round(sum(home_curve) * 0.25, 1)
    fleet_miles = round(sum(sum(t.distance_miles for t in p.employee.trips)
                            for p in sim.profiles), 0)

    return {
        "employees": employees,
        "adoption_rate": round(adoption_rate, 3),
        "ev_drivers": n_evs,
        "recommended_l2_chargers": int(peak_work_vehicles),
        "peak_workplace_power_kw": round(peak_work_kw, 1),
        "workplace_energy_kwh_per_day": total_work_kwh,
        "home_energy_kwh_per_day": total_home_kwh,
        "total_ev_energy_kwh_per_day": round(total_work_kwh + total_home_kwh, 1),
        "avg_commute_miles": round(float(np.mean([p.employee.commute_distance_mi
                                                  for p in sim.profiles])), 1),
        "fleet_daily_miles": fleet_miles,
        "l2_charger_kw": L2_RATE_KW,
        "curve_labels": grid,
        "workplace_load_curve_kw": work_curve,
        "home_load_curve_kw": home_curve,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(estimate(150), indent=2)[:1200])
