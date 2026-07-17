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
from collections import defaultdict, Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# locate the NHTS generator (works in both repo layouts: bundled or sibling)
_NHTS = next((c for c in (os.path.join(_HERE, "nhts_generator"), os.path.dirname(_HERE),
                          "/Users/ashishstephen/nhts_analysis")
              if os.path.isdir(os.path.join(c, "lbnl_model"))), os.path.dirname(_HERE))
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
             site_id: str = "site", run_period_days: int = 2,
             site_type: str = "Office", location: str = "",
             parking_spaces: int = 0) -> dict:
    """Run the full estimate. Returns a JSON-serializable payload for the
    Site / Transportation / Vehicle-Electrification / Infrastructure pages."""
    employees = int(max(1, min(employees, 5000)))
    adoption_rate = float(min(max(adoption_rate, 0.01), 1.0))
    parking_spaces = int(parking_spaces) if parking_spaces and parking_spaces > 0 else employees

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

    emps = [p.employee for p in sim.profiles]
    base = {
        "employees": employees,
        "adoption_rate": round(adoption_rate, 3),
        "ev_drivers": n_evs,
        "recommended_l2_chargers": int(peak_work_vehicles),
        "peak_workplace_power_kw": round(peak_work_kw, 1),
        "workplace_energy_kwh_per_day": total_work_kwh,
        "home_energy_kwh_per_day": total_home_kwh,
        "total_ev_energy_kwh_per_day": round(total_work_kwh + total_home_kwh, 1),
        "avg_commute_miles": round(float(np.mean([e.commute_distance_mi for e in emps])), 1),
        "fleet_daily_miles": fleet_miles,
        "l2_charger_kw": L2_RATE_KW,
        "curve_labels": grid,
        "workplace_load_curve_kw": work_curve,
        "home_load_curve_kw": home_curve,
    }
    base.update(_page_data(emps, base, site_type, location, parking_spaces))
    return base


# ---------------------------------------------------------------------------
# Extra analytics for the dashboard pages
# ---------------------------------------------------------------------------
FUEL_LABELS = {"gas": "Gasoline", "hybrid": "Hybrid", "ev": "Electric (BEV)",
               "phev": "Plug-in hybrid (PHEV)"}
LOC_DISPLAY = {"Home": "Home", "Work": "Work", "": "Driving"}


def _commute_mode_distribution():
    """NHTS national usual-commute-mode split (context for the fleet)."""
    import pandas as pd
    try:
        m = pd.read_csv(S.DEFAULT_TABLES)
        d = m[(m["distribution"] == "usual_commute_mode") & (m["geography"] == "National")]
        rows = [{"mode": str(r["category"]), "pct": round(float(r["weighted_pct"]), 1)}
                for _, r in d.iterrows()]
        rows.sort(key=lambda x: -x["pct"])
        top = rows[:6]
        other = round(sum(r["pct"] for r in rows[6:]), 1)
        if other > 0:
            top.append({"mode": "Other", "pct": other})
        return top
    except Exception:
        return [{"mode": "Car", "pct": 85.0}, {"mode": "Public transit", "pct": 5.0},
                {"mode": "Walk", "pct": 3.0}, {"mode": "Other", "pct": 7.0}]


def _profile_activities(emp):
    out = []
    for a in emp.activities:
        loc = "Driving" if a.state == "Driving" else LOC_DISPLAY.get(a.location, a.location or "Stop")
        out.append({"state": a.state, "start": round(a.start_min / 60.0, 3),
                    "end": round(a.end_min / 60.0, 3), "location": loc,
                    "distance": round(a.distance_miles, 1) if a.state == "Driving" else 0})
    return out


def _sample_profiles(emps, k=6):
    """A diverse handful of synthetic drivers with their daily timeline."""
    chained = [e for e in emps if e.morning_chain != "direct" or e.evening_chain != "direct"]
    evs = [e for e in emps if e.vehicle.is_ev]
    picks, seen = [], set()
    for pool in (chained[:2], evs[:1], emps):
        for e in pool:
            if e.employee_id in seen:
                continue
            seen.add(e.employee_id); picks.append(e)
            if len(picks) >= k:
                break
        if len(picks) >= k:
            break
    return [{
        "id": i + 1, "archetype": e.archetype_name, "age": e.age, "sex": e.sex,
        "income": e.income_range, "fuel": FUEL_LABELS.get(e.vehicle.fuel_type, e.vehicle.fuel_type),
        "is_ev": bool(e.vehicle.is_ev), "commute_mi": round(e.commute_distance_mi, 1),
        "depart": S.m2t(e.depart_home_min), "return": S.m2t(e.depart_work_min),
        "telework": e.telework_status, "activities": _profile_activities(e),
    } for i, e in enumerate(picks)]


def _page_data(emps, base, site_type, location, parking_spaces):
    n = len(emps)
    # driving clusters (archetypes)
    arch = Counter(e.archetype_name for e in emps)
    archetype_dist = [{"name": k, "count": v, "pct": round(100 * v / n, 1)}
                      for k, v in arch.most_common()]
    # fuel mix (NHTS-natural, income-conditioned)
    fuel = Counter(e.vehicle.fuel_type for e in emps)
    fuel_breakdown = [{"fuel": FUEL_LABELS.get(k, k), "count": v, "pct": round(100 * v / n, 1)}
                      for k, v in fuel.most_common()]
    natural_ev_pct = round(100 * sum(1 for e in emps if e.vehicle.is_ev) / n, 1)
    # driving characteristics
    dists = np.array([e.commute_distance_mi for e in emps])
    durs = np.array([e.commute_duration_min for e in emps])
    direct = np.mean([e.morning_chain == "direct" for e in emps])
    dep_hours = [int(e.depart_home_min // 60) for e in emps]
    depart_hist = [{"hour": h, "pct": round(100 * dep_hours.count(h) / n, 1)}
                   for h in range(4, 13)]
    driving_characteristics = {
        "avg_commute_mi": round(float(dists.mean()), 1),
        "median_commute_mi": round(float(np.median(dists)), 1),
        "avg_duration_min": round(float(durs.mean()), 1),
        "median_duration_min": round(float(np.median(durs)), 1),
        "avg_daily_miles": round(base["fleet_daily_miles"] / n, 1),
        "direct_pct": round(100 * float(direct), 0),
        "chained_pct": round(100 * (1 - float(direct)), 0),
        "pct_hybrid_workers": round(100 * float(np.mean([e.telework_status == "hybrid" for e in emps])), 0),
    }
    # charger types: L2 for the bulk, a few DC-fast for quick top-ups
    l2 = base["recommended_l2_chargers"]
    l3 = max(1, round(l2 * 0.1)) if l2 else 0
    chargers_by_type = [
        {"type": "Level 2 (7 kW)", "count": l2, "role": "Primary workplace charging"},
        {"type": "DC fast (50 kW)", "count": l3, "role": "Optional quick top-ups"},
    ]
    # infrastructure sizing
    peak = base["peak_workplace_power_kw"]
    transformer_kva = int(round(peak / 0.9 / 5) * 5) if peak else 0   # 0.9 PF, round to 5
    suggestions = [
        f"Install ~{l2} Level-2 (7 kW) ports to serve the {peak:.0f} kW design-day peak "
        f"without queueing.",
        f"Add ~{l3} DC-fast port(s) for drivers arriving low on charge who need a quick top-up.",
        f"Size the service for ~{transformer_kva} kVA (peak {peak:.0f} kW at ~0.9 power factor); "
        f"stagger or add load management to shave the mid-morning peak.",
        f"~{base['workplace_energy_kwh_per_day']:.0f} kWh/day of workplace energy — consider "
        f"time-of-use rates and on-site solar to offset the daytime peak.",
    ]
    return {
        "site": {"type": site_type or "Office", "location": location or "—",
                 "employees": base["employees"], "parking_spaces": parking_spaces,
                 "space_ratio": round(parking_spaces / base["employees"], 2)},
        "commute_mode_dist": _commute_mode_distribution(),
        "archetype_dist": archetype_dist,
        "sample_profiles": _sample_profiles(emps),
        "driving_characteristics": driving_characteristics,
        "depart_hist": depart_hist,
        "fuel_breakdown": fuel_breakdown,
        "natural_ev_pct": natural_ev_pct,
        "chargers_by_type": chargers_by_type,
        "infrastructure": {"transformer_kva": transformer_kva, "suggestions": suggestions},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(estimate(150), indent=2)[:1200])
