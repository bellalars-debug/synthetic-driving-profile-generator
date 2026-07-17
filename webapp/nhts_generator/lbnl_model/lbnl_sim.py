"""
LBNL Workplace Synthetic Driving-Profile Generator  (v2: archetype-driven)
==========================================================================
Stochastic synthetic-population model. Given a parking-lot vehicle count, it
generates one synthetic office employee per parked vehicle, assigns each a
DEMOGRAPHIC ARCHETYPE derived from the NHTS (see ../archetypes.py), builds a
full chronological daily driving profile, and estimates EV charging demand.

What changed in v2
------------------
Each synthetic driver is now a coherent PERSON, not a bag of independent draws.
An archetype (e.g. "Parent with School Drop-off", "Young Single Professional")
is sampled first from its NHTS prevalence, and every downstream attribute
(income, telework, age, household, commute distance/duration/times, trip
chaining) is then drawn from that archetype's NHTS-conditional distribution.
Because the archetypes PARTITION the office-worker population, the population-
weighted mixture of the conditionals reproduces the national NHTS marginals by
construction -- so aggregate validation still holds.

Outputs are written as a small RELATIONAL DATABASE:
  UserDemographicKey.csv   UserID -> ArchetypeID (thin link table)
  ArchetypeDefinitions.csv one row per archetype (written by archetypes.py)
  DriverCharacteristics.csv one row per UserID, all static attributes
  DriverProfiles.csv       V2G-Sim itinerary: one row per activity, 0h -> 24h
(plus the original SyntheticEmployees/Trips/ChargingDemand/summary files.)

- NOT machine learning. NOT BEAM. Pure weighted sampling from NHTS 2022 tables.
- Reproducible via --seed.  No real employee data or home addresses.
Author deliverable for Lawrence Berkeley National Laboratory.
"""
from __future__ import annotations
import argparse, os, math, json
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration & documented assumptions
# ---------------------------------------------------------------------------
_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_TABLES = os.path.join(_ROOT, "Driving_Profile_Probability_Tables.csv")
TABLES_DIR = os.path.join(_ROOT, "tables")
INCOME_FUEL_CSV = os.path.join(_ROOT, "income_based_vehicle_type_probabilities.csv")
INCOME_DIST_CSV = os.path.join(_ROOT, "worker_household_income_distribution.csv")
ARCHETYPE_JSON = os.path.join(_ROOT, "archetype_params.json")
GEO = "National"          # best sample size for behavioural *shape* (see report)

# Charging assumptions (documented; simple by design)
EV_EFF_KWH_PER_MI   = 0.30    # battery-electric efficiency
PHEV_EFF_KWH_PER_MI = 0.25    # plug-in hybrid electric-mode efficiency
L2_CHARGER_KW       = 6.6     # workplace Level-2 charger power
PHEV_ELECTRIC_RANGE = 30.0    # mi of electric range assumed for a PHEV/day

# V2G-Sim charger availability by parking location (Watts). Matches the
# Tennessee.xlsx reference itinerary: Home = Level-1 (1440 W), Work = L2 (6600 W).
LOC_PMAX_W = {"Home": 1440, "Work": 6600}      # all other stop locations: 0 W

# income-table fuel column -> whether that fuel plugs in (used for charging state)
EV_FUELS = {"ev", "phev"}

# Validation targets. Distance/duration/depart come straight from the office-
# worker NHTS marginals (identical to the earlier analysis). morning/evening
# direct are the OFFICE-worker directness rates (higher than the all-worker
# figures) since the model's universe is office workers.
TARGETS = {
    "avg_commute_distance_mi": 12.1, "median_commute_distance_mi": 8.9,
    "avg_commute_duration_min": 25.2, "median_commute_duration_min": 20.0,
    "morning_direct_pct": 95.0, "evening_direct_pct": 91.0,
    "midday_trip_pct": 3.5, "avg_daily_trips": 2.76,
    "avg_daily_driver_miles": 26.9, "ev_phev_share_pct": 1.7,
    "hybrid_share_pct": 3.0, "depart_before_8am_pct": 70.5,
}


# ===========================================================================
# ProbabilitySampler  -- reads the master table, samples categories & bins
# ===========================================================================
class ProbabilitySampler:
    """Loads the NHTS-derived probability tables and provides weighted draws.
    Falls back to target-calibrated parametric distributions if a table is
    missing or malformed."""

    def __init__(self, master_csv: str, rng: np.random.Generator, geo: str = GEO):
        self.rng = rng
        self.geo = geo
        self.ok = True
        try:
            m = pd.read_csv(master_csv)
            self._m = m
            self._idx = {k: g for k, g in m.groupby(["distribution", "geography"])}
        except Exception as e:                                   # fallback mode
            print(f"[sampler] WARNING master table unreadable ({e}); using fallbacks")
            self.ok = False
            self._idx = {}

    def _get(self, dist: str, geo: Optional[str] = None):
        return self._idx.get((dist, geo or self.geo))

    def sample_category(self, dist: str, geo: Optional[str] = None):
        df = self._get(dist, geo)
        if df is None or df.empty:
            return None
        p = df["weighted_pct"].to_numpy(float); p = p / p.sum()
        return str(df["category"].to_numpy()[self.rng.choice(len(df), p=p)])

    def sample_bin(self, dist: str, geo: Optional[str] = None, fallback=None):
        df = self._get(dist, geo)
        if df is None or df.empty or df["bin_start"].isna().all():
            return fallback() if fallback else None
        d = df.dropna(subset=["bin_start", "bin_end"])
        p = d["weighted_pct"].to_numpy(float); p = p / p.sum()
        i = self.rng.choice(len(d), p=p)
        lo = float(d["bin_start"].to_numpy()[i]); hi = float(d["bin_end"].to_numpy()[i])
        return float(self.rng.uniform(lo, hi))

    def sample_percentiles(self, stats_row: dict, cap: float = 1.0):
        """Inverse-CDF draw from a percentile row. `cap`<1 clips the upper tail."""
        if cap < 1.0:
            qs = [(0.0, "min"), (.05, "p5"), (.10, "p10"), (.25, "p25"),
                  (.50, "median"), (.75, "p75"), (.90, "p90"), (.95, "p95"), (.99, "p99")]
        else:
            qs = [(0.0, "min"), (.05, "p5"), (.10, "p10"), (.25, "p25"),
                  (.50, "median"), (.75, "p75"), (.90, "p90"), (.95, "p95"),
                  (.99, "p99"), (1.0, "max")]
        xs = [stats_row[k] for _, k in qs]
        ps = [p for p, _ in qs]
        u = min(float(self.rng.uniform(0, 1)), cap)
        return float(np.interp(u, ps, xs))


# ===========================================================================
# ArchetypeSampler -- samples an archetype then draws archetype-conditional
#                     attributes, with national fallback for any missing cell.
# ===========================================================================
class ArchetypeSampler:
    def __init__(self, national: ProbabilitySampler, rng, json_path=ARCHETYPE_JSON):
        self.s = national; self.rng = rng
        self.ok = True
        try:
            with open(json_path) as f:
                self.P = json.load(f)
        except Exception as e:
            print(f"[archetype] WARNING params unreadable ({e}); national fallback")
            self.ok = False; self.P = {}
        # pooled national office duration percentile row: preserves the heaped
        # 20-min median that small per-archetype rows quantize away. Duration is
        # only weakly archetype-specific (it tracks distance), so it is drawn from
        # the national row while DISTANCE stays archetype-conditional.
        self._natl_dur = None
        try:
            df = pd.read_csv(os.path.join(TABLES_DIR, "03_HWoffice_summary_National.csv"))
            self._natl_dur = df.set_index("metric").loc["duration_min"].to_dict()
        except Exception:
            pass
        self.ids = sorted([k for k in self.P if not k.startswith("_")])
        self.prev = np.array([self.P[a]["prevalence"] for a in self.ids]) \
            if self.ids else np.array([])
        if self.prev.size:
            self.prev = self.prev / self.prev.sum()

    # -- pick the archetype for one driver --
    def draw(self) -> str:
        if not self.ids:
            return "A00"
        return self.ids[self.rng.choice(len(self.ids), p=self.prev)]

    def _p(self, aid):
        return self.P.get(aid, {})

    def _inv(self, row, cap=1.0, lo=None):
        v = self.s.sample_percentiles(row, cap=cap)
        return max(lo, v) if lo is not None else v

    # -- conditional attribute draws --
    def income(self, aid):
        p = self._p(aid)
        br = p.get("income_brackets"); pr = p.get("income_probs")
        if not br:
            return None
        pr = np.array(pr, float); pr = pr / pr.sum()
        return br[self.rng.choice(len(br), p=pr)]

    def telework(self, aid):
        p = self._p(aid).get("telework_probs")
        if not p:
            return None
        keys = ["onsite", "hybrid", "remote"]
        pr = np.array([p[k] for k in keys], float); pr = pr / pr.sum()
        return keys[self.rng.choice(3, p=pr)]

    def age(self, aid):
        r = self._p(aid).get("age")
        return int(round(self._inv(r, lo=16))) if r else None

    def household(self, aid):
        """Return (household_size, num_workers, num_children) as small integers
        drawn around the archetype means with Poisson-ish jitter but bounded."""
        p = self._p(aid)
        hs = p.get("hhsize_mean"); wk = p.get("workers_mean"); ch = p.get("children_mean")
        if hs is None:
            return None, None, None
        num_children = int(self.rng.poisson(max(ch, 0.0))) if ch and ch > 0.05 else 0
        num_children = min(num_children, 5)
        # workers: 1 or 2+ around the mean (bounded to household adults)
        n_workers = 2 if self.rng.uniform() < max(min(wk - 1, 1.0), 0.0) else 1
        adults = max(1, int(round(hs - ch)) if hs and ch is not None else 1)
        adults = max(adults, n_workers)
        hhsize = adults + num_children
        return hhsize, n_workers, num_children

    def sex(self, aid):
        # gender is fixed for a gender-split archetype; otherwise drawn from the
        # archetype's observed female share.
        s = self._p(aid).get("sex")
        if s:
            return s
        fs = self._p(aid).get("female_share")
        if fs is None:
            return ""
        return "Female" if self.rng.uniform() < fs else "Male"

    def commute_distance(self, aid):
        r = self._p(aid).get("commute_distance_pct")
        if r:
            return max(0.3, self._inv(r, cap=0.99))
        return max(0.3, self.s.sample_bin("HWoffice_distance_2p5mi", GEO,
                   fallback=lambda: float(self.rng.lognormal(2.1, 0.9))))

    def commute_duration(self, aid):
        # national office row (heaped median 20); falls back to a lognormal
        # calibrated to median 20 / mean 24.6 if the table is missing.
        if self._natl_dur:
            return max(3.0, self.s.sample_percentiles(self._natl_dur, cap=0.99))
        mu = math.log(20.0)
        sg = math.sqrt(max(2 * (math.log(24.6) - mu), 0.04))
        return max(3.0, float(self.rng.lognormal(mu, sg)))

    def depart_home(self, aid):
        r = self._p(aid).get("depart_home_pct")
        if r:
            return float(np.clip(self._inv(r, cap=0.99), 4 * 60, 11 * 60))
        return self.s.sample_bin("HWoffice_departure_15min", GEO,
                                 fallback=lambda: float(self.rng.normal(7.7 * 60, 90)))

    def depart_work(self, aid):
        r = self._p(aid).get("depart_work_pct")
        if r:
            return float(np.clip(self._inv(r, cap=0.99), 11 * 60, 23 * 60))
        return self.s.sample_bin("WHoffice_departure_15min", GEO,
                                 fallback=lambda: float(self.rng.normal(16.75 * 60, 110)))

    def morning_stop(self, aid):
        """Return a single stop purpose (str) or None for a direct commute."""
        p = self._p(aid)
        pd_ = p.get("morning_direct_p")
        if pd_ is None:
            return None
        if self.rng.uniform() < pd_:
            return None
        purposes = p.get("morning_purposes"); probs = p.get("morning_purpose_probs")
        pr = np.array(probs, float); pr = pr / pr.sum()
        return purposes[self.rng.choice(len(purposes), p=pr)]

    def evening_stop(self, aid):
        p = self._p(aid)
        pd_ = p.get("evening_direct_p")
        if pd_ is None:
            return None
        if self.rng.uniform() < pd_:
            return None
        purposes = p.get("evening_purposes"); probs = p.get("evening_purpose_probs")
        pr = np.array(probs, float); pr = pr / pr.sum()
        return purposes[self.rng.choice(len(purposes), p=pr)]

    def name(self, aid):
        return self._p(aid).get("name", aid)


# ===========================================================================
# Domain objects
# ===========================================================================
@dataclass
class Vehicle:
    vehicle_id: str
    fuel_type: str            # 'gas' | 'hybrid' | 'ev' | 'phev'
    fuel_detail: str
    is_ev: bool
    income_bracket: str = ""

@dataclass
class Trip:
    employee_id: str
    vehicle_id: str
    trip_id: str
    origin: str
    destination: str
    purpose: str
    depart_time: str
    arrive_time: str
    distance_miles: float
    duration_minutes: float
    stop_duration_minutes: Optional[float]
    fuel_type: str
    is_ev: bool

@dataclass
class Activity:
    """One contiguous timeline segment for the V2G-Sim itinerary (absolute min)."""
    state: str                # 'Parked' | 'Driving' | 'Charging'
    start_min: float
    end_min: float
    distance_miles: float
    location: str             # '' for Driving
    pmax_w: int

@dataclass
class Employee:
    employee_id: str
    vehicle: Vehicle
    worker_type: str
    telework_status: str
    office_name: str
    office_location: str
    home_location: str
    # demographic archetype + attributes
    archetype_id: str = "A00"
    archetype_name: str = ""
    age: int = 0
    sex: str = ""
    household_size: int = 1
    num_workers: int = 1
    num_children: int = 0
    income_bracket: str = ""
    income_range: str = ""
    # assigned commute behaviour
    commute_distance_mi: float = 0.0
    commute_duration_min: float = 0.0
    depart_home_min: float = 0.0
    arrive_work_min: float = 0.0
    depart_work_min: float = 0.0
    arrive_home_min: float = 0.0
    morning_chain: str = "direct"
    evening_chain: str = "direct"
    had_midday: bool = False
    trips: list = field(default_factory=list)
    activities: list = field(default_factory=list)

@dataclass
class DrivingProfile:
    employee: Employee
    trips: list


def m2t(m: float) -> str:
    m = int(round(m)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


INCOME_RANGES = {"1_<$50k": "<$50,000", "2_$50-100k": "$50,000-99,999",
                 "3_$100-150k": "$100,000-149,999", "4_$150-200k": "$150,000-199,999",
                 "5_$200k+": "$200,000+"}

# stop purpose bucket -> V2G-Sim parking Location label
PURPOSE_LOC = {
    "School": "School", "Daycare/Care": "Daycare", "Drop/Pickup person": "Other",
    "Meals/Coffee": "Restaurant", "Shopping": "Shopping/Errands",
    "Health/Personal errand": "Medical", "Social/Rec": "Gym",
    "Work-related": "Other", "Change mode": "Other", "Other": "Other",
}


# ===========================================================================
# Generators
# ===========================================================================
class VehicleFactory:
    """Assigns vehicle fuel type CONDITIONED ON HOUSEHOLD INCOME."""
    FUEL_COLS = {
        "p_gasoline": ("gas", False, "Gas"),
        "p_hybrid": ("hybrid", False, "Hybrid non-plugin (HEV)"),
        "p_phev": ("phev", True, "Plug-in hybrid (PHEV)"),
        "p_bev": ("ev", True, "Electric only (BEV)"),
        "p_diesel_other": ("gas", False, "Diesel/Other"),
    }
    _FALLBACK = {"p_gasoline": 0.9247, "p_hybrid": 0.0304, "p_phev": 0.0049,
                 "p_bev": 0.0125, "p_diesel_other": 0.0275}

    def __init__(self, rng, income_fuel_csv: str = INCOME_FUEL_CSV):
        self.rng = rng
        self.cols = list(self.FUEL_COLS.keys())
        self.by_bracket = {}
        try:
            df = pd.read_csv(income_fuel_csv)
            for _, r in df.iterrows():
                self.by_bracket[r["income_bracket"]] = np.array(
                    [float(r[c]) for c in self.cols])
        except Exception as e:
            print(f"[vehicle] WARNING income-fuel table unreadable ({e}); fallback")
            self.by_bracket["ALL"] = np.array([self._FALLBACK[c] for c in self.cols])

    def make(self, vid: str, income_bracket: str) -> Vehicle:
        p = self.by_bracket.get(income_bracket, self.by_bracket.get("ALL"))
        p = p / p.sum()
        col = self.cols[self.rng.choice(len(self.cols), p=p)]
        bucket, is_ev, detail = self.FUEL_COLS[col]
        return Vehicle(vid, bucket, detail, is_ev, income_bracket)


class EmployeeGenerator:
    """Creates one synthetic office employee per parked vehicle.
    Flow: sample archetype -> archetype-conditional income -> vehicle | income,
    plus archetype-conditional age / household / sex / telework."""
    def __init__(self, arch: ArchetypeSampler, national: ProbabilitySampler, rng,
                 office_name, office_location,
                 income_dist_csv: str = INCOME_DIST_CSV,
                 income_distribution: Optional[dict] = None):
        self.arch = arch; self.s = national; self.rng = rng
        self.office_name = office_name; self.office_location = office_location
        self.vf = VehicleFactory(rng)
        self.inc_brackets, self.inc_probs = self._load_income(
            income_dist_csv, income_distribution)

    def _load_income(self, csv, override):
        if override:
            b = list(override.keys()); p = np.array([override[k] for k in b], float)
            return b, p / p.sum()
        try:
            df = pd.read_csv(csv)
            b = df["income_bracket"].tolist()
            p = df["weighted_pct"].to_numpy(float); p = p / p.sum()
            return b, p
        except Exception:
            b = list(INCOME_RANGES.keys())
            p = np.array([0.240, 0.316, 0.204, 0.105, 0.134])
            return b, p / p.sum()

    def _telework(self, aid):
        # archetype-conditional WKFMHM22; a parked car => commuted today, so the
        # fully-remote class is treated as hybrid for scheduling.
        status = self.arch.telework(aid)
        if status is None:                      # national fallback
            cat = self.s.sample_category("wfh_frequency", GEO)
            status = {"Never WFH (commute 5d)": "onsite",
                      "WFH 1-2 d/wk (hybrid)": "hybrid",
                      "WFH 3-4 d/wk (hybrid)": "hybrid",
                      "WFH 5+ d/wk (remote)": "remote"}.get(cat, "onsite")
        return "hybrid" if status == "remote" else status

    def make(self, i: int) -> Employee:
        eid = f"LBNL_EMP_{i:04d}"
        aid = self.arch.draw()
        # income: archetype-conditional, else national marginal
        bracket = self.arch.income(aid)
        if bracket is None:
            bracket = self.inc_brackets[self.rng.choice(len(self.inc_brackets),
                                                        p=self.inc_probs)]
        veh = self.vf.make(f"VEH_{i:04d}", bracket)
        age = self.arch.age(aid) or int(self.rng.integers(22, 64))
        hhsize, nworkers, nchildren = self.arch.household(aid)
        if hhsize is None:
            hhsize, nworkers, nchildren = 1, 1, 0
        sex = self.arch.sex(aid)
        return Employee(
            employee_id=eid, vehicle=veh,
            worker_type="office worker (fixed workplace)",
            telework_status=self._telework(aid),
            office_name=self.office_name, office_location=self.office_location,
            home_location=f"Synthetic Home Location {i:04d}",
            archetype_id=aid, archetype_name=self.arch.name(aid),
            age=age, sex=sex, household_size=hhsize, num_workers=nworkers,
            num_children=nchildren,
            income_bracket=bracket, income_range=INCOME_RANGES.get(bracket, ""),
        )


class TripChainGenerator:
    """Supplies stop dwell times (by purpose) and midday sub-tours. Chain
    directness itself is now decided by the ArchetypeSampler."""
    def __init__(self, sampler, rng):
        self.s = sampler; self.rng = rng
        self.dwell = self._load_dwell()
        self.midday_stats = self._load_midday_stats()

    def _load_dwell(self):
        try:
            df = pd.read_csv(os.path.join(TABLES_DIR, "07_stop_dwell_by_purpose.csv"))
            return {r["purpose"]: r for _, r in df.iterrows()}
        except Exception:
            return {}

    def _load_midday_stats(self):
        try:
            df = pd.read_csv(os.path.join(TABLES_DIR, "05_midday_stats.csv"))
            return {r["metric"]: r for _, r in df.iterrows()}
        except Exception:
            return {}

    def dwell_for(self, purpose: str) -> float:
        key = {
            "Drop/Pickup person": "Drop/Pickup person", "Meals/Coffee": "Meals/Coffee",
            "Shopping": "Shopping", "School": "School", "Daycare/Care": "Daycare/Care",
            "Social/Rec": "Social/Rec", "Health/Personal errand": "Health/Personal errand",
            "Work-related": "Work-related", "Change mode": "Change mode",
        }.get(purpose, "Other")
        r = self.dwell.get(key)
        val = self.s.sample_percentiles(r) if r is not None else float(self.rng.uniform(10, 45))
        return float(min(max(val, 3.0), 90.0))

    def midday(self, emp):
        p = 0.035
        try:
            pp = pd.read_csv(os.path.join(TABLES_DIR, "05_midday_probability.csv"))
            p = float(pp["value"].iloc[0]) / 100.0
        except Exception:
            pass
        if self.rng.uniform() >= p:
            return None
        emp.had_midday = True
        purpose = self.s.sample_category("midday_purpose", GEO) or "Meals/Coffee"
        def stat(metric, dflt):
            r = self.midday_stats.get(metric)
            return self.s.sample_percentiles(r) if r is not None else dflt()
        dist = max(0.2, stat("midday_trip_distance_mi", lambda: self.rng.uniform(1, 8)))
        dur = max(2.0, stat("midday_trip_duration_min", lambda: self.rng.uniform(5, 20)))
        dwell = min(max(5.0, stat("midday_stop_dwell_min", lambda: self.rng.uniform(20, 40))), 120.0)
        return {"purpose": purpose, "dist": dist, "dur": dur, "dwell": dwell}


class RouteGenerator:
    """v1 straight-line routes. Swap for GoogleMapsRouteGenerator later."""
    def route(self, origin, destination, distance_miles, duration_minutes, depart_min):
        return {"origin": origin, "destination": destination,
                "distance_miles": round(distance_miles, 2),
                "duration_minutes": round(duration_minutes, 1),
                "depart_time": m2t(depart_min),
                "arrive_time": m2t(depart_min + duration_minutes)}


class GoogleMapsRouteGenerator(RouteGenerator):
    def __init__(self, api_key: str):
        self.api_key = api_key
    def route(self, *a, **k):                                   # pragma: no cover
        raise NotImplementedError("Provide Google Maps Directions API integration")


class ChargingDemandEstimator:
    """Estimates EV/PHEV energy & workplace charging demand. Simple assumptions."""
    def estimate(self, emp: Employee, daily_miles: float, work_dwell_hours: float):
        v = emp.vehicle
        if not v.is_ev:
            return None
        if v.fuel_type == "ev":
            eff = EV_EFF_KWH_PER_MI; elec_miles = daily_miles
        else:
            eff = PHEV_EFF_KWH_PER_MI
            elec_miles = min(daily_miles, PHEV_ELECTRIC_RANGE)
        energy = elec_miles * eff
        home_leg = emp.commute_distance_mi
        workplace_need_kwh = min(home_leg * eff, energy)
        home_need_kwh = max(energy - workplace_need_kwh, 0.0)
        deliverable = L2_CHARGER_KW * max(work_dwell_hours, 0)
        workplace_kwh_demand = min(workplace_need_kwh, deliverable)
        needs_workplace = workplace_need_kwh > 0.05
        return {
            "employee_id": emp.employee_id, "vehicle_id": v.vehicle_id,
            "fuel_type": v.fuel_type, "daily_miles": round(daily_miles, 2),
            "efficiency_kwh_per_mi": eff, "total_energy_kwh": round(energy, 2),
            "workplace_charging_need": needs_workplace,
            "workplace_kwh_demand": round(workplace_kwh_demand, 2),
            "home_kwh_demand": round(home_need_kwh, 2),
            "work_dwell_hours": round(work_dwell_hours, 2),
            "l2_charger_kw": L2_CHARGER_KW,
        }


# ===========================================================================
# DrivingProfileGenerator -- orchestrates one full daily timeline per employee
# ===========================================================================
DAY_MIN = 24 * 60


class DrivingProfileGenerator:
    def __init__(self, national, arch, rng, office_name, office_location,
                 income_distribution=None):
        self.rng = rng
        self.office = office_name
        self.arch = arch
        self.empgen = EmployeeGenerator(arch, national, rng, office_name,
                                        office_location, income_distribution=income_distribution)
        self.chain = TripChainGenerator(national, rng)
        self.router = RouteGenerator()
        self.charger = ChargingDemandEstimator()

    # ---- record both a Trip (legacy output) and Activity segments (timeline) ----
    def _drive(self, emp, tid, origin, dest, purpose, depart_min, dist, dur):
        r = self.router.route(origin, dest, dist, dur, depart_min)
        emp.trips.append(Trip(
            emp.employee_id, emp.vehicle.vehicle_id, f"{emp.employee_id}_T{tid}",
            r["origin"], r["destination"], purpose, r["depart_time"], r["arrive_time"],
            r["distance_miles"], r["duration_minutes"], None,
            emp.vehicle.fuel_type, emp.vehicle.is_ev))
        emp.activities.append(Activity("Driving", depart_min, depart_min + dur,
                                       round(dist, 2), "", 0))
        return depart_min + dur

    def _park(self, emp, location, start_min, end_min):
        if end_min <= start_min:
            return start_min
        pmax = LOC_PMAX_W.get(location, 0)
        charging = emp.vehicle.is_ev and pmax > 0      # EV/PHEV plugged in where a charger exists
        state = "Charging" if charging else "Parked"
        emp.activities.append(Activity(state, start_min, end_min, 0.0, location, pmax))
        return end_min

    def generate(self, i: int) -> DrivingProfile:
        emp = self.empgen.make(i)
        aid = emp.archetype_id
        HOME, WORK = emp.home_location, self.office
        emp.commute_distance_mi = self.arch.commute_distance(aid)
        emp.commute_duration_min = self.arch.commute_duration(aid)
        emp.depart_home_min = self.arch.depart_home(aid)
        tid = 1

        # ---------------- MORNING ----------------
        m_stop = self.arch.morning_stop(aid)
        emp.morning_chain = "direct" if not m_stop else f"Home->Work via {m_stop}"
        D, M = emp.commute_distance_mi, emp.commute_duration_min
        t = self._park(emp, "Home", 0.0, emp.depart_home_min)
        if not m_stop:
            t = self._drive(emp, tid, HOME, WORK, "Home->Work commute", t, D, M); tid += 1
        else:
            detour = 1.20; per_d = D * detour / 2; per_t = M * detour / 2
            loc = PURPOSE_LOC.get(m_stop, "Other")
            t = self._drive(emp, tid, HOME, f"{m_stop} stop", f"Morning stop: {m_stop}",
                            t, per_d, per_t); tid += 1
            t = self._park(emp, loc, t, t + self.chain.dwell_for(m_stop))
            t = self._drive(emp, tid, f"{m_stop} stop", WORK,
                            "Home->Work commute (final leg)", t, per_d, per_t); tid += 1
        emp.arrive_work_min = t

        # ---------------- decide evening work-departure ----------------
        dep_w = self.arch.depart_work(aid)
        if dep_w <= emp.arrive_work_min + 240:              # enforce >=4h workday
            dep_w = emp.arrive_work_min + self.rng.uniform(360, 540)
        # keep the whole day inside 24h (leave room for the evening commute).
        # Evening distance = morning distance (same home<->work route); only the
        # duration is redrawn, so travel time can differ with traffic.
        De = emp.commute_distance_mi
        Me = self.arch.commute_duration(aid)
        dep_w = min(dep_w, DAY_MIN - Me - 20)
        emp.depart_work_min = dep_w

        # ---------------- MID-DAY ----------------
        md = self.chain.midday(emp)
        cursor = emp.arrive_work_min
        if md and emp.arrive_work_min + 60 < dep_w:
            md_depart = min(emp.arrive_work_min + self.rng.uniform(120, 240), dep_w - 30)
            cursor = self._park(emp, "Work", cursor, md_depart)
            loc = PURPOSE_LOC.get(md["purpose"], "Other")
            back = self._drive(emp, tid, WORK, f"Midday: {md['purpose']}",
                               f"Midday {md['purpose']}", md_depart, md["dist"], md["dur"]); tid += 1
            back = self._park(emp, loc, back, back + md["dwell"])
            cursor = self._drive(emp, tid, f"Midday: {md['purpose']}", WORK,
                                 "Return to work", back, md["dist"], md["dur"]); tid += 1
        # park at work until the evening departure
        cursor = self._park(emp, "Work", cursor, dep_w)

        # ---------------- EVENING ----------------
        e_stop = self.arch.evening_stop(aid)
        emp.evening_chain = "direct" if not e_stop else f"Work->Home via {e_stop}"
        t = dep_w
        if not e_stop:
            t = self._drive(emp, tid, WORK, HOME, "Work->Home commute", t, De, Me); tid += 1
        else:
            detour = 1.20; per_d = De * detour / 2; per_t = Me * detour / 2
            loc = PURPOSE_LOC.get(e_stop, "Other")
            t = self._drive(emp, tid, WORK, f"{e_stop} stop", f"Evening stop: {e_stop}",
                            t, per_d, per_t); tid += 1
            t = self._park(emp, loc, t, t + self.chain.dwell_for(e_stop))
            t = self._drive(emp, tid, f"{e_stop} stop", HOME,
                            "Work->Home commute (final leg)", t, per_d, per_t); tid += 1
        emp.arrive_home_min = min(t, DAY_MIN)
        # final Home parked segment to exactly 24h
        self._park(emp, "Home", min(t, DAY_MIN), DAY_MIN)
        emp.trips = emp.trips           # (already populated)
        return DrivingProfile(emp, emp.trips)


# ===========================================================================
# Validator
# ===========================================================================
class Validator:
    def __init__(self, employees):
        self.emps = employees

    def compute(self):
        n = len(self.emps)
        dists = np.array([e.commute_distance_mi for e in self.emps])
        durs = np.array([e.commute_duration_min for e in self.emps])
        m_direct = np.mean([e.morning_chain == "direct" for e in self.emps]) * 100
        e_direct = np.mean([e.evening_chain == "direct" for e in self.emps]) * 100
        midday = np.mean([e.had_midday for e in self.emps]) * 100
        trips = np.mean([len(e.trips) for e in self.emps])
        daily_miles = np.mean([sum(t.distance_miles for t in e.trips) for e in self.emps])
        ev_phev = np.mean([e.vehicle.is_ev for e in self.emps]) * 100
        hybrid = np.mean([e.vehicle.fuel_type == "hybrid" for e in self.emps]) * 100
        before8 = np.mean([e.depart_home_min < 8 * 60 for e in self.emps]) * 100
        return {
            "n_employees": n,
            "avg_commute_distance_mi": round(float(dists.mean()), 2),
            "median_commute_distance_mi": round(float(np.median(dists)), 2),
            "avg_commute_duration_min": round(float(durs.mean()), 2),
            "median_commute_duration_min": round(float(np.median(durs)), 2),
            "morning_direct_pct": round(float(m_direct), 2),
            "evening_direct_pct": round(float(e_direct), 2),
            "midday_trip_pct": round(float(midday), 2),
            "avg_daily_trips": round(float(trips), 2),
            "avg_daily_driver_miles": round(float(daily_miles), 2),
            "ev_phev_share_pct": round(float(ev_phev), 2),
            "hybrid_share_pct": round(float(hybrid), 2),
            "depart_before_8am_pct": round(float(before8), 2),
        }

    def report(self):
        got = self.compute()
        rows = []
        keymap = [
            ("avg_commute_distance_mi", 1.5), ("median_commute_distance_mi", 2.0),
            ("avg_commute_duration_min", 3.0), ("median_commute_duration_min", 3.0),
            ("morning_direct_pct", 3.0), ("evening_direct_pct", 3.0),
            ("midday_trip_pct", 2.5), ("avg_daily_trips", 0.6),
            ("avg_daily_driver_miles", 6.0), ("ev_phev_share_pct", 2.0),
            ("hybrid_share_pct", 2.5), ("depart_before_8am_pct", 8.0),
        ]
        for k, tol in keymap:
            g = got[k]; tgt = TARGETS[k]
            ok = abs(g - tgt) <= tol
            rows.append({"metric": k, "synthetic": g, "nhts_target": tgt,
                         "abs_diff": round(abs(g - tgt), 2), "tolerance": tol,
                         "pass": "PASS" if ok else "CHECK"})
        return got, pd.DataFrame(rows)


# ===========================================================================
# Simulation driver
# ===========================================================================
class LBNLSimulation:
    def __init__(self, parking_count, office_name, office_location,
                 tables_csv, seed=None, income_distribution=None):
        self.parking_count = parking_count
        self.office_name = office_name
        self.office_location = office_location
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sampler = ProbabilitySampler(tables_csv, self.rng)
        self.arch = ArchetypeSampler(self.sampler, self.rng)
        self.gen = DrivingProfileGenerator(self.sampler, self.arch, self.rng,
                                           office_name, office_location,
                                           income_distribution=income_distribution)
        self.profiles = []

    def run(self):
        self.profiles = [self.gen.generate(i + 1) for i in range(self.parking_count)]
        return self.profiles

    # ---- output writers ----
    def write_outputs(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        emps = [p.employee for p in self.profiles]

        # ---------- legacy SyntheticEmployees.csv ----------
        erows = []
        for e in emps:
            erows.append({
                "employee_id": e.employee_id, "vehicle_id": e.vehicle.vehicle_id,
                "archetype_id": e.archetype_id, "archetype_name": e.archetype_name,
                "worker_type": e.worker_type, "telework_status": e.telework_status,
                "age": e.age, "sex": e.sex, "household_size": e.household_size,
                "num_workers": e.num_workers, "num_children": e.num_children,
                "income_bracket": e.income_bracket, "income_range": e.income_range,
                "fuel_type": e.vehicle.fuel_type, "fuel_detail": e.vehicle.fuel_detail,
                "is_ev": e.vehicle.is_ev, "home_location": e.home_location,
                "office_name": e.office_name, "office_location": e.office_location,
                "commute_distance_mi": round(e.commute_distance_mi, 2),
                "commute_duration_min": round(e.commute_duration_min, 1),
                "depart_home": m2t(e.depart_home_min), "arrive_work": m2t(e.arrive_work_min),
                "depart_work": m2t(e.depart_work_min), "arrive_home": m2t(e.arrive_home_min),
                "morning_chain": e.morning_chain, "evening_chain": e.evening_chain,
                "had_midday_trip": e.had_midday, "n_trips": len(e.trips),
            })
        pd.DataFrame(erows).to_csv(os.path.join(outdir, "SyntheticEmployees.csv"), index=False)

        # ---------- legacy SyntheticTrips.csv ----------
        trows = [asdict(t) for e in emps for t in e.trips]
        pd.DataFrame(trows).to_csv(os.path.join(outdir, "SyntheticTrips.csv"), index=False)

        # ================= RELATIONAL DATABASE =================
        uid = {e.employee_id: n + 1 for n, e in enumerate(emps)}   # UserID = 1..N

        # 1) UserDemographicKey.csv  (thin link: UserID -> archetype)
        pd.DataFrame([{
            "UserID": uid[e.employee_id], "ArchetypeID": e.archetype_id,
            "ArchetypeName": e.archetype_name,
            "Description": self._arch_desc(e.archetype_id),
        } for e in emps]).to_csv(os.path.join(outdir, "UserDemographicKey.csv"), index=False)

        # 2) DriverCharacteristics.csv  (one row per UserID, all static attributes)
        pd.DataFrame([{
            "UserID": uid[e.employee_id], "ArchetypeID": e.archetype_id,
            "Age": e.age, "Sex": e.sex, "IncomeBracket": e.income_range,
            "HouseholdSize": e.household_size, "NumWorkers": e.num_workers,
            "NumChildren": e.num_children, "VehicleType": e.vehicle.fuel_detail,
            "FuelType": e.vehicle.fuel_type, "IsEV": e.vehicle.is_ev,
            "CommuteDistance_mi": round(e.commute_distance_mi, 2),
            "CommuteDuration_min": round(e.commute_duration_min, 1),
            "DepartHome": m2t(e.depart_home_min), "ArriveWork": m2t(e.arrive_work_min),
            "DepartWork": m2t(e.depart_work_min), "TeleworkStatus": e.telework_status,
            "MorningChain": e.morning_chain, "EveningChain": e.evening_chain,
            "NHTS_HH_Weight": self._weight(e),
        } for e in emps]).to_csv(os.path.join(outdir, "DriverCharacteristics.csv"), index=False)

        # 3) DriverProfiles.csv  (V2G-Sim itinerary: one row per activity, 0h->24h)
        prows = []
        for e in emps:
            w = self._weight(e)
            for a in e.activities:
                prows.append({
                    "User ID": uid[e.employee_id], "State": a.state,
                    "Start time (hour)": round(a.start_min / 60.0, 4),
                    "End time (hour)": round(a.end_min / 60.0, 4),
                    "Distance (mi)": round(a.distance_miles, 2),
                    "Nothing": 0, "P_max (W)": a.pmax_w,
                    "Location": a.location if a.location else "",
                    "NHTS HH Wt": w,
                })
        pd.DataFrame(prows, columns=["User ID", "State", "Start time (hour)",
            "End time (hour)", "Distance (mi)", "Nothing", "P_max (W)", "Location",
            "NHTS HH Wt"]).to_csv(os.path.join(outdir, "DriverProfiles.csv"), index=False)

        # ---------- legacy DrivingProfiles.csv (trip-sequence view) ----------
        drows = []
        for e in emps:
            for seq, t in enumerate(e.trips, 1):
                drows.append({
                    "employee_id": e.employee_id, "user_id": uid[e.employee_id],
                    "seq": seq, "trip_id": t.trip_id, "purpose": t.purpose,
                    "origin": t.origin, "destination": t.destination,
                    "depart_time": t.depart_time, "arrive_time": t.arrive_time,
                    "distance_miles": t.distance_miles, "duration_minutes": t.duration_minutes,
                    "fuel_type": t.fuel_type, "is_ev": t.is_ev,
                })
        pd.DataFrame(drows).to_csv(os.path.join(outdir, "DrivingProfiles_trips.csv"), index=False)

        # ---------- ChargingDemand.csv ----------
        crows = []
        for e in emps:
            daily_miles = sum(t.distance_miles for t in e.trips)
            work_dwell_h = max(e.depart_work_min - e.arrive_work_min, 0) / 60.0
            est = self.gen.charger.estimate(e, daily_miles, work_dwell_h)
            if est:
                est["user_id"] = uid[e.employee_id]; crows.append(est)
        cdf = pd.DataFrame(crows)
        cdf.to_csv(os.path.join(outdir, "ChargingDemand.csv"), index=False)

        # ---------- PopulationSummary.csv ----------
        val = Validator(emps); got = val.compute()
        fuel_counts = pd.Series([e.vehicle.fuel_type for e in emps]).value_counts()
        tele_counts = pd.Series([e.telework_status for e in emps]).value_counts()
        summ = {**got,
                "n_gas": int(fuel_counts.get("gas", 0)), "n_hybrid": int(fuel_counts.get("hybrid", 0)),
                "n_ev": int(fuel_counts.get("ev", 0)), "n_phev": int(fuel_counts.get("phev", 0)),
                "n_onsite": int(tele_counts.get("onsite", 0)),
                "n_hybrid_worker": int(tele_counts.get("hybrid", 0)),
                "total_daily_miles": round(sum(sum(t.distance_miles for t in e.trips)
                                               for e in emps), 1)}
        if not cdf.empty:
            summ["total_workplace_kwh_demand"] = round(cdf["workplace_kwh_demand"].sum(), 1)
            summ["total_ev_daily_kwh"] = round(cdf["total_energy_kwh"].sum(), 1)
            summ["n_needing_workplace_charging"] = int(cdf["workplace_charging_need"].sum())
        pd.DataFrame([summ]).to_csv(os.path.join(outdir, "PopulationSummary.csv"), index=False)

        # ---------- IncomeVehicleBreakdown.csv ----------
        idf = pd.DataFrame([{"income_bracket": e.income_bracket, "fuel_type": e.vehicle.fuel_type,
                             "is_ev": e.vehicle.is_ev} for e in emps])
        brk = (idf.groupby("income_bracket")
                  .apply(lambda g: pd.Series({
                      "n_employees": len(g),
                      "pct_gas": round(100 * (g.fuel_type == "gas").mean(), 1),
                      "pct_hybrid": round(100 * (g.fuel_type == "hybrid").mean(), 1),
                      "pct_phev": round(100 * (g.fuel_type == "phev").mean(), 1),
                      "pct_ev": round(100 * (g.fuel_type == "ev").mean(), 1),
                      "pct_ev_phev": round(100 * g.is_ev.mean(), 1)}),
                         include_groups=False)
                  .reset_index())
        brk.to_csv(os.path.join(outdir, "IncomeVehicleBreakdown.csv"), index=False)

        # ---------- ArchetypePopulation.csv (realised counts vs NHTS prevalence) ----------
        arows = []
        acnt = pd.Series([e.archetype_id for e in emps]).value_counts()
        for aid in sorted(self.arch.ids):
            nrow = int(acnt.get(aid, 0))
            arows.append({"ArchetypeID": aid, "ArchetypeName": self.arch.name(aid),
                          "NHTS_prevalence_pct": round(100 * self.arch.P[aid]["prevalence"], 2),
                          "n_generated": nrow,
                          "generated_pct": round(100 * nrow / len(emps), 2)})
        pd.DataFrame(arows).to_csv(os.path.join(outdir, "ArchetypePopulation.csv"), index=False)

        # ---------- reports ----------
        got, vdf = val.report()
        self._write_validation_md(outdir, vdf, emps)
        self._write_sim_md(outdir, summ, cdf)
        return summ, vdf

    # -- helpers --
    def _weight(self, e):
        # deterministic pseudo-weight per driver (households vary in weight);
        # placeholder scaled to NHTS person-weight magnitude for V2G-Sim column.
        h = abs(hash((self.seed, e.employee_id))) % 300 + 30
        return int(h)

    def _arch_desc(self, aid):
        try:
            defs = pd.read_csv(os.path.join(_ROOT, "ArchetypeDefinitions.csv"))
            row = defs[defs["ArchetypeID"] == aid]
            return row.iloc[0]["Description"] if len(row) else ""
        except Exception:
            return ""

    def _write_validation_md(self, outdir, vdf, emps):
        n_pass = (vdf["pass"] == "PASS").sum()
        lines = [f"# Validation Report — {self.office_name}",
                 f"Synthetic population validated against NHTS 2022 office-worker targets.  ",
                 f"**{n_pass}/{len(vdf)} metrics within tolerance.**\n",
                 "| Metric | Synthetic | NHTS Target | |Diff| | Tol | Result |",
                 "|---|---|---|---|---|---|"]
        for _, r in vdf.iterrows():
            lines.append(f"| {r['metric']} | {r['synthetic']} | {r['nhts_target']} "
                         f"| {r['abs_diff']} | {r['tolerance']} | {r['pass']} |")
        # archetype-distribution check
        lines += ["", "## Archetype mix (generated vs NHTS prevalence)",
                  "| Archetype | NHTS % | Generated % |", "|---|---|---|"]
        acnt = pd.Series([e.archetype_id for e in emps]).value_counts()
        for aid in sorted(self.arch.ids):
            gp = 100 * int(acnt.get(aid, 0)) / len(emps)
            lines.append(f"| {aid} {self.arch.name(aid)} | "
                         f"{100*self.arch.P[aid]['prevalence']:.1f} | {gp:.1f} |")
        lines += ["", "*Tolerances are Monte-Carlo sampling bands; small samples "
                  "(midday %, EV %, rare archetypes) fluctuate more.*"]
        open(os.path.join(outdir, "ValidationReport.md"), "w").write("\n".join(lines))

    def _write_sim_md(self, outdir, summ, cdf):
        L = [f"# Simulation Summary — {self.office_name}",
             f"- **Location:** {self.office_location}",
             f"- **Parking count (input):** {self.parking_count} vehicles",
             f"- **Random seed:** {self.seed}",
             f"- **Synthetic employees generated:** {summ['n_employees']}",
             f"- **Demographic archetypes:** {len(self.arch.ids)} (see ArchetypeDefinitions.csv)", "",
             "## Fleet composition",
             f"- Gas: {summ['n_gas']} · Hybrid: {summ['n_hybrid']} · "
             f"EV: {summ['n_ev']} · PHEV: {summ['n_phev']}",
             f"- EV+PHEV share: **{summ['ev_phev_share_pct']}%** (target 1.7%)", "",
             "## Travel",
             f"- Avg commute distance: **{summ['avg_commute_distance_mi']} mi** "
             f"(median {summ['median_commute_distance_mi']})",
             f"- Avg commute duration: **{summ['avg_commute_duration_min']} min** "
             f"(median {summ['median_commute_duration_min']})",
             f"- Avg trips/employee/day: **{summ['avg_daily_trips']}**",
             f"- Avg daily driver miles: **{summ['avg_daily_driver_miles']}**",
             f"- Morning direct commute: {summ['morning_direct_pct']}% · "
             f"Evening direct: {summ['evening_direct_pct']}% · "
             f"Midday trips: {summ['midday_trip_pct']}%", "",
             "## Workplace charging demand (EV/PHEV)"]
        if not cdf.empty:
            L += [f"- Vehicles needing workplace charging: "
                  f"**{summ.get('n_needing_workplace_charging',0)}**",
                  f"- Total workplace L2 kWh demand/day: "
                  f"**{summ.get('total_workplace_kwh_demand',0)} kWh**",
                  f"- Total EV/PHEV energy/day: {summ.get('total_ev_daily_kwh',0)} kWh"]
        else:
            L += ["- No EV/PHEV vehicles in this sample draw."]
        L += ["", "## Documented assumptions",
              "- Each driver is assigned a NHTS-derived demographic **archetype** first; "
              "income, telework, age, household, commute and trip-chaining are then drawn "
              "from that archetype's NHTS-conditional distribution. The archetypes partition "
              "the office-worker population, so the weighted mixture reproduces the national "
              "NHTS marginals (validated above).",
              f"- EV efficiency {EV_EFF_KWH_PER_MI} kWh/mi; PHEV {PHEV_EFF_KWH_PER_MI} "
              f"kWh/mi (electric range {PHEV_ELECTRIC_RANGE} mi/day).",
              f"- Charger availability: Home {LOC_PMAX_W['Home']} W (L1), "
              f"Work {LOC_PMAX_W['Work']} W (L2); other stops 0 W.",
              "- Charging demand = daily_miles × efficiency; workplace share = energy to "
              "complete the return-home leg, capped by L2 power × work dwell.",
              "- Vehicle fuel type is conditioned on household income; EV/PHEV ownership rises "
              "steeply with income (0.5% at <$50k to 5% at $200k+).",
              "- Geography: National NHTS distributions for behavioural shape (2022 public "
              "NHTS has no county/Bay-Area identifier)."]
        open(os.path.join(outdir, "SimulationSummary.md"), "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="LBNL synthetic driving-profile generator")
    ap.add_argument("--parking_count", type=int, default=250)
    ap.add_argument("--office_name", default="Lawrence Berkeley National Laboratory")
    ap.add_argument("--office_location", default="Berkeley, California / Alameda County")
    ap.add_argument("--tables", default=DEFAULT_TABLES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "outputs"))
    args = ap.parse_args()

    sim = LBNLSimulation(args.parking_count, args.office_name, args.office_location,
                         args.tables, seed=args.seed)
    sim.run()
    summ, vdf = sim.write_outputs(args.outdir)
    print(f"Generated {summ['n_employees']} synthetic employees "
          f"({args.office_name}); seed={args.seed}")
    print(f"Outputs -> {args.outdir}")
    print("\nValidation:")
    print(vdf.to_string(index=False))


if __name__ == "__main__":
    main()
