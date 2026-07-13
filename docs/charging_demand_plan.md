# Workplace EV Charging Demand Plan

Status: design only — no code written yet. This document specifies how
`scenarios/charging_demand.py` should convert the finalized mobility outputs
(`data/processed/synthetic_employees.parquet`,
`data/processed/synthetic_activity.parquet`, both as of commit `059386a`,
see `docs/model_status.md`) into per-visit workplace charging sessions and a
15-minute aggregate workplace load profile. It does not implement anything.

---

## 1. Objective

The mobility generator answers "where is each synthetic employee's vehicle,
and when." This stage answers a different question layered on top: "if some
fraction of these vehicles were EVs charging at the workplace, what would
that look like as a load curve." The two questions must stay separated —
mobility is a *behavioral* model validated against NHTS; charging demand is
a *scenario* model layered on top of it, whose assumptions (EV adoption,
vehicle efficiency, charger power) are inputs the user picks, not facts
recovered from data.

## 2. Separation of concerns

- **NHTS mobility generation remains unchanged.** `generator/sample.py` and
  `generator/activity.py` are not modified by this work. This stage only
  reads their finalized outputs.
- **EV ownership and charging assumptions are scenario inputs**, not
  properties inferred from the mobility data. Adoption rate, vehicle
  efficiency, charging efficiency, charger power, and charging strategy are
  all configurable parameters of a named scenario (§4), not constants
  derived from `synthetic_employees.parquet`.
- **Donor `vehicle_type`/`vehicle_fuel` must not determine EV status.**
  These two columns live on `synthetic_activity.parquet` (not on
  `synthetic_employees.parquet`) and are per-*leg* NHTS vehicle codes
  inherited from whichever real donor chain a leg was rescaled from
  (`docs/activity_generation_plan.md` §3). They are not stable even within
  one synthetic employee's own day — e.g. `SYN-00000001`'s legs carry
  `vehicle_type` 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0 across its nine
  legs, because different legs were rescaled from different donors' vehicle
  records. Treating that as "this employee's vehicle" would be both
  conceptually wrong (it describes the *donor's* historical vehicle, not
  the synthetic employee's) and structurally incoherent (it isn't even
  constant per employee). EV ownership is assigned independently in §5.
- **All charging logic belongs in `src/driving_profiles/scenarios/charging_demand.py`.**
  This is currently a documented placeholder
  (see file header) with no implementation; this plan is what fills it in
  next, once approved.
- **Remote sensing and parking-space estimation remain future work.** This
  stage assumes unlimited charger availability at the workplace (§4); it
  does not model how many physical charging spaces or parking spots exist.
  See §11.

## 3. Required inputs

### From `synthetic_employees.parquet` (5,000 rows, one per synthetic employee)

| Field | dtype | Role |
|---|---|---|
| `synthetic_employee_id` | str | Join key to `synthetic_activity.parquet`; primary key of every output table. |
| `cluster_id` | Int64 | Carried through to outputs; enables cluster-specific adoption rates later (§5). |
| `total_daily_miles` | float64 | Drives `miles_to_replenish` (§6). **52.3% null** in the current dataset (2,617/5,000) — see §6 for handling. |
| `used_household_vehicle` | bool | Drives non-driver exclusion (§5). 289/5,000 (5.8%) are `False`. |

`work_arrival_time`/`work_departure_time` on this table are the
employee's *summary* single arrival/departure (HHMM-encoded, see below) —
useful for context but not used directly for charging math, since a given
employee can have several distinct workplace visits in a day that this
summary pair collapses to one. The per-visit detail comes from
`synthetic_activity.parquet` instead.

### From `synthetic_activity.parquet` (15,201 rows, one per trip leg)

| Field | dtype | Role |
|---|---|---|
| `synthetic_employee_id` | str | Join key back to `synthetic_employees.parquet`. |
| `trip_number` | int64 | Orders an employee's legs; used to derive `workplace_visit_number` (below). |
| `arrival_time` | float64 | **HHMM-encoded**, not minutes-since-midnight — see unit note below. This *is* the workplace arrival time for rows where `is_workplace_arrival` is `True`. |
| `is_workplace_arrival` | bool | `True` on the leg whose destination is the workplace. This is the workplace-visit row. 6,242 such legs across 5,000 employees. |
| `is_workplace_departure` | bool | `True` on the leg whose *origin* is the workplace (i.e. the next leg after a visit). Not needed for the dwell calculation (see below) but useful for sanity-checking. |
| `workplace_dwell_minutes` | float64 | Already-computed dwell duration for that specific visit, in **true minutes** (not HHMM). Populated exactly on `is_workplace_arrival == True` rows. |

**Unit note — this is the single most important gotcha for this stage.**
`arrival_time`/`departure_time` (and `synthetic_employees.work_arrival_time`/
`work_departure_time`) are stored **HHMM-encoded** (e.g. `1226.258017`
decodes to 12:26, not "1226 minutes"), per
`generator/time_utils.py`'s `minutes_to_hhmm`/`hhmm_to_minutes` — the
generator jitters in true minutes internally but re-encodes to HHMM before
writing output, to stay consistent with NHTS's own convention. By contrast,
`workplace_dwell_minutes`, `duration`, `dwell_time_after`,
`total_driving_minutes`, and `commute_duration_minutes` are already **true
minutes**. Every timestamp this stage reads must be converted with the
existing `hhmm_to_minutes()` helper (reused, not reimplemented) before any
interval-bucketing or duration arithmetic; mixing the two unit systems would
silently produce nonsense (e.g. treating 12:26 as "1,226 minutes" instead of
746).

### Deriving per-visit arrival/departure without a second join

A visit's departure time does **not** need to be read off the
`is_workplace_departure` row that follows it — it is fully recoverable from
the arrival row alone:

```
departure_time_minutes = hhmm_to_minutes(arrival_time) + workplace_dwell_minutes
```

(Verified against the data: for `SYN-00000001`'s first visit,
`hhmm_to_minutes(814.513314) = 494.51`; `494.51 + 251.744702 = 746.26`,
which re-encodes to exactly `1226.258017` — the next leg's own
`departure_time`.) This keeps the per-visit extraction a single filter
(`is_workplace_arrival == True`) with no row-pairing logic.

### Handling multiple workplace visits

An employee can have more than one workplace visit in a day (e.g. leaving
for a midday errand and returning). Distribution in the current dataset:

| Visits/day | Employees |
|---:|---:|
| 1 | 4,300 |
| 2 | 406 |
| 3 | 175 |
| 4 | 50 |
| 5+ | 69 |

All `is_workplace_arrival == True` rows for a given `synthetic_employee_id`
are extracted, sorted by `trip_number` (equivalently, `arrival_time` once
decoded), and numbered `workplace_visit_number = 1, 2, 3, ...`. Every visit
becomes its own row in `ev_charging_sessions.parquet` (§7); §6 defines how
one employee's single daily energy need is allocated across their visits.

**Open-ended last visit of the day.** 296 of the 6,242 workplace-arrival
legs are the *last* recorded leg for their employee — the synthetic day's
trip chain simply ends at the workplace, so there is no subsequent leg to
close the dwell window and `workplace_dwell_minutes` is `NaN` for exactly
these rows (confirmed: `NaN` count on arrival rows exactly equals the
last-leg-of-day arrival count, 296). Baseline assumption: treat the vehicle
as remaining parked at the workplace through the end of the simulated day,
i.e. `available_dwell_minutes = 1440 - arrival_time_minutes` for this case
only. This is a documented MVP simplification (§11), not a data error.

## 4. Baseline scenario assumptions

A "scenario" is a named bundle of these parameters, so this module can be
called repeatedly with different assumptions without touching mobility
data. Defaults for the baseline:

| Parameter | Default | Notes |
|---|---|---|
| `scenario_name` | `"baseline"` | Carried into every output row for traceability across scenario runs. |
| `ev_adoption_rate` | `0.20` | Single project-wide rate for MVP (§5); cluster-specific rates are a documented future extension. |
| `vehicle_efficiency_kwh_per_mile` | `0.30` | Used in §6. |
| `charging_efficiency` | `0.90` | AC-to-battery conversion loss. |
| `charger_power_kw` | `7.2` | Level 2, single tier for MVP. |
| `interval_minutes` | `15` | Fixed at 96 intervals/day (§8). |
| `charging_strategy` | `"unmanaged_immediate"` | Vehicle draws full charger power from the moment it connects until either the battery need is met or it disconnects — no smoothing, no queuing, no site cap. |
| `charger_availability` | unlimited | Every simultaneously-present EV can charge; no contention modeled (§11). |
| `random_seed` | project default (`42`, via `utils/random_seed.get_seed`) | Reused, not reimplemented — same helper the mobility generator uses, so a caller who doesn't override the seed gets the same EV assignment on every run. |

These belong in a scenario config (e.g. a `charging:` block in
`config/default.yaml`, alongside the existing `scenarios:` placeholder) —
not hard-coded in the module — so a future run can vary adoption rate or
charger power without a code change. Config wiring is left to
implementation time; this plan only fixes the parameter set and defaults.

## 5. EV eligibility and assignment

**Eligible for workplace EV charging:** a synthetic employee who (a) is a
worker (`is_worker == True` — true for all 5,000 rows currently, but check
explicitly rather than assume), (b) drives to work
(`used_household_vehicle == True`), and (c) has at least one
`is_workplace_arrival == True` row in `synthetic_activity.parquet`. In the
current dataset this excludes 289 non-drivers (5.8%); a further ~small
number could in principle have `used_household_vehicle == True` but zero
recorded workplace-arrival legs (e.g. a fully remote day), and should be
excluded on the same basis — no vehicle presence at the workplace means no
charging opportunity, regardless of EV ownership.

**Why not filter on `total_daily_miles` here:** its 52.3% null rate is
*not* correlated with non-driving status (2,397 of the 2,617 nulls still
have `used_household_vehicle == True`) — it reflects a separate data-quality
gap in the mileage summary field, unrelated to whether the employee drives
or visits the workplace. Conflating the two would incorrectly exclude over
half the driving population from EV eligibility. Mileage nullness is
instead handled downstream, purely as an energy-calculation gap (§6), kept
distinct from eligibility so the summary metrics (§9) can separately report
"non-eligible (doesn't drive)" vs. "eligible but energy unknown."

**Assignment, reproducibly:** among eligible employees, draw EV ownership
as an independent Bernoulli(`ev_adoption_rate`) per employee, using
`utils/random_seed.get_rng(seed)` (the same helper already used by
`generator/sample.py`) seeded once per run. Employees are sorted by
`synthetic_employee_id` before drawing so assignment doesn't depend on row
order in the parquet file. The same seed always reproduces the same set of
EV employees; a different seed produces a different set (both required as
explicit checks, §10).

**Cluster-specific adoption rates (future):** `ev_adoption_rate` can become
a `dict[cluster_id, rate]` instead of a scalar, drawing each employee's
Bernoulli trial with their own cluster's rate. The eligibility filter and
RNG-per-employee approach are unchanged; only the probability parameter
becomes cluster-dependent. Deferred because the baseline scenario has no
stated cluster-differentiated adoption assumption yet.

## 6. Energy calculation

```
traction_energy_kwh = miles_to_replenish × vehicle_efficiency_kwh_per_mile
grid_energy_requested_kwh = traction_energy_kwh / charging_efficiency
```

`miles_to_replenish` for a *day* is the employee's `total_daily_miles`
(the whole day's driving, since the baseline models charging as replenishing
whatever the vehicle used that day, not a specific commute leg).

**Missing `total_daily_miles` (NaN, 2,617/5,000 = 52.3%):** the employee is
excluded from `ev_charging_sessions.parquet` entirely for that run — no
rows emitted for them, EV-assigned or not. This is a large fraction and
should be surfaced prominently, not buried: it's the single biggest
limitation of the baseline (§11) and is reported explicitly in the summary
(`employees_excluded_missing_mileage`, §9), separate from and not blended
into the adoption-rate or demand-served metrics.

**Zero miles (`total_daily_miles == 0`, 43 employees currently):** a valid,
kept case — an EV owner who simply didn't drive that day. Produces session
rows with `requested_energy_kwh = delivered_energy_kwh = unmet_energy_kwh = 0`
rather than being excluded, since zero is a real, informative answer (unlike
NaN, which is an unknown one).

**Multiple workplace visits:** one `grid_energy_requested_kwh` is computed
per employee per day (from their single `total_daily_miles`) and then
allocated sequentially across that employee's visits, in chronological
(`workplace_visit_number`) order — see §7 for the exact allocation loop.

**Short dwell windows / insufficient charging time / unmet energy:** a
visit's deliverable energy is capped by both what's still owed and what the
dwell window physically allows at `charger_power_kw`:

```
capacity_kwh = charger_power_kw × (available_dwell_minutes / 60)
delivered_energy_kwh = min(remaining_requested_kwh, capacity_kwh)
unmet_energy_kwh = remaining_requested_kwh − delivered_energy_kwh
```

Any energy still unmet after an employee's last visit of the day is not
carried anywhere further — it's recorded as unmet demand for that day and
rolled into the summary's `total_unmet_energy_kwh` / `percent_demand_served`
(§9). The baseline does not model next-day carryover or home charging as a
backstop; both are out of scope (§11).

## 7. Charging sessions

One row per EV-assigned employee per workplace visit in
`ev_charging_sessions.parquet`:

| Field | Description |
|---|---|
| `synthetic_employee_id` | Join key. |
| `cluster_id` | Carried through from `synthetic_employees.parquet`. |
| `workplace_visit_number` | 1-indexed, chronological, per §3. |
| `arrival_time_minutes` | `hhmm_to_minutes(arrival_time)`. |
| `departure_time_minutes` | `arrival_time_minutes + available_dwell_minutes` (§3/§6). |
| `available_dwell_minutes` | From `workplace_dwell_minutes`, or the end-of-day fallback (§3) when null. |
| `miles_to_replenish` | The employee's `total_daily_miles` (same value repeated on every visit row for that employee — informational, not per-visit-allocated). |
| `requested_energy_kwh` | Remaining requested energy *owed going into this visit* (§6 allocation). |
| `delivered_energy_kwh` | Energy actually delivered this visit. |
| `unmet_energy_kwh` | `requested_energy_kwh − delivered_energy_kwh` for this visit. |
| `charger_power_kw` | Scenario constant, repeated for traceability. |
| `charging_start_minutes` | `= arrival_time_minutes` (unmanaged strategy: charging begins immediately on connect). |
| `charging_end_minutes` | `charging_start_minutes + charging_duration_minutes`. |
| `charging_duration_minutes` | `(delivered_energy_kwh / charger_power_kw) × 60`, always `≤ available_dwell_minutes`. |
| `scenario_name` | Scenario identifier (§4). |
| `ev_adoption_rate` | Scenario parameter, repeated for traceability/filterability across scenario runs stacked in one file. |

**Allocation across multiple visits** (the loop implied by §6), run once
per employee in `workplace_visit_number` order:

```
remaining = grid_energy_requested_kwh   # employee's whole-day total
for each visit, in order:
    requested_energy_kwh[visit] = remaining
    capacity_kwh = charger_power_kw × (available_dwell_minutes[visit] / 60)
    delivered_energy_kwh[visit] = min(remaining, capacity_kwh)
    unmet_energy_kwh[visit] = requested_energy_kwh[visit] - delivered_energy_kwh[visit]
    remaining -= delivered_energy_kwh[visit]
```

Interim visits that fully cover `remaining` leave `0` unmet and `remaining`
at `0` for every subsequent visit that day (which then correctly produce
all-zero rows); only a visit that exhausts its own dwell capacity before
`remaining` reaches zero shows nonzero `unmet_energy_kwh`, and that nonzero
amount only ever appears on the final visit of a day in practice (since
`remaining` cannot become positive again once it hits zero).

## 8. 15-minute load profile

`ev_charging_load_profile.parquet` is a fixed 96-row (24h × 4/hr) table:

| Field | Description |
|---|---|
| `interval_start_minutes` | `0, 15, 30, ..., 1425`. |
| `interval_end_minutes` | `interval_start_minutes + 15`. |
| `connected_ev_count` | Count of sessions whose **dwell window** (`arrival_time_minutes` → `departure_time_minutes`) overlaps this interval — i.e. physically present, whether or not still drawing power. |
| `charging_ev_count` | Count of sessions whose **charging window** (`charging_start_minutes` → `charging_end_minutes`) overlaps this interval — i.e. actively drawing power. |
| `charging_power_kw` | `charging_ev_count × charger_power_kw` (unmanaged strategy: every actively-charging vehicle draws full power, no throttling). |
| `interval_energy_kwh` | Sum, across sessions overlapping this interval, of `charger_power_kw × (overlap_minutes / 60)`. |
| `cumulative_energy_kwh` | Running `cumsum` of `interval_energy_kwh` across the 96 intervals, in order. |

**Proration for partial overlap:** for a session with charging window
`[start, end)` and an interval `[t, t+15)`:

```
overlap_minutes = max(0, min(end, t + 15) − max(start, t))
```

Because unmanaged charging draws constant full power for the entire
`charging_start_minutes`→`charging_end_minutes` span, summing
`charger_power_kw × overlap_minutes/60` across all 96 intervals for one
session reconstructs exactly that session's `delivered_energy_kwh` — this
is what the "interval energy reconciles with session energy" validation
check (§10) verifies structurally, not just approximately.

**Scope simplification:** since `available_dwell_minutes` is already capped
at end-of-day (§3) and all decoded arrival/departure times fall within
`[0, 1440)`, no session can cross midnight in the baseline — the 96-interval
day is self-contained with no wraparound logic needed. Documented as a
baseline simplification (§11), not a general guarantee for future
multi-day or overnight-charging extensions.

## 9. Summary metrics

`ev_charging_summary.csv`, one row (or one column of labeled values) per
scenario run:

| Metric | Definition |
|---|---|
| `total_employees` | `len(synthetic_employees)`. |
| `charging_eligible_employees` | Passes §5 eligibility (driving worker with ≥1 workplace visit). |
| `ev_employees` | Eligible employees assigned an EV (§5). |
| `ev_adoption_rate_actual` | `ev_employees / charging_eligible_employees` — reported alongside the configured `ev_adoption_rate` as a sanity check they're close. |
| `total_requested_energy_kwh` | Sum of `requested_energy_kwh` across all sessions. |
| `total_delivered_energy_kwh` | Sum of `delivered_energy_kwh`. |
| `total_unmet_energy_kwh` | Sum of `unmet_energy_kwh`. |
| `percent_demand_served` | `total_delivered_energy_kwh / total_requested_energy_kwh`. |
| `peak_charging_power_kw` | `max(charging_power_kw)` across the load profile. |
| `peak_time_minutes` | `interval_start_minutes` at the peak. |
| `max_simultaneous_charging_vehicles` | `max(charging_ev_count)`. |
| `avg_session_energy_kwh` | Mean `delivered_energy_kwh` across sessions. |
| `avg_charging_duration_minutes` | Mean `charging_duration_minutes` across sessions. |
| `employees_excluded_missing_mileage` | Eligible+EV-assigned employees with `total_daily_miles` null, excluded per §6 (expect ~52% given current data — reported explicitly, not folded into `ev_employees`). |

## 10. Outputs

- `data/processed/ev_charging_sessions.parquet` — schema per §7.
- `data/processed/ev_charging_load_profile.parquet` — schema per §8.
- `data/processed/ev_charging_summary.csv` — schema per §9.

**Future Excel sheets**, following the existing pattern in
`utils/export_excel.py` (one workbook, a README sheet listing every other
sheet, data sheets as Excel Tables, a key-value sheet for summaries —
`_write_data_sheet`/`_write_key_value_sheet`/`_write_readme_sheet`, reused
rather than reinvented) into the already-scaffolded
`reports/xlsx/charging_demand/` directory:

- **Charging Scenario** — key-value sheet of the scenario's config values (§4), so a reviewer can see at a glance which assumptions produced the attached numbers.
- **Charging Sessions** — data sheet, `ev_charging_sessions.parquet` as an Excel Table.
- **Charging Load Profile** — data sheet, `ev_charging_load_profile.parquet` as an Excel Table.
- **Charging Summary** — key-value sheet, `ev_charging_summary.csv`.

## 11. Validation

Checks to implement in `tests/`, following the existing project convention
of one test module per pipeline stage (e.g. `tests/test_generator.py`,
`tests/test_validation_activity.py`):

- **EV assignment count** — `ev_employees` count is close to
  `charging_eligible_employees × ev_adoption_rate` (binomial tolerance, not
  exact equality).
- **Reproducibility with fixed seed** — same seed → byte-identical EV
  assignment set across two runs.
- **Different seed changes assignments** — a different seed produces a
  materially different assignment set (not identical).
- **No negative values** — every energy, duration, count, and power column
  is `≥ 0` across all three outputs.
- **Non-driving employees receive no charging demand** — no
  `synthetic_employee_id` with `used_household_vehicle == False` appears in
  `ev_charging_sessions.parquet`.
- **`delivered_energy_kwh ≤ requested_energy_kwh`** — per session row.
- **`delivered_energy_kwh ≤` dwell-window charging capacity** — per session
  row, `delivered_energy_kwh ≤ charger_power_kw × available_dwell_minutes/60`.
- **`charging_duration_minutes ≤ available_dwell_minutes`** — per session row.
- **Load profile has exactly 96 intervals** spanning `[0, 1440)` with no
  gaps or overlaps.
- **Load power never negative** — `charging_power_kw ≥ 0` every interval.
- **Interval energy reconciles with session energy** — summing
  `interval_energy_kwh` across all 96 intervals equals
  `total_delivered_energy_kwh` from §9, within floating-point tolerance.
- **Output files are created** — all three paths in §10 exist and are
  non-empty after a run.
- **Mobility outputs are not modified** — `synthetic_employees.parquet` and
  `synthetic_activity.parquet` are byte-identical (or checksum-identical)
  before and after a charging-demand run, confirming this stage is
  read-only with respect to mobility data.

## 12. MVP and future extensions

**MVP (first implementation):** exactly the baseline defaults in §4 — one
scalar adoption rate, one vehicle efficiency, one charger power tier,
unlimited chargers, unmanaged immediate charging, fixed 15-minute
intervals. No config file wiring is required to ship the MVP; a single
hard-coded `Scenario` dataclass/namedtuple with these defaults, overridable
by keyword arguments, is sufficient — config-file plumbing can follow once
the shape is proven.

**Future extensions (explicitly out of scope here):**
- Charger constraints (finite chargers per site, contention).
- Queues / wait times when chargers are full.
- Managed charging (load-shifting, off-peak scheduling, V1G/V2G).
- Site power caps (aggregate demand limits, load shedding).
- Solar generation coordination (net-load-following charging).
- Company-specific inputs (site-specific charger counts, adoption rates, shift schedules).
- Remote-sensing-derived parking capacity (replacing the "unlimited chargers" assumption in §4 with an actual physical constraint).
