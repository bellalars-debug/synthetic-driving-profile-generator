# Daily Activity Generation Plan

Status: design only — no code written yet. This document specifies how
`generator/activity.py` should expand `data/processed/synthetic_employees.parquet`
(the `sample.py` output: one row per synthetic employee, summary statistics
only) into a full daily trip-chain activity profile per synthetic employee,
for workplace EV charging demand estimation. It supersedes the sketch in
`docs/synthetic_generation_plan.md` §5 now that `sample.py` is implemented
and the actual shape of `synthetic_employees.parquet` is known; it does not
implement anything.

---

## 1. Objective

`synthetic_employees.parquet` gives each synthetic employee a *summary* day:
one arrival time, one departure time, a trip count, a total mileage. That is
enough to describe an employee in the aggregate, but a workplace EV charging
demand model needs to know the actual *shape* of the day — when the vehicle
leaves home, whether it goes straight to work or runs an errand first,
whether it leaves work and comes back at midday, and when it finally parks
for the night. None of that is recoverable from a single arrival/departure
pair.

The objective of this stage is to convert each synthetic employee's static
summary row into an ordered **daily trip chain**: a sequence of legs, each
with a departure time, arrival time, purpose, distance, and duration, that
is internally consistent (times don't overlap or run backwards, distances
sum to the employee's own total) and consistent with that employee's own
drawn attributes (arrival time, departure time, trip count, total miles).
The chain's *shape* — how many legs, in what purpose order — is not invented
from scratch; it is borrowed from a real NHTS respondent's chain in the same
behavioral cluster and then rescaled to the synthetic employee's own values
(§3). This keeps the output physically plausible by construction rather than
by post-hoc validation alone.

This document is scoped to **driving activity only**. No EV ownership,
charging behavior, or fuel-type assumption is introduced here — that is
deferred to `scenarios/charging_demand.py` (§7), consistent with
`docs/feature_engineering_plan.md` §3 and `docs/clustering_plan.md` §7.

---

## 2. Inputs

Three processed tables, each playing a distinct role:

| Table | Grain | Role |
|---|---|---|
| `data/processed/synthetic_employees.parquet` | One row per synthetic employee | The **target**. Supplies the summary values each reconstructed chain must be rescaled to match: `cluster_id` (which donor pool to draw from, §3), `work_arrival_time`, `work_departure_time`, `trips_per_day`, `number_of_stops`, `total_daily_miles`, `commute_distance_survey_miles`, `total_driving_minutes`, plus vehicle-availability context (`vehicles_per_driver`, `vehicle_per_driver_adequate`, `used_household_vehicle`). `synthetic_employee_id` is the join key the output activity table hangs off of. `source_houseid`/`source_personid` (the real respondent each synthetic employee was resampled from in `sample.py`) are also present and are useful here as a *starting point* for donor lookup (§3), though the donor chain ultimately selected need not be that same respondent's. |
| `data/interim/trips_clean.parquet` | One row per real trip leg | The **structural reference library**. Supplies real, internally-consistent chain *shapes* — ordered `WHYFROM`/`WHYTO` purpose sequences, timestamped with `STRTTIME`/`ENDTIME`, each leg carrying `TRPMILES`, `TRVLCMIN`, `DWELTIME`, and vehicle linkage (`VEHID`/`TRPHHVEH`, and via the vehicle file, `VEHTYPE`/`VEHFUEL`). This table is read only to borrow chain *structure* — it is never a source of new per-person facts about a specific synthetic employee, since those already came from `synthetic_employees.parquet`. |
| `data/processed/employee_clusters.parquet` | One row per real `HOUSEID`+`PERSONID` | The **bridge** between the other two. `trips_clean.parquet` has no `cluster_id` of its own; this table is the only place a real trip chain's `HOUSEID`+`PERSONID` can be looked up against a `cluster_id`. Joining `trips_clean.parquet` to this table by `HOUSEID`+`PERSONID` is what lets donor-chain selection be restricted to real people in the same behavioral archetype as the synthetic employee being expanded (§3). |

---

## 3. Trip-chain reconstruction strategy

**Use real NHTS trip patterns as chain templates, not generated from
scratch.** Every real respondent in `trips_clean.parquet` already has a
physically coherent day — their chain doesn't arrive at work before it
departs, and their leg count matches their own stop pattern, because it's a
real diary. Generating a synthetic chain by drawing purposes, times, and
distances as independent random values risks producing a day that is
individually plausible-looking per field but incoherent as a whole (e.g. a
trip that ends after the next one starts). Borrowing a real chain's shape
and rescaling it avoids that failure mode structurally, rather than relying
on validation (§6) to catch it after the fact.

**How trip chains are selected:**

1. Join `employee_clusters.parquet` to `trips_clean.parquet` on
   `HOUSEID`+`PERSONID`, then group by person to reconstruct each real
   respondent's full ordered chain for their surveyed day (all their
   `TRIPID` rows, ordered by `SEQ_TRIPID`/`STRTTIME`), tagged with that
   person's `cluster_id`. This produces a **donor pool**: one candidate
   chain per real clustered respondent, grouped by `cluster_id`.
2. For a given synthetic employee, restrict the donor pool to donors sharing
   that employee's `cluster_id`, then pick the donor whose chain shape most
   closely matches the synthetic employee's own `trips_per_day` and
   `number_of_stops` (e.g. exact match first, widening to ±1 if the exact
   combination has no donor).
3. **Fall back gracefully** when no reasonably close donor exists in that
   cluster (a real risk for cluster/trip-count combinations that are sparse
   in a ~7,800-person sample): widen the match tolerance first; if that
   still fails, fall back to a minimal synthesized chain (direct
   home→work→home, with the employee's own arrival/departure/distance
   values) rather than failing the run for that employee. This means every
   synthetic employee gets a chain, but not every chain is guaranteed to be
   donor-derived — that should be tracked (e.g. a `chain_source` flag:
   donor-matched vs. fallback) so §6 validation can check whether fallback
   usage is rare enough to ignore or common enough to mean the donor pool
   needs to be larger (a bigger synthetic `n` doesn't help here — the donor
   pool is bounded by real NHTS respondents per cluster, not by how many
   synthetic employees are being generated).

**Why sample donors from the same behavioral cluster, specifically:** a
cluster (`docs/clustering_plan.md` §1) is defined by exactly the fields that
also describe chain shape and timing — commute distance, arrival/departure
time, daily mobility, vehicle availability. A chain shape is not independent
of those fields: a long-commute, low-stop archetype's real chains look
different from a short-commute, multi-stop archetype's. Pooling donors
across clusters (or picking a donor at random from the whole population)
would let a chain shape typical of one archetype get attached to a synthetic
employee whose own drawn attributes belong to a different archetype —
reintroducing exactly the joint-structure-breaking problem that
`docs/synthetic_generation_plan.md` §4 already identified for independent
per-column sampling, just relocated to the chain-shape step instead of the
attribute-sampling step.

**Rescaling the selected donor chain** (never copy it verbatim — verbatim
reuse would just be resampling a real record again):

- **Times**: shift the donor's work-arrival and work-departure legs so they
  land exactly on the synthetic employee's own `work_arrival_time` /
  `work_departure_time` (already drawn in `sample.py`); shift every other
  leg in the chain by the same offset so relative spacing between legs is
  preserved.
- **Distances**: scale each leg's `TRPMILES` proportionally so the chain's
  total matches the synthetic employee's `total_daily_miles`, anchoring the
  work-purpose leg(s) specifically to `commute_distance_survey_miles` (the
  more authoritative, independently-drawn value for that leg).
- **Durations**: scale `TRVLCMIN` consistently with the distance rescaling
  (or recompute from the rescaled time deltas directly) so a leg's implied
  speed stays plausible rather than drifting arbitrarily from the rescaling.
- **Trip count / stop count**: already fixed by donor selection; no further
  reconciliation needed beyond the match tolerance in step 2 above.

---

## 4. Daily activity profile structure

Output: `data/processed/synthetic_activity.parquet`, long format — one row
per synthetic-employee-leg (the trip-level analog of `trips_clean.parquet`
for the synthetic population). Columns:

| Column | Description |
|---|---|
| `synthetic_employee_id` | FK to `synthetic_employees.parquet`. |
| `trip_number` | 1-indexed leg sequence within that employee's day (ordered by departure time). |
| `departure_time` | Rescaled leg start time (HHMM, matching `work_arrival_time`/`work_departure_time`'s existing encoding for consistency). |
| `arrival_time` | Rescaled leg end time (HHMM). |
| `trip_purpose` | Destination purpose for the leg, derived from the donor's `WHYFROM`/`WHYTO` (or `WHYTRP1S`), collapsed to a small alphabet (home / work / other) for the first pass, consistent with `docs/feature_engineering_plan.md` §2.D's simplified trip-chain-pattern treatment. |
| `distance` | Rescaled leg distance (miles), from `TRPMILES`. |
| `duration` | Rescaled leg travel time (minutes), from `TRVLCMIN`. |
| `dwell_time_after` | Minutes spent at the leg's destination before the next leg departs (from `DWELTIME`, rescaled consistently with the time shift) — this is what §5 uses to derive parking duration at work specifically. |
| `donor_houseid`, `donor_personid` | The real respondent (`HOUSEID`+`PERSONID`) the chain shape was borrowed from, for traceability during development — mirrors the same dev-traceability rationale `sample.py` already applies to `source_houseid`/`source_personid`, and lets a validation failure be traced back to a specific donor chain. |
| `chain_source` | `"donor"` or `"fallback"` (§3 step 3) — whether this employee's chain came from a matched real donor or a synthesized minimal chain. |
| `vehicle_type`, `vehicle_fuel` | Descriptive-only fields carried through from the donor leg's `VEHTYPE`/`VEHFUEL` (via `VEHID`/`TRPHHVEH`), where available. These describe **the donor's real vehicle**, not an assumption about the synthetic employee's own vehicle — no EV/fuel-type logic should read these columns; they exist only as inspectable provenance alongside `donor_houseid`/`donor_personid`, consistent with `docs/feature_engineering_plan.md` §3 deferring fuel-type logic entirely to `scenarios/charging_demand.py`. |

Employee-level vehicle-availability context (`vehicles_per_driver`,
`vehicle_per_driver_adequate`, `used_household_vehicle`) is **not**
duplicated onto every leg row here — it already lives in
`synthetic_employees.parquet`, one row per employee, and a consumer needing
both should join on `synthetic_employee_id` rather than have it repeated
redundantly across every leg.

---

## 5. Workplace parking behavior

Workplace dwell time is the direct determinant of charging opportunity — a
vehicle can only charge while it's parked at the workplace — so it must be
derived explicitly from the reconstructed chain rather than left as an
implicit byproduct of §4's leg table:

- **Workplace arrival time**: the `arrival_time` of the leg whose purpose is
  work. By construction (§3's rescaling step), this equals the synthetic
  employee's own `work_arrival_time` from `synthetic_employees.parquet` —
  the chain reconstruction is anchored to that value, not the other way
  around, so this is a consistency check as much as a derivation.
- **Workplace departure time**: the `departure_time` of the leg that leaves
  the work-purpose location, similarly anchored to `work_departure_time`.
- **Parking duration**: `dwell_time_after` on the work-purpose leg —
  equivalently, workplace departure time minus workplace arrival time when
  there is exactly one continuous work-purpose dwell segment that day.

**Fragmented dwell windows.** Not every employee's day has one continuous
work-purpose dwell segment — a donor chain (or a synthetic employee's own
`number_of_stops` > the minimum) can include a midday departure from and
return to the work-purpose location (`docs/feature_engineering_plan.md`
§2.D's "midday workplace departure flag" — the vehicle leaves work and comes
back within the same day). When the reconstructed chain contains this
pattern, workplace parking duration must be computed as **the set of all
contiguous dwell windows at the work-purpose location that day**, not a
single arrival-to-departure span — each window is an independently bounded
charging opportunity, and collapsing them into one span would overstate how
long the vehicle is continuously available in any single window (materially
relevant to whether a charging session can complete, not just how much total
dwell time exists). This is why `dwell_time_after` is computed per-leg in
§4 rather than only as a single derived employee-level pair of timestamps —
the per-leg detail is what a fragmented-day case needs to be represented at
all.

---

## 6. Validation

Before `synthetic_activity.parquet` is used downstream (§7), compare it
against the real trip chains it was derived from — `trips_clean.parquet`,
restricted to the clustered population via `employee_clusters.parquet` —
the same way `docs/synthetic_generation_plan.md` §6 validates
`synthetic_employees.parquet` against `employee_clusters.parquet` directly.
All comparisons should be run **within cluster**, not only on the pooled
population, since a synthetic population could match the pooled distribution
while being systematically wrong within any individual cluster:

- **Trip counts** — compare the distribution (not just the mean) of legs per
  synthetic employee-day against real `trips_per_day`/`number_of_stops`,
  per cluster. A generator that reproduces the mean but flattens the
  variance would understate how often a workplace dwell window is
  interrupted (§5's fragmented-window case).
- **Commute distances** — compare the rescaled work-purpose leg distance
  distribution against real `commute_distance_survey_miles`/`TRPMILES`
  (mean, median, and tail behavior specifically, since long-commute
  outliers matter more for peak charging-energy estimates than for central
  tendency).
- **Arrival/departure distributions** — compare the full time-of-day
  histogram of reconstructed `departure_time`/`arrival_time` for the
  work-purpose leg against real NHTS arrival/departure times, per cluster.
  The *shape* of the peak (sharp rush-hour spike vs. wide spread) is what
  determines whether a downstream charging model sees a coincident load
  peak or a smoothed one — matching only the mean arrival time would miss
  this.
- **Daily miles** — compare total reconstructed chain distance per
  synthetic employee-day against real `total_daily_miles`.
- **Travel duration** — compare total reconstructed chain `duration` against
  real `total_driving_minutes`, and per-leg `duration` against real
  `TRVLCMIN` for legs of comparable purpose/distance.
- **Chain-shape plausibility** (structural, not distributional) — every
  reconstructed chain should have non-decreasing leg times (`departure_time`
  of leg *n+1* ≥ `arrival_time` of leg *n*), non-negative distances/durations
  that sum to the employee's own totals within a small tolerance, and a
  purpose sequence that starts and ends at a plausible location (typically
  home). This is a correctness check on the reconstruction logic itself
  (§3), distinct from the distributional checks above, and should fail loud
  (not just get logged) if violated, since a chain that fails this check is
  not usable input to §7 regardless of how well the pooled distributions
  compare.
- **Fallback rate** — report what fraction of synthetic employees fell back
  to a synthesized minimal chain (§3 step 3) rather than a donor-matched one,
  overall and per cluster. A high fallback rate in any cluster signals that
  cluster's donor pool is too sparse at that `trips_per_day`/`number_of_stops`
  combination for reconstruction fidelity to be trusted there.

Any systematic mismatch found here should feed back into §3's donor-matching
tolerance or rescaling logic, not be treated as acceptable approximation
error — validation is a gate before `synthetic_activity.parquet` is trusted
by §7.

---

## 7. Future charging integration

`synthetic_activity.parquet`, joined to `synthetic_employees.parquet` on
`synthetic_employee_id`, is designed to carry exactly what
`scenarios/charging_demand.py` needs and nothing more: per-leg timing and
distance, and — critically for charging specifically — the workplace parking
windows derived in §5. No EV ownership, fuel type, or charging behavior is
assumed anywhere in this document; every column above is a statement about
*where a vehicle goes and when*, never about what kind of vehicle it is
(the `vehicle_type`/`vehicle_fuel` columns in §4 are explicitly donor
provenance metadata, not a synthetic employee's own vehicle assumption).
This mirrors the same deferred-scope boundary `docs/feature_engineering_plan.md`
§3 and `docs/synthetic_generation_plan.md` §8 already established.

Concretely, the future integration step will:

1. Take `synthetic_activity.parquet` (plus `synthetic_employees.parquet`) as
   a fixed input, unchanged by which EV scenario is being run.
2. Apply an EV-penetration draw (flag X% of `synthetic_employee_id`s as EV
   drivers) and a fuel-type/efficiency assumption per EV-flagged employee —
   entirely within `scenarios/charging_demand.py`, requiring no change to
   activity generation.
3. For each EV-flagged employee, convert their workplace parking window(s)
   from §5 into one or more candidate charging sessions: session start/end
   bounded by the parking window, energy required derived from that
   employee's commute distance/VMT (§4's `distance` column, aggregated) and
   the assumed vehicle efficiency. A fragmented dwell day (§5) naturally
   yields multiple shorter candidate sessions instead of one long one.
4. Aggregate candidate sessions across the synthetic workforce, weighted by
   how many employees' parking windows overlap at each time of day, into a
   workplace charging **load curve** — the deliverable this whole pipeline
   has been building toward.

Because steps 2–4 are scenario parameters applied after the fact, re-running
a different EV penetration or fuel-type assumption never requires
regenerating `synthetic_activity.parquet` — the same activity table can be
reused across every scenario, exactly as `docs/clustering_plan.md` §7 and
`docs/synthetic_generation_plan.md` §8 anticipated when they deferred this
same overlay out of every earlier stage.
