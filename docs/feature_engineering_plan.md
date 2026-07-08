# Feature Engineering Plan

Status: design only — no code written yet. This document specifies what
`features/build_features.py` should compute from `trips_clean.parquet` (the
`clean.py` output: one row per trip, joined with person/household/vehicle
attributes). It does not implement anything.

Goal: transform cleaned trip records into **employee-level driving behavior
profiles** that can later be clustered into archetypes, used to generate
synthetic employees, and ultimately feed a workplace EV charging demand
estimate.

All NHTS variable names below match what `ingest.py` loads and `clean.py`
preserves (see `docs/data_requirements.md` §2–3). Nothing here introduces a
variable not already present in the cleaned dataset.

---

## 1. Unit of analysis: person, not trip

`trips_clean.parquet` is trip-level (one row per trip). Feature engineering
must **aggregate up to one row per person** (`HOUSEID` + `PERSONID`) before
any clustering happens. Reasons:

- **The clustering question is about people, not trips.** Workplace charging
  demand is a per-employee question: does this employee drive to work, how
  long is their car parked there, how much range did their day consume. A
  trip is a sub-event inside that story, not the unit the business question
  is asked about. Clustering on trips would produce clusters of "kinds of
  trips" (e.g., short errands vs. long commutes), not "kinds of employees" —
  the wrong output for generating a synthetic workforce.
- **Trips within a person are not independent observations.** A person with
  a 5-stop chain contributes 5 correlated trip rows; a person who drove
  straight home contributes 1. Clustering at the trip level would silently
  overweight people with more trips and treat within-person correlation as
  between-group signal, biasing the clusters.
- **Downstream consumers need a person-shaped object.** Synthetic employee
  generation samples *employees*, each with one commute distance, one
  arrival/departure time, one vehicle-availability window. That object only
  exists after aggregation — it can't be assembled from a bag of unlinked
  trip rows.
- **NHTS is a one-day diary, so "person-level" here means "person, using
  their one surveyed day as a proxy for a typical day."** NHTS collects
  exactly one travel day per person (`TRAVDAY`/`TDAYDATE` on the trip file).
  There is no way to observe day-to-day variability for a given individual
  in this dataset. The person-level profile below is therefore a single-day
  snapshot per person, not a multi-day behavioral average — clustering
  treats the surveyed day as representative, and this assumption should be
  stated explicitly wherever profiles are used downstream (`methodology.md`
  should record it once a clustering approach is chosen).
- Profiles should be built **only from weekday records** (`TRAVDAY`), since
  workplace charging demand is a weekday phenomenon; weekend travel diaries
  are out of scope for this feature set.

The grain for every feature below is therefore **one row per
`HOUSEID` + `PERSONID`** (a person's single weekday travel diary), built by
aggregating that person's trip rows and carrying through their static
person/household/vehicle attributes.

---

## 2. Feature categories

### A. Demographics

| Feature | NHTS source variable(s) | Calculation | Why it matters for workplace charging |
|---|---|---|---|
| Age | `R_AGE` | Pass through | Segments schedule regularity and commute-distance tendency; needed to build a realistic synthetic workforce age distribution. |
| Age band | `R_AGE` | Bucket (e.g., <25, 25–34, 35–44, 45–54, 55–64, 65+) | Coarser, cluster-friendly categorical version of age; reduces noise from single-year age. |
| Sex | `R_SEX` | Pass through | Segmentation/realism input for the synthetic workforce; not expected to directly drive charging demand but needed so generated employees match NHTS demographic mix. |
| Education | `EDUC` | Pass through (ordinal) | Correlates with occupation type and commute regularity; segmentation input for realistic persona generation. |
| Household relationship | `R_RELAT` | Pass through | Distinguishes household head vs. other members; feeds `LIF_CYC`-style household-role context (see B) rather than driving charging demand on its own. |

`R_RACE`/`R_HISP` are in the cleaned dataset but are **not** proposed as
clustering features here — they carry no direct behavioral mechanism for
commute or charging behavior, and using them as clustering inputs risks
encoding demographic bias into the synthetic workforce. Keep them available
for post-hoc representativeness checks (does the generated workforce match
NHTS's demographic composition?) rather than as inputs to the archetype
model itself.

### B. Household characteristics

| Feature | NHTS source variable(s) | Calculation | Why it matters for workplace charging |
|---|---|---|---|
| Household size | `HHSIZE` | Pass through | Proxy for scheduling constraints (caregiving, shared errands) that shape trip-chain complexity around the commute. |
| Household income | `HHFAMINC` | Pass through (ordinal bracket) | Segmentation input — income correlates with commute distance and housing location choice, both of which shape charging-relevant commute patterns. |
| Household vehicle count | `HHVEHCNT` | Pass through | Numerator for vehicle-per-driver ratio (below); a household with more vehicles than drivers has less schedule contention over any single vehicle. |
| Driver count | `DRVRCNT` | Pass through | Denominator for vehicle-per-driver ratio. |
| Vehicles per driver | `HHVEHCNT`, `DRVRCNT` | `HHVEHCNT / DRVRCNT` (household-level, joined onto each person) | Direct proxy for whether the commute vehicle is "dedicated" to this person or shared/contended — shared vehicles are less reliably available at the workplace on any given day. |
| Number of workers in household | `WRKCOUNT` | Pass through | Households with multiple workers may have competing schedules/vehicle needs, relevant when combined with vehicles-per-driver. |
| Life cycle stage | `LIF_CYC` | Pass through (categorical) | Proxy for household caregiving obligations (e.g., presence of children) that predict multi-stop chains (daycare drop-off, errands) which shorten or interrupt workplace dwell time. |
| Urban/rural context | `URBAN`/`URBRUR`, `URBANSIZE` | Pass through (categorical) | Segmentation context for calibrating synthetic employees to a specific site type (dense urban core vs. suburban/rural workplace). |
| Metro area size | `MSACAT`, `MSASIZE` | Pass through (categorical) | Same as above — lets later scenario work target a specific metro size class. |
| Census division/region | `CENSUS_R`, `CENSUS_D` | Pass through (categorical) | Regional segmentation for site-specific or region-specific demand scenarios. |

`HOMEOWN` is intentionally **excluded** here (see §3) — its main behavioral
relevance is home-charging access, which is a charging-behavior question,
not a driving-behavior one.

### C. Work commute behavior

| Feature | NHTS source variable(s) | Calculation | Why it matters for workplace charging |
|---|---|---|---|
| Employee flag | `WORKER`, `PRMACT`, `PAYPROF` | Boolean: `WORKER = 01` (optionally tightened with `PRMACT`/`PAYPROF`, per the open decision in `docs/data_requirements.md` §7) | Defines the population in scope at all — non-workers have no workplace to charge at. |
| Work location arrangement | `WRKLOC` | Pass through (categorical: onsite / hybrid / telework-only / drives for work) | Determines whether a workplace parking spot — and therefore a charging opportunity — exists on a given day. `WRKLOC = 03` (full telework) implies zero workplace charging demand for that person. |
| Telework frequency | `WKFMHM22` | Pass through (days/week worked from home) | Scales workplace charging opportunity down from 5 days/week for hybrid workers; needed to convert a per-day profile into a weekly demand estimate. |
| Usual commute mode | `WRKTRANS` | Pass through (categorical) | Filters to vehicle-driving commute modes — the population that could plausibly charge an EV at work in the first place. |
| Weekly work hours proxy | `EMPLOYMENT2` | Pass through (categorical: full-time/part-time) | Part-time schedules imply shorter or less regular workplace dwell time, affecting available charging windows. |
| Commute distance (survey estimate) | `GCDWORK` | Pass through | Direct driver of how much battery range the commute consumes — the core input to any later charging-need estimate. |
| Commute distance (trip-based) | `TRPMILES` on the outbound work-purpose leg(s) (`WHYTRP1S = 10` or `WHYTRP90`) | Sum `TRPMILES` over the person's outbound work-purpose trip(s) for the day | Independent, road-network-based cross-check against `GCDWORK` (which is great-circle/straight-line); the two can be compared or reconciled during modeling. |
| Commute arrival time | `STRTTIME`/`ENDTIME` of the trip arriving at the work-purpose destination | `ENDTIME` of the leg where `WHYTO` (or `WHYTRP1S`) = work | Marks the start of the workplace charging window. |
| Commute departure time | `STRTTIME` of the trip leaving the work-purpose location | `STRTTIME` of the leg where `WHYFROM` = work | Marks the end of the workplace charging window (subject to D's midday-departure feature). |
| Work dwell time | `DWELTIME` on the work-purpose destination leg | Pass through (minutes) | Direct proxy for maximum available charging session duration at the workplace. |
| Weekday flag | `TRAVDAY` | Boolean: weekday vs. weekend | Used as a filter (§1), not a clustering input — kept here only for completeness/traceability of the day being profiled. |

### D. Daily mobility behavior

| Feature | NHTS source variable(s) | Calculation | Why it matters for workplace charging |
|---|---|---|---|
| Daily trip count | `TRIPID` | Count of trip rows per person-day (optionally excluding `LOOP_TRIP` trips, which don't change location) | Overall mobility intensity; more trips generally means more chances the commute vehicle leaves the workplace mid-day. |
| Trip chain pattern | `WHYFROM`→`WHYTO` (or `WHYTRP1S`) sequence | Ordered sequence per person-day, collapsed to a simplified alphabet (home / work / other) for the first pass | Captures whether the day is a direct home↔work commute or a multi-stop chain; multi-stop chains around the commute reduce or fragment the workplace charging window. |
| Total daily VMT | `TRPMILES` (or `VMT_MILE`) | Sum over trips where `TRPTRANS` is a driving mode | Total daily range consumed; combined with commute distance, indicates how much of the day's driving happens outside the commute itself (errands, other driving) that also draws down battery state of charge before or after the workplace charging window. |
| Total daily driving time | `TRVLCMIN` | Sum over driving-mode trips (`TRPTRANS`) | Complements VMT; distinguishes short/congested driving from long/highway driving for the same distance. |
| Midday workplace departure flag | `WHYFROM`/`WHYTO` sequence, `STRTTIME` | Boolean: does the person leave and return to the work-purpose location within the same day (a trip out of and back into the work leg) | Indicates the vehicle isn't parked at the workplace continuously — interrupts what would otherwise be one long charging session into two shorter ones. |
| Time-of-day distribution | `STRTTIME`, `ENDTIME` across all trips | Departure/arrival time histogram per person-day | Rush-hour clustering signal; helps validate that generated synthetic commute times match real arrival/departure peaks. |

### E. Vehicle availability behavior

| Feature | NHTS source variable(s) | Calculation | Why it matters for workplace charging |
|---|---|---|---|
| Vehicles per driver (household) | `HHVEHCNT`, `DRVRCNT` | Same derived ratio as in B, restated here as the vehicle-availability lens | A ratio near or above 1 suggests the commute vehicle is reliably the same one day to day; a ratio well below 1 suggests contention, which matters for whether "this person's car" is a stable concept at all. |
| Commute vehicle identifier | `VEHID`/`TRPHHVEH` on the work-purpose trip leg | Pass through (which household vehicle carried the commute trip) | Links the person-level commute profile to a specific vehicle record, needed later to attach vehicle attributes (deferred — see §3) once fuel-type/EV logic is introduced. |
| Vehicle household-day overlap | `VEHID`/`TRPHHVEH` + `STRTTIME`/`ENDTIME` across **all household members'** trips for the day | For the vehicle used on the person's commute, compute whether any other household member's trip uses the same `VEHID` that day, and if so, whether the time windows overlap | Flags shared-vehicle contention: if another driver needs the same car mid-day, the vehicle may not actually be parked at the workplace for the whole dwell-time window computed in C. Must be computed at the household+vehicle level (joining across all persons in the household), not per person alone. |
| Vehicle annual mileage | `ANNMILES` (vehicle file) | Pass through, joined via the commute `VEHID` | Proxy for how heavily-used/primary this vehicle is within the household; a low-mileage vehicle is more likely a secondary/backup vehicle with a less predictable daily schedule. |

Note: E deliberately stops at *availability* (is the vehicle physically at
the workplace and free to charge) and does not touch *whether* it is
electric or *how* it would charge — that's excluded per §3.

---

## 3. Features intentionally excluded for now

These are present in the cleaned dataset (or derivable from it) but are
**out of scope for this feature set**, per `docs/methodology.md`'s deferred
scope. Including them now would bake unvalidated assumptions into the
clustering/generation stages before there's a reason to:

- **EV ownership** — nothing in NHTS 2022 identifies whether a household
  vehicle is currently an EV in a way this project should treat as ground
  truth for a *future* synthetic workforce; EV *penetration* is a scenario
  parameter (5%/10%/20%) to be applied later in `scenarios/`, not a feature
  of today's observed behavior.
- **Charging behavior** — session timing, power level, home-vs-work
  charging split, and charging-station availability are all downstream
  outputs of the demand model this feature set feeds, not inputs to it.
  `HOMEOWN` (home-charging feasibility) is excluded for the same reason —
  it's a charging-location question, not a driving-behavior one.
- **Vehicle fuel type assumptions** — `VEHFUEL` and `VEHTYPE` exist in the
  cleaned dataset and are cheap to carry through, but no feature here should
  branch on them. Driving-behavior clustering (commute distance, dwell time,
  vehicle availability) should be computed identically regardless of what a
  vehicle is powered by; fuel-type/vehicle-type logic is explicitly reserved
  for `src/driving_profiles/scenarios/charging_demand.py`.

Carrying `VEHID` (E) forward without resolving it to a fuel type is
intentional: it's the join key that will let the scenarios phase attach an
EV-penetration assumption to a specific vehicle-availability window later,
without needing to redo the driving-behavior feature engineering.

---

## 4. How these features support downstream stages

**Clustering.** The person-level feature vector (A–E) is what gets
clustered into a small number of employee archetypes (e.g., "short-commute
onsite full-timer with a dedicated vehicle," "long-commute hybrid worker
with a shared household vehicle," "midday-errand commuter with a fragmented
workplace dwell window"). Commute distance, arrival/departure times, work
dwell time, and vehicle availability (C, D, E) are the primary axes that
should separate clusters, since they map most directly onto charging
opportunity; demographics and household characteristics (A, B) mostly serve
to make each cluster's *composition* realistic rather than to define the
cluster boundaries themselves.

**Synthetic employee generation.** Once clusters are defined, each cluster
becomes a template: synthetic employees are generated by sampling from the
within-cluster distribution of every feature above (e.g., draw a commute
distance from cluster 2's `GCDWORK` distribution, a dwell time from its
`DWELTIME` distribution, an age from its `R_AGE` distribution), rather than
from the marginal distribution across the whole population. This is what
lets the generator produce an arbitrarily large synthetic workforce that
still reproduces the *joint* structure NHTS shows (e.g., long commutes
co-occurring with earlier arrivals and longer dwell times), not just
correct marginals for each feature independently.

**Charging demand estimation.** The scenarios phase will take each
synthetic employee's cluster-derived profile — work arrival/departure time
(C) defining the charging window, work dwell time (C) bounding session
length, midday-departure flag (D) determining whether that window is
interrupted, commute distance/VMT (C, D) determining energy needed per
session, and vehicle availability (E) determining whether the vehicle is
even present to charge — and overlay the deferred EV-penetration and
fuel-type assumptions (§3) on top. This feature set is designed so that
step can be added later without recomputing any driving-behavior feature:
everything a charging-demand model needs about *when the vehicle is at
work and how long it stays* is already captured here.
