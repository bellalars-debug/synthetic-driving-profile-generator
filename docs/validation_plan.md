# Synthetic Population Validation Plan

Status: design only — no validation code written yet. This document
specifies how to check whether `synthetic_employees.parquet` and
`synthetic_activity.parquet` (the `generator/sample.py` and
`generator/activity.py` outputs) realistically reproduce the NHTS 2022
travel behavior patterns they were generated from, before
`scenarios/charging_demand.py` is built on top of them. It does not
implement anything; see `docs/synthetic_generation_plan.md` §6 and
`docs/activity_generation_plan.md` §6 for the validation sketches this
document supersedes with concrete metrics, datasets, and pass/fail
thresholds.

**Relationship to existing tests.** `tests/test_cluster.py`,
`tests/test_generator.py`, and `tests/test_activity.py` already cover
*structural/mechanistic* correctness — reproducibility given a fixed seed,
chronological leg ordering, no real NHTS IDs in output, jitter leaving NaNs
alone, fallback triggering correctly, etc. Those are unit tests and stay in
`tests/`. This plan is about a different question those tests cannot
answer: **does the generated population look statistically like the real
population it claims to model?** That requires comparing distributions
across the full generated dataset, not asserting properties of a handful of
constructed test fixtures.

---

## 1. Validation philosophy

### Why validation is required before charging demand modeling

`scenarios/charging_demand.py` (not yet built) will take arrival time,
dwell time, and commute VMT per synthetic employee and convert them
directly into charging session timing and energy estimates. Every one of
those inputs is a number this pipeline invented — resampled from a cluster,
jittered, or rescaled from a donor chain (`docs/synthetic_generation_plan.md`
§4, `docs/activity_generation_plan.md` §3). If the generation process has
silently distorted a distribution (flattened the arrival-time peak,
shrunk the commute-distance tail, broken the joint correlation between
commute distance and dwell time), that distortion propagates directly into
a charging demand curve with no further opportunity to catch it — the
demand model has no independent way to know its inputs are wrong. Validating
now, before that model exists, is the only point in the pipeline where the
generated population can be checked against ground truth (NHTS) directly;
once EV/fuel-type assumptions are layered on top, there is no "real"
charging-demand baseline to compare against at all.

Concretely, validation is what turns "the code ran without raising an
exception" into "the numbers this code produced are trustworthy inputs to a
demand estimate." Those are different claims, and the existing unit test
suite only supports the first one.

### What "successful synthetic generation" means

Per `docs/synthetic_generation_plan.md` §1, the design goal was never
"reproduce NHTS exactly" — a synthetic population that matched NHTS
row-for-row would defeat the privacy and arbitrary-population-size goals
that motivated generation in the first place. Success instead means:

- **Distributional fidelity**: for every feature that matters to charging
  demand, the synthetic population's distribution (not just its mean) is
  statistically indistinguishable from the real distribution it was drawn
  from, *within each cluster* — matching only the pooled/marginal
  distribution while being wrong within a cluster is exactly the failure
  mode `docs/synthetic_generation_plan.md` §6 already warns about.
- **Joint-structure preservation**: correlations between features (long
  commute co-occurring with earlier arrival and a dedicated vehicle) survive
  generation — this is the entire justification for resampling whole rows
  and donor chains rather than sampling each column independently
  (`sample.py`'s module docstring, `docs/synthetic_generation_plan.md` §4).
  A validation pass that only checks marginals per column would miss this
  failure mode entirely.
- **Structural plausibility**: every synthetic employee-day is internally
  consistent (arrival before departure, distances non-negative and summing
  to the employee's own total) — already covered by existing unit tests,
  but worth re-confirming at the full-population scale rather than on
  hand-built fixtures.
- **No new individual re-identification**: jitter and rescaling should have
  moved every synthetic employee's continuous fields measurably away from
  the single real respondent's row it was drawn from (§1's privacy
  motivation) — a *different* success criterion from distributional
  fidelity, and in tension with it (more jitter = better privacy separation,
  worse fidelity to the source row), so both need to be checked, not just
  one.
- **Scalability under population-size change**: since `n` is a free
  parameter (`docs/synthetic_generation_plan.md` §3), fidelity should hold
  whether `n=500` or `n=50,000` — the cluster weights and within-cluster
  distributions don't change with `n`, only the number of draws does, so
  validation run at the default `n=5,000` should be sanity-checked once at
  another `n` to confirm nothing size-dependent slipped in.

### Limitations of validating against NHTS 2022

Every comparison in this plan is ultimately "does synthetic data resemble
NHTS 2022," which bounds what validation can actually prove:

- **One surveyed day per respondent, not a travel history.** NHTS captures
  a single weekday diary per person. `total_daily_miles`,
  `trips_per_day`, etc. are that one day's behavior, not a personal
  average — a respondent having an unusually light or heavy travel day
  (a doctor's appointment, working from home the next day) is
  indistinguishable in the source data from someone whose behavior is
  always like that. The clustering and generation pipeline necessarily
  treats each surveyed day as if it were representative of that person's
  "typical" day, which NHTS's design does not actually guarantee. Any
  validation showing a good match to NHTS 2022 confirms internal
  consistency of the pipeline, not that the synthetic population reflects
  *stable, repeatable* employee behavior over time (e.g., day-to-day
  variance for a single real person, telework schedules, seasonal
  commuting changes) — that variance simply does not exist in a one-day
  survey to validate against.
- **A fixed, non-representative sample.** NHTS 2022's ~7,800 raw employee
  respondents (2,434 clustered — `docs/synthetic_generation_plan.md` §2)
  is a national sample, not a sample matched to any specific employer or
  site this project might eventually model. A synthetic population that
  matches NHTS 2022 well is validated against *national* commuting
  patterns; it says nothing about whether that matches a specific
  workplace's actual employee base (geography, industry, income mix all
  affect commute distance and vehicle availability). This is a
  generalizability limitation inherent to the data source, not something
  validation can fix — it should be stated as an assumption boundary (§8),
  not treated as a validation failure.
- **NHTS 2022 was fielded during/just after COVID-19 disruption to
  commuting patterns** (elevated telework, atypical peak-hour spread
  relative to pre-2020 NHTS waves). Validating against NHTS 2022 confirms
  the synthetic population matches *this* survey wave, not necessarily
  "normal" or projected-future commuting behavior — worth flagging as an
  assumption if this pipeline's output is later compared against
  pre-pandemic benchmarks or used for forward-looking demand projections.
- **Small cluster count limits how fine-grained a validation claim can
  be.** With `k=2` currently chosen (§3 below), cluster-level validation
  has exactly two archetypes to check — real behavioral heterogeneity
  within a 4,176-employee cluster (e.g., a wide range of commute distances
  all landing in "cluster 0") won't be caught by a cluster-level summary
  statistic; only the full within-cluster distribution comparisons (§2)
  catch that.
- **Donor-pool sparsity for activity generation is bounded by NHTS, not by
  synthetic population size.** `docs/activity_generation_plan.md` §3 notes
  the donor pool for chain reconstruction is fixed at however many real
  NHTS respondents fall in a given cluster/trip-count combination — a
  validation pass showing good aggregate fidelity does not mean every
  individual chain shape is well-supported; the fallback rate (§4, §5) is
  the specific metric that surfaces this.

---

## 2. Synthetic employee population validation

**Source dataset:** `data/processed/employee_clusters.parquet`, filtered to
`cluster_id.notna()` (the same population `sample.py` draws from — comparing
against the *unfiltered* `employee_features.parquet` would incorrectly
include non-workers and no-commute workers who were never eligible to be
resampled in the first place).

**Synthetic dataset:** `data/processed/synthetic_employees.parquet`.

**Join key for "which real rows fed which synthetic rows":**
`source_houseid`/`source_personid` (present per `sample.py`'s module
docstring — a deliberate deviation from the plan's original privacy
recommendation, kept for dev traceability). Useful for spot-checking
individual resample-jitter behavior, not for the population-level
comparisons below (those should compare the full source distribution
against the full synthetic distribution, not resampled row against its
own source row).

All comparisons in this section should be run **both pooled and per
`cluster_id`** — pooled catches gross sampling bugs (e.g., wrong cluster
proportions); per-cluster is what actually validates distributional
fidelity, per §1.

### Demographics

| Metric | Source column(s) | Synthetic column(s) | Comparison method | Acceptable validation metric |
|---|---|---|---|---|
| Age distribution | `age` | `age` | Overlaid histogram/KDE + two-sample KS test, per cluster | KS test `p > 0.05` (fail to reject "same distribution"); visually, overlapping histograms with no systematic shift in mean/spread |
| Age band | `age_band` | `age_band` | Categorical proportion comparison (bar chart) + chi-square goodness-of-fit | Chi-square `p > 0.05`; every band's synthetic share within ~2 percentage points of source share |
| Worker status | `worker_status` | `worker_status` | Proportion comparison | Expected to be constant/near-constant (`docs/clustering_plan.md` §3 notes this is near-constant post-filter) — flag as a *non-informative* check, not a real validation gate |
| Income categories | `household_income_bracket` | `household_income_bracket` | Categorical proportion comparison (ordinal-aware — compare cumulative distribution, not just per-bracket share) + chi-square | Chi-square `p > 0.05`; cumulative distribution (e.g., % below/above median bracket) within ~3 percentage points |
| Household size | `household_size` | `household_size` | Discrete distribution comparison (bar chart of counts) + KS test | KS test `p > 0.05`; mean/median within source's own standard error |

### Household characteristics

| Metric | Source column(s) | Synthetic column(s) | Comparison method | Acceptable validation metric |
|---|---|---|---|---|
| Household vehicle count | `household_vehicle_count` | `household_vehicle_count` | Discrete distribution comparison + KS test | KS test `p > 0.05` |
| Vehicles per driver | `vehicles_per_driver` | `vehicles_per_driver` (a `JITTER_FEATURES` column — see `sample.py:71`) | Overlaid histogram + KS test, per cluster | KS test `p > 0.05`; specifically check the jittered distribution hasn't drifted the mean (jitter is mean-zero Gaussian noise, so mean should be ~unchanged, only spread slightly widened) |
| Vehicle availability indicators | `vehicle_per_driver_adequate`, `used_household_vehicle` | same | Proportion comparison (both are booleans, `sample.py:83-86` — passed through unjittered from the resampled row) | Proportions should match source **within cluster** almost exactly (±1 percentage point) since these are not jittered — any larger gap indicates a resampling bug, not natural variation |

### Commute behavior

| Metric | Source column(s) | Synthetic column(s) | Comparison method | Acceptable validation metric |
|---|---|---|---|---|
| Commute distance | `commute_distance_survey_miles` | same (jittered) | Overlaid histogram/KDE + KS test, per cluster; separately compare tail behavior (e.g., 90th/95th percentile) since long commutes matter disproportionately for energy estimates (`docs/synthetic_generation_plan.md` §6) | KS test `p > 0.05`; 90th-percentile values within ~10% of source |
| Commute duration | `commute_duration_minutes` | same (jittered) | Overlaid histogram + KS test, per cluster | KS test `p > 0.05` |
| Work arrival time | `work_arrival_time` (HHMM) | same (jittered, clamped `[0, 2359]` — `sample.py:222-223`) | Time-of-day histogram (not mean) + KS test on `hhmm_to_minutes`-converted values, per cluster; specifically check the rush-hour peak shape survives (sharp vs. smoothed), not just the mean | KS test `p > 0.05`; peak histogram bin (e.g., 15-minute buckets) within ~15% of source's peak-bin share |
| Work departure time | `work_departure_time` (HHMM) | same | Same as arrival time | Same as arrival time |

### Daily mobility behavior

| Metric | Source column(s) | Synthetic column(s) | Comparison method | Acceptable validation metric |
|---|---|---|---|---|
| Trips per day | `trips_per_day` | same (`COUNT_FEATURES`, rounded + clamped `≥1` after jitter — `sample.py:88,216-219`) | Discrete distribution comparison (not just mean — a generator that flattens variance here understates dwell-window interruption, per `docs/synthetic_generation_plan.md` §6) + KS test | KS test `p > 0.05`; variance within ~20% of source variance |
| Total daily miles | `total_daily_miles` | same | Overlaid histogram + KS test, **run only on the non-null subset** (see §5 — this column has structural NaNs that must not be dropped-then-ignored, they need their own validation) | KS test `p > 0.05` on non-null subset; missingness rate itself validated separately (§5) |
| Total driving minutes | `total_driving_minutes` | same | Same as total daily miles | Same as total daily miles |
| Number of stops | `number_of_stops` | same (`COUNT_FEATURES`) | Discrete distribution comparison + KS test | KS test `p > 0.05` |
| Average trip distance | `average_trip_distance_miles` | same | Overlaid histogram + KS test, non-null subset only (derived from `total_daily_miles` — same missingness, see §5) | KS test `p > 0.05` on non-null subset |

---

## 3. Cluster validation

**Do not assume `k=2` is correct.** The current `cluster_evaluation.csv`
(from `cluster.py`'s own evaluation run) shows:

| k | inertia | silhouette |
|---|---|---|
| 2 | 18,511.0 | **0.417** |
| 3 | 16,313.8 | 0.355 |
| 4 | 14,507.9 | 0.362 |
| 5 | 12,890.5 | 0.327 |
| 6 | 11,746.0 | 0.302 |
| 7 | 10,842.7 | 0.248 |
| 8 | 9,950.8 | 0.267 |

`k=2` has the highest silhouette score among the candidates evaluated, which
is why it was selected — but 0.417 is a **moderate, not strong**, silhouette
score (above ~0.5 is usually considered reasonably well-separated; above
~0.7 strong). This is exactly the situation `docs/clustering_plan.md` §6
anticipated when it insisted on a domain-interpretability review, not just
picking the max-silhouette `k` mechanically — that review does not appear to
have been documented anywhere yet, and this validation pass is the place to
do it before trusting cluster-conditioned sampling any further.

### Determining whether clusters represent meaningful driving patterns (not assumed)

1. **Profile each cluster's centroid/summary statistics** on the *raw,
   unstandardized* `CONTINUOUS_FEATURES` (`cluster.py:69-79`: commute
   distance, commute duration, arrival/departure time, trips per day, total
   daily miles, total driving minutes, number of stops, vehicles per
   driver) — mean, median, and IQR per cluster, both in
   `employee_clusters.parquet` and reproduced in `synthetic_employees.parquet`.
   A meaningful cluster split should produce clusters whose profiles a
   non-technical reader could label (e.g., "cluster 0: shorter commute, more
   stops" vs. "cluster 1: longer commute, direct trip") — per
   `docs/clustering_plan.md` §6's domain-interpretability criterion. If the
   two clusters differ mainly in one feature by a small margin with heavy
   overlap in every other feature, that is evidence `k=2` is capturing a
   weak or arbitrary split rather than a real behavioral distinction, even
   though it has the best silhouette among the candidates tried.
2. **Check separation, not just existence, on each primary feature
   individually** — overlaid histograms of commute distance, arrival time,
   etc., split by cluster. Look for genuinely bimodal/separated
   distributions, not just a shifted mean with heavy overlap (a shifted
   mean with 80% distributional overlap is a weak archetype distinction in
   practice, even if it nudges a centroid-distance-based silhouette score
   upward).
3. **Cross-check against a method-agnostic view.** Per
   `docs/clustering_plan.md` §5's recommendation, run a hierarchical
   clustering dendrogram on the same preprocessed feature matrix
   (`preprocess_features` output) as a diagnostic — does the dendrogram's
   natural cut point agree with `k=2`, or does it suggest KMeans's spherical
   assumption forced a 2-way split where the real structure is different
   (e.g., 3+ groups, or no clear structure at all)? This check has not yet
   been run and should be, given the moderate silhouette score above.
4. **If clusters do not hold up under 1-3**, this is a finding for this
   validation pass to surface, not something to fix by picking a different
   `k` unilaterally — `docs/clustering_plan.md` §6 is explicit that the
   final call should combine quantitative metrics with domain review, so a
   negative finding here should be reported back to that decision, with the
   silhouette-vs-`k` table above and the qualitative profile differences
   as the evidence, rather than the validation pass silently re-clustering.

### Cluster population proportions

| Metric | Source | Synthetic | Comparison method | Acceptable validation metric |
|---|---|---|---|---|
| Cluster share | `employee_clusters.parquet` cluster_id value_counts (normalized) | `synthetic_employees.parquet` cluster_id value_counts (normalized) | Direct proportion comparison — should match by construction for default (non-overridden) sampling (`docs/synthetic_generation_plan.md` §6: "mainly a check that sampling code has no bug") | Synthetic share within the largest-remainder rounding error of the target weight (effectively exact for `n=5,000`); current run: cluster 0 ≈ 83.5% source / 83.5% synthetic (4,176/5,000), cluster 1 ≈ 16.5% source / 16.5% synthetic (824/5,000) |

### Cluster feature distributions / commute patterns by cluster / mobility patterns by cluster

Repeat every metric in §2's four tables **filtered to each `cluster_id`
separately** (source vs. synthetic, same cluster). This is the actual
"does synthetic preserve archetype behavior" check — pooled-population
comparisons in §2 can pass even if within-cluster distributions are wrong,
if cluster proportions happen to offset the error (this is the specific
failure mode `docs/synthetic_generation_plan.md` §6 calls out, and why this
section exists separately from §2).

Specifically for the two current clusters, validate that these known
source-population differences (to be confirmed/re-measured, not assumed)
survive into the synthetic population:

- Whichever cluster has the longer mean commute distance in the source data
  should still have the longer mean commute distance in the synthetic data,
  by a comparable margin (not just "still longer," but a similar effect
  size).
- Whichever cluster has more stops/higher `trips_per_day` in the source
  should still show that pattern synthetically.
- The `total_daily_miles`/`total_driving_minutes` missingness rate
  differs materially by cluster in the source data (§5 measures this
  precisely) — this per-cluster missingness *rate* is itself part of "the
  cluster's archetype" and should be preserved, not just the non-null
  values' distribution.

---

## 4. Activity profile validation

**Source dataset:** `data/interim/trips_clean.parquet`, restricted to
respondents with a `cluster_id` via `employee_clusters.parquet` (i.e., the
same `build_donor_legs` restriction `activity.py:196-267` applies — the
donor pool itself, not the full unfiltered trip file, since that's the
actual "ground truth" chains generation drew from).

**Synthetic dataset:** `data/processed/synthetic_activity.parquet`.

All comparisons should be run **per cluster** for the same reason as §3, and
**per `chain_source`** (`"donor"` vs. `"fallback"`) — a fallback chain is
built from the employee's own summary values directly
(`build_fallback_chain`, `activity.py:497-542`), not borrowed structure, so
it is a structurally different generation process and mixing it with
donor-derived chains in one comparison would obscure whether either
mechanism individually is producing realistic output.

### Trip chain structure

| Metric | Source | Synthetic | Comparison method | Acceptable validation metric |
|---|---|---|---|---|
| Trips (legs) per employee-day | Donor-pool legs per person (`trips_clean` grouped by `HOUSEID`+`PERSONID`) | Legs per `synthetic_employee_id` in `synthetic_activity.parquet` | Discrete distribution comparison + KS test, per cluster | KS test `p > 0.05` |
| Number of stops | `summarize_donor_chains`'s `stop_count` (`activity.py:270-286`) definition applied to real donor pool | Same definition applied to synthetic chains | Discrete distribution comparison + KS test, per cluster | KS test `p > 0.05` |
| Trip sequence length | Same as trips-per-employee above (a chain's leg count *is* its sequence length here) | Same | — (duplicate of trips-per-employee; keep as one metric, not two, to avoid double-counting in a summary report) | — |
| Home→work→home pattern share | Proportion of donor chains whose purpose sequence is exactly `home, work, home` (or starts/ends at home generally) vs. more complex chains | Same proportion in synthetic chains, split by `chain_source` (fallback chains are *always* this exact 2-leg pattern by construction — `build_fallback_chain` — so this metric is only meaningful for `chain_source == "donor"`) | Categorical proportion comparison | Donor-sourced synthetic share within ~5 percentage points of the real donor-pool share |

### Travel behavior

| Metric | Source | Synthetic | Comparison method | Acceptable validation metric |
|---|---|---|---|---|
| Trip distance distribution | `TRPMILES` per leg, donor pool | `distance` per leg | Overlaid histogram/KDE + KS test, per cluster and per leg purpose (home/work/other) | KS test `p > 0.05` |
| Trip duration distribution | `TRVLCMIN` per leg, donor pool | `duration` per leg | Overlaid histogram + KS test, per cluster and per leg purpose; also check implied speed (`distance/duration`) stays within `MIN_PLAUSIBLE_SPEED_MPH`/`MAX_PLAUSIBLE_SPEED_MPH` (`activity.py:96-97`) for the whole synthetic population, not just the legs the rescaling logic itself guards | 100% of legs within the plausible-speed band (this one is a hard structural check, not a KS test — any violation is a bug, not natural variation) |
| Departure time distribution | `STRTTIME` (converted to minutes), donor pool | `departure_time` (converted via `hhmm_to_minutes`) | Time-of-day histogram + KS test, per cluster | KS test `p > 0.05` |
| Arrival time distribution | `ENDTIME`, donor pool | `arrival_time` | Same as departure time | Same as departure time |

### Workplace behavior

| Metric | Source | Synthetic | Comparison method | Acceptable validation metric | Why it matters most for EV charging |
|---|---|---|---|---|---|
| Workplace arrival timing | `ENDTIME` of the work-purpose leg, donor pool | `arrival_time` where `is_workplace_arrival` | Time-of-day histogram + KS test, per cluster; **this is a consistency check as much as a distribution check** — by construction (`rescale_chain_times`), synthetic workplace arrival should equal the employee's own `work_arrival_time` from `synthetic_employees.parquet` exactly (within floating-point tolerance) | 100% match to source employee's own `work_arrival_time` (structural check) — this is *not* a KS-test question, since it's deterministic by construction; a mismatch would indicate a rescaling bug | Defines when a charging session could begin — the single most load-bearing timing input to the eventual demand curve |
| Workplace departure timing | `STRTTIME` of the leg departing work, donor pool | `departure_time` where `is_workplace_departure` | Same structural check against `work_departure_time` | 100% match (structural), or documented explanation via `rescale_chain_times`'s fallback-to-arrival-offset path (`activity.py:342-346` — used when no leg follows arrival or the target departure is missing/before arrival) | Defines when a session must end — bounds maximum session duration |
| Workplace dwell periods | `DWELTIME` on work-purpose legs, donor pool (aggregated to one value per continuous dwell window) | `workplace_dwell_minutes`, non-null subset (see §5 for why this is NaN on ~4.4% of work legs — end-of-chain artifact, not an error) | Overlaid histogram + KS test, per cluster; separately validate the **fragmented-dwell-window case** (`docs/activity_generation_plan.md` §5) — count how often a synthetic employee has more than one work-purpose leg in a day, and confirm each such case reports multiple independent dwell windows rather than one collapsed span | KS test `p > 0.05` on total dwell-per-employee-day; fragmented-window share within a few percentage points of real donor pool's fragmented-window share | Directly bounds maximum charging session length — the dwell-time distribution shape (not just its mean) determines what fraction of arriving vehicles could plausibly complete a full charge before departure, which is the crux of a workplace charging demand curve |

**Which activity metrics matter most for EV charging, ranked:** (1)
workplace arrival time distribution — determines coincident charging-start
load and peak timing; (2) workplace dwell duration distribution — determines
how much of the arriving fleet can complete a session at all, and directly
gates session-count/energy-delivered estimates; (3) commute distance (from
§2/§3, carried into the work-purpose leg's `distance`) — determines energy
required per session; (4) fragmented-dwell-window rate — determines whether
"one long session" or "two shorter sessions" is the right model for a given
employee, which changes both feasibility and total deliverable energy per
day. Trip counts/chain shape away from the workplace matter much less
directly (they mostly just need to be *plausible*, per the structural
checks, since they don't feed the charging model at all).

---

## 5. Employees with missing driving summary features

This section validates the pattern identified during initial data
inspection: `total_daily_miles`, `total_driving_minutes`, and
`average_trip_distance_miles` are missing together for **2,637 of 5,000**
synthetic employees in the current default-seed run. **Missingness here is
not automatically treated as an error** — `build_features.py:258-261`
computes these columns by summing over driving-mode trips only
(`DRIVING_MODE_TRPTRANS_CODES`) with `min_count=1`, so a worker who commuted
by a non-driving mode (walk/bike/transit) that day legitimately has no
driving miles to report. The purpose of this section is to confirm that
documented intent actually holds at the full-population level, and to
surface the one place it interacts questionably with a downstream step
(activity generation), rather than assume either "it's fine" or "it's a
bug" without checking.

### Does missingness match the original NHTS population?

| Check | Method | Current observed result (this run) | Acceptable validation metric |
|---|---|---|---|
| Pooled missingness rate | `synthetic_employees.parquet`'s null rate on `total_daily_miles` vs. `employee_clusters.parquet` (clustered population only) null rate | Synthetic 2,637/5,000 = 52.7% vs. source 1,264/2,434 = 51.9% | Within ~2-3 percentage points (resampling noise at `n=5,000` is expected to produce some deviation from the exact source rate) |
| Per-cluster missingness rate | Same comparison, split by `cluster_id` | Cluster 0: synthetic 2,362/4,176 = 56.6% vs. source 56.1%; cluster 1: synthetic 275/824 = 33.4% vs. source 30.9% | Within ~3-5 percentage points per cluster (smaller cluster 1 has a smaller sample, so more sampling variance is expected and acceptable there specifically) |
| Missingness co-occurrence | Confirm all three columns are null together, never partially | All 2,637 affected rows have exactly 3/3 columns null; the remaining 2,363 rows have 0/3 null | Should be exactly 100% co-occurring — `average_trip_distance_miles` is arithmetically derived from `total_daily_miles` (`build_features.py:281-282`), so any row with one null and not the others indicates a pipeline inconsistency, not natural variation |
| Preservation through jitter | Confirm `sample_employees`'s jitter step left every source-NaN as NaN rather than jittering it to some non-NaN value | Code inspection confirms `not_na` masking (`sample.py:210-214`); should be re-confirmed empirically against the full 5,000-row output, not just trusted from reading the code | 100% — any synthetic row with a non-null value where its source row was null (traceable via `source_houseid`/`source_personid`) indicates a jitter-masking bug |

### Does activity generation produce plausible chains for these employees?

This is the more important open question, and the answer is genuinely
mixed — worth stating plainly rather than glossing over:

- **Trip counts**: plausible. Donor selection (`select_donor`,
  `activity.py:289-319`) matches on `trips_per_day`/`number_of_stops`, which
  are *not* null for these 2,637 employees (only the mileage/duration
  summary columns are null) — so these employees get a normally-matched
  donor chain with a realistic leg count, same as anyone else in their
  cluster.
- **Distances**: plausible in isolation, but **not derived from this
  employee's own target** — `rescale_chain_distances` (`activity.py:405-410`)
  explicitly falls back to the donor's raw, unscaled `TRPMILES`/`TRVLCMIN`
  whenever `total_daily_miles` is NaN, since there is no target to rescale
  to. Empirically (checked against the current run), these 2,637 employees'
  synthetic chains have a mean total leg distance of ~31 miles (min 0, max
  ~874, 25th/50th/75th percentiles ~13/23/37 miles) — individually
  plausible-looking numbers, but they describe *the donor's* day, not a
  rescaled version of anything specific to this synthetic employee.
- **Modes**: **this is the actual gap.** `build_donor_legs`
  (`activity.py:196-267`) does not filter donor legs to driving-mode trips
  the way `build_features.py` does when computing `total_daily_miles` —
  there is no `TRPTRANS`/`DRIVING_MODE_TRPTRANS_CODES` restriction anywhere
  in the donor-pool construction. This means the donor whose chain shape
  gets borrowed for one of these 2,637 employees was selected purely on
  `trips_per_day`/`number_of_stops` match, with no consideration of whether
  the donor's *own* day was a driving day. Two distinct scenarios are
  currently indistinguishable in the output:
  1. The donor drove that day (its `total_daily_miles` in the source table
     is *not* null) even though the synthetic employee's own resampled
     `total_daily_miles` happens to be null — in this case the synthetic
     employee's activity chain shows real driving-mode legs, which is
     arguably fine (the chain *shape*, per `docs/activity_generation_plan.md`
     §3, was never meant to imply the donor and synthetic employee share
     every attribute).
  2. The donor *also* had `total_daily_miles == NaN` (also a non-driving
     day) — in which case the donor's `TRPMILES`/`TRVLCMIN` values being
     carried through as `distance`/`duration` describe a walk/bike/transit
     trip, not a drive, and labeling them `distance`/`duration` in a
     *driving* activity table without any mode flag is misleading at best
     for a downstream charging model that will read every leg as
     "the vehicle went here."
  Validation should measure how often case 2 occurs (join
  `synthetic_activity.parquet`'s implied donor back through
  `employee_clusters.parquet`'s `total_daily_miles`/`TRPTRANS`, or add a
  temporary diagnostic column) and report that rate distinctly from case 1.
- **Workplace arrival/departure**: plausible and unaffected — arrival/
  departure rescaling (`rescale_chain_times`) is anchored to
  `work_arrival_time`/`work_departure_time`, which are independent draws
  from the cluster's fitted distribution and are not null for these
  employees (only the mileage/duration columns are null). Workplace timing
  for this subgroup should validate identically to the rest of the
  population under §4's workplace-timing checks.

### Determination: meaningful behavior, modeling limitation, or future design decision?

- **The missingness itself is meaningful behavior**, faithfully carried
  through from a real, documented NHTS phenomenon (non-driving-mode
  commute day), preserved deliberately at every pipeline stage
  (`cluster.py`'s zero-impute-for-clustering-math-only,
  `sample.py`'s NaN-preserving jitter, `activity.py`'s NaN-aware rescaling
  fallback). **Do not impute a value here** — there is no real driving
  distance to impute, and doing so would fabricate data for someone who
  didn't drive.
- **The mode-blind donor pool in `activity.py` is a modeling limitation**,
  not an intentional design decision (nothing in `docs/activity_generation_plan.md`
  discusses restricting the donor pool by driving mode — it appears to have
  simply not been considered, since `build_features.py`'s own driving-mode
  filter for a materially related computation exists a few files away).
  This should be measured (the case-2 rate above) before deciding whether
  it's material enough to fix — if case 2 is rare, it may be an acceptable
  known limitation to document; if it affects a large share of the 2,637
  employees, it likely warrants adding a driving-mode restriction (or at
  minimum a `donor_is_driving_day` flag) to `build_donor_legs` in a future
  change.
- **A future design decision worth flagging explicitly** (not this
  validation pass's job to resolve): should a synthetic employee whose own
  `total_daily_miles` is null even receive a full driving-activity chain at
  all, or should `synthetic_activity.parquet` represent them with an
  explicit "no driving that day" row/flag instead of a donor-borrowed chain
  that may or may not itself represent driving? This is a legitimate
  alternative design, not clearly superior to the current approach, and
  should be raised as an open question for whoever owns `docs/activity_generation_plan.md`
  rather than decided unilaterally inside a validation report.

---

## 6. Reproducibility validation

| Check | Command | Expected output |
|---|---|---|
| Fixed seed produces identical results | Run generation twice with the same explicit seed, diff the outputs | `python -m driving_profiles.generator.sample --seed 42 -n 5000` then move/rename the output, rerun identically, `diff` (or `pd.testing.assert_frame_equal`) the two `synthetic_employees.parquet` files — byte-identical (already asserted at the unit level in `test_generator.py`'s `test_create_synthetic_employee_table_is_reproducible_with_fixed_seed`; this is the same check at full population scale) |
| Same seed flows through activity generation too | `python -m driving_profiles.generator.activity --seed 42`, run twice against the same `synthetic_employees.parquet`, diff `synthetic_activity.parquet` | Byte-identical (unit-level equivalent: `test_activity.py`'s `test_generate_synthetic_activity_is_reproducible_with_same_seed`) |
| Changing seed produces different but statistically similar populations | Run with `--seed 42` and `--seed 43` (or any two distinct seeds), compare the two synthetic populations against each other using the same distributional tests as §2/§3/§4 | Row-level values differ (not identical to each other), but KS tests between the *two synthetic* populations should show no significant difference (`p > 0.05`) — confirming seed variation samples the same underlying distributions rather than producing systematically different populations |
| Cluster proportions stable across seeds | Compare cluster share (§3) across the two seeded runs above | Should match almost exactly regardless of seed (proportional sampling via `determine_cluster_sampling` is deterministic given `n` and the source population; only the largest-remainder tie-break and within-cluster row draws vary by seed) |
| Full pipeline regenerates cleanly from a fresh environment | From a clean checkout (no `data/processed/`, `data/interim/` populated): `python -m driving_profiles.data.download` → `python -m driving_profiles.data.clean` → `python -m driving_profiles.features.build_features` → `python -m driving_profiles.features.cluster` → `python -m driving_profiles.generator.sample` → `python -m driving_profiles.generator.activity` (or the equivalent `scripts/run_pipeline.py` invocation, per `tests/test_run_pipeline.py`'s stage coverage) | Every stage completes without error and produces its documented output file; final `synthetic_employees.parquet`/`synthetic_activity.parquet` pass §2-§5 validation identically to the currently-committed run (same default seed) |
| Population size scaling doesn't break fidelity | Regenerate at `n=500` and `n=50,000` with the same seed, re-run §2's per-cluster distributional comparisons at each size | KS test results should not depend materially on `n` — cluster weights and within-cluster distributions are independent of population size by design (`docs/synthetic_generation_plan.md` §3); only the p-values' sensitivity to sample size is expected to change, not the underlying pass/fail conclusion |

---

## 7. Validation metrics table

Summary of every metric above, for a validation report to work through
top-to-bottom.

| Metric | Dataset comparison | Method | Why it matters for EV charging model |
|---|---|---|---|
| Cluster proportions | `employee_clusters.parquet` vs. `synthetic_employees.parquet` | Direct proportion comparison | Confirms the archetype mix feeding every downstream estimate is correct; a proportion bug silently biases the whole demand curve toward the over/under-represented archetype |
| Cluster separation/meaningfulness | `employee_clusters.parquet`, silhouette/dendrogram diagnostics | Silhouette score table, hierarchical dendrogram cross-check, per-cluster profile inspection | If clusters aren't behaviorally real, every "per-cluster" validation and every cluster-conditioned sampling step downstream is validating against a distinction that doesn't exist |
| Age / age band / income / household size | Source clustered pop vs. synthetic, per cluster | KS test / chi-square | Demographic realism for reporting/stakeholder credibility; low direct mechanism for charging demand itself |
| Household vehicle count / vehicles per driver / availability flags | Same | KS test / proportion comparison | Determines whether a vehicle is even present to charge — a precondition for every charging session, not just a timing input |
| Commute distance | Same, plus tail-percentile check | KS test + percentile comparison | Direct driver of energy required per charging session; tail (long-commute) behavior disproportionately affects peak energy estimates |
| Commute duration | Same | KS test | Secondary context for commute distance; less directly load-bearing |
| Work arrival time | Same, time-of-day histogram | KS test on peak shape, not just mean | Defines charging session start time — directly determines whether the demand model sees a sharp coincident peak or a smoothed one |
| Work departure time | Same | KS test | Defines charging session end / max duration bound |
| Trips per day / number of stops | Same, discrete distribution | KS test on full distribution (not just mean) | A generator that flattens variance here understates how often the workplace dwell window is interrupted, mis-shaping session-duration estimates |
| Total daily miles / driving minutes / avg trip distance | Same, non-null subset | KS test on non-null values | Total daily VMT context for energy estimates beyond the commute leg itself |
| Missingness rate (driving summary columns) | Source vs. synthetic, pooled and per-cluster | Proportion comparison | If missingness rate drifts from source, the "no driving that day" subpopulation is mis-sized, which directly changes what fraction of the workforce the charging model should expect to see arrive with any commute VMT at all |
| Donor mode-blindness (case 2 rate, §5) | `synthetic_activity.parquet` legs traced to donors with their own null `total_daily_miles` | Diagnostic join + rate calculation | Determines whether a meaningful share of "driving activity" legs actually describe a non-driving trip mislabeled as driving distance/duration — directly inflates or distorts VMT-derived energy estimates if material |
| Trip/leg distance & duration distributions | `trips_clean.parquet` (donor pool) vs. `synthetic_activity.parquet` | KS test, per cluster and leg purpose | Underlying data for every per-leg energy/timing calculation the charging model will do |
| Implied leg speed plausibility | `synthetic_activity.parquet` distance/duration | Structural check against `MIN_PLAUSIBLE_SPEED_MPH`/`MAX_PLAUSIBLE_SPEED_MPH` | A leg with an implausible implied speed would produce a nonsensical duration input to session-timing logic |
| Workplace arrival/departure consistency | `synthetic_activity.parquet` vs. `synthetic_employees.parquet`'s own drawn values | Structural equality check (not KS test — deterministic by construction) | The single most load-bearing timing input to the demand curve; any drift here is a rescaling bug, not natural variation |
| Workplace dwell duration distribution | Donor pool `DWELTIME` vs. synthetic `workplace_dwell_minutes` | KS test on non-null values, per cluster | Directly bounds maximum charging session length — the crux metric for whether a session can complete before departure |
| Fragmented dwell-window rate | Donor pool vs. synthetic, share of employees with >1 work-purpose leg | Proportion comparison | Determines whether "one long session" or "multiple shorter sessions" is the right per-employee charging model |
| Fallback chain rate | `synthetic_activity.parquet`'s `chain_source` proportion, overall and per cluster | Proportion + logged rate (already logged at generation time — `activity.py:630-636`) | A high fallback rate in a cluster means donor-derived realism can't be trusted there — fallback chains are a coarser model (direct home→work→home) that understates chain complexity |
| Reproducibility (fixed seed) | Two runs, same seed | Exact-equality diff | Prerequisite for any regenerate-and-recompare workflow the demand model or its own validation will need later |
| Reproducibility (varying seed) | Two runs, different seeds | KS test between the two synthetic populations | Confirms seed is a genuine randomness control, not accidentally deterministic or accidentally non-representative |

---

## 8. Recommendations

**What passes validation (expected, based on this session's inspection —
to be confirmed by actually running the comparisons above rather than
assumed):**

- Cluster proportions in `synthetic_employees.parquet` match the source
  clustered population closely (83.5%/16.5% split reproduced almost
  exactly at `n=5,000`) — the sampling mechanism itself has no evident bug.
- Missingness rate and co-occurrence for `total_daily_miles`/
  `total_driving_minutes`/`average_trip_distance_miles` is faithfully
  preserved from source, both pooled and per-cluster, within ordinary
  resampling variance.
- Workplace arrival/departure timing in `synthetic_activity.parquet` is
  structurally anchored to each employee's own drawn `work_arrival_time`/
  `work_departure_time` by construction — this is a correctness property of
  the code, not something that needs a distributional test to confirm, just
  a structural equality check at scale.
- Structural chain validity (chronological legs, non-negative distances,
  no real NHTS IDs in output) is already covered by the existing unit test
  suite at the fixture level; this plan's contribution is confirming the
  same properties hold at full-population scale, which should be a
  low-risk confirmation rather than a likely source of new findings.

**What requires investigation before this population is trusted further:**

- **Cluster meaningfulness at `k=2`** (§3) — a silhouette score of 0.417 is
  the best among the candidates tried, but "best available" is not the same
  as "well-separated." The domain-interpretability review
  `docs/clustering_plan.md` §6 calls for does not appear to be documented
  anywhere; this should happen before any further validation work leans on
  "cluster 0 vs. cluster 1" being a meaningful distinction, since a large
  share of this document's own checks are structured around per-cluster
  comparison.
- **Donor mode-blindness (§5)** — measure the case-2 rate (synthetic
  employee with null `total_daily_miles`, matched to a donor who *also* had
  a non-driving day) before deciding whether `synthetic_activity.parquet`'s
  `distance`/`duration` columns need a caveat, a mode flag, or a
  `build_donor_legs` change restricting donors to driving-mode legs.
- **Actual KS-test/distributional results for every metric in §2/§4** —
  this document specifies the comparisons; none have actually been run yet.
  Several (arrival-time peak shape, commute-distance tail behavior, dwell
  duration distribution) are exactly the kind of check that can look fine
  on a mean/median summary while failing a full-distribution test, per this
  project's own stated validation philosophy (`docs/synthetic_generation_plan.md`
  §6) — they should be run, not assumed to pass by extension of the
  proportion/missingness checks that were already spot-checked this
  session.
- **Fallback rate under different `n`/seed** — the current default run
  reported all 5,000 employees using donor-matched chains (0% fallback per
  the earlier inspection), but this should be re-checked at smaller `n`
  and with different seeds/cluster-weight overrides, since a skewed
  cluster-weight scenario (`docs/synthetic_generation_plan.md` §3's
  "what if this site skews toward long-commute hybrid workers" case) could
  plausibly push more draws into a sparser corner of the donor pool.

**What should be fixed before charging demand modeling:**

- Nothing identified in this session rises to "must fix before proceeding"
  on its own — but this is a *plan*, not a completed validation run.
  The two investigation items above (cluster meaningfulness, donor
  mode-blindness) should be resolved — either confirmed benign or
  addressed in code — before `scenarios/charging_demand.py` is built on
  top of `workplace_dwell_minutes` and per-leg `distance`, since both
  columns are exactly the ones a donor mode-blindness issue would corrupt,
  and both are the primary charging-relevant metrics per §4's ranking.

**What assumptions should be documented in a research paper:**

- NHTS 2022 is a single-day travel diary per respondent; this pipeline
  treats each respondent's one surveyed day as representative of their
  typical day, and validation against NHTS 2022 can only confirm internal
  consistency with that survey, not real day-to-day behavioral stability.
- NHTS 2022's national sample is not matched to any specific employer,
  site, or industry; validation against NHTS confirms fidelity to national
  commuting patterns, not to any particular workplace this pipeline might
  eventually be used to model.
- NHTS 2022 was fielded during a period of atypical telework/commuting
  prevalence relative to pre-2020 surveys; this should be stated
  explicitly if the resulting demand estimates are ever compared against
  pre-pandemic benchmarks or used in a forward-looking projection.
- The cluster count (`k=2`) was chosen by highest silhouette score among
  `k=2..8`, with a domain-interpretability review recommended but not yet
  documented as completed — the paper should state what that review found
  (or that it's outstanding) rather than presenting `k=2` as an
  uncontested choice.
- `total_daily_miles`/`total_driving_minutes` missingness (~52% of the
  clustered population) reflects real non-driving-mode commute days, is
  intentionally preserved (not imputed) through every generation stage,
  and — pending the §5 investigation above — the activity-generation
  donor pool is not currently restricted by driving mode, which should be
  either resolved or explicitly caveated as a known limitation of the
  `distance`/`duration` columns for this subpopulation.
