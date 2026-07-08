# Synthetic Employee Generation Plan

Status: design only — no code written yet. This document specifies how
`generator/sample.py` and `generator/activity.py` should turn
`data/processed/employee_clusters.parquet` (the `cluster.py` output: one row
per `HOUSEID`+`PERSONID`, labeled with a `cluster_id`) into an arbitrarily
large synthetic employee population with full daily driving activity
profiles. It does not implement anything; see `docs/clustering_plan.md` for
how the clusters referenced below were built and
`docs/feature_engineering_plan.md` for how the underlying columns were
derived from NHTS.

---

## 1. Objective

NHTS 2022 gives us a fixed, non-representative sample of ~7,800 real
employees, each with a single surveyed weekday. That is enough to *learn*
what employee travel behavior looks like, but it is the wrong shape to feed
a workplace charging demand model directly: a real employer's workforce has
a different size, and reusing real respondents' exact records one-for-one
would tie any downstream demand estimate to NHTS's specific sample (a
privacy problem, since individual diary days would be traceable, and a
generalizability problem, since the estimate could never scale beyond
~7,800 people or adapt to a workforce with a different size or mix).

Synthetic employee generation solves this by treating the clustered NHTS
data as a **fitted model of employee travel behavior**, not a lookup table.
Each cluster (`docs/clustering_plan.md` §1) is a coherent archetype with its
own joint distribution over commute distance, arrival/departure time,
vehicle availability, and demographics. Generation draws new, synthetic
individuals from those distributions rather than resampling real rows, so
the output can be:

- **Arbitrarily sized** — a synthetic workforce of 50, 5,000, or 50,000
  employees, matched to whatever site or scenario is being modeled.
- **Representative in aggregate** — cluster proportions, and the
  within-cluster feature distributions, are calibrated to the real NHTS
  population (§6), so synthetic employees "look like" real ones in
  distribution without being any specific real person.
- **A stable base for demand modeling** — a workplace EV charging demand
  model needs many employee-days of arrival time, dwell time, and commute
  VMT to estimate charging load and timing. Synthetic generation is what
  produces that volume of employee-days from a one-day, fixed-size survey.

This document is scoped to **travel behavior only**. No EV ownership,
charging behavior, or fuel-type assumption is introduced anywhere in
generation — that is deferred to `scenarios/charging_demand.py` (§8),
consistent with `docs/feature_engineering_plan.md` §3 and
`docs/clustering_plan.md` §7.

---

## 2. Inputs

**Source:** `data/processed/employee_clusters.parquet`, produced by
`cluster.py`. This is the *only* input generation should read — no raw NHTS
files, and no re-derivation of features that `build_features.py` /
`cluster.py` already computed.

**Grain:** one row per `HOUSEID`+`PERSONID`, identical to
`employee_features.parquet` (`docs/feature_engineering_plan.md` §1), plus
one additional column: `cluster_id` (nullable `Int64`).

**Population split.** Not every row has a `cluster_id`. Per
`docs/clustering_plan.md` §2/§7, only workers with an observed weekday
commute were clustered; non-workers and workers without a commute that day
keep every other column but have `cluster_id = <NA>`. Generation must
**filter to `cluster_id.notna()`** before sampling (§3) — an unclustered row
has no archetype distribution to sample from and would otherwise pull the
generator back toward reusing a real row directly, defeating §1's purpose.
As of the current `cluster.py` run, this filter keeps 2,434 of 7,845 rows
(the current evaluation picked `k=2` by silhouette score, saved to
`data/processed/cluster_evaluation.csv`); the exact split will change if a
different `k` is chosen after the domain-interpretability review
`docs/clustering_plan.md` §6 calls for, but the *shape* of the input
(a filterable, cluster-labeled table) will not.

**Columns available**, grouped as they'll be used in §4–5:

| Group | Columns | Role in generation |
|---|---|---|
| Identifiers | `HOUSEID`, `PERSONID` | Not sampled — real-respondent identifiers, dropped after generation (see §4's privacy note). |
| Archetype | `cluster_id` | The sampling unit (§3) — every other column is sampled *conditional on* this. |
| Demographics | `age`, `age_band`, `worker_status` | §4 demographic attributes. |
| Household | `household_income_bracket`, `household_size`, `household_vehicle_count` | §4 household attributes. |
| Commute behavior | `commute_distance_survey_miles`, `commute_distance_trip_miles`, `commute_duration_minutes`, `work_arrival_time`, `work_departure_time` | §5 charging-window inputs (arrival/departure, commute distance). |
| Daily mobility | `work_trip_count`, `trips_per_day`, `total_daily_miles`, `total_driving_minutes`, `number_of_stops`, `average_trip_distance_miles` | §5 trip-count/mileage inputs. |
| Vehicle availability | `vehicles_per_driver`, `vehicle_per_driver_adequate`, `household_vehicle_trip_count`, `used_household_vehicle` | §4/§5 vehicle-sharing context. |

**What this table does *not* have** — and generation must not invent —
is trip-by-trip detail: the per-person row is already aggregated
(`docs/feature_engineering_plan.md` §1), so `number_of_stops` and
`trips_per_day` are counts, not an ordered sequence of legs. §5 addresses
where the ordered-sequence information for reconstructing a trip chain
should come from (the trip-level `data/interim/trips_clean.parquet`, used
as a structural reference library, not as more per-person facts).

---

## 3. Sampling strategy

**Proportional cluster sampling, by default.** Each synthetic employee is
generated by first drawing a `cluster_id`, weighted by that cluster's share
of the real clustered population (e.g., if cluster 0 is 83% of clustered
NHTS employees, roughly 83% of synthetic employees should be drawn from
cluster 0). This preserves the real archetype mix and is what makes
aggregate synthetic statistics (§6) comparable to NHTS at all. The generator
should also support an explicit override — a caller-supplied cluster-weight
vector — so a scenario can deliberately model a workforce with a different
archetype mix (e.g., "what if this site skews toward long-commute hybrid
workers"); proportional-to-NHTS should be the default, not the only option.

**User-specified population size.** The number of synthetic employees to
generate (`n`) must be a caller-supplied parameter, not fixed to the size of
the clustered NHTS population. This is the entire point of generation
(§1) — a workforce of 500 or 50,000 should be produced from the same fitted
distributions without re-clustering or re-fitting anything. `n` only
controls how many draws are taken in §4; it has no effect on the cluster
weights or within-cluster distributions themselves.

**Randomness control.** All sampling must be seeded through the existing
`driving_profiles.utils.random_seed` module (`get_seed`/`get_rng`), the same
convention `cluster.py` already follows, rather than any ad hoc seeding.
Concretely:

- A single `seed` parameter (default `random_seed.DEFAULT_SEED`) should
  flow into every random draw a generation run makes — cluster assignment,
  within-cluster feature sampling (§4), and activity reconstruction (§5) —
  so that the same `(seed, n, cluster weights)` reproduces an identical
  synthetic population byte-for-byte. Reproducibility matters here because
  synthetic populations feed a demand model downstream (§8); being able to
  regenerate the exact same population is what makes a demand estimate
  re-runnable and diffable across code changes.
- A caller that wants genuine variation across runs (e.g., generating
  several independent synthetic workforces to characterize demand
  uncertainty) should do so by varying `seed` explicitly, not by leaving it
  unset — an unset seed still resolves to the same project default via
  `get_seed`, so "no seed" and "reproducible" are the same thing by design,
  matching how `cluster.py` already behaves.

---

## 4. Employee generation

Each synthetic employee is built in two steps: pick an archetype, then draw
that employee's attributes from within that archetype's fitted
distribution.

1. **Draw a `cluster_id`** per §3's proportional (or overridden) weights.
2. **Draw every attribute jointly from that cluster's within-cluster
   distribution**, not independently column-by-column from the
   population-wide marginal. This is the same principle
   `docs/clustering_plan.md` §7 establishes for why clusters exist in the
   first place: a cluster's whole value is that it captures *joint*
   structure (e.g., long commutes co-occurring with earlier arrival times
   and a dedicated, high-availability vehicle). Sampling each column
   independently — even from the correct per-cluster marginal — would
   destroy exactly that correlation and could, for instance, pair a
   30-mile commute with a 5-minute dwell time that never occurs in the real
   data. Two ways to implement joint sampling, in increasing order of
   fidelity (a later implementation phase should pick one, informed by how
   well-separated the clusters turn out to be):
   - **Resample-with-noise**: pick a real within-cluster row at random and
     jitter its continuous fields (e.g., add small Gaussian noise to
     commute distance and arrival time, scaled to that cluster's own
     within-cluster standard deviation). Simple, and guarantees the joint
     structure of *some* real respondent is preserved, but risks staying
     close to individual real records unless the jitter is large enough —
     the privacy motivation in §1 means jitter magnitude is a real design
     parameter, not just a smoothing nicety.
   - **Fit-and-sample**: fit a joint distribution per cluster (e.g., a
     multivariate Gaussian over the standardized continuous features, or —
     if `docs/clustering_plan.md` §5's GMM follow-up is adopted for
     clustering itself — sample directly from each component's fitted
     Gaussian) and draw new points from it, then map categorical/discrete
     columns (age band, income bracket) from the cluster's empirical
     conditional frequencies given the drawn continuous values. Better
     privacy separation from individual respondents and smoother output
     distributions, at the cost of needing a per-cluster model-fitting step
     before sampling can begin.
3. **Assign a synthetic identifier** in place of `HOUSEID`/`PERSONID` (e.g.,
   a generated UUID or sequential `synthetic_employee_id`) — the real IDs
   exist in the input for traceability back to `employee_clusters.parquet`
   during development/validation, but must not appear in generator output,
   both because they're meaningless for a synthetic person and to keep any
   accidental exact-row reuse from step 2's resample approach from being
   traceable to a real NHTS respondent.

Attributes fall into the three groups the task calls out, all produced by
the same joint-sampling step above (they aren't three separate draws):

- **Demographic attributes** — `age`/`age_band`, drawn per cluster; the
  cluster generally under-determines these (`docs/clustering_plan.md` §3
  treats them as secondary/contextual), so their main role is making the
  synthetic population's demographic composition realistic (§6), not
  differentiating archetypes.
- **Household attributes** — `household_income_bracket`, `household_size`,
  `household_vehicle_count` (and derived `vehicles_per_driver`), drawn per
  cluster for the same reason.
- **Travel behavior** — `commute_distance_survey_miles`,
  `commute_duration_minutes`, `work_arrival_time`, `work_departure_time`,
  `trips_per_day`, `total_daily_miles`, `number_of_stops`,
  `vehicle_per_driver_adequate`, `used_household_vehicle`. These *are* the
  columns that define a cluster (`docs/clustering_plan.md` §3's "primary"
  groups), so they should show the most cluster-to-cluster variation and
  the strongest within-cluster joint correlation — this is the group where
  getting joint (not marginal) sampling right in step 2 matters most.

The output of this stage is a **synthetic employee profile table**: one row
per synthetic employee, same column shape as the travel-behavior columns of
`employee_clusters.parquet` but with a synthetic ID instead of
`HOUSEID`/`PERSONID`, and no trip-by-trip detail yet — that's §5.

---

## 5. Driving activity generation

The employee-profile table from §4 has *summary* travel statistics per
employee (one arrival time, one departure time, a trip count, a total
mileage) but not a reconstructed day — no ordered sequence of legs. §5's
job is to expand each synthetic employee's summary row into a plausible
**daily trip chain** consistent with those summary statistics, since a
workplace charging demand model needs to know not just "this employee
arrives around 8:15am" but the actual shape of their day (do they leave
work and come back, do they run an errand before or after the commute).

**Approach: reconstruct chains from real NHTS chain shapes, not from
scratch.** `data/interim/trips_clean.parquet` (the `clean.py` output,
trip-level, with `WHYFROM`/`WHYTO` purpose sequences and
`STRTTIME`/`ENDTIME` per leg) already contains real, internally-consistent
trip chains for every person in the underlying survey. Rather than
inventing chain logic independently (e.g., generating trip purposes and
times as if they were unrelated random draws, which risks producing
physically incoherent days — a chain that arrives at work after it departs,
or a stop sequence with a trip count that doesn't match `number_of_stops`),
generation should:

1. **Select a chain template** from real trip chains belonging to the same
   cluster as the synthetic employee (join `trips_clean.parquet` back to
   `employee_clusters.parquet` by `HOUSEID`+`PERSONID`, filtered to that
   `cluster_id`), chosen to match the synthetic employee's `trips_per_day`
   and `number_of_stops` as closely as possible. This guarantees the
   *shape* of the day (how many legs, in what purpose order — home→work,
   home→other→work, work→other→work→home, etc.) is one that genuinely
   occurs in NHTS for that archetype.
2. **Rescale the template's times and distances** to the synthetic
   employee's own drawn values from §4, rather than copying the template
   verbatim (verbatim copying would just be resampling real records again,
   the problem §1 identifies). Concretely:
   - **Arrival time / departure time**: shift the template's work-arrival
     and work-departure legs so they land on the synthetic employee's own
     drawn `work_arrival_time`/`work_departure_time`, and shift every other
     leg in the template by the same offset so relative spacing between
     legs is preserved.
   - **Daily miles**: scale each leg's distance proportionally so the
     chain's total matches the synthetic employee's drawn
     `total_daily_miles`, anchoring the work-purpose leg(s) to the drawn
     commute distance specifically (since that's independently sampled and
     should be the more authoritative of the two for the commute leg).
   - **Trip counts / stops**: already fixed by the template-selection step;
     no separate reconciliation needed as long as template selection
     matched on these fields within a reasonable tolerance (an
     implementation detail — e.g., "same `number_of_stops`, `trips_per_day`
     within ±1" — to be tuned once real template availability per cluster
     is inspected).
3. **Fall back gracefully when no close template exists** in a given
   cluster (e.g., an unusual combination of high stop count and short
   total mileage) by relaxing the match tolerance before falling back to a
   simpler generated chain (e.g., a direct home→work→home chain with
   inserted stops), so generation never fails outright for an edge-case
   employee — it should degrade to a less chain-realistic but still valid
   output rather than raising an error.

This keeps `generator/activity.py`'s job scoped to *reconstruction*
(borrow real structure, rescale to synthetic values) rather than
*simulation from first principles*, which is both more consistent with how
`cluster.py`/`build_features.py` already treat NHTS as ground truth to be
resampled/re-fit rather than modeled mechanistically, and cheaper to
validate (§6) since chain shapes are drawn from shapes that provably occur
in the source data.

The output of this stage is a **synthetic driving activity table**: one row
per synthetic-employee-leg (long format), with purpose, start time, end
time, and distance per leg — the trip-level analog of `trips_clean.parquet`
for the synthetic population.

---

## 6. Validation

Before any synthetic population is used downstream (§8), it must be checked
against the real NHTS clustered population it was sampled from — the goal
is confirming the generator reproduces distributions, not individual
records (reproducing individual records would be the privacy failure §1
exists to avoid). Validation should run automatically as part of
generation (e.g., as a report emitted alongside the output files in §7),
not as a one-off manual check.

Comparisons to run, real (`employee_clusters.parquet`, clustered rows only)
vs. synthetic (`generator/sample.py` + `generator/activity.py` output):

- **Cluster proportions** — compare the synthetic population's realized
  cluster-share breakdown against the input weights from §3 (should match
  by construction, so this is mainly a check that sampling code has no
  bug) and against the real NHTS cluster shares if an overridden weight
  vector was *not* used.
- **Distribution comparisons per feature** — for every continuous field
  sampled in §4 (commute distance, arrival time, departure time, daily
  miles, dwell time), compare real vs. synthetic distributions both
  visually (overlaid histograms/KDE plots) and with a quantitative
  goodness-of-fit test (e.g., a two-sample Kolmogorov–Smirnov test),
  computed **within each cluster**, not just on the pooled population —
  a synthetic population could match the pooled marginal while being wrong
  within every individual cluster if cluster proportions happen to offset
  the errors.
- **Commute distance** — specifically worth calling out beyond the general
  per-feature check above, since it's the direct input to energy-per-session
  in the eventual charging model (§8): compare mean/median/tail behavior
  (long-commute outliers matter for peak demand), not just central
  tendency.
- **Trip counts / number of stops** — compare the discrete distribution
  (not just the mean) of `trips_per_day`/`number_of_stops` between real and
  synthetic, since these are integer-valued and a generator that gets the
  mean right but flattens the variance (e.g., always generating exactly the
  cluster's mean stop count) would understate how often a workplace dwell
  window gets interrupted (`docs/feature_engineering_plan.md` §2's midday-
  departure feature).
- **Arrival time / departure time** — compare the full time-of-day
  distribution (not just mean arrival time) against real NHTS, since the
  *shape* of the arrival/departure peak (a sharp rush-hour spike vs. a wide
  spread) directly determines whether a charging-demand model sees a sharp
  coincident-load peak or a smoothed one — getting the mean right but the
  spread wrong would materially bias a peak-load estimate.
- **Chain-shape plausibility (§5-specific)** — spot-check that reconstructed
  synthetic chains are internally consistent (arrival before departure,
  leg distances non-negative and summing to the employee's total, purpose
  sequence starts and ends at a plausible location e.g. home) — a
  correctness check on the reconstruction logic itself, distinct from the
  distributional checks above.

Any systematic mismatch found here should feed back into §4/§5 (e.g., "the
resample-with-noise jitter is too large and is flattening the commute
distance distribution") rather than being treated as an acceptable
approximation error — validation is a gate before the output is trusted by
§8, not a report to file away.

---

## 7. Output files

Two pipeline artifacts, mirroring the `sample.py` / `activity.py` module
split already stubbed in `src/driving_profiles/generator/`:

- **`generator/sample.py`** → `data/processed/synthetic_employees.parquet`
  — one row per synthetic employee: a synthetic ID, `cluster_id`, and the
  full demographic/household/travel-behavior column set described in §4
  (the same shape as the travel-behavior columns of
  `employee_clusters.parquet`, minus `HOUSEID`/`PERSONID`). This is the
  direct output of §4 and the required input to §5.
- **`generator/activity.py`** → `data/processed/synthetic_activity.parquet`
  — one row per synthetic-employee-leg (long format): synthetic employee
  ID (joining back to `synthetic_employees.parquet`), leg sequence number,
  purpose, start time, end time, and distance, as described in §5's output.

Both stages should also write a **validation report** (§6) — e.g.,
`reports/synthetic_validation.{html,csv}` or similar, following whatever
convention `reports/` already establishes elsewhere in the pipeline —
summarizing the distribution comparisons so a generation run's fidelity can
be checked without re-running the comparisons manually every time.

`n` (population size), `seed`, and any cluster-weight override (§3) are
run parameters, not persisted output — but should be recorded (e.g., in
a small run-metadata sidecar or logged at generation time) so a given
`synthetic_employees.parquet`/`synthetic_activity.parquet` pair can be
traced back to the parameters that produced it, the same way `cluster.py`
logs its chosen `k` today.

---

## 8. Future integration

`synthetic_employees.parquet` and `synthetic_activity.parquet` are designed
to be exactly what `scenarios/charging_demand.py` needs and nothing more:
per-employee arrival/departure time (defining a charging window), dwell
time (bounding session length), commute distance/VMT (determining energy
consumed), and vehicle availability (determining whether a vehicle is
present to charge at all). This mirrors the same downstream contract
`docs/clustering_plan.md` §7 already established for clusters feeding
generation — generation is simply the next stage that turns cluster-shaped
distributions into individual synthetic employee-days at the volume a
demand model needs.

Critically, **no EV ownership, fuel type, or charging behavior is assumed
anywhere in this document** — every table and column described above is a
statement about *driving* behavior only (where a vehicle goes and when),
never about what kind of vehicle it is. This is intentional and mirrors
`docs/feature_engineering_plan.md` §3's deferred scope: EV penetration
(5%/10%/20%) and vehicle fuel-type assumptions are scenario parameters to
be overlaid *on top of* a completed synthetic population, not baked into
generation itself. Concretely, the future integration step will:

1. Take `synthetic_employees.parquet` + `synthetic_activity.parquet` as a
   fixed input, unchanged by which EV scenario is being run.
2. Apply an EV-penetration draw (e.g., flag X% of synthetic employees as
   EV drivers, weighted however realism requires — uniformly at random, or
   correlated with commute distance/income if a future study motivates
   that) and a vehicle fuel-type/efficiency assumption per EV-flagged
   employee.
3. Convert each EV-flagged employee's workplace dwell window and commute
   VMT into an estimated charging session (timing, energy required, and —
   if a charger-power assumption is added — session duration), aggregating
   across the synthetic workforce to produce a workplace charging demand
   curve.

Because steps 2–3 are scenario parameters applied after the fact, re-running
a different EV penetration or fuel-type assumption never requires
regenerating the synthetic population — the same `synthetic_employees` /
`synthetic_activity` pair can be reused across every scenario, exactly as
`docs/clustering_plan.md` §7 anticipated when it deferred this same
overlay out of the clustering/generation stages in the first place.
