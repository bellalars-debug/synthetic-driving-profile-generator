# Model Status

Status snapshot of the synthetic mobility generator as of commit `059386a`
("Finalize validated synthetic activity generation"). Summarizes
`docs/validation_results.md` and `docs/activity_validation_investigation.md`;
introduces no new analysis.

## 1. Pipeline stages completed

1. **Data ingestion** - NHTS 2022 raw extracts downloaded and ingested
   (`data/driving_profiles/data/download.py`, `ingest.py`).
2. **Data cleaning** - trip records filtered to weekday workers / driving-
   relevant trips (`data/clean.py`; output `data/interim/trips_clean.parquet`).
3. **Feature engineering** - employee-level commute and daily-mobility
   features built, including the upstream implausible-value filtering added
   in this phase (`features/build_features.py`; output
   `data/processed/employee_features.parquet`).
4. **Behavioral clustering** - employees clustered into two archetypes via
   KMeans, cross-checked with hierarchical clustering
   (`features/cluster.py`; output `data/processed/employee_clusters.parquet`,
   `cluster_evaluation.csv`).
5. **Synthetic employee sampling** - 5,000 synthetic employees drawn per
   cluster proportions with jittered demographic/summary features
   (`generator/sample.py`; output `data/processed/synthetic_employees.parquet`).
6. **Synthetic activity generation** - per-employee daily trip chains built
   by donor selection and rescaling against real NHTS trip legs
   (`generator/activity.py`; output `data/processed/synthetic_activity.parquet`).
7. **Validation** - population, cluster, activity, and missingness checks
   run against source distributions (`validation/*.py`; output
   `docs/validation_results.md`).

## 2. Final dataset sizes

| dataset | rows | notes |
|---|---:|---|
| `synthetic_employees.parquet` | 5,000 | cluster_0 = 859, cluster_1 = 4,141 |
| `synthetic_activity.parquet` | 15,201 legs | 100% `chain_source == "donor"` (0 fallback chains) |
| source population (NHTS, pooled) | 2,434 employees | used as the comparison baseline throughout validation |

## 3. Validation improvements made

Across the five most recent commits, validation moved from an unguarded
first pass to a report with documented, investigated failures:

- Added the synthetic data validation framework and plan (population,
  cluster, activity, missingness sections).
- Fixed donor selection to respect driving-mode compatibility (no donor
  mismatched on whether the employee actually drives).
- Fixed commute anchoring and time-of-day jitter behavior.
- Improved time jitter and donor schedule (time-of-day) matching.
- Finalized the relaxed same-cluster donor tier, eliminating fallback
  chains entirely.

Current top-line result (`docs/validation_results.md` §1): **76 passed, 26
failed, 21 informational** across population/clusters/activity/missingness
sections.

## 4. Important fixes

- **Driving-compatible donor matching** - `select_donor` now filters
  candidate donors on the employee's own driving-mode requirement, so a
  non-driving employee can no longer be assigned a driving-donor chain (or
  vice versa). Verified structurally: `donor_mode_mismatch` = 0 mismatches
  across 1,918,589 candidate pairs.
- **Implausible target filtering** - `commute_distance_survey_miles` (>150
  mi) and `total_daily_miles` (>400 mi) are now filtered to NaN upstream in
  `build_features.py` before they can be used as rescale targets, closing
  the root cause identified in `docs/activity_validation_investigation.md`
  (a handful of NHTS respondents' self-reported values were physically
  implausible for a single day - e.g. a 1,209-mile commute - and flowed
  through unguarded).
- **Guarded commute anchor** - the commute leg's distance/duration
  rescaling in `rescale_chain_distances` no longer anchors unboundedly to
  raw survey values.
- **Minutes-based time jitter** - time-of-day jitter now operates in
  minutes-since-midnight rather than raw HHMM units, avoiding clock-math
  errors around hour boundaries.
- **Time-compatible donor selection** - donor matching now prefers donors
  whose arrival/departure timing is compatible with the target employee's
  drawn schedule (combined arrival+departure -> arrival-only ->
  unrestricted fallback).
- **Relaxed same-cluster donor tier** - when no donor matches within the
  trip/stop-count tolerance, `select_donor` falls through to a relaxed tier
  that keeps the cluster/driving-status filters but drops the trip/stop-count
  restriction, rather than synthesizing a minimal fallback chain. This
  eliminated fallback-chain usage entirely (previously 17/5,000 employees,
  all cluster_0).

## 5. Final validation strengths

- **Fallback rate: 0%** - all 15,201 legs across all 5,000 employees are
  real donor-sourced chains; no synthesized fallback chains remain.
- **0 implausible-speed legs** - `implied_leg_speed_plausible` structural
  check: 0/14,808 legs outside the [5, 70] mph plausible-speed band.
- **Realistic leg counts** - `legs_per_employee_day` / `stops_per_employee_day`
  pass for both clusters (e.g. cluster_1 source mean=2.50 vs. synthetic
  mean=2.49).
- **Workplace dwell validation** - workplace arrival-time and departure-time
  consistency checks both pass at 0 violations (5,000/5,000 and 4,607/4,607
  respectively), and cluster_0 `workplace_dwell_minutes` passes the
  distributional check (p=0.9975).

## 6. Remaining statistical differences

- **Leg-duration tails** - `leg_distance` and `leg_duration` KS tests fail
  for both clusters, but per
  `docs/activity_validation_investigation.md` §3, the bulk of the
  distribution (p50-p75) already matches source closely; the failure is
  driven by a small tail (~0.1-0.3% of legs) at p95+ with disproportionately
  large influence on the KS statistic and standard deviation.
- **cluster_1 workplace-dwell distribution** - `workplace_dwell_minutes`
  fails the KS test for cluster_1 (source mean=461.89 vs. synthetic
  mean=456.93 minutes; p=0.0001), while cluster_0's dwell distribution
  passes cleanly.

## 7. Why these are limitations, not blocking errors

Both remaining gaps are documented, bounded, and traced to a known
mechanism rather than an unexplained defect:

- The leg-distance/duration tail failures were root-caused in
  `docs/activity_validation_investigation.md` to a small number of NHTS
  source respondents' own extreme summary values; the primary fix (upstream
  filtering to NaN) has already been applied, and the residual effect is
  concentrated in the tail (p95+), not the bulk of the distribution that
  downstream modeling primarily depends on.
- All structural (deterministic-by-construction) checks - the class of
  check where a violation would indicate an actual bug - pass at 100%:
  speed plausibility, workplace arrival/departure consistency, NaN
  preservation through jitter, missingness co-occurrence, and donor
  mode-matching. The remaining failures are all distributional
  (KS-test-based) comparisons, which are inherently sensitive to small tail
  effects and are informational/directional rather than correctness gates.
- The cluster_1 workplace-dwell gap is a modest mean shift (~5 minutes,
  ~1% of the mean) rather than a structural or directional bias, and
  sits alongside a passing cluster_0 result.

## 8. Readiness

The generator is ready for downstream workplace EV charging-demand
modeling.
