# Activity Distribution Failure Investigation

Investigation only - no generation logic changed. Scope: the `leg_distance`,
`leg_duration`, `departure_time_minutes`, `arrival_time_minutes` KS-test
failures in `docs/validation_results.md` §2 (activity profile validation),
following up on the donor mode-mismatch fix (`37b0c0b`, `020642f`). Uses the
committed `data/processed/synthetic_activity.parquet` /
`data/processed/synthetic_employees.parquet` and the real donor pool
(`generator/activity.py`'s `build_donor_legs`, reused read-only).

## 1. Summary

The four failing metrics are not four independent problems. All of them are
downstream of the same mechanism: a small number of synthetic legs (roughly
0.1-0.3% of donor-sourced legs) take on physically implausible absolute
magnitude - single legs of 200-1200+ miles and up to 10,538 minutes (~7.3
days) - and those few legs dominate the standard deviation, which is what the
KS test is picking up (the bulk of the distribution, p50-p75, already
matches source closely). The root trigger is upstream of `generator/`
entirely: a handful of real NHTS respondents have self-reported/derived
travel values (`GCDWORK`-sourced `commute_distance_survey_miles`, and
`TRPMILES`-summed `total_daily_miles`) that are themselves implausible for a
single day, and no stage of the pipeline caps them before they're used as
rescale targets.

| rank | mechanism | evidence | scope |
|---|---|---|---|
| 1 | Extreme, uncapped source outliers in `commute_distance_survey_miles`/`total_daily_miles` flow through as rescale *targets* | source max commute = 1209 mi, max total_daily_miles = 612 mi; both pass through jitter almost unchanged | root cause - everything below is this value hitting under-guarded code |
| 2 | `rescale_chain_distances` anchors the commute leg to that target **unscaled**, with no upper bound | `commute_distance_survey_miles=1201.5` -> leg `distance=1201.51` directly | activity.py:484-485 |
| 3 | The `[5,70]` mph plausibility guard only catches *scale-induced* speed distortion, not a preserved-but-absurd donor ratio at extreme absolute magnitude | 1201.5 mi / 10,537.7 min = 6.84 mph - "plausible," still ~7.3 days in one leg | activity.py:494-507 |
| 4 | When `trips_per_day` is low (58.6% of donor-sourced employees have <=2), the entire "remaining budget" (`total_daily_miles` minus commute) concentrates onto a single non-commute leg | `SYN-00004312`: total=623.7, commute=120.3, remaining 503.3 all on one "home" leg | activity.py:487-493 |
| 5 (separate, minor) | Fallback chains (`build_fallback_chain`) never pass through the speed-plausibility guard at all | `SYN-00004370`: distance=0, duration=931.4 min; `SYN-00004259`: 1.89 mi / 949.5 min (0.12 mph) | activity.py:540-585 |
| 6 (secondary amplifier) | Jitter (`sample.py`) sigma is `0.15 x within-cluster std`, and that std is itself inflated by the same outliers, occasionally pushing values *past* the source's own max | `total_daily_miles`: 2/2363 synthetic values exceed source's max (623.7, 618.9 vs. 612.0) | sample.py:205-214 |

## 2. Which legs are causing the inflated SDs

Not "many slightly-off legs" - a handful of extreme ones. Excluding just the
37 donor-chain legs (out of 15,203, i.e. 0.24%) whose `distance` exceeds the
donor pool's own observed max (132.5 mi), and the same for `duration` vs. the
donor pool's max (735.0 min):

| metric | full synthetic SD | SD excluding those outlier legs | source SD |
|---|---:|---:|---:|
| leg_distance (pooled donor legs) | 23.21 | 13.70 | ~12-16 (per cluster/purpose) |
| leg_duration (pooled donor legs) | 104.05 | 39.14 | ~18-27 (per cluster) |

37 legs (0.24% of 15,203) account for the majority of the `leg_distance` SD
inflation; the picture is even starker for `leg_duration`, where 14 legs
(0.09%) - all with duration > the donor pool's max of 735 min - drop the SD
from 104.05 to 39.14, i.e. they're responsible for more than half of the
remaining excess over source (source SD is ~18-27).

Worst offenders (all donor-sourced, all pass the `[5,70]` mph guard):

| synthetic_employee_id | trip_purpose | distance (mi) | duration (min) | implied mph |
|---|---|---:|---:|---:|
| SYN-00002338 | work | 1201.51 | 10537.74 | 6.84 |
| SYN-00001547 | work | 391.46 | 3592.74 | 6.54 |
| SYN-00002826 | work | 874.41 | 2129.88 | 24.63 |
| SYN-00004020 | work | 391.04 | 1572.56 | 14.92 |
| SYN-00001122 | work | 406.62 | 1301.21 | 18.75 |
| SYN-00004849 | other | 200.54 | 1257.81 | 9.57 |
| SYN-00004312 | home | 503.34 | 876.77 | 34.44 |

Every one of these traces back to an extreme `commute_distance_survey_miles`
or `total_daily_miles` draw for that employee (§3), not to a donor-selection
or jitter-noise coincidence.

## 3. Source vs. synthetic percentiles

Donor-sourced legs only (`chain_source == "donor"`), the population the four
failing metrics are measured on. p50/p75 track source closely everywhere;
the gap opens at p95+ and is worst at p99 - exactly where a handful of
extreme legs would show up.

**leg_distance (mi)**

| group | p50 src/syn | p75 src/syn | p90 src/syn | p95 src/syn | p99 src/syn |
|---|---|---|---|---|---|
| cluster_0 | 7.04 / 6.62 | 15.33 / 14.83 | 25.09 / 24.63 | 33.84 / 33.38 | 54.98 / 56.11 |
| cluster_1 | 5.09 / 4.43 | 13.00 / 13.45 | 26.40 / 27.16 | 39.71 / 41.64 | 77.71 / 106.43 |

**leg_duration (min)**

| group | p50 src/syn | p75 src/syn | p90 src/syn | p95 src/syn | p99 src/syn |
|---|---|---|---|---|---|
| cluster_0 | 20.00 / 17.22 | 30.00 / 34.69 | 45.00 / 60.01 | 60.00 / 90.00 | 80.00 / 198.99 |
| cluster_1 | 15.00 / 14.68 | 30.00 / 30.00 | 45.00 / 56.06 | 60.00 / 89.16 | 100.00 / 209.05 |

**departure_time_minutes**

| group | p50 src/syn | p75 src/syn | p90 src/syn | p95 src/syn | p99 src/syn |
|---|---|---|---|---|---|
| cluster_0 | 855 / 869 | 1020 / 1049 | 1140 / 1187 | 1230 / 1321 | 1365 / 1439 |
| cluster_1 | 825 / 836 | 1020 / 1054 | 1125 / 1251 | 1200 / 1371 | 1320 / 1439 |

**arrival_time_minutes**

| group | p50 src/syn | p75 src/syn | p90 src/syn | p95 src/syn | p99 src/syn |
|---|---|---|---|---|---|
| cluster_0 | 878 / 890 | 1050 / 1075 | 1155 / 1212 | 1245 / 1344 | 1380 / 1439 |
| cluster_1 | 845 / 857 | 1040 / 1076 | 1150 / 1275 | 1230 / 1397 | 1335 / 1439 |

Reading this: `leg_distance`/`leg_duration` are a **tail** problem
(p50-p75 essentially match; p99 and max are where it breaks). The
timing metrics are a **milder, broader** shift (every percentile from p75 up
runs 15-150 minutes hot, and p99 pins at 1439 = the `minutes_to_hhmm` clamp,
i.e. multiple employees' shifted times are hitting the end-of-day ceiling).
The timing drift is a distinct, smaller-magnitude issue from the
distance/duration blowup - see §4.

## 4. Attribution by candidate mechanism

- **Extreme source outliers (primary root cause).** `commute_distance_survey_miles`
  comes from `GCDWORK` (`build_features.py:191`), a per-respondent
  great-circle work-distance field carried straight through with no
  plausibility cap anywhere in `clean.py`/`build_features.py`. Source max is
  1209.4 mi (mean 11.3, sd 37.3 - already a very heavy tail). `total_daily_miles`
  is `TRPMILES` summed over driving-mode legs (`build_features.py:260`);
  source max is 612.0 mi. Critically, `MAX_PLAUSIBLE_LEG_MILES` (150 mi) is
  only applied when building the **donor pool** (`build_donor_legs`, dropping
  a respondent entirely if any single leg exceeds it) - it is never applied
  to `commute_distance_survey_miles`/`total_daily_miles` themselves, so an
  employee whose derived summary value came from a >150mi leg still carries
  that value forward as a rescale *target*, even though a donor with that
  same leg would have been rejected. 6/2241 (0.27%) source employees exceed
  150mi on `commute_distance_survey_miles`; 25/1170 (2.14%) do on
  `total_daily_miles`. Confirmed by tracing the outputs: `SYN-00002338`'s
  `commute_distance_survey_miles=1201.51` is the (lightly jittered) source
  respondent's actual `GCDWORK=1209.45` value, the single largest value in
  the whole source population.

- **`rescale_chain_distances` (primary amplifier).** Per its own docstring,
  the commute leg is "used as-is" (activity.py:484-485) with no upper bound -
  whatever `commute_distance_survey_miles` says becomes that leg's `distance`
  verbatim, regardless of magnitude. The proportional "remaining budget"
  scaling (activity.py:487-493) has the same property: `other_scale =
  remaining_budget / donor_other_sum` is unbounded, so when `total_daily_miles`
  is an outlier and the donor chain has few/small "other" legs, the scale
  factor is large and lands entirely on whichever legs exist. This is the
  function that actually turns an upstream data problem into an
  implausible-looking leg.

- **The speed-plausibility guard is necessary but insufficient.** It compares
  `implied_speed_mph` against `[5, 70]` and re-derives duration from a flat
  30 mph assumption when that fails - but when a *donor's own* recorded
  distance/duration ratio is already within `[5,70]` (e.g., a slow 6.8 mph
  in-town trip), scaling both distance and duration by the same large factor
  preserves that ratio exactly, so the guard never fires even though the
  resulting absolute duration (10,537 min = 7.3 days) cannot fit in a
  single-day chain. All 14 legs with duration exceeding the donor pool's own
  max (735 min) pass this guard. The guard checks a *rate*, not a magnitude,
  and the function has no check at all for "does this leg fit inside one
  day" or "is this leg's duration plausible in absolute terms."

- **Low trip counts concentrate the effect.** 2,915/4,976 (58.6%) of
  donor-sourced employees have `trips_per_day <= 2`, meaning as few as one
  non-commute leg exists to absorb `remaining_budget`. This doesn't create
  outliers by itself, but it's why the outlier `total_daily_miles` cases
  (e.g. `SYN-00004312`, `SYN-00004499`, `SYN-00004435`, all `cluster_1`,
  `trips_per_day` 1-4) show up as a *single* extreme leg rather than being
  diluted across many legs - it's a magnitude multiplier on root cause 1, not
  an independent mechanism.

- **Fallback chains bypass the speed guard entirely (separate, smaller
  issue).** `build_fallback_chain` (activity.py:540-585) uses the employee's
  own `commute_duration_minutes`/`commute_distance_survey_miles`/
  `total_daily_miles` directly with no speed check at all (that check lives
  only in `rescale_chain_distances`, which fallback chains never call).
  Concretely: `SYN-00004370` gets `distance=0, duration=931.4` and
  `SYN-00004259` gets `distance=1.89, duration=949.5` (0.12 mph) - both are
  almost certainly already among the 22 structural
  `implied_leg_speed_plausible` violations in `docs/validation_results.md`
  §2. Fallback is only 0.48% of employees overall but 2.91% of `cluster_1`
  (24/824), and `cluster_1` is exactly the high-mileage/long-commute archetype
  where a 2-leg minimal chain (mean 2.00 legs vs. source mean 5.97) is least
  representative - this is why `legs_per_employee_day`/`stops_per_employee_day`
  fail hard for `cluster_1`/fallback in the report, though that specific
  failure is "expected by construction" per the module docstring, not a bug
  in itself.

- **Jitter is a minor secondary amplifier, not the driver.** `sample.py`'s
  `JITTER_SCALE=0.15` is applied to each cluster's own within-cluster std,
  which is itself inflated by these same outliers (cluster_0's
  `commute_distance_survey_miles` std = 39.98, driven up by the 1209-mile
  row) - so jitter noise is somewhat wider than a "clean" distribution would
  warrant, and it occasionally pushes an already-extreme value past the
  observed source max (`total_daily_miles`: 2/2363 synthetic values exceed
  source's 612.0 max, at 623.7 and 618.9). But jitter did not *create* any of
  the extreme values in §2/§3's worst-offender table - those are
  (near-)unmodified real respondent values. Fixing jitter alone would not
  resolve the failures.

- **Donor selection is not implicated.** `select_donor`'s trip/stop-count
  matching and the now-fixed driving-mode requirement (`020642f`) operate
  correctly here; the donor chosen for e.g. `SYN-00002338` is a normal
  cluster_0 chain shape-matched donor. The problem is entirely in what target
  values that donor's legs get rescaled *to*, not which donor was picked.

- **Timing metrics (`departure_time_minutes`/`arrival_time_minutes`) are a
  distinct, smaller issue**, not explained by the distance/duration
  mechanism above. The drift is broad (every percentile from p75 up runs
  hot) rather than tail-dominated, and `p99` pinning at exactly 1439 in three
  of four groups points at the `minutes_to_hhmm` end-of-day clamp
  (activity.py:174-187) being hit by a non-trivial share of employees, likely
  because `work_departure_time` jitter (raw HHMM units, clamped only to
  `[0, 2359]` in `sample.py`, not to a plausible clock progression) pushes
  some employees' shifted late-chain legs past midnight, where they clip
  instead of naturally wrapping or wpressed shorter. This deserves its own
  targeted look (not scoped further here) but is unrelated to root causes
  1-4 above.

## 5. Recommended fixes, ranked by importance

1. **Cap or trim implausible `commute_distance_survey_miles`/`total_daily_miles`
   values before they become rescale targets** (upstream, in
   `build_features.py`/`clean.py`, or as an explicit documented exclusion
   rule applied consistently to both the donor pool *and* the employee
   summary values it's rescaling toward). Currently `MAX_PLAUSIBLE_LEG_MILES`
   protects only the donor pool; the same standard should decide what's a
   usable *target*, not just what's a usable *donor*. This is the actual
   root cause - fixing only the two items below would still let extreme
   targets exist, just handled more gracefully.

2. **Add an explicit upper bound (or a documented re-derivation rule) to the
   commute-leg anchor and the total-distance rescale in
   `rescale_chain_distances`.** "Used as-is" is fine for normal values but
   has no safety net for a bad target; an employee-level plausibility check
   (e.g., cap `commute_distance_survey_miles` at some multiple of the
   cluster's own commute distribution, or refuse to anchor above
   `MAX_PLAUSIBLE_LEG_MILES` and fall back to the donor's own commute value
   scaled by a bounded factor instead) would directly fix the worst
   `leg_distance` outliers regardless of whether item 1 is also done.

3. **Extend the plausibility guard from "implied speed" to "implied
   duration relative to a single day."** A 6.84 mph implied speed is fine in
   isolation; a 10,537-minute leg is not, for a dataset that models one
   calendar day. A magnitude check (e.g., a single leg's duration can't
   exceed some fraction of `MINUTES_PER_DAY`) would catch the cases the
   speed-ratio guard structurally cannot.

4. **Apply the same speed/magnitude guard to `build_fallback_chain`.**
   It currently has zero plausibility checking; even a small fix (skip the
   `FALLBACK_*` constants' bypass and run the same guard `rescale_chain_distances`
   uses) would remove the 0.12 mph and 0-mph fallback legs from the
   structural violation count.

5. **Revisit whether `total_daily_miles`'s "remaining budget" should ever be
   concentrated onto a single leg.** Lower priority than 1-3 since it's a
   magnitude multiplier, not an independent source of implausibility - but
   worth a minimum-legs-to-spread-across rule or a per-leg cap once items
   1-3 shrink how large "remaining budget" can get in the first place.

6. **Investigate the `JITTER_SCALE`/within-cluster-std feedback loop
   separately** (lower priority, smaller effect size) - once item 1 trims
   the outliers that inflate `cluster_std`, jitter's contribution to the
   tail should shrink on its own; only worth a dedicated look if it doesn't.

7. **Timing metrics deserve their own follow-up** (out of scope for this
   investigation) - likely centers on `work_departure_time` jitter combined
   with the `minutes_to_hhmm` end-of-day clamp, not the distance/duration
   mechanism above.

## 6. Fix implemented: item 1 (upstream plausibility bounds)

Implemented in `src/driving_profiles/features/build_features.py`
(`build_commute_features`/`build_daily_mobility_features`); tests added in
`tests/test_build_features.py`. **`generator/activity.py`/`generator/sample.py`
are untouched** - donor selection, jitter, and rescaling logic are exactly
as they were; only the upstream feature values they consume are now bounded.

### What was filtered

Two fields, each pass-through/derived from a raw NHTS variable with no prior
plausibility cap:

| field | source | new bound | above-bound values become |
|---|---|---:|---|
| `commute_distance_survey_miles` | `GCDWORK` (survey great-circle estimate) | `MAX_PLAUSIBLE_COMMUTE_MILES = 150.0` mi | NaN |
| `total_daily_miles` | `TRPMILES` summed over driving-mode legs | `MAX_PLAUSIBLE_TOTAL_DAILY_MILES = 400.0` mi | NaN |

Filtering (not clipping) to NaN was chosen because NaN already has an
established, gracefully-handled meaning throughout this pipeline
("not applicable" - `sample.py`'s jitter skips it, `rescale_chain_distances`
falls back to the donor's own raw value when a target is NaN); clipping to
a hard cap would instead pile many different respondents onto one artificial
value and manufacture a new distributional spike.

When `total_daily_miles` is filtered, `total_driving_minutes` and
`average_trip_distance_miles` are set to NaN in lockstep (not
independently re-checked against their own bound) - `validation/missingness.py`
enforces that these three columns are always null together (0 rows with a
partial 1/3 or 2/3 null pattern, checked on both source and synthetic
populations); filtering only one would have introduced the exact kind of
partial-null inconsistency that check exists to catch.
`commute_distance_trip_miles` (a separate `TRPMILES`-based cross-check
against `GCDWORK`, per `build_commute_features`'s docstring) is deliberately
*not* filtered alongside `commute_distance_survey_miles` - the two are
independent signals, and only the survey-based one is used downstream as a
jitter/rescale input.

### Why these thresholds

Both bounds were chosen from two converging lines of evidence per field,
not an arbitrary round number:

- **`commute_distance_survey_miles` = 150.0 mi.** Reuses
  `generator/activity.py`'s existing `MAX_PLAUSIBLE_LEG_MILES` (150.0) - the
  standard the donor pool already enforces for "a single one-way trip
  distance that's a representative local-commute template" - rather than
  inventing an independent number, directly per this investigation's own
  recommendation ("the same standard should decide what's a usable target,
  not just what's a usable donor"). It also independently lands in this
  field's own empirical gap: the clustered population's values cluster up
  to 156.9 mi, then jump to 396.3 mi+ with nothing in between - 150 cleanly
  separates the two groups.
- **`total_daily_miles` = 400.0 mi.** No existing project constant applies
  (it's a whole-day sum across potentially several legs, so the single-leg
  cap above would be too strict). Grounded in: (1) empirical - the clustered
  population's values cluster up to 356.8 mi, then jump to 456.4 mi+ with
  nothing in between; (2) physical - FMCSA hours-of-service rules cap a
  single day's driving at 11 hours, and covering 400 mi in that time needs
  only a ~36 mph blended city/highway average (a conservative, easily
  achievable speed) - so 400 mi/day is a generous but physically groundable
  ceiling for "local daily driving," sitting comfortably above even the
  long-commute cluster's own observed mean+3sd (cluster_1: mean=73.75,
  sd=87.65 -> mean+3sd=336.7).

### How many records were affected

Measured against the full `employee_features.parquet` output (all weekday
workers, not just the clustered subset that reaches `sample.py`):

| field | non-null records | records filtered | % filtered | of which in the clustered population reaching generation |
|---|---:|---:|---:|---:|
| `commute_distance_survey_miles` | 3,214 | 9 | 0.28% | 6 |
| `total_daily_miles` | 3,853 | 25 | 0.65% | 5 |

Both are small minorities of their respective populations, consistent with
§2/§4's finding that this is a tail problem affecting a handful of records,
not a systematic distributional issue.

### Verification

`pytest tests/` (240 tests, including 8 new tests covering: filtered above
the bound, kept exactly at the bound, not filtered when plausible, the
independent `commute_distance_trip_miles` cross-check staying unaffected,
and the three-column co-occurring-null lockstep) and `ruff check src/ tests/`
both pass clean. `data/processed/employee_features.parquet` and every
downstream artifact (`employee_clusters.parquet`, `synthetic_employees.parquet`,
`synthetic_activity.parquet`) still reflect the pre-fix pipeline run - they
are not regenerated as part of this change; re-running the pipeline
(`build_features.py` -> `cluster.py` -> `sample.py` -> `generator/activity.py`)
end to end is a separate, deliberate next step, not done here so its blast
radius (re-running clustering/sampling/rescaling on updated inputs) can be
reviewed on its own.
