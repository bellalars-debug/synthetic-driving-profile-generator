# Profile-Based Generation Plan: DriverProfiles.csv / CombinedDriverProfiles.csv

**Scope of this document:** investigation only. No files in `data/external/nhts_datasetanalysis`
were modified, no external scripts were executed, and no production pipeline files were touched.
This builds on the provenance/leakage work already done in
[`nhts_datasetanalysis_assessment.md`](nhts_datasetanalysis_assessment.md) — that document
established the repo is same-source (not an independent NHTS sample); this document goes one level
deeper into whether `DriverProfiles.csv` specifically is *physically valid* at the individual-leg
level, which the prior assessment did not check.

**Bottom line up front:** the time encoding is confirmed decimal hours by direct source-code
evidence — this is not an encoding-artifact problem. The extreme implied speeds are a **real defect
in the external repo's generator**: driving distance and driving duration are sampled from two
statistically independent distributions with no joint physical constraint, and — for duration
specifically — the generator ignores an archetype-conditional duration table that already exists in
its own `archetype_params.json` in favor of an unrelated national-pooled distribution. 46% of the
250 synthetic users (115/250) have at least one driving leg outside a 5–70 mph plausibility band.
`DriverProfiles.csv` is 100% synthetic model output, not observed NHTS trip chains — it is
unsuitable as a distance/duration source but its state/location/schedule backbone is usable.
`CombinedDriverProfiles.csv` is genuine weighted-NHTS aggregate data, but it is a **coarsening
hierarchy**, not a flat partition — sampling all 97 rows by weight without respecting the hierarchy
double-counts the population.

---

## 1. True time encoding

**Source-code evidence (unambiguous, two independent call sites):**

- `lbnl_model/lbnl_sim.py:829-830` (writer used by `lbnl_model/outputs/DriverProfiles.csv`):
  ```python
  "Start time (hour)": round(a.start_min / 60.0, 4),
  "End time (hour)": round(a.end_min / 60.0, 4),
  ```
- `driver_profile_build.py:567-568` (writer used by the file actually in question,
  `driver_profile_analysis/DriverProfiles.csv`):
  ```python
  "Start time (hour)": round(a.start_min / 60, 4),
  "End time (hour)": round(a.end_min / 60, 4),
  ```

Both divide `start_min`/`end_min` (absolute minutes-past-midnight, as defined on the `Activity`
dataclass, `lbnl_sim.py:308-315`) by 60 and round to 4 decimals. **`Start time (hour)` / `End time
(hour)` are plain decimal hours** — `7.0286` means 7 h + 0.0286×60 = 7 h 1.72 min, not "7:02" or any
HH.MM-style value. This is the interpretation already used in the numbers reported in the prompt.

**Quantitative test of the plausible alternative (HH.MM-style, i.e. treat the fractional part ×100
as minutes) — run on the actual 560 driving rows, not selected for favorable results:**

| Interpretation | Duration median (min) | Duration p10 (min) | Speed p50 | p75 | p90 | p95 | p99 | max | <5 mph | >70 mph | >100 mph |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A. Decimal hours (as coded)** | 18.51 | 6.22 | 28.5 mph | 59.8 | 121.4 | 198.4 | 476.0 | 2,848 mph | 72 (12.9%) | 115 (20.5%) | 72 (12.9%) |
| B. HH.MM-style (fraction×100 = minutes) | 20.92 | **-7.78** | 16.7 mph | 41.8 | 98.5 | 207.6 | 601.4 | 3,749 mph | 166 (29.6%) | 83 (14.8%) | 55 (9.8%) |

Candidate B is not just worse on average — it is **structurally invalid**: interpreting the decimal
part as minutes requires every fractional value to be `< 0.60`. In the actual data, 210/560 driving
rows (37.5%) have a fractional part that rounds to a "minute" value above 60 (max 99.62), which is
impossible for HH.MM notation, and it produces negative durations at the 10th percentile. Decimal
hours is the only interpretation that is both source-code-confirmed and internally consistent.
**Conclusion: the encoding is correct. The bad speeds are a genuine data-generation problem, not a
misread column.**

---

## 2. Observed vs. synthetic

`driver_profile_analysis/DriverProfiles.csv` is produced by `driver_profile_build.py:499-504`:

```python
import lbnl_model.lbnl_sim as S
sim = S.LBNLSimulation(250, "Lawrence Berkeley National Laboratory", ...)
sim.run()
emps = [p.employee for p in sim.profiles]
```

This is **the same `LBNLSimulation` / `DrivingProfileGenerator` engine** that produces
`lbnl_model/outputs/DriverProfiles.csv`, just invoked with a fixed `parking_count=250` from a
different driver script and a different RNG draw (the two files have identical schema/shape but are
row-for-row different — confirmed by diff). It is not a second, independently observed dataset.

- **User IDs:** `eid = f"LBNL_EMP_{i:04d}"`, then remapped to sequential `UserID = 1..250`
  (`lbnl_sim.py:798`). **Not real NHTS respondent IDs** — purely synthetic sequence numbers.
- **Distances:** `emp.commute_distance_mi = self.arch.commute_distance(aid)`
  (`lbnl_sim.py:619`), which inverse-CDF-samples from `archetype_params.json`'s
  `commute_distance_pct` — a real, NHTS-derived, archetype-conditional percentile table. So the
  distance *marginal* is grounded in observed NHTS trip distances, but each individual value is an
  independent random draw from that marginal, not an observed trip.
- **Durations:** `emp.commute_duration_min = self.arch.commute_duration(aid)`
  (`lbnl_sim.py:620`) → `ArchetypeSampler.commute_duration()` (`lbnl_sim.py:230-237`):
  ```python
  def commute_duration(self, aid):
      # national office row (heaped median 20); falls back to a lognormal ...
      if self._natl_dur:
          return max(3.0, self.s.sample_percentiles(self._natl_dur, cap=0.99))
  ```
  This **ignores `aid` (the archetype) entirely** and always draws from the single national-pooled
  office-worker duration table, with its own independent `rng.uniform(0,1)` draw. Notably,
  `archetype_params.json` *does* contain an archetype-specific `commute_duration_pct` field per
  archetype (confirmed present, e.g. archetype `A01F`) — **it is computed but never used**. Each
  employee's duration is drawn from a distribution completely decoupled from the same employee's
  distance draw, with no shared random seed, percentile rank, or joint model tying them together.
  The code's own comment (`lbnl_sim.py:146-149`) acknowledges the design choice: "Duration is only
  weakly archetype-specific (it tracks distance), so it is drawn from the national row while
  DISTANCE stays archetype-conditional" — i.e., the independence is intentional, not an oversight,
  but it has no joint-plausibility safeguard.
- **Charging state / `P_max`:** added downstream by `_park()` (`lbnl_sim.py:606-613`) purely from a
  location→power lookup (`LOC_PMAX_W`) and `emp.vehicle.is_ev` — not from any NHTS field.

**Validity classification:**
- ❌ Not valid as observed donor chains — zero individual records are observed; every value is a
  model draw.
- ✅ Valid as a synthetic *activity-chain template* (state/location/schedule sequence) — this part
  is built by separate, robust scheduling logic (`_park`/`_drive`, contiguity enforced structurally)
  and is internally consistent (every profile 0→24h, no gaps — confirmed for all 250 users).
- ✅ Valid as a benchmark output for cross-generator comparison (as already used in
  `nhts_datasetanalysis_assessment.md` §5).
- ❌ Unsuitable as training/calibration input for distance or duration, because those two fields are
  independently sampled with no joint constraint (see §3).

The repo's own validation (`Driver_Profile_Attribute_Analysis_Report.md` §17: "9/9 checks pass")
only checks **distributional marginals** — income/age/sex/distance/commute/fuel shares against
NHTS — never per-leg speed or distance/duration joint consistency. This exact defect was never
caught by the source repo's own test suite. Separately, `METHODOLOGY_REPORT.md:76` notes that the
*upstream NHTS trip-speed analysis* (a different stage, `analyze.py`) does clip computed speed to
1–90 mph "to drop coding outliers" — so the toolkit's authors clearly know speed needs bounding, but
that safeguard was never carried into the `DriverProfiles.csv` generator.

---

## 3. Distance–duration inconsistency

**Root cause, confirmed from code (`lbnl_sim.py:619-620`, `230-237`, `223-228`):** distance and
duration for a driving leg are two separate random draws from two separate distributions:
`commute_distance()` draws from an archetype-conditional percentile table; `commute_duration()`
draws from a national-pooled percentile table, regardless of archetype. There is no rank
correlation, shared draw, or plausible-speed check tying them together. This holds even for the
simplest direct 2-leg commutes, not only chained/detour legs:

| Leg group | n | median speed | % >100 mph | % <5 mph |
|---|---|---|---|---|
| 2-leg (direct commute) | 402 | 30.2 mph | 12.7% | 13.4% |
| 3–4 leg (stop/detour/midday) | 158 | 24.3 mph | 13.3% | 11.4% |

Roughly equal implausibility rates in both groups confirm the defect is architectural (independent
sampling everywhere), not specific to detour-leg splitting.

**Checked mechanisms, with findings:**

- **Distance duplicated for outbound/return:** Confirmed, by design, not a bug per se. Evening
  distance is explicitly set `De = emp.commute_distance_mi` (same value as morning, `lbnl_sim.py:648`,
  comment: "same home<->work route"). Verified empirically: **all 201/201 (100%) of users with
  exactly 2 driving legs have byte-identical morning/evening distances.** Duration (`Me =
  self.arch.commute_duration(aid)`, line 649) is independently *re-drawn* for the return leg, so
  even a physically consistent morning leg can pair the same distance with an implausible evening
  duration, and vice versa.
- **Duration independently sampled:** Confirmed above — this is the primary root cause.
- **Minimum trip durations imposed:** Confirmed, and this compounds the problem for detour legs.
  `commute_duration()` floors at `max(3.0, ...)` minutes; when a stop is present, that duration is
  split `per_t = M * 1.20 / 2` (`lbnl_sim.py:632`), so a floored 3.0-minute base duration produces a
  1.8-minute leg — the observed minimum leg duration in the file is exactly **1.8 minutes** (34/560
  legs, 6.1%, are under 3.5 minutes). Distance for that same split leg is drawn independently and
  has no matching floor, so short legs can carry large distances.
- **Time rounding:** Not a meaningful contributor — rounding is to 4 decimal places (~0.004 min),
  far too small to explain speeds in the hundreds/thousands of mph.
- **Start/end times from arrival/departure probability tables:** Confirmed as the source of
  `depart_home`, `depart_work` (`lbnl_sim.py:239-251`, from `depart_home_pct`/`depart_work_pct`
  archetype tables) — these anchor the *schedule*, not the driving-leg duration, which is instead
  the independently-sampled `commute_duration_min` described above.
- **Distance/duration from different profile attributes:** Effectively yes — distance comes from an
  archetype-specific field, duration from a national-pooled field never conditioned on the same
  archetype.

**Quantified impact:**
- Users with **at least one** implausible leg (speed <5 or >70 mph): **115 / 250 (46.0%)**.
- Users where **every** driving leg satisfies `duration > 0`, `distance >= 0`, and `5 ≤ speed ≤ 70`
  mph: **135 / 250 (54.0%)**. No leg has `duration <= 0` or `distance < 0` — the failures are purely
  speed-implausibility, not missing/negative values.

---

## 4. Evaluate possible uses

| Option | Scientific validity | Data retained | Circularity risk | Inherited-error risk | What it demonstrates |
|---|---|---|---|---|---|
| **A. Use directly as donors** | Invalid — 46% of users carry at least one physically impossible leg; using the raw distance/duration fields would inject impossible speeds straight into any downstream feature or model that consumes them. | 100% (1,370 rows) | Low (not circular — it's wrong for a different reason) | **High** — directly inherits the independent-sampling defect | Nothing valid; would need to be caught later by validation anyway |
| **B. Keep only fully-plausible profiles** | Valid on the retained subset, but the retained 135/250 (54%) is not a random subsample — archetypes with longer archetype-conditional distances are more likely to collide with the national-pooled (generally short) duration draw, so filtering plausibly biases the sample toward shorter/typical commutes. | 54% of users, and only those with **no** long/short-tail archetype draws | Low | Low on retained rows, but **selection bias** risk | That a filtered, physically-consistent subset of profile-based schedules can be produced — but the subset is not representative of the archetype mix |
| **C. Keep schedule, recalculate duration/distance from a documented speed model** | Valid if the speed model is transparent and cited, but introduces a new, project-authored assumption (an implied-speed-by-purpose or -by-distance-band curve) not present in NHTS itself — this project would be inventing the joint distribution the external repo failed to model. | 100% of profiles, all legs correctable | Low | **Medium** — replaces one unvalidated assumption with another, self-authored one | That the profile's schedule *shape* is usable once paired with *any* internally-consistent speed model — doesn't prove the model is right |
| **D. Keep only state/location/schedule sequence, substitute this pipeline's own validated NHTS donor legs for distance/duration** | Valid — reuses this project's already-validated donor-selection machinery (0% fallback-chain usage per the current production pipeline) instead of inventing a new speed model, and never touches the flawed `Distance (mi)`/duration fields at all. | 100% of profiles (schedule backbone is independently well-formed for all 250 users — verified 0-gap, 0→24h) | Low | **Lowest** — distance/duration never come from the external repo | That an externally-sourced schedule *shape* (state sequence, location sequence, start/end timestamps) is compatible with this pipeline's own donor-leg engine — a genuine input-format compatibility test |
| **E. Benchmark only, not an input** | Valid, safest, zero contamination | 0% used as input (comparison only) | None | None | Cross-generator distributional consistency (already partially done in §5 of `nhts_datasetanalysis_assessment.md`) — adds no new training signal |

**Assessment:** A is disqualified outright. B is defensible but discards nearly half the data and
skews the retained sample. C is defensible but requires authoring and defending a new speed model —
a nontrivial, separate piece of methodology. **D is the best fit for "test profile-based input
compatibility without contaminating the pipeline"**: it takes only the one part of
`DriverProfiles.csv` that is well-formed (the state/location/schedule backbone), discards the part
that is broken (distance/duration), and reuses machinery this pipeline has already validated rather
than inventing new assumptions. E remains valuable as a parallel, zero-risk comparison and should be
run alongside D, not instead of it.

---

## 5. `CombinedDriverProfiles.csv`

**Structure:** 97 rows, columns include `Unweighted n`, `Weighted population`, `Weighted population
share %`, `Effective sample size`, `Source variables` (literally lists `HHFAMINC, R_AGE, R_SEX,
TRPMILES, WHYTO, WTPERFIN, WTTRDFIN`), and a `Cross-class level` column with values `L0`/`L1`/`L2`/`L3`.
This is a genuine weighted NHTS cross-tabulation, not a simulation artifact — every numeric column
traces to a named NHTS variable and a Kish effective-n calculation, consistent with the "Aggregated
observed" tier already established in `nhts_datasetanalysis_assessment.md` §2.

- **Do the archetypes overlap?** **Yes.** `Cross-class level` shows a coarsening hierarchy (84 rows
  at `L3`/finest, 9 at `L2`, 2 at `L1`, 2 at `L0`/coarsest). Confirmed concretely: for the
  `High ($150k+)` × `25-39` cell, row `DP003` (`Sex=Mixed`, `Distance=Mixed`, level `L1`, share
  0.81%) coexists with `DP004`–`DP007` (`L3`, sex- and distance-specific children of that same
  cell). `DP003` is a **marginal roll-up of its own children**, not a disjoint segment.
- **Can weighted shares be sampled directly?** **Not naively.** Summing `Weighted population share
  %` across all 97 rows lands at 100.03% — which looks like a clean partition but is coincidental:
  it works only because `driver_profile_build.py`'s own report states each of the 250 synthetic
  drivers is matched to "exactly one archetype... through the same coarsening hierarchy" (`Driver_
  Profile_Attribute_Analysis_Report.md:36`) — i.e., the hierarchy is meant to be walked top-down,
  taking the *finest available* cell per person and backing off only when a cell is too sparse
  (`n<30`). Flatly sampling all 97 rows by weight (ignoring `Cross-class level`) double-counts
  anyone captured by both a parent and a child row.
- **Are the means/probabilities observed or synthetic?** **Observed, weighted NHTS summaries** —
  `Direct-commute prob`, `Trip-chain prob`, `Midday-trip prob`, `Mean Car-Trip Distance (mi)`,
  `Mean Daily Miles` are all cell-level weighted statistics computed directly from NHTS trip
  records grouped by the cross-classification, not model output.
- **Usable even though `DriverProfiles.csv` has bad legs?** **Yes, for archetype-level marginals
  only.** The archetype table's own numbers (mean distance, mean daily miles, chain probabilities)
  are unaffected by the downstream generator bug — that bug is introduced only when `lbnl_sim.py`
  independently *samples* an individual distance and an individual duration per synthetic driver.
  Using `CombinedDriverProfiles.csv` to draw population *shares* or *cell-level means* (respecting
  the coarsening hierarchy) is legitimate; using it as a source of individual driving-leg
  distance/duration pairs is not — it has no per-record data of that shape at all.

---

## 6. Recommended safest experiment

**Profile-based schedule adapter test, using Option D, run entirely under
`data/validation/profile_based/`.**

- **Inputs:** `data/external/nhts_datasetanalysis/driver_profile_analysis/DriverProfiles.csv`
  (state/location/schedule backbone only) — **read-only**.
- **Fields used:** `User ID`, `State`, `Start time (hour)`, `End time (hour)`, `Location`. **Fields
  explicitly excluded:** `Distance (mi)`, `Nothing`, `P_max (W)`, `NHTS HH Wt` — these either carry
  the confirmed defect or are downstream artifacts not needed to test schedule compatibility.
- **Adapter logic:** for each `User ID`, reconstruct the `Parked`/`Driving`/`Charging` state
  sequence and location sequence exactly as given (already verified 0→24h, no gaps, for all 250
  users — this part of the source file is trustworthy). For each `Driving` segment, keep the
  segment's clock start/end time as the *scheduling slot* but discard its `Distance (mi)`, and
  instead attach a distance/duration pair from this pipeline's own validated NHTS donor-leg pool
  (the same relaxed-tier donor-selection machinery already confirmed running at 0% fallback usage),
  matched on trip purpose (inferred from the `Location` transition, e.g. `Home→Work`) and
  time-of-day band. This never touches the external repo's flawed fields.
- **Outputs:** write only to `data/validation/profile_based/` (e.g.
  `profile_schedule_adapter_output.parquet`), never to `data/processed/synthetic_employees.parquet`
  or `data/processed/synthetic_activity.parquet`.
- **Validation metrics:** (1) 100% of reconstructed timelines remain 0→24h contiguous after
  substitution; (2) resulting implied speed for every substituted leg falls in a documented
  plausible range (sanity check on the *substitution*, not a re-test of the source file); (3)
  resulting distance/duration marginals compared against this pipeline's existing NHTS-office-worker
  validation targets; (4) resulting chain-type shares (direct vs. stop) compared against
  `CombinedDriverProfiles.csv`'s `Direct-commute prob`/`Trip-chain prob`, using the correct
  coarsening-hierarchy lookup per synthetic driver (§5), as an external structural cross-check.
- **Required tests:** (a) guard test asserting no output row's distance/duration was sourced from
  the external `Distance (mi)` column (prevents silent regression back to Option A); (b) byte-diff
  test confirming `data/processed/synthetic_employees.parquet` and
  `data/processed/synthetic_activity.parquet` are unchanged before/after the experiment runs; (c)
  contiguity test on all 250 reconstructed timelines.
- **Expected limitations:** only 250 schedule templates available (small n relative to the
  production pipeline's population); the schedule backbone still reflects one external generator's
  archetype/scheduling choices (e.g. its stop-purpose mix, its dwell-time model), so this tests
  *format and structural compatibility*, not a new independent validation of this pipeline's own
  distance/duration modeling — consistent with the "same-source, not independent" conclusion of
  `nhts_datasetanalysis_assessment.md`.

Run **Option E (benchmark-only) in parallel** at no extra risk: compare this pipeline's own
`synthetic_activity.parquet` distributions against `DriverProfiles.csv`'s *distributions* (not
individual legs) for schedule-shape metrics (dwell durations, chain-type share, departure-time
histogram) where the independent-sampling defect doesn't apply — this adds a second, free
cross-check without any adapter code.

---

## 7. Implementation roadmap (not implemented — planning only)

- **Exact input files:**
  `data/external/nhts_datasetanalysis/driver_profile_analysis/DriverProfiles.csv` (schedule
  source, read-only); this pipeline's existing donor-leg pool (already in
  `data/processed`/`data/raw` per the current production pipeline — no new external input needed
  for distances).
- **Exact fields used:** `User ID`, `State`, `Start time (hour)`, `End time (hour)`, `Location`
  from `DriverProfiles.csv`, joined against donor legs on purpose (derived from `Location`
  transition) and time-of-day band.
- **Filtering/reconstruction policy:** no row-level filtering by speed needed (Option D discards the
  flawed fields entirely rather than filtering the 46% of implausible users) — every one of the 250
  schedules is usable structurally, since only its state/location/timing (which is well-formed) is
  retained; `CombinedDriverProfiles.csv` used only via the coarsening-hierarchy lookup, never a flat
  weight sample.
- **Adapter output schema:** one row per reconstructed activity segment —
  `user_id, state, start_hour, end_hour, location, purpose (derived), distance_mi (from donor leg),
  duration_min (from donor leg), source = "profile_schedule+donor_leg"` — written under
  `data/validation/profile_based/`.
- **Validation metrics:** contiguity (0→24h, no gaps/overlaps) on 100% of reconstructed timelines;
  implied-speed sanity check on substituted legs; distance/duration marginals vs. existing
  NHTS-office-worker targets; chain-type share vs. `CombinedDriverProfiles.csv` (hierarchy-respecting
  lookup).
- **Required tests:** guard against reuse of `Distance (mi)`; byte-diff protection on the two
  production parquet files; contiguity test; purpose-matching join-coverage test (what fraction of
  segments find a donor-leg match vs. fall back, and at what rate — mirroring the fallback-chain
  metric already tracked for the production donor-selection pipeline).
- **Expected limitations:** small template count (250); schedule shape still inherits one external
  generator's stop/dwell modeling choices; this is a structural-compatibility test, not new
  independent validation evidence (per §6 of `nhts_datasetanalysis_assessment.md`, nothing in this
  external repo is a second, independent NHTS sample).
