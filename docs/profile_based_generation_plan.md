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

---

## 8. Refined reconciliation policy (design only — supersedes §6–§7's adapter sketch)

**Status:** design/specification only. Nothing in this section has been implemented; no code has
been written or run against `data/external/nhts_datasetanalysis` or the production pipeline. This
section refines Option D's "adapter logic" (§6) and "roadmap" (§7) into an exact, position-aware
algorithm, per an explicit request to design the reconciliation before writing it. §6/§7's core
conclusion is unchanged (distance/duration must come from this pipeline's own validated donor-leg
data, never from `DriverProfiles.csv`'s `Distance (mi)` column); this section replaces their
coarser "match on purpose + time-of-day band" sketch wherever the two disagree.

**Goal:** implement the external driving activity profiles — not benchmark them, not replace them
with an NHTS donor schedule. `DriverProfiles.csv` contains complete, well-formed activity schedules
(§1–§2: every one of the 250 profiles is chronologically valid, 0→24h, no gaps). The one confirmed
defect (§3) is that `Distance (mi)` and each leg's implied duration are two independently sampled
values with no joint physical constraint, producing impossible speeds on 46% of users. The
reconciliation therefore repairs only the two physically-inconsistent fields per driving leg and
otherwise preserves the external schedule exactly — it must be possible to point at any output leg
and show its sequence, purpose, and (whenever feasible) its clock times came from
`DriverProfiles.csv`, and only its distance/duration came from this pipeline's donor pool.

### 8.1 Preservation contract

Each item the external profile must keep, and what in `DriverProfiles.csv` it maps to:

| # | Item | Source field(s) | How it is preserved |
|---|---|---|---|
| 1 | Ordered `State` sequence (`Parked`/`Driving`/`Charging`) | `State` | Copied verbatim, in order. Never reordered, merged, split, inserted, or dropped. |
| 2 | All destination/location categories | `Location` (`Home`, `Work`, `Other`, `Restaurant`, `Medical`, `Shopping/Errands`, `Gym`, `Daycare`, `School`; `-1` for `Driving` rows) | Copied verbatim, in order. Never reclassified or reordered. Collapsed to `home`/`work`/`other` only as an internal matching key (§8.3), never in the output sequence. |
| 3 | Number of driving legs | count of `State == "Driving"` rows per `User ID` | Fixed at the external count. The reconciliation only ever changes a leg's distance/duration/clock-times — it never adds, removes, or splits a leg. |
| 4 | Number and ordering of stops | non-`Home`/non-`Work` `Location` values and their position | Unchanged — stops are never reordered, added, or dropped. |
| 5 | Direct vs. chained commute structure | leg count of the pre-first-`Work` chain and the post-last-`Work` chain (§8.3) | The leg count of each chain is fixed at the external value; "direct" (1 leg) and "chained" (>1 leg) are structural facts about the external profile, not modeled or altered. |
| 6 | Midday activity structure | any `Work → ... → Work` sub-chain (departs `Work`, later returns to `Work`, before the day's final `Work` departure) | Leg count and destination sequence of every midday sub-chain unchanged. Confirmed present in 9/250 profiles (`Work` appears exactly twice; never more than twice in this file). |
| 7 | Parked-window start/end times | `Start time (hour)` / `End time (hour)` on `Parked`/`Charging` rows | Preserved exactly unless a chronology reconciliation (§8.6) must adjust one side of a window to absorb a duration change — and even then, only the non-anchored side of the *immediately adjacent* window moves (§8.6). |
| 8 | Workplace arrival/departure times | `End time (hour)` of the `Parked`/`Charging` row where `Location == "Work"` begins (arrival); its `Start time (hour)` (departure) | Explicitly the two clock times the anchor rule (§8.6) protects first — adjustments are steered away from these whenever any other placement of the residual is possible. |
| 9 | Overall 0–24h daily schedule | full per-`User ID` timeline | Stays contiguous (no gaps/overlaps) and spans the full day by construction — §8.6 only ever changes an existing segment's duration, never removes, inserts, or reorders segments. |

### 8.2 What is discarded, and why only two fields

Per driving leg, only `Distance (mi)` and the implied duration (`End time (hour) − Start time
(hour)`, in the `Driving` row) are discarded and replaced — never partially kept or blended. §3
established that the independent-sampling defect is architectural, applying uniformly across
direct and chained legs alike (30.2 mph median / 12.7% >100 mph on direct 2-leg commutes vs. 24.3
mph median / 13.3% >100 mph on chained legs — materially the same rate). There is no reliable
per-leg signal to decide "this particular leg's numbers happen to be physically fine, keep them" —
so every driving leg's distance and duration are substituted, unconditionally, rather than
conditionally filtered by an implied-speed check on the *external* value (which would silently keep
some external legs and not others, undermining the "prove the reconstruction is uniform" goal).

### 8.3 Leg annotation — deriving position/role, shared by both sides

Before matching, every driving leg — on **both** the external profile and this pipeline's donor
pool — is tagged with the same four attributes, computed by the same rule applied to each side's
own leg sequence. This symmetry (one annotation rule, two inputs) is what makes "leg position" and
"direct vs. chained structure" usable as matching keys at all.

**Step 1 — locate workplace anchors.** Within one person's chronologically ordered legs, find every
leg whose destination purpose is `work` (external: `Location == "Work"` on the following `Parked`/
`Charging` row; donor: `trip_purpose == "work"`, i.e. `build_donor_legs`'s existing
`classify_trip_purpose` collapse of `WHYTRP1S`). Call these **work-occurrence legs**, in order:
`w_1 ... w_k` (`k ≥ 1` — confirmed 250/250 external profiles have at least one, at most two, `Work`
occurrence; the donor pool must be similarly checked but is expected to satisfy this since
`summarize_donor_chains` already requires `has_work_leg`).

**Step 2 — assign `chain_segment`.** Every leg falls into exactly one of:
- `commute_out` — every leg from the start of the day up to and including `w_1`.
- `midday_i` — every leg strictly between `w_i` and `w_{i+1}`, for each consecutive pair of
  work-occurrence legs (only possible for `i < k`; only observed for `k = 2` in this file, i.e. at
  most one midday segment per profile, but the rule generalizes to any `k`).
- `commute_return` — every leg after `w_k` to the end of the day.

**Step 3 — assign `leg_index_in_segment` / `chain_type`.** Within a `chain_segment`, legs are
numbered 1..`n` in chronological order; `chain_type = "direct"` when `n == 1`, else `"chained"`.
This is exactly item 5/6 of the preservation contract, expressed as a matching key rather than
prose.

**Step 4 — assign `purpose_transition`.** For each leg, `(origin_purpose, destination_purpose)`,
each collapsed to `{home, work, other}` (external: `Location` collapsed the same way `Sources.csv`'s
own `WHYTRP1S`-derived categories already are — `Home→home`, `Work→work`, everything else→`other`;
donor: the existing `trip_purpose` column). `origin_purpose` is the previous leg's
`destination_purpose`, or `home` for the first leg of the day (every profile and every valid donor
chain starts at `Home`, confirmed for the external file in §1/§2 and already assumed by
`rescale_chain_times`'s own arrival-leg logic for donors).

**Step 5 — `is_arrival_at_work` flag.** `destination_purpose == "work"` — true exactly for `w_1
... w_k` themselves. Used only to pick the anchor direction (§8.6), not as a matching key (it's
already implied by `purpose_transition`).

### 8.4 Donor pool and tiered matching

**Pool.** Reuse `build_donor_legs(trips_clean, employee_clusters)` unchanged — same chronological-
validity filter, same `MAX_PLAUSIBLE_LEG_MILES` cap — since that is "the existing validated donor
machinery" the policy is required to reuse, not a new dataset. From its output, keep only legs with
`is_driving_leg == True` (a leg-level restriction, tighter than `select_donor`'s whole-donor
`has_driving_leg`, since this reconciliation borrows single legs, not whole chains — a donor whose
day otherwise had no driving-mode trips is irrelevant here, but so is a *non-driving* leg belonging
to a donor who drove elsewhere that day). Additionally restrict to legs whose own implied speed
(`TRPMILES / (TRVLCMIN / 60)`) falls in `[MIN_PLAUSIBLE_SPEED_MPH, MAX_PLAUSIBLE_SPEED_MPH]` (the
same 5–70 mph constants `rescale_chain_distances` already validates against) — filtering at
pool-construction time, rather than relying on a post-hoc fallback, is what makes "100% of
substituted legs are speed-plausible" true *by construction*, with the `ASSUMED_AVERAGE_SPEED_MPH`
duration fallback (§8.5) reduced to a defensive branch expected to fire at ~0% (mirroring the
already-validated 0% fallback-chain rate for whole-donor selection). Tag every pool leg with
`chain_segment` / `chain_type` / `purpose_transition` (§8.3, computed once from `donor_legs`'
existing per-person grouping) and its own `STRTTIME` (minutes since midnight) for time-of-day
matching. No `cluster_id` restriction is applied — external profiles have no cluster assignment,
and nothing in the preservation contract calls for one.

**Tiers.** For each external driving leg, search progressively wider tiers, stopping at the first
non-empty one (same "widen only until non-empty, never let a later widening override an earlier
match" philosophy as `MATCH_TOLERANCES` / the Tier A/B/C time preference in `select_donor`):

| Tier | Requires match on | Time-of-day restriction |
|---|---|---|
| 1a | `purpose_transition` + `chain_segment` + `chain_type` | `\|leg start − candidate STRTTIME\| ≤ 60 min` |
| 1b | same as 1a | ≤ 120 min |
| 1c | same as 1a | unrestricted |
| 2 | `purpose_transition` + `chain_segment` (drop `chain_type`) | unrestricted |
| 3 | `purpose_transition` only (drop `chain_segment`) | unrestricted |
| 4 | `destination_purpose` only (drop `origin_purpose`) | unrestricted |

Tier 4 is expected to never be empty in practice (`home`/`work`/`other` destination categories are
common in the real-trip donor pool at every time of day), so it is the guaranteed terminus — there
is no synthesized/fallback leg for this reconciliation the way `build_fallback_chain` exists for
whole-donor selection, because unlike whole-chain donor selection (bounded by `cluster_id` ×
`has_driving_leg`, which *can* be empty for a sparse cluster), this leg-level pool is unrestricted by
cluster and is drawn from the full plausible-and-driving donor-leg universe. If a future run ever
does exhaust Tier 4 (e.g. an unexpected destination category), that leg is flagged
`distance_duration_source = "external_unrepaired"` and its original `Distance (mi)`/duration are
kept verbatim with an explicit implausibility flag — a documented last resort, not the expected
path.

**Selection within a tier.** Sort the tier's candidates by `(HOUSEID, PERSONID, TRIPID)` and draw
one uniformly with a seeded `rng.integers(len(candidates))` — the same reproducible tie-break
convention `_select_by_time_preference` already uses, so a fixed seed reproduces the same
reconstruction. A single donor leg is sampled (not a tier-wide mean/median) so the substituted
distances retain realistic donor-to-donor variance instead of flattening to a point estimate.

### 8.5 Distance and duration reconciliation

For the matched donor leg:

```
new_distance_mi = donor_leg.TRPMILES
donor_speed_mph = donor_leg.TRPMILES / (donor_leg.TRVLCMIN / 60)
if MIN_PLAUSIBLE_SPEED_MPH <= donor_speed_mph <= MAX_PLAUSIBLE_SPEED_MPH:
    new_duration_min = new_distance_mi / donor_speed_mph * 60
else:
    new_duration_min = new_distance_mi / ASSUMED_AVERAGE_SPEED_MPH * 60   # expected ~0% of legs
```

This is the existing `rescale_chain_distances` per-leg plausibility branch (lines 605–621 of
`generator/activity.py`), reused verbatim rather than re-derived — the same validated constants
(`MIN_PLAUSIBLE_SPEED_MPH = 5`, `MAX_PLAUSIBLE_SPEED_MPH = 70`, `ASSUMED_AVERAGE_SPEED_MPH = 30`)
this pipeline already ships. Because the §8.4 pool is pre-filtered to plausible-speed legs, the
`else` branch is a defensive guard, not the normal path.

### 8.6 Chronology reconciliation — the anchor rule and cascade

Replacing a leg's duration moves the boundary between it and one of its two neighboring
Parked/Charging windows. Which side moves is decided by a single rule, then a cascade handles the
rare case where the immediately adjacent window can't absorb the whole change.

**Anchor rule.** For each driving leg:
- If `is_arrival_at_work` (destination is a `Work` occurrence, `w_1 ... w_k`) — **anchor the leg's
  end time** (preserves item 8: workplace arrival). The **preceding** Parked/Charging window absorbs
  the duration change (its own start time, and everything before it, stays untouched).
- Otherwise (every other leg, including the leg that departs `Work` — `commute_out`'s non-final
  legs, `midday` departures, and every `commute_return` leg) — **anchor the leg's start time**
  (preserves item 8's other half: workplace departure, and item 7 generally). The **following**
  Parked/Charging window absorbs the change (its own end time, and everything after it, stays
  untouched).

This single rule is why both halves of item 8 (arrival *and* departure) are protected: an arrival
leg and the leg that immediately departs from that same `Work` window are two different legs with
opposite anchor directions, so the `Work` window's own two boundary times are each the *anchored*
side of one of its neighboring legs and are therefore never touched by an ordinary (non-cascading)
adjustment.

**Cascade (only when the immediately adjacent window can't absorb the full change).** The adjacent
window's duration is clamped to a floor, `MIN_DWELL_FLOOR_MINUTES` (a new constant, proposed at 1.0
minute, matching the order of magnitude of the shortest real donor legs already tolerated elsewhere
in this pipeline). If the required change exceeds what clamping to the floor allows:
1. The unabsorbed residual ripples forward (start-anchored case) or backward (arrival-anchored
   case) as a uniform time translation applied to every subsequent (or preceding) leg and window,
   preserving each of their own internal durations — only their clock placement shifts.
2. The ripple stops the moment it reaches the next **protected anchor** — any workplace arrival or
   departure time, or the day boundary (`0:00`/`24:00`) — without needing to move it.
3. If the ripple reaches a protected anchor and residual still remains, that anchor itself is
   shifted by the minimal leftover amount. This is the one case where a "whole day" shift can occur,
   and it is logged distinctly (`anchor_shifted = true`, with the shift in minutes) — expected to be
   rare, since realistic donor-leg durations (bounded by `MAX_PLAUSIBLE_LEG_MILES` and the 5–70 mph
   band) rarely exceed a stop's original external dwell time by more than a few tens of minutes.

Because the reconciliation only ever changes existing segment durations (never inserts, removes, or
reorders), the timeline stays contiguous and 0→24h by construction (item 9), independent of how far
any given cascade travels.

### 8.7 Per-leg audit record

Every reconstructed driving leg carries:

| Field | Meaning |
|---|---|
| `sequence_source` | Always `"external"` in this design — no leg is invented or reordered. |
| `schedule_status` | `"preserved"` if this leg's own departure/arrival clock times are unchanged from `DriverProfiles.csv`, else `"adjusted"`. |
| `distance_duration_source` | `"nhts_donor"` (normal case) or `"external_unrepaired"` (Tier-4-exhausted last resort, §8.4). |
| `match_tier` | Which of Tiers 1a–4 supplied the donor leg (diagnostic only). |
| `adjustment_minutes` | Signed minutes by which this leg's non-anchored boundary moved from the external value (`0` when `schedule_status == "preserved"`). |
| `anchor_shifted` | `true` only on the rare cascade case (§8.6, step 3) where a protected workplace anchor itself had to move; carries its own shift in minutes. |

### 8.8 Validation metrics

- **% of location sequences preserved exactly** — always 100% by construction (the `Location`/
  `State` sequence is never modified); reported as an assertion-style check, not a tolerance metric.
- **% of workplace arrival times preserved within 5 / 15 / 30 minutes** — computed across all
  work-occurrence legs (`w_1 ... w_k`, every profile), comparing reconstructed vs. original `End
  time (hour)` of that `Work` window.
- **% of workplace departure times preserved within 5 / 15 / 30 minutes** — same, on the `Start time
  (hour)` of each `Work` window's departure.
- **Mean and maximum schedule adjustment** — mean/max of `abs(adjustment_minutes)` across all
  non-anchored window boundaries (i.e., every reconstructed leg's one adjustable side).
- **% of driving legs whose original schedule required adjustment** — share of legs with
  `schedule_status == "adjusted"` vs. `"preserved"`.
- **Chronological validity** — 100% of reconstructed timelines remain monotonic, non-overlapping,
  and span 0→24h (should be 100% by construction, per item 9 — measured anyway to catch a
  cascade-logic defect rather than assumed).
- **Implied speed plausibility** — 100% of substituted legs' `new_distance_mi` /
  `new_duration_min` fall in `[MIN_PLAUSIBLE_SPEED_MPH, MAX_PLAUSIBLE_SPEED_MPH]` (should be 100% by
  construction per §8.4/§8.5's pre-filtering, again measured rather than assumed).

### 8.9 Updated implementation roadmap (supersedes §7's adapter-logic bullets)

Not implemented — planning only, per instruction to design before coding.

- **New leg annotation step** (§8.3), applied identically to `build_donor_legs`' output and to a
  newly parsed `DriverProfiles.csv` frame — one shared function, two call sites.
- **New leg-level donor pool** (§8.4): `build_donor_legs` output, filtered to `is_driving_leg` and
  in-band implied speed, tagged with `chain_segment`/`chain_type`/`purpose_transition`/`STRTTIME`.
  Distinct from (and simpler than) `donor_summary`'s whole-chain, cluster-scoped table — no new
  external inputs.
- **New per-leg matching function** (§8.4): the Tier 1a–4 search, reusing the seeded-draw
  reproducibility convention already established in `_select_by_time_preference`.
  distance/duration reconciliation reusing `rescale_chain_distances`'s existing plausibility-branch
  constants verbatim (§8.5).
- **New chronology reconciliation function** (§8.6): anchor-rule + cascade, operating purely on
  minutes-since-midnight, with `MIN_DWELL_FLOOR_MINUTES` as its one new constant.
- **Outputs:** unchanged from §6 — written only under `data/validation/profile_based/`, never to
  `data/processed/synthetic_employees.parquet` or `data/processed/synthetic_activity.parquet`.
- **Required tests:** everything in §6/§7's guard-test list, plus: (a) anchor-protection test — for
  every profile, every workplace arrival/departure time is either unchanged or the shift is recorded
  under `anchor_shifted`, never silently moved; (b) cascade-floor test —
  `MIN_DWELL_FLOOR_MINUTES` is never violated (no window duration goes negative or below the floor);
  (c) leg-count/location-sequence byte-identity test between input and output per profile (item
  1–4 of the preservation contract); (d) the validation metrics of §8.8 computed and asserted
  against target thresholds before this experiment is considered to demonstrate the policy works.
