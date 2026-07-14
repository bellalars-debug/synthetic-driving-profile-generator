# Profile-Based Mobility Generation: Test Results

**Scope of this document:** measurement report only, for the experiment designed in
[`profile_based_generation_plan.md`](profile_based_generation_plan.md) §8 and implemented in
`src/driving_profiles/generator/profile_adapter.py`, `profile_based.py`, and
`src/driving_profiles/validation/profile_based.py`. All outputs described below were written under
`data/validation/profile_based/`. No file under `data/processed/` was modified, and no code in this
document was changed to produce a more favorable result.

---

## 1. Purpose

`data/external/nhts_datasetanalysis/driver_profile_analysis/DriverProfiles.csv` provides a detailed,
per-employee activity schedule format — an ordered sequence of `Parked`/`Driving`/`Charging` states,
locations, and clock times — that this pipeline does not itself produce as an input. The prior
assessment (`profile_based_generation_plan.md` §1–§3) established that this schedule *structure* is
well-formed (chronologically valid, 0→24h, no gaps, for all 250 profiles) but that its
`Distance (mi)` and per-leg duration fields are unusable: they are drawn from two statistically
independent distributions with no joint physical constraint, producing implausible speeds on 46% of
the 250 synthetic users.

This experiment tests whether the pipeline can **ingest that external schedule format as an input**,
preserve everything about it that is behaviorally meaningful, and repair only the one part that is
broken — without touching production outputs. It is a compatibility test of the reconciliation policy
(§8 of the plan), not a new independent validation of this pipeline's own distance/duration modeling.

---

## 2. External profile representation

Each of the 250 employees in `DriverProfiles.csv` is represented as an ordered list of activity
segments (rows), each with:

- **`State`** — `Parked`, `Driving`, or `Charging`.
- **`Location`** — `Home`, `Work`, `Other`, `Restaurant`, `Medical`, `Shopping/Errands`, `Gym`,
  `Daycare`, `School`, or `-1` for `Driving` rows.
- **`Start time (hour)` / `End time (hour)`** — decimal hours (confirmed by source-code evidence,
  `profile_based_generation_plan.md` §1), i.e. `7.0286` means 7 h 1.72 min, not "7:02".
- **`Distance (mi)`** — the field this experiment discards (see §4).

This is a full-day activity chain per person, not a single trip record — the same representational
level this pipeline's own donor-leg chains use internally, which is what makes a like-for-like
reconciliation possible.

---

## 3. Features preserved

The reconciliation preserves everything about the external schedule except the two flawed numeric
fields on driving legs. Concretely, per profile:

- The ordered `State` sequence, verbatim.
- Every destination/location category, verbatim and in order.
- The exact number of driving legs (none added, removed, or split).
- The number and ordering of stops.
- Direct-vs-chained commute structure (leg count of the outbound and return chains).
- Midday activity structure (any `Work → ... → Work` sub-chain).
- Parked-window start/end times, except where a chronology adjustment must move the non-anchored
  side of an immediately adjacent window.
- Workplace arrival and departure clock times specifically — the two times the reconciliation's
  anchor rule protects first.
- The overall 0–24h contiguous daily schedule.

In short: sequence, location, and schedule *shape* come entirely from the external profile. Only a
leg's distance and duration are ever replaced.

---

## 4. Why distance and duration were replaced

`profile_based_generation_plan.md` §3 traced the defect to source code: `DriverProfiles.csv`'s
generator samples each driving leg's distance from an archetype-conditional table and its duration
from an unrelated, national-pooled table, with no shared draw or plausibility check tying the two
together. This produces speeds ranging from near-zero to thousands of mph, and it applies uniformly
across direct and chained legs alike (30.2 mph median on direct 2-leg commutes vs. 24.3 mph median on
chained legs — materially the same implausibility rate), so there is no reliable per-leg signal for
"this particular leg's numbers happen to be fine, keep them." Every driving leg's distance and
duration are therefore substituted unconditionally, rather than conditionally filtered — substituting
selectively would silently keep some external legs and not others, undermining the goal of proving
the reconstruction is uniform.

---

## 5. Sourcing physically plausible values from validated NHTS donor legs

Rather than inventing a new speed model, the reconciliation reuses this pipeline's own
already-validated donor-selection machinery — the same relaxed-tier donor pool already confirmed
running at 0% fallback-chain usage in production. For each external driving leg:

1. The leg is tagged with its position (`commute_out` / `midday` / `commute_return`), its
   direct-vs-chained chain type, and its purpose transition (e.g. `home→work`), using the same
   annotation rule applied to both the external profile and the donor pool.
2. A donor leg is matched using a tiered search — tightest tier requires matching purpose transition,
   chain segment, chain type, and time-of-day within 60 minutes; each subsequent tier relaxes one
   constraint, stopping at the first non-empty tier.
3. The donor pool itself is pre-filtered to legs whose own implied speed
   (`TRPMILES / (TRVLCMIN / 60)`) falls in the 5–70 mph plausible band, so a matched leg's distance
   and duration are physically consistent *by construction*, not by a post-hoc check.
4. The external leg's distance and duration are replaced with the matched donor leg's values (with
   duration recomputed from the donor's own plausible speed); the leg's role as an anchor (workplace
   arrival or departure) determines which side of the adjacent parked/charging window absorbs the
   resulting clock-time change.

This is the same donor-leg engine that already supplies distance/duration for the production
synthetic population — no new dataset or model was introduced for this experiment.

---

## 6. Results

Run against all 250 employees in `DriverProfiles.csv`:

| Metric | Result |
|---|---|
| Employees processed | 250 |
| Reconstructed driving legs | 560 |
| Total schedule rows | 1,370 |
| Location-sequence preservation | 100% |
| Workplace arrivals within 5 min | 257 / 259 |
| Workplace arrivals within 15 min | 258 / 259 |
| Workplace arrivals within 30 min | 259 / 259 |
| Workplace departures within 5 min | 258 / 259 |
| Workplace departures within 15 min | 259 / 259 |
| Workplace departures within 30 min | 259 / 259 |
| Mean schedule adjustment | 15.89 minutes |
| Maximum schedule adjustment | 146.11 minutes |
| Driving legs requiring some adjustment | 560 / 560 (100%) |
| Chronological validity | 100% |
| Implausible-speed legs | 0 / 560 |
| Source User IDs leaked into output | 0 |
| Production `synthetic_employees`/`synthetic_activity` parquet files | unchanged |
| Test suite | 405 passing |
| Lint (`ruff`) | clean |

**What the schedule-adjustment numbers mean:** every one of the 560 driving legs needed its
distance/duration substituted (by design — see §4), and substituting a leg's duration necessarily
moves the boundary of whichever adjacent parked/charging window is not protected by the anchor rule.
The mean adjustment of 15.89 minutes shows this is typically a modest nudge to a dwell window, not a
schedule rewrite; the 146.11-minute maximum reflects the rare case where a donor leg's duration
differs substantially from the external profile's original (physically implausible) duration and the
adjacent window has to absorb a correspondingly larger change. The workplace arrival/departure
figures (257–259 out of 259 preserved within 5 minutes, 259/259 within 30 minutes) confirm the anchor
rule is doing its job: even though every leg's timing changed somewhat, the two clock times that
matter most for a person's day — when they arrive at and leave work — were preserved almost exactly,
because the reconciliation deliberately steers adjustment away from those two anchors first.

100% chronological validity and 0/560 implausible-speed legs confirm the reconstruction holds its two
"by construction" guarantees in practice, not just in design. 0 leaked source User IDs and unchanged
production parquet files confirm the experiment is fully isolated from both the external repo's
identifiers and this pipeline's production outputs.

---

## 7. Limitation: same NHTS source

This experiment demonstrates *format compatibility* — that an externally structured activity schedule
can be ingested, preserved, and repaired using this pipeline's own machinery — not a second,
independent validation of that machinery. `nhts_datasetanalysis_assessment.md` already established
that nothing in `data/external/nhts_datasetanalysis` is an independent NHTS sample: `DriverProfiles.csv`'s
schedule backbone and this pipeline's donor-leg pool both ultimately derive from the same underlying
NHTS survey data, just processed differently. Additionally, the schedule *shape* being tested here
(250 templates) still reflects one external generator's own archetype and scheduling choices — its
stop-purpose mix, its dwell-time model — so a good result shows the reconciliation policy works on a
realistic, well-formed external schedule, not that this pipeline's distance/duration modeling has been
cross-validated against an independent population.

---

## 8. Conclusion

The pipeline can ingest a detailed profile-based activity representation, preserve its behavioral
structure, and reconstruct physically plausible mobility without modifying production outputs.
