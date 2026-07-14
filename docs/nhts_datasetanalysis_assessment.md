# Assessment: Can `data/external/nhts_datasetanalysis` Be Used to Test My Pipeline?

**Scope of this document:** read-only assessment. No files in `data/external/nhts_datasetanalysis`
were modified, no scripts in that repository were executed, and no changes were made to this
project's pipeline or data.

**Bottom line up front:** this repository was built by a different analyst from the **exact same
2022 NHTS national public-use file release** that this pipeline already uses (`hhv2pub.csv`,
`perv2pub.csv`, `tripv2pub.csv`, `vehv2pub.csv` — identical row counts: 7,893 households / 16,997
persons / 31,074 trips / 14,684 vehicles). It is **not an independent data source**. Every
"observed" table in it is a re-derivation of the same source population my pipeline already
draws from. Its synthetic outputs (`SyntheticEmployees.csv`, `DriverProfiles.csv`, etc.) are a
**different generator's** synthetic population, not observed data at all. This shapes every
recommendation below: nothing in this repo can serve as an *independent generalization test*.
It can serve narrower, still-useful purposes — described in §3, §5, and §7.

---

## 1. Strongest candidate test files — profile

| File | Rows | Cols | Grain | Identifier(s) | Data type |
|---|---|---|---|---|---|
| `driver_profile_analysis/DriverProfiles.csv` | 1,370 (250 users × ~5 segments) | 9 | one row per activity segment within a driver-day; 250 distinct `User ID`s | `User ID` | **Synthetic** — rule/archetype-generated 24h timelines |
| `driver_profile_analysis/CombinedDriverProfiles.csv` | 97 | 24 | one row per archetype (income × age × sex × distance cross-class) | `Archetype ID` | **Aggregated** — observed NHTS cross-tab summary (means, shares, effective n) |
| `driver_profile_analysis/csv_exports/Driver_Profiles.csv` | 1,370 | 9 | identical grain/content to `DriverProfiles.csv` above | `User ID` | **Synthetic** (CSV export of the same table) |
| `driver_profile_analysis/csv_exports/Combined_Driver_Profiles.csv` | 97 | 24 | identical grain/content to `CombinedDriverProfiles.csv` | `Archetype ID` | **Aggregated** (CSV export of the same table) |
| `activity_profile_library/All_Activity_Profiles.csv` | 761 (151 categories × 4–5 segments) | 11 | one row per timeline segment within a **representative template category** (not a person) | `Profile ID` + `Attribute Type`/`Category` | **Aggregated → synthesized template** — one deterministic timeline per category built from weighted-median NHTS values (explicitly "deterministic, not sampled" per the repo's own README) |
| `activity_profile_library/TripPurpose_Activity_Profiles.csv` | 60 (12 purposes × 4–5 segments) | 9 | one row per timeline segment within a trip-purpose template | `Profile ID` | Same as above, purpose-anchored subset |
| `lbnl_model/outputs/DriverProfiles.csv` | 1,370 | 9 | same schema as driver_profile_analysis version but **byte-different content** (separate generator run) | `User ID` | **Synthetic** |
| `lbnl_model/outputs/DrivingProfiles.csv` | 594 | 13 | one row per trip leg, 250 employees | `employee_id`, `trip_id` | **Synthetic** (different seed/run than `SyntheticTrips.csv`) |
| `lbnl_model/outputs/DrivingProfiles_trips.csv` | 560 | 13 | one row per trip leg, 250 employees | `employee_id`, `user_id`, `trip_id` | **Synthetic** |
| `lbnl_model/outputs/SyntheticEmployees.csv` | 250 | 28 | one row per synthetic employee | `employee_id` (`LBNL_EMP_####`) | **Synthetic** |
| `lbnl_model/outputs/SyntheticTrips.csv` | 560 | 13 | one row per trip leg | `employee_id`, `trip_id` | **Synthetic** |

Notes on grain confirmation:
- `DriverProfiles.csv`/`Driver_Profiles.csv`: verified 250 unique `User ID`s, 5–7 rows each, states restricted to `{Parked, Driving, Charging}`, timelines run 0→24h.
- `All_Activity_Profiles.csv`: verified 151 unique `Profile ID`s (matches the repo's documented "151 representative profiles" — 7 income + 6 age + 2 sex + 6 distance + 12 purpose + 3 commute-type + ~115 archetype, before rounding overlaps).
- `SyntheticEmployees.csv`: verified 250 unique `employee_id`s, all prefixed `LBNL_EMP_`, home locations literally labeled `"Synthetic Home Location 000N"`, office fixed to `"Lawrence Berkeley National Laboratory, Berkeley, California / Alameda County"`.

---

## 2. Provenance

**Confirmed via `README.md`, `METHODOLOGY_REPORT.md`, `driver_profile_analysis/Sources.csv`, and
byte-identical row counts against my own `data/raw/*.csv` + `data/raw/manifest.json`:**

- Survey: **2022 NextGen NHTS**, national public-use files, same four files my pipeline ingests
  (`hhv2pub`, `perv2pub`, `tripv2pub`, `vehv2pub`).
- Row counts match exactly: 7,893 households / 16,997 persons / 31,074 trips / 14,684 vehicles —
  identical to `data/raw/hhv2pub.csv` (7,894 lines incl. header), `perv2pub.csv` (16,998),
  `tripv2pub.csv` (31,075), `vehv2pub.csv` (14,685) in this project. This is the single official
  ORNL release, not a resampled or restricted-use variant — anyone who downloaded the 2022 public
  file gets the same records.
- The external repo's own git history shows a single commit (`432f662`, "NHTS synthetic
  driving-profile toolkit + representative-template library"), remote
  `github.com/ashishstephen777-debug/nhts_datasetanalysis`, dated 2026-07-13 — a separate analyst's
  independent build against the same public download, not a fork of or dependency on this project.

Classification of every candidate file by provenance tier:

| Tier | Files | Description |
|---|---|---|
| **Aggregated observed** | `CombinedDriverProfiles.csv` / `Combined_Driver_Profiles.csv` (`ArchetypeDefinitions.csv`), `Driving_Profile_Probability_Tables.csv`, everything under `tables/`, `VALIDATION.csv` | Weighted statistics computed directly from the raw NHTS microdata (means, medians, shares, effective-n). Directly observed, but pre-aggregated — no per-record microdata survives. |
| **Aggregated → deterministic template** | `All_Activity_Profiles.csv`, `TripPurpose_Activity_Profiles.csv`, and all other `activity_profile_library/*` sheets | One synthesized timeline per category, built by inserting weighted-median NHTS values into a fixed template shape. Not sampled, not a population — a single "typical day" per category. |
| **Fully synthetic (this repo's own generator)** | `DriverProfiles.csv` (both copies), `DriverCharacteristics.csv`, `lbnl_model/outputs/*` (all files) | Model output from `lbnl_sim.py`, seeded (seed=42 for the canonical run), validated against NHTS marginals but containing zero observed individual records. |
| **Raw observed microdata** | *(none present)* | The repository does **not** redistribute `hhv2pub.csv`/`perv2pub.csv`/`tripv2pub.csv`/`vehv2pub.csv` themselves — confirmed absent from the file listing. Only derived products are checked in. |

**Key implication:** there is no raw or record-level observed data anywhere in this repository
that isn't already a rebuild of the exact NHTS 2022 file my pipeline reads from `data/raw/`. The
"observed" tables here are aggregates of the same underlying records, not a second, independent
observed dataset.

---

## 3. Scientific validity of the five proposed experiments

**A. Use observed driver profiles from this repository as a second source population.**
Not valid as an *independent* second source — there are no record-level observed driver profiles
here at all (see §2); the closest analog, `CombinedDriverProfiles.csv`, is an aggregate of the
identical NHTS 2022 file already backing my pipeline. Using it as a "second population" would
silently compare my pipeline to itself under a different aggregation scheme, not to new data.

**B. Use observed trip-level records as an external holdout.**
Not available. No trip-level *observed* microdata (record-per-trip with individual `TRPMILES`,
`WHYTO`, timestamps, etc.) is redistributed in this repo — only pre-aggregated distributions
(`tables/`) and fully synthetic trip files (`lbnl_model/outputs/*Trips.csv`). A holdout requires
individual records withheld from training; none exist here that aren't already inside my own
`data/raw/tripv2pub.csv`.

**C. Use aggregated probability tables to calibrate my generator.**
Technically possible but **not scientifically informative** as a validation step, and risky as a
calibration step: `Driving_Profile_Probability_Tables.csv`, `ArchetypeDefinitions.csv`, and the
`tables/` directory are recomputed from the same source records my pipeline's `analyze`/`clean`
stages already consume. Calibrating against them would not add information — at best it
reproduces numbers I can already regenerate from `data/raw/`, at worst it introduces a second,
independently-coded (and therefore only approximately reconciled) implementation of the same
weighting logic as a hidden dependency.

**D. Compare my synthetic output against this repository's synthetic output.**
**This is the one genuinely valid use of this repository.** `SyntheticEmployees.csv` /
`SyntheticTrips.csv` / `DriverProfiles.csv` are a *different pipeline's* synthetic population,
built with different modeling choices (rule-based archetype cross-classification vs. whatever
donor/cluster method my pipeline uses — see `data/processed/synthetic_activity.parquet`'s
`chain_source == "donor"` column). Comparing distributions between two *independently coded*
synthetic generators that both target the same NHTS 2022 marginals is a legitimate
implementation cross-check. It proves consistency between two modeling approaches, not
generalization to new data (both are ultimately grounded in the same source survey).

**E. Use it only as a format-adapter test.**
Valid and low-risk. The `activity_profile_library` and `lbnl_model/outputs` schemas (columns,
timeline conventions, `Parked/Driving/Charging` states, 0→24h contiguous segments) are a
reasonable independent schema to test format conversion / ingestion code against, without
treating the content as new evidence.

---

## 4. Field mapping to my pipeline's schema

### Employee-level features

My pipeline (`data/processed/employee_features.parquet` / `employee_clusters.parquet`) expects:
`commute_distance_survey_miles`, `commute_distance_trip_miles`, `commute_duration_minutes`,
`work_arrival_time`, `work_departure_time`, `trips_per_day`, `total_daily_miles`,
`total_driving_minutes`, `number_of_stops`, `vehicles_per_driver`,
`vehicle_per_driver_adequate`, `used_household_vehicle`, `cluster_id`.

| My field | `SyntheticEmployees.csv` | `CombinedDriverProfiles.csv` (archetype-level) | `DriverCharacteristics.csv` (250-driver) |
|---|---|---|---|
| `commute_distance_survey_miles` | unavailable (no separate survey-reported distance field; only `commute_distance_mi`, which is trip-derived) | unavailable | unavailable (`Commute Distance (mi)` is trip-derived) |
| `commute_distance_trip_miles` | directly available (`commute_distance_mi`) | derivable (`Mean Car-Trip Distance (mi)`, archetype mean not per-record) | directly available (`Commute Distance (mi)`) |
| `commute_duration_minutes` | directly available (`commute_duration_min`) | unavailable | directly available (`Commute Duration (min)`) |
| `work_arrival_time` | directly available (`arrive_work`) | unavailable | directly available (`Typical Arrival Time`) |
| `work_departure_time` | directly available (`depart_work`) | unavailable | directly available (`Typical Work Departure Time`) |
| `trips_per_day` | directly available (`n_trips`) | derivable (`Typical Daily Trip Count`, archetype mean) | unavailable (no explicit count field) |
| `total_daily_miles` | derivable (sum `SyntheticTrips.csv` `distance_miles` per `employee_id`) | derivable (`Mean Daily Miles`, archetype mean) | directly available (`Typical Daily Miles`) |
| `total_driving_minutes` | derivable (sum trip `duration_minutes` per employee) | unavailable | unavailable |
| `number_of_stops` | derivable (`n_trips` − 2, approx.) | unavailable | unavailable |
| `vehicles_per_driver` | unavailable (one `vehicle_id` per employee assumed; no household vehicle count field) | unavailable | unavailable |
| `vehicle_per_driver_adequate` | unavailable | unavailable | unavailable |
| `used_household_vehicle` | unavailable (no household-vehicle-sharing flag) | unavailable | unavailable |
| `cluster_id` | unavailable (uses `archetype_id`, a different taxonomy, not directly comparable to my `cluster_id`) | unavailable (`Archetype ID` is a different scheme) | unavailable (`Archetype ID`, same caveat) |

### Activity-level fields

My pipeline (`data/processed/synthetic_activity.parquet`) expects: trip number, departure time,
arrival time, trip purpose, distance, duration, workplace-arrival/departure flags, workplace
dwell minutes.

| My field | `SyntheticTrips.csv` / `DrivingProfiles_trips.csv` | `All_Activity_Profiles.csv` / `TripPurpose_Activity_Profiles.csv` |
|---|---|---|
| trip number (`trip_number`) | derivable (`seq`, present in `DrivingProfiles.csv`/`DrivingProfiles_trips.csv`; absent from `SyntheticTrips.csv` but derivable from row order per `employee_id`) | unavailable (segments are `Parked`/`Driving` states within a template, not enumerated trips) |
| departure time | directly available (`depart_time`) | directly available (`Start Time`, in decimal hours, for `Driving` rows) |
| arrival time | directly available (`arrive_time`) | directly available (`End Time`, decimal hours) |
| trip purpose | directly available (`purpose`, but phrased as e.g. `"Home->Work commute"` rather than a single-word category — derivable with a string-parse) | directly available for `TripPurpose_Activity_Profiles.csv` (`Category`); derivable elsewhere via `Attribute Type`/`Location` transitions |
| distance | directly available (`distance_miles`) | directly available (`Distance (mi)`) |
| duration | directly available (`duration_minutes`) | derivable (`End Time` − `Start Time`, ×60) |
| workplace-arrival/departure flags | unavailable directly; derivable by matching `destination`/`origin` == the fixed office name string | derivable by matching `Location == "Work"` transitions |
| workplace dwell minutes | derivable (`stop_duration_minutes` on the work leg, or gap between consecutive commute trips) | directly available (duration of the `Parked` row with `Location == "Work"`) |

**Overall fit:** the *synthetic* files (`SyntheticEmployees.csv`, `SyntheticTrips.csv`) map
reasonably well structurally (most fields directly available or one join/derivation away), but
they describe a **different generator's synthetic population**, not observed ground truth — see
§6. The *aggregated* files (`CombinedDriverProfiles.csv`, activity templates) map poorly at the
individual-record level because they carry only per-category means/medians, not per-person
values — most employee-level fields are "unavailable" at that grain.

---

## 5. Recommended experiment

**Use this repository only as Option E (format-compatibility / schema-adapter test), extended
into a limited Option D (cross-generator structural comparison) — do not use it for anything
framed as external validation or generalization.**

- **Exact source files:** `lbnl_model/outputs/SyntheticEmployees.csv` (employee grain, 250 rows)
  and `lbnl_model/outputs/SyntheticTrips.csv` (trip grain, 560 rows), joined on `employee_id`.
  These are the most structurally complete synthetic files and the only ones with a natural
  employee→trips join.
- **Exact input grain:** one synthetic employee with its associated trip legs — mirrors my own
  `employee_features` ↔ `synthetic_activity` relationship.
- **Train/calibration split:** none — no training occurs against this data. This is a
  post-hoc structural/distributional comparison, not a model-fitting exercise.
- **Validation holdout:** none — nothing here is held out from or fed into my pipeline's
  training; it is an entirely separate artifact compared side-by-side.
- **Target synthetic population size:** generate an equivalent **250-employee** run from my own
  pipeline (matching the external repo's population size) to keep sample-size effects out of any
  distributional comparison (e.g., via K-S test power).
- **Features to compare (distributional, not row-matched):** commute distance, commute duration,
  morning/evening direct-commute share, trips/day, daily miles, EV/PHEV share, departure-time
  histogram, workplace dwell time. All of these exist in both my `employee_features`/
  `synthetic_activity` outputs and the external `SyntheticEmployees`/`SyntheticTrips` files.
- **Structural checks:** (1) every employee's trip timeline is chronologically ordered with no
  overlaps; (2) commute distance/duration are internally consistent with implied trip legs; (3)
  schema round-trips cleanly through my ingestion code without silent NaN coercion.
- **Success criteria:** distributions of the compared features fall within the same order of
  tolerance both pipelines already validate against NHTS marginals independently (e.g. commute
  distance within ~1.5 mi of each other, EV share within ~2 pp) — success here demonstrates that
  two independently-coded generators targeting the same survey converge, which is a useful
  implementation sanity check, not evidence of external generalization.
- **Privacy precautions:** minimal residual risk — all compared records are synthetic
  (`LBNL_EMP_####` IDs, placeholder home/office locations), and my own outputs are already
  synthetic. No NHTS respondent-level data changes hands. Standard practice still applies: don't
  publish the external repo's raw files outside this read-only comparison, and don't merge or
  redistribute its `data/` contents into this project's tracked outputs.

This experiment is fast (no new modeling, just two summary tables and a handful of comparison
plots) and defensible: it tells me whether my generator's outputs are broadly consistent with an
independently engineered generator that used the same underlying survey, which is useful for
catching gross implementation bugs — but it must be reported as a **cross-implementation
consistency check**, not as validation against new data.

---

## 6. Leakage check

**Yes — every "observed" candidate file in this repository is derived from the identical NHTS
2022 national public-use release my pipeline already ingests from `data/raw/`.** Verified by
exact row-count match (§2) and by the external repo's own documentation citing the same four
source files with the same record counts.

**Is a test against the aggregated/observed files circular?**
Yes, for any test framed as "external validation." `CombinedDriverProfiles.csv`,
`ArchetypeDefinitions.csv`, `Driving_Profile_Probability_Tables.csv`, and the `tables/` directory
are alternate aggregations of the *same* underlying person/trip/household/vehicle records my
pipeline's `clean`/`ingest`/`build_features` stages already process. Any distributional match
would be partly guaranteed by both parties reading the same weighted marginals from the same
survey, not by genuine independent agreement.

**What limited claim could it still support?**
- That my pipeline's weighted statistics (means, medians, shares) are computed correctly, by
  cross-checking against a second, independently-coded implementation of the same NHTS weighting
  methodology (`WTPERFIN`/`WTTRDFIN`/`WTHHFIN`) — a code-correctness check, not a data-generalization
  check.
- That two independently built synthetic generators targeting the same survey converge on similar
  outputs (§5) — an implementation consistency check.

**What it cannot prove:**
- That my synthetic driving profiles generalize beyond the 2022 NHTS national sample.
- That my pipeline performs well on data it has not seen — there is no unseen data here.
- Anything about geographic transferability (this repo has the same "no county/CBSA identifier"
  limitation as my own raw files, per its own `METHODOLOGY_REPORT.md` §3, §9).

---

## 7. Final verdict

- **Best file(s) to use:** `lbnl_model/outputs/SyntheticEmployees.csv` +
  `lbnl_model/outputs/SyntheticTrips.csv`, for a cross-generator structural/distributional
  consistency check (§5) and as a format-adapter test (schema, timeline conventions, join keys).
- **Unsuitable files (for validation purposes):** `CombinedDriverProfiles.csv` /
  `Combined_Driver_Profiles.csv`, `ArchetypeDefinitions.csv`, `Driving_Profile_Probability_Tables.csv`,
  everything in `tables/`, and `All_Activity_Profiles.csv` / `TripPurpose_Activity_Profiles.csv` —
  all are aggregates or deterministic templates derived from the same source records already in
  `data/raw/`; useful only as a code-correctness cross-check (§6), never as new evidence.
- **Is this an independent generalization test?** **No.** Same NHTS 2022 national release, same
  respondents, same known geography/occupation limitations as my own pipeline's source data.
- **Is it a same-source robustness / implementation-consistency test?** **Yes** — that is the
  most this repository can offer: a second, independently coded pipeline against the identical
  underlying survey, useful for catching bugs and confirming methodological agreement, not for
  demonstrating that my pipeline works beyond the 2022 NHTS national sample.
- **Next implementation step:** if this cross-check is worth pursuing, write a small comparison
  script (outside this assessment, on request) that (a) generates a 250-employee run from my own
  pipeline, (b) loads `SyntheticEmployees.csv`/`SyntheticTrips.csv` read-only from
  `data/external/nhts_datasetanalysis/`, and (c) plots/tabulates the feature distributions listed
  in §5 side by side, labeling the result explicitly as a cross-implementation consistency check
  rather than external validation.
