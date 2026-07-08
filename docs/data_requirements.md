# NHTS 2022 Data Requirements Plan

Status: planning only — no data downloaded, no ingestion code written yet.

Sources verified against the official 2022 NextGen NHTS documentation:
- [2022 NHTS Data User Guide](https://nhts.ornl.gov/assets/2022/doc/2022%20NextGen%20NHTS%20User's%20Guide%20V201_PubUse.pdf)
- [2022 NextGen NHTS Public Use Codebook](https://nhts.ornl.gov/assets/2022/doc/codebook.pdf) (covers all four files; v2.0.1 Dec 2024, v2.1 Apr 2025 supersedes it)
- [2022 NHTS Derived Variables](https://nhts.ornl.gov/assets/2022/doc/2022%20NextGen%20NHTS%20Derived%20Variables-PubUse.pdf)
- [NHTS downloads page](https://nhts.ornl.gov/downloads)

Note: variable names changed between the 2017 and 2022 waves (e.g. the old
`WKFTPT`/`TIMETOWK`/`OCCAT` commute variables do not exist in 2022 — they were
replaced by `WRKLOC`, `WKFMHM22`, `GCDWORK`). Anything below is grounded in
the 2022 codebook specifically, not carried over from an earlier wave.

## 1. Files needed

The 2022 NHTS public-use release has exactly **four** data files, output as
CSV/SAS/SPSS, hierarchically linked by shared ID variables:

| File | Filename | Record level | Key |
|---|---|---|---|
| Household | `hhpub.csv` | one row per household | `HOUSEID` |
| Person | `perpub.csv` | one row per household member | `HOUSEID` + `PERSONID` |
| Vehicle | `vehpub.csv` | one row per household vehicle | `HOUSEID` + `VEHID` |
| Trip | `trippub.csv` | one row per trip on the travel day | `HOUSEID` + `PERSONID` + `TRIPID` |

All four are needed for this project — there is no separate "day" file (the
survey day's date/weekday are fields on the trip file), and the state/MSA
oversample ("add-on") households are already folded into these same four
public files, not distributed separately.

Geography beyond Census division/region/MSA-size/urban-rural (e.g. Census
tract-level home/work locations) exists only in a **restricted-use** file
requiring a separate data license from FHWA/ORNL. Out of scope for now — flag
if a future requirement needs tract-level siting precision.

Missing-value convention used throughout all four files (handle in cleaning):
`-1` valid skip, `-7` refused, `-8` don't know, `-9` not ascertained. A few
derived numeric variables (e.g. `ANNMILES`) additionally use `-77`/`-88` —
check the codebook per-variable rather than assuming only four codes.

### Download source (verified, not yet fetched)
- CSV bundle: `https://nhts.ornl.gov/assets/2022/download/csv.zip`
  (HEAD-checked: HTTP 200, `Content-Type: application/x-zip-compressed`,
  ~3.9 MB, `Last-Modified: 2024-12-20` — matches the v2.0.1 codebook date).
  A newer v2.1 (April 2025) release also exists at
  `https://nhts.ornl.gov/media/2022/doc/codebook.pdf`; confirm whether the
  root download URL has been updated to serve v2.1 before pinning it.
- **Unverified — resolve when `download.py` is written**: the four CSVs
  inside the zip are referred to as `hhpub.csv`/`perpub.csv`/`vehpub.csv`/
  `trippub.csv` throughout the official documentation, but at least one
  third-party source references `hhv2pub.csv`/`perv2pub.csv` for a revised
  release. Since listing zip contents requires fetching the archive (out of
  scope for this planning pass — no data is being downloaded yet),
  `download.py` should extract and log the actual filenames it finds rather
  than hard-coding an assumption.
- No published checksum was found on the NHTS site. `download.py` should
  compute and persist a sha256 of the fetched zip (per the TODO already in
  `data/README.md`) so the extract is reproducible from a fresh clone.

## 2. Where each content area lives

| Content area | File(s) | Key variables |
|---|---|---|
| Demographics | Person | `R_AGE`, `R_SEX`, `R_RACE`, `R_HISP`, `EDUC`, `R_RELAT` |
| Household characteristics | Household | `HHSIZE`, `HHFAMINC`, `HOMEOWN`, `HHVEHCNT`, `DRVRCNT`, `LIF_CYC`, `HH_RACE`, `HH_HISP`, `URBAN`, `URBRUR`, `URBANSIZE`, `MSACAT`, `MSASIZE`, `CENSUS_R`, `CENSUS_D` |
| Employment information | Person | `WORKER`, `PAYPROF`, `PRMACT`, `EMPLOYMENT2` (hours/week), `WRKLOC`, `WKFMHM22`, `EMPPASS`, `PARKHOME*` |
| Trip behavior | Trip | `WHYFROM`, `WHYTO`, `WHYTRP1S`, `WHYTRP90`, `TRIPPURP`, `TRPTRANS`, `NUMONTRP`, `LOOP_TRIP` |
| Commute patterns | Person + Trip | Person: `WRKTRANS` (usual commute mode), `WRKLOC`, `WKFMHM22`, `GCDWORK`. Trip: work-purpose legs identified via `WHYTRP1S=10` or `WHYTRP90` |
| Travel distances | Trip + Vehicle + Person | Trip: `TRPMILES`, `VMT_MILE`. Vehicle: `ANNMILES`. Person: `GCDWORK` (great-circle home↔work distance) |
| Departure/arrival times | Trip | `STRTTIME`, `ENDTIME`, `TRVLCMIN`, `DWELTIME`, `TRAVDAY`, `TDAYDATE` |

## 3. Variables useful per pipeline stage

The three consumers below draw on overlapping files but distinct variables.
This matrix makes the separation explicit — a checkmark means that variable
is a direct input to that stage (not just "present in the same file"):

| Variable | Employee demographics | Trip-chain reconstruction | Clustering |
|---|---|---|---|
| `R_AGE`, `R_SEX`, `EDUC`, `HHFAMINC`, `LIF_CYC` | ✓ | | |
| `WORKER`, `PRMACT`, `PAYPROF` | ✓ | | |
| `WRKLOC`, `WKFMHM22` | ✓ | | ✓ (schedule regularity) |
| `WRKTRANS` | ✓ | | ✓ (mode-based segment) |
| `GCDWORK` | ✓ | | ✓ (commute-distance segment) |
| `HHVEHCNT`, `DRVRCNT`, `WHODROVE`/`WHODROVE_IMP` | ✓ | ✓ (which vehicle) | |
| `URBAN`/`URBRUR`, `MSASIZE`, `CENSUS_R` | ✓ | | ✓ (context segment) |
| `WHYFROM`, `WHYTO`, `WHYTRP1S`, `WHYTRP90` | | ✓ | ✓ (chain shape) |
| `STRTTIME`, `ENDTIME`, `TRAVDAY`, `TDAYDATE` | | ✓ | ✓ (time-of-day) |
| `TRPMILES`, `TRVLCMIN`, `VMT_MILE` | | ✓ | ✓ (distance/duration) |
| `DWELTIME` | | ✓ | ✓ (dwell-time segment) |
| `VEHID`, `TRPHHVEH` | | ✓ | |
| `TRIPID`/`TDCASEID`/`SEQ_TRIPID` (chain ordering keys) | | ✓ | |

Read top-to-bottom: employee demographics variables are all person/household
level and set once per synthetic employee; trip-chain reconstruction
variables are all trip-level and ordered within a person-day; clustering
variables are aggregates or segment features computed *from* the trip-chain
and demographic variables, not a fourth independent variable set.

### Synthetic employee generation
Need to define who counts as an "employee" and build a realistic persona:
- `WORKER = 01` — filters to workers (combine with `PRMACT`/`PAYPROF` if a
  stricter "worked for pay last week" definition is wanted)
- `WRKLOC` — onsite / varies / **telework-only (self-employed)** / drives for
  work. Full-telework workers (`03`) should probably be excluded from
  workplace-charging demand entirely.
- `WKFMHM22` — days/week worked from home; drives hybrid-schedule modeling
  for everyone who isn't `WRKLOC=03`
- `WRKTRANS` — usual commute mode; filter to vehicle-driving modes to define
  the population that could plausibly charge an EV at work
- `GCDWORK` — commute distance, a strong persona/clustering signal
- Persona demographics: `R_AGE`, `R_SEX`, `EDUC`, `HHFAMINC`, `LIF_CYC`
- Vehicle access: household `HHVEHCNT`/`DRVRCNT`, joined via `HOUSEID` to the
  Vehicle file, and `WHODROVE`/`WHODROVE_IMP` on trips to confirm the person
  actually drives
- Segmentation context: `URBAN`/`URBRUR`, `MSASIZE`, `CENSUS_R` — useful if
  synthetic employees should be calibrated to a specific region/site type

### Clustering driving behavior
- Per-person-day trip chain: ordered `WHYFROM`→`WHYTO` (or `WHYTRP1S`)
  sequence, timestamped with `STRTTIME`/`ENDTIME`
- Commute leg distance/duration: `TRPMILES` + `TRVLCMIN` on the trip(s) where
  `WHYTRP1S = 10` (work) or `WHYTRP90 = 01`
- Daily trip count and total VMT per person (aggregate `TRPMILES` over
  vehicle-mode trips, `TRPTRANS` in the driving codes)
- Time-of-day distribution of departures/arrivals (rush-hour clustering)
- Dwell time at destinations, `DWELTIME` — especially the work-location dwell,
  which maps directly to charging session duration later
- Trip chain complexity: multi-stop chains vs. direct home↔work vs.
  home↔work↔errand↔home
- `TRAVDAY` — weekday vs. weekend flag, since only weekday patterns matter
  for workplace charging

### Generating daily trip chains
- Full ordered trip sequence per `HOUSEID`+`PERSONID`+day (`TRIPID`/`TDCASEID`)
- `WHYFROM`→`WHYTO` transition pairs to build the activity chain
- `STRTTIME`/`ENDTIME` to place each leg on a 24-hour timeline
- `TRPMILES`/`TRVLCMIN` per leg
- `VEHID`/`TRPHHVEH` to identify which household vehicle carried each leg
  (kept for later; per `docs/methodology.md` vehicle type/EV logic is
  deliberately deferred to the scenarios phase)
- `DWELTIME` at the workplace leg specifically, feeding the future charging-
  demand model

## 4. Files/variables to ignore initially

- **SAS/SPSS copies** of the same four files — use CSV only
- **Restricted-use geographic add-on** (Census tract-level) — requires a
  separate FHWA/ORNL data license; not needed for archetype-level modeling
- Detailed vehicle identity: `MAKE`, `MODEL`, `VEHYEAR` — irrelevant until
  the deferred EV/vehicle-type scenario phase; `VEHFUEL`/`VEHTYPE` are cheap
  to retain now but shouldn't drive any features yet
- Medical condition / mobility device variables (`MEDCOND`, `MEDCOND6`,
  `W_CANE`, `W_CHAIR`, `W_NONE`, `W_SCCH`, `W_VISIMP`, `W_WKCR`) — only
  relevant for an accessibility-focused extension, not this phase
- COVID-era behavior-change variables (`USAGE1`, `USAGE2_1`…`USAGE2_10`) —
  survey-context artifacts, not steady-state commute signal
- Micromobility / shared-mode 30-day usage flags (`LAST30_*`,
  `ESCOOTERUSED`, `RIDESHARE22`, `TAXISERVICE`, `MCTRANSIT`, `PTUSED`,
  `WALKTRANSIT`, `TRNPASS`, `USEPUBTR`) — unless a multi-modal commute
  extension is added later
- Online shopping / delivery variables (`DELIVER`, `RET_AMZ`, `RET_HOME`,
  `RET_PUF`, `RET_STORE`) — not driving-relevant
- School-related variables (`SCHOOL1`, `SCHOOL1C`, `SCHTRN1`, `SCHTYP`,
  `STUDE`) — out of scope for an employee-focused model
- National weight columns (`WTHHFIN`/`WTHHFIN2D`/`WTHHFIN5D` on Household,
  `WTPERFIN*` on Person, `WTTRDFIN*` on Trip) — these are ordinary columns
  in each file, not a separate download. Safe and cheap to carry through
  `ingest.py` unchanged, but don't build features from them yet; revisit
  only if synthetic employees need to reproduce nationally-representative
  prevalence rather than an even/analyst-chosen sample.

## 5. Implementation notes for `download.py` / `ingest.py` / `clean.py`

- **`download.py`**: fetch the zip from the URL in §1, verify it extracts
  to exactly four CSVs, log whatever filenames are actually present (see
  the `hhpub.csv` vs. `hhv2pub.csv` naming caveat in §1) rather than
  hard-coding names, and persist a sha256 + source URL + `Last-Modified`
  value alongside the extract for reproducibility.
- **`ingest.py`**: every ID/key column (`HOUSEID`, `PERSONID`, `VEHID`,
  `TRIPID`, `VEHCASEID`, `TDCASEID`, `SEQ_TRIPID`) is codebook type `C`
  (character), fixed-width, and zero-padded (e.g. `PERSONID` is `"01"`, not
  `1`). These **must** be read as strings — if any of them are inferred as
  integers, leading zeros are silently dropped and later merges across
  Household/Person/Vehicle/Trip will under- or mis-match. Suggested
  boundary: `ingest.py` only loads the four CSVs into typed DataFrames
  (string dtype for all `C`-type columns per the codebook, numeric for
  `N`-type) — it should not yet interpret missing-value sentinels or filter
  rows; that belongs in `clean.py`.
- **`clean.py`**: apply the missing-value convention from §1 (`-1`/`-7`/
  `-8`/`-9`, plus per-variable exceptions like `ANNMILES`'s `-77`/`-88`);
  join Household → Person → Vehicle → Trip on the string keys from
  `ingest.py`; and apply the "employee" filter once it's decided (see §7
  below — this is still open, so `clean.py` shouldn't hard-code a filter
  until that decision is made).

## 6. Initial Feature Engineering Plan

These are the features `features/build_features.py` is expected to derive
downstream of `clean.py` — not implemented yet, but named here so clustering
and generator work has a stable target. Each is one row per person-day
unless noted otherwise.

| Feature | Definition | Source variables | Notes |
|---|---|---|---|
| Commute distance | One-way home↔work distance | `GCDWORK` (survey estimate); cross-check against summed `TRPMILES` on the outbound work-purpose leg(s) | Two independent estimates — worth comparing since `GCDWORK` is great-circle (straight-line), not road distance |
| Arrival/departure times | Clock time of each trip's start and end | `STRTTIME`, `ENDTIME` | Already in 24-hour local time; no timezone conversion needed |
| Number of daily trips | Count of trip records per person-day | `TRIPID` count grouped by `HOUSEID`+`PERSONID`+`TDAYDATE` | Exclude/flag loop trips (`LOOP_TRIP`) separately since they don't change location |
| Trip chain pattern | Ordered sequence of purposes, e.g. `Home→Work→Shop→Home` | `WHYFROM`→`WHYTO` (or `WHYTRP1S`) sequence per person-day | Primary categorical input to clustering; consider a simplified/collapsed alphabet (home, work, other) for the first pass per the open decision in §7 |
| Vehicle availability periods | Windows when a household vehicle is not in use by any member (available for charging) | `VEHID`/`TRPHHVEH` + `STRTTIME`/`ENDTIME` across **all household members' trips**, joined to `HHVEHCNT` | Must be computed at the household+vehicle level, not per-person — a shared vehicle's free time depends on every driver's schedule that day |
| Total daily driving time | Sum of trip duration for vehicle-mode trips in a day | `TRVLCMIN` summed where `TRPTRANS` is a driving mode, grouped by person-day | Complements total daily VMT (`TRPMILES`/`VMT_MILE` summed the same way) |
| Work dwell time | Minutes spent at the workplace location | `DWELTIME` on the trip whose destination purpose is work (`WHYTRP1S=10`) | Direct proxy for charging session duration in the later scenarios phase |
| Weekday flag | Whether the person-day is a weekday | `TRAVDAY` | Workplace charging demand should likely be modeled on weekdays only |

## 7. Open decisions for `docs/methodology.md`
This plan answers "which files/variables," but the following still need an
explicit decision before ingestion code is written:
- Exact "employee" filter (just `WORKER=01`, or also exclude `WRKLOC=03`
  full telework, or also require a driving `WRKTRANS`?)
- Whether to keep multi-stop chains as-is or collapse to home↔work↔home for
  the first clustering pass
- Which NHTS release version to pin (v2.0.1 vs. v2.1, given ORNL has
  revised the codebook/files since initial 2023 release)
