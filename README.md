# Synthetic Driving Profile Generator

A synthetic employee driving/mobility profile generator built on NHTS 2022
travel behavior data, produced as the second stage of a longer pipeline
toward workplace EV charging-demand estimation.

## 1. Project purpose

This project generates statistically realistic, privacy-preserving synthetic
daily mobility profiles for a workforce, calibrated against real-world travel
survey data (NHTS 2022). Given a target employee count, it produces a
population of synthetic employees with plausible demographics, commute
patterns, and full daily trip-leg chains (departure/arrival times, distances,
durations, trip purposes) - without using or exposing any real individual's
travel record.

## 2. Current scope

This phase covers **general daily travel behavior only**:

- Data ingestion and cleaning of NHTS 2022 public-use extracts
- Employee-level feature engineering and behavioral clustering
- Synthetic employee population sampling
- Synthetic daily activity (trip-chain) generation
- A validation framework comparing synthetic output against source
  distributions
- A human-readable `.xlsx` export of all outputs

**Out of scope for this phase**: vehicle type, EV ownership/penetration
scenarios, and charging-demand modeling. These are deferred to
`src/driving_profiles/scenarios/charging_demand.py`, which currently exists
only as a documented placeholder.

## 3. Long-term system vision

This generator is one stage in a larger intended pipeline:

```
parking-space estimation
        -> employee-count estimation
                -> synthetic mobility generation   <-- this repository
                        -> workplace EV charging demand
```

The synthetic mobility profiles produced here are meant to be the direct
input to a future workplace EV charging-demand model, once an employee count
for a given site is available from the upstream stages.

## 4. Implemented pipeline stages

1. **Download** - fetch and verify the NHTS 2022 public-use CSV extract
   (`src/driving_profiles/data/download.py`)
2. **Ingest** - load the four required raw CSVs
   (`src/driving_profiles/data/ingest.py`)
3. **Clean** - filter to weekday-worker, driving-relevant trip records
   (`src/driving_profiles/data/clean.py` -> `data/interim/trips_clean.parquet`)
4. **Feature engineering** - build employee-level commute/daily-mobility
   features, with upstream implausible-value filtering
   (`src/driving_profiles/features/build_features.py` ->
   `data/processed/employee_features.parquet`)
5. **Clustering** - assign employees to behavioral archetypes via KMeans
   (cross-checked with hierarchical clustering)
   (`src/driving_profiles/features/cluster.py` ->
   `data/processed/employee_clusters.parquet`, `cluster_evaluation.csv`)
6. **Synthetic employee sampling** - draw a synthetic population per cluster
   proportions with jittered demographic/summary features
   (`src/driving_profiles/generator/sample.py` ->
   `data/processed/synthetic_employees.parquet`)
7. **Synthetic activity generation** - build per-employee daily trip chains by
   donor selection and rescaling against real NHTS trip legs
   (`src/driving_profiles/generator/activity.py` ->
   `data/processed/synthetic_activity.parquet`)
8. **Validation** - population, cluster, activity, and missingness checks
   against source distributions (`src/driving_profiles/validation/`)
9. **Excel export** - human-readable `.xlsx` workbook of all outputs
   (`src/driving_profiles/utils/export_excel.py`)

`scripts/run_pipeline.py` orchestrates stages 1-9 end to end.

## 5. Data source

[NHTS 2022](https://nhts.ornl.gov/) (National Household Travel Survey),
public-use CSV extract. The pipeline downloads
`https://nhts.ornl.gov/assets/2022/download/csv.zip` and reads four files:
`hhv2pub.csv` (household), `perv2pub.csv` (person), `vehv2pub.csv`
(vehicle), and `tripv2pub.csv` (trip). A `manifest.json` recording the
source URL, sha256, and download timestamp is written alongside the raw
extract for reproducibility. See `data/README.md` and
`docs/data_requirements.md` for details.

## 6. Installation (uv)

```bash
uv sync --extra dev
```

This creates `.venv` and installs the package plus test/lint dependencies
(`pytest`, `ruff`) per `pyproject.toml`. Add `--extra notebook` if you also
want Jupyter for `notebooks/01_explore_nhts_2022.ipynb`.

## 7. Commands

### Run the full pipeline

```bash
uv run python scripts/run_pipeline.py
```

Downloads/ingests/cleans NHTS data, builds features, clusters employees,
samples 5,000 synthetic employees, generates their activity chains, and
exports the `.xlsx` report. Each stage is skipped if its output already
exists.

### Regenerate outputs (force overwrite)

```bash
uv run python scripts/run_pipeline.py --force
```

Add `-n <count>` to change the synthetic employee count (default 5,000) and
`--seed <int>` to control the random seed.

### Run tests

```bash
uv run pytest
```

### Run validation

```bash
uv run python -m driving_profiles.validation.report
```

Writes `docs/validation_results.md`, comparing synthetic output against
source NHTS distributions across population, cluster, activity, and
missingness checks.

### Generate the Excel workbook

```bash
uv run python -m driving_profiles.utils.export_excel
```

Writes `reports/xlsx/synthetic_mobility_report.xlsx` from the existing
`data/processed/` outputs plus the current validation docs. Add `--force` to
overwrite an existing workbook.

## 8. Final outputs

| file | description |
|---|---|
| `data/processed/employee_features.parquet` | employee-level commute/daily-mobility features derived from cleaned NHTS trips |
| `data/processed/employee_clusters.parquet` | employee features with assigned behavioral cluster (archetype) |
| `data/processed/synthetic_employees.parquet` | sampled synthetic employee population |
| `data/processed/synthetic_activity.parquet` | full synthetic daily trip-leg chains per employee |
| `reports/xlsx/synthetic_mobility_report.xlsx` | human-readable workbook of all of the above, plus validation summary, for non-Python review |

## 9. Final headline results

- **5,000** synthetic employees generated
- **15,201** synthetic activity (trip) legs
- **0%** fallback-chain rate (all trip chains are real donor-sourced chains)
- **0** implausible-speed legs (0/14,808 legs outside the [5, 70] mph
  plausible-speed band)
- **306** tests passing

See `docs/model_status.md` and `docs/validation_results.md` for the full
validation breakdown (76 passed / 26 failed / 21 informational checks across
population, cluster, activity, and missingness sections).

## 10. Methodology summary

- **Cleaning**: NHTS 2022 trip records are filtered to weekday workers and
  driving-relevant trips; a handful of physically implausible self-reported
  values (e.g. commute distances >150 mi, daily totals >400 mi) are filtered
  to NaN upstream rather than used as rescale targets.
- **Feature engineering**: employee-level commute and daily-mobility summary
  features are built from the cleaned trip records.
- **Clustering**: employees are grouped into two behavioral archetypes via
  KMeans on the engineered features, cross-checked against a hierarchical
  (Ward-linkage) clustering.
- **Synthetic sampling**: a target-sized synthetic population is drawn
  respecting source cluster proportions, with jittered demographic and
  summary features.
- **Activity generation**: each synthetic employee's daily trip chain is
  built by selecting a compatible real NHTS "donor" record (matched on
  cluster, driving-mode compatibility, and schedule/trip-count tolerance,
  relaxing trip-count tolerance before falling back further) and rescaling
  its legs' distances/durations/times to the employee's own drawn targets.
- **Validation**: synthetic output is statistically compared against source
  NHTS distributions (KS tests, chi-square, proportion/effect-size
  comparisons) plus a set of structural checks that must pass at 100% by
  construction (speed plausibility, workplace arrival/departure consistency,
  NaN preservation, donor mode-matching).

Full detail: `docs/methodology.md`, `docs/clustering_plan.md`,
`docs/synthetic_generation_plan.md`, `docs/activity_generation_plan.md`.

## 11. Privacy note

Synthetic employees and their activity chains are generated profiles, not
real individuals. Trip-leg *shapes* are sourced from real NHTS donor
records, but each synthetic employee's own demographic/summary values are
independently sampled and jittered, and donor-linkage identifiers
(`source_houseid`/`source_personid`) are excluded from the Excel export by
design (see `EMPLOYEE_EXPORT_COLUMNS` in
`src/driving_profiles/utils/export_excel.py`). No output is a copy of, or
directly traceable to, a single real NHTS respondent's record.

## 12. Validation approach

Validation is implemented as a dedicated, non-generative measurement layer
(`src/driving_profiles/validation/`) that never imputes missing values or
changes generation logic to produce a passing result. It compares synthetic
output to source NHTS distributions across four sections - population,
clusters, activity, and missingness - using:

- **Distributional tests**: two-sample Kolmogorov-Smirnov (`ks_2samp`) and
  chi-square goodness-of-fit against source proportions
- **Proportion/effect-size comparisons**: simple share differences and
  Cohen's d between clusters (informational)
- **Structural checks**: deterministic-by-construction invariants (e.g.
  implied leg speed within a plausible band, workplace arrival/departure
  times matching drawn targets, NaN preservation through jitter, donor
  driving-mode compatibility) where any failure indicates an actual bug
  rather than natural sampling variation

Running `uv run python -m driving_profiles.validation.report` regenerates
`docs/validation_results.md` from the current pipeline outputs.

## 13. Known limitations

- **Leg-distance/duration tails**: KS tests fail for `leg_distance` and
  `leg_duration` in both clusters. Per
  `docs/activity_validation_investigation.md`, the bulk of the distribution
  (p50-p75) matches source closely; the failure is driven by a small tail
  (~0.1-0.3% of legs) at p95+ that disproportionately affects the KS
  statistic.
- **Cluster 1 workplace-dwell distribution**: `workplace_dwell_minutes`
  fails its KS test for cluster_1 (source mean=461.89 vs. synthetic
  mean=456.93 minutes, ~1% mean shift), while cluster_0's dwell distribution
  passes cleanly.
- **Several population-level KS tests fail on statistical significance**
  despite very small practical mean/sd differences (e.g. pooled
  `vehicles_per_driver`: source mean=1.12 vs. synthetic mean=1.13) - a
  consequence of large sample sizes making KS tests sensitive to small
  distributional differences.
- **Cluster count (k=2) meaningfulness** has a documented diagnostic
  (hierarchical cross-check, silhouette scores in `cluster_evaluation.csv`)
  but no separately written-up domain-interpretability review
  (`docs/clustering_plan.md` §6).
- All of the above are distributional (KS-test-based) findings, not
  structural failures - every deterministic-by-construction check (speed
  plausibility, workplace-time consistency, NaN preservation, donor
  mode-matching) passes at 100%. See `docs/model_status.md` §6-7 for the
  full reasoning on why these are treated as limitations rather than
  blocking defects.
- **EV/charging-demand logic is not implemented** - `scenarios/` is a
  placeholder only.
- `pyproject.toml`, `CITATION.cff`, and `LICENSE` still have unfilled
  `TODO` fields (author, license text, citation metadata) that should be
  completed before public release.

## 14. Repository structure

```
config/                   default.yaml - pipeline configuration
data/                     raw/ interim/ processed/ (not committed; see data/README.md)
docs/                     methodology, planning, and validation-result documents
notebooks/                exploratory NHTS 2022 notebook
reports/xlsx/             generated Excel workbook output
scripts/run_pipeline.py   end-to-end pipeline CLI
src/driving_profiles/
    data/                 download, ingest, clean
    features/             build_features, cluster
    generator/            sample, activity, model, time_utils
    scenarios/            charging_demand (placeholder, not yet implemented)
    utils/                export_excel, io, random_seed
    validation/           activity, clusters, common, missingness, population, report
tests/                    306 tests covering every module above
```

## 15. Future work

- **Charging-demand model** - implement `scenarios/charging_demand.py` to
  translate synthetic mobility profiles into workplace EV charging demand
  under vehicle-type and EV-penetration scenarios (e.g. 5%/10%/20%),
  consuming an employee count from the upstream parking/employee-estimation
  stages.
- **Company-specific inputs** - allow a target site's own employee count,
  shift patterns, or regional travel characteristics to condition the
  synthetic population, rather than sampling from the pooled national NHTS
  distribution only.
- **Remote-sensing integration** - connect the upstream parking-space
  estimation stage (e.g. satellite/aerial imagery-derived parking-space
  counts) to the employee-count estimation stage that feeds this generator,
  closing the loop described in §3.

## 16. Workplace charging estimation — ev-tool station/queue backend

The final pipeline stage turns the synthetic activity profiles into workplace
EV **charging estimates**. Two backends are available:

- `src/driving_profiles/scenarios/charging_demand.py` — built-in scenario energy
  model (EV adoption, efficiency, unmanaged-immediate delivery).
- `src/driving_profiles/scenarios/ev_tool_charging.py` — **station/queue
  simulation** using Rongxin Yin's ev-infrastructure-tool (MIT), vendored under
  `third_party/ev_infrastructure_tool/`. Models discrete L2/L3 stations, a
  first-come queue, contention, and waiting time.

Both read the finalized `data/processed/synthetic_activity.parquet` /
`synthetic_employees.parquet` and never modify the generator. The ev-tool stage
reconstructs each employee's day into the tool's `pov_driving_pattern.json`
schema (work legs → `On-Site`, stops → `Off-Site`) — preserving trip chains the
ev-tool's own generator can't produce — then runs its charging simulator.

```bash
# after scripts/run_pipeline.py has produced the activity profiles:
python scripts/run_ev_tool_charging.py --adoption-rate 0.36 --run-period 30
```

Outputs `ev_tool_vehicle_status_{rate}.csv` + `ev_tool_summary.json` (vehicles,
EVs selected, L2 station-count scenarios, peak simultaneous charging). See
`docs/ev_tool_charging_integration.md`; smoke test in
`tests/test_ev_tool_charging.py`, runnable demo in
`scripts/demo_ev_tool_charging.py`.
