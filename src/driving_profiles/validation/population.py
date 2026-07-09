"""Validate synthetic employee distributions against the NHTS-derived
clustered employee population (docs/validation_plan.md §2).

Source dataset: `data/processed/employee_clusters.parquet`, filtered to
`cluster_id.notna()` - the same population `generator/sample.py` draws
from (comparing against the unfiltered `employee_features.parquet` would
incorrectly include non-workers and no-commute workers who were never
eligible to be resampled). Synthetic dataset:
`data/processed/synthetic_employees.parquet`. Every comparison here is run
both pooled and per `cluster_id`, per the plan: pooled catches gross
sampling bugs, per-cluster is what actually validates distributional
fidelity (a pooled comparison can pass even when a per-cluster distribution
is wrong, if cluster proportions happen to offset the error).

Read-only: this module only loads `generator/sample.py`'s existing output
and reuses its loader/column constants - it never regenerates or perturbs
any value.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from driving_profiles.generator import sample as sample_module
from driving_profiles.generator.activity import hhmm_to_minutes
from driving_profiles.validation import common

SECTION = "population"

DEMOGRAPHIC_KS_FEATURES = ["age", "household_size"]
DEMOGRAPHIC_CATEGORICAL_FEATURES = ["age_band", "household_income_bracket"]
HOUSEHOLD_KS_FEATURES = ["household_vehicle_count", "vehicles_per_driver"]
HOUSEHOLD_BOOLEAN_FEATURES = ["vehicle_per_driver_adequate", "used_household_vehicle"]
COMMUTE_KS_FEATURES = ["commute_distance_survey_miles", "commute_duration_minutes"]
TIME_OF_DAY_FEATURES = ["work_arrival_time", "work_departure_time"]
MOBILITY_COUNT_FEATURES = ["trips_per_day", "number_of_stops"]
MOBILITY_NULLABLE_FEATURES = [
    "total_daily_miles",
    "total_driving_minutes",
    "average_trip_distance_miles",
]


def load_source_population(
    processed_dir: Path = sample_module.DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """The clustered population sample.py draws from (plan §2) - reuses
    sample.py's own loader/filter rather than re-deriving it."""
    return sample_module.load_clustered_employees(processed_dir)


def load_synthetic_population(
    processed_dir: Path = sample_module.DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """Read sample.py's output (`synthetic_employees.parquet`)."""
    path = Path(processed_dir) / sample_module.SYNTHETIC_EMPLOYEE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic employee table not found: {path}. Run "
            "`python -m driving_profiles.generator.sample` first."
        )
    return pd.read_parquet(path)


def iter_groups(source: pd.DataFrame, synthetic: pd.DataFrame):
    """Yield (group_label, source_subset, synthetic_subset): pooled first,
    then one per `cluster_id` present in either table."""
    yield "pooled", source, synthetic
    cluster_ids = sorted(
        set(source["cluster_id"].dropna().unique())
        | set(synthetic["cluster_id"].dropna().unique())
    )
    for cluster_id in cluster_ids:
        yield (
            f"cluster_{cluster_id}",
            source.loc[source["cluster_id"] == cluster_id],
            synthetic.loc[synthetic["cluster_id"] == cluster_id],
        )


def validate_demographics(source: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Age, age band, income, household size (plan §2 "Demographics")."""
    rows = []
    for group, s, y in iter_groups(source, synthetic):
        for column in DEMOGRAPHIC_KS_FEATURES:
            rows.append(common.ks_result(SECTION, column, s[column], y[column], group=group))
        for column in DEMOGRAPHIC_CATEGORICAL_FEATURES:
            rows.append(
                common.chi_square_result(SECTION, column, s[column], y[column], group=group)
            )
        # worker_status/is_worker is expected constant post-filter (plan §2's
        # table calls this a non-informative check, not a real gate) -
        # reported for completeness with a wide tolerance, not a fail signal.
        rows.append(
            common.proportion_result(
                SECTION,
                "is_worker",
                s["is_worker"],
                y["is_worker"],
                group=group,
                max_diff_pp=100.0,
            )
        )
    return common.results_frame(rows)


def validate_household(source: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Vehicle count, vehicles per driver, availability flags (plan §2
    "Household characteristics")."""
    rows = []
    for group, s, y in iter_groups(source, synthetic):
        for column in HOUSEHOLD_KS_FEATURES:
            rows.append(common.ks_result(SECTION, column, s[column], y[column], group=group))
        for column in HOUSEHOLD_BOOLEAN_FEATURES:
            # Unjittered pass-through fields (sample.py:83-86): should match
            # almost exactly within cluster, not just within ordinary
            # resampling tolerance (plan §2).
            rows.append(
                common.proportion_result(
                    SECTION, column, s[column], y[column], group=group, max_diff_pp=1.0
                )
            )
    return common.results_frame(rows)


def validate_commute(source: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Commute distance/duration/arrival/departure (plan §2 "Commute
    behavior") - includes the 90th-percentile tail check on commute
    distance and time-of-day KS tests on HHMM columns converted to minutes.
    """
    rows = []
    for group, s, y in iter_groups(source, synthetic):
        for column in COMMUTE_KS_FEATURES:
            rows.append(common.ks_result(SECTION, column, s[column], y[column], group=group))
        rows.append(
            common.percentile_result(
                SECTION,
                "commute_distance_survey_miles",
                s["commute_distance_survey_miles"],
                y["commute_distance_survey_miles"],
                percentile=90,
                group=group,
            )
        )
        for column in TIME_OF_DAY_FEATURES:
            s_minutes = s[column].apply(hhmm_to_minutes)
            y_minutes = y[column].apply(hhmm_to_minutes)
            rows.append(
                common.ks_result(SECTION, f"{column}_minutes", s_minutes, y_minutes, group=group)
            )
    return common.results_frame(rows)


def validate_daily_mobility(source: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Trips/stops per day, total daily miles/minutes, average trip distance
    (plan §2 "Daily mobility behavior"). `MOBILITY_NULLABLE_FEATURES` are
    compared on their non-null subset only - their missingness rate itself
    is validated separately in missingness.py, per the plan.
    """
    rows = []
    for group, s, y in iter_groups(source, synthetic):
        for column in MOBILITY_COUNT_FEATURES:
            rows.append(common.ks_result(SECTION, column, s[column], y[column], group=group))
            rows.append(
                common.variance_ratio_result(SECTION, column, s[column], y[column], group=group)
            )
        for column in MOBILITY_NULLABLE_FEATURES:
            rows.append(
                common.ks_result(SECTION, f"{column}_nonnull", s[column], y[column], group=group)
            )
    return common.results_frame(rows)


def run_population_validation(
    source: pd.DataFrame | None = None,
    synthetic: pd.DataFrame | None = None,
    processed_dir: Path = sample_module.DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """Run every §2 check and return one combined result table.

    Loads `employee_clusters.parquet`/`synthetic_employees.parquet` from
    `processed_dir` if `source`/`synthetic` aren't supplied directly (tests
    typically pass small in-memory frames instead).
    """
    if source is None:
        source = load_source_population(processed_dir)
    if synthetic is None:
        synthetic = load_synthetic_population(processed_dir)

    return pd.concat(
        [
            validate_demographics(source, synthetic),
            validate_household(source, synthetic),
            validate_commute(source, synthetic),
            validate_daily_mobility(source, synthetic),
        ],
        ignore_index=True,
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    results = run_population_validation()
    gating = results.loc[results["passed"].notna()]
    logger = logging.getLogger(__name__)
    logger.info(
        "population validation: %d/%d check(s) passed",
        int(gating["passed"].sum()),
        len(gating),
    )
    print(results.to_string())
