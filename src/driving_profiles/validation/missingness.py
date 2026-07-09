"""Investigate employees with missing `total_daily_miles` /
`total_driving_minutes` / `average_trip_distance_miles`
(docs/validation_plan.md §5).

`build_features.py` computes these columns by summing over driving-mode
trips only, with `min_count=1` - a worker who commuted by a non-driving
mode (walk/bike/transit) that day legitimately has no driving miles to
report, so missingness here is expected real behavior, not a bug. This
module only measures whether that missingness pattern is faithfully
preserved through sampling and reports how it interacts with activity
generation's donor-selection step - it never imputes a value or otherwise
changes what generation produced, per docs/validation_plan.md §5's
explicit "do not impute" determination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driving_profiles.generator import activity as activity_module
from driving_profiles.validation import common

SECTION = "missingness"

MISSING_COLUMNS = ["total_daily_miles", "total_driving_minutes", "average_trip_distance_miles"]
PRIMARY_MISSING_COLUMN = "total_daily_miles"


def validate_pooled_missingness_rate(
    source: pd.DataFrame, synthetic: pd.DataFrame, max_diff_pp: float = 3.0
) -> pd.DataFrame:
    """Pooled null rate on `total_daily_miles`, source vs. synthetic (plan
    §5's first table row)."""
    s_null = source[PRIMARY_MISSING_COLUMN].isna()
    y_null = synthetic[PRIMARY_MISSING_COLUMN].isna()
    s_rate = float(s_null.mean()) * 100
    y_rate = float(y_null.mean()) * 100
    diff = abs(y_rate - s_rate)
    return common.results_frame(
        [
            common.result_row(
                SECTION,
                "missingness_rate",
                group="pooled",
                test="proportion_diff",
                statistic=diff,
                n_source=len(source),
                n_synthetic=len(synthetic),
                threshold=f"diff <= {max_diff_pp}pp",
                passed=bool(diff <= max_diff_pp),
                detail=(
                    f"source={s_rate:.1f}% ({int(s_null.sum())}/{len(source)}), "
                    f"synthetic={y_rate:.1f}% ({int(y_null.sum())}/{len(synthetic)})"
                ),
            )
        ]
    )


def validate_per_cluster_missingness_rate(
    source: pd.DataFrame, synthetic: pd.DataFrame, max_diff_pp: float = 5.0
) -> pd.DataFrame:
    """Per-cluster null rate on `total_daily_miles` (plan §5) - the smaller
    cluster is expected to show more sampling-driven deviation, hence the
    wider default tolerance than the pooled check.
    """
    rows = []
    cluster_ids = sorted(
        set(source["cluster_id"].dropna().unique()) | set(synthetic["cluster_id"].dropna().unique())
    )
    for cluster_id in cluster_ids:
        s = source.loc[source["cluster_id"] == cluster_id, PRIMARY_MISSING_COLUMN]
        y = synthetic.loc[synthetic["cluster_id"] == cluster_id, PRIMARY_MISSING_COLUMN]
        s_rate = float(s.isna().mean()) * 100
        y_rate = float(y.isna().mean()) * 100
        diff = abs(y_rate - s_rate)
        rows.append(
            common.result_row(
                SECTION,
                "missingness_rate",
                group=f"cluster_{cluster_id}",
                test="proportion_diff",
                statistic=diff,
                n_source=len(s),
                n_synthetic=len(y),
                threshold=f"diff <= {max_diff_pp}pp",
                passed=bool(diff <= max_diff_pp),
                detail=f"source={s_rate:.1f}% synthetic={y_rate:.1f}%",
            )
        )
    return common.results_frame(rows)


def validate_missingness_cooccurrence(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """All three `MISSING_COLUMNS` should be null together, never partially
    (plan §5: `average_trip_distance_miles` is arithmetically derived from
    `total_daily_miles`, so a row with one null and not the others indicates
    a pipeline inconsistency, not natural variation).
    """
    null_counts = df[MISSING_COLUMNS].isna().sum(axis=1)
    fully_null = int((null_counts == len(MISSING_COLUMNS)).sum())
    none_null = int((null_counts == 0).sum())
    partial = len(df) - fully_null - none_null
    return common.results_frame(
        [
            common.structural_result(
                SECTION,
                "missingness_cooccurrence",
                n_violations=partial,
                n_checked=len(df),
                group=dataset_label,
                threshold="0 rows with a partial (1/3 or 2/3) null pattern across the 3 columns",
                detail=f"fully_null={fully_null}, none_null={none_null}, partial={partial}",
            )
        ]
    )


def validate_jitter_preserves_nan(source: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Confirm `sample_employees`'s jitter left every source-NaN as NaN
    (sample.py's `not_na` masking), empirically, joined via
    `source_houseid`/`source_personid` (plan §5) - a code-inspection claim
    re-verified against the full synthetic population rather than trusted
    from reading the code alone.
    """
    src_null_by_person = (
        source.set_index(["HOUSEID", "PERSONID"])[MISSING_COLUMNS].isna()
    )
    violations = 0
    checked = 0
    for column in MISSING_COLUMNS:
        key = pd.MultiIndex.from_frame(
            synthetic[["source_houseid", "source_personid"]].rename(
                columns={"source_houseid": "HOUSEID", "source_personid": "PERSONID"}
            )
        )
        source_was_null = src_null_by_person[column].reindex(key).to_numpy()
        synthetic_is_null = synthetic[column].isna().to_numpy()
        bad = source_was_null & ~synthetic_is_null
        violations += int(np.nansum(bad))
        checked += int(np.nansum(source_was_null))

    return common.results_frame(
        [
            common.structural_result(
                SECTION,
                "jitter_preserves_source_nan",
                n_violations=violations,
                n_checked=checked,
                threshold="0 synthetic rows with a non-null value where the source row was null",
                detail=f"{violations} violation(s) out of {checked} source-null-traced row(s)",
            )
        ]
    )


def _candidate_donor_pool(
    donor_summary: pd.DataFrame, cluster_id, trips_per_day: int, number_of_stops: int
) -> pd.DataFrame:
    """Reproduce select_donor's cluster/tolerance-widening candidate search
    (generator/activity.py's MATCH_TOLERANCES), but return every candidate
    at the tolerance level that first found a match, rather than picking
    one - the set select_donor's own tie-break would have drawn from.
    """
    pool = donor_summary.loc[donor_summary["cluster_id"] == cluster_id]
    if pool.empty:
        return pool
    trip_diff = (pool["trip_count"] - trips_per_day).abs()
    stop_diff = (pool["stop_count"] - number_of_stops).abs()
    for tolerance in activity_module.MATCH_TOLERANCES:
        candidates = pool.loc[(trip_diff <= tolerance) & (stop_diff <= tolerance)]
        if not candidates.empty:
            return candidates
    return pool.iloc[0:0]


def estimate_donor_mode_blindness_rate(
    synthetic_employees: pd.DataFrame,
    employee_clusters: pd.DataFrame,
    trips_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate how often a null-`total_daily_miles` synthetic employee's
    donor chain is *also* a non-driving day for the donor (plan §5's "case
    2"): `build_donor_legs`/`select_donor` don't filter by driving mode, so
    a donor whose own day had no driving-mode trips can still be selected
    purely on trip/stop-count match, and its `TRPMILES`/`TRVLCMIN` would
    then describe a walk/bike/transit trip mislabeled as `distance`/
    `duration` in the driving-activity table.

    This does *not* replay the actual RNG draws recorded in
    `synthetic_activity.parquet` - `generator/activity.py`'s module
    docstring explains why donor identity is deliberately not stored in
    that output (no real IDs in generated artifacts), so there is nothing
    to join back to directly. Instead, for every null-`total_daily_miles`
    synthetic employee, this reproduces the *candidate donor pool*
    `select_donor` would have searched (same cluster/trip-count/stop-count
    tolerance widening) and computes the share of those candidates that are
    themselves null-`total_daily_miles` donors. Since `select_donor` breaks
    ties uniformly at random among candidates, the mean of this per-employee
    share is the expected case-2 rate under the actual selection mechanism
    - an honest estimate of the real quantity, not a replay of one
    particular run's random draws.
    """
    donor_legs = activity_module.build_donor_legs(trips_clean, employee_clusters)
    donor_summary = activity_module.summarize_donor_chains(donor_legs)

    daily_miles_by_person = employee_clusters.set_index(["HOUSEID", "PERSONID"])[
        PRIMARY_MISSING_COLUMN
    ]
    donor_key = pd.MultiIndex.from_frame(donor_summary[["HOUSEID", "PERSONID"]])
    donor_summary = donor_summary.copy()
    donor_summary["donor_total_daily_miles_null"] = (
        daily_miles_by_person.reindex(donor_key).isna().to_numpy()
    )

    null_employees = synthetic_employees.loc[synthetic_employees[PRIMARY_MISSING_COLUMN].isna()]

    per_employee_rates = []
    n_no_candidates = 0
    for _, employee in null_employees.iterrows():
        candidates = _candidate_donor_pool(
            donor_summary,
            employee["cluster_id"],
            int(employee["trips_per_day"]),
            int(employee["number_of_stops"]),
        )
        if candidates.empty:
            n_no_candidates += 1
            continue
        per_employee_rates.append(float(candidates["donor_total_daily_miles_null"].mean()))

    mean_rate = float(np.mean(per_employee_rates)) if per_employee_rates else float("nan")
    return common.results_frame(
        [
            common.result_row(
                SECTION,
                "donor_mode_blindness_case2_rate_estimate",
                group="pooled",
                test="diagnostic_estimate",
                statistic=mean_rate,
                n_synthetic=len(null_employees),
                threshold=(
                    "informational - not a pass/fail gate (plan §5: measure before "
                    "deciding whether build_donor_legs needs a driving-mode restriction)"
                ),
                passed=None,
                detail=(
                    f"expected case-2 rate across {len(per_employee_rates)} "
                    f"null-{PRIMARY_MISSING_COLUMN} employee(s) with >=1 candidate donor; "
                    f"{n_no_candidates} had no matching donor (would use a fallback chain, "
                    "which is unaffected by donor mode-blindness)"
                ),
            )
        ]
    )


def run_missingness_validation(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    employee_clusters: pd.DataFrame,
    trips_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Run every §5 check and return one combined result table."""
    return pd.concat(
        [
            validate_pooled_missingness_rate(source, synthetic),
            validate_per_cluster_missingness_rate(source, synthetic),
            validate_missingness_cooccurrence(source, "source"),
            validate_missingness_cooccurrence(synthetic, "synthetic"),
            validate_jitter_preserves_nan(source, synthetic),
            estimate_donor_mode_blindness_rate(synthetic, employee_clusters, trips_clean),
        ],
        ignore_index=True,
    )


if __name__ == "__main__":
    import logging

    from driving_profiles.generator import activity as ac
    from driving_profiles.validation import population as population_validation

    logging.basicConfig(level=logging.INFO)
    source_population = population_validation.load_source_population()
    synthetic_population = population_validation.load_synthetic_population()
    employee_clusters_full = ac.load_employee_clusters()
    trips = ac.load_trips_clean()

    results = run_missingness_validation(
        source_population, synthetic_population, employee_clusters_full, trips
    )
    print(results.to_string())
