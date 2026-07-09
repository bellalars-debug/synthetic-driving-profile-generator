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
    donor_summary: pd.DataFrame,
    cluster_id,
    require_driving_leg: bool,
    trips_per_day: int,
    number_of_stops: int,
) -> pd.DataFrame:
    """Reproduce `select_donor`'s current candidate search
    (generator/activity.py): `cluster_id` and `has_driving_leg ==
    require_driving_leg` (the donor mode-blindness fix), then the same
    trip/stop-count tolerance widening (`MATCH_TOLERANCES`). Returns every
    candidate at the tolerance level that first found a match, rather than
    picking one - the set `select_donor`'s own tie-break would have drawn
    from.
    """
    pool = donor_summary.loc[
        (donor_summary["cluster_id"] == cluster_id)
        & (donor_summary["has_driving_leg"] == require_driving_leg)
    ]
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
    """Confirm no donor mode mismatch remains after `generator/activity.py`'s
    `select_donor` fix (`docs/validation_results.md` §6 "donor
    mode-blindness"): `select_donor` now restricts its candidate pool to
    donors whose own `has_driving_leg` matches whether the requesting
    employee's `total_daily_miles` is non-null (`require_driving_leg`), so
    a donor's raw `TRPMILES`/`TRVLCMIN` can no longer describe the wrong
    kind of day (a non-driving donor's walk/bike/transit trip standing in
    for a driving employee's `distance`/`duration`, or vice versa).

    This does *not* replay the actual RNG draws recorded in
    `synthetic_activity.parquet` - `generator/activity.py`'s module
    docstring explains why donor identity is deliberately not stored in
    that output (no real IDs in generated artifacts), so there is nothing
    to join back to directly. Instead, for every synthetic employee, this
    reproduces the *candidate donor pool* `select_donor` would have
    searched (`_candidate_donor_pool`, same cluster/has_driving_leg/
    trip-count/stop-count tolerance widening) and checks each candidate's
    own `total_daily_miles` nullness (from `employee_clusters`, a field
    computed independently of `has_driving_leg`, which is instead derived
    from `trips_clean`'s `TRPTRANS`) against what the employee's
    `require_driving_leg` implies. Every candidate is expected to agree - a
    disagreement would mean `has_driving_leg` and `total_daily_miles`
    nullness have drifted out of sync for some donor, which would let a
    mismatch slip through `select_donor`'s filter despite it operating
    correctly.
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

    n_violations = 0
    n_checked = 0
    n_no_candidates = 0
    n_employees_checked = 0
    for _, employee in synthetic_employees.iterrows():
        require_driving_leg = bool(pd.notna(employee[PRIMARY_MISSING_COLUMN]))
        candidates = _candidate_donor_pool(
            donor_summary,
            employee["cluster_id"],
            require_driving_leg,
            int(employee["trips_per_day"]),
            int(employee["number_of_stops"]),
        )
        if candidates.empty:
            n_no_candidates += 1
            continue
        n_employees_checked += 1
        n_checked += len(candidates)
        # A driving employee's candidates must all have driven
        # (donor_total_daily_miles_null == False); a non-driving employee's
        # candidates must all not have (== True) - so a mismatch is where
        # the candidate's nullness equals require_driving_leg itself.
        mismatched = candidates["donor_total_daily_miles_null"] == require_driving_leg
        n_violations += int(mismatched.sum())

    return common.results_frame(
        [
            common.structural_result(
                SECTION,
                "donor_mode_mismatch",
                n_violations=n_violations,
                n_checked=n_checked,
                threshold=(
                    "0 candidate donors whose total_daily_miles nullness contradicts the "
                    "requesting employee's own driving-mode requirement"
                ),
                detail=(
                    f"{n_violations} mismatched candidate(s) across {n_checked} candidate(s) "
                    f"for {n_employees_checked}/{len(synthetic_employees)} synthetic "
                    f"employee(s) with >=1 candidate donor; {n_no_candidates} had no matching "
                    "donor (would use a fallback chain, unaffected by donor mode-blindness)"
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
