"""Validate synthetic activity chains against NHTS trip behavior
(docs/validation_plan.md §4).

Source dataset: `data/interim/trips_clean.parquet`, restricted to
respondents with a `cluster_id` via `employee_clusters.parquet` - the same
donor pool `generator/activity.py`'s `build_donor_legs` builds, reused here
(not re-derived) so validation compares against exactly the pool generation
drew from, not the full unfiltered trip file. Synthetic dataset:
`data/processed/synthetic_activity.parquet`.

Every comparison is run per `cluster_id` (same reasoning as clusters.py) and
per `chain_source` ("donor" vs. "fallback") - a fallback chain is built
directly from the employee's own summary values, not borrowed structure, so
mixing it with donor-derived chains in one comparison would obscure whether
either mechanism individually produces realistic output.

Read-only: reuses `generator/activity.py`'s own donor-pool construction,
purpose classification, and constants rather than re-deriving them - never
regenerates or rescales a chain.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from driving_profiles.generator import activity as activity_module
from driving_profiles.validation import common

SECTION = "activity"

CHAIN_SOURCES = (activity_module.DONOR_CHAIN_SOURCE, activity_module.FALLBACK_CHAIN_SOURCE)

# Purpose sequence build_fallback_chain always produces (activity.py:529-539)
# and the pattern the plan's "home->work->home pattern share" metric tracks
# for donor-sourced chains - a direct, no-stop round-trip commute.
DIRECT_COMMUTE_PURPOSE_SEQUENCE = [
    activity_module.TRIP_PURPOSE_WORK,
    activity_module.TRIP_PURPOSE_HOME,
]


def load_synthetic_activity(
    processed_dir: Path = activity_module.DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """Read generator/activity.py's output (`synthetic_activity.parquet`)."""
    path = Path(processed_dir) / activity_module.ACTIVITY_TABLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic activity table not found: {path}. Run "
            "`python -m driving_profiles.generator.activity` first."
        )
    return pd.read_parquet(path)


def build_donor_pool(employee_clusters: pd.DataFrame, trips_clean: pd.DataFrame) -> pd.DataFrame:
    """The real ground-truth donor pool (plan §4) - reuses
    generator/activity.py's own construction so this validates against
    exactly what generation drew from."""
    return activity_module.build_donor_legs(trips_clean, employee_clusters)


def attach_cluster(
    synthetic_activity: pd.DataFrame, synthetic_employees: pd.DataFrame
) -> pd.DataFrame:
    """Join each synthetic activity leg back to its employee's `cluster_id`
    (synthetic_activity.parquet deliberately doesn't carry cluster_id - see
    generator/activity.py's OUTPUT_COLUMNS)."""
    lookup = synthetic_employees.set_index("synthetic_employee_id")["cluster_id"]
    out = synthetic_activity.copy()
    out["cluster_id"] = out["synthetic_employee_id"].map(lookup)
    return out


def _per_employee_chain_shape(legs: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    """One row per employee-day: cluster_id, leg_count, stop_count."""
    grouped = legs.groupby(id_columns)
    return grouped.agg(
        cluster_id=("cluster_id", "first"),
        leg_count=("trip_purpose", "size"),
        stop_count=(
            "trip_purpose",
            lambda s: int((s != activity_module.TRIP_PURPOSE_HOME).sum()),
        ),
    ).reset_index()


def validate_chain_length(
    donor_legs: pd.DataFrame, synthetic_activity_c: pd.DataFrame
) -> pd.DataFrame:
    """Legs per employee-day and number of stops (plan §4 "Trip chain
    structure": trips/stops per employee-day, trip sequence length - the
    latter is the same quantity as leg count so it isn't duplicated here,
    per the plan's own note against double-counting).
    """
    rows = []
    donor_shape = _per_employee_chain_shape(donor_legs, ["HOUSEID", "PERSONID"])

    for chain_source in (None, *CHAIN_SOURCES):
        label = "all" if chain_source is None else chain_source
        subset = (
            synthetic_activity_c
            if chain_source is None
            else synthetic_activity_c.loc[synthetic_activity_c["chain_source"] == chain_source]
        )
        if subset.empty:
            continue
        synthetic_shape = _per_employee_chain_shape(subset, ["synthetic_employee_id"])

        cluster_ids = sorted(
            set(donor_shape["cluster_id"].dropna()) | set(synthetic_shape["cluster_id"].dropna())
        )
        for cluster_id in cluster_ids:
            d = donor_shape.loc[donor_shape["cluster_id"] == cluster_id]
            s = synthetic_shape.loc[synthetic_shape["cluster_id"] == cluster_id]
            rows.append(
                common.ks_result(
                    SECTION,
                    "legs_per_employee_day",
                    d["leg_count"],
                    s["leg_count"],
                    group=f"cluster_{cluster_id}",
                    chain_source=label,
                )
            )
            rows.append(
                common.ks_result(
                    SECTION,
                    "stops_per_employee_day",
                    d["stop_count"],
                    s["stop_count"],
                    group=f"cluster_{cluster_id}",
                    chain_source=label,
                )
            )
    return common.results_frame(rows)


def validate_direct_commute_share(
    donor_legs: pd.DataFrame, synthetic_activity_c: pd.DataFrame
) -> pd.DataFrame:
    """Share of chains that are a direct 2-leg work-then-home round trip
    (plan §4 "Home->work->home pattern share"), donor-sourced chains only -
    fallback chains are *always* this exact pattern by construction
    (`build_fallback_chain`), so comparing them would be tautological, per
    the plan.
    """

    def _direct_share(legs: pd.DataFrame, id_columns: list[str]) -> tuple[float, int]:
        sequences = legs.sort_values(id_columns).groupby(id_columns)["trip_purpose"].apply(list)
        is_direct = sequences.apply(lambda seq: seq == DIRECT_COMMUTE_PURPOSE_SEQUENCE)
        return float(is_direct.mean()) if len(is_direct) else float("nan"), len(is_direct)

    donor_chain = synthetic_activity_c.loc[
        synthetic_activity_c["chain_source"] == activity_module.DONOR_CHAIN_SOURCE
    ]
    rows = []
    cluster_lookup = donor_legs[["HOUSEID", "PERSONID", "cluster_id"]].drop_duplicates()
    cluster_ids = sorted(
        set(cluster_lookup["cluster_id"].dropna()) | set(donor_chain["cluster_id"].dropna())
    )
    for cluster_id in cluster_ids:
        donor_subset = donor_legs.loc[donor_legs["cluster_id"] == cluster_id]
        syn_subset = donor_chain.loc[donor_chain["cluster_id"] == cluster_id]
        src_share, n_src = _direct_share(donor_subset, ["HOUSEID", "PERSONID"])
        syn_share, n_syn = _direct_share(syn_subset, ["synthetic_employee_id"])
        if n_src == 0 or n_syn == 0:
            continue
        diff_pp = abs(syn_share - src_share) * 100
        rows.append(
            common.result_row(
                SECTION,
                "direct_commute_pattern_share",
                group=f"cluster_{cluster_id}",
                chain_source=activity_module.DONOR_CHAIN_SOURCE,
                test="proportion_diff",
                statistic=diff_pp,
                n_source=n_src,
                n_synthetic=n_syn,
                threshold="diff <= 5pp",
                passed=bool(diff_pp <= 5.0),
                detail=f"source={src_share:.4f} synthetic={syn_share:.4f}",
            )
        )
    return common.results_frame(rows)


def _with_donor_minutes_and_dwell(donor_legs: pd.DataFrame) -> pd.DataFrame:
    """Add `_departure_minutes`/`_arrival_minutes`/`dwell_time_after` to a
    donor-pool leg table, computed the same way
    `generator/activity.py`'s `compute_dwell_time_after` derives them for
    the synthetic side: from consecutive legs' own times, not NHTS's raw
    `DWELTIME` (which `build_donor_legs`'s `DONOR_LEG_COLUMNS` doesn't even
    carry through) - comparing dwell computed the same way on both sides is
    what makes the comparison apples-to-apples.

    Relies on `donor_legs` already being sorted into each person's
    chronological order, which `build_donor_legs` guarantees.
    """
    df = donor_legs.copy()
    df["_departure_minutes"] = df["STRTTIME"].apply(activity_module.hhmm_to_minutes)
    df["_arrival_minutes"] = df["ENDTIME"].apply(activity_module.hhmm_to_minutes)
    next_departure = df.groupby(["HOUSEID", "PERSONID"])["_departure_minutes"].shift(-1)
    df["dwell_time_after"] = (next_departure - df["_arrival_minutes"]).clip(lower=0)
    return df


def validate_leg_distributions(
    donor_legs: pd.DataFrame, synthetic_activity_c: pd.DataFrame
) -> pd.DataFrame:
    """Per-leg distance/duration/departure/arrival time distributions,
    per cluster and per chain_source (plan §4 "Travel behavior").
    """
    rows = []
    donor = _with_donor_minutes_and_dwell(donor_legs)

    syn_dep = synthetic_activity_c["departure_time"].apply(activity_module.hhmm_to_minutes)
    syn_arr = synthetic_activity_c["arrival_time"].apply(activity_module.hhmm_to_minutes)
    synthetic = synthetic_activity_c.assign(_departure_minutes=syn_dep, _arrival_minutes=syn_arr)

    for chain_source in CHAIN_SOURCES:
        syn_subset = synthetic.loc[synthetic["chain_source"] == chain_source]
        if syn_subset.empty:
            continue
        cluster_ids = sorted(
            set(donor["cluster_id"].dropna()) | set(syn_subset["cluster_id"].dropna())
        )
        for cluster_id in cluster_ids:
            d = donor.loc[donor["cluster_id"] == cluster_id]
            s = syn_subset.loc[syn_subset["cluster_id"] == cluster_id]
            rows.append(
                common.ks_result(
                    SECTION, "leg_distance", d["TRPMILES"], s["distance"],
                    group=f"cluster_{cluster_id}", chain_source=chain_source,
                )
            )
            rows.append(
                common.ks_result(
                    SECTION, "leg_duration", d["TRVLCMIN"], s["duration"],
                    group=f"cluster_{cluster_id}", chain_source=chain_source,
                )
            )
            rows.append(
                common.ks_result(
                    SECTION,
                    "departure_time_minutes",
                    d["_departure_minutes"],
                    s["_departure_minutes"],
                    group=f"cluster_{cluster_id}",
                    chain_source=chain_source,
                )
            )
            rows.append(
                common.ks_result(
                    SECTION, "arrival_time_minutes", d["_arrival_minutes"], s["_arrival_minutes"],
                    group=f"cluster_{cluster_id}", chain_source=chain_source,
                )
            )
            for purpose in (
                activity_module.TRIP_PURPOSE_HOME,
                activity_module.TRIP_PURPOSE_WORK,
                activity_module.TRIP_PURPOSE_OTHER,
            ):
                d_purpose = d.loc[d["trip_purpose"] == purpose]
                s_purpose = s.loc[s["trip_purpose"] == purpose]
                rows.append(
                    common.ks_result(
                        SECTION, "leg_distance", d_purpose["TRPMILES"], s_purpose["distance"],
                        group=f"cluster_{cluster_id}_purpose_{purpose}", chain_source=chain_source,
                    )
                )
    return common.results_frame(rows)


def validate_implied_speed_plausibility(synthetic_activity: pd.DataFrame) -> pd.DataFrame:
    """Structural check (plan §4): every leg's implied speed
    (distance/duration) should fall within
    [MIN_PLAUSIBLE_SPEED_MPH, MAX_PLAUSIBLE_SPEED_MPH] - a hard rule
    rescale_chain_distances is supposed to guarantee for every rescaled
    leg, not just the ones its own near-zero-distance guard targets, so
    this is worth re-confirming at full-population scale.
    """
    legs = synthetic_activity.loc[synthetic_activity["duration"] > 0]
    implied_speed = legs["distance"] / (legs["duration"] / 60.0)
    violations = (
        (implied_speed < activity_module.MIN_PLAUSIBLE_SPEED_MPH)
        | (implied_speed > activity_module.MAX_PLAUSIBLE_SPEED_MPH)
    ).sum()
    return common.results_frame(
        [
            common.structural_result(
                SECTION,
                "implied_leg_speed_plausible",
                n_violations=int(violations),
                n_checked=len(legs),
                threshold=(
                    f"100% within [{activity_module.MIN_PLAUSIBLE_SPEED_MPH}, "
                    f"{activity_module.MAX_PLAUSIBLE_SPEED_MPH}] mph"
                ),
                detail=f"{int(violations)}/{len(legs)} leg(s) outside the plausible-speed band",
            )
        ]
    )


def validate_workplace_timing_consistency(
    synthetic_activity: pd.DataFrame, synthetic_employees: pd.DataFrame
) -> pd.DataFrame:
    """Structural check (plan §4): each employee's *first* workplace-arrival/
    departure leg should equal their own drawn `work_arrival_time`/
    `work_departure_time` exactly (within floating-point tolerance) -
    deterministic by construction (`rescale_chain_times`), not a
    distributional question.

    Only the first work-purpose leg is anchored this way
    (`rescale_chain_times`'s docstring: "the chain's first work-purpose leg
    lands on target_arrival_hhmm"); `is_workplace_arrival`/
    `is_workplace_departure` flag *every* work-purpose leg, so a fragmented-
    dwell chain's second work visit (plan §5) is correctly unanchored and
    must be excluded here, not counted as a mismatch.

    Compares in minutes-since-midnight (`hhmm_to_minutes`), not raw HHMM -
    `sample.py`'s jitter perturbs `work_arrival_time`/`work_departure_time`
    as a raw HHMM number, so a jittered value routinely has an
    out-of-range minute component (e.g. `880.0`); `rescale_chain_times`
    re-encodes that through `minutes_to_hhmm` into a valid clock reading
    (e.g. `920.0`) when it stores `arrival_time`/`departure_time`, so a raw
    HHMM equality check would flag these as false mismatches even though
    they denote the identical time of day (see `hhmm_to_minutes`'s own
    docstring in generator/activity.py).
    """
    employees = synthetic_employees.set_index("synthetic_employee_id")
    rows = []

    arrivals = synthetic_activity.loc[synthetic_activity["is_workplace_arrival"]].sort_values(
        ["synthetic_employee_id", "trip_number"]
    )
    first_arrivals = arrivals.groupby("synthetic_employee_id", as_index=False).first()
    first_arrivals["target"] = first_arrivals["synthetic_employee_id"].map(
        employees["work_arrival_time"]
    )
    arrival_minutes = first_arrivals["arrival_time"].apply(activity_module.hhmm_to_minutes)
    target_arrival_minutes = first_arrivals["target"].apply(activity_module.hhmm_to_minutes)
    arrival_mismatch = (
        (arrival_minutes - target_arrival_minutes).abs() > 1e-6
    ) & first_arrivals["target"].notna()
    rows.append(
        common.structural_result(
            SECTION,
            "workplace_arrival_matches_drawn_value",
            n_violations=int(arrival_mismatch.sum()),
            n_checked=len(first_arrivals),
            threshold=(
                "100% match, in minutes-since-midnight, on each employee's "
                "first workplace-arrival leg"
            ),
        )
    )

    departures = synthetic_activity.loc[synthetic_activity["is_workplace_departure"]].sort_values(
        ["synthetic_employee_id", "trip_number"]
    )
    first_departures = departures.groupby("synthetic_employee_id", as_index=False).first()
    first_departures["target"] = first_departures["synthetic_employee_id"].map(
        employees["work_departure_time"]
    )
    first_departures["target_arrival"] = first_departures["synthetic_employee_id"].map(
        employees["work_arrival_time"]
    )
    # rescale_chain_times falls back to the arrival-anchored offset (not the
    # departure target) whenever the target departure is missing or would
    # land at/before the (already-shifted) arrival time (activity.py:342-346)
    # - a documented, not buggy, deviation, so those rows are excluded from
    # this check (no valid anchor to compare against) rather than counted as
    # mismatches.
    target_arrival_min = first_departures["target_arrival"].apply(activity_module.hhmm_to_minutes)
    target_departure_min = first_departures["target"].apply(activity_module.hhmm_to_minutes)
    departure_minutes = first_departures["departure_time"].apply(activity_module.hhmm_to_minutes)
    has_valid_anchor = first_departures["target"].notna() & (
        target_departure_min > target_arrival_min
    )
    departure_mismatch = (
        (departure_minutes - target_departure_min).abs() > 1e-6
    ) & has_valid_anchor
    rows.append(
        common.structural_result(
            SECTION,
            "workplace_departure_matches_drawn_value",
            n_violations=int(departure_mismatch.sum()),
            n_checked=int(has_valid_anchor.sum()),
            threshold=(
                "100% match target departure on each employee's first workplace-departure "
                "leg, restricted to employees whose target departure was actually usable as "
                "an anchor (activity.py:342-346 documents when it isn't)"
            ),
        )
    )
    return common.results_frame(rows)


def validate_dwell_time(
    donor_legs: pd.DataFrame, synthetic_activity_c: pd.DataFrame
) -> pd.DataFrame:
    """Workplace dwell duration distribution (plan §4 "Workplace dwell
    periods") - the crux metric for whether a charging session can complete
    before departure. Both sides compute dwell the same way (consecutive
    legs' own times via `_with_donor_minutes_and_dwell`/
    `compute_dwell_time_after`), not NHTS's raw `DWELTIME`, so the
    comparison is apples-to-apples. Synthetic side uses
    `workplace_dwell_minutes` (non-null subset - NaN is an end-of-chain
    artifact, not an error).
    """
    donor = _with_donor_minutes_and_dwell(donor_legs)
    donor_work = donor.loc[donor["trip_purpose"] == activity_module.TRIP_PURPOSE_WORK]
    rows = []
    for chain_source in CHAIN_SOURCES:
        syn_subset = synthetic_activity_c.loc[
            (synthetic_activity_c["chain_source"] == chain_source)
            & synthetic_activity_c["is_workplace_arrival"]
        ]
        if syn_subset.empty:
            continue
        cluster_ids = sorted(
            set(donor_work["cluster_id"].dropna()) | set(syn_subset["cluster_id"].dropna())
        )
        for cluster_id in cluster_ids:
            d = donor_work.loc[donor_work["cluster_id"] == cluster_id, "dwell_time_after"]
            s = syn_subset.loc[syn_subset["cluster_id"] == cluster_id, "workplace_dwell_minutes"]
            rows.append(
                common.ks_result(
                    SECTION, "workplace_dwell_minutes", d, s,
                    group=f"cluster_{cluster_id}", chain_source=chain_source,
                )
            )
    return common.results_frame(rows)


def validate_fragmented_dwell_rate(
    donor_legs: pd.DataFrame, synthetic_activity_c: pd.DataFrame, max_diff_pp: float = 5.0
) -> pd.DataFrame:
    """Share of employee-days with more than one work-purpose leg (plan §5's
    fragmented-dwell-window case) - determines whether "one long session" or
    "multiple shorter sessions" is the right per-employee charging model.
    """

    def _fragmented_share(df: pd.DataFrame, id_columns: list[str]) -> tuple[float, int]:
        work_leg_counts = (
            df.loc[df["trip_purpose"] == activity_module.TRIP_PURPOSE_WORK]
            .groupby(id_columns)
            .size()
        )
        if work_leg_counts.empty:
            return float("nan"), 0
        return float((work_leg_counts > 1).mean()), len(work_leg_counts)

    rows = []
    donor_share, n_donor = _fragmented_share(donor_legs, ["HOUSEID", "PERSONID"])
    donor_chain = synthetic_activity_c.loc[
        synthetic_activity_c["chain_source"] == activity_module.DONOR_CHAIN_SOURCE
    ]
    syn_share, n_syn = _fragmented_share(donor_chain, ["synthetic_employee_id"])
    if n_donor and n_syn:
        diff_pp = abs(syn_share - donor_share) * 100
        rows.append(
            common.result_row(
                SECTION,
                "fragmented_dwell_window_share",
                chain_source=activity_module.DONOR_CHAIN_SOURCE,
                test="proportion_diff",
                statistic=diff_pp,
                n_source=n_donor,
                n_synthetic=n_syn,
                threshold=f"diff <= {max_diff_pp}pp",
                passed=bool(diff_pp <= max_diff_pp),
                detail=f"source={donor_share:.4f} synthetic={syn_share:.4f}",
            )
        )
    return common.results_frame(rows)


def validate_fallback_rate(synthetic_activity: pd.DataFrame) -> pd.DataFrame:
    """Fallback chain rate, overall and per cluster (plan §4/§7) -
    informational, not a pass/fail gate: a high fallback rate in a cluster
    means donor-derived realism can't be trusted there, which is a finding
    to report, not a threshold to enforce mechanically.
    """
    per_employee = synthetic_activity.drop_duplicates("synthetic_employee_id")
    rows = []
    overall_rate = float(
        (per_employee["chain_source"] == activity_module.FALLBACK_CHAIN_SOURCE).mean()
    )
    rows.append(
        common.result_row(
            SECTION,
            "fallback_chain_rate",
            group="pooled",
            test="proportion",
            statistic=overall_rate,
            n_synthetic=len(per_employee),
            threshold="informational",
            passed=None,
            detail=f"{overall_rate:.4f}",
        )
    )
    if "cluster_id" in per_employee.columns:
        for cluster_id, group in per_employee.groupby("cluster_id"):
            rate = float((group["chain_source"] == activity_module.FALLBACK_CHAIN_SOURCE).mean())
            rows.append(
                common.result_row(
                    SECTION,
                    "fallback_chain_rate",
                    group=f"cluster_{cluster_id}",
                    test="proportion",
                    statistic=rate,
                    n_synthetic=len(group),
                    threshold="informational",
                    passed=None,
                    detail=f"{rate:.4f}",
                )
            )
    return common.results_frame(rows)


def run_activity_validation(
    employee_clusters: pd.DataFrame,
    trips_clean: pd.DataFrame,
    synthetic_activity: pd.DataFrame,
    synthetic_employees: pd.DataFrame,
) -> pd.DataFrame:
    """Run every §4 check and return one combined result table."""
    donor_legs = build_donor_pool(employee_clusters, trips_clean)
    synthetic_activity_c = attach_cluster(synthetic_activity, synthetic_employees)

    return pd.concat(
        [
            validate_chain_length(donor_legs, synthetic_activity_c),
            validate_direct_commute_share(donor_legs, synthetic_activity_c),
            validate_leg_distributions(donor_legs, synthetic_activity_c),
            validate_implied_speed_plausibility(synthetic_activity),
            validate_workplace_timing_consistency(synthetic_activity, synthetic_employees),
            validate_dwell_time(donor_legs, synthetic_activity_c),
            validate_fragmented_dwell_rate(donor_legs, synthetic_activity_c),
            validate_fallback_rate(synthetic_activity_c),
        ],
        ignore_index=True,
    )


if __name__ == "__main__":
    import logging

    from driving_profiles.validation import population as population_validation

    logging.basicConfig(level=logging.INFO)
    employee_clusters_full = activity_module.load_employee_clusters()
    trips = activity_module.load_trips_clean()
    synthetic_activity_table = load_synthetic_activity()
    synthetic_employees_table = population_validation.load_synthetic_population()

    results = run_activity_validation(
        employee_clusters_full, trips, synthetic_activity_table, synthetic_employees_table
    )
    print(results.to_string())
