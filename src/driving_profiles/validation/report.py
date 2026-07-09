"""Aggregate every validation module into one summary report
(docs/validation_plan.md §7-8).

Read-only end to end: loads the pipeline's existing outputs
(`data/processed/*.parquet`, `data/interim/trips_clean.parquet`), runs
population.py/clusters.py/activity.py/missingness.py, and renders the
combined result as Markdown. Never regenerates, imputes, or otherwise
changes a pipeline artifact - this module only measures and reports, per
docs/validation_plan.md's explicit instruction not to auto-fix findings.

Running this module's `main()` (or `python -m driving_profiles.validation.report`)
is what produces `docs/validation_results.md`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from driving_profiles.features import cluster as cluster_module
from driving_profiles.generator import activity as activity_module
from driving_profiles.generator import sample as sample_module
from driving_profiles.validation import activity as activity_validation
from driving_profiles.validation import clusters as cluster_validation
from driving_profiles.validation import missingness as missingness_validation
from driving_profiles.validation import population as population_validation

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = Path("docs/validation_results.md")

SECTION_TITLES = {
    "population": "Synthetic employee population (plan §2)",
    "clusters": "Cluster validation (plan §3)",
    "activity": "Activity profile validation (plan §4)",
    "missingness": "Missing driving-summary features (plan §5)",
}


def load_all_inputs(
    processed_dir: Path = sample_module.DEFAULT_PROCESSED_DIR,
    interim_dir: Path = activity_module.DEFAULT_INTERIM_DIR,
) -> dict[str, pd.DataFrame]:
    """Load every pipeline artifact this report reads, once, so each
    validation module's `run_*` function can be called against in-memory
    frames rather than re-reading Parquet per module.
    """
    return {
        "source_population": population_validation.load_source_population(processed_dir),
        "synthetic_employees": population_validation.load_synthetic_population(processed_dir),
        "employee_clusters_full": activity_module.load_employee_clusters(processed_dir),
        "synthetic_activity": activity_validation.load_synthetic_activity(processed_dir),
        "trips_clean": activity_module.load_trips_clean(interim_dir),
    }


def run_all_validations(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Run every module's pass/fail-gated checks. Returns one DataFrame per
    section, all sharing `common.RESULT_COLUMNS`.
    """
    return {
        "population": population_validation.run_population_validation(
            data["source_population"], data["synthetic_employees"]
        ),
        "clusters": cluster_validation.run_cluster_validation(
            data["source_population"], data["synthetic_employees"]
        ),
        "activity": activity_validation.run_activity_validation(
            data["employee_clusters_full"],
            data["trips_clean"],
            data["synthetic_activity"],
            data["synthetic_employees"],
        ),
        "missingness": missingness_validation.run_missingness_validation(
            data["source_population"],
            data["synthetic_employees"],
            data["employee_clusters_full"],
            data["trips_clean"],
        ),
    }


def run_diagnostics(
    data: dict[str, pd.DataFrame], processed_dir: Path = sample_module.DEFAULT_PROCESSED_DIR
) -> dict[str, pd.DataFrame]:
    """Informational tables that don't carry their own pass/fail verdict
    (plan §3 items 1/3) - a human reads these alongside the gated results.
    """
    diagnostics = {
        "cluster_centroids_source": cluster_validation.profile_cluster_centroids(
            data["source_population"], "source"
        ),
        "cluster_centroids_synthetic": cluster_validation.profile_cluster_centroids(
            data["synthetic_employees"], "synthetic"
        ),
    }
    evaluation_path = Path(processed_dir) / cluster_module.CLUSTER_EVALUATION_FILENAME
    if evaluation_path.exists():
        diagnostics["cluster_evaluation"] = pd.read_csv(evaluation_path)
    try:
        diagnostics["hierarchical_cross_check"] = cluster_validation.hierarchical_cross_check(
            processed_dir
        )
    except FileNotFoundError:
        logger.warning("hierarchical_cross_check: employee_features.parquet not found, skipping")
    return diagnostics


def summarize(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pass/fail counts per section, over gating checks only (`passed`
    not-null) - informational rows (`passed is None`) are excluded from the
    denominator rather than counted as failures.
    """
    rows = []
    for section, df in results.items():
        gating = df.loc[df["passed"].notna()]
        # `passed` is an object-dtype column whenever a section mixes
        # True/False/None (informational rows) - `~` on Python bool objects
        # in an object array invokes int.__invert__ (~True == -2), not
        # boolean negation, so n_fail is computed via subtraction instead
        # of bitwise-not to stay correct regardless of dtype.
        n_pass = int(gating["passed"].sum())
        n_fail = len(gating) - n_pass
        n_info = int(df["passed"].isna().sum())
        rows.append(
            {
                "section": section,
                "checks_passed": n_pass,
                "checks_failed": n_fail,
                "checks_informational": n_info,
                "total_checks": len(df),
            }
        )
    return pd.DataFrame(rows)


def _format_result_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No checks produced results for this section._\n"
    display = df.copy()
    display["statistic"] = display["statistic"].map(
        lambda v: f"{v:.4f}" if pd.notna(v) else ""
    )
    display["p_value"] = display["p_value"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
    display["passed"] = display["passed"].map(
        lambda v: "PASS" if v is True else ("FAIL" if v is False else "info")
    )
    columns = [
        "metric",
        "group",
        "chain_source",
        "test",
        "statistic",
        "p_value",
        "n_source",
        "n_synthetic",
        "threshold",
        "passed",
        "detail",
    ]
    return display[columns].fillna("").to_markdown(index=False)


def render_markdown(
    results: dict[str, pd.DataFrame], diagnostics: dict[str, pd.DataFrame]
) -> str:
    """Render the combined results into the Markdown report saved as
    `docs/validation_results.md` - metrics calculated, pass/fail
    interpretation, findings, and issues requiring investigation, per the
    task's required doc sections.
    """
    summary = summarize(results)
    lines: list[str] = []

    lines.append("# Synthetic Population Validation Results")
    lines.append("")
    lines.append(
        "Generated by `driving_profiles.validation.report` against the committed "
        "pipeline outputs in `data/processed/` and `data/interim/`. Implements the "
        "checks specified in `docs/validation_plan.md`. This document is a measurement "
        "report only - no missing value was imputed and no generation logic was changed "
        "to produce it (`docs/validation_plan.md`'s explicit instruction)."
    )
    lines.append("")

    lines.append("## 1. Summary")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    total_pass = int(summary["checks_passed"].sum())
    total_fail = int(summary["checks_failed"].sum())
    total_info = int(summary["checks_informational"].sum())
    lines.append(
        f"**{total_pass} passed, {total_fail} failed, {total_info} informational** "
        "across all sections. See §2-5 for the full per-metric table; §6 for "
        "narrative findings; §7 for issues requiring further investigation."
    )
    lines.append("")

    lines.append("## 2. Metrics calculated and pass/fail interpretation")
    lines.append("")
    lines.append(
        "Every row below follows the same schema: `test` names the comparison method "
        "(`ks_2samp` = two-sample Kolmogorov-Smirnov, `chi_square` = goodness-of-fit "
        "against source proportions, `proportion_diff` = simple share comparison, "
        "`structural` = a deterministic-by-construction check where any violation is a "
        "bug rather than natural variation, `diagnostic_estimate`/`cohens_d`/`proportion` "
        "= informational, no pass/fail gate). `passed` is blank (`info`) for informational "
        "rows - these are not counted as failures in §1's summary."
    )
    lines.append("")
    for section, df in results.items():
        lines.append(f"### {SECTION_TITLES.get(section, section)}")
        lines.append("")
        lines.append(_format_result_table(df))
        lines.append("")

    lines.append("## 3. Cluster centroid profiles (diagnostic)")
    lines.append("")
    lines.append(
        "Mean/median/IQR of each clustering feature, per cluster, source vs. synthetic "
        "(plan §3 item 1) - informational input to judging whether the clusters are "
        "behaviorally meaningful, not a pass/fail row."
    )
    lines.append("")
    centroids = pd.concat(
        [diagnostics.get("cluster_centroids_source", pd.DataFrame()),
         diagnostics.get("cluster_centroids_synthetic", pd.DataFrame())],
        ignore_index=True,
    )
    if not centroids.empty:
        centroids = centroids.round(2)
        lines.append(centroids.to_markdown(index=False))
    lines.append("")

    if "cluster_evaluation" in diagnostics:
        lines.append("## 4. Cluster count (k) evaluation")
        lines.append("")
        lines.append(
            "From `cluster_evaluation.csv` (already produced by `features/cluster.py`), "
            "reproduced here for reference alongside the hierarchical cross-check below."
        )
        lines.append("")
        lines.append(diagnostics["cluster_evaluation"].to_markdown(index=False))
        lines.append("")

    if "hierarchical_cross_check" in diagnostics:
        lines.append("## 5. Hierarchical clustering cross-check (diagnostic)")
        lines.append("")
        lines.append(
            "Ward-linkage merge height at which cutting the dendrogram yields each "
            "candidate k (plan §3 item 3), fit on the same preprocessed feature matrix "
            "`features/cluster.py`'s KMeans uses. A large `height_drop_to_next_k` is "
            "evidence for a natural cut at that k; this table is a diagnostic input to a "
            "human domain review, not a k-selection algorithm."
        )
        lines.append("")
        lines.append(diagnostics["hierarchical_cross_check"].round(2).to_markdown(index=False))
        lines.append("")

    lines.append("## 6. Findings")
    lines.append("")
    lines.append(_render_findings(results, diagnostics))
    lines.append("")

    lines.append("## 7. Issues requiring investigation")
    lines.append("")
    lines.append(_render_open_issues(results, diagnostics))
    lines.append("")

    return "\n".join(lines)


def _render_findings(
    results: dict[str, pd.DataFrame], diagnostics: dict[str, pd.DataFrame]
) -> str:
    findings = []

    proportions = results["clusters"]
    share_rows = proportions.loc[proportions["metric"] == "cluster_share"]
    if not share_rows.empty and share_rows["passed"].all():
        findings.append(
            "- **Cluster proportions match by construction.** Every cluster's synthetic "
            "share is within the pass threshold of its source share - sampling has no "
            "evident proportion bug."
        )

    missingness = results["missingness"]
    pooled = missingness.loc[
        (missingness["metric"] == "missingness_rate") & (missingness["group"] == "pooled")
    ]
    if not pooled.empty:
        row = pooled.iloc[0]
        verdict = "within tolerance" if row["passed"] else "OUTSIDE tolerance"
        findings.append(
            f"- **Missingness rate is {verdict}.** Pooled `total_daily_miles` null rate: "
            f"{row['detail']} (diff={row['statistic']:.2f}pp, threshold {row['threshold']})."
        )

    cooccur = missingness.loc[missingness["metric"] == "missingness_cooccurrence"]
    if not cooccur.empty and cooccur["passed"].all():
        findings.append(
            "- **Missingness co-occurs 100% of the time** across "
            "`total_daily_miles`/`total_driving_minutes`/`average_trip_distance_miles`, "
            "in both source and synthetic populations - no partial-null pipeline "
            "inconsistency found."
        )

    jitter = missingness.loc[missingness["metric"] == "jitter_preserves_source_nan"]
    if not jitter.empty:
        row = jitter.iloc[0]
        verdict = "confirmed" if row["passed"] else "VIOLATED"
        findings.append(
            f"- **Jitter-preserves-NaN is {verdict}** at full population scale: "
            f"{row['detail']}."
        )

    workplace = results["activity"]
    arrival = workplace.loc[workplace["metric"] == "workplace_arrival_matches_drawn_value"]
    departure = workplace.loc[workplace["metric"] == "workplace_departure_matches_drawn_value"]
    if not arrival.empty:
        row = arrival.iloc[0]
        verdict = "holds" if row["passed"] else "DOES NOT hold"
        findings.append(
            f"- **Workplace arrival-time consistency {verdict}**: "
            f"{row['detail']}."
        )
    if not departure.empty:
        row = departure.iloc[0]
        verdict = "holds" if row["passed"] else "DOES NOT hold"
        findings.append(
            f"- **Workplace departure-time consistency {verdict}**: "
            f"{row['detail']}."
        )

    speed = workplace.loc[workplace["metric"] == "implied_leg_speed_plausible"]
    if not speed.empty:
        row = speed.iloc[0]
        verdict = "holds" if row["passed"] else "DOES NOT hold"
        findings.append(f"- **Implied leg-speed plausibility {verdict}**: {row['detail']}.")

    fallback = workplace.loc[
        (workplace["metric"] == "fallback_chain_rate") & (workplace["group"] == "pooled")
    ]
    if not fallback.empty:
        rate = fallback.iloc[0]["statistic"]
        findings.append(f"- **Fallback chain rate (pooled): {rate:.2%}.**")

    donor_blind = results["missingness"].loc[
        results["missingness"]["metric"] == "donor_mode_mismatch"
    ]
    if not donor_blind.empty:
        row = donor_blind.iloc[0]
        verdict = "holds" if row["passed"] else "DOES NOT hold"
        findings.append(
            f"- **Donor mode-blindness fix {verdict} (donor mode mismatch check)**: "
            f"{row['detail']}."
        )

    if not findings:
        findings.append("- No findings synthesized (see §2 tables for raw results).")

    return "\n".join(findings)


def _render_open_issues(
    results: dict[str, pd.DataFrame], diagnostics: dict[str, pd.DataFrame]
) -> str:
    issues = []

    for section, df in results.items():
        failed = df.loc[df["passed"] == False]  # noqa: E712
        for _, row in failed.iterrows():
            issues.append(
                f"- **[{section}] {row['metric']}** (group={row['group']}"
                + (f", chain_source={row['chain_source']}" if row["chain_source"] else "")
                + f"): {row['detail']} - failed threshold `{row['threshold']}`."
            )

    if "hierarchical_cross_check" in diagnostics:
        table = diagnostics["hierarchical_cross_check"]
        if not table.empty and "height_drop_to_next_k" in table.columns:
            issues.append(
                "- **Cluster count (k=2) meaningfulness** (plan §3): review the "
                "hierarchical cross-check table (§5) and cluster centroid profiles (§3) "
                "against `cluster_evaluation.csv`'s silhouette scores before treating "
                "`cluster_id` as an uncontested behavioral archetype - the domain-"
                "interpretability review `docs/clustering_plan.md` §6 calls for has not "
                "been separately documented."
            )

    donor_blind = results["missingness"].loc[
        results["missingness"]["metric"] == "donor_mode_mismatch"
    ]
    if not donor_blind.empty:
        row = donor_blind.iloc[0]
        if not row["passed"]:
            issues.append(
                f"- **Donor mode mismatch ({int(row['statistic'])} violation(s))**: "
                "`generator/activity.py`'s `select_donor` restricts candidate donors by "
                "`has_driving_leg`, but a candidate donor's own `total_daily_miles` "
                "nullness disagreed with that flag for at least one synthetic employee - "
                f"{row['detail']}. Indicates `has_driving_leg` (derived from `TRPTRANS`) "
                "and `total_daily_miles` nullness (derived in `build_features.py`) have "
                "drifted out of sync for some donor."
            )

    if not issues:
        issues.append("- No open issues found by these checks (see §2 for the full results).")

    return "\n".join(issues)


def save_report(text: str, path: Path = DEFAULT_REPORT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def main(
    processed_dir: Path = sample_module.DEFAULT_PROCESSED_DIR,
    interim_dir: Path = activity_module.DEFAULT_INTERIM_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    data = load_all_inputs(processed_dir, interim_dir)
    results = run_all_validations(data)
    diagnostics = run_diagnostics(data, processed_dir)
    text = render_markdown(results, diagnostics)
    return save_report(text, report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_path = main()
    logger.info("Wrote validation report to %s", output_path)
