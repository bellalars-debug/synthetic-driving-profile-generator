"""Validate cluster proportions and behavioral differences survive synthetic
generation (docs/validation_plan.md §3).

Three separate questions, kept as separate functions since they answer
different parts of the plan:

1. `validate_cluster_proportions` - did sampling draw the right *number* of
   employees from each cluster (a mechanical sampling-bug check, expected to
   pass almost exactly by construction).
2. `profile_cluster_centroids` / `validate_cluster_separation` - are the
   clusters behaviorally *meaningful* in the first place (plan §3's
   domain-interpretability question), and does that separation survive into
   the synthetic population.
3. `hierarchical_cross_check` - a method-agnostic diagnostic (plan §3 item
   3) for whether KMeans's k=2 spherical-cluster assumption forced a 2-way
   split where the data doesn't obviously support one.

Read-only: reuses features/cluster.py's own feature-selection/preprocessing
functions rather than re-deriving them, and never re-fits or overwrites the
committed cluster assignments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from driving_profiles.features import cluster as cluster_module
from driving_profiles.validation import common

SECTION = "clusters"


def validate_cluster_proportions(
    source: pd.DataFrame, synthetic: pd.DataFrame, max_diff_pp: float = 2.0
) -> pd.DataFrame:
    """Cluster share comparison (plan §3 "Cluster population proportions") -
    expected to match almost exactly for default (non-overridden) sampling;
    this is mainly a check that the sampling code has no bug, not a
    fidelity test.
    """
    src_share = source["cluster_id"].value_counts(normalize=True).sort_index()
    syn_share = synthetic["cluster_id"].value_counts(normalize=True).sort_index()
    rows = []
    for cluster_id in sorted(set(src_share.index) | set(syn_share.index)):
        s = float(src_share.get(cluster_id, 0.0))
        y = float(syn_share.get(cluster_id, 0.0))
        diff_pp = abs(y - s) * 100
        rows.append(
            common.result_row(
                SECTION,
                "cluster_share",
                group=f"cluster_{cluster_id}",
                test="proportion_diff",
                statistic=diff_pp,
                n_source=int((source["cluster_id"] == cluster_id).sum()),
                n_synthetic=int((synthetic["cluster_id"] == cluster_id).sum()),
                threshold=f"diff <= {max_diff_pp}pp",
                passed=bool(diff_pp <= max_diff_pp),
                detail=f"source={s:.4f} synthetic={y:.4f}",
            )
        )
    return common.results_frame(rows)


def profile_cluster_centroids(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """Mean/median/IQR of `CONTINUOUS_FEATURES` per cluster (plan §3 item 1):
    informational input to a domain-interpretability review, not a pass/fail
    check - a validator reads this table to judge whether each cluster is
    something a non-technical reader could label.
    """
    rows = []
    for cluster_id, group in df.groupby("cluster_id"):
        for column in cluster_module.CONTINUOUS_FEATURES:
            values = group[column].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "dataset": dataset_label,
                    "cluster_id": cluster_id,
                    "feature": column,
                    "n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def validate_cluster_separation(source: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Effect size (Cohen's d) between the two clusters' means on every
    `CONTINUOUS_FEATURES` column, computed separately in source and
    synthetic (plan §3 item 2: "genuinely separated... not just a shifted
    mean with heavy overlap"). Informational, not a pass/fail gate - the
    plan wants a human to compare the source vs. synthetic effect size, not
    a mechanical threshold, since "how much separation is enough" is a
    domain judgment call.

    Only defined when there are exactly two clusters (this project's
    current `k=2`); returns an empty frame otherwise rather than guessing
    which pair of clusters to compare.
    """
    cluster_ids = sorted(source["cluster_id"].dropna().unique())
    rows = []
    if len(cluster_ids) != 2:
        return common.results_frame(rows)
    low, high = cluster_ids
    for label, df in (("source", source), ("synthetic", synthetic)):
        for column in cluster_module.CONTINUOUS_FEATURES:
            a = df.loc[df["cluster_id"] == low, column].dropna()
            b = df.loc[df["cluster_id"] == high, column].dropna()
            if a.empty or b.empty:
                continue
            pooled_sd = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2)
            effect = float((b.mean() - a.mean()) / pooled_sd) if pooled_sd > 0 else float("nan")
            rows.append(
                common.result_row(
                    SECTION,
                    f"{column}_effect_size",
                    group=label,
                    test="cohens_d",
                    statistic=effect,
                    n_source=len(a),
                    n_synthetic=len(b),
                    threshold="informational - compare source vs. synthetic effect size",
                    passed=None,
                    detail=(
                        f"cluster_{low} mean={a.mean():.2f}, cluster_{high} mean={b.mean():.2f}"
                    ),
                )
            )
    return common.results_frame(rows)


def hierarchical_cross_check(
    processed_dir: Path = cluster_module.DEFAULT_PROCESSED_DIR,
    sample_size: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit a Ward-linkage hierarchical clustering on the same preprocessed
    feature matrix cluster.py's KMeans fits (plan §3 item 3), and report the
    merge height at which cutting the dendrogram would yield each candidate
    `k`.

    This is a diagnostic table, not a k-selection algorithm: the plan is
    explicit that k should not be picked by a single mechanical metric, so
    this function deliberately stops at reporting merge heights (a big gap
    between the height for k and k+1 clusters is evidence for a natural cut
    at k) rather than asserting a "correct" k itself - a human should read
    the table alongside `profile_cluster_centroids`/`validate_cluster_separation`.

    Runs on a random sample of `sample_size` rows (Ward linkage is O(n^2)
    memory) rather than the full clustering population.
    """
    employee_features = cluster_module.load_employee_features(processed_dir)
    selected = cluster_module.select_clustering_features(employee_features)
    rng = np.random.default_rng(seed)
    if len(selected) > sample_size:
        idx = rng.choice(len(selected), size=sample_size, replace=False)
        selected = selected.iloc[idx].reset_index(drop=True)
    _, X = cluster_module.preprocess_features(selected)

    Z = linkage(X.to_numpy(), method="ward")
    n = len(X)

    rows = []
    for k in range(2, 9):
        merge_index = n - k - 1
        if merge_index < 0 or merge_index >= len(Z):
            continue
        rows.append({"k": k, "n_sampled": n, "merge_height": float(Z[merge_index, 2])})
    result = pd.DataFrame(rows)
    if not result.empty:
        # merge_height decreases as k increases (cutting closer to the
        # leaves); diff(-1)[i] = merge_height[i] - merge_height[i+1] is the
        # (positive) drop from k to k+1 - a large drop is evidence for a
        # natural cut point at k.
        result["height_drop_to_next_k"] = result["merge_height"].diff(-1)
    return result


def run_cluster_validation(
    source: pd.DataFrame, synthetic: pd.DataFrame, processed_dir: Path | None = None
) -> pd.DataFrame:
    """Run the pass/fail-gated §3 checks (proportions + separation) and
    return one combined result table. `profile_cluster_centroids` and
    `hierarchical_cross_check` return separate diagnostic tables (not this
    schema) since they're informational inputs to a human review, not
    results with their own pass/fail row - see report.py for how they're
    surfaced.
    """
    return pd.concat(
        [
            validate_cluster_proportions(source, synthetic),
            validate_cluster_separation(source, synthetic),
        ],
        ignore_index=True,
    )


if __name__ == "__main__":
    import logging

    from driving_profiles.validation import population as population_validation

    logging.basicConfig(level=logging.INFO)
    source_population = population_validation.load_source_population()
    synthetic_population = population_validation.load_synthetic_population()

    results = run_cluster_validation(source_population, synthetic_population)
    print(results.to_string())

    dendrogram_table = hierarchical_cross_check()
    print(dendrogram_table.to_string())
