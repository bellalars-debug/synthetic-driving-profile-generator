"""Shared statistical helpers and result-row schema for the validation package.

Every validation module (population.py, clusters.py, activity.py,
missingness.py) returns a `pd.DataFrame` of result rows in the schema
defined by `RESULT_COLUMNS`, so `report.py` can concatenate every module's
output into one table without per-module special-casing. Per
docs/validation_plan.md, the recurring pass criterion for a distributional
check is "KS test p > 0.05" (fail to reject "same distribution"); helpers
here follow that convention unless a specific metric's plan entry calls for
something else (proportion/percentile/structural checks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

RESULT_COLUMNS = [
    "section",
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

DEFAULT_ALPHA = 0.05


def result_row(
    section: str,
    metric: str,
    group: str = "pooled",
    chain_source: str | None = None,
    test: str = "",
    statistic: float | None = None,
    p_value: float | None = None,
    n_source: int | None = None,
    n_synthetic: int | None = None,
    threshold: str = "",
    passed: bool | None = None,
    detail: str = "",
) -> dict:
    """Build one row of the shared validation-result schema.

    `passed` is `None` for informational/diagnostic checks that aren't a
    pass/fail gate (e.g. cluster effect-size reporting, the donor
    mode-blindness rate) - report.py treats `None` as "not counted" in its
    pass/fail summary rather than as a failure.
    """
    return {
        "section": section,
        "metric": metric,
        "group": group,
        "chain_source": chain_source,
        "test": test,
        "statistic": statistic,
        "p_value": p_value,
        "n_source": n_source,
        "n_synthetic": n_synthetic,
        "threshold": threshold,
        "passed": passed,
        "detail": detail,
    }


def results_frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble result rows into a DataFrame with a stable column order,
    even when `rows` is empty (so callers can always rely on the schema)."""
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def ks_result(
    section: str,
    metric: str,
    source: pd.Series,
    synthetic: pd.Series,
    group: str = "pooled",
    chain_source: str | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Two-sample KS test between `source` and `synthetic`, NaN dropped from
    both sides before comparison (non-null-subset comparisons per plan §2/§4
    are the caller's responsibility to slice before calling this - dropping
    NaN here is what makes that slicing safe either way).
    """
    src = pd.to_numeric(source, errors="coerce").dropna().to_numpy(dtype=float)
    syn = pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy(dtype=float)
    if len(src) < 2 or len(syn) < 2:
        return result_row(
            section,
            metric,
            group,
            chain_source,
            test="ks_2samp",
            n_source=len(src),
            n_synthetic=len(syn),
            threshold=f"p > {alpha}",
            passed=None,
            detail="insufficient non-null observations for a KS test",
        )
    stat, p = stats.ks_2samp(src, syn)
    return result_row(
        section,
        metric,
        group,
        chain_source,
        test="ks_2samp",
        statistic=float(stat),
        p_value=float(p),
        n_source=len(src),
        n_synthetic=len(syn),
        threshold=f"p > {alpha}",
        passed=bool(p > alpha),
        detail=(
            f"source mean={src.mean():.2f} sd={src.std(ddof=0):.2f}; "
            f"synthetic mean={syn.mean():.2f} sd={syn.std(ddof=0):.2f}"
        ),
    )


def variance_ratio_result(
    section: str,
    metric: str,
    source: pd.Series,
    synthetic: pd.Series,
    group: str = "pooled",
    chain_source: str | None = None,
    max_rel_diff: float = 0.20,
) -> dict:
    """Compare variance, not just the KS statistic - a generator that
    flattens a discrete count column's spread (plan §2's trips_per_day
    concern) can still pass a KS test on the CDF while understating spread.
    """
    src = pd.to_numeric(source, errors="coerce").dropna()
    syn = pd.to_numeric(synthetic, errors="coerce").dropna()
    if len(src) < 2 or len(syn) < 2:
        return result_row(
            section,
            metric,
            group,
            chain_source,
            test="variance_ratio",
            n_source=len(src),
            n_synthetic=len(syn),
            threshold=f"variance within {max_rel_diff:.0%} of source",
            passed=None,
            detail="insufficient non-null observations",
        )
    src_var = float(src.var(ddof=0))
    syn_var = float(syn.var(ddof=0))
    rel_diff = abs(syn_var - src_var) / src_var if src_var != 0 else float("inf")
    return result_row(
        section,
        metric,
        group,
        chain_source,
        test="variance_ratio",
        statistic=rel_diff,
        n_source=len(src),
        n_synthetic=len(syn),
        threshold=f"variance within {max_rel_diff:.0%} of source",
        passed=bool(rel_diff <= max_rel_diff),
        detail=f"source variance={src_var:.2f}, synthetic variance={syn_var:.2f}",
    )


def chi_square_result(
    section: str,
    metric: str,
    source: pd.Series,
    synthetic: pd.Series,
    group: str = "pooled",
    chain_source: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    max_share_diff_pp: float = 2.0,
) -> dict:
    """Chi-square goodness-of-fit: does synthetic's categorical distribution
    match source's, using source's observed shares as expected proportions.

    Pass requires both `p > alpha` and every category's synthetic share
    within `max_share_diff_pp` percentage points of its source share - a
    large `n` can make the chi-square test reject on a practically
    negligible share difference, so the plan's per-band percentage-point
    tolerance is checked directly rather than relying on the p-value alone.
    """
    src = source.dropna()
    syn = synthetic.dropna()
    categories = sorted(set(src.unique()) | set(syn.unique()), key=str)
    src_counts = src.value_counts().reindex(categories, fill_value=0)
    syn_counts = syn.value_counts().reindex(categories, fill_value=0)

    if len(src) == 0 or len(syn) == 0:
        return result_row(
            section,
            metric,
            group,
            chain_source,
            test="chi_square",
            n_source=len(src),
            n_synthetic=len(syn),
            threshold=f"p > {alpha} and max category-share diff <= {max_share_diff_pp}pp",
            passed=None,
            detail="no observations to compare",
        )

    src_share = src_counts / len(src)
    syn_share = syn_counts / len(syn)
    expected = src_share * len(syn)

    valid = expected > 0
    if not valid.any():
        return result_row(
            section,
            metric,
            group,
            chain_source,
            test="chi_square",
            n_source=len(src),
            n_synthetic=len(syn),
            threshold=f"p > {alpha} and max category-share diff <= {max_share_diff_pp}pp",
            passed=None,
            detail="no category has a positive expected count",
        )
    stat, p = stats.chisquare(f_obs=syn_counts[valid], f_exp=expected[valid])

    max_diff_pp = float((100 * (syn_share - src_share)).abs().max())
    passed = bool(p > alpha) and max_diff_pp <= max_share_diff_pp
    return result_row(
        section,
        metric,
        group,
        chain_source,
        test="chi_square",
        statistic=float(stat),
        p_value=float(p),
        n_source=len(src),
        n_synthetic=len(syn),
        threshold=f"p > {alpha} and max category-share diff <= {max_share_diff_pp}pp",
        passed=passed,
        detail=f"max category-share diff = {max_diff_pp:.2f}pp",
    )


def proportion_result(
    section: str,
    metric: str,
    source: pd.Series,
    synthetic: pd.Series,
    group: str = "pooled",
    chain_source: str | None = None,
    max_diff_pp: float = 1.0,
) -> dict:
    """Boolean/binary proportion comparison (source share vs. synthetic
    share), e.g. `vehicle_per_driver_adequate` (plan §2's household table).
    """
    src = source.dropna().astype(bool)
    syn = synthetic.dropna().astype(bool)
    if len(src) == 0 or len(syn) == 0:
        return result_row(
            section,
            metric,
            group,
            chain_source,
            test="proportion_diff",
            n_source=len(src),
            n_synthetic=len(syn),
            threshold=f"diff <= {max_diff_pp}pp",
            passed=None,
            detail="no observations to compare",
        )
    src_share = float(src.mean())
    syn_share = float(syn.mean())
    diff_pp = abs(syn_share - src_share) * 100
    return result_row(
        section,
        metric,
        group,
        chain_source,
        test="proportion_diff",
        statistic=diff_pp,
        n_source=len(src),
        n_synthetic=len(syn),
        threshold=f"diff <= {max_diff_pp}pp",
        passed=bool(diff_pp <= max_diff_pp),
        detail=f"source={src_share:.4f} synthetic={syn_share:.4f}",
    )


def percentile_result(
    section: str,
    metric: str,
    source: pd.Series,
    synthetic: pd.Series,
    percentile: int = 90,
    group: str = "pooled",
    chain_source: str | None = None,
    max_rel_diff: float = 0.10,
) -> dict:
    """Tail-percentile comparison (plan §2: commute-distance 90th percentile
    within ~10% of source) - a KS test alone can pass while the tail, which
    matters disproportionately for energy estimates, has drifted.
    """
    src = pd.to_numeric(source, errors="coerce").dropna()
    syn = pd.to_numeric(synthetic, errors="coerce").dropna()
    if src.empty or syn.empty:
        return result_row(
            section,
            metric,
            group,
            chain_source,
            test=f"p{percentile}_diff",
            n_source=len(src),
            n_synthetic=len(syn),
            threshold=f"within {max_rel_diff:.0%} of source p{percentile}",
            passed=None,
            detail="insufficient observations",
        )
    src_p = float(np.percentile(src, percentile))
    syn_p = float(np.percentile(syn, percentile))
    rel_diff = abs(syn_p - src_p) / abs(src_p) if src_p != 0 else float("inf")
    return result_row(
        section,
        metric,
        group,
        chain_source,
        test=f"p{percentile}_diff",
        statistic=rel_diff,
        n_source=len(src),
        n_synthetic=len(syn),
        threshold=f"within {max_rel_diff:.0%} of source p{percentile}",
        passed=bool(rel_diff <= max_rel_diff),
        detail=f"source p{percentile}={src_p:.2f}, synthetic p{percentile}={syn_p:.2f}",
    )


def structural_result(
    section: str,
    metric: str,
    n_violations: int,
    n_checked: int,
    group: str = "pooled",
    chain_source: str | None = None,
    threshold: str = "0 violations",
    detail: str = "",
) -> dict:
    """A deterministic-by-construction check (plan calls these out
    separately from distributional tests, e.g. workplace arrival/departure
    consistency, missingness co-occurrence) - any violation indicates a bug,
    not natural sampling variation.
    """
    return result_row(
        section,
        metric,
        group,
        chain_source,
        test="structural",
        statistic=float(n_violations),
        n_synthetic=n_checked,
        threshold=threshold,
        passed=bool(n_violations == 0),
        detail=detail or f"{n_violations}/{n_checked} violation(s)",
    )
