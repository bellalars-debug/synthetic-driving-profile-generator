"""Generate synthetic employee profiles: static per-employee attributes plus
cluster/archetype membership.

Produces the "synthetic employee profiles" pipeline artifact
(`data/processed/synthetic_employees.parquet`). Reads only
`data/processed/employee_clusters.parquet` (the `cluster.py` output) - no
raw NHTS files, and no re-derivation of anything `build_features.py` /
`cluster.py` already computed. Trip-chain reconstruction is out of scope
here; see `generator/activity.py` (not yet implemented) for that, which is
the only stage that should read `data/interim/trips_clean.parquet`.
Methodology: `docs/synthetic_generation_plan.md`.

## Sampling strategy (plan §3-4)

Each synthetic employee is generated in two steps:

1. Draw a `cluster_id`, weighted by that cluster's share of the real
   clustered population by default (`determine_cluster_sampling`), or by an
   explicit caller-supplied override for scenario modeling.
2. Draw a **whole real row** at random from that cluster (not independent
   per-column sampling - see `sample_employees`), then jitter its continuous
   travel-behavior fields (`JITTER_FEATURES`) with Gaussian noise scaled to
   that cluster's own within-cluster standard deviation. This is the plan's
   "resample-with-noise" approach (§4 step 2): it preserves the joint
   structure between commute distance, arrival time, vehicle availability,
   etc. that would be destroyed by sampling each column from its marginal,
   while the jitter keeps a synthetic employee from being a byte-identical
   copy of the real respondent it was drawn from.

## ID handling: a deliberate deviation from plan §4 step 3

Plan §4 step 3 argues real `HOUSEID`/`PERSONID` should not appear in
generator output at all, for privacy (individual NHTS diary days
shouldn't be traceable) and to avoid the resample step accidentally
reproducing an identifiable real row. The task that authorized this
implementation explicitly requires the opposite - retaining the original
`PERSONID`/`HOUSEID` in the output for traceability during development.
`assign_unique_employee_ids` follows the task requirement: it assigns a new
`synthetic_employee_id` as the table's primary key and keeps the source IDs
in clearly separate `source_houseid`/`source_personid` columns rather than
overwriting the primary key with them, so the two concerns (a synthetic
primary key vs. a traceability breadcrumb) stay visibly distinct. Anyone
hardening this for a non-development release should revisit this choice.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from driving_profiles.features.cluster import CONTINUOUS_FEATURES, PERSON_KEY
from driving_profiles.generator.time_utils import MINUTES_PER_DAY, hhmm_to_minutes, minutes_to_hhmm
from driving_profiles.utils import random_seed

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")
CLUSTER_TABLE_FILENAME = "employee_clusters.parquet"
SYNTHETIC_EMPLOYEE_FILENAME = "synthetic_employees.parquet"

DEFAULT_N = 5000

# Continuous travel-behavior fields perturbed during resampling (plan §4
# step 2). Reused directly from cluster.py's CONTINUOUS_FEATURES - these are
# exactly the columns that define a cluster archetype, so they're also the
# ones where getting joint (not marginal) sampling right matters most, and
# where privacy jitter matters most (they're the fields closest to being
# individually identifying, e.g. an exact commute distance).
JITTER_FEATURES = list(CONTINUOUS_FEATURES)

# Fraction of a cluster's own within-cluster standard deviation used as the
# jitter's Gaussian sigma (plan §4: "jitter magnitude is a real design
# parameter, not just a smoothing nicety"). Small enough to preserve
# cluster-level distributional shape, large enough that a synthetic
# employee's continuous fields are not byte-identical to the real
# respondent it was resampled from.
JITTER_SCALE = 0.15

# HHMM-coded time-of-day columns: jittered in true minutes-since-midnight
# space (converted via hhmm_to_minutes, perturbed, clamped to
# [0, MINUTES_PER_DAY - 1], converted back via minutes_to_hhmm), never
# rounded to integer. Jittering the raw HHMM number directly (e.g.
# 830 + noise) was this project's original approach and is exactly wrong:
# a clock's minute digit rolls over at 60, not 100, so raw-HHMM noise
# routinely produces an invalid minute component (e.g. 880 = "8:80") and,
# more importantly, a systematic later-time bias (a negative HHMM offset
# crossing the encoding's 100-boundary is decoded ~40 minutes later than a
# true clock rollback would be - see docs/activity_validation_investigation.md
# for the measured effect). Converting to minutes first makes the jitter's
# additive noise genuinely additive in elapsed time, eliminating both
# problems.
TIME_OF_DAY_FEATURES = ["work_arrival_time", "work_departure_time"]

# Integer count columns: rounded after jitter and clamped to their observed
# minimum (every clustered row has an observed commute, so both are >= 1 in
# the source population - see module-level checks in tests).
COUNT_FEATURES = ["trips_per_day", "number_of_stops"]

# Remaining continuous fields: clamped to >= 0 after jitter (a distance,
# duration, or ratio can't be negative).
NON_NEGATIVE_FEATURES = [
    c for c in JITTER_FEATURES if c not in TIME_OF_DAY_FEATURES and c not in COUNT_FEATURES
]

SYNTHETIC_ID_COLUMN = "synthetic_employee_id"
SOURCE_HOUSEID_COLUMN = "source_houseid"
SOURCE_PERSONID_COLUMN = "source_personid"


def load_clustered_employees(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Read cluster.py's output and filter to the clustered population.

    Per docs/synthetic_generation_plan.md §2, only rows with a non-null
    `cluster_id` have an archetype distribution to sample from; unclustered
    rows (non-workers, workers without an observed weekday commute) are
    dropped here rather than passed through, since resampling them would
    have no distribution to draw from.
    """
    path = Path(processed_dir) / CLUSTER_TABLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Clustered employee table not found: {path}. Run "
            "`python -m driving_profiles.features.cluster` first."
        )
    employee_clusters = pd.read_parquet(path)
    clustered = employee_clusters.loc[employee_clusters["cluster_id"].notna()].reset_index(
        drop=True
    )
    logger.info(
        "load_clustered_employees: %d/%d employee(s) have a cluster_id (clustered population)",
        len(clustered),
        len(employee_clusters),
    )
    return clustered


def determine_cluster_sampling(
    clustered_employees: pd.DataFrame,
    n: int,
    cluster_weights: pd.Series | dict[int, float] | None = None,
    seed: int | None = None,
) -> pd.Series:
    """Decide how many synthetic employees to draw from each cluster.

    Defaults to proportional sampling: each cluster's share of `n` matches
    its share of `clustered_employees` (plan §3). A caller-supplied
    `cluster_weights` overrides this with an explicit archetype mix instead
    (plan §3's "what if this site skews toward long-commute hybrid workers"
    case); weights need not already sum to 1, they're normalized here.

    Counts are allocated by largest-remainder rounding so they always sum to
    exactly `n` while staying as close as possible to the weighted
    proportions. Ties among equal fractional remainders are broken via a
    seeded random permutation rather than index order, so the tie-break
    itself is reproducible but not silently biased toward whichever cluster
    happens to sort first.
    """
    observed_counts = clustered_employees["cluster_id"].value_counts()
    if cluster_weights is None:
        weights = (observed_counts / observed_counts.sum()).sort_index()
    else:
        weights = pd.Series(cluster_weights, dtype=float).sort_index()
        weights = weights / weights.sum()

    raw = weights * n
    floor_counts = np.floor(raw).astype(int)
    remainder = int(n - floor_counts.sum())

    if remainder > 0:
        rng = random_seed.get_rng(seed)
        shuffled_ids = rng.permutation(weights.index.to_numpy())
        fractional = (raw - floor_counts).reindex(shuffled_ids)
        top_up = fractional.sort_values(ascending=False, kind="mergesort").index[:remainder]
        floor_counts.loc[top_up] += 1

    floor_counts.index.name = "cluster_id"
    floor_counts.name = "n_synthetic"
    return floor_counts.sort_index()


def sample_employees(
    clustered_employees: pd.DataFrame,
    cluster_sampling: pd.Series,
    jitter_scale: float = JITTER_SCALE,
    seed: int | None = None,
) -> pd.DataFrame:
    """Draw synthetic employee profiles by resampling real rows within each
    cluster and jittering their continuous travel-behavior fields.

    Implements the "resample-with-noise" approach (plan §4 step 2): a whole
    real row is drawn per synthetic employee - not one independent draw per
    column - so every non-jittered column (demographics, household,
    categorical/boolean travel-behavior fields) keeps exactly the joint
    relationship it had in the real respondent's record. Only
    `JITTER_FEATURES` are perturbed, with Gaussian noise scaled to
    `jitter_scale` times that cluster's own within-cluster standard
    deviation; a `NaN` source value (e.g. an unobserved
    `total_daily_miles`) is left as `NaN` rather than jittered.

    `TIME_OF_DAY_FEATURES` are the one exception to "jitter the column's own
    value directly": they're converted to minutes-since-midnight first
    (`hhmm_to_minutes`), jittered there with sigma from the *minutes-space*
    within-cluster standard deviation, clamped to
    `[0, MINUTES_PER_DAY - 1]`, then converted back (`minutes_to_hhmm`) -
    see `TIME_OF_DAY_FEATURES`'s module-level comment for why jittering the
    raw HHMM number directly is wrong.
    """
    rng = random_seed.get_rng(seed)
    cluster_std = clustered_employees.groupby("cluster_id")[JITTER_FEATURES].std(ddof=0)

    time_of_day_minutes = clustered_employees[TIME_OF_DAY_FEATURES].apply(
        lambda column: column.apply(hhmm_to_minutes)
    )
    cluster_std_minutes = time_of_day_minutes.groupby(clustered_employees["cluster_id"]).std(
        ddof=0
    )

    sampled_parts = []
    for cluster_id, n_draw in cluster_sampling.items():
        if n_draw == 0:
            continue
        pool = clustered_employees.loc[clustered_employees["cluster_id"] == cluster_id]
        if pool.empty:
            raise ValueError(f"cluster_id={cluster_id!r} has no source rows to sample from")

        draw_idx = rng.integers(0, len(pool), size=int(n_draw))
        draw = pool.iloc[draw_idx].reset_index(drop=True)
        draw_minutes = time_of_day_minutes.loc[pool.index].iloc[draw_idx].reset_index(drop=True)

        sigma = cluster_std.loc[cluster_id]
        sigma_minutes = cluster_std_minutes.loc[cluster_id]
        for column in JITTER_FEATURES:
            if column in TIME_OF_DAY_FEATURES:
                col_sigma = sigma_minutes[column]
                if not np.isfinite(col_sigma) or col_sigma == 0:
                    continue
                values = draw_minutes[column].to_numpy(dtype=float).copy()
                not_na = ~np.isnan(values)
                noise = rng.normal(loc=0.0, scale=jitter_scale * col_sigma, size=len(draw))
                values[not_na] = values[not_na] + noise[not_na]
                values = np.clip(values, 0, MINUTES_PER_DAY - 1)
                draw[column] = pd.Series(values, index=draw.index).apply(minutes_to_hhmm)
            else:
                col_sigma = sigma[column]
                if not np.isfinite(col_sigma) or col_sigma == 0:
                    continue
                values = draw[column].to_numpy(dtype=float).copy()
                not_na = ~np.isnan(values)
                noise = rng.normal(loc=0.0, scale=jitter_scale * col_sigma, size=len(draw))
                values[not_na] = values[not_na] + noise[not_na]
                draw[column] = values

        for column in COUNT_FEATURES:
            draw[column] = draw[column].round().clip(lower=1).astype(
                clustered_employees[column].dtype
            )
        for column in NON_NEGATIVE_FEATURES:
            draw[column] = draw[column].clip(lower=0)
        # TIME_OF_DAY_FEATURES already produced a valid HHMM value via
        # minutes_to_hhmm's own [0, MINUTES_PER_DAY - 1] clamp above - no
        # separate raw-HHMM clip(0, 2359) needed anymore.

        sampled_parts.append(draw)

    return pd.concat(sampled_parts, ignore_index=True)


def assign_unique_employee_ids(sampled: pd.DataFrame) -> pd.DataFrame:
    """Replace the resampled real HOUSEID/PERSONID with a synthetic primary
    key, retaining the source identifiers in separate columns.

    `synthetic_employee_id` is a sequential `SYN-XXXXXXXX` string, unique by
    construction within a single generation run. `source_houseid` /
    `source_personid` retain the real respondent IDs the row was resampled
    from, for traceability back to `employee_clusters.parquet` (see the
    module docstring for how/why this differs from plan §4 step 3).
    """
    result = sampled.copy()
    result[SOURCE_HOUSEID_COLUMN] = result["HOUSEID"]
    result[SOURCE_PERSONID_COLUMN] = result["PERSONID"]
    result[SYNTHETIC_ID_COLUMN] = [f"SYN-{i:08d}" for i in range(1, len(result) + 1)]
    return result.drop(columns=PERSON_KEY)


def create_synthetic_employee_table(
    n: int,
    cluster_weights: pd.Series | dict[int, float] | None = None,
    seed: int | None = None,
    jitter_scale: float = JITTER_SCALE,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """Run the full sample.py pipeline end to end: load, decide cluster
    counts, resample-with-noise, assign IDs, and order columns.

    `seed` flows into both `determine_cluster_sampling`'s remainder tie-
    break and `sample_employees`'s row draws/jitter, per plan §3's
    reproducibility requirement - the same `(seed, n, cluster_weights)`
    against the same `employee_clusters.parquet` always produces the same
    synthetic population.
    """
    clustered_employees = load_clustered_employees(processed_dir)
    cluster_sampling = determine_cluster_sampling(clustered_employees, n, cluster_weights, seed)
    sampled = sample_employees(clustered_employees, cluster_sampling, jitter_scale, seed)
    synthetic = assign_unique_employee_ids(sampled)

    other_columns = [c for c in clustered_employees.columns if c not in PERSON_KEY]
    column_order = [
        SYNTHETIC_ID_COLUMN,
        SOURCE_HOUSEID_COLUMN,
        SOURCE_PERSONID_COLUMN,
        *other_columns,
    ]
    return synthetic[column_order].reset_index(drop=True)


def save_synthetic_employees(
    synthetic_employees: pd.DataFrame, processed_dir: Path = DEFAULT_PROCESSED_DIR
) -> Path:
    """Write the synthetic employee table to `processed_dir / SYNTHETIC_EMPLOYEE_FILENAME`."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / SYNTHETIC_EMPLOYEE_FILENAME
    synthetic_employees.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate a synthetic employee population.")
    parser.add_argument(
        "-n",
        "--number-of-synthetic-employees",
        dest="n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of synthetic employees to generate (default: {DEFAULT_N}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: driving_profiles.utils.random_seed.DEFAULT_SEED).",
    )
    args = parser.parse_args()

    synthetic_employees = create_synthetic_employee_table(n=args.n, seed=args.seed)
    output_path = save_synthetic_employees(synthetic_employees)

    proportions = synthetic_employees["cluster_id"].value_counts(normalize=True).sort_index()
    logger.info(
        "Wrote %d synthetic employee(s) to %s (cluster proportions: %s)",
        len(synthetic_employees),
        output_path,
        proportions.round(4).to_dict(),
    )
