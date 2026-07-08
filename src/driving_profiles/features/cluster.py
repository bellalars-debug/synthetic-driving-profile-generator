"""Cluster daily travel feature vectors into behavior archetypes.

Produces the "cluster assignments" pipeline artifact, which the generator
samples from when creating synthetic employee profiles. Reads only
`data/processed/employee_features.parquet` (the `build_features.py`
output) - no raw NHTS files are read here, and `build_features.py` itself
is not modified. Synthetic employee sampling, EV penetration, and charging
demand are out of scope; see `generator/` and `scenarios/` for those
stages. Methodology: `docs/clustering_plan.md`.

## Feature selection (docs/clustering_plan.md §3)

Clustering runs only on the "primary" feature groups the plan identifies as
directly determining workplace charging opportunity - commute behavior,
daily mobility, and vehicle availability (`CONTINUOUS_FEATURES` +
`BOOLEAN_FEATURES`, together `CLUSTERING_FEATURES`). Demographic and
household columns (`age`, `age_band`, `household_income_bracket`,
`household_size`, `household_vehicle_count`, ...) are the plan's
"secondary/contextual" group: informative for *describing* a cluster after
the fact, but with no direct mechanism that changes charging opportunity,
so they are excluded from the KMeans feature matrix entirely rather than
merely down-weighted - KMeans has no native per-feature weighting, so
exclusion is the only way to guarantee they can't dominate distances.
They, and every other column in `employee_features.parquet`, are still
carried through to the final output (`save_clustered_profiles`) for
post-hoc cluster profiling. `EXCLUDED_FEATURES` documents, column by
column, why each non-selected column is left out (identifiers,
near-constant columns, or a column that would double-count a concept
already captured by another selected column).

## Population filter (docs/clustering_plan.md §2)

Clustering is restricted to workers with an observed weekday commute
(`is_worker == True` and `work_trip_count > 0`): a non-worker or a worker
who didn't travel to work that day has no workplace dwell window to
characterize, and their commute columns are structurally NaN, not a value
to cluster on. This filter is applied inside `select_clustering_features`;
rows outside it are excluded from the KMeans fit but retained (with a null
`cluster_id`) in the final saved output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from driving_profiles.utils import random_seed

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")
FEATURE_TABLE_FILENAME = "employee_features.parquet"
CLUSTER_TABLE_FILENAME = "employee_clusters.parquet"
CLUSTER_EVALUATION_FILENAME = "cluster_evaluation.csv"

PERSON_KEY = ["HOUSEID", "PERSONID"]

# --- Feature selection (docs/clustering_plan.md §3) -------------------------

# Commute behavior + daily mobility + vehicle availability: the plan's
# "primary" groups, standardized before clustering (see PREPROCESSING notes
# in preprocess_features).
CONTINUOUS_FEATURES = [
    "commute_distance_survey_miles",
    "commute_duration_minutes",
    "work_arrival_time",
    "work_departure_time",
    "trips_per_day",
    "total_daily_miles",
    "total_driving_minutes",
    "number_of_stops",
    "vehicles_per_driver",
]

# Already 0/1-valued; pass through unscaled per plan §4 ("Booleans ... pass
# through as 0/1 with no encoding step needed").
BOOLEAN_FEATURES = [
    "vehicle_per_driver_adequate",
    "used_household_vehicle",
]

CLUSTERING_FEATURES = CONTINUOUS_FEATURES + BOOLEAN_FEATURES

# "No driving trips that day" is a real zero, not missing data - impute 0
# rather than median (docs/clustering_plan.md §4).
ZERO_FILL_FEATURES = ("total_daily_miles", "total_driving_minutes")

# Every column in employee_features.parquet not in CLUSTERING_FEATURES,
# with the reason it's excluded from the clustering matrix. Columns not
# excluded here and not in CLUSTERING_FEATURES would be a bug (see the
# assertion in select_clustering_features).
EXCLUDED_FEATURES: dict[str, str] = {
    "HOUSEID": "identifier, not a behavioral feature.",
    "PERSONID": "identifier, not a behavioral feature.",
    "age": (
        "raw continuous age has a wide, largely non-behavioral range (~16-90) "
        "that would dominate Euclidean distance purely due to spread, not "
        "informativeness (plan §3)."
    ),
    "age_band": "demographic/contextual only - no direct charging-opportunity mechanism (plan §3).",
    "worker_status": (
        "constant 'worker' after the population filter (plan §2) - no separating signal."
    ),
    "is_worker": "constant True after the population filter (plan §2) - no separating signal.",
    "household_income_bracket": (
        "demographic/contextual only - no direct charging-opportunity mechanism (plan §3)."
    ),
    "household_size": (
        "demographic/contextual only - no direct charging-opportunity mechanism (plan §3)."
    ),
    "household_vehicle_count": (
        "redundant with vehicles_per_driver, which already normalizes vehicle "
        "count by driver count; the plan prefers the ratio (plan §3)."
    ),
    "work_trip_count": (
        "near-constant at 1 within the filtered population - carries almost no "
        "separating signal (plan §3)."
    ),
    "commute_distance_trip_miles": (
        "redundant with commute_distance_survey_miles (same underlying concept); "
        "the survey estimate (GCDWORK) is preferred because it's complete for "
        "every worker with a GCDWORK response, while the trip-based sum is NaN "
        "whenever WHYTRP1S didn't tag a leg as work-purpose that day (plan §3)."
    ),
    "average_trip_distance_miles": (
        "derived from total_daily_miles (already included directly); including "
        "both would double-count one underlying concept (plan §3)."
    ),
    "household_vehicle_trip_count": (
        "redundant with used_household_vehicle, which already captures the same "
        "concept as a boolean; the raw count would double-count alongside "
        "trips_per_day/number_of_stops (plan §3)."
    ),
}

DEFAULT_K_RANGE = range(2, 9)


def select_clustering_features(employee_features: pd.DataFrame) -> pd.DataFrame:
    """Filter to the clustering population and select the clustering columns.

    Applies the docs/clustering_plan.md §2 population filter (workers with
    an observed weekday commute) and drops the §4 `vehicles_per_driver`
    NaN edge case (a zero-driver household - a data anomaly, not a typical
    case to impute). Returns `PERSON_KEY + CLUSTERING_FEATURES`, unimputed
    and unscaled - see `preprocess_features` for that step.
    """
    unaccounted = (
        set(employee_features.columns) - set(CLUSTERING_FEATURES) - set(EXCLUDED_FEATURES)
    )
    assert not unaccounted, (
        f"Column(s) {sorted(unaccounted)} are neither selected for clustering nor "
        "documented in EXCLUDED_FEATURES - classify them before proceeding."
    )

    is_worker = employee_features["is_worker"].fillna(False).astype(bool)
    has_commute = employee_features["work_trip_count"] > 0
    population = employee_features.loc[is_worker & has_commute].copy()
    logger.info(
        "select_clustering_features: %d/%d employee(s) are workers with an observed "
        "commute (clustering population)",
        len(population),
        len(employee_features),
    )

    no_vehicle_ratio = population["vehicles_per_driver"].isna()
    if no_vehicle_ratio.any():
        logger.info(
            "select_clustering_features: dropping %d row(s) with an undefined "
            "vehicles_per_driver (zero-driver household)",
            int(no_vehicle_ratio.sum()),
        )
        population = population.loc[~no_vehicle_ratio]

    return population[PERSON_KEY + CLUSTERING_FEATURES].reset_index(drop=True)


def preprocess_features(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate IDs from model features and prepare the clustering matrix.

    - IDs (`PERSON_KEY`) are split out and returned separately so no
      identifier reaches the model input.
    - Missing values: `ZERO_FILL_FEATURES` (no driving trips that day) are
      filled with 0; any other remaining NaN in a continuous feature (a
      true missing/unascertained survey response, e.g. `GCDWORK` refusals)
      is median-imputed (plan §4).
    - Categorical encoding: none needed - `CLUSTERING_FEATURES` has no
      nominal/ordinal categoricals (those live in the excluded demographic
      group); `BOOLEAN_FEATURES` are cast 0/1.
    - Numerical scaling: `CONTINUOUS_FEATURES` are standardized (zero mean,
      unit variance) via `StandardScaler`, required because
      `work_arrival_time`/`work_departure_time` are HHMM-coded (range
      ~0-2400) and would otherwise dwarf features like `vehicles_per_driver`
      (plan §4).

    Returns `(ids, X)` with matching row order/length: `ids` has
    `PERSON_KEY`; `X` has `CLUSTERING_FEATURES` ready for `run_clustering`.
    """
    ids = selected[PERSON_KEY].reset_index(drop=True)
    features = selected[CLUSTERING_FEATURES].reset_index(drop=True).copy()

    features[list(ZERO_FILL_FEATURES)] = features[list(ZERO_FILL_FEATURES)].fillna(0)
    for column in CONTINUOUS_FEATURES:
        missing = features[column].isna()
        if missing.any():
            median = features[column].median()
            logger.info(
                "preprocess_features: median-imputing %d missing value(s) in %s",
                int(missing.sum()),
                column,
            )
            features[column] = features[column].fillna(median)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[CONTINUOUS_FEATURES])
    X = pd.DataFrame(scaled, columns=CONTINUOUS_FEATURES, index=features.index)
    for column in BOOLEAN_FEATURES:
        X[column] = features[column].astype(int)
    X = X[CLUSTERING_FEATURES]

    return ids, X


def determine_optimal_clusters(
    X: pd.DataFrame,
    k_range: range = DEFAULT_K_RANGE,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Evaluate candidate cluster counts via inertia (elbow) and silhouette.

    Fits a KMeans model for every `k` in `k_range` and records within-
    cluster inertia and silhouette score for each, per docs/clustering_plan.md
    §6. Returns a DataFrame (`k`, `inertia`, `silhouette_score`) for review
    - the caller (or a human reviewing `save_cluster_evaluation`'s output)
    picks the final `k`, since the plan is explicit that no single metric,
    and no metric alone, should decide `k` without a domain-interpretability
    check on the resulting clusters.
    """
    seed = random_seed.get_seed(random_state)
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette_score": score})
        logger.info(
            "determine_optimal_clusters: k=%d inertia=%.2f silhouette=%.4f",
            k,
            model.inertia_,
            score,
        )
    return pd.DataFrame(rows)


def run_clustering(
    X: pd.DataFrame, k: int, random_state: int | None = None
) -> tuple[np.ndarray, KMeans]:
    """Fit the final KMeans model with `k` clusters and return (labels, model).

    Uses `random_seed.get_seed` so identical inputs and seed produce
    identical assignments across runs (KMeans's own `n_init` restarts are
    themselves seeded off `random_state`, so this alone is sufficient for
    reproducibility - no separate seeding of the restarts is needed).
    """
    seed = random_seed.get_seed(random_state)
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(X)
    return labels, model


def save_cluster_evaluation(
    evaluation: pd.DataFrame, processed_dir: Path = DEFAULT_PROCESSED_DIR
) -> Path:
    """Write the `determine_optimal_clusters` evaluation table to CSV for review."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / CLUSTER_EVALUATION_FILENAME
    evaluation.to_csv(path, index=False)
    return path


def save_clustered_profiles(
    employee_features: pd.DataFrame,
    ids: pd.DataFrame,
    cluster_labels: np.ndarray,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Join cluster assignments back onto the full employee feature table.

    `employee_features` is the full, unfiltered table (every employee, not
    just the clustered population) so that the employee count is preserved
    end to end (docs/clustering_plan.md §2: non-worker/no-commute rows stay
    in the table for other project uses). `cluster_id` is a nullable
    integer: populated for rows in the clustering population (`ids`), and
    null for every row excluded by `select_clustering_features` (non-
    workers, workers without an observed commute, or the zero-driver
    edge case) - a person who wasn't clustered has no archetype to report,
    not archetype 0. Writes to `processed_dir / CLUSTER_TABLE_FILENAME`.
    """
    labels = ids[PERSON_KEY].copy()
    labels["cluster_id"] = pd.array(cluster_labels, dtype="Int64")

    result = employee_features.merge(labels, on=PERSON_KEY, how="left", validate="1:1")

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / CLUSTER_TABLE_FILENAME
    result.to_parquet(path, index=False)
    return path


def load_employee_features(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Read the build_features.py output (`employee_features.parquet`)."""
    path = Path(processed_dir) / FEATURE_TABLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Employee feature table not found: {path}. Run "
            "`python -m driving_profiles.features.build_features` first."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    employee_features = load_employee_features()
    selected = select_clustering_features(employee_features)
    ids, X = preprocess_features(selected)

    evaluation = determine_optimal_clusters(X)
    evaluation_path = save_cluster_evaluation(evaluation)
    logger.info("Wrote cluster evaluation table to %s", evaluation_path)

    best_k = int(evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"])
    logger.info(
        "Best silhouette score in %s is at k=%d (review %s before trusting this "
        "for production - plan §6 requires a domain-interpretability check, not "
        "just the quantitative optimum)",
        list(DEFAULT_K_RANGE),
        best_k,
        evaluation_path,
    )

    labels, model = run_clustering(X, best_k)
    output_path = save_clustered_profiles(employee_features, ids, labels)
    logger.info(
        "Wrote %d employee row(s) (%d clustered into k=%d archetypes) to %s",
        len(employee_features),
        len(ids),
        best_k,
        output_path,
    )
