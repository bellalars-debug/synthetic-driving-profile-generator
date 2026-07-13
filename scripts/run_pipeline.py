"""Thin CLI orchestrating the full pipeline:

download -> ingest -> clean -> features -> cluster -> sample ->
generate activity profiles -> (optional) export .xlsx reports.

Each stage calls straight into the already-implemented module functions
(`driving_profiles.data.*`, `driving_profiles.features.*`,
`driving_profiles.generator.*`) - no scientific/statistical logic lives
here, only sequencing, logging, existing-output checks, and error
reporting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from driving_profiles.data import clean, download, ingest  # noqa: E402
from driving_profiles.features import build_features, cluster  # noqa: E402
from driving_profiles.generator import activity  # noqa: E402
from driving_profiles.generator import sample as sample_module  # noqa: E402
from driving_profiles.utils import export_excel  # noqa: E402

# Same four files download.py's REQUIRED_CSV_FILENAMES guarantees fetch()
# extracts and ingest.py reads by these exact names.
REQUIRED_RAW_FILES = tuple(sorted(download.REQUIRED_CSV_FILENAMES))

DEFAULT_REPORTS_DIR = Path("reports/xlsx")

# export_excel.py exposes a callable named `export_all(...)` as its pipeline
# entry point (see its module docstring); detected via hasattr rather than
# hard-wired so a caller that stubs it back out in a test still gets a
# correctly-numbered, export-skipping pipeline run.
EXPORT_EXCEL_ENTRY_POINT = "export_all"


def _stage(index: int, total: int, label: str) -> None:
    print(f"[{index}/{total}] {label}")


def _run_stage(index: int, total: int, label: str, func, *args, **kwargs):
    """Run one pipeline stage, printing its banner and turning any
    exception it raises into a clear, pipeline-halting error message.
    """
    _stage(index, total, label)
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        print(
            f"\nPipeline failed at stage [{index}/{total}] {label}\n"
            f"  {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _skip_existing(path: Path, force: bool) -> bool:
    """Return True (and print an explanatory message) if `path` already
    exists and `force` was not requested - the caller should skip
    recomputation rather than silently overwrite it.
    """
    if path.exists() and not force:
        print(f"  Output already exists at {path} - skipping (use --force to regenerate).")
        return True
    return False


def _export_excel_implemented() -> bool:
    return hasattr(export_excel, EXPORT_EXCEL_ENTRY_POINT)


# --- stage bodies ------------------------------------------------------------


def stage_download(raw_dir: Path, force: bool) -> None:
    required = [raw_dir / name for name in REQUIRED_RAW_FILES]
    if not force and all(path.exists() for path in required):
        print(f"  Raw NHTS files already present in {raw_dir} - skipping download.")
        return
    manifest = download.fetch(dest_dir=raw_dir)
    print(f"  Downloaded and extracted {len(manifest.extracted_files)} file(s) to {raw_dir}")


def stage_ingest(raw_dir: Path) -> None:
    tables = ingest.load_all(raw_dir)
    for name, table in tables.items():
        print(f"  {name}: {len(table)} row(s), {len(table.columns)} column(s)")


def stage_clean(raw_dir: Path, interim_dir: Path, force: bool) -> None:
    output_path = interim_dir / clean.ANALYSIS_DATASET_FILENAME
    if _skip_existing(output_path, force):
        return
    dataset = clean.create_analysis_dataset(raw_dir)
    path = clean.save_analysis_dataset(dataset, interim_dir)
    print(f"  Wrote {len(dataset)} cleaned trip record(s) to {path}")


def stage_build_features(interim_dir: Path, processed_dir: Path, force: bool) -> None:
    output_path = processed_dir / build_features.FEATURE_TABLE_FILENAME
    if _skip_existing(output_path, force):
        return
    cleaned = build_features.load_cleaned_trips(interim_dir)
    features = build_features.create_employee_feature_table(cleaned)
    path = build_features.save_feature_table(features, processed_dir)
    print(f"  Wrote {len(features)} employee feature row(s) to {path}")


def stage_cluster(processed_dir: Path, k: int | None, force: bool) -> None:
    output_path = processed_dir / cluster.CLUSTER_TABLE_FILENAME
    if _skip_existing(output_path, force):
        return
    employee_features = cluster.load_employee_features(processed_dir)
    selected = cluster.select_clustering_features(employee_features)
    ids, X = cluster.preprocess_features(selected)

    evaluation = cluster.determine_optimal_clusters(X)
    evaluation_path = cluster.save_cluster_evaluation(evaluation, processed_dir)
    print(f"  Wrote cluster evaluation table to {evaluation_path}")

    chosen_k = k
    if chosen_k is None:
        chosen_k = int(evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"])
        print(f"  Auto-selected k={chosen_k} by best silhouette score")
    else:
        print(f"  Using caller-specified k={chosen_k}")

    labels, _model = cluster.run_clustering(X, chosen_k)
    path = cluster.save_clustered_profiles(employee_features, ids, labels, processed_dir)
    print(f"  Wrote {len(employee_features)} employee row(s) ({len(ids)} clustered) to {path}")


def stage_sample(processed_dir: Path, n: int, seed: int | None, force: bool) -> None:
    output_path = processed_dir / sample_module.SYNTHETIC_EMPLOYEE_FILENAME
    if _skip_existing(output_path, force):
        return
    synthetic_employees = sample_module.create_synthetic_employee_table(
        n=n, seed=seed, processed_dir=processed_dir
    )
    path = sample_module.save_synthetic_employees(synthetic_employees, processed_dir)
    print(f"  Wrote {len(synthetic_employees)} synthetic employee(s) to {path}")


def stage_activity(interim_dir: Path, processed_dir: Path, seed: int | None, force: bool) -> None:
    output_path = processed_dir / activity.ACTIVITY_TABLE_FILENAME
    if _skip_existing(output_path, force):
        return
    synthetic_employees = activity.load_synthetic_employees(processed_dir)
    employee_clusters = activity.load_employee_clusters(processed_dir)
    trips_clean = activity.load_trips_clean(interim_dir)
    result = activity.generate_synthetic_activity(
        synthetic_employees, employee_clusters, trips_clean, seed=seed
    )
    path = activity.save_synthetic_activity(result, processed_dir)
    print(
        f"  Wrote {len(result)} activity leg(s) for "
        f"{result['synthetic_employee_id'].nunique()} synthetic employee(s) to {path}"
    )


def stage_export_excel(
    interim_dir: Path, processed_dir: Path, reports_dir: Path, force: bool
) -> None:
    export_all = getattr(export_excel, EXPORT_EXCEL_ENTRY_POINT)
    export_all(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        force=force,
    )


# --- orchestration -------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full synthetic driving profile pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=ingest.DEFAULT_RAW_DIR)
    parser.add_argument("--interim-dir", type=Path, default=clean.DEFAULT_INTERIM_DIR)
    parser.add_argument("--processed-dir", type=Path, default=build_features.DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "-n",
        "--num-employees",
        dest="n",
        type=int,
        default=sample_module.DEFAULT_N,
        help=f"Number of synthetic employees to generate (default: {sample_module.DEFAULT_N}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: driving_profiles.utils.random_seed.DEFAULT_SEED).",
    )
    parser.add_argument(
        "-k",
        "--num-clusters",
        dest="k",
        type=int,
        default=None,
        help="Number of clusters (default: auto-select via best silhouette score).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite every stage's output, even if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    # Keep our own print() banners interleaved correctly with the stage
    # modules' own logging output when stdout is redirected/piped (not a
    # tty), since fully-buffered stdout otherwise flushes only at exit
    # while logging's stderr handler flushes per record.
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    total = 8 if _export_excel_implemented() else 7

    _run_stage(1, total, "Downloading data...", stage_download, args.raw_dir, args.force)
    _run_stage(2, total, "Ingesting raw NHTS files...", stage_ingest, args.raw_dir)
    _run_stage(
        3, total, "Cleaning...", stage_clean, args.raw_dir, args.interim_dir, args.force
    )
    _run_stage(
        4,
        total,
        "Building employee features...",
        stage_build_features,
        args.interim_dir,
        args.processed_dir,
        args.force,
    )
    _run_stage(
        5, total, "Clustering...", stage_cluster, args.processed_dir, args.k, args.force
    )
    _run_stage(
        6,
        total,
        "Sampling synthetic employees...",
        stage_sample,
        args.processed_dir,
        args.n,
        args.seed,
        args.force,
    )
    _run_stage(
        7,
        total,
        "Generating activity profiles...",
        stage_activity,
        args.interim_dir,
        args.processed_dir,
        args.seed,
        args.force,
    )
    if total == 8:
        _run_stage(
            8,
            total,
            "Exporting .xlsx reports...",
            stage_export_excel,
            args.interim_dir,
            args.processed_dir,
            args.reports_dir,
            args.force,
        )
    else:
        print("(export_excel.py not yet implemented - skipping .xlsx export)")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
