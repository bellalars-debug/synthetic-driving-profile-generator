"""CLI entry point for the profile-based mobility-generation experiment
(`docs/profile_based_generation_plan.md` §8).

Reads `DriverProfiles.csv` (external, read-only) plus this pipeline's own
already-validated `data/interim/trips_clean.parquet` /
`data/processed/employee_clusters.parquet` (read-only), runs the §8
reconciliation, and writes results only under `data/validation/profile_based/`.
Never touches `data/processed/synthetic_employees.parquet` or
`data/processed/synthetic_activity.parquet` - verified explicitly via a
before/after hash check (see `_hash_file`) as an extra runtime guard on top
of the dedicated pytest byte-diff test.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from driving_profiles.generator import activity as activity_module  # noqa: E402
from driving_profiles.generator import profile_adapter  # noqa: E402
from driving_profiles.generator import profile_based  # noqa: E402
from driving_profiles.validation import profile_based as profile_based_validation  # noqa: E402

PRODUCTION_FILES = (
    Path("data/processed/synthetic_employees.parquet"),
    Path("data/processed/synthetic_activity.parquet"),
)


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the profile-based mobility-generation experiment (plan §8)."
    )
    parser.add_argument(
        "--n-employees",
        type=int,
        default=250,
        help="Number of DriverProfiles.csv users to process (default: 250, the full file).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for donor matching.")
    parser.add_argument(
        "--driver-profiles-path",
        type=Path,
        default=profile_adapter.DEFAULT_DRIVER_PROFILES_PATH,
    )
    parser.add_argument("--interim-dir", type=Path, default=activity_module.DEFAULT_INTERIM_DIR)
    parser.add_argument("--processed-dir", type=Path, default=activity_module.DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--validation-dir", type=Path, default=profile_based.DEFAULT_VALIDATION_DIR
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite the output/report even if they already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    pre_hashes = {p: _hash_file(p) for p in PRODUCTION_FILES}

    output_path = args.validation_dir / profile_based.OUTPUT_FILENAME
    if output_path.exists() and not args.force:
        print(f"Output already exists at {output_path} - skipping (use --force to regenerate).")
        return

    print(f"[1/5] Loading DriverProfiles.csv from {args.driver_profiles_path}...")
    driver_profiles = profile_adapter.load_driver_profiles(args.driver_profiles_path)
    user_ids = profile_adapter.select_profile_user_ids(driver_profiles, args.n_employees, args.seed)
    print(f"  Selected {len(user_ids)} user(s) from {driver_profiles['user_id'].nunique()} available.")

    print("[2/5] Building donor-leg pool from trips_clean.parquet + employee_clusters.parquet...")
    trips_clean = activity_module.load_trips_clean(args.interim_dir)
    employee_clusters = activity_module.load_employee_clusters(args.processed_dir)
    donor_pool = profile_adapter.build_donor_leg_pool(trips_clean, employee_clusters)
    print(f"  Donor pool: {len(donor_pool)} plausible-speed driving leg(s).")

    print("[3/5] Running §8 reconciliation (annotate, match, reconcile distance/duration/schedule)...")
    output = profile_based.run_profile_based_reconciliation(
        driver_profiles, donor_pool, user_ids, seed=args.seed
    )
    output_written_path = profile_based.save_profile_based_output(output, args.validation_dir)
    n_driving = int((output["state"] == profile_adapter.STATE_DRIVING).sum())
    print(
        f"  Wrote {len(output)} row(s) ({n_driving} driving leg(s)) for "
        f"{output['profile_employee_id'].nunique()} employee(s) to {output_written_path}"
    )

    print("[4/5] Running §8.8 validation...")
    results = profile_based_validation.run_profile_based_validation(driver_profiles, output, user_ids)
    report_path = profile_based_validation.save_validation_report(results, args.validation_dir)
    print(f"  Wrote validation report to {report_path}")

    print("[5/5] Verifying production files were not touched...")
    post_hashes = {p: _hash_file(p) for p in PRODUCTION_FILES}
    all_unchanged = True
    for p in PRODUCTION_FILES:
        unchanged = pre_hashes[p] == post_hashes[p]
        all_unchanged = all_unchanged and unchanged
        status = "unchanged" if unchanged else "CHANGED (!)"
        print(f"  {p}: {status}")

    print("\n--- Summary ---")
    print(f"Employees processed: {output['profile_employee_id'].nunique()}")
    print(f"Activity-leg (Driving) rows: {n_driving}")
    print(f"Total output rows: {len(output)}")

    for _, row in results.iterrows():
        print(f"  [{row['metric']}] {row['detail'] or row['statistic']}")

    print(f"\nProduction files unchanged: {all_unchanged}")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
