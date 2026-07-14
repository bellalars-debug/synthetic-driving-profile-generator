"""Tests for scripts/run_profile_based_test.py.

Mirrors test_run_pipeline.py's approach: load the script as a module and
test its orchestration logic (hashing, skip/force behavior, argument
parsing) plus one full end-to-end run against small tmp-path fixtures
standing in for DriverProfiles.csv / trips_clean.parquet /
employee_clusters.parquet.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_profile_based_test.py"
SRC_PATH = Path(__file__).resolve().parent.parent / "src"


def _load_script():
    sys.path.insert(0, str(SRC_PATH))
    spec = importlib.util.spec_from_file_location("run_profile_based_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


# --- _hash_file -------------------------------------------------------------------


def test_hash_file_missing_returns_none(tmp_path):
    assert script._hash_file(tmp_path / "does_not_exist.parquet") is None


def test_hash_file_stable_for_same_content(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello")
    h1 = script._hash_file(path)
    h2 = script._hash_file(path)
    assert h1 == h2
    assert h1 is not None


def test_hash_file_changes_with_content(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello")
    h1 = script._hash_file(path)
    path.write_text("goodbye")
    h2 = script._hash_file(path)
    assert h1 != h2


# --- parse_args ---------------------------------------------------------------------


def test_parse_args_defaults():
    args = script.parse_args([])
    assert args.n_employees == 250
    assert args.seed is None
    assert args.force is False


def test_parse_args_overrides():
    args = script.parse_args(["--n-employees", "10", "--seed", "42", "--force"])
    assert args.n_employees == 10
    assert args.seed == 42
    assert args.force is True


# --- end-to-end run against tmp-path fixtures ---------------------------------------


def _write_driver_profiles_csv(path: Path) -> None:
    path.write_text(
        "User ID,State,Start time (hour),End time (hour),Distance (mi),"
        "Nothing,P_max (W),Location,NHTS HH Wt\n"
        "1,Parked,0.0,8.0,-1.0,0,0,Home,100\n"
        "1,Driving,8.0,8.5,15.0,0,0,-1,100\n"
        "1,Parked,8.5,17.0,-1.0,0,0,Work,100\n"
        "1,Driving,17.0,17.5,15.0,0,0,-1,100\n"
        "1,Parked,17.5,24.0,-1.0,0,0,Home,100\n"
    )


def _write_trips_and_clusters(interim_dir: Path, processed_dir: Path) -> None:
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    trips = pd.DataFrame(
        [
            {
                "HOUSEID": "D1",
                "PERSONID": "01",
                "TRIPID": "01",
                "LOOP_TRIP": 2,
                "STRTTIME": 800,
                "ENDTIME": 830,
                "TRVLCMIN": 24.0,
                "TRPMILES": 12.0,
                "WHYTRP1S": 10,
                "TRPTRANS": 3,
                "VEHTYPE": 1.0,
                "VEHFUEL": 1.0,
            },
            {
                "HOUSEID": "D1",
                "PERSONID": "01",
                "TRIPID": "02",
                "LOOP_TRIP": 2,
                "STRTTIME": 1700,
                "ENDTIME": 1730,
                "TRVLCMIN": 24.0,
                "TRPMILES": 12.0,
                "WHYTRP1S": 1,
                "TRPTRANS": 3,
                "VEHTYPE": 1.0,
                "VEHFUEL": 1.0,
            },
        ]
    )
    trips.to_parquet(interim_dir / "trips_clean.parquet", index=False)

    clusters = pd.DataFrame(
        [{"HOUSEID": "D1", "PERSONID": "01", "cluster_id": pd.array([0], dtype="Int64")[0]}]
    )
    clusters.to_parquet(processed_dir / "employee_clusters.parquet", index=False)


def test_main_end_to_end_writes_output_and_report(tmp_path, capsys):
    driver_profiles_path = tmp_path / "DriverProfiles.csv"
    _write_driver_profiles_csv(driver_profiles_path)
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    validation_dir = tmp_path / "validation"
    _write_trips_and_clusters(interim_dir, processed_dir)

    script.main(
        [
            "--n-employees",
            "1",
            "--seed",
            "42",
            "--driver-profiles-path",
            str(driver_profiles_path),
            "--interim-dir",
            str(interim_dir),
            "--processed-dir",
            str(processed_dir),
            "--validation-dir",
            str(validation_dir),
            "--force",
        ]
    )

    from driving_profiles.generator import profile_based

    assert (validation_dir / profile_based.OUTPUT_FILENAME).exists()
    assert (validation_dir / "profile_based_validation_report.csv").exists()
    out = capsys.readouterr().out
    assert "Experiment complete." in out


def test_main_skips_when_output_exists_without_force(tmp_path, capsys):
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir(parents=True)
    from driving_profiles.generator import profile_based

    (validation_dir / profile_based.OUTPUT_FILENAME).touch()

    script.main(
        [
            "--driver-profiles-path",
            str(tmp_path / "does_not_matter.csv"),
            "--validation-dir",
            str(validation_dir),
        ]
    )
    out = capsys.readouterr().out
    assert "already exists" in out
