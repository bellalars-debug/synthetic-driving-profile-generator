"""Tests for scripts/run_pipeline.py.

Only the orchestration logic itself is covered here (existing-output
skip/force behavior, error handling, optional export-stage detection) -
each stage's own scientific/statistical logic is already tested where it
lives (test_download.py, test_clean.py, test_build_features.py,
test_cluster.py, test_generator.py, test_activity.py).
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py"


def _load_run_pipeline():
    spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_pipeline = _load_run_pipeline()


# --- _skip_existing -----------------------------------------------------------


def test_skip_existing_true_when_present_and_not_forced(tmp_path, capsys):
    path = tmp_path / "output.parquet"
    path.touch()

    assert run_pipeline._skip_existing(path, force=False) is True
    assert "already exists" in capsys.readouterr().out


def test_skip_existing_false_when_forced(tmp_path):
    path = tmp_path / "output.parquet"
    path.touch()

    assert run_pipeline._skip_existing(path, force=True) is False


def test_skip_existing_false_when_missing(tmp_path):
    path = tmp_path / "output.parquet"

    assert run_pipeline._skip_existing(path, force=False) is False


# --- _export_excel_implemented -------------------------------------------------


def test_export_excel_implemented_by_default():
    assert run_pipeline._export_excel_implemented() is True


def test_export_excel_not_detected_when_entry_point_missing(monkeypatch):
    monkeypatch.delattr(run_pipeline.export_excel, "export_all", raising=False)
    assert run_pipeline._export_excel_implemented() is False


def test_export_excel_detected_once_entry_point_exists(monkeypatch):
    monkeypatch.setattr(
        run_pipeline.export_excel, "export_all", lambda **kwargs: None, raising=False
    )
    assert run_pipeline._export_excel_implemented() is True


# --- _run_stage -----------------------------------------------------------------


def test_run_stage_returns_function_result(capsys):
    result = run_pipeline._run_stage(1, 7, "Doing a thing...", lambda x: x + 1, 41)

    assert result == 42
    assert "[1/7] Doing a thing..." in capsys.readouterr().out


def test_run_stage_halts_pipeline_on_exception(capsys):
    def failing_stage():
        raise ValueError("bad input")

    with pytest.raises(SystemExit) as exc_info:
        run_pipeline._run_stage(3, 7, "Cleaning...", failing_stage)

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "[3/7] Cleaning..." in err
    assert "ValueError: bad input" in err


# --- stage_download -------------------------------------------------------------


def test_stage_download_skips_when_raw_files_already_present(tmp_path, monkeypatch):
    for name in run_pipeline.REQUIRED_RAW_FILES:
        (tmp_path / name).touch()

    def fail_if_called(**kwargs):
        raise AssertionError("download.fetch should not be called when raw files exist")

    monkeypatch.setattr(run_pipeline.download, "fetch", fail_if_called)

    run_pipeline.stage_download(tmp_path, force=False)


def test_stage_download_fetches_when_raw_files_missing(tmp_path, monkeypatch):
    calls = []

    class FakeManifest:
        extracted_files = ["hhv2pub.csv", "perv2pub.csv", "vehv2pub.csv", "tripv2pub.csv"]

    def fake_fetch(dest_dir):
        calls.append(dest_dir)
        return FakeManifest()

    monkeypatch.setattr(run_pipeline.download, "fetch", fake_fetch)

    run_pipeline.stage_download(tmp_path, force=False)

    assert calls == [tmp_path]


def test_stage_download_force_refetches_even_when_present(tmp_path, monkeypatch):
    for name in run_pipeline.REQUIRED_RAW_FILES:
        (tmp_path / name).touch()

    calls = []

    class FakeManifest:
        extracted_files = ["hhv2pub.csv", "perv2pub.csv", "vehv2pub.csv", "tripv2pub.csv"]

    def fake_fetch(dest_dir):
        calls.append(dest_dir)
        return FakeManifest()

    monkeypatch.setattr(run_pipeline.download, "fetch", fake_fetch)

    run_pipeline.stage_download(tmp_path, force=True)

    assert calls == [tmp_path]


# --- stage_clean (representative existing-output check) -----------------------


def test_stage_clean_skips_when_output_already_exists(tmp_path, monkeypatch):
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    (interim_dir / run_pipeline.clean.ANALYSIS_DATASET_FILENAME).touch()

    def fail_if_called(raw_dir):
        raise AssertionError("clean.create_analysis_dataset should not be called")

    monkeypatch.setattr(run_pipeline.clean, "create_analysis_dataset", fail_if_called)

    run_pipeline.stage_clean(tmp_path / "raw", interim_dir, force=False)


def test_stage_clean_runs_when_output_missing(tmp_path, monkeypatch):
    interim_dir = tmp_path / "interim"
    calls = []

    monkeypatch.setattr(
        run_pipeline.clean, "create_analysis_dataset", lambda raw_dir: calls.append(raw_dir) or []
    )
    monkeypatch.setattr(
        run_pipeline.clean,
        "save_analysis_dataset",
        lambda df, interim_dir: interim_dir / run_pipeline.clean.ANALYSIS_DATASET_FILENAME,
    )

    run_pipeline.stage_clean(tmp_path / "raw", interim_dir, force=False)

    assert calls == [tmp_path / "raw"]
