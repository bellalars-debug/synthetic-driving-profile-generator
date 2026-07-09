"""Tests for driving_profiles.validation.report."""

import pandas as pd
import pytest

from driving_profiles.generator import activity as ac
from driving_profiles.generator import sample as sample_module
from driving_profiles.validation import common
from driving_profiles.validation import report as rpt

# --- summarize -----------------------------------------------------------------
#
# Regression coverage for a real bug found while building this report: a
# `passed` column mixing True/False/None (object dtype) made `~gating["passed"]`
# invoke Python's bitwise-not on booleans (`~True == -2`) instead of boolean
# negation, producing negative fail counts. summarize() must compute n_fail
# via subtraction, not `~`, so this must hold regardless of dtype.


def test_summarize_counts_are_never_negative_with_mixed_object_dtype():
    mixed = common.results_frame(
        [
            common.result_row("s", "m1", passed=True),
            common.result_row("s", "m2", passed=False),
            common.result_row("s", "m3", passed=None),
        ]
    )
    assert mixed["passed"].dtype == object  # the exact condition that triggered the bug

    summary = rpt.summarize({"section": mixed})

    row = summary.iloc[0]
    assert row["checks_passed"] == 1
    assert row["checks_failed"] == 1
    assert row["checks_informational"] == 1
    assert row["checks_failed"] >= 0


def test_summarize_matches_manual_counts_for_all_pass_bool_dtype():
    # A section with no informational rows can end up with a pure-bool
    # dtype column - summarize() must handle that case too.
    all_pass = common.results_frame(
        [common.result_row("s", "m1", passed=True), common.result_row("s", "m2", passed=True)]
    )

    summary = rpt.summarize({"section": all_pass})

    row = summary.iloc[0]
    assert row["checks_passed"] == 2
    assert row["checks_failed"] == 0


def test_summarize_totals_sum_to_total_checks():
    df = common.results_frame(
        [
            common.result_row("s", "m1", passed=True),
            common.result_row("s", "m2", passed=False),
            common.result_row("s", "m3", passed=False),
            common.result_row("s", "m4", passed=None),
        ]
    )

    summary = rpt.summarize({"section": df})

    row = summary.iloc[0]
    counted = row["checks_passed"] + row["checks_failed"] + row["checks_informational"]
    assert counted == row["total_checks"]
    assert row["total_checks"] == 4


# --- render_markdown -----------------------------------------------------------


def test_render_markdown_includes_all_required_sections():
    results = {
        "population": common.results_frame([common.result_row("population", "m1", passed=True)]),
        "clusters": common.results_frame([common.result_row("clusters", "m2", passed=False)]),
        "activity": common.results_frame([common.result_row("activity", "m3", passed=True)]),
        "missingness": common.results_frame([common.result_row("missingness", "m4", passed=None)]),
    }

    text = rpt.render_markdown(results, diagnostics={})

    assert "# Synthetic Population Validation Results" in text
    assert "## 1. Summary" in text
    assert "## 2. Metrics calculated and pass/fail interpretation" in text
    assert "## 6. Findings" in text
    assert "## 7. Issues requiring investigation" in text


def test_render_markdown_lists_failed_checks_in_open_issues():
    results = {
        "population": common.results_frame(
            [
                common.result_row(
                    "population", "bad_metric", group="cluster_0", passed=False, detail="oops"
                )
            ]
        ),
        "clusters": common.results_frame([]),
        "activity": common.results_frame([]),
        "missingness": common.results_frame([]),
    }

    text = rpt.render_markdown(results, diagnostics={})

    assert "bad_metric" in text
    assert "oops" in text


# --- save_report -----------------------------------------------------------------


def test_save_report_writes_file(tmp_path):
    path = rpt.save_report("# hello", tmp_path / "sub" / "out.md")

    assert path.exists()
    assert path.read_text() == "# hello"


# --- main (integration) -----------------------------------------------------------


def _trip_row(
    house_id, person_id, trip_id, strttime, endtime, trvlcmin, trpmiles, whytrp1s, loop_trip=2
):
    return {
        "HOUSEID": house_id, "PERSONID": person_id, "TRIPID": trip_id,
        "LOOP_TRIP": loop_trip, "STRTTIME": strttime, "ENDTIME": endtime,
        "TRVLCMIN": trvlcmin, "TRPMILES": trpmiles, "WHYTRP1S": whytrp1s,
        "VEHTYPE": 1.0, "VEHFUEL": 1.0,
    }


@pytest.fixture
def small_pipeline(tmp_path):
    processed_dir = tmp_path / "processed"
    interim_dir = tmp_path / "interim"
    processed_dir.mkdir()
    interim_dir.mkdir()

    trips = pd.DataFrame(
        [
            _trip_row("D1", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
            _trip_row("D1", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
            _trip_row("D2", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
            _trip_row("D2", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
        ]
    )
    for col in ("HOUSEID", "PERSONID", "TRIPID"):
        trips[col] = trips[col].astype(str)
    trips.to_parquet(interim_dir / ac.clean.ANALYSIS_DATASET_FILENAME, index=False)

    employee_clusters = pd.DataFrame(
        {
            "HOUSEID": ["D1", "D2"],
            "PERSONID": ["01", "01"],
            "cluster_id": pd.array([0, 0], dtype="Int64"),
            "age": [30, 40],
            "age_band": ["25-34", "35-44"],
            "worker_status": ["worker", "worker"],
            "is_worker": pd.array([True, True], dtype="boolean"),
            "household_income_bracket": [5.0, 6.0],
            "household_size": [2, 3],
            "household_vehicle_count": [2, 2],
            "vehicles_per_driver": [1.0, 1.0],
            "vehicle_per_driver_adequate": [True, True],
            "used_household_vehicle": [True, True],
            "commute_distance_survey_miles": [10.0, 10.0],
            "commute_duration_minutes": [30.0, 30.0],
            "work_arrival_time": [830.0, 830.0],
            "work_departure_time": [1700.0, 1700.0],
            "trips_per_day": [2, 2],
            "number_of_stops": [1, 1],
            "total_daily_miles": [20.0, 20.0],
            "total_driving_minutes": [60.0, 60.0],
            "average_trip_distance_miles": [10.0, 10.0],
        }
    )
    employee_clusters.to_parquet(processed_dir / sample_module.CLUSTER_TABLE_FILENAME, index=False)

    synthetic_employees = employee_clusters.rename(
        columns={"HOUSEID": "source_houseid", "PERSONID": "source_personid"}
    ).copy()
    synthetic_employees["synthetic_employee_id"] = ["SYN-001", "SYN-002"]
    synthetic_employees.to_parquet(
        processed_dir / sample_module.SYNTHETIC_EMPLOYEE_FILENAME, index=False
    )

    synthetic_activity = ac.generate_synthetic_activity(
        synthetic_employees, employee_clusters, trips, seed=0
    )
    synthetic_activity.to_parquet(processed_dir / ac.ACTIVITY_TABLE_FILENAME, index=False)

    return processed_dir, interim_dir


def test_main_produces_a_readable_report_file(small_pipeline):
    processed_dir, interim_dir = small_pipeline
    report_path = processed_dir / "validation_results.md"

    result_path = rpt.main(
        processed_dir=processed_dir, interim_dir=interim_dir, report_path=report_path
    )

    assert result_path == report_path
    text = report_path.read_text()
    assert "# Synthetic Population Validation Results" in text
    assert "## 6. Findings" in text
