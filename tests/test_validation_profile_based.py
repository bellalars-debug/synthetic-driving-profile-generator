"""Tests for driving_profiles.validation.profile_based (plan §8.8)."""

from pathlib import Path

import pandas as pd
import pytest

from driving_profiles.generator import profile_adapter as pa
from driving_profiles.generator import profile_based as pb
from driving_profiles.validation import profile_based as pbv


def _write_driver_profiles_csv(tmp_path) -> Path:
    path = tmp_path / "DriverProfiles.csv"
    path.write_text(
        "User ID,State,Start time (hour),End time (hour),Distance (mi),"
        "Nothing,P_max (W),Location,NHTS HH Wt\n"
        "1,Parked,0.0,8.0,-1.0,0,0,Home,100\n"
        "1,Driving,8.0,8.5,15.0,0,0,-1,100\n"
        "1,Parked,8.5,17.0,-1.0,0,0,Work,100\n"
        "1,Driving,17.0,17.5,15.0,0,0,-1,100\n"
        "1,Parked,17.5,24.0,-1.0,0,0,Home,100\n"
        "2,Parked,0.0,9.0,-1.0,0,0,Home,100\n"
        "2,Driving,9.0,9.3,10.0,0,0,-1,100\n"
        "2,Parked,9.3,18.0,-1.0,0,0,Work,100\n"
        "2,Driving,18.0,18.3,10.0,0,0,-1,100\n"
        "2,Parked,18.3,24.0,-1.0,0,0,Home,100\n"
    )
    return path


def _small_donor_pool() -> pd.DataFrame:
    def leg(house, transition, segment, ctype, dest, start_min, miles, minutes):
        return {
            "HOUSEID": house,
            "PERSONID": "01",
            "TRIPID": "01",
            "purpose_transition": transition,
            "chain_segment": segment,
            "chain_type": ctype,
            "destination_purpose": dest,
            "start_min": start_min,
            "TRPMILES": miles,
            "TRVLCMIN": minutes,
        }

    return pd.DataFrame(
        [
            leg("D1", "home->work", "commute_out", "direct", "work", 480, 12.0, 24.0),
            leg("D2", "work->home", "commute_return", "direct", "home", 1020, 12.0, 24.0),
        ]
    )


@pytest.fixture
def fixture(tmp_path):
    driver_profiles = pa.load_driver_profiles(_write_driver_profiles_csv(tmp_path))
    pool = _small_donor_pool()
    user_ids = [1, 2]
    output = pb.run_profile_based_reconciliation(driver_profiles, pool, user_ids, seed=42)
    return driver_profiles, output, user_ids


# --- validate_sequence_preserved --------------------------------------------------


def test_validate_sequence_preserved_passes_on_untouched_output(fixture):
    driver_profiles, output, user_ids = fixture
    result = pbv.validate_sequence_preserved(driver_profiles, output, user_ids)
    row = result.iloc[0]
    assert bool(row["passed"]) is True
    assert row["statistic"] == 0.0


def test_validate_sequence_preserved_detects_tampered_location(fixture):
    driver_profiles, output, user_ids = fixture
    tampered = output.copy()
    idx = tampered.loc[tampered["profile_employee_id"] == "PROF-0001"].index[0]
    tampered.loc[idx, "location"] = "Shopping/Errands"
    result = pbv.validate_sequence_preserved(driver_profiles, tampered, user_ids)
    assert bool(result.iloc[0]["passed"]) is False
    assert result.iloc[0]["statistic"] == 1.0


# --- validate_workplace_arrival_preserved / departure ------------------------------


def test_validate_workplace_arrival_preserved_100pct_when_unchanged(fixture):
    driver_profiles, output, user_ids = fixture
    result = pbv.validate_workplace_arrival_preserved(driver_profiles, output, user_ids, (5,))
    row = result.iloc[0]
    assert row["n_synthetic"] == 2  # one arrival leg per user
    assert row["statistic"] == pytest.approx(100.0)


def test_validate_workplace_departure_preserved_100pct_when_unchanged(fixture):
    driver_profiles, output, user_ids = fixture
    result = pbv.validate_workplace_departure_preserved(driver_profiles, output, user_ids, (5,))
    row = result.iloc[0]
    assert row["n_synthetic"] == 2
    assert row["statistic"] == pytest.approx(100.0)


def test_validate_workplace_arrival_preserved_detects_large_shift(fixture):
    driver_profiles, output, user_ids = fixture
    tampered = output.copy()
    is_prof_1 = tampered["profile_employee_id"] == "PROF-0001"
    mask = is_prof_1 & tampered["is_arrival_at_work"].astype(bool)
    tampered.loc[mask, "end_hour"] = tampered.loc[mask, "end_hour"] + 1.0  # shift by 60 min
    result = pbv.validate_workplace_arrival_preserved(driver_profiles, tampered, user_ids, (5, 30))
    within_5 = result.loc[result["metric"] == "workplace_arrival_preserved_within_5min"].iloc[0]
    within_30 = result.loc[result["metric"] == "workplace_arrival_preserved_within_30min"].iloc[0]
    assert within_5["statistic"] == pytest.approx(50.0)  # 1 of 2 users now outside 5 min
    assert within_30["statistic"] == pytest.approx(50.0)  # still outside 30 min (60 min shift)


# --- validate_schedule_adjustment --------------------------------------------------


def test_validate_schedule_adjustment_reports_mean_and_max(fixture):
    _, output, _ = fixture
    result = pbv.validate_schedule_adjustment(output)
    mean_row = result.loc[result["metric"] == "mean_schedule_adjustment_minutes"].iloc[0]
    max_row = result.loc[result["metric"] == "max_schedule_adjustment_minutes"].iloc[0]
    driving = output.loc[output["state"] == "Driving"]
    expected_mean = driving["adjustment_minutes"].abs().mean()
    expected_max = driving["adjustment_minutes"].abs().max()
    assert mean_row["statistic"] == pytest.approx(expected_mean)
    assert max_row["statistic"] == pytest.approx(expected_max)


# --- validate_chronological_validity -----------------------------------------------


def test_validate_chronological_validity_passes(fixture):
    _, output, user_ids = fixture
    result = pbv.validate_chronological_validity(output, user_ids)
    assert bool(result.iloc[0]["passed"]) is True


def test_validate_chronological_validity_detects_gap(fixture):
    _, output, user_ids = fixture
    tampered = output.copy()
    prof_1_rows = tampered.loc[tampered["profile_employee_id"] == "PROF-0001"]
    idx = prof_1_rows.sort_values("row_index").index[-1]
    tampered.loc[idx, "end_hour"] = 23.0  # no longer spans to 24h
    result = pbv.validate_chronological_validity(tampered, user_ids)
    assert bool(result.iloc[0]["passed"]) is False


# --- validate_speed_plausibility ---------------------------------------------------


def test_validate_speed_plausibility_passes_on_generated_output(fixture):
    _, output, _ = fixture
    result = pbv.validate_speed_plausibility(output)
    assert bool(result.iloc[0]["passed"]) is True


def test_validate_speed_plausibility_detects_violation():
    output = pd.DataFrame(
        {
            "state": ["Driving"],
            "distance_mi": [1000.0],
            "duration_min": [1.0],  # 60,000 mph
            "distance_duration_source": [pb.DISTANCE_DURATION_SOURCE_DONOR],
        }
    )
    result = pbv.validate_speed_plausibility(output)
    assert bool(result.iloc[0]["passed"]) is False
    assert result.iloc[0]["statistic"] == 1.0


def test_validate_speed_plausibility_ignores_unrepaired_legs():
    output = pd.DataFrame(
        {
            "state": ["Driving"],
            "distance_mi": [1000.0],
            "duration_min": [1.0],
            "distance_duration_source": [pb.DISTANCE_DURATION_SOURCE_UNREPAIRED],
        }
    )
    result = pbv.validate_speed_plausibility(output)
    assert result.iloc[0]["n_synthetic"] == 0
    assert bool(result.iloc[0]["passed"]) is True


# --- validate_no_source_user_ids ---------------------------------------------------


def test_validate_no_source_user_ids_passes_on_generated_output(fixture):
    _, output, user_ids = fixture
    result = pbv.validate_no_source_user_ids(output, user_ids)
    assert bool(result.iloc[0]["passed"]) is True


def test_validate_no_source_user_ids_detects_leak(fixture):
    _, output, user_ids = fixture
    tampered = output.copy()
    tampered["profile_employee_id"] = tampered["profile_employee_id"].replace({"PROF-0001": "1"})
    result = pbv.validate_no_source_user_ids(tampered, user_ids)
    assert bool(result.iloc[0]["passed"]) is False


# --- run_profile_based_validation ---------------------------------------------------


def test_run_profile_based_validation_returns_all_sections(fixture):
    driver_profiles, output, user_ids = fixture
    result = pbv.run_profile_based_validation(driver_profiles, output, user_ids)
    assert set(result["metric"]) >= {
        "location_sequence_preserved",
        "mean_schedule_adjustment_minutes",
        "max_schedule_adjustment_minutes",
        "chronological_validity",
        "implied_speed_plausible",
        "no_source_user_ids_in_output",
    }


def test_save_validation_report_writes_csv(tmp_path, fixture):
    driver_profiles, output, user_ids = fixture
    result = pbv.run_profile_based_validation(driver_profiles, output, user_ids)
    path = pbv.save_validation_report(result, tmp_path)
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(result)
