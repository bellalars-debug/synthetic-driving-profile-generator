"""Tests for driving_profiles.generator.profile_based (plan §8.5/§8.6/§8.7)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driving_profiles.generator import activity as activity_module
from driving_profiles.generator import profile_adapter as pa
from driving_profiles.generator import profile_based as pb

# --- reconcile_distance_duration (§8.5) ------------------------------------------


def test_reconcile_distance_duration_plausible_speed_recovers_donor_duration():
    donor_leg = pd.Series({"TRPMILES": 10.0, "TRVLCMIN": 20.0})  # 30 mph
    distance, duration = pb.reconcile_distance_duration(donor_leg)
    assert distance == 10.0
    assert duration == pytest.approx(20.0)


def test_reconcile_distance_duration_implausible_speed_falls_back_to_assumed_average():
    donor_leg = pd.Series({"TRPMILES": 100.0, "TRVLCMIN": 1.0})  # 6000 mph, implausible
    distance, duration = pb.reconcile_distance_duration(donor_leg)
    assert distance == 100.0
    assert duration == pytest.approx(100.0 / activity_module.ASSUMED_AVERAGE_SPEED_MPH * 60.0)


def test_reconcile_distance_duration_zero_donor_minutes_falls_back():
    donor_leg = pd.Series({"TRPMILES": 5.0, "TRVLCMIN": 0.0})
    distance, duration = pb.reconcile_distance_duration(donor_leg)
    assert distance == 5.0
    assert duration == pytest.approx(5.0 / activity_module.ASSUMED_AVERAGE_SPEED_MPH * 60.0)


# --- _ripple_forward / _ripple_backward (§8.6 cascade) ---------------------------


def test_ripple_forward_no_cascade_when_adjacent_window_has_room():
    boundary = np.array([0, 50, 90, 200, 1440], dtype=float)  # plenty of room after the leg
    protected = np.array([True, False, False, False, True])
    events = []
    pb._ripple_forward(boundary, protected, 1.0, 1, 40.0, events)
    # far boundary of the adjacent window (index 2) and everything beyond it
    # untouched - the adjacent window alone absorbs the change.
    assert boundary.tolist() == [0, 50, 90, 200, 1440]
    assert events == []


def test_ripple_forward_cascade_clamps_to_floor_and_preserves_intervening_duration():
    # Adjacent window (boundary[1..2], originally 1 min) is too tight to
    # absorb; the ripple must pass through it, and the window after it
    # (boundary[2..3], originally 3 min) absorbs the residual since its far
    # side (boundary[3]) is protected.
    boundary = np.array([0, 50, 70, 71, 74, 1440], dtype=float)
    protected = np.array([True, False, False, False, True, True])
    boundary[2] = 90.0  # leg's own end moved from 70 to 90 (delta=+20)
    events = []
    pb._ripple_forward(boundary, protected, 1.0, 2, 20.0, events)

    durations = np.diff(boundary)
    assert (durations >= 1.0 - 1e-9).all()
    assert events == [(4, pytest.approx(18.0))]
    # boundary[3] preserved its original 1-min span via pure translation
    assert boundary[3] - boundary[2] == pytest.approx(1.0)


def test_ripple_backward_mirrors_ripple_forward():
    boundary = np.array([0, 1440 - 74, 1440 - 71, 1440 - 70, 1440 - 50, 1440], dtype=float)
    protected = np.array([True, True, False, False, False, True])
    boundary[3] -= 20.0  # leg's own start moved earlier by 20
    events = []
    pb._ripple_backward(boundary, protected, 1.0, 3, -20.0, events)
    durations = np.diff(boundary)
    assert (durations >= 1.0 - 1e-9).all()
    assert len(events) == 1
    assert events[0][0] == 1  # the protected boundary that had to move


# --- reconcile_user_schedule (§8.6/§8.7) -----------------------------------------


def _timeline(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"state": s, "start_min": a, "end_min": b} for s, a, b in rows]
    )


def _leg_row(row_index, is_arrival_at_work, origin_purpose):
    return {
        "row_index": row_index,
        "is_arrival_at_work": is_arrival_at_work,
        "origin_purpose": origin_purpose,
        "chain_segment": "commute_out",
        "chain_type": "direct",
        "purpose_transition": "home->work" if is_arrival_at_work else "work->home",
    }


def _donor(trpmiles, trvlcmin):
    return pd.Series({"TRPMILES": trpmiles, "TRVLCMIN": trvlcmin})


def test_reconcile_user_schedule_arrival_anchored_preserves_end_time():
    timeline = _timeline([("Parked", 0, 480), ("Driving", 480, 500), ("Parked", 500, 1440)])
    legs = pd.DataFrame([_leg_row(1, is_arrival_at_work=True, origin_purpose="home")])
    donor_matches = {1: (_donor(10.0, 30.0), "1a")}  # 20 mph -> 30 min duration (was 20)

    boundary, audit = pb.reconcile_user_schedule(timeline, legs, donor_matches)

    assert boundary[2] == 500.0  # arrival (end) time exactly preserved
    assert boundary[1] == pytest.approx(470.0)  # start moved earlier to absorb +10 min
    assert audit[1]["schedule_status"] == pb.SCHEDULE_STATUS_ADJUSTED
    assert audit[1]["adjustment_minutes"] == pytest.approx(-10.0)
    assert audit[1]["anchor_shifted"] is False


def test_reconcile_user_schedule_start_anchored_preserves_start_time():
    timeline = _timeline([("Parked", 0, 480), ("Driving", 480, 500), ("Parked", 500, 1440)])
    legs = pd.DataFrame([_leg_row(1, is_arrival_at_work=False, origin_purpose="work")])
    donor_matches = {1: (_donor(10.0, 30.0), "1a")}

    boundary, audit = pb.reconcile_user_schedule(timeline, legs, donor_matches)

    assert boundary[1] == 480.0  # departure (start) time exactly preserved
    assert boundary[2] == pytest.approx(510.0)  # end moved later
    assert audit[1]["adjustment_minutes"] == pytest.approx(10.0)


def test_reconcile_user_schedule_no_change_when_donor_duration_matches_original():
    timeline = _timeline([("Parked", 0, 480), ("Driving", 480, 500), ("Parked", 500, 1440)])
    legs = pd.DataFrame([_leg_row(1, is_arrival_at_work=True, origin_purpose="home")])
    donor_matches = {1: (_donor(10.0, 20.0), "1a")}  # 30 mph -> 20 min, same as original

    boundary, audit = pb.reconcile_user_schedule(timeline, legs, donor_matches)
    assert audit[1]["schedule_status"] == pb.SCHEDULE_STATUS_PRESERVED
    assert audit[1]["adjustment_minutes"] == 0.0


def test_reconcile_user_schedule_unrepaired_leg_keeps_original_schedule():
    timeline = _timeline([("Parked", 0, 480), ("Driving", 480, 500), ("Parked", 500, 1440)])
    legs = pd.DataFrame([_leg_row(1, is_arrival_at_work=True, origin_purpose="home")])
    donor_matches = {1: (None, pa.MATCH_TIER_UNREPAIRED)}

    boundary, audit = pb.reconcile_user_schedule(timeline, legs, donor_matches)
    assert boundary[1] == 480.0
    assert boundary[2] == 500.0
    assert audit[1]["schedule_status"] == pb.SCHEDULE_STATUS_PRESERVED
    assert audit[1]["distance_duration_source"] == pb.DISTANCE_DURATION_SOURCE_UNREPAIRED
    assert audit[1]["new_distance_mi"] is None


def test_reconcile_user_schedule_cascade_never_violates_floor_and_flags_anchor_shift():
    # Matches the hand-derived scenario in test_ripple_forward_cascade_*:
    # a large duration increase on a non-work-anchored leg (row 1) must
    # ripple through a too-tight window (row 2, 1 min) and finally shift the
    # protected workplace-arrival boundary that terminates row 3.
    timeline = _timeline(
        [
            ("Parked", 0, 50),
            ("Driving", 50, 70),
            ("Parked", 70, 71),
            ("Driving", 71, 74),
            ("Parked", 74, 1440),
        ]
    )
    legs = pd.DataFrame(
        [
            _leg_row(1, is_arrival_at_work=False, origin_purpose="other"),
            _leg_row(3, is_arrival_at_work=True, origin_purpose="other"),
        ]
    )
    donor_matches = {
        1: (_donor(30.0, 60.0), "1a"),  # 30 mph, 60 min (was 20 min) -> +40 min
        3: (None, pa.MATCH_TIER_UNREPAIRED),  # leave leg 2 alone
    }

    boundary, audit = pb.reconcile_user_schedule(timeline, legs, donor_matches)

    durations = np.diff(boundary)
    floor = pb.MIN_DWELL_FLOOR_MINUTES
    assert (durations >= floor - 1e-9).all(), "MIN_DWELL_FLOOR_MINUTES violated"
    assert audit[1]["anchor_shifted"] is True
    assert audit[1]["anchor_shift_minutes"] == pytest.approx(38.0)
    # the workplace-arrival boundary (end of leg row 3) did move, and it's
    # exactly the shift the audit record claims - not silently different.
    assert boundary[4] - 74.0 == pytest.approx(audit[1]["anchor_shift_minutes"])


# --- run_profile_based_reconciliation (integration) ------------------------------


def _write_driver_profiles_csv(tmp_path) -> Path:
    path = tmp_path / "DriverProfiles.csv"
    path.write_text(
        "User ID,State,Start time (hour),End time (hour),Distance (mi),"
        "Nothing,P_max (W),Location,NHTS HH Wt\n"
        "1,Parked,0.0,8.0,-1.0,0,0,Home,100\n"
        "1,Driving,8.0,8.5,-99999.0,0,0,-1,100\n"
        "1,Parked,8.5,17.0,-1.0,0,0,Work,100\n"
        "1,Driving,17.0,17.5,-99999.0,0,0,-1,100\n"
        "1,Parked,17.5,24.0,-1.0,0,0,Home,100\n"
    )
    return path


def _small_donor_pool() -> pd.DataFrame:
    def leg(house, trip, transition, segment, ctype, dest, start_min, miles, minutes):
        return {
            "HOUSEID": house,
            "PERSONID": "01",
            "TRIPID": trip,
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
            leg("D1", "01", "home->work", "commute_out", "direct", "work", 480, 12.0, 24.0),
            leg("D2", "01", "work->home", "commute_return", "direct", "home", 1020, 12.0, 24.0),
        ]
    )


def test_run_profile_based_reconciliation_never_reuses_external_distance_mi(tmp_path):
    # Guard test (plan §8.9 required test a): the external Distance (mi)
    # column is set to an impossible sentinel (-99999) for both driving
    # legs; since both legs get a real donor match here, that sentinel must
    # never appear in the output.
    driver_profiles = pa.load_driver_profiles(_write_driver_profiles_csv(tmp_path))
    pool = _small_donor_pool()
    output = pb.run_profile_based_reconciliation(driver_profiles, pool, [1], seed=42)

    driving = output.loc[output["state"] == "Driving"]
    assert len(driving) == 2
    assert not (driving["distance_mi"] == -99999.0).any()
    assert set(driving["distance_mi"]) == {12.0}
    assert (driving["distance_duration_source"] == pb.DISTANCE_DURATION_SOURCE_DONOR).all()


def test_run_profile_based_reconciliation_no_source_user_id_in_output(tmp_path):
    driver_profiles = pa.load_driver_profiles(_write_driver_profiles_csv(tmp_path))
    pool = _small_donor_pool()
    output = pb.run_profile_based_reconciliation(driver_profiles, pool, [1], seed=42)
    assert output["profile_employee_id"].unique().tolist() == ["PROF-0001"]


def test_run_profile_based_reconciliation_preserves_location_sequence(tmp_path):
    driver_profiles = pa.load_driver_profiles(_write_driver_profiles_csv(tmp_path))
    pool = _small_donor_pool()
    output = pb.run_profile_based_reconciliation(driver_profiles, pool, [1], seed=42)
    assert output.sort_values("row_index")["location"].tolist() == [
        "Home",
        "-1",
        "Work",
        "-1",
        "Home",
    ]
    assert output.sort_values("row_index")["state"].tolist() == [
        "Parked",
        "Driving",
        "Parked",
        "Driving",
        "Parked",
    ]


def test_run_profile_based_reconciliation_stays_contiguous_0_to_24h(tmp_path):
    driver_profiles = pa.load_driver_profiles(_write_driver_profiles_csv(tmp_path))
    pool = _small_donor_pool()
    output = pb.run_profile_based_reconciliation(driver_profiles, pool, [1], seed=42)
    rows = output.sort_values("row_index")
    starts = rows["start_hour"].to_numpy()
    ends = rows["end_hour"].to_numpy()
    assert starts[0] == pytest.approx(0.0)
    assert ends[-1] == pytest.approx(24.0)
    assert np.allclose(ends[:-1], starts[1:])


# --- I/O --------------------------------------------------------------------------


def test_save_and_load_profile_based_output_round_trip(tmp_path):
    output = pd.DataFrame({"profile_employee_id": ["PROF-0001"], "row_index": [0]})
    path = pb.save_profile_based_output(output, tmp_path)
    assert path == tmp_path / pb.OUTPUT_FILENAME
    loaded = pb.load_profile_based_output(tmp_path)
    pd.testing.assert_frame_equal(loaded, output)


def test_load_profile_based_output_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pb.load_profile_based_output(tmp_path)


# --- byte-diff guard: production files untouched (plan §8.9 required test b) ----


PRODUCTION_FILES = (
    Path("data/processed/synthetic_employees.parquet"),
    Path("data/processed/synthetic_activity.parquet"),
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(
    not all(p.exists() for p in PRODUCTION_FILES),
    reason="production parquet files not present in this checkout",
)
def test_profile_based_run_does_not_touch_production_parquet_files(tmp_path):
    pre_hashes = {p: _hash(p) for p in PRODUCTION_FILES}

    driver_profiles = pa.load_driver_profiles(_write_driver_profiles_csv(tmp_path))
    pool = _small_donor_pool()
    output = pb.run_profile_based_reconciliation(driver_profiles, pool, [1], seed=42)
    pb.save_profile_based_output(output, tmp_path / "profile_based_validation_output")

    post_hashes = {p: _hash(p) for p in PRODUCTION_FILES}
    assert pre_hashes == post_hashes
