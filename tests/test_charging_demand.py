"""Tests for driving_profiles.scenarios.charging_demand."""

from __future__ import annotations

import pandas as pd
import pytest

from driving_profiles.generator.time_utils import hhmm_to_minutes
from driving_profiles.scenarios import charging_demand as cd

# --- fixtures -----------------------------------------------------------------


def _activity_row(
    employee_id: str,
    trip_number: int,
    arrival_time: float,
    workplace_dwell_minutes: float | None,
    is_workplace_arrival: bool = True,
    vehicle_type: float = 1.0,
    vehicle_fuel: float = 1.0,
) -> dict:
    return {
        "synthetic_employee_id": employee_id,
        "trip_number": trip_number,
        "arrival_time": arrival_time,
        "is_workplace_arrival": is_workplace_arrival,
        "workplace_dwell_minutes": workplace_dwell_minutes,
        "vehicle_type": vehicle_type,
        "vehicle_fuel": vehicle_fuel,
    }


def _activity_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _employee_row(
    employee_id: str,
    cluster_id: int = 0,
    is_worker: bool = True,
    used_household_vehicle: bool = True,
    total_daily_miles: float = 20.0,
) -> dict:
    return {
        "synthetic_employee_id": employee_id,
        "cluster_id": pd.array([cluster_id], dtype="Int64")[0],
        "is_worker": is_worker,
        "used_household_vehicle": used_household_vehicle,
        "total_daily_miles": total_daily_miles,
    }


def _employees_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _window_row(
    employee_id: str,
    visit_number: int,
    arrival_time_minutes: float,
    departure_time_minutes: float,
    open_ended_window: bool = False,
) -> dict:
    return {
        "synthetic_employee_id": employee_id,
        "workplace_visit_number": visit_number,
        "arrival_time_minutes": arrival_time_minutes,
        "departure_time_minutes": departure_time_minutes,
        "available_dwell_minutes": departure_time_minutes - arrival_time_minutes,
        "open_ended_window": open_ended_window,
    }


def _windows_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _eligibility_row(
    employee_id: str,
    cluster_id: int = 0,
    total_daily_miles: float = 20.0,
    driving_eligible: bool = True,
    has_workplace_window: bool = True,
    charging_eligible: bool = True,
    ev_assigned: bool = True,
) -> dict:
    return {
        "synthetic_employee_id": employee_id,
        "cluster_id": pd.array([cluster_id], dtype="Int64")[0],
        "total_daily_miles": total_daily_miles,
        "driving_eligible": driving_eligible,
        "has_workplace_window": has_workplace_window,
        "charging_eligible": charging_eligible,
        "ev_assigned": ev_assigned,
    }


def _eligibility_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


DEFAULT_CONFIG = cd.ChargingScenarioConfig(random_seed=42)


# --- ChargingScenarioConfig -----------------------------------------------------


def test_config_defaults_match_spec():
    config = cd.ChargingScenarioConfig()
    assert config.scenario_name == "baseline_unmanaged"
    assert config.ev_adoption_rate == 0.20
    assert config.vehicle_efficiency_kwh_per_mile == 0.30
    assert config.charging_efficiency == 0.90
    assert config.charger_power_kw == 7.2
    assert config.interval_minutes == 15
    assert config.open_ended_workplace_departure_minutes == 1439.0


# --- build_workplace_windows: unit conversion (items 1-3) ----------------------


def test_arrival_time_is_hhmm_converted():
    activity = _activity_df(_activity_row("SYN-1", 1, 814.513314, 251.744702))
    windows = cd.build_workplace_windows(activity, DEFAULT_CONFIG)

    assert windows.loc[0, "arrival_time_minutes"] == pytest.approx(
        hhmm_to_minutes(814.513314)
    )
    # Sanity: HHMM misread-as-minutes would give ~814.5, not ~494.5.
    assert windows.loc[0, "arrival_time_minutes"] == pytest.approx(494.51, abs=0.01)


def test_workplace_dwell_is_not_hhmm_converted():
    # A dwell of 251.744702 true minutes must be used as-is, not decoded as
    # if it were HHMM (which would silently corrupt it to ~2:51 -> 171 min).
    activity = _activity_df(_activity_row("SYN-1", 1, 814.513314, 251.744702))
    windows = cd.build_workplace_windows(activity, DEFAULT_CONFIG)

    assert windows.loc[0, "available_dwell_minutes"] == pytest.approx(251.744702)


def test_departure_computed_from_arrival_plus_dwell():
    activity = _activity_df(_activity_row("SYN-1", 1, 814.513314, 251.744702))
    windows = cd.build_workplace_windows(activity, DEFAULT_CONFIG)

    expected_departure = hhmm_to_minutes(814.513314) + 251.744702
    assert windows.loc[0, "departure_time_minutes"] == pytest.approx(expected_departure)
    # Matches the plan's worked example: re-encodes to exactly 1226.258017.
    assert expected_departure == pytest.approx(746.258016, abs=1e-3)


# --- open-ended / invalid windows (items 4-5) -----------------------------------


def test_open_ended_final_arrival_uses_configured_default():
    activity = _activity_df(_activity_row("SYN-1", 1, 900.0, workplace_dwell_minutes=float("nan")))
    windows = cd.build_workplace_windows(activity, DEFAULT_CONFIG)

    assert len(windows) == 1
    assert windows.loc[0, "open_ended_window"]
    assert windows.loc[0, "departure_time_minutes"] == 1439.0


def test_zero_or_negative_dwell_excluded():
    activity = _activity_df(
        _activity_row("SYN-1", 1, 800.0, workplace_dwell_minutes=0.0),
        _activity_row("SYN-2", 1, 800.0, workplace_dwell_minutes=30.0),
    )
    windows = cd.build_workplace_windows(activity, DEFAULT_CONFIG)

    assert windows["synthetic_employee_id"].tolist() == ["SYN-2"]


def test_invalid_arrival_time_excluded():
    activity = _activity_df(
        _activity_row("SYN-1", 1, float("nan"), workplace_dwell_minutes=30.0),
        _activity_row("SYN-2", 1, 800.0, workplace_dwell_minutes=30.0),
    )
    windows = cd.build_workplace_windows(activity, DEFAULT_CONFIG)

    assert windows["synthetic_employee_id"].tolist() == ["SYN-2"]


def test_visit_numbers_assigned_per_employee_sorted_by_arrival():
    activity = _activity_df(
        _activity_row("SYN-1", 3, 1300.0, workplace_dwell_minutes=30.0),
        _activity_row("SYN-1", 1, 800.0, workplace_dwell_minutes=30.0),
    )
    windows = cd.build_workplace_windows(activity, DEFAULT_CONFIG)

    windows = windows.sort_values("workplace_visit_number")
    assert windows["arrival_time_minutes"].tolist() == sorted(windows["arrival_time_minutes"])
    assert windows["workplace_visit_number"].tolist() == [1, 2]


# --- assign_evs: eligibility (items 6-7) ----------------------------------------


def test_driving_eligibility_independent_of_mileage_nullness():
    employees = _employees_df(
        _employee_row("SYN-1", used_household_vehicle=True, total_daily_miles=float("nan")),
    )
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 700.0))

    eligibility = cd.assign_evs(employees, windows, DEFAULT_CONFIG)

    row = eligibility.iloc[0]
    assert row["driving_eligible"]
    assert row["charging_eligible"]


def test_non_driving_employees_excluded_from_eligibility_and_sessions():
    employees = _employees_df(
        _employee_row("SYN-1", used_household_vehicle=False, total_daily_miles=20.0),
    )
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 700.0))

    eligibility = cd.assign_evs(employees, windows, DEFAULT_CONFIG)
    assert not eligibility.iloc[0]["driving_eligible"]
    assert not eligibility.iloc[0]["charging_eligible"]
    assert not eligibility.iloc[0]["ev_assigned"]

    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)
    assert sessions.empty


# --- assign_evs: deterministic assignment (items 8-10) --------------------------


def _pool_employees_and_windows(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"SYN-{i:04d}" for i in range(n)]
    employees = _employees_df(*[_employee_row(eid) for eid in ids])
    windows = _windows_df(*[_window_row(eid, 1, 500.0, 700.0) for eid in ids])
    return employees, windows


def test_ev_assignment_count_is_deterministic_round():
    employees, windows = _pool_employees_and_windows(10)
    config = cd.ChargingScenarioConfig(ev_adoption_rate=0.3, random_seed=1)

    eligibility = cd.assign_evs(employees, windows, config)

    assert int(eligibility["ev_assigned"].sum()) == round(10 * 0.3)


def test_ev_assignment_reproducible_with_fixed_seed():
    employees, windows = _pool_employees_and_windows(30)
    config = cd.ChargingScenarioConfig(ev_adoption_rate=0.4, random_seed=123)

    first = cd.assign_evs(employees, windows, config)
    second = cd.assign_evs(employees, windows, config)

    first_ids = set(first.loc[first["ev_assigned"], "synthetic_employee_id"])
    second_ids = set(second.loc[second["ev_assigned"], "synthetic_employee_id"])
    assert first_ids == second_ids


def test_ev_assignment_differs_with_different_seed():
    employees, windows = _pool_employees_and_windows(50)
    config_a = cd.ChargingScenarioConfig(ev_adoption_rate=0.5, random_seed=1)
    config_b = cd.ChargingScenarioConfig(ev_adoption_rate=0.5, random_seed=2)

    a = cd.assign_evs(employees, windows, config_a)
    b = cd.assign_evs(employees, windows, config_b)

    a_ids = set(a.loc[a["ev_assigned"], "synthetic_employee_id"])
    b_ids = set(b.loc[b["ev_assigned"], "synthetic_employee_id"])
    assert a_ids != b_ids


# --- donor vehicle_type/vehicle_fuel ignored (item 11) --------------------------


def test_donor_vehicle_columns_not_required_or_used():
    activity = pd.DataFrame(
        [
            {
                "synthetic_employee_id": "SYN-1",
                "trip_number": 1,
                "arrival_time": 800.0,
                "is_workplace_arrival": True,
                "workplace_dwell_minutes": 30.0,
                # deliberately no vehicle_type/vehicle_fuel columns at all
            }
        ]
    )
    windows = cd.build_workplace_windows(activity, DEFAULT_CONFIG)
    assert len(windows) == 1

    employees = _employees_df(_employee_row("SYN-1"))
    eligibility = cd.assign_evs(employees, windows, DEFAULT_CONFIG)
    assert eligibility.iloc[0]["charging_eligible"]


# --- energy calculation (items 12-14) -------------------------------------------


def test_requested_energy_calculation():
    config = cd.ChargingScenarioConfig(
        vehicle_efficiency_kwh_per_mile=0.30, charging_efficiency=0.90, random_seed=1
    )
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 700.0))
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=20.0))

    sessions = cd.create_charging_sessions(windows, eligibility, config)

    traction = 20.0 * 0.30
    expected_requested = traction / 0.90
    assert sessions.loc[0, "employee_requested_energy_kwh"] == pytest.approx(expected_requested)


def test_charging_loss_increases_requested_over_traction_energy():
    config = cd.ChargingScenarioConfig(
        vehicle_efficiency_kwh_per_mile=0.30, charging_efficiency=0.90, random_seed=1
    )
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 900.0))
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=20.0))

    sessions = cd.create_charging_sessions(windows, eligibility, config)

    traction = 20.0 * 0.30
    # charging_efficiency < 1 means grid energy requested > traction energy.
    assert sessions.loc[0, "employee_requested_energy_kwh"] > traction


@pytest.mark.parametrize("bad_miles", [float("nan"), float("inf"), -5.0])
def test_missing_or_unusable_mileage_excluded_from_sessions(bad_miles):
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 700.0))
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=bad_miles))

    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)

    assert sessions.empty


def test_zero_mileage_produces_single_zero_valued_session():
    windows = _windows_df(
        _window_row("SYN-1", 1, 500.0, 700.0),
        _window_row("SYN-1", 2, 800.0, 900.0),
    )
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=0.0))

    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)

    assert len(sessions) == 1
    assert sessions.loc[0, "delivered_energy_kwh"] == 0.0
    assert sessions.loc[0, "employee_unmet_energy_kwh"] == 0.0


# --- multi-visit allocation (items 15-20) ---------------------------------------


def test_multiple_visits_allocate_sequentially():
    config = cd.ChargingScenarioConfig(charger_power_kw=7.2, random_seed=1)
    # Visit 1: 60 min dwell -> max deliverable 7.2 kWh. Visit 2: long dwell,
    # can cover the remainder.
    windows = _windows_df(
        _window_row("SYN-1", 1, 500.0, 560.0),
        _window_row("SYN-1", 2, 800.0, 1000.0),
    )
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=100.0))

    sessions = cd.create_charging_sessions(windows, eligibility, config).sort_values(
        "workplace_visit_number"
    )

    total_requested = sessions.iloc[0]["employee_requested_energy_kwh"]
    visit1, visit2 = sessions.iloc[0], sessions.iloc[1]

    assert visit1["visit_requested_energy_kwh"] == pytest.approx(total_requested)
    assert visit1["delivered_energy_kwh"] == pytest.approx(7.2 * 60 / 60)
    assert visit2["visit_requested_energy_kwh"] == pytest.approx(
        total_requested - visit1["delivered_energy_kwh"]
    )
    assert visit2["remaining_energy_after_visit_kwh"] == pytest.approx(
        sessions.iloc[-1]["employee_unmet_energy_kwh"]
    )


def test_delivered_never_exceeds_requested():
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 1000.0))
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=1.0))

    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)

    assert (sessions["delivered_energy_kwh"] <= sessions["visit_requested_energy_kwh"] + 1e-9).all()


def test_delivered_never_exceeds_dwell_capacity():
    config = cd.ChargingScenarioConfig(charger_power_kw=7.2, random_seed=1)
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 530.0))  # 30 min dwell
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=1000.0))

    sessions = cd.create_charging_sessions(windows, eligibility, config)

    capacity = config.charger_power_kw * 30 / 60
    assert sessions.loc[0, "delivered_energy_kwh"] <= capacity + 1e-9


def test_charging_duration_never_exceeds_dwell():
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 530.0))
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=1000.0))

    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)

    assert sessions.loc[0, "charging_duration_minutes"] <= 30.0 + 1e-9


def test_charging_start_equals_arrival():
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 700.0))
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=20.0))

    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)

    assert sessions.loc[0, "charging_start_minutes"] == sessions.loc[0, "arrival_time_minutes"]


def test_charging_end_never_exceeds_workplace_departure():
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 700.0))
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=1000.0))

    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)

    assert (
        sessions.loc[0, "charging_end_minutes"]
        <= sessions.loc[0, "departure_time_minutes"] + 1e-9
    )


# --- load profile (items 21-25) -------------------------------------------------


def test_load_profile_has_exactly_96_intervals():
    profile = cd.build_load_profile(cd._empty_sessions_frame(), DEFAULT_CONFIG)
    assert len(profile) == 96
    assert profile["interval_start_minutes"].tolist() == [i * 15 for i in range(96)]
    assert profile["interval_end_minutes"].tolist() == [i * 15 + 15 for i in range(96)]


def test_partial_interval_energy_prorating():
    # A session charging 500->530 (spans intervals [495,510) and [510,525)
    # partially, plus [525,540) partially): interval boundaries at 15-min
    # marks are 495, 510, 525, 540.
    config = cd.ChargingScenarioConfig(charger_power_kw=7.2, random_seed=1)
    sessions = pd.DataFrame(
        [
            {
                "arrival_time_minutes": 500.0,
                "departure_time_minutes": 560.0,
                "charging_start_minutes": 500.0,
                "charging_end_minutes": 530.0,
                "charger_power_kw": 7.2,
                "delivered_energy_kwh": 7.2 * 30 / 60,
            }
        ]
    )
    profile = cd.build_load_profile(sessions, config)

    # Interval [495, 510): overlap = 10 min (500->510).
    interval_33 = profile.loc[profile["interval_start_minutes"] == 495].iloc[0]
    assert interval_33["interval_energy_kwh"] == pytest.approx(7.2 * 10 / 60)


def test_connected_ev_count_overlap_logic():
    config = cd.ChargingScenarioConfig(random_seed=1)
    sessions = pd.DataFrame(
        [
            {
                "arrival_time_minutes": 500.0,
                "departure_time_minutes": 560.0,
                "charging_start_minutes": 500.0,
                "charging_end_minutes": 500.0,  # zero-length charging window
                "charger_power_kw": 7.2,
                "delivered_energy_kwh": 0.0,
            }
        ]
    )
    profile = cd.build_load_profile(sessions, config)

    connected_interval = profile.loc[profile["interval_start_minutes"] == 495].iloc[0]
    assert connected_interval["connected_ev_count"] == 1
    assert connected_interval["charging_ev_count"] == 0


def test_charging_ev_count_overlap_logic():
    config = cd.ChargingScenarioConfig(random_seed=1)
    sessions = pd.DataFrame(
        [
            {
                "arrival_time_minutes": 500.0,
                "departure_time_minutes": 560.0,
                "charging_start_minutes": 500.0,
                "charging_end_minutes": 540.0,
                "charger_power_kw": 7.2,
                "delivered_energy_kwh": 7.2 * 40 / 60,
            }
        ]
    )
    profile = cd.build_load_profile(sessions, config)

    active_interval = profile.loc[profile["interval_start_minutes"] == 495].iloc[0]
    assert active_interval["charging_ev_count"] == 1


def test_interval_energy_reconciles_with_session_delivered_energy():
    windows = _windows_df(
        _window_row("SYN-1", 1, 500.0, 560.0),
        _window_row("SYN-2", 1, 700.0, 850.0),
    )
    eligibility = _eligibility_df(
        _eligibility_row("SYN-1", total_daily_miles=5.0),
        _eligibility_row("SYN-2", total_daily_miles=50.0),
    )
    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)
    profile = cd.build_load_profile(sessions, DEFAULT_CONFIG)

    assert profile["interval_energy_kwh"].sum() == pytest.approx(
        sessions["delivered_energy_kwh"].sum(), abs=1e-6
    )
    assert profile["cumulative_energy_kwh"].iloc[-1] == pytest.approx(
        profile["interval_energy_kwh"].sum()
    )


# --- no negative values (item 26) -----------------------------------------------


def test_no_negative_values_across_outputs():
    windows = _windows_df(
        _window_row("SYN-1", 1, 500.0, 560.0),
        _window_row("SYN-2", 1, 700.0, 705.0),  # tiny dwell, likely unmet
        _window_row("SYN-3", 1, 0.0, 100.0),
    )
    eligibility = _eligibility_df(
        _eligibility_row("SYN-1", total_daily_miles=5.0),
        _eligibility_row("SYN-2", total_daily_miles=500.0),
        _eligibility_row("SYN-3", total_daily_miles=0.0),
    )
    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)
    profile = cd.build_load_profile(sessions, DEFAULT_CONFIG)

    numeric_session_cols = [
        "available_dwell_minutes",
        "delivered_energy_kwh",
        "charging_duration_minutes",
        "employee_unmet_energy_kwh",
    ]
    for col in numeric_session_cols:
        assert (sessions[col] >= -1e-9).all()

    numeric_profile_cols = [
        "connected_ev_count",
        "charging_ev_count",
        "charging_power_kw",
        "interval_energy_kwh",
        "cumulative_energy_kwh",
    ]
    for col in numeric_profile_cols:
        assert (profile[col] >= -1e-9).all()


# --- no double counting (item 27) -----------------------------------------------


def test_summary_does_not_double_count_multi_visit_employee_energy():
    windows = _windows_df(
        _window_row("SYN-1", 1, 500.0, 560.0),
        _window_row("SYN-1", 2, 700.0, 900.0),
    )
    employees = _employees_df(_employee_row("SYN-1", total_daily_miles=100.0))
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=100.0))

    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)
    assert len(sessions) == 2  # both visits used (first doesn't fully cover)

    load_profile = cd.build_load_profile(sessions, DEFAULT_CONFIG)
    summary = cd.summarize_charging_scenario(
        employees, eligibility, windows, sessions, load_profile, DEFAULT_CONFIG
    )

    expected_requested = sessions.iloc[0]["employee_requested_energy_kwh"]
    assert summary.loc[0, "total_requested_energy_kwh"] == pytest.approx(expected_requested)
    expected_unmet = sessions.iloc[0]["employee_unmet_energy_kwh"]
    assert summary.loc[0, "total_unmet_energy_kwh"] == pytest.approx(expected_unmet)


# --- output files (item 28) -----------------------------------------------------


def test_save_charging_outputs_creates_all_three_files(tmp_path):
    windows = _windows_df(_window_row("SYN-1", 1, 500.0, 700.0))
    eligibility = _eligibility_df(_eligibility_row("SYN-1", total_daily_miles=20.0))
    employees = _employees_df(_employee_row("SYN-1", total_daily_miles=20.0))

    sessions = cd.create_charging_sessions(windows, eligibility, DEFAULT_CONFIG)
    load_profile = cd.build_load_profile(sessions, DEFAULT_CONFIG)
    summary = cd.summarize_charging_scenario(
        employees, eligibility, windows, sessions, load_profile, DEFAULT_CONFIG
    )

    paths = cd.save_charging_outputs(sessions, load_profile, summary, tmp_path)

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


# --- input files untouched (item 29) --------------------------------------------


def test_run_charging_scenario_does_not_modify_input_parquet_files(tmp_path):
    from driving_profiles.generator import sample as sample_module
    from driving_profiles.generator.activity import ACTIVITY_TABLE_FILENAME

    employees = _employees_df(
        _employee_row("SYN-1", total_daily_miles=20.0),
        _employee_row("SYN-2", used_household_vehicle=False, total_daily_miles=float("nan")),
    )
    activity = _activity_df(
        _activity_row("SYN-1", 1, 800.0, 30.0),
    )
    employees_path = tmp_path / sample_module.SYNTHETIC_EMPLOYEE_FILENAME
    activity_path = tmp_path / ACTIVITY_TABLE_FILENAME
    employees.to_parquet(employees_path, index=False)
    activity.to_parquet(activity_path, index=False)

    before_employees = employees_path.read_bytes()
    before_activity = activity_path.read_bytes()

    cd.run_charging_scenario(tmp_path, DEFAULT_CONFIG)

    assert employees_path.read_bytes() == before_employees
    assert activity_path.read_bytes() == before_activity


def test_load_charging_inputs_fails_clearly_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Synthetic employee table not found"):
        cd.load_charging_inputs(tmp_path)


# --- run_pipeline.py integration (item 30) --------------------------------------


def test_run_pipeline_includes_charging_stage_and_cli_options():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline_charging_check", script_path)
    run_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_pipeline)

    assert hasattr(run_pipeline, "stage_charging_demand")

    args = run_pipeline.parse_args([])
    assert hasattr(args, "ev_adoption_rate")
    assert hasattr(args, "charger_power_kw")
    assert hasattr(args, "vehicle_efficiency_kwh_per_mile")
    assert hasattr(args, "charging_efficiency")
    assert args.ev_adoption_rate is None
    assert args.charger_power_kw is None
