"""Tests for driving_profiles.validation.activity."""

import pandas as pd
import pytest

from driving_profiles.generator import activity as ac
from driving_profiles.validation import activity as av

# --- fixtures (mirrors tests/test_activity.py's conventions) -----------------


def _trip_row(
    house_id, person_id, trip_id, strttime, endtime, trvlcmin, trpmiles, whytrp1s,
    loop_trip=2, vehtype=1.0, vehfuel=1.0,
):
    return {
        "HOUSEID": house_id, "PERSONID": person_id, "TRIPID": trip_id,
        "LOOP_TRIP": loop_trip, "STRTTIME": strttime, "ENDTIME": endtime,
        "TRVLCMIN": trvlcmin, "TRPMILES": trpmiles, "WHYTRP1S": whytrp1s,
        "VEHTYPE": vehtype, "VEHFUEL": vehfuel,
    }


def _trips_clean_df(*rows):
    df = pd.DataFrame(list(rows))
    df["HOUSEID"] = df["HOUSEID"].astype(str)
    df["PERSONID"] = df["PERSONID"].astype(str)
    df["TRIPID"] = df["TRIPID"].astype(str)
    return df


def _employee_clusters_df(rows):
    df = pd.DataFrame(rows, columns=["HOUSEID", "PERSONID", "cluster_id"])
    df["HOUSEID"] = df["HOUSEID"].astype(str)
    df["PERSONID"] = df["PERSONID"].astype(str)
    df["cluster_id"] = pd.array(df["cluster_id"], dtype="Int64")
    return df


def _donor_population():
    """cluster 0: D1 (2-leg direct commute), D2 (3-leg with a stop);
    cluster 1: D3 (2-leg direct commute)."""
    trips = _trips_clean_df(
        _trip_row("D1", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("D1", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
        _trip_row("D2", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("D2", "01", "02", 1200, 1215, 15, 3.0, whytrp1s=20),
        _trip_row("D2", "01", "03", 1230, 1300, 30, 4.0, whytrp1s=1),
        _trip_row("D3", "01", "01", 800, 830, 30, 12.0, whytrp1s=10),
        _trip_row("D3", "01", "02", 1700, 1730, 30, 12.0, whytrp1s=1),
    )
    clusters = _employee_clusters_df(
        [("D1", "01", 0), ("D2", "01", 0), ("D3", "01", 1)]
    )
    return trips, clusters


def _employee_row(
    synthetic_employee_id, cluster_id, trips_per_day, number_of_stops,
    work_arrival_time=830.0, work_departure_time=1700.0, total_daily_miles=20.0,
    commute_distance_survey_miles=10.0, total_driving_minutes=60.0,
    commute_duration_minutes=30.0,
):
    return {
        "synthetic_employee_id": synthetic_employee_id,
        "cluster_id": pd.array([cluster_id], dtype="Int64")[0],
        "trips_per_day": trips_per_day,
        "number_of_stops": number_of_stops,
        "work_arrival_time": work_arrival_time,
        "work_departure_time": work_departure_time,
        "total_daily_miles": total_daily_miles,
        "commute_distance_survey_miles": commute_distance_survey_miles,
        "total_driving_minutes": total_driving_minutes,
        "commute_duration_minutes": commute_duration_minutes,
    }


@pytest.fixture
def population():
    trips, clusters = _donor_population()
    employees = pd.DataFrame(
        [
            _employee_row("SYN-001", cluster_id=0, trips_per_day=2, number_of_stops=1),
            _employee_row("SYN-002", cluster_id=0, trips_per_day=3, number_of_stops=2),
            _employee_row("SYN-003", cluster_id=1, trips_per_day=2, number_of_stops=1),
            _employee_row(
                "SYN-004", cluster_id=0, trips_per_day=2, number_of_stops=1,
                work_arrival_time=880.0,  # jittered, "invalid" minute component
            ),
        ]
    )
    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)
    return trips, clusters, employees, activity


# --- load_synthetic_activity -------------------------------------------------


def test_load_synthetic_activity_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        av.load_synthetic_activity(tmp_path)


def test_load_synthetic_activity_reads_written_parquet(tmp_path, population):
    _, _, _, activity = population
    ac.save_synthetic_activity(activity, tmp_path)

    result = av.load_synthetic_activity(tmp_path)

    assert len(result) == len(activity)


# --- build_donor_pool / attach_cluster -----------------------------------------


def test_build_donor_pool_matches_generator_output(population):
    trips, clusters, _, _ = population

    donor_legs = av.build_donor_pool(clusters, trips)

    assert set(donor_legs["HOUSEID"]) == {"D1", "D2", "D3"}
    assert "cluster_id" in donor_legs.columns


def test_attach_cluster_maps_employee_to_cluster(population):
    _, _, employees, activity = population

    result = av.attach_cluster(activity, employees)

    syn1 = result.loc[result["synthetic_employee_id"] == "SYN-003"]
    assert (syn1["cluster_id"] == 1).all()


# --- validate_chain_length --------------------------------------------------------


def test_validate_chain_length_reports_per_cluster_and_chain_source(population):
    trips, clusters, employees, activity = population
    donor_legs = av.build_donor_pool(clusters, trips)
    activity_c = av.attach_cluster(activity, employees)

    result = av.validate_chain_length(donor_legs, activity_c)

    assert "legs_per_employee_day" in result["metric"].tolist()
    assert "donor" in result["chain_source"].tolist()
    assert "all" in result["chain_source"].tolist()


# --- validate_direct_commute_share ------------------------------------------------


def test_validate_direct_commute_share_only_covers_donor_chains(population):
    trips, clusters, employees, activity = population
    donor_legs = av.build_donor_pool(clusters, trips)
    activity_c = av.attach_cluster(activity, employees)

    result = av.validate_direct_commute_share(donor_legs, activity_c)

    assert (result["chain_source"] == ac.DONOR_CHAIN_SOURCE).all()


# --- validate_leg_distributions ---------------------------------------------------


def test_validate_leg_distributions_covers_distance_duration_and_times(population):
    trips, clusters, employees, activity = population
    donor_legs = av.build_donor_pool(clusters, trips)
    activity_c = av.attach_cluster(activity, employees)

    result = av.validate_leg_distributions(donor_legs, activity_c)

    expected = {"leg_distance", "leg_duration", "departure_time_minutes", "arrival_time_minutes"}
    assert expected <= set(result["metric"])


# --- validate_implied_speed_plausibility -------------------------------------------


def test_validate_implied_speed_plausibility_passes_for_normal_chain(population):
    _, _, _, activity = population

    result = av.validate_implied_speed_plausibility(activity)

    assert bool(result.iloc[0]["passed"]) is True

def test_validate_implied_speed_plausibility_fails_for_implausible_leg():
    activity = pd.DataFrame({"distance": [100.0], "duration": [1.0]})  # 6000 mph

    result = av.validate_implied_speed_plausibility(activity)

    assert bool(result.iloc[0]["passed"]) is False

# --- validate_workplace_timing_consistency -----------------------------------------


def test_validate_workplace_timing_consistency_passes_for_donor_chains(population):
    _, _, employees, activity = population

    result = av.validate_workplace_timing_consistency(activity, employees)

    arrival = result.loc[result["metric"] == "workplace_arrival_matches_drawn_value"].iloc[0]
    assert bool(arrival["passed"]) is True

def test_validate_workplace_timing_consistency_handles_jittered_invalid_minute_value(population):
    # SYN-004 has work_arrival_time=880.0 (an "8:80" jittered value) -
    # regression test for comparing in minutes-since-midnight rather than
    # raw HHMM, since rescale_chain_times re-encodes 880.0 -> 920.0.
    _, _, employees, activity = population

    result = av.validate_workplace_timing_consistency(activity, employees)

    arrival = result.loc[result["metric"] == "workplace_arrival_matches_drawn_value"].iloc[0]
    assert arrival["statistic"] == 0.0
    assert bool("0/" in arrival["detail"] or arrival["passed"]) is True

def test_validate_workplace_timing_consistency_only_checks_first_work_leg():
    # A fragmented-dwell chain (2 work-purpose legs): rescale_chain_times
    # anchors target_departure_hhmm to the leg immediately after the FIRST
    # work-purpose leg (departure_idx = arrival_idx + 1) - in this chain
    # that's the "leave for lunch" leg (index 1), not the final departure
    # from work (index 3), which is correctly left unanchored. So the
    # target 1245.0 belongs on index 1, and index 3's own value (1830.0)
    # must NOT match the target for this to be a meaningful test.
    activity = pd.DataFrame(
        {
            "synthetic_employee_id": ["SYN-X"] * 4,
            "trip_number": [1, 2, 3, 4],
            "trip_purpose": ["work", "other", "work", "home"],
            "arrival_time": [830.0, 1200.0, 1230.0, 1900.0],
            "departure_time": [800.0, 1245.0, 1215.0, 1830.0],
            "is_workplace_arrival": [True, False, True, False],
            "is_workplace_departure": [False, True, False, True],
        }
    )
    employees = pd.DataFrame(
        [
            {
                "synthetic_employee_id": "SYN-X",
                "work_arrival_time": 830.0,
                "work_departure_time": 1245.0,
            }
        ]
    )

    result = av.validate_workplace_timing_consistency(activity, employees)

    arrival = result.loc[result["metric"] == "workplace_arrival_matches_drawn_value"].iloc[0]
    departure = result.loc[result["metric"] == "workplace_departure_matches_drawn_value"].iloc[0]
    assert bool(arrival["passed"]) is True
    assert bool(departure["passed"]) is True
    assert departure["n_synthetic"] == 1  # only the first is_workplace_departure leg is checked


# --- validate_dwell_time -----------------------------------------------------------


def test_validate_dwell_time_uses_consistent_definition_both_sides(population):
    trips, clusters, employees, activity = population
    donor_legs = av.build_donor_pool(clusters, trips)
    activity_c = av.attach_cluster(activity, employees)

    result = av.validate_dwell_time(donor_legs, activity_c)

    assert (result["metric"] == "workplace_dwell_minutes").all()


# --- validate_fragmented_dwell_rate -------------------------------------------------


def test_validate_fragmented_dwell_rate_returns_a_row_when_data_available(population):
    trips, clusters, employees, activity = population
    donor_legs = av.build_donor_pool(clusters, trips)
    activity_c = av.attach_cluster(activity, employees)

    result = av.validate_fragmented_dwell_rate(donor_legs, activity_c)

    assert len(result) == 1
    assert result.iloc[0]["chain_source"] == ac.DONOR_CHAIN_SOURCE


# --- validate_fallback_rate ---------------------------------------------------------


def test_validate_fallback_rate_is_zero_when_all_donor_matched(population):
    _, _, _, activity = population
    activity_c = activity.assign(cluster_id=0)

    result = av.validate_fallback_rate(activity_c)

    pooled = result.loc[result["group"] == "pooled"].iloc[0]
    assert pooled["statistic"] == 0.0
    assert pooled["passed"] is None  # informational


def test_validate_fallback_rate_reports_nonzero_when_fallback_used():
    trips, clusters = _donor_population()
    employees = pd.DataFrame(
        [_employee_row("SYN-FB", cluster_id=99, trips_per_day=2, number_of_stops=1)]
    )
    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)
    activity_c = av.attach_cluster(activity, employees)

    result = av.validate_fallback_rate(activity_c)

    pooled = result.loc[result["group"] == "pooled"].iloc[0]
    assert pooled["statistic"] == 1.0


# --- run_activity_validation ---------------------------------------------------------


def test_run_activity_validation_combines_every_check(population):
    trips, clusters, employees, activity = population

    result = av.run_activity_validation(clusters, trips, activity, employees)

    assert result["section"].eq("activity").all()
    assert "implied_leg_speed_plausible" in result["metric"].tolist()
    assert "workplace_arrival_matches_drawn_value" in result["metric"].tolist()
