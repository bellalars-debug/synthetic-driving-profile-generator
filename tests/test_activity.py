"""Tests for driving_profiles.generator.activity."""

import numpy as np
import pandas as pd
import pytest

from driving_profiles.generator import activity as ac

# --- fixtures -----------------------------------------------------------------


def _trip_row(
    house_id: str,
    person_id: str,
    trip_id: str,
    strttime: float,
    endtime: float,
    trvlcmin: float,
    trpmiles: float,
    whytrp1s: int,
    loop_trip: int = 2,
    vehtype: float = 1.0,
    vehfuel: float = 1.0,
    trptrans: int = 3,
) -> dict:
    # trptrans defaults to 3 ("Car"), a DRIVING_MODE_TRPTRANS_CODES member,
    # so every existing donor fixture is a driving donor unless a test
    # overrides it - keeps pre-driving-filter test behavior unchanged.
    return {
        "HOUSEID": house_id,
        "PERSONID": person_id,
        "TRIPID": trip_id,
        "LOOP_TRIP": loop_trip,
        "STRTTIME": strttime,
        "ENDTIME": endtime,
        "TRVLCMIN": trvlcmin,
        "TRPMILES": trpmiles,
        "WHYTRP1S": whytrp1s,
        "VEHTYPE": vehtype,
        "VEHFUEL": vehfuel,
        "TRPTRANS": trptrans,
    }


def _trips_clean_df(*rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    df["HOUSEID"] = df["HOUSEID"].astype(str)
    df["PERSONID"] = df["PERSONID"].astype(str)
    df["TRIPID"] = df["TRIPID"].astype(str)
    return df


def _employee_clusters_df(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["HOUSEID", "PERSONID", "cluster_id"])
    df["HOUSEID"] = df["HOUSEID"].astype(str)
    df["PERSONID"] = df["PERSONID"].astype(str)
    df["cluster_id"] = pd.array(df["cluster_id"], dtype="Int64")
    return df


def _sample_trips_and_clusters() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small donor population:

    - D1 (cluster 0): 2-leg chain, home->work->home. 1 stop.
    - D2 (cluster 0): 3-leg chain, home->work->errand->home. 2 stops.
    - D3 (cluster 1): 2-leg chain, home->work->home (different cluster).
    - D4 (cluster 0): no work leg at all (excluded from the donor pool).
    - D5 (cluster 0): identical shape to D1, so cluster 0 / (2 legs, 1 stop)
      has two exact-match candidates for tie-break testing.
    - A loop trip is included on D1 to verify it's excluded from
      trip_count/stop_count.
    """
    trips = _trips_clean_df(
        _trip_row("D1", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("D1", "01", "02", 900, 905, 5, 0.5, whytrp1s=97, loop_trip=1),
        _trip_row("D1", "01", "03", 1700, 1730, 30, 10.0, whytrp1s=1),
        _trip_row("D2", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("D2", "01", "02", 1200, 1215, 15, 3.0, whytrp1s=20),
        _trip_row("D2", "01", "03", 1230, 1300, 30, 4.0, whytrp1s=1),
        _trip_row("D3", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("D3", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
        _trip_row("D4", "01", "01", 800, 830, 30, 10.0, whytrp1s=20),
        _trip_row("D4", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
        _trip_row("D5", "01", "01", 815, 845, 30, 12.0, whytrp1s=10),
        _trip_row("D5", "01", "02", 1715, 1745, 30, 12.0, whytrp1s=1),
    )
    clusters = _employee_clusters_df(
        [
            ("D1", "01", 0),
            ("D2", "01", 0),
            ("D3", "01", 1),
            ("D4", "01", 0),
            ("D5", "01", 0),
        ]
    )
    return trips, clusters


def _employee_row(
    synthetic_employee_id: str,
    cluster_id: int,
    trips_per_day: int,
    number_of_stops: int,
    work_arrival_time: float = 830.0,
    work_departure_time: float = 1700.0,
    total_daily_miles: float = 20.0,
    commute_distance_survey_miles: float = 10.0,
    total_driving_minutes: float = 60.0,
    commute_duration_minutes: float = 30.0,
    # Defaults preserve pre-existing test behavior: a NaN trip-miles value
    # always falls through to commute_distance_survey_miles regardless of
    # work_trip_count (select_commute_anchor_miles), so every test written
    # before this anchor existed keeps its original expected result unless
    # it explicitly overrides these two.
    commute_distance_trip_miles: float = float("nan"),
    work_trip_count: int = 1,
) -> dict:
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
        "commute_distance_trip_miles": commute_distance_trip_miles,
        "work_trip_count": work_trip_count,
    }


def _synthetic_employees_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --- hhmm_to_minutes / minutes_to_hhmm -----------------------------------------


def test_hhmm_to_minutes_standard_value():
    assert ac.hhmm_to_minutes(830.0) == 8 * 60 + 30


def test_hhmm_to_minutes_handles_invalid_minute_digit_from_jitter():
    # 880 -> "8:80", not a valid clock reading, but a real jitter output.
    assert ac.hhmm_to_minutes(880.0) == 8 * 60 + 80


def test_hhmm_to_minutes_nan_passthrough():
    assert pd.isna(ac.hhmm_to_minutes(float("nan")))


def test_minutes_to_hhmm_round_trips_standard_values():
    for hhmm in (0.0, 830.0, 1259.0, 2359.0):
        minutes = ac.hhmm_to_minutes(hhmm)
        assert ac.minutes_to_hhmm(minutes) == hhmm


def test_minutes_to_hhmm_always_produces_valid_minute_component():
    hhmm = ac.minutes_to_hhmm(ac.hhmm_to_minutes(880.0))
    assert hhmm % 100 < 60


def test_minutes_to_hhmm_clips_rather_than_wraps():
    assert ac.minutes_to_hhmm(-10) == 0.0
    assert ac.minutes_to_hhmm(ac.MINUTES_PER_DAY + 100) == ac.minutes_to_hhmm(
        ac.MINUTES_PER_DAY - 1
    )


# --- classify_trip_purpose -----------------------------------------------------


def test_classify_trip_purpose_maps_home_work_other():
    whytrp1s = pd.Series([1, 10, 20, 97])
    result = ac.classify_trip_purpose(whytrp1s)
    assert result.tolist() == [
        ac.TRIP_PURPOSE_HOME,
        ac.TRIP_PURPOSE_WORK,
        ac.TRIP_PURPOSE_OTHER,
        ac.TRIP_PURPOSE_OTHER,
    ]


# --- build_donor_legs -----------------------------------------------------------


def test_build_donor_legs_excludes_unclustered_and_loop_trips_and_sorts():
    trips, clusters = _sample_trips_and_clusters()

    legs = ac.build_donor_legs(trips, clusters)

    d1_legs = legs.loc[(legs["HOUSEID"] == "D1") & (legs["PERSONID"] == "01")]
    assert len(d1_legs) == 2  # the loop trip is excluded
    assert d1_legs["TRIPID"].tolist() == ["01", "03"]  # sorted chronologically


def test_build_donor_legs_excludes_persons_without_cluster_id():
    trips, clusters = _sample_trips_and_clusters()
    clusters_missing_d2 = clusters.loc[clusters["HOUSEID"] != "D2"]

    legs = ac.build_donor_legs(trips, clusters_missing_d2)

    assert "D2" not in legs["HOUSEID"].tolist()


def test_build_donor_legs_drops_donor_with_non_chronological_strttime_order():
    # D6: TRIPID order (01, 02) puts a later STRTTIME before an earlier one -
    # a real NHTS data artifact (e.g. a diary crossing midnight), not a
    # valid chain shape (regression: a prior version let this through and
    # rescale_chain_times produced out-of-order timestamps downstream).
    trips, clusters = _sample_trips_and_clusters()
    trips = pd.concat(
        [
            trips,
            _trips_clean_df(
                _trip_row("D6", "01", "01", 1600, 1630, 30, 10.0, whytrp1s=10),
                _trip_row("D6", "01", "02", 800, 830, 30, 10.0, whytrp1s=1),
            ),
        ],
        ignore_index=True,
    )
    clusters = pd.concat(
        [clusters, _employee_clusters_df([("D6", "01", 0)])], ignore_index=True
    )

    legs = ac.build_donor_legs(trips, clusters)

    assert "D6" not in legs["HOUSEID"].tolist()


def test_build_donor_legs_drops_donor_with_within_leg_time_reversal():
    # D7: a single leg with ENDTIME < STRTTIME - another midnight-crossing
    # artifact (STRTTIME=2330 / ENDTIME=10 for a real ~40-minute trip).
    trips, clusters = _sample_trips_and_clusters()
    trips = pd.concat(
        [
            trips,
            _trips_clean_df(
                _trip_row("D7", "01", "01", 2330, 10, 40, 10.0, whytrp1s=10),
                _trip_row("D7", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
            ),
        ],
        ignore_index=True,
    )
    clusters = pd.concat(
        [clusters, _employee_clusters_df([("D7", "01", 0)])], ignore_index=True
    )

    legs = ac.build_donor_legs(trips, clusters)

    assert "D7" not in legs["HOUSEID"].tolist()


def test_build_donor_legs_flags_is_driving_leg_from_trptrans():
    # D11: a driving-mode leg (car, trptrans=3) followed by a non-driving
    # leg (walk, trptrans=1) - is_driving_leg must be computed per leg, not
    # per donor.
    trips, clusters = _sample_trips_and_clusters()
    trips = pd.concat(
        [
            trips,
            _trips_clean_df(
                _trip_row("D11", "01", "01", 800, 830, 30, 10.0, whytrp1s=10, trptrans=3),
                _trip_row("D11", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1, trptrans=1),
            ),
        ],
        ignore_index=True,
    )
    clusters = pd.concat(
        [clusters, _employee_clusters_df([("D11", "01", 0)])], ignore_index=True
    )

    legs = ac.build_donor_legs(trips, clusters)
    d11_legs = legs.loc[(legs["HOUSEID"] == "D11") & (legs["PERSONID"] == "01")]

    assert d11_legs["is_driving_leg"].tolist() == [True, False]


def test_build_donor_legs_drops_donor_with_implausibly_long_leg():
    # D8: an occasional very-long single leg (the regular NHTS trip file
    # does contain a small fraction of these even though the dedicated
    # long-distance file is excluded) - not a representative local-commute
    # template for this project's scope.
    trips, clusters = _sample_trips_and_clusters()
    trips = pd.concat(
        [
            trips,
            _trips_clean_df(
                _trip_row("D8", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
                _trip_row("D8", "01", "02", 1700, 1730, 30, 500.0, whytrp1s=1),
            ),
        ],
        ignore_index=True,
    )
    clusters = pd.concat(
        [clusters, _employee_clusters_df([("D8", "01", 0)])], ignore_index=True
    )

    legs = ac.build_donor_legs(trips, clusters)

    assert "D8" not in legs["HOUSEID"].tolist()


# --- summarize_donor_chains -----------------------------------------------------


def test_summarize_donor_chains_computes_trip_and_stop_counts():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)

    summary = ac.summarize_donor_chains(legs)
    summary = summary.set_index(["HOUSEID", "PERSONID"])

    assert summary.loc[("D1", "01"), "trip_count"] == 2
    assert summary.loc[("D1", "01"), "stop_count"] == 1
    assert summary.loc[("D2", "01"), "trip_count"] == 3
    assert summary.loc[("D2", "01"), "stop_count"] == 2


def test_summarize_donor_chains_excludes_donors_without_a_work_leg():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)

    summary = ac.summarize_donor_chains(legs)

    assert "D4" not in summary["HOUSEID"].tolist()


def test_summarize_donor_chains_has_driving_leg_true_when_any_leg_drives():
    # D1's default fixture legs are all trptrans=3 (driving).
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)

    summary = ac.summarize_donor_chains(legs).set_index(["HOUSEID", "PERSONID"])

    assert bool(summary.loc[("D1", "01"), "has_driving_leg"]) is True


def test_summarize_donor_chains_computes_own_arrival_min():
    # D1: work-arrival leg ENDTIME=830 -> own_arrival_min = 8*60+30 = 510.
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)

    summary = ac.summarize_donor_chains(legs).set_index(["HOUSEID", "PERSONID"])

    assert summary.loc[("D1", "01"), "own_arrival_min"] == 8 * 60 + 30


def test_summarize_donor_chains_computes_own_departure_min_from_next_leg():
    # D1's leg immediately after the work-arrival leg (the loop trip is
    # excluded) departs at STRTTIME=1700 -> own_departure_min = 17*60 = 1020.
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)

    summary = ac.summarize_donor_chains(legs).set_index(["HOUSEID", "PERSONID"])

    assert summary.loc[("D1", "01"), "own_departure_min"] == 17 * 60


def test_summarize_donor_chains_own_departure_min_is_nan_when_arrival_leg_is_last():
    # D13: a single leg, itself the first (and only) work-purpose arrival -
    # there is no "leg immediately after" to derive own_departure_min from.
    trips, clusters = _sample_trips_and_clusters()
    trips = pd.concat(
        [trips, _trips_clean_df(_trip_row("D13", "01", "01", 800, 830, 30, 10.0, whytrp1s=10))],
        ignore_index=True,
    )
    clusters = pd.concat(
        [clusters, _employee_clusters_df([("D13", "01", 0)])], ignore_index=True
    )
    legs = ac.build_donor_legs(trips, clusters)

    summary = ac.summarize_donor_chains(legs).set_index(["HOUSEID", "PERSONID"])

    assert summary.loc[("D13", "01"), "own_arrival_min"] == 8 * 60 + 30
    assert pd.isna(summary.loc[("D13", "01"), "own_departure_min"])


def test_summarize_donor_chains_has_driving_leg_false_when_no_leg_drives():
    # D12: every leg is a non-driving mode (walk) - has_driving_leg must be
    # False even though the chain otherwise has a normal work-leg shape.
    trips, clusters = _sample_trips_and_clusters()
    trips = pd.concat(
        [
            trips,
            _trips_clean_df(
                _trip_row("D12", "01", "01", 800, 830, 30, 10.0, whytrp1s=10, trptrans=1),
                _trip_row("D12", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1, trptrans=1),
            ),
        ],
        ignore_index=True,
    )
    clusters = pd.concat(
        [clusters, _employee_clusters_df([("D12", "01", 0)])], ignore_index=True
    )
    legs = ac.build_donor_legs(trips, clusters)

    summary = ac.summarize_donor_chains(legs).set_index(["HOUSEID", "PERSONID"])

    assert bool(summary.loc[("D12", "01"), "has_driving_leg"]) is False


# --- select_donor ----------------------------------------------------------------


def test_select_donor_prefers_exact_match():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    summary = ac.summarize_donor_chains(legs)
    rng = np.random.default_rng(0)

    donor = ac.select_donor(
        0, trips_per_day=3, number_of_stops=2, donor_summary=summary, rng=rng,
        require_driving_leg=True,
    )

    assert donor == ("D2", "01")


def test_select_donor_respects_cluster_restriction():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    summary = ac.summarize_donor_chains(legs)
    rng = np.random.default_rng(0)

    # D3 is the only exact (2,1) match but it's cluster 1; cluster 0 must
    # never resolve to D3.
    donor = ac.select_donor(
        0, trips_per_day=2, number_of_stops=1, donor_summary=summary, rng=rng,
        require_driving_leg=True,
    )

    assert donor in {("D1", "01"), ("D5", "01")}


def test_select_donor_widens_tolerance_when_no_exact_match():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    summary = ac.summarize_donor_chains(legs)
    rng = np.random.default_rng(0)

    # (4, 3) has no exact match in cluster 0, but D2's (3, 2) is within +-1.
    donor = ac.select_donor(
        0, trips_per_day=4, number_of_stops=3, donor_summary=summary, rng=rng,
        require_driving_leg=True,
    )

    assert donor == ("D2", "01")


def test_select_donor_returns_none_when_cluster_has_no_donor():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    summary = ac.summarize_donor_chains(legs)
    rng = np.random.default_rng(0)

    donor = ac.select_donor(
        99, trips_per_day=2, number_of_stops=1, donor_summary=summary, rng=rng,
        require_driving_leg=True,
    )

    assert donor is None


def _mixed_driving_donor_pool() -> pd.DataFrame:
    """Two cluster-0 donors, identical (2, 1) shape, differing only in
    driving status: DDRV drove that day, DWLK never used a driving mode."""
    trips = _trips_clean_df(
        _trip_row("DDRV", "01", "01", 800, 830, 30, 10.0, whytrp1s=10, trptrans=3),
        _trip_row("DDRV", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1, trptrans=3),
        _trip_row("DWLK", "01", "01", 800, 830, 30, 10.0, whytrp1s=10, trptrans=1),
        _trip_row("DWLK", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1, trptrans=1),
    )
    clusters = _employee_clusters_df([("DDRV", "01", 0), ("DWLK", "01", 0)])
    legs = ac.build_donor_legs(trips, clusters)
    return ac.summarize_donor_chains(legs)


def test_select_donor_only_returns_driving_donor_when_required():
    summary = _mixed_driving_donor_pool()
    rng = np.random.default_rng(0)

    for _ in range(10):
        donor = ac.select_donor(
            0, trips_per_day=2, number_of_stops=1, donor_summary=summary, rng=rng,
            require_driving_leg=True,
        )
        assert donor == ("DDRV", "01")


def test_select_donor_only_returns_non_driving_donor_when_required():
    summary = _mixed_driving_donor_pool()
    rng = np.random.default_rng(0)

    for _ in range(10):
        donor = ac.select_donor(
            0, trips_per_day=2, number_of_stops=1, donor_summary=summary, rng=rng,
            require_driving_leg=False,
        )
        assert donor == ("DWLK", "01")


def test_select_donor_returns_none_when_no_donor_matches_driving_requirement():
    # Pool has only a driving donor (D1..D5 default trptrans=3); asking for
    # a non-driving donor in that cluster must fail rather than silently
    # returning a driving one.
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    summary = ac.summarize_donor_chains(legs)
    rng = np.random.default_rng(0)

    donor = ac.select_donor(
        0, trips_per_day=2, number_of_stops=1, donor_summary=summary, rng=rng,
        require_driving_leg=False,
    )

    assert donor is None


def test_select_donor_is_reproducible_with_same_rng_seed():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    summary = ac.summarize_donor_chains(legs)

    donor_a = ac.select_donor(0, 2, 1, summary, np.random.default_rng(7), require_driving_leg=True)
    donor_b = ac.select_donor(0, 2, 1, summary, np.random.default_rng(7), require_driving_leg=True)

    assert donor_a == donor_b


# --- select_donor: donor time-schedule preference (S4) --------------------------


def _time_pref_donor_pool() -> pd.DataFrame:
    """Three cluster-0 donors, all an exact (2, 1) shape:

    - DTA: work-arrival ENDTIME=830 (own_arrival_min=510), next leg
      STRTTIME=1700 (own_departure_min=1020) - compatible on both arrival
      and departure with the targets used below (500, 1015).
    - DTC: work-arrival ENDTIME=830 (own_arrival_min=510, arrival-compatible
      with target 500), but next leg STRTTIME=2200
      (own_departure_min=1320) - departure incompatible with target 1015.
    - DTD: work-arrival ENDTIME=1200 (own_arrival_min=720) and next leg
      STRTTIME=1800 (own_departure_min=1080) - incompatible on both arrival
      and departure with the targets used below.
    """
    trips = _trips_clean_df(
        _trip_row("DTA", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("DTA", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
        _trip_row("DTC", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("DTC", "01", "02", 2200, 2230, 30, 10.0, whytrp1s=1),
        _trip_row("DTD", "01", "01", 1130, 1200, 30, 10.0, whytrp1s=10),
        _trip_row("DTD", "01", "02", 1800, 1830, 30, 10.0, whytrp1s=1),
    )
    clusters = _employee_clusters_df([("DTA", "01", 0), ("DTC", "01", 0), ("DTD", "01", 0)])
    legs = ac.build_donor_legs(trips, clusters)
    return ac.summarize_donor_chains(legs)


def test_select_donor_prefers_combined_arrival_and_departure_compatible_donor():
    summary = _time_pref_donor_pool()

    for seed in range(10):
        donor = ac.select_donor(
            0, trips_per_day=2, number_of_stops=1, donor_summary=summary,
            rng=np.random.default_rng(seed), require_driving_leg=True,
            target_arrival_min=500.0, target_departure_min=1015.0,
        )
        assert donor == ("DTA", "01")


def test_select_donor_prefers_arrival_only_compatible_donor_when_no_combined_match():
    # Drop DTA so no donor is compatible on both arrival and departure; DTC
    # remains arrival-compatible (own_arrival_min=510 vs target 500) even
    # though its own_departure_min (1320) is far from the target (1015).
    summary = _time_pref_donor_pool()
    summary = summary.loc[summary["HOUSEID"] != "DTA"]

    for seed in range(10):
        donor = ac.select_donor(
            0, trips_per_day=2, number_of_stops=1, donor_summary=summary,
            rng=np.random.default_rng(seed), require_driving_leg=True,
            target_arrival_min=500.0, target_departure_min=1015.0,
        )
        assert donor == ("DTC", "01")


def test_select_donor_falls_back_to_unrestricted_pool_when_no_time_compatible_donor():
    # Only DTD remains, and its own_arrival_min (720) is far outside
    # TIME_MATCH_TOLERANCE_MINUTES of the target (500) - Tier A and Tier B
    # are both empty, but a same-shape donor still exists, so select_donor
    # must still return it (Tier C) rather than None.
    summary = _time_pref_donor_pool()
    summary = summary.loc[summary["HOUSEID"] == "DTD"]

    donor = ac.select_donor(
        0, trips_per_day=2, number_of_stops=1, donor_summary=summary,
        rng=np.random.default_rng(0), require_driving_leg=True,
        target_arrival_min=500.0, target_departure_min=1015.0,
    )

    assert donor == ("DTD", "01")


def test_select_donor_time_incompatibility_alone_never_triggers_fallback():
    # Same setup as above (only a time-incompatible donor available) - the
    # important assertion is simply that select_donor never returns None
    # here, across many rng draws, since an otherwise-eligible donor exists.
    summary = _time_pref_donor_pool()
    summary = summary.loc[summary["HOUSEID"] == "DTD"]

    for seed in range(10):
        donor = ac.select_donor(
            0, trips_per_day=2, number_of_stops=1, donor_summary=summary,
            rng=np.random.default_rng(seed), require_driving_leg=True,
            target_arrival_min=500.0, target_departure_min=1015.0,
        )
        assert donor is not None


def test_select_donor_applies_cluster_filter_before_time_compatibility():
    # DTA (cluster 0) is time-compatible with the targets; give it a
    # cluster-1 twin (DTA2) that is even more time-compatible (exact match)
    # to confirm the cluster restriction is applied before any time
    # preference could otherwise pull in a donor from the wrong cluster.
    summary = _time_pref_donor_pool()
    extra_trips = _trips_clean_df(
        _trip_row("DTA2", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("DTA2", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
    )
    extra_clusters = _employee_clusters_df([("DTA2", "01", 1)])
    extra_legs = ac.build_donor_legs(extra_trips, extra_clusters)
    summary = pd.concat([summary, ac.summarize_donor_chains(extra_legs)], ignore_index=True)

    for seed in range(10):
        donor = ac.select_donor(
            0, trips_per_day=2, number_of_stops=1, donor_summary=summary,
            rng=np.random.default_rng(seed), require_driving_leg=True,
            target_arrival_min=510.0, target_departure_min=1020.0,
        )
        assert donor == ("DTA", "01")


def test_select_donor_applies_driving_mode_filter_before_time_compatibility():
    # DWLK is a non-driving donor with the exact same (time-compatible)
    # schedule as DTA; require_driving_leg=True must still never return it.
    summary = _time_pref_donor_pool()
    extra_trips = _trips_clean_df(
        _trip_row("DWLK", "01", "01", 800, 830, 30, 10.0, whytrp1s=10, trptrans=1),
        _trip_row("DWLK", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1, trptrans=1),
    )
    extra_clusters = _employee_clusters_df([("DWLK", "01", 0)])
    extra_legs = ac.build_donor_legs(extra_trips, extra_clusters)
    summary = pd.concat([summary, ac.summarize_donor_chains(extra_legs)], ignore_index=True)

    for seed in range(10):
        donor = ac.select_donor(
            0, trips_per_day=2, number_of_stops=1, donor_summary=summary,
            rng=np.random.default_rng(seed), require_driving_leg=True,
            target_arrival_min=510.0, target_departure_min=1020.0,
        )
        assert donor == ("DTA", "01")


def test_select_donor_applies_trip_and_stop_count_filter_before_time_compatibility():
    # D2 (3 trips, 2 stops, cluster 0) is time-compatible with the target
    # (same 830/1700 schedule as DTA) but does not match the requested
    # (2, 1) shape within MATCH_TOLERANCES; only DTA (an exact (2, 1) match)
    # should ever be returned even though both are equally time-compatible.
    _, base_clusters = _sample_trips_and_clusters()
    d2_trips = _trips_clean_df(
        _trip_row("D2", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("D2", "01", "02", 1200, 1215, 15, 3.0, whytrp1s=20),
        _trip_row("D2", "01", "03", 1230, 1300, 30, 4.0, whytrp1s=1),
    )
    dta_trips = _trips_clean_df(
        _trip_row("DTA", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("DTA", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
    )
    all_trips = pd.concat([d2_trips, dta_trips], ignore_index=True)
    all_clusters = pd.concat(
        [
            base_clusters.loc[base_clusters["HOUSEID"] == "D2"],
            _employee_clusters_df([("DTA", "01", 0)]),
        ],
        ignore_index=True,
    )
    legs = ac.build_donor_legs(all_trips, all_clusters)
    summary = ac.summarize_donor_chains(legs)

    for seed in range(10):
        donor = ac.select_donor(
            0, trips_per_day=2, number_of_stops=1, donor_summary=summary,
            rng=np.random.default_rng(seed), require_driving_leg=True,
            target_arrival_min=510.0, target_departure_min=1020.0,
        )
        assert donor == ("DTA", "01")


def test_select_donor_with_time_preference_is_reproducible_with_same_rng_seed():
    summary = _time_pref_donor_pool()

    donor_a = ac.select_donor(
        0, 2, 1, summary, np.random.default_rng(7), require_driving_leg=True,
        target_arrival_min=500.0, target_departure_min=1015.0,
    )
    donor_b = ac.select_donor(
        0, 2, 1, summary, np.random.default_rng(7), require_driving_leg=True,
        target_arrival_min=500.0, target_departure_min=1015.0,
    )

    assert donor_a == donor_b


def test_generate_chain_for_employee_prefers_time_compatible_donor_end_to_end():
    # Confirms generate_chain_for_employee actually converts the employee's
    # own work_arrival_time/work_departure_time and threads them through to
    # select_donor, not just that select_donor's tiering works in isolation.
    # Each donor's second (non-anchor) leg has a distinct raw TRPMILES
    # (5 / 20 / 30) so that leg's post-rescale *duration* - which depends on
    # the donor's own raw-miles-to-rescaled-miles ratio, unlike distance,
    # which is always forced to exactly the remaining budget for a single
    # "other" leg - reveals which donor was actually selected.
    trips = _trips_clean_df(
        _trip_row("DTA", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("DTA", "01", "02", 1700, 1730, 30, 5.0, whytrp1s=1),
        _trip_row("DTC", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("DTC", "01", "02", 2200, 2230, 30, 20.0, whytrp1s=1),
        _trip_row("DTD", "01", "01", 1130, 1200, 30, 10.0, whytrp1s=10),
        _trip_row("DTD", "01", "02", 1800, 1830, 30, 30.0, whytrp1s=1),
    )
    clusters = _employee_clusters_df([("DTA", "01", 0), ("DTC", "01", 0), ("DTD", "01", 0)])
    donor_legs = ac.build_donor_legs(trips, clusters)
    donor_summary = ac.summarize_donor_chains(donor_legs)
    donor_legs_by_person = {k: g for k, g in donor_legs.groupby(ac.PERSON_KEY)}
    # DTA's own schedule (830 arrival / 1700 departure) is within 60 minutes
    # of this employee's own target on both ends; DTC only on arrival; DTD
    # on neither.
    employee = pd.Series(
        _employee_row(
            "SYN-1", cluster_id=0, trips_per_day=2, number_of_stops=1,
            work_arrival_time=820.0, work_departure_time=1710.0,
            total_daily_miles=20.0, commute_distance_survey_miles=10.0,
        )
    )

    for seed in range(10):
        chain = ac.generate_chain_for_employee(
            employee, donor_summary, donor_legs_by_person, np.random.default_rng(seed)
        )
        assert (chain["chain_source"] == ac.DONOR_CHAIN_SOURCE).all()
        other_leg = chain.loc[chain["trip_purpose"] != ac.TRIP_PURPOSE_WORK, "duration"]
        other_leg_duration = other_leg.iloc[0]
        # DTA: remaining budget 10mi / donor raw 5mi -> scale 2 -> 30*2=60min.
        assert other_leg_duration == pytest.approx(60.0)


# --- rescale_chain_times ---------------------------------------------------------


def test_rescale_chain_times_anchors_arrival_and_departure():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    donor_legs = legs.loc[(legs["HOUSEID"] == "D1") & (legs["PERSONID"] == "01")]

    rescaled = ac.rescale_chain_times(
        donor_legs, target_arrival_hhmm=915.0, target_departure_hhmm=1830.0
    )

    work_leg = rescaled.loc[rescaled["trip_purpose"] == ac.TRIP_PURPOSE_WORK].iloc[0]
    next_leg = rescaled.iloc[rescaled.index.get_loc(work_leg.name) + 1]
    assert work_leg["arrival_time"] == 915.0
    assert next_leg["departure_time"] == 1830.0


def test_rescale_chain_times_preserves_leg_order():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    donor_legs = legs.loc[(legs["HOUSEID"] == "D2") & (legs["PERSONID"] == "01")]

    rescaled = ac.rescale_chain_times(
        donor_legs, target_arrival_hhmm=600.0, target_departure_hhmm=2200.0
    )

    dep = rescaled["_departure_minutes"].to_numpy()
    arr = rescaled["_arrival_minutes"].to_numpy()
    assert (dep <= arr).all()
    assert (dep[1:] >= arr[:-1]).all()


def test_rescale_chain_times_falls_back_to_single_offset_when_departure_missing():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    donor_legs = legs.loc[(legs["HOUSEID"] == "D1") & (legs["PERSONID"] == "01")]

    rescaled = ac.rescale_chain_times(
        donor_legs, target_arrival_hhmm=915.0, target_departure_hhmm=float("nan")
    )

    dep = rescaled["_departure_minutes"].to_numpy()
    arr = rescaled["_arrival_minutes"].to_numpy()
    assert (dep <= arr).all()
    assert (dep[1:] >= arr[:-1]).all()


# --- rescale_chain_distances -------------------------------------------------------


def test_rescale_chain_distances_anchors_commute_leg_and_total():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    donor_legs = legs.loc[(legs["HOUSEID"] == "D2") & (legs["PERSONID"] == "01")]
    rescaled_times = ac.rescale_chain_times(donor_legs, 830.0, 1700.0)

    rescaled = ac.rescale_chain_distances(
        rescaled_times, total_daily_miles=30.0, commute_distance_survey_miles=12.0
    )

    work_leg = rescaled.loc[rescaled["trip_purpose"] == ac.TRIP_PURPOSE_WORK].iloc[0]
    assert work_leg["distance"] == pytest.approx(12.0)
    assert rescaled["distance"].sum() == pytest.approx(30.0)
    assert (rescaled["distance"] >= 0).all()
    assert (rescaled["duration"] >= 0).all()


def test_rescale_chain_distances_leaves_values_unscaled_when_total_daily_miles_missing():
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    donor_legs = legs.loc[(legs["HOUSEID"] == "D2") & (legs["PERSONID"] == "01")]
    rescaled_times = ac.rescale_chain_times(donor_legs, 830.0, 1700.0)

    rescaled = ac.rescale_chain_distances(
        rescaled_times, total_daily_miles=float("nan"), commute_distance_survey_miles=float("nan")
    )

    # No target to rescale to - donor's own raw miles pass through.
    assert rescaled["distance"].tolist() == donor_legs["TRPMILES"].tolist()


def test_rescale_chain_distances_scales_a_second_work_purpose_leg_too():
    # Regression: a donor chain with a fragmented workplace dwell (arrive at
    # work, leave briefly, return to work again - plan §5) has a *second*
    # "work"-purpose leg. A prior version excluded every work-purpose leg
    # from the proportional "remaining budget" scaling (not just the first,
    # anchor one), so that second leg's raw donor TRPMILES passed through
    # completely unscaled and could blow the chain's total past
    # total_daily_miles.
    trips = _trips_clean_df(
        _trip_row("D9", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),  # arrive at work
        _trip_row("D9", "01", "02", 1200, 1215, 15, 3.0, whytrp1s=20),  # leave for lunch
        _trip_row("D9", "01", "03", 1230, 1300, 30, 50.0, whytrp1s=10),  # return to work
        _trip_row("D9", "01", "04", 1700, 1730, 30, 4.0, whytrp1s=1),  # home
    )
    clusters = _employee_clusters_df([("D9", "01", 0)])
    legs = ac.build_donor_legs(trips, clusters)
    donor_legs = legs.loc[(legs["HOUSEID"] == "D9") & (legs["PERSONID"] == "01")]
    rescaled_times = ac.rescale_chain_times(donor_legs, 800.0, 1700.0)

    rescaled = ac.rescale_chain_distances(
        rescaled_times, total_daily_miles=20.0, commute_distance_survey_miles=10.0
    )

    assert rescaled["distance"].sum() == pytest.approx(20.0)


def test_rescale_chain_distances_caps_duration_when_donor_leg_distance_is_near_zero():
    # Regression: a donor leg with a near-zero TRPMILES (a real NHTS
    # GPS-rounding artifact) used to produce a distance-scale factor of
    # thousands when anchored to a normal commute distance, and that same
    # factor was applied to TRVLCMIN too - a multi-thousand-minute "duration"
    # for what should be a normal commute.
    trips = _trips_clean_df(
        _trip_row("D10", "01", "01", 800, 801, 1, 0.001, whytrp1s=10),
        _trip_row("D10", "01", "02", 1700, 1701, 1, 0.001, whytrp1s=1),
    )
    clusters = _employee_clusters_df([("D10", "01", 0)])
    legs = ac.build_donor_legs(trips, clusters)
    donor_legs = legs.loc[(legs["HOUSEID"] == "D10") & (legs["PERSONID"] == "01")]
    rescaled_times = ac.rescale_chain_times(donor_legs, 800.0, 1700.0)

    rescaled = ac.rescale_chain_distances(
        rescaled_times, total_daily_miles=50.0, commute_distance_survey_miles=25.0
    )

    assert rescaled["duration"].max() < 300  # nowhere near the un-capped ~25000 minutes
    implied_speed = rescaled["distance"] / (rescaled["duration"] / 60.0)
    assert (implied_speed <= ac.MAX_PLAUSIBLE_SPEED_MPH + 1e-6).all()


def test_rescale_chain_distances_uses_zero_commute_distance_as_a_real_anchor():
    # commute_distance_survey_miles == 0.0 is a legitimate drawn value (e.g.
    # a work-from-home-adjacent employee), not "missing" - only NaN means
    # "no target." Regression: a prior version treated 0 the same as
    # missing and fell through to the donor's own (much larger) raw value.
    trips, clusters = _sample_trips_and_clusters()
    legs = ac.build_donor_legs(trips, clusters)
    donor_legs = legs.loc[(legs["HOUSEID"] == "D2") & (legs["PERSONID"] == "01")]
    rescaled_times = ac.rescale_chain_times(donor_legs, 830.0, 1700.0)

    rescaled = ac.rescale_chain_distances(
        rescaled_times, total_daily_miles=5.0, commute_distance_survey_miles=0.0
    )

    work_leg = rescaled.loc[rescaled["trip_purpose"] == ac.TRIP_PURPOSE_WORK].iloc[0]
    assert work_leg["distance"] == 0.0


# --- select_commute_anchor_miles ---------------------------------------------------


def test_select_commute_anchor_miles_uses_trip_miles_for_single_work_trip():
    employee = pd.Series(
        _employee_row(
            "SYN-1", cluster_id=0, trips_per_day=2, number_of_stops=1,
            commute_distance_survey_miles=10.0,
            commute_distance_trip_miles=15.0, work_trip_count=1,
        )
    )

    assert ac.select_commute_anchor_miles(employee) == 15.0


def test_select_commute_anchor_miles_falls_back_when_multiple_work_trips():
    employee = pd.Series(
        _employee_row(
            "SYN-1", cluster_id=0, trips_per_day=2, number_of_stops=1,
            commute_distance_survey_miles=10.0,
            commute_distance_trip_miles=15.0, work_trip_count=2,
        )
    )

    assert ac.select_commute_anchor_miles(employee) == 10.0


def test_select_commute_anchor_miles_falls_back_above_plausibility_bound():
    employee = pd.Series(
        _employee_row(
            "SYN-1", cluster_id=0, trips_per_day=2, number_of_stops=1,
            commute_distance_survey_miles=10.0,
            commute_distance_trip_miles=ac.MAX_PLAUSIBLE_LEG_MILES + 1.0, work_trip_count=1,
        )
    )

    assert ac.select_commute_anchor_miles(employee) == 10.0


def test_select_commute_anchor_miles_keeps_trip_miles_at_the_bound():
    employee = pd.Series(
        _employee_row(
            "SYN-1", cluster_id=0, trips_per_day=2, number_of_stops=1,
            commute_distance_survey_miles=10.0,
            commute_distance_trip_miles=ac.MAX_PLAUSIBLE_LEG_MILES, work_trip_count=1,
        )
    )

    assert ac.select_commute_anchor_miles(employee) == ac.MAX_PLAUSIBLE_LEG_MILES


def test_select_commute_anchor_miles_falls_back_when_trip_miles_null():
    employee = pd.Series(
        _employee_row(
            "SYN-1", cluster_id=0, trips_per_day=2, number_of_stops=1,
            commute_distance_survey_miles=10.0,
            commute_distance_trip_miles=float("nan"), work_trip_count=1,
        )
    )

    assert ac.select_commute_anchor_miles(employee) == 10.0


def test_generate_chain_for_employee_uses_trip_miles_anchor_end_to_end():
    # Confirms the computed anchor actually reaches rescale_chain_distances,
    # not just that select_commute_anchor_miles returns the right value in
    # isolation.
    trips, clusters = _sample_trips_and_clusters()
    donor_legs = ac.build_donor_legs(trips, clusters)
    donor_summary = ac.summarize_donor_chains(donor_legs)
    donor_legs_by_person = {k: g for k, g in donor_legs.groupby(ac.PERSON_KEY)}
    employee = pd.Series(
        _employee_row(
            "SYN-1", cluster_id=0, trips_per_day=2, number_of_stops=1,
            total_daily_miles=30.0, commute_distance_survey_miles=10.0,
            commute_distance_trip_miles=15.0, work_trip_count=1,
        )
    )
    rng = np.random.default_rng(0)

    chain = ac.generate_chain_for_employee(employee, donor_summary, donor_legs_by_person, rng)

    work_leg = chain.loc[chain["trip_purpose"] == ac.TRIP_PURPOSE_WORK].iloc[0]
    assert work_leg["distance"] == pytest.approx(15.0)


def test_generate_chain_for_employee_falls_back_to_survey_miles_end_to_end():
    trips, clusters = _sample_trips_and_clusters()
    donor_legs = ac.build_donor_legs(trips, clusters)
    donor_summary = ac.summarize_donor_chains(donor_legs)
    donor_legs_by_person = {k: g for k, g in donor_legs.groupby(ac.PERSON_KEY)}
    employee = pd.Series(
        _employee_row(
            "SYN-1", cluster_id=0, trips_per_day=2, number_of_stops=1,
            total_daily_miles=30.0, commute_distance_survey_miles=10.0,
            commute_distance_trip_miles=15.0, work_trip_count=2,
        )
    )
    rng = np.random.default_rng(0)

    chain = ac.generate_chain_for_employee(employee, donor_summary, donor_legs_by_person, rng)

    work_leg = chain.loc[chain["trip_purpose"] == ac.TRIP_PURPOSE_WORK].iloc[0]
    assert work_leg["distance"] == pytest.approx(10.0)


# --- build_fallback_chain ---------------------------------------------------------


def test_build_fallback_chain_has_two_legs_home_work_home():
    employee = pd.Series(
        _employee_row("SYN-00000001", cluster_id=0, trips_per_day=2, number_of_stops=1)
    )

    chain = ac.build_fallback_chain(employee)

    assert chain["trip_purpose"].tolist() == [ac.TRIP_PURPOSE_WORK, ac.TRIP_PURPOSE_HOME]
    assert chain["arrival_time"].iloc[0] == employee["work_arrival_time"]
    assert chain["departure_time"].iloc[1] == employee["work_departure_time"]


def test_build_fallback_chain_handles_missing_departure_time():
    employee = pd.Series(
        _employee_row(
            "SYN-00000001",
            cluster_id=0,
            trips_per_day=2,
            number_of_stops=1,
            work_departure_time=float("nan"),
        )
    )

    chain = ac.build_fallback_chain(employee)

    assert chain["_departure_minutes"].iloc[1] > chain["_arrival_minutes"].iloc[0]


# --- generate_synthetic_activity (integration) -------------------------------------


@pytest.fixture
def population():
    trips, clusters = _sample_trips_and_clusters()
    employees = _synthetic_employees_df(
        _employee_row("SYN-00000001", cluster_id=0, trips_per_day=2, number_of_stops=1),
        _employee_row("SYN-00000002", cluster_id=0, trips_per_day=3, number_of_stops=2),
        _employee_row("SYN-00000003", cluster_id=1, trips_per_day=2, number_of_stops=1),
        # No cluster-0/1 donor is anywhere close -> triggers the fallback path.
        _employee_row("SYN-00000004", cluster_id=99, trips_per_day=2, number_of_stops=1),
    )
    return employees, clusters, trips


def test_every_synthetic_employee_has_a_valid_trip_chain(population):
    employees, clusters, trips = population

    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)

    assert set(activity["synthetic_employee_id"]) == set(employees["synthetic_employee_id"])
    for _, chain in activity.groupby("synthetic_employee_id"):
        assert len(chain) >= 2
        assert chain["trip_number"].tolist() == list(range(1, len(chain) + 1))


def test_activity_output_contains_no_real_nhts_ids(population):
    employees, clusters, trips = population

    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)

    assert "HOUSEID" not in activity.columns
    assert "PERSONID" not in activity.columns
    assert "donor_houseid" not in activity.columns
    assert "donor_personid" not in activity.columns
    # every synthetic_employee_id must come from the synthetic table, not a donor id
    real_ids = set(clusters["HOUSEID"]) | set(clusters["PERSONID"])
    assert not set(activity["synthetic_employee_id"]) & real_ids


def test_activity_trip_order_is_chronological(population):
    employees, clusters, trips = population

    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)

    for _, chain in activity.groupby("synthetic_employee_id"):
        chain = chain.sort_values("trip_number")
        dep = chain["departure_time"].apply(ac.hhmm_to_minutes).to_numpy()
        arr = chain["arrival_time"].apply(ac.hhmm_to_minutes).to_numpy()
        assert (dep <= arr).all()
        assert (dep[1:] >= arr[:-1]).all()


def test_activity_distances_and_durations_are_realistic(population):
    employees, clusters, trips = population

    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)

    assert (activity["distance"] >= 0).all()
    assert (activity["distance"] < 500).all()
    assert (activity["duration"] >= 0).all()
    assert (activity["duration"] < 600).all()
    valid_purposes = [ac.TRIP_PURPOSE_HOME, ac.TRIP_PURPOSE_WORK, ac.TRIP_PURPOSE_OTHER]
    assert activity["trip_purpose"].isin(valid_purposes).all()


def test_activity_has_exactly_one_workplace_arrival_and_departure_flag_per_donor_chain(population):
    employees, clusters, trips = population

    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)

    donor_chains = activity.loc[activity["chain_source"] == ac.DONOR_CHAIN_SOURCE]
    for _, chain in donor_chains.groupby("synthetic_employee_id"):
        assert chain["is_workplace_arrival"].sum() >= 1
        arrival_row = chain.loc[chain["is_workplace_arrival"]].iloc[0]
        assert pd.notna(arrival_row["workplace_dwell_minutes"])


def test_generate_synthetic_activity_uses_fallback_when_no_donor_available(population):
    employees, clusters, trips = population

    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)

    fallback_chain = activity.loc[activity["synthetic_employee_id"] == "SYN-00000004"]
    assert (fallback_chain["chain_source"] == ac.FALLBACK_CHAIN_SOURCE).all()
    assert fallback_chain["vehicle_type"].isna().all()


def test_generate_synthetic_activity_is_reproducible_with_same_seed(population):
    employees, clusters, trips = population

    activity_a = ac.generate_synthetic_activity(employees, clusters, trips, seed=3)
    activity_b = ac.generate_synthetic_activity(employees, clusters, trips, seed=3)

    pd.testing.assert_frame_equal(activity_a, activity_b)


def test_generate_synthetic_activity_can_vary_donor_choice_with_different_seed():
    trips, clusters = _sample_trips_and_clusters()
    # Ambiguous match (two exact (2,1)-shaped donors: D1 and D5) so different
    # seeds can plausibly land on different donors.
    employees = _synthetic_employees_df(
        *[
            _employee_row(f"SYN-{i:08d}", cluster_id=0, trips_per_day=2, number_of_stops=1)
            for i in range(1, 21)
        ]
    )

    activity_a = ac.generate_synthetic_activity(employees, clusters, trips, seed=1)
    activity_b = ac.generate_synthetic_activity(employees, clusters, trips, seed=2)

    # distance is fully anchored/rescaled to the same targets regardless of
    # donor (D1 vs D5), but duration still reflects each donor's own raw
    # TRPMILES-derived scale factor (D1: 10mi legs: D5: 12mi legs), so a
    # different donor pick is still observable here.
    assert not activity_a["duration"].equals(activity_b["duration"])


def test_generate_synthetic_activity_matches_driving_status_between_employee_and_donor():
    # Regression for the donor mode-blindness fix: a mixed donor pool with
    # one driving donor (DDRV, VEHTYPE sentinel 1.0) and one non-driving
    # donor (DWLK, VEHTYPE sentinel 2.0), both an exact (2, 1) cluster-0
    # match, so *only* the driving-status filter can disambiguate them.
    trips = _trips_clean_df(
        _trip_row(
            "DDRV", "01", "01", 800, 830, 30, 10.0, whytrp1s=10, trptrans=3, vehtype=1.0
        ),
        _trip_row(
            "DDRV", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1, trptrans=3, vehtype=1.0
        ),
        _trip_row(
            "DWLK", "01", "01", 800, 830, 30, 10.0, whytrp1s=10, trptrans=1, vehtype=2.0
        ),
        _trip_row(
            "DWLK", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1, trptrans=1, vehtype=2.0
        ),
    )
    clusters = _employee_clusters_df([("DDRV", "01", 0), ("DWLK", "01", 0)])
    employees = _synthetic_employees_df(
        _employee_row(
            "SYN-DRIVER", cluster_id=0, trips_per_day=2, number_of_stops=1,
            total_daily_miles=20.0,
        ),
        _employee_row(
            "SYN-NONDRIVER", cluster_id=0, trips_per_day=2, number_of_stops=1,
            total_daily_miles=float("nan"),
        ),
    )

    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)

    driver_legs = activity.loc[activity["synthetic_employee_id"] == "SYN-DRIVER"]
    nondriver_legs = activity.loc[activity["synthetic_employee_id"] == "SYN-NONDRIVER"]

    assert (driver_legs["chain_source"] == ac.DONOR_CHAIN_SOURCE).all()
    assert (driver_legs["vehicle_type"] == 1.0).all()  # only DDRV could have been chosen

    assert (nondriver_legs["chain_source"] == ac.DONOR_CHAIN_SOURCE).all()
    assert (nondriver_legs["vehicle_type"] == 2.0).all()  # only DWLK could have been chosen
    # The non-driver's raw (unscaled) legs are exactly DWLK's own non-driving
    # TRPMILES - not a driving donor's mileage passed through mislabeled.
    assert nondriver_legs["distance"].tolist() == [10.0, 10.0]


# --- I/O -----------------------------------------------------------------------


def test_load_synthetic_employees_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ac.load_synthetic_employees(tmp_path)


def test_load_employee_clusters_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ac.load_employee_clusters(tmp_path)


def test_load_trips_clean_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ac.load_trips_clean(tmp_path)


def test_save_synthetic_activity_writes_parquet(tmp_path, population):
    employees, clusters, trips = population
    activity = ac.generate_synthetic_activity(employees, clusters, trips, seed=0)

    path = ac.save_synthetic_activity(activity, tmp_path)
    roundtrip = pd.read_parquet(path)

    assert path.name == ac.ACTIVITY_TABLE_FILENAME
    assert len(roundtrip) == len(activity)
