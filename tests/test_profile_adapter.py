"""Tests for driving_profiles.generator.profile_adapter (plan §8.3/§8.4)."""

import numpy as np
import pandas as pd
import pytest

from driving_profiles.generator import profile_adapter as pa

# --- collapse_location ---------------------------------------------------------


def test_collapse_location_home():
    assert (pa.collapse_location(pd.Series(["Home"])) == "home").all()


def test_collapse_location_work():
    assert (pa.collapse_location(pd.Series(["Work"])) == "work").all()


def test_collapse_location_other_categories_collapse_to_other():
    others = pd.Series(["Restaurant", "Medical", "Shopping/Errands", "School", "Gym", "Daycare"])
    assert (pa.collapse_location(others) == "other").all()


def test_collapse_location_driving_sentinel_is_nan():
    result = pa.collapse_location(pd.Series(["-1"]))
    assert result.isna().all()


# --- annotate_chain_segments ----------------------------------------------------


def _legs(*destinations: str, group: str = "U1") -> pd.DataFrame:
    return pd.DataFrame({"group": [group] * len(destinations), "dest": list(destinations)})


def test_annotate_direct_commute_single_work_occurrence():
    # home -> work -> home: one work-occurrence leg (w_1), commute_out = [0],
    # commute_return = [1], both direct (n=1 per segment).
    legs = _legs("work", "home")
    out = pa.annotate_chain_segments(legs, "group", "dest")
    assert out["chain_segment"].tolist() == ["commute_out", "commute_return"]
    assert out["chain_type"].tolist() == ["direct", "direct"]
    assert out["is_arrival_at_work"].tolist() == [True, False]
    assert out["origin_purpose"].tolist() == ["home", "work"]
    assert out["purpose_transition"].tolist() == ["home->work", "work->home"]


def test_annotate_chained_commute_out():
    # home -> other -> work -> home: commute_out has 2 legs (chained), then
    # commute_return has 1 leg (direct).
    legs = _legs("other", "work", "home")
    out = pa.annotate_chain_segments(legs, "group", "dest")
    assert out["chain_segment"].tolist() == ["commute_out", "commute_out", "commute_return"]
    assert out["chain_type"].tolist() == ["chained", "chained", "direct"]
    assert out["leg_index_in_segment"].tolist() == [1, 2, 1]


def test_annotate_midday_segment_includes_return_to_work_leg():
    # home->work (w_1) ->other->work (w_2) ->home: the midday segment is
    # everything strictly after w_1 up to and including w_2 (this module's
    # documented resolution of the "strictly between" boundary ambiguity -
    # see annotate_chain_segments' docstring).
    legs = _legs("work", "other", "work", "home")
    out = pa.annotate_chain_segments(legs, "group", "dest")
    assert out["chain_segment"].tolist() == [
        "commute_out",
        "midday_1",
        "midday_1",
        "commute_return",
    ]
    assert out["chain_type"].tolist() == ["direct", "chained", "chained", "direct"]
    assert out["is_arrival_at_work"].tolist() == [True, False, True, False]


def test_annotate_no_work_leg_falls_back_to_single_commute_out_segment():
    legs = _legs("other", "home")
    out = pa.annotate_chain_segments(legs, "group", "dest")
    assert out["chain_segment"].tolist() == ["commute_out", "commute_out"]
    assert not out["is_arrival_at_work"].any()


def test_annotate_handles_multiple_independent_groups():
    legs = pd.concat(
        [_legs("work", "home", group="U1"), _legs("other", "work", "home", group="U2")],
        ignore_index=True,
    )
    out = pa.annotate_chain_segments(legs, "group", "dest")
    u1 = out.loc[out["group"] == "U1"]
    u2 = out.loc[out["group"] == "U2"]
    assert u1["chain_segment"].tolist() == ["commute_out", "commute_return"]
    assert u2["chain_segment"].tolist() == ["commute_out", "commute_out", "commute_return"]
    # origin_purpose resets to "home" independently per group, not carried
    # over from the previous group's last leg.
    assert u2["origin_purpose"].iloc[0] == "home"


# --- load_driver_profiles / select_profile_user_ids -----------------------------


def _write_driver_profiles_csv(tmp_path) -> "Path":  # noqa: F821
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


def test_load_driver_profiles_columns_and_row_index(tmp_path):
    path = _write_driver_profiles_csv(tmp_path)
    df = pa.load_driver_profiles(path)
    assert list(df.loc[df["user_id"] == 1, "row_index"]) == [0, 1, 2, 3, 4]
    assert df.loc[0, "start_min"] == 0.0
    assert df.loc[0, "end_min"] == pytest.approx(480.0)
    assert df.loc[0, "collapsed_location"] == "home"
    assert pd.isna(df.loc[1, "collapsed_location"])


def test_load_driver_profiles_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pa.load_driver_profiles(tmp_path / "does_not_exist.csv")


def test_select_profile_user_ids_returns_all_when_n_at_or_above_population(tmp_path):
    path = _write_driver_profiles_csv(tmp_path)
    df = pa.load_driver_profiles(path)
    assert pa.select_profile_user_ids(df, 250) == [1, 2]
    assert pa.select_profile_user_ids(df, 2) == [1, 2]


def test_select_profile_user_ids_seeded_subset_is_reproducible(tmp_path):
    path = _write_driver_profiles_csv(tmp_path)
    df = pa.load_driver_profiles(path)
    a = pa.select_profile_user_ids(df, 1, seed=7)
    b = pa.select_profile_user_ids(df, 1, seed=7)
    assert a == b
    assert len(a) == 1
    assert a[0] in (1, 2)


# --- build_external_driving_legs -------------------------------------------------


def test_build_external_driving_legs_destination_from_following_window(tmp_path):
    path = _write_driver_profiles_csv(tmp_path)
    df = pa.load_driver_profiles(path)
    legs = pa.build_external_driving_legs(df)

    u1 = legs.loc[legs["user_id"] == 1].sort_values("row_index")
    assert u1["destination_purpose"].tolist() == ["work", "home"]
    assert u1["is_arrival_at_work"].tolist() == [True, False]
    assert u1["chain_segment"].tolist() == ["commute_out", "commute_return"]


# --- build_donor_leg_pool / match_donor_leg --------------------------------------


def _trip_row(
    house_id, person_id, trip_id, strttime, endtime, trvlcmin, trpmiles, whytrp1s, trptrans=3
):
    return {
        "HOUSEID": house_id,
        "PERSONID": person_id,
        "TRIPID": trip_id,
        "LOOP_TRIP": 2,
        "STRTTIME": strttime,
        "ENDTIME": endtime,
        "TRVLCMIN": trvlcmin,
        "TRPMILES": trpmiles,
        "WHYTRP1S": whytrp1s,
        "TRPTRANS": trptrans,
        "VEHTYPE": 1.0,
        "VEHFUEL": 1.0,
    }


def _trips_clean_df(*rows):
    df = pd.DataFrame(list(rows))
    for col in ("HOUSEID", "PERSONID", "TRIPID"):
        df[col] = df[col].astype(str)
    return df


def _employee_clusters_df(rows):
    df = pd.DataFrame(rows, columns=["HOUSEID", "PERSONID", "cluster_id"])
    df["HOUSEID"] = df["HOUSEID"].astype(str)
    df["PERSONID"] = df["PERSONID"].astype(str)
    df["cluster_id"] = pd.array(df["cluster_id"], dtype="Int64")
    return df


def test_build_donor_leg_pool_filters_to_driving_and_plausible_speed():
    # D1: both legs a plausible 30 mph. D2: leg 1 is implausibly fast (300
    # mph, excluded), leg 2 is a plausible 30 mph (kept) - filtering is
    # per-leg (§8.4), not per-donor, so D2 contributes exactly one leg.
    trips = _trips_clean_df(
        _trip_row("D1", "01", "01", 800, 830, 30, 15.0, whytrp1s=10),  # 30 mph
        _trip_row("D1", "01", "02", 1700, 1730, 30, 15.0, whytrp1s=1),  # 30 mph
        _trip_row("D2", "01", "01", 800, 801, 1, 5.0, whytrp1s=10),  # 300 mph
        _trip_row("D2", "01", "02", 1700, 1730, 30, 15.0, whytrp1s=1),  # 30 mph
    )
    clusters = _employee_clusters_df([("D1", "01", 0), ("D2", "01", 0)])
    pool = pa.build_donor_leg_pool(trips, clusters)
    assert set(pool["HOUSEID"]) == {"D1", "D2"}
    assert len(pool.loc[pool["HOUSEID"] == "D2"]) == 1
    assert pool.loc[pool["HOUSEID"] == "D2", "TRIPID"].iloc[0] == "02"
    assert "chain_segment" in pool.columns
    assert "purpose_transition" in pool.columns
    assert (pool["is_driving_leg"]).all()


def _pool_leg(
    purpose_transition,
    chain_segment,
    chain_type,
    destination_purpose,
    start_min,
    trpmiles=10.0,
    trvlcmin=20.0,
    house="H",
    person="P",
    trip="1",
):
    return {
        "HOUSEID": house,
        "PERSONID": person,
        "TRIPID": trip,
        "purpose_transition": purpose_transition,
        "chain_segment": chain_segment,
        "chain_type": chain_type,
        "destination_purpose": destination_purpose,
        "start_min": start_min,
        "TRPMILES": trpmiles,
        "TRVLCMIN": trvlcmin,
    }


def _external_leg(purpose_transition, chain_segment, chain_type, destination_purpose, start_min):
    return pd.Series(
        {
            "purpose_transition": purpose_transition,
            "chain_segment": chain_segment,
            "chain_type": chain_type,
            "destination_purpose": destination_purpose,
            "start_min": start_min,
        }
    )


POOL_COLUMNS = [
    "HOUSEID",
    "PERSONID",
    "TRIPID",
    "purpose_transition",
    "chain_segment",
    "chain_type",
    "destination_purpose",
    "start_min",
    "TRPMILES",
    "TRVLCMIN",
]


def test_match_donor_leg_prefers_tier_1a_within_60_minutes():
    pool = pd.DataFrame(
        [
            _pool_leg("home->work", "commute_out", "direct", "work", 500, house="H1"),
            _pool_leg("home->work", "commute_out", "direct", "work", 1000, house="H2"),
        ]
    )
    leg = _external_leg("home->work", "commute_out", "direct", "work", start_min=480)
    rng = np.random.default_rng(0)
    donor, tier = pa.match_donor_leg(leg, pool, rng)
    assert tier == pa.MATCH_TIER_1A
    assert donor["HOUSEID"] == "H1"


def test_match_donor_leg_widens_to_tier_1c_when_no_time_match():
    pool = pd.DataFrame(
        [_pool_leg("home->work", "commute_out", "direct", "work", 1000, house="H1")]
    )
    leg = _external_leg("home->work", "commute_out", "direct", "work", start_min=480)
    rng = np.random.default_rng(0)
    donor, tier = pa.match_donor_leg(leg, pool, rng)
    assert tier == pa.MATCH_TIER_1C
    assert donor["HOUSEID"] == "H1"


def test_match_donor_leg_falls_to_tier_2_when_chain_type_differs():
    pool = pd.DataFrame(
        [_pool_leg("home->work", "commute_out", "chained", "work", 480, house="H1")]
    )
    leg = _external_leg("home->work", "commute_out", "direct", "work", start_min=480)
    rng = np.random.default_rng(0)
    donor, tier = pa.match_donor_leg(leg, pool, rng)
    assert tier == pa.MATCH_TIER_2


def test_match_donor_leg_falls_to_tier_3_when_chain_segment_differs():
    pool = pd.DataFrame([_pool_leg("home->work", "midday_1", "direct", "work", 480, house="H1")])
    leg = _external_leg("home->work", "commute_out", "direct", "work", start_min=480)
    rng = np.random.default_rng(0)
    donor, tier = pa.match_donor_leg(leg, pool, rng)
    assert tier == pa.MATCH_TIER_3


def test_match_donor_leg_falls_to_tier_4_when_transition_differs():
    pool = pd.DataFrame([_pool_leg("other->work", "midday_1", "direct", "work", 480, house="H1")])
    leg = _external_leg("home->work", "commute_out", "direct", "work", start_min=480)
    rng = np.random.default_rng(0)
    donor, tier = pa.match_donor_leg(leg, pool, rng)
    assert tier == pa.MATCH_TIER_4


def test_match_donor_leg_unrepaired_when_pool_empty():
    pool = pd.DataFrame(columns=POOL_COLUMNS)
    leg = _external_leg("home->work", "commute_out", "direct", "work", start_min=480)
    rng = np.random.default_rng(0)
    donor, tier = pa.match_donor_leg(leg, pool, rng)
    assert donor is None
    assert tier == pa.MATCH_TIER_UNREPAIRED


def test_match_donor_leg_seeded_draw_is_reproducible():
    pool = pd.DataFrame(
        [
            _pool_leg("home->work", "commute_out", "direct", "work", 480, house="H1"),
            _pool_leg("home->work", "commute_out", "direct", "work", 480, house="H2"),
            _pool_leg("home->work", "commute_out", "direct", "work", 480, house="H3"),
        ]
    )
    leg = _external_leg("home->work", "commute_out", "direct", "work", start_min=480)
    donor_a, _ = pa.match_donor_leg(leg, pool, np.random.default_rng(123))
    donor_b, _ = pa.match_donor_leg(leg, pool, np.random.default_rng(123))
    assert donor_a["HOUSEID"] == donor_b["HOUSEID"]
