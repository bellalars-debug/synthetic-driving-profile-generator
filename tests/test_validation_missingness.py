"""Tests for driving_profiles.validation.missingness."""

import pandas as pd
import pytest

from driving_profiles.validation import missingness as mv


def _source_population():
    return pd.DataFrame(
        {
            "HOUSEID": ["H1", "H2", "H3", "H4", "H5", "H6"],
            "PERSONID": ["01"] * 6,
            "cluster_id": pd.array([0, 0, 0, 1, 1, 1], dtype="Int64"),
            "total_daily_miles": [10.0, float("nan"), 20.0, float("nan"), 30.0, float("nan")],
            "total_driving_minutes": [20.0, float("nan"), 40.0, float("nan"), 60.0, float("nan")],
            "average_trip_distance_miles": [
                5.0, float("nan"), 10.0, float("nan"), 15.0, float("nan"),
            ],
        }
    )


def _synthetic_from_source(source: pd.DataFrame) -> pd.DataFrame:
    synthetic = source.rename(
        columns={"HOUSEID": "source_houseid", "PERSONID": "source_personid"}
    ).copy()
    synthetic["synthetic_employee_id"] = [f"SYN-{i}" for i in range(len(synthetic))]
    synthetic["trips_per_day"] = 2
    synthetic["number_of_stops"] = 1
    return synthetic


# --- validate_pooled_missingness_rate -----------------------------------------------


def test_validate_pooled_missingness_rate_passes_when_matched():
    source = _source_population()
    synthetic = _synthetic_from_source(source)

    result = mv.validate_pooled_missingness_rate(source, synthetic)

    assert bool(result.iloc[0]["passed"]) is True
    assert result.iloc[0]["statistic"] == pytest.approx(0.0)


def test_validate_pooled_missingness_rate_fails_when_drifted():
    source = _source_population()
    synthetic = _synthetic_from_source(source)
    synthetic["total_daily_miles"] = 10.0  # no missingness at all

    result = mv.validate_pooled_missingness_rate(source, synthetic, max_diff_pp=3.0)

    assert bool(result.iloc[0]["passed"]) is False

# --- validate_per_cluster_missingness_rate ------------------------------------------


def test_validate_per_cluster_missingness_rate_returns_one_row_per_cluster():
    source = _source_population()
    synthetic = _synthetic_from_source(source)

    result = mv.validate_per_cluster_missingness_rate(source, synthetic)

    assert set(result["group"]) == {"cluster_0", "cluster_1"}


# --- validate_missingness_cooccurrence ----------------------------------------------


def test_validate_missingness_cooccurrence_passes_when_fully_paired():
    source = _source_population()

    result = mv.validate_missingness_cooccurrence(source, "source")

    assert bool(result.iloc[0]["passed"]) is True
    assert "partial=0" in result.iloc[0]["detail"]


def test_validate_missingness_cooccurrence_fails_on_partial_null_row():
    df = _source_population()
    df.loc[0, "total_daily_miles"] = float("nan")  # now only 1/3 columns null on row 0

    result = mv.validate_missingness_cooccurrence(df, "source")

    assert bool(result.iloc[0]["passed"]) is False

# --- validate_jitter_preserves_nan --------------------------------------------------


def test_validate_jitter_preserves_nan_passes_when_preserved():
    source = _source_population()
    synthetic = _synthetic_from_source(source)

    result = mv.validate_jitter_preserves_nan(source, synthetic)

    assert bool(result.iloc[0]["passed"]) is True

def test_validate_jitter_preserves_nan_fails_when_a_null_was_filled_in():
    source = _source_population()
    synthetic = _synthetic_from_source(source)
    # H2's source total_daily_miles is NaN; synthetic row now has a value.
    synthetic.loc[synthetic["source_houseid"] == "H2", "total_daily_miles"] = 99.0

    result = mv.validate_jitter_preserves_nan(source, synthetic)

    assert bool(result.iloc[0]["passed"]) is False
    assert result.iloc[0]["statistic"] >= 1.0


# --- estimate_donor_mode_blindness_rate ---------------------------------------------


def _trip_row(
    house_id, person_id, trip_id, strttime, endtime, trvlcmin, trpmiles, whytrp1s, loop_trip=2
):
    return {
        "HOUSEID": house_id, "PERSONID": person_id, "TRIPID": trip_id,
        "LOOP_TRIP": loop_trip, "STRTTIME": strttime, "ENDTIME": endtime,
        "TRVLCMIN": trvlcmin, "TRPMILES": trpmiles, "WHYTRP1S": whytrp1s,
        "VEHTYPE": 1.0, "VEHFUEL": 1.0,
    }


def _trips_clean_df(*rows):
    df = pd.DataFrame(list(rows))
    for col in ("HOUSEID", "PERSONID", "TRIPID"):
        df[col] = df[col].astype(str)
    return df


def test_estimate_donor_mode_blindness_rate_detects_null_donor_candidates():
    # Donor pool: D1 drove (total_daily_miles not null), D2 did not drive
    # that day (total_daily_miles null) - both same (2,1) chain shape in
    # cluster 0, so a null-total_daily_miles synthetic employee matching
    # that shape has a 50% chance of landing on the non-driving donor D2.
    trips = _trips_clean_df(
        _trip_row("D1", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("D1", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
        _trip_row("D2", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("D2", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
    )
    employee_clusters = pd.DataFrame(
        {
            "HOUSEID": ["D1", "D2"],
            "PERSONID": ["01", "01"],
            "cluster_id": pd.array([0, 0], dtype="Int64"),
            "total_daily_miles": [20.0, float("nan")],
        }
    )
    synthetic_employees = pd.DataFrame(
        [
            {
                "synthetic_employee_id": "SYN-1",
                "cluster_id": pd.array([0], dtype="Int64")[0],
                "trips_per_day": 2,
                "number_of_stops": 1,
                "total_daily_miles": float("nan"),
            }
        ]
    )

    result = mv.estimate_donor_mode_blindness_rate(synthetic_employees, employee_clusters, trips)

    row = result.iloc[0]
    assert row["statistic"] == pytest.approx(0.5)
    assert row["passed"] is None  # diagnostic, not a gate


def test_estimate_donor_mode_blindness_rate_handles_no_null_employees():
    trips = _trips_clean_df(
        _trip_row("D1", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("D1", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
    )
    employee_clusters = pd.DataFrame(
        {
            "HOUSEID": ["D1"],
            "PERSONID": ["01"],
            "cluster_id": pd.array([0], dtype="Int64"),
            "total_daily_miles": [20.0],
        }
    )
    synthetic_employees = pd.DataFrame(
        [
            {
                "synthetic_employee_id": "SYN-1",
                "cluster_id": pd.array([0], dtype="Int64")[0],
                "trips_per_day": 2,
                "number_of_stops": 1,
                "total_daily_miles": 15.0,
            }
        ]
    )

    result = mv.estimate_donor_mode_blindness_rate(synthetic_employees, employee_clusters, trips)

    assert pd.isna(result.iloc[0]["statistic"])
    assert result.iloc[0]["n_synthetic"] == 0


# --- run_missingness_validation -------------------------------------------------------


def test_run_missingness_validation_combines_every_check():
    source = _source_population()
    synthetic = _synthetic_from_source(source)
    trips = _trips_clean_df(
        _trip_row("H1", "01", "01", 800, 830, 30, 10.0, whytrp1s=10),
        _trip_row("H1", "01", "02", 1700, 1730, 30, 10.0, whytrp1s=1),
    )
    employee_clusters = source.copy()

    result = mv.run_missingness_validation(source, synthetic, employee_clusters, trips)

    assert result["section"].eq("missingness").all()
    assert "missingness_rate" in result["metric"].tolist()
    assert "missingness_cooccurrence" in result["metric"].tolist()
    assert "jitter_preserves_source_nan" in result["metric"].tolist()
    assert "donor_mode_blindness_case2_rate_estimate" in result["metric"].tolist()
