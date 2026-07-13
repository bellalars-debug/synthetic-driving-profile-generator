"""Tests for driving_profiles.generator.time_utils (hhmm_to_minutes/
minutes_to_hhmm), and that generator/activity.py and generator/sample.py
both re-export the same shared objects rather than duplicating them.
"""

import pandas as pd

from driving_profiles.generator import time_utils as tu


def test_hhmm_to_minutes_standard_value():
    assert tu.hhmm_to_minutes(830.0) == 8 * 60 + 30


def test_hhmm_to_minutes_handles_invalid_minute_digit_from_jitter():
    # 880 isn't a valid clock reading ("8:80") but must still decode
    # deterministically rather than error.
    assert tu.hhmm_to_minutes(880.0) == 8 * 60 + 80


def test_hhmm_to_minutes_nan_passthrough():
    assert pd.isna(tu.hhmm_to_minutes(float("nan")))


def test_minutes_to_hhmm_round_trips_standard_values():
    for hhmm in (0.0, 830.0, 1259.0, 2359.0):
        minutes = tu.hhmm_to_minutes(hhmm)
        assert tu.minutes_to_hhmm(minutes) == hhmm


def test_minutes_to_hhmm_always_produces_valid_minute_component():
    hhmm = tu.minutes_to_hhmm(tu.hhmm_to_minutes(880.0))
    assert hhmm % 100 < 60


def test_minutes_to_hhmm_clips_rather_than_wraps():
    assert tu.minutes_to_hhmm(-10) == 0.0
    assert tu.minutes_to_hhmm(tu.MINUTES_PER_DAY + 100) == tu.minutes_to_hhmm(
        tu.MINUTES_PER_DAY - 1
    )


def test_activity_module_reexports_the_same_objects():
    from driving_profiles.generator import activity as ac

    assert ac.hhmm_to_minutes is tu.hhmm_to_minutes
    assert ac.minutes_to_hhmm is tu.minutes_to_hhmm
    assert ac.MINUTES_PER_DAY == tu.MINUTES_PER_DAY


def test_sample_module_reexports_the_same_objects():
    from driving_profiles.generator import sample as sm

    assert sm.hhmm_to_minutes is tu.hhmm_to_minutes
    assert sm.minutes_to_hhmm is tu.minutes_to_hhmm
    assert sm.MINUTES_PER_DAY == tu.MINUTES_PER_DAY
