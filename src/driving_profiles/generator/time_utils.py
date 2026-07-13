"""Shared NHTS-style HHMM <-> minutes-since-midnight conversion helpers.

Centralized here (rather than living in `generator/activity.py`) so
`generator/sample.py` can also convert to/from true minutes-since-midnight
when jittering time-of-day fields, without an import cycle: `activity.py`
already imports `sample.py` (for `SYNTHETIC_EMPLOYEE_FILENAME` etc.), so
`sample.py` cannot import back from `activity.py`.
"""

from __future__ import annotations

import pandas as pd

MINUTES_PER_DAY = 24 * 60


def hhmm_to_minutes(hhmm: float) -> float:
    """Convert an NHTS-style HHMM-encoded time-of-day value to minutes since
    midnight.

    Handles jittered inputs whose "minutes" component isn't in [0, 60) -
    real in this project's `work_arrival_time`/`work_departure_time` prior
    to `generator/sample.py`'s minutes-space jitter fix, and still possible
    from arbitrary caller input - by treating the hundreds-and-up digits as
    hours and the remainder as minutes literally: `880` decodes as hour=8,
    minute=80 -> 560 minutes (9:20am), not as an error. `minutes_to_hhmm` is
    this function's exact inverse and always re-encodes into a valid
    (minute in [0, 60)) HHMM value.
    """
    if pd.isna(hhmm):
        return float("nan")
    hours, minutes = divmod(float(hhmm), 100)
    return hours * 60 + minutes


def minutes_to_hhmm(minutes: float) -> float:
    """Inverse of `hhmm_to_minutes`: minutes since midnight -> HHMM, always
    with a valid (< 60) minute component.

    Clips (never wraps) to [0, MINUTES_PER_DAY - 1) - clipping is monotonic
    non-decreasing, so it can never reorder two already-ordered timestamps,
    unlike a modulo wraparound which could place a late offset-shifted leg
    before an earlier one.
    """
    if pd.isna(minutes):
        return float("nan")
    minutes = min(max(float(minutes), 0.0), MINUTES_PER_DAY - 1)
    hours, mins = divmod(minutes, 60)
    return hours * 100 + mins
