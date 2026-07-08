# Data directory

Raw and derived data are **not** committed to version control. This
directory documents how to obtain and regenerate them.

## `raw/`
Original NHTS 2022 public-use files.
TODO: document exact extract/version used, download source, and any
checksums, so results are reproducible from a fresh clone.

## `interim/`
Cleaned and joined intermediate tables (output of `data/clean.py`).

## `processed/`
Feature tables ready for clustering and profile generation (output of
`features/build_features.py`).
