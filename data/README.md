# Data directory

Raw and derived data are **not** committed to version control. This
directory documents how to obtain and regenerate them.

## `raw/`
Original NHTS 2022 public-use files, populated by running
`python -m driving_profiles.data.download` (see
`src/driving_profiles/data/download.py`). That run also writes
`manifest.json` alongside the CSVs, recording the source URL, sha256 of
the downloaded zip, its `Last-Modified` header, and the extracted
filenames — so the extract is traceable from a fresh clone.

## `interim/`
Cleaned and joined intermediate tables (output of `data/clean.py`).

## `processed/`
Feature tables ready for clustering and profile generation (output of
`features/build_features.py`).
