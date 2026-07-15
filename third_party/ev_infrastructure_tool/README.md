# Vendored: ev-infrastructure-tool (charging backend)

`charging_backend/` is a vendored subset of the **ev-infrastructure-tool** by
**Rongxin Yin** — specifically the Python charging-simulation core from
`server/python-backend/scripts/`. It is used as the final, station/queue-based
charging-estimation backend of this pipeline (see
`src/driving_profiles/scenarios/ev_tool_charging.py`).

- **Upstream:** https://github.com/rongxinyin/ev-infrastructure-tool
- **License:** MIT (© 2024 rongxinyin) — see `LICENSE` in this folder.
- **What it does:** given a `pov_driving_pattern.json` fleet, it simulates
  discrete L2/L3 charging stations with a first-come queue over a multi-day
  horizon, tracking state-of-charge, station assignment, and waiting time.

## What was vendored (unmodified)

```
charging_backend/
  models/         vehicle.py, charging_station.py     (EV + station state models)
  utilities/      helpers.py, queue.py, activity.py, pattern_generator.py
  standalone-pov-charging-management.py   run_charging_management(...) entry point
  requirements.txt
```

The code is **unmodified** from upstream. Only the React client, Node server,
Google-Maps employee-data ingestion, and post-processing scripts were left out —
this pipeline supplies the driving patterns itself (from NHTS-derived synthetic
activity profiles), so no Google Maps API key is required.

The adapter that converts this repo's `synthetic_activity.parquet` into the
`pov_driving_pattern.json` schema and calls `run_charging_management` lives at
`src/driving_profiles/scenarios/ev_tool_charging.py`.
