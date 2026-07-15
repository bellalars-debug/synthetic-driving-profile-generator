# ev-tool charging integration (final stage)

Adds a **station/queue charging-simulation** backend to the end of the pipeline,
so the synthetic activity profiles this repo produces can be turned into
workplace EV charging estimates that account for **discrete L2/L3 stations, a
first-come queue, station contention, and waiting time**.

This is an *alternative* to `src/driving_profiles/scenarios/charging_demand.py`
(which applies an idealized unmanaged-immediate energy model). Both read the same
finalized artifacts; neither modifies the generator.

## Data flow

```
data/processed/synthetic_activity.parquet      (generator/activity.py output)
data/processed/synthetic_employees.parquet     (generator/sample.py output)
        │
        ▼   src/driving_profiles/scenarios/ev_tool_charging.py
reconstruct each employee's day  →  pov_driving_pattern.json   (ev-tool schema)
        │
        ▼   third_party/ev_infrastructure_tool/charging_backend  (MIT, vendored)
run_charging_management(...)  →  L2/L3 station+queue simulation
        │
        ▼
ev_tool_vehicle_status_{rate}.csv  +  ev_tool_summary.json
```

## Schema mapping (this repo → ev-tool)

| this pipeline (`synthetic_activity.parquet`) | ev-tool `pov_driving_pattern.json` |
|---|---|
| one row per **trip leg** (`trip_number` ordered) | per-vehicle daily `activities` list |
| `departure_time` / `arrival_time` (HHMM) | `start_time` / `end_time` (`HH:MM`), via `hhmm_to_minutes` |
| `trip_purpose == "work"` | parked `location = "On-Site"` (workplace charging happens here) |
| `trip_purpose == "home"` | parked `location = "Home"` |
| `trip_purpose == "other"` | parked `location = "Off-Site"` (stop; no charging) |
| driving between legs | `activity_type = "Driving"`, `location = "Off-Site"`, `driving_distance` (mi) |
| `distance` (mi) | `distance.value` (m) for ranking/eligibility |
| one representative day | replicated across `days_drive` weekdays; weekends idle |

Energy fields (`equivalent electricity kWh`, consumption rate, mpge) use the
ev-tool's own Light-Duty formulas so the physics matches the vendored simulator.
Trip chains are preserved: a School→Work→Shopping→Home day becomes extra
`Off-Site` legs, which changes the state-of-charge on arrival at work — richer
than the ev-tool's own generator, which only makes direct Home→Work→Home days.

## Run it

After `scripts/run_pipeline.py` has produced the activity profiles:

```bash
python scripts/run_ev_tool_charging.py \
  --activity data/processed/synthetic_activity.parquet \
  --employees data/processed/synthetic_employees.parquet \
  --adoption-rate 0.36 --run-period 30
```

Or call `driving_profiles.scenarios.ev_tool_charging.estimate_charging(...)`.
`scripts/demo_ev_tool_charging.py` runs it end-to-end on a schema-accurate
fixture (no pipeline data required) and `tests/test_ev_tool_charging.py` is the
smoke test.

## Output

- `pov_driving_pattern.json` — the fleet handed to the simulator
- `ev_tool_vehicle_status_{rate}.csv` — per-vehicle 15-min status (SOC, location,
  charging station, status, queue) across a sweep of L2 station counts
- `ev_tool_summary.json` — vehicles, EVs selected at the adoption rate, the L2
  station-count scenarios, and peak simultaneous charging per scenario

## Attribution

The charging simulator is Rongxin Yin's **ev-infrastructure-tool** (MIT), vendored
unmodified under `third_party/ev_infrastructure_tool/`. Only the driving-pattern
adapter and the runner in this repo are new. No Google Maps API key is needed —
this pipeline supplies the driving patterns itself.
