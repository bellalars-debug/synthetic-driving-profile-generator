# Workplace EV Charging Demand Estimator (web app)

A four-page dashboard with a tree sidebar. Enter a **company headcount** on the
Site page and it runs the whole pipeline, populating:

- **Site** — type, location, employees vs. parking spaces
- **Transportation** — commute-mode split, driving clusters (archetypes),
  sampled driving profiles (24h activity timelines), driving characteristics
- **Vehicle Electrification** — EV adoption, fuel mix, chargers by station type
- **Infrastructure** — design-day metrics, load curve, suggested EV infrastructure

No employee data, Google Maps, or parking-lot detection required.

```
headcount (+ EV-adoption slider)
  → synthetic NHTS workforce            (nhts_generator/, fast in-process generator)
  → per-driver daily activity profiles
  → ev-tool station/queue charging sim  (ev_charging_sim/, MIT, vendored)
  → chargers needed · peak kW · kWh/day · 24h load curve
```

This is the responsive demo front-end. It is self-contained and uses the fast
`lbnl_sim` generator bundled in `nhts_generator/` (the same NHTS foundation as
the main pipeline), so a result returns in seconds. The heavier, full-fidelity
path is the repo's own pipeline (`src/driving_profiles/…`) feeding
`scenarios/ev_tool_charging.py`.

## Run it

```bash
cd webapp
pip install -r requirements.txt
python app.py                 # -> http://127.0.0.1:5001
# share on your LAN:   HOST=0.0.0.0 PORT=5001 python app.py
```

## Files

```
webapp/
  app.py               Flask server (serves the UI, exposes /api/estimate)
  index.html           single-page UI + canvas load-curve chart (no CDN)
  pipeline.py          headcount -> synthetic drivers -> ev-tool sim -> summary
  requirements.txt
  ev_charging_sim/     vendored ev-infrastructure-tool charging simulator (MIT, R. Yin)
  nhts_generator/      bundled NHTS synthetic-driver generator + probability data
```

## What the numbers mean

- **Chargers recommended** — peak number of vehicles charging at once on a design
  day (so nobody waits). All workplace charging is Level-2 (7 kW).
- **Peak charging power** — the highest instantaneous workplace draw (kW).
- **EV drivers** — share of the headcount driving electric, at the adoption rate.
- **Total EV energy/day** — workplace + overnight home charging (kWh).

Drivers with home Level-2 charging arrive well-charged; those without top up at
work, which creates the mid-morning peak. Input is capped at 5,000 drivers to
keep the web tool responsive — the per-vehicle simulation is real and its cost
grows super-linearly with fleet size.

## Attribution

`ev_charging_sim/` is Rongxin Yin's **ev-infrastructure-tool** (MIT), vendored
unmodified. See its `LICENSE`.
