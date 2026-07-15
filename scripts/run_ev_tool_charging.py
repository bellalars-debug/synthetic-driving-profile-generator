"""Final pipeline stage (CLI): synthetic activity profiles -> ev-tool charging
estimates.

Runs AFTER `scripts/run_pipeline.py` has produced the synthetic activity output.
Uses the ev-infrastructure-tool station/queue simulator (vendored, MIT) as an
alternative to the built-in `scenarios/charging_demand.py` scenario model.

    python scripts/run_ev_tool_charging.py \
        --activity data/processed/synthetic_activity.parquet \
        --employees data/processed/synthetic_employees.parquet \
        --adoption-rate 0.36 --run-period 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from driving_profiles.scenarios import ev_tool_charging as ev  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activity", default="data/processed/synthetic_activity.parquet")
    ap.add_argument("--employees", default="data/processed/synthetic_employees.parquet")
    ap.add_argument("--out-dir", default="reports/xlsx/charging_demand/ev_tool")
    ap.add_argument("--site-id", default="bldg-90")
    ap.add_argument("--adoption-rate", type=float, default=0.36)
    ap.add_argument("--run-period", type=int, default=30)
    ap.add_argument("--l2-rate", type=float, default=7.0)
    ap.add_argument("--l3-rate", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not Path(args.activity).exists():
        sys.exit(f"Activity profiles not found at {args.activity}. "
                 f"Run scripts/run_pipeline.py first to generate them.")

    emp = args.employees if Path(args.employees).exists() else None
    summary = ev.estimate_charging(
        args.activity, emp, out_dir=args.out_dir, site_id=args.site_id,
        adoption_rate=args.adoption_rate, run_period_days=args.run_period,
        l2_max_rate_kw=args.l2_rate, l3_max_rate_kw=args.l3_rate, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
