
#!/usr/bin/env python3
"""Run the drone racing simulation multiple times and report stats.

Usage:
    python run_sim_stats.py --config level2.toml --n-runs 20 --render 0
    # Optional:
    #   --controller <controller.py> to override the controller from the config
    #   --render 1 to enable rendering

This script expects to be placed in the same directory as `sim.py`.
"""

import argparse
import math
from typing import List, Optional

# Import the simulate() function from sim.py (assumed to be in the same folder)
try:
    from sim import simulate
except ModuleNotFoundError as e:
    raise SystemExit("Could not import simulate from sim.py. Place this script next to sim.py.") from e


def main():
    parser = argparse.ArgumentParser(description="Run simulation and report statistics.")
    parser.add_argument("--config", type=str, default="level2.toml", help="Config filename under config/.")
    parser.add_argument("--controller", type=str, default=None, help="Override controller filename (optional).")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of episodes to run.")
    parser.add_argument("--render", type=int, choices=[0,1], default=None,
                        help="1 to enable rendering, 0 to disable. Defaults to config value if omitted.")

    args = parser.parse_args()
    render: Optional[bool] = None if args.render is None else bool(args.render)

    # Run the simulation; simulate() returns a list where each element is the
    # completion time (float seconds) on success, or None on failure.
    ep_times: List[Optional[float]] = simulate(
        config=args.config,
        controller=args.controller,
        n_runs=args.n_runs,
        render=render,
    )

    # Compute statistics
    successes = [t for t in ep_times if t is not None]
    success_count = len(successes)
    fail_count = args.n_runs - success_count
    success_rate = success_count / args.n_runs if args.n_runs > 0 else 0.0
    avg_time = (sum(successes) / success_count) if success_count > 0 else math.nan

    # Report per-run results
    print("\nPer-run results:")
    for i, t in enumerate(ep_times, start=1):
        if t is None:
            print(f"Run {i:>3}: FAIL")
        else:
            print(f"Run {i:>3}: SUCCESS in {t:.3f} s")

    # Summary
    print("\nSummary:")
    print(f"  Total runs:     {args.n_runs}")
    print(f"  Successes:      {success_count}")
    print(f"  Failures:       {fail_count}")
    print(f"  Success rate:   {success_rate:.2%}")
    if success_count > 0:
        print(f"  Avg time (s):   {avg_time:.3f}")
    else:
        print("  Avg time (s):   n/a (no successful runs)")


if __name__ == "__main__":
    main()
