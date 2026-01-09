#!/usr/bin/env python3
"""Helper script to find and display the latest MPC diagnostic plots."""

import glob
import subprocess
from pathlib import Path


def find_latest_diagnostics():
    """Find the most recent mpc_diagnostics directory."""
    dirs = glob.glob("mpc_diagnostics_*")
    if not dirs:
        print("No diagnostic directories found!")
        print("Run your simulation first: python scripts/sim.py --controller tunnel_mpc.py")
        return None

    # Sort by modification time, get the latest
    latest = max(dirs, key=lambda d: Path(d).stat().st_mtime)
    return latest


def main():
    diag_dir = find_latest_diagnostics()

    if not diag_dir:
        return

    print(f"Found latest diagnostics: {diag_dir}\n")

    # List files
    files = sorted(Path(diag_dir).glob("*"))
    print("Files in directory:")
    for f in files:
        size = f.stat().st_size
        print(f"  - {f.name} ({size} bytes)")

    # Show plot
    control_plot = Path(diag_dir) / "control_values.png"
    if control_plot.exists():
        print(f"\nOpening {control_plot}...")
        try:
            subprocess.run(["xdg-open", str(control_plot)])
        except Exception as e:
            print(f"Could not open with xdg-open: {e}")
            print(f"Try opening manually: {control_plot}")

    # Print log preview
    log_file = Path(diag_dir) / "control_log.json"
    if log_file.exists():
        import json

        with open(log_file) as f:
            data = json.load(f)

        print("\nControl Log Summary:")
        print(f"  - Total steps: {len(data['timestamps'])}")
        print(f"  - Duration: {data['timestamps'][-1]:.3f}s")
        print(f"  - Avg thrust: {sum(data['thrust_c']) / len(data['thrust_c']):.4f} N")
        print(f"  - Max lateral error (w1): {max(abs(x) for x in data['w1']):.4f} m")
        print(f"  - Max vertical error (w2): {max(abs(x) for x in data['w2']):.4f} m")

        failures = sum(1 for s in data["solver_status"] if s != 0)
        if failures > 0:
            print(f"  ⚠ Solver failures: {failures}/{len(data['solver_status'])}")
        else:
            print("  ✓ All solver runs successful!")


if __name__ == "__main__":
    main()
