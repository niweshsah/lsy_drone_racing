#!/usr/bin/env python3
"""Quick test script for tunnel_mpc plotting functionality."""

import sys
import subprocess
from pathlib import Path

# Run sim with tunnel_mpc controller for 1 run
result = subprocess.run(
    [sys.executable, "scripts/sim.py", "--config", "level0.toml", "--controller", "tunnel_mpc.py", "--n_runs", "1"],
    cwd=Path(__file__).parent,
    capture_output=True,
    text=True
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

# Check if plots were generated
import glob
plots = glob.glob("mpc_diagnostics_*/control_values.png")
logs = glob.glob("control_log_*.json")

print(f"\n{'='*60}")
print(f"Generated plots: {len(plots)}")
for p in plots:
    print(f"  - {p}")
print(f"Generated logs: {len(logs)}")
for l in logs:
    print(f"  - {l}")
print(f"{'='*60}")
