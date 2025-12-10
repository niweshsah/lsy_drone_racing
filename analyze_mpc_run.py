#!/usr/bin/env python3
"""
Complete example showing MPC debugging workflow.

This demonstrates:
1. Running simulation with automatic plot generation
2. Accessing control logs programmatically  
3. Manual plot generation
4. Analysis of solver performance
"""

import sys
from pathlib import Path

# Example of accessing the controller after simulation
def analyze_latest_run():
    """Analyze the latest MPC run."""
    import glob
    import json
    
    # Find latest log
    logs = glob.glob("mpc_diagnostics_*/control_log.json")
    if not logs:
        print("No control logs found. Run simulation first:")
        print("  python scripts/sim.py --controller tunnel_mpc.py --config level0.toml")
        return
    
    latest_log = sorted(logs)[-1]
    parent_dir = Path(latest_log).parent
    
    print(f"\n{'='*70}")
    print(f"ANALYZING MPC RUN: {parent_dir}")
    print(f"{'='*70}\n")
    
    # Load and analyze
    with open(latest_log) as f:
        log = json.load(f)
    
    # Basic stats
    n_steps = len(log['timestamps'])
    duration = log['timestamps'][-1]
    
    print(f"📊 BASIC STATISTICS")
    print(f"  Total steps: {n_steps}")
    print(f"  Duration: {duration:.3f} seconds")
    print(f"  Average frequency: {n_steps/duration:.1f} Hz")
    
    # Control statistics
    import numpy as np
    
    print(f"\n🎮 CONTROL COMMANDS")
    print(f"  Roll (φ_c):   min={min(log['phi_c']):+.4f}, max={max(log['phi_c']):+.4f}, "
          f"mean={np.mean(log['phi_c']):+.4f}")
    print(f"  Pitch (θ_c):  min={min(log['theta_c']):+.4f}, max={max(log['theta_c']):+.4f}, "
          f"mean={np.mean(log['theta_c']):+.4f}")
    print(f"  Yaw (ψ_c):    min={min(log['psi_c']):+.4f}, max={max(log['psi_c']):+.4f}, "
          f"mean={np.mean(log['psi_c']):+.4f}")
    print(f"  Thrust (T_c): min={min(log['thrust_c']):.4f}, max={max(log['thrust_c']):.4f}, "
          f"mean={np.mean(log['thrust_c']):.4f}")
    
    # Solver statistics
    print(f"\n⚙️ SOLVER PERFORMANCE")
    failures = sum(1 for s in log['solver_status'] if s != 0)
    success_rate = 100 * (n_steps - failures) / n_steps
    print(f"  Success rate: {success_rate:.1f}% ({n_steps - failures}/{n_steps})")
    if failures > 0:
        print(f"  ⚠️  Failed solves: {failures}")
        unique_statuses = set(log['solver_status'])
        print(f"  Failure codes: {sorted(unique_statuses)}")
    else:
        print(f"  ✓ All solves successful!")
    
    # Path tracking statistics
    print(f"\n📍 PATH TRACKING")
    print(f"  Arc length (s): {log['s'][-1]:.2f} m")
    print(f"  Max lateral error (w1): {max(abs(x) for x in log['w1']):.4f} m")
    print(f"  Max vertical error (w2): {max(abs(x) for x in log['w2']):.4f} m")
    print(f"  RMS lateral error: {np.sqrt(np.mean(np.array(log['w1'])**2)):.4f} m")
    print(f"  RMS vertical error: {np.sqrt(np.mean(np.array(log['w2'])**2)):.4f} m")
    
    # Speed statistics
    print(f"\n🚀 SPEED TRACKING")
    print(f"  Target speed: ~4.0 m/s")
    print(f"  Achieved ds: min={min(log['ds']):.3f}, max={max(log['ds']):.3f}, "
          f"mean={np.mean(log['ds']):.3f}")
    
    print(f"\n{'='*70}")
    print(f"📁 Output directory: {parent_dir}")
    print(f"  - control_values.png (6-subplot figure)")
    print(f"  - solver_status.png (success/failure timeline)")
    print(f"  - control_log.json (raw data)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    analyze_latest_run()
