# MPC Debugging & Visualization Guide

## What Was Added

Your `SpatialMPCController` now has comprehensive debugging and visualization capabilities:

### 1. **Real-time Console Debugging**
- Each control step prints a formatted debug line with:
  - Step count and elapsed time
  - Control commands: φ_c (roll), θ_c (pitch), ψ_c (yaw), T_c (thrust)
  - State feedback: s (arc length), w1 & w2 (lateral/vertical errors), ds (speed)
  - MPC solver status (0 = success, non-zero = failure)

Example output:
```
[Step 42] t=0.840s | φ_c=+0.0234 θ_c=-0.0156 ψ_c=+0.0000 T_c=0.4266 | s=3.45 w1=-0.0023 w2=+0.0045 ds=4.123 | Status=0
```

Enable/disable with: `controller.debug = True/False`

### 2. **Automatic Plot Generation**
At the end of each episode, plots are automatically generated showing:

#### **Control Values Plot** (`control_values.png`)
- Roll & Pitch commands over time
- Thrust command (with hover thrust reference)
- Yaw command
- Arc length progression
- Lateral/Vertical errors (with ±0.5m constraint bounds)

#### **Solver Status Plot** (`solver_status.png`)
- Green dots = successful MPC solve
- Red dots = solver failure
- Helps identify optimization issues

#### **Control Log** (`control_log.json`)
- Raw data export of all control values and states
- Useful for offline analysis

### 3. **Output Organization**
All plots and logs are saved to a timestamped directory:
```
mpc_diagnostics_20251210_143022/
├── control_log.json
├── control_values.png
└── solver_status.png
```

## How It Works

The integration is automatic:

1. **During Episode**: Each call to `compute_control()` logs data via `_log_control_step()`
2. **After Episode**: `episode_callback()` is automatically called by the simulator
3. **Plots Generated**: `plot_all_diagnostics()` creates all visualizations

## Manual Usage

You can also manually trigger plotting:

```python
# After episode is done
controller.plot_control_values()          # Just control plots
controller.plot_solver_status()           # Just solver status
controller.plot_all_diagnostics()         # Everything
controller.save_control_log("my_log.json") # Save raw data
```

## How to Run

Simply run as normal - plots will be auto-generated:
```bash
python scripts/sim.py --controller tunnel_mpc.py --config level0.toml
```

## Example Analysis

Check the generated plots to debug:
- **Thrust oscillations?** Check the thrust plot stability
- **Solver failing?** Look at the solver status plot for patterns
- **Bad path tracking?** Check w1/w2 errors against the bounds
- **Strange attitude commands?** Review φ_c and θ_c plots

All numerical data is in `control_log.json` for custom analysis.
