# MPC Debugging Implementation Summary

## Problem
You wanted debugging prints for MPC control values at each timestep with graphs.

## Solution Implemented

### 1. **Real-time Console Debugging** ✓
- Each MPC step logs control values and state to console
- Format: `[Step N] t=X.XXXs | φ_c=+X.XXXX θ_c=+X.XXXX ψ_c=+X.XXXX T_c=X.XXXX | s=X.XX w1=+X.XXXX w2=+X.XXXX ds=X.XXX | Status=0`
- Implementation: `_log_control_step()` method called from `compute_control()`
- Control via: `controller.debug = True/False`

### 2. **Automatic Data Logging** ✓
- All control values stored in `control_log` dictionary
- Tracks: timestamps, φ_c, θ_c, ψ_c, T_c, solver_status, s, w1, w2, ds
- Data persisted for end-of-episode analysis

### 3. **Automatic Plot Generation** ✓
- `episode_callback()` automatically calls `plot_all_diagnostics()` at episode end
- Creates timestamped directory: `mpc_diagnostics_YYYYMMDD_HHMMSS/`

### 4. **Generated Visualizations**

**control_values.png** - 6 subplot figure:
- Roll command (φ_c) vs time
- Pitch command (θ_c) vs time  
- Thrust command (T_c) vs time with hover reference
- Yaw command (ψ_c) vs time
- Arc length (s) vs time
- Lateral/Vertical errors (w1, w2) with constraint bounds (±0.5m)

**solver_status.png** - Solver success/failure:
- Green dots = successful solve (status=0)
- Red dots = solver failure (status≠0)
- Helps identify optimization issues

**control_log.json** - Raw data:
- JSON export of all logged values
- Useful for post-processing analysis

## Files Modified
- `lsy_drone_racing/control/tunnel_mpc.py` - Added all debugging features

## New Helper Scripts
- `view_latest_plots.py` - View and summarize latest diagnostic run
- `verify_tunnel_mpc.py` - Verify all methods are implemented
- `test_tunnel_mpc.py` - Simple test runner

## Usage

### Automatic (Default)
```bash
python scripts/sim.py --controller tunnel_mpc.py --config level0.toml
# Plots auto-generated at episode end in mpc_diagnostics_TIMESTAMP/
```

### Manual Control
```python
controller.debug = False  # Disable console prints
controller.plot_control_values(save_path="my_plot.png")
controller.save_control_log("my_log.json")
controller.plot_all_diagnostics("my_output_dir")
```

### View Results
```bash
python view_latest_plots.py
# Shows latest diagnostics summary and opens plots
```

## Output Structure
```
mpc_diagnostics_20251210_143022/
├── control_log.json          (raw data)
├── control_values.png        (6-subplot control & state)
└── solver_status.png         (solver success/failure over time)
```

## Integration with Simulator
The simulator automatically calls:
1. `compute_control()` - Logs data via `_log_control_step()`
2. `step_callback()` - Called after each step (returns False to continue)
3. `episode_callback()` - Called at episode end (generates plots)
4. `episode_reset()` - Called to reset for next episode

All integration happens automatically - no code changes to sim.py needed!
