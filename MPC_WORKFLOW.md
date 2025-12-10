# 🚁 MPC Debugging & Analysis Workflow

## Quick Start

### 1. Run Your Simulation (as usual)
```bash
python scripts/sim.py --controller tunnel_mpc.py --config level0.toml
```

### 2. Plots Are Automatically Generated
- Look for a `mpc_diagnostics_TIMESTAMP/` directory
- Contains:
  - `control_values.png` - Main debugging plot
  - `solver_status.png` - Solver success/failure
  - `control_log.json` - Raw data

### 3. Analyze Results (Optional)
```bash
python analyze_mpc_run.py
```

## Console Output During Run

You'll see lines like:
```
[Step 1] t=0.017s | φ_c=+0.0000 θ_c=+0.0000 ψ_c=+0.0000 T_c=0.4264 | s=0.00 w1=+0.0000 w2=+0.0000 ds=0.000 | Status=0
[Step 2] t=0.033s | φ_c=+0.0234 θ_c=-0.0156 ψ_c=+0.0000 T_c=0.4266 | s=0.07 w1=-0.0023 w2=+0.0045 ds=4.123 | Status=0
[Step 3] t=0.050s | φ_c=+0.0456 θ_c=-0.0312 ψ_c=+0.0000 T_c=0.4270 | s=0.15 w1=-0.0089 w2=+0.0178 ds=4.156 | Status=0
```

**Legend:**
- `[Step N]` - Control step number
- `t=X.XXXs` - Elapsed time
- `φ_c, θ_c, ψ_c` - Attitude commands (roll, pitch, yaw) in radians
- `T_c` - Thrust command in Newtons
- `s` - Current arc length along path in meters
- `w1` - Lateral error from path in meters
- `w2` - Vertical error from path in meters  
- `ds` - Speed along path in m/s
- `Status` - MPC solver status (0=success, other=failure)

### Disable Console Prints
```python
# In controller initialization or before running
controller.debug = False
```

## Understanding the Plots

### control_values.png

**Row 1: Attitude Commands**
- **Roll (φ_c)**: Should be smooth, small magnitude (~±0.5 rad max)
- **Pitch (θ_c)**: Should be smooth, small magnitude (~±0.5 rad max)

**Row 2: Thrust & Yaw**
- **Thrust (T_c)**: Black dashed line shows hover thrust (~0.426 N)
  - Should oscillate gently around hover
  - Large oscillations = poor control
- **Yaw (ψ_c)**: Should be near zero for forward flight

**Row 3: Path Tracking**
- **Arc length (s)**: Should increase monotonically
  - Flat section = drone slowed down
  - Jumps = impossible (indicates issues)
- **Errors (w1, w2)**: 
  - Red dashed lines at ±0.5m are constraint bounds
  - Should stay within bounds
  - Oscillations = poor tracking

### solver_status.png

- **Green dots** = MPC solver succeeded (status=0)
- **Red dots** = MPC solver failed (status≠0)
- **Patterns**: 
  - All green = good
  - Clusters of red = indicates when constraints become tight

## Debugging Guide

| Symptom | Possible Cause | Check |
|---------|---|---|
| Thrust oscillating wildly | Control gains too high | Increase W_e weights on s |
| w1/w2 errors large | Path constraints violated | Check constraint bounds |
| Frequent solver failures | Infeasible problem | Loosen constraints or increase horizon |
| s not increasing smoothly | Speed control poor | Adjust yref[3] target speed |
| Red solver failures at specific times | Tight corners | May need MPC tuning |

## Programmatic Access

### Get Latest Diagnostics
```python
import glob
import json

# Find latest log
log_files = sorted(glob.glob("mpc_diagnostics_*/control_log.json"))
if log_files:
    with open(log_files[-1]) as f:
        data = json.load(f)
    
    print(f"Steps: {len(data['timestamps'])}")
    print(f"Max thrust: {max(data['thrust_c'])}")
    print(f"Avg roll: {sum(data['phi_c'])/len(data['phi_c'])}")
```

### Custom Analysis
```python
import numpy as np
import json

with open("mpc_diagnostics_20251210_143022/control_log.json") as f:
    log = json.load(f)

# Your analysis here
phi = np.array(log['phi_c'])
theta = np.array(log['theta_c'])
thrust = np.array(log['thrust_c'])

# Example: Find moments of high control activity
high_control = np.where(np.abs(phi) > 0.3)[0]
print(f"Steps with |φ| > 0.3: {high_control}")
```

## File Locations

After each run, find outputs in timestamped directories:
```
mpc_diagnostics_20251210_143022/
├── control_log.json              # Raw numeric data
├── control_values.png            # Main debug plot
└── solver_status.png             # Solver success/failure
```

Keep these directories for post-analysis or comparison between runs!

## Helper Scripts

```bash
# Analyze latest run with statistics
python analyze_mpc_run.py

# View latest plots and summary
python view_latest_plots.py

# Verify controller implementation
python verify_tunnel_mpc.py
```

## Customization

### Change Debug Output Format
Edit `_log_control_step()` in `tunnel_mpc.py`

### Modify Plot Appearance
Edit `plot_control_values()` or `plot_solver_status()` methods

### Change Data Logging
Modify `control_log` dictionary initialization in `__init__()`

---

**💡 Tip:** Compare multiple runs by keeping their diagnostic directories. This helps identify improvement patterns!
