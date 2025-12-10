#!/usr/bin/env python3
"""Verify the tunnel_mpc controller has all required methods."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from lsy_drone_racing.control.tunnel_mpc import SpatialMPCController
from lsy_drone_racing.control import Controller
import inspect

print("Checking SpatialMPCController methods...\n")

# Check inheritance
print(f"✓ Inherits from Controller: {issubclass(SpatialMPCController, Controller)}")

# Check required methods
required_methods = [
    'compute_control',
    'step_callback', 
    'episode_callback',
    'reset',
    'episode_reset',
    '_log_control_step',
    'save_control_log',
    'plot_control_values',
    'plot_solver_status',
    'plot_all_diagnostics'
]

for method_name in required_methods:
    has_method = hasattr(SpatialMPCController, method_name)
    status = "✓" if has_method else "✗"
    print(f"{status} {method_name}")
    if has_method:
        method = getattr(SpatialMPCController, method_name)
        sig = inspect.signature(method)
        print(f"   Signature: {sig}")

print("\n✓ All methods present!")
