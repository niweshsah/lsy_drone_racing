# import sys
# from pprint import pprint

# try:
#     import lsy_drone_racing.control
#     print(f"DEBUG: Control path is: {lsy_drone_racing.control.__path__}")
#     from lsy_drone_racing.control.common_functions.yaml_import import load_yaml
    
#     from lsy_drone_racing.control.GeometryEngines.geometryEngine1 import GeometryEngine
#     from lsy_drone_racing.control.model_dynamics.mpc1 import SpatialMPC, get_drone_params
    
#     print("✅ All modules imported successfully!")
#     print(f"GeometryEngine location: {GeometryEngine.__module__}")
# except ImportError as e:
#     print(f"❌ Import failed: {e}")
#     print("\nCurrent Python Path:")
#     pprint(sys.path)


# """
# Diagnostic script to check package structure and identify import issues
# """
# import sys
# import os
# from pathlib import Path
# from pprint import pprint

# print("=" * 60)
# print("PACKAGE STRUCTURE DIAGNOSTICS")
# print("=" * 60)

# # 1. Check if lsy_drone_racing can be imported
# print("\n1. Base package import:")
# try:
#     import lsy_drone_racing
#     print(f"   ✅ lsy_drone_racing imported successfully")
#     print(f"   Location: {lsy_drone_racing.__file__}")
#     print(f"   Package path: {lsy_drone_racing.__path__}")
# except ImportError as e:
#     print(f"   ❌ Failed: {e}")
#     sys.exit(1)

# # 2. Check control module
# print("\n2. Control module:")
# try:
#     import lsy_drone_racing.control
#     print(f"   ✅ lsy_drone_racing.control imported")
#     print(f"   Location: {lsy_drone_racing.control.__file__}")
#     print(f"   Control path: {lsy_drone_racing.control.__path__}")
# except ImportError as e:
#     print(f"   ❌ Failed: {e}")

# # 3. List actual directory structure
# print("\n3. Physical directory structure:")
# pkg_path = Path(lsy_drone_racing.__file__).parent
# print(f"   Base path: {pkg_path}")

# control_path = pkg_path / "control"
# if control_path.exists():
#     print(f"\n   Contents of {control_path}:")
#     for item in sorted(control_path.iterdir()):
#         indicator = "📁" if item.is_dir() else "📄"
#         print(f"   {indicator} {item.name}")
        
#         # Check for __init__.py in directories
#         if item.is_dir():
#             init_file = item / "__init__.py"
#             if init_file.exists():
#                 print(f"      ✅ Has __init__.py")
#             else:
#                 print(f"      ❌ Missing __init__.py")
# else:
#     print(f"   ❌ Control directory not found at {control_path}")

# # 4. Try to import common_functions
# print("\n4. Attempting to import common_functions:")
# try:
#     from lsy_drone_racing.control import common_functions
#     print(f"   ✅ common_functions imported")
#     print(f"   Location: {common_functions.__file__}")
# except ImportError as e:
#     print(f"   ❌ Failed: {e}")
    
#     # Check if the directory exists but lacks __init__.py
#     common_funcs_path = control_path / "common_functions"
#     if common_funcs_path.exists():
#         print(f"\n   Directory exists at: {common_funcs_path}")
#         init_file = common_funcs_path / "__init__.py"
#         if not init_file.exists():
#             print(f"   ⚠️  ISSUE: Missing __init__.py in common_functions/")
#             print(f"   This is likely the cause of the import error!")

# # 5. Try the specific import from the test script
# print("\n5. Attempting yaml_import:")
# try:
#     from lsy_drone_racing.control.common_functions.yaml_import import load_yaml
#     print(f"   ✅ load_yaml imported successfully")
# except ImportError as e:
#     print(f"   ❌ Failed: {e}")

# # 6. Check sys.path
# print("\n6. Python sys.path:")
# for i, p in enumerate(sys.path[:5], 1):
#     print(f"   {i}. {p}")
# if len(sys.path) > 5:
#     print(f"   ... and {len(sys.path) - 5} more paths")

# print("\n" + "=" * 60)
# print("RECOMMENDATIONS:")
# print("=" * 60)

# # Check for missing __init__.py files
# if control_path.exists():
#     missing_init = []
#     for item in control_path.rglob("*"):
#         if item.is_dir() and not (item / "__init__.py").exists():
#             # Check if it's a Python package (contains .py files)
#             if any(item.glob("*.py")):
#                 missing_init.append(item.relative_to(pkg_path))
    
#     if missing_init:
#         print("\n⚠️  Missing __init__.py files in:")
#         for path in missing_init:
#             print(f"   - {path}/")
#         print("\nTo fix: Create empty __init__.py files in these directories:")
#         for path in missing_init:
#             full_path = pkg_path / path / "__init__.py"
#             print(f"   touch {full_path}")
#     else:
#         print("\n✅ All package directories have __init__.py files")
# else:
#     print("\n❌ Control directory structure needs to be checked")

# print("\n" + "=" * 60)




"""
Comprehensive test script to verify all module imports
"""
import sys

print("=" * 60)
print("TESTING MODULE IMPORTS")
print("=" * 60)

# Track results
results = []

def test_import(description, import_func):
    """Helper function to test imports and track results"""
    try:
        result = import_func()
        print(f"✅ {description}")
        results.append((description, True, None))
        return result
    except Exception as e:
        print(f"❌ {description}")
        print(f"   Error: {e}")
        results.append((description, False, str(e)))
        return None

# Test 1: Base package
print("\n1. Base Package:")
lsy_drone_racing = test_import(
    "lsy_drone_racing",
    lambda: __import__('lsy_drone_racing')
)

# Test 2: Control module
print("\n2. Control Module:")
control = test_import(
    "lsy_drone_racing.control",
    lambda: __import__('lsy_drone_racing.control', fromlist=[''])
)

# Test 3: Common functions
print("\n3. Common Functions:")
load_yaml = test_import(
    "load_yaml from yaml_import",
    lambda: __import__('lsy_drone_racing.control.common_functions.yaml_import', fromlist=['load_yaml']).load_yaml
)

# Test 4: Geometry Engine
print("\n4. Geometry Engine:")
GeometryEngine = test_import(
    "GeometryEngine from geometryEngine1",
    lambda: __import__('lsy_drone_racing.control.GeometryEngines.geometryEngine1', fromlist=['GeometryEngine']).GeometryEngine
)

# Test 5: MPC modules
print("\n5. MPC Modules:")
SpatialMPC = test_import(
    "SpatialMPC from mpc1",
    lambda: __import__('lsy_drone_racing.control.model_dynamics.mpc1', fromlist=['SpatialMPC']).SpatialMPC
)

get_drone_params = test_import(
    "get_drone_params from mpc1",
    lambda: __import__('lsy_drone_racing.control.model_dynamics.mpc1', fromlist=['get_drone_params']).get_drone_params
)

# Test 6: Check if classes/functions are callable
print("\n6. Verifying Callability:")
if GeometryEngine:
    test_import(
        "GeometryEngine is a class",
        lambda: callable(GeometryEngine) or None
    )
    if callable(GeometryEngine):
        print(f"   Location: {GeometryEngine.__module__}")

if SpatialMPC:
    test_import(
        "SpatialMPC is a class",
        lambda: callable(SpatialMPC) or None
    )
    if callable(SpatialMPC):
        print(f"   Location: {SpatialMPC.__module__}")

if get_drone_params:
    test_import(
        "get_drone_params is callable",
        lambda: callable(get_drone_params) or None
    )

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

passed = sum(1 for _, success, _ in results if success)
failed = sum(1 for _, success, _ in results if not success)

print(f"\n✅ Passed: {passed}/{len(results)}")
print(f"❌ Failed: {failed}/{len(results)}")

if failed > 0:
    print("\nFailed imports:")
    for desc, success, error in results:
        if not success:
            print(f"  - {desc}")
            print(f"    {error}")
    sys.exit(1)
else:
    print("\n🎉 All imports successful!")
    print("\nYou can now use these modules in your code:")
    print("  from lsy_drone_racing.control.common_functions.yaml_import import load_yaml")
    print("  from lsy_drone_racing.control.GeometryEngines.geometryEngine1 import GeometryEngine")
    print("  from lsy_drone_racing.control.model_dynamics.mpc1 import SpatialMPC, get_drone_params")
    
print("\n" + "=" * 60)