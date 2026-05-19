#!/usr/bin/env bash
# Check that the shared workflow environment is portable across group users.
# Run after activating /gpfs/group/shenvi/envs/workflow312, or pass the Python
# executable explicitly as the first argument.
set -euo pipefail

PY="${1:-/gpfs/group/shenvi/envs/workflow312/bin/python}"
GROUP_ROOT="/gpfs/group/shenvi"
EXPECTED_SW_ROOT="${SCRIPPS_WORKFLOW_ROOT:-/gpfs/group/shenvi/code/scripps-workflow}"
EXPECTED_NMR_ROOT="${NMR_HPC_DATA_ROOT:-/gpfs/group/shenvi/nmr-data/runs}"

echo "python: ${PY}"
"${PY}" - <<'PY'
import os
import shutil
import sys

GROUP_ROOT = "/gpfs/group/shenvi"
EXPECTED_SW_ROOT = os.environ.get("SCRIPPS_WORKFLOW_ROOT", "/gpfs/group/shenvi/code/scripps-workflow")
EXPECTED_NMR_ROOT = os.environ.get("NMR_HPC_DATA_ROOT", "/gpfs/group/shenvi/nmr-data/runs")

print("sys.executable:", sys.executable)

try:
    import scripps_workflow
except Exception as exc:
    raise SystemExit(f"ERROR: could not import scripps_workflow: {exc!r}")

sw_file = os.path.realpath(scripps_workflow.__file__)
print("scripps_workflow:", sw_file)
if not sw_file.startswith(os.path.realpath(EXPECTED_SW_ROOT) + os.sep):
    raise SystemExit(
        "ERROR: scripps_workflow is not imported from the shared group repo:\n"
        f"  expected under: {EXPECTED_SW_ROOT}\n"
        f"  got:            {sw_file}"
    )

try:
    import nmr_data
except Exception as exc:
    raise SystemExit(f"ERROR: could not import nmr_data: {exc!r}")

nmr_file = os.path.realpath(nmr_data.__file__)
print("nmr_data:", nmr_file)
if not nmr_file.startswith(os.path.realpath("/gpfs/group/shenvi/code/nmr-data") + os.sep):
    raise SystemExit(
        "ERROR: nmr_data is not imported from the shared group repo:\n"
        f"  expected under: /gpfs/group/shenvi/code/nmr-data\n"
        f"  got:            {nmr_file}"
    )

print("SCRIPPS_WORKFLOW_ROOT:", os.environ.get("SCRIPPS_WORKFLOW_ROOT", ""))
if os.environ.get("SCRIPPS_WORKFLOW_ROOT") != EXPECTED_SW_ROOT:
    raise SystemExit(
        "ERROR: SCRIPPS_WORKFLOW_ROOT is unset or unexpected:\n"
        f"  expected: {EXPECTED_SW_ROOT}\n"
        f"  got:      {os.environ.get('SCRIPPS_WORKFLOW_ROOT', '')}"
    )

print("NMR_HPC_DATA_ROOT:", os.environ.get("NMR_HPC_DATA_ROOT", ""))
if os.environ.get("NMR_HPC_DATA_ROOT") != EXPECTED_NMR_ROOT:
    raise SystemExit(
        "ERROR: NMR_HPC_DATA_ROOT is unset or unexpected:\n"
        f"  expected: {EXPECTED_NMR_ROOT}\n"
        f"  got:      {os.environ.get('NMR_HPC_DATA_ROOT', '')}"
    )

print("NMR_DATABASE_URL set:", bool(os.environ.get("NMR_DATABASE_URL")))
if not os.environ.get("NMR_DATABASE_URL"):
    raise SystemExit("ERROR: NMR_DATABASE_URL is not set")

for exe in ("xtb", "crest"):
    path = shutil.which(exe)
    print(f"{exe}:", path)
    if not path:
        raise SystemExit(f"ERROR: {exe} not found on PATH")
    real = os.path.realpath(path)
    if not real.startswith(GROUP_ROOT + os.sep):
        raise SystemExit(
            f"ERROR: {exe} resolves outside the shared group tree:\n"
            f"  command -v: {path}\n"
            f"  realpath:   {real}\n"
            f"  expected under: {GROUP_ROOT}"
        )

print("Group portability check passed.")
PY
