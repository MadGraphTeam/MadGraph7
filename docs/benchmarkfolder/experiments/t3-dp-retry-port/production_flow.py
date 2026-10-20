#!/usr/bin/env python3
"""Run the frozen mock scenarios on a fresh production-template CLI export."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FROZEN = HERE.parent/"t3-dp-retry"
WORK = ROOT/".ninjars-build/abi3-private"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location("frozen_mock", FROZEN/"mock_flow.py")
mock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock)

source = WORK/"exports/qcd-1"
copy = WORK/"flow-source/qcd-1"
assert not copy.exists()
copy.parent.mkdir()
subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(copy)], check=True)
for path in copy.rglob("*"):
    if path.is_symlink() and path.resolve().is_relative_to(source):
        target = copy/path.resolve().relative_to(source)
        path.unlink()
        path.symlink_to(os.path.relpath(target, path.parent))
original = next(source.glob("SubProcesses/P*_*"))
directory = next(copy.glob("SubProcesses/P*_*"))
# The physics/state-machine source comes entirely from the current CLI export.
# Only the recording driver comes from the frozen, validated test harness.
driver = FROZEN/"qcd-1/check_sa.f"
shutil.copy2(driver, directory/"check_sa.f")
mock.process.WORK = copy.parent
mock.WORK = WORK/"production-flow"
mock.OUTPUT = mock.WORK/"qcd-1"
mock.REPORT = HERE/"mock-flow"
mock.prepare()
metadata = {
    "source": str(source.relative_to(ROOT)),
    "command_file_sha256": sha(HERE/"qcd1.mg5"),
    "production_templates": {
        name: sha(ROOT/"madgraph/iolibs/template_files/loop_optimized"/name)
        for name in ["loop_matrix_standalone.inc", "NINJARS_interface.inc"]},
    "fresh_generated_source": {
        name: sha(original/name) for name in ["loop_matrix.f", "CT_interface.f"]},
    "recording_driver": str(driver.relative_to(ROOT)),
    "recording_driver_sha256": sha(driver),
    "scenario_harness": str((FROZEN/"mock_flow.py").relative_to(ROOT)),
    "scenario_harness_sha256": sha(FROZEN/"mock_flow.py"),
    "scope": "Actual current generated production state machine, with mocked reducer values and phase counters. No genuine MP claim.",
}
(HERE/"mock-flow/origin.json").write_text(json.dumps(metadata, indent=2)+"\n")
mock.run(5)
