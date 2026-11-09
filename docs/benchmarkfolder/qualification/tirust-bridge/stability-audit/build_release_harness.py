#!/usr/bin/env python3
"""Replay the unchanged diagnostic build against the installed release DSO."""
import importlib.util
import json
from pathlib import Path

source = Path(__file__).resolve().with_name("build_harness.py")
spec = importlib.util.spec_from_file_location("tirust_stability_build", source)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
builder.BRIDGE = builder.ROOT / "HEPTools/tirust/lib/libmadloop_tirust.so"
builder.BUILD = builder.ROOT / ".tirust-build/stability-audit-release"
builder.main()
metadata = builder.BUILD / "build-metadata.json"
data = json.loads(metadata.read_text())
launcher = Path(__file__).resolve()
data["input_sha256"][str(launcher.relative_to(builder.ROOT))] = builder.digest(launcher)
data["profile"] = "Installed release DSO; diagnostic Fortran flags unchanged from debug audit"
metadata.write_text(json.dumps(data, indent=2) + "\n")
