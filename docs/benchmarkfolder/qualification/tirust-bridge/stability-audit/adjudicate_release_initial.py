#!/usr/bin/env python3
"""Run the unchanged adjudication assertions and label the release replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

source = Path(__file__).resolve().with_name("adjudicate.py")
spec = importlib.util.spec_from_file_location("tirust_stability_adjudication", source)
adjudicator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adjudicator)
adjudicator.main()
destination = Path(sys.argv[2])
data = json.loads(destination.read_text())
assert data["limits"][0] == "Debug bridge correctness evidence, not release performance qualification"
data["limits"][0] = "Installed release bridge correctness replay, not performance qualification"
data["release_adjudicator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
destination.write_text(json.dumps(data, indent=2) + "\n")
