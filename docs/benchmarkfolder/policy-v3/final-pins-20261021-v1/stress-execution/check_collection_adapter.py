#!/usr/bin/env python3
"""Source/ownership adapter checks only; no compiler or numerical engine."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import collect_final as adapter
import collect_nrs_first as nrs_first


class AdapterChecks(unittest.TestCase):
    def test_live_production_matches_frozen_generation(self):
        inventory = adapter.numerical_sources(adapter.final.REVISION)
        registration = adapter.final.read(adapter.final.BASE / "stress-generation/early-registration.json")
        self.assertEqual(inventory["production"], registration["numerical_sources_sha256"]["production"])

    def test_another_numerical_revision_is_rejected(self):
        with self.assertRaises(ValueError):
            adapter.numerical_sources("0" * 40)

    def test_dirty_production_fails_before_any_worker(self):
        with patch.object(adapter.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "git diff")):
            with self.assertRaises(subprocess.CalledProcessError):
                adapter.numerical_sources(adapter.final.REVISION)

    def test_untracked_generator_input_is_rejected(self):
        with patch.object(adapter.subprocess, "run"), patch.object(
                adapter.subprocess, "check_output", return_value="madgraph/injected.py\n"):
            with self.assertRaises(ValueError):
                adapter.numerical_sources(adapter.final.REVISION)

    def test_numerical_functions_and_target_inventory_are_preserved(self):
        old = (adapter.raw.workers.Worker, adapter.raw.workers.run_case,
               adapter.raw.stress.scale_adapter, adapter.raw.stress.configuration)
        adapter.configure_adapters()
        self.assertEqual(old, (adapter.raw.workers.Worker, adapter.raw.workers.run_case,
                              adapter.raw.stress.scale_adapter, adapter.raw.stress.configuration))
        self.assertEqual(adapter.raw.BACKENDS, (1, 6, 7, 8))

    def test_single_job_adapter_restores_dispatch_on_failure(self):
        original, calls = adapter.bench.run, []
        def run(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError("expected control")
        with patch.object(adapter.pair, "owned_preparation", return_value=nullcontext()), \
                patch.object(adapter.bench, "run", run):
            with self.assertRaisesRegex(RuntimeError, "expected control"):
                with adapter.owned_processes(None, None):
                    adapter.bench.run(["make", "-j4", "check"], "cwd", "log", timeout=7)
            self.assertIs(adapter.bench.run, run)
        self.assertIs(adapter.bench.run, original)
        self.assertEqual(calls, [((["make", "-j1", "check"], "cwd", "log"), {"timeout": 7})])

    def test_nrs_first_targets_preserve_independent_derived_controls(self):
        self.assertEqual({name: nrs_first.target_backends(name)
                          for name in adapter.final.stress.VARIANTS},
                         {"ordinary": (8,), "rotation": (6, 8), "scale": (6, 8)})
        with self.assertRaises(ValueError):
            nrs_first.target_backends("unregistered")

    def test_resealed_backend_relabeling_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=adapter.ROOT / ".ninjars-build/tmp") as directory:
            path = Path(directory) / "base.json"
            base = adapter.policy.seal({"schema": adapter.raw.SCHEMA, "variant": "ordinary"})
            path.write_text(json.dumps(base))
            spec = nrs_first.selected_spec(base, adapter.supplement.bound(path))
            self.assertEqual(nrs_first.validate_selection(spec), (8,))
            spec["explicit_target_backends"] = [6]
            with self.assertRaises(ValueError):
                nrs_first.validate_selection(adapter.policy.seal(spec))

    def test_target_policy_requires_immutable_base_specification(self):
        with tempfile.TemporaryDirectory(dir=adapter.ROOT / ".ninjars-build/tmp") as directory:
            path = Path(directory) / "base.json"
            base = adapter.policy.seal({"schema": adapter.raw.SCHEMA, "variant": "scale"})
            path.write_text(json.dumps(base))
            spec = nrs_first.selected_spec(base, adapter.supplement.bound(path))
            path.write_text(json.dumps(adapter.policy.seal({**base, "variant": "ordinary"})))
            with self.assertRaises((ValueError, RuntimeError)):
                nrs_first.validate_selection(spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
