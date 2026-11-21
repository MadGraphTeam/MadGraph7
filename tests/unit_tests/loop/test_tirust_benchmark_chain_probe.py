"""Historical optimization probes cannot mutate exports or qualify acceptance."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
import tirust_benchmark_chain_probe as probe


class TestChainProbe(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.original = self.root / "original"
        self.output = self.root / "private"
        self.directory = self.original / "SubProcesses/P0_example"
        self.directory.mkdir(parents=True)
        (self.original / "Source").mkdir()
        (self.original / "Cards").mkdir()
        (self.original / "lib").mkdir()
        (self.original / "Cards/MadLoopParams.dat").write_text("private card")
        (self.original / "lib/backend.a").write_bytes(b"immutable archive")
        self.template = self.root / "historical.f"
        self.template.write_text("program @PREFIX@\nend\n")
        parameters = {"prefix": "example", "nexternal": 4, "nborn": 1}
        rendered = bench.render_worker(self.template.read_text(), parameters)
        (self.directory / "check_sa.f").write_text(rendered)
        (self.directory / "check_sa.o").write_bytes(b"original driver object")
        (self.directory / "loop.o").write_bytes(b"original numerical object")
        (self.directory / "check").write_bytes(b"original executable")
        dependencies = bench.dependency_inputs(self.directory)
        self.compilation = {"numerical_build_from_sources": True,
            "executable_sha256": bench.digest(self.directory / "check"),
            "sources": bench.generated_sources(self.original),
            "worker_template_sha256": bench.digest(self.template),
            "worker_render_parameters": parameters,
            "driver_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "link_inputs_before": dependencies, "link_inputs_after": dependencies,
            "effective_compiler_flags": {}}
        bench.dump(self.original / "compilation.json", self.compilation)
        bench.dump(self.original / "generation.json", {"source": "baseline"})
        for name, replacement in (("capture_effective_flags", lambda *a, **kw: {}),):
            item = patch.object(probe, name, replacement)
            item.start()
            self.addCleanup(item.stop)
        for name, replacement in (("same_compiler_policy", lambda *a: True),
                                   ("runtime_identity", lambda *a: {"libraries": {}, "archives": {}, "metadata": {}})):
            item = patch.object(bench, name, replacement)
            item.start()
            self.addCleanup(item.stop)

    def test_clone_preserves_binary_objects_and_archive_identity(self):
        record = probe.clone_export(self.original, self.output, self.template)
        self.assertFalse(record["compiled_or_relinked"])
        self.assertEqual(probe.objects(self.original), probe.objects(self.output))
        self.assertEqual(bench.dependency_inputs(self.directory),
                         bench.dependency_inputs(self.output / "SubProcesses/P0_example"))
        self.assertTrue((self.output / "lib").is_symlink())
        (self.output / "Cards/MadLoopParams.dat").write_text("changed private settings")
        self.assertEqual((self.original / "Cards/MadLoopParams.dat").read_text(), "private card")

    def test_changed_executable_is_rejected(self):
        (self.directory / "check").write_bytes(b"different executable")
        with self.assertRaisesRegex(ValueError, "Original executable changed"):
            probe.clone_export(self.original, self.output, self.template)
        self.assertFalse(self.output.exists())

    def test_prepared_runtime_and_provenance_cannot_change_before_run(self):
        record = probe.clone_export(self.original, self.output, self.template)
        probe.verify_prepared_clone(self.output, record)
        with patch.object(bench, "runtime_identity", return_value={
                "libraries": {"/different/libtirust.so": "changed"}, "archives": {}, "metadata": {}}):
            with self.assertRaisesRegex(ValueError, "Prepared native runtime changed"):
                probe.verify_prepared_clone(self.output, record)
        bench.dump(self.output / "generation.json", {"source": "different historical claim"})
        with self.assertRaisesRegex(ValueError, "Private build provenance changed"):
            probe.verify_prepared_clone(self.output, record)

    def test_same_named_libraries_cannot_evade_equality_using_other_paths(self):
        inputs = {source: {"runtime": {"libraries": {f"/{source}/libninja.so": checksum},
                                      "archives": {}, "metadata": {}}}
                  for source, checksum in (("baseline", "old"), ("candidate", "new"))}
        with self.assertRaisesRegex(ValueError, "Shared runtime library bytes differ"):
            probe.validate_shared_runtime(inputs)

    def test_changed_static_archive_is_rejected(self):
        (self.original / "lib/backend.a").write_bytes(b"different archive")
        with self.assertRaisesRegex(ValueError, "link-time dependencies changed"):
            probe.clone_export(self.original, self.output, self.template)

    def test_mutable_external_card_is_rejected(self):
        card = self.original / "Cards/MadLoopParams.dat"
        card.unlink()
        outside = self.root / "shared-card.dat"
        outside.write_text("must not mutate")
        card.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Mutable resource escapes"):
            probe.clone_export(self.original, self.output, self.template)

    def test_live_original_is_rejected(self):
        (self.directory / "worker-protocol.fifo").write_text("active sentinel")
        with self.assertRaisesRegex(ValueError, "Original output is active"):
            probe.clone_export(self.original, self.output, self.template)

    def test_statistics_are_descriptive_without_a_gate(self):
        stats = probe.descriptive_statistics({"baseline": [1.] * 7, "candidate": [1.12] * 7})
        self.assertAlmostEqual(stats["geometric_mean_ratio"], 1.12)
        for key in ("passed", "acceptance_eligible", "threshold", "minimum_pairs"):
            self.assertNotIn(key, stats)
        self.assertEqual(stats["median_seconds_per_point"], {"baseline": 1., "candidate": 1.12})

    def test_workspace_restores_on_failure(self):
        before = bench.WORK
        with self.assertRaises(ValueError):
            with probe.private_workspace(self.root):
                self.assertEqual(bench.WORK, self.root)
                raise ValueError("intentional")
        self.assertEqual(bench.WORK, before)


if __name__ == "__main__":
    unittest.main()
