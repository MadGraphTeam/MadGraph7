"""A new campaign may copy proven baseline objects, never mutable old cards."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
import tirust_benchmark_compiler as compiler
import tirust_benchmark_relink as relink


class TestBaselineReuse(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.work, self.report = root / "work", root / "report"
        self.case = {"id": "example", "nfinal": 1}
        self.original = self.work / "outputs/baseline/example"
        self.target = self.work / "campaigns/new/outputs/baseline/example"
        self.sub = self.original / "SubProcesses/P0_example"
        self.sub.mkdir(parents=True)
        for name, content in (("check", "executable"), ("loop.f", "source"), ("loop.o", "object")):
            (self.sub / name).write_text(content)
        (self.original / "Cards").mkdir()
        (self.original / "Cards/param_card.dat").write_text("physical card")
        (self.original / "Cards/MadLoopParams.dat").write_text("mutable reducer card")
        self.sources = {"SubProcesses/P0_example/loop.f": bench.digest(self.sub / "loop.f")}
        self.dependencies = {"archives": {}, "metadata": {}}
        self.runtime = {**self.dependencies, "libraries": {"/lib/native.so": "frozen"}}
        bench.dump(self.original / "generation.json", {"case": self.case, "source": "baseline",
            "source_identity": {"revision": bench.BASELINE_REVISION}})
        bench.dump(self.original / "compilation.json", {"sources": self.sources,
            "executable_sha256": bench.digest(self.sub / "check"), "numerical_build_from_sources": True,
            "link_inputs_before": self.dependencies, "link_inputs_after": self.dependencies,
            "effective_compiler_flags": {"policy": "same"}})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("CAMPAIGN_ID", "new"), ("CAMPAIGN_MANIFEST_SHA256", "manifest"),
                ("CAMPAIGN_MANIFEST", {"predecessor": {"id": "legacy"}}),
                ("LEGACY_WORK", self.work), ("WORK", self.work / "campaigns/new"), ("REPORT", self.report),
                ("campaign_guard", lambda stage: None), ("output_dir", lambda *args: self.target),
                ("generated_sources", lambda root: self.sources),
                ("dependency_inputs", lambda root: self.dependencies),
                ("runtime_identity", lambda root: self.runtime),
                ("same_compiler_policy", lambda left, right: left == right)):
            self.stack.enter_context(patch.object(bench, name, value))
        self.stack.enter_context(patch.object(compiler, "capture_effective_flags",
                                              return_value={"policy": "same"}))
        def proof(directory, expected, destination, env):
            destination.mkdir(parents=True)
            for name in ("proof.json", "make.log", "link.log"):
                (destination / name).write_text("retained link proof")
            return {"passed": True, "expected_executable_sha256": expected}
        self.proof = self.stack.enter_context(patch.object(relink, "prove_object_relink", side_effect=proof))

    def test_copies_exact_objects_and_private_cards_then_requires_relink(self):
        before = {str(path.relative_to(self.original)): bench.digest(path)
                  for path in self.original.rglob("*") if path.is_file()}
        record = bench.reuse_baseline(self.case)
        self.assertEqual(record["campaign_binding"], {"id": "new", "manifest_sha256": "manifest"})
        self.assertTrue(record["reused_from"]["worker_only_relink_required"])
        self.assertTrue(record["reused_from"]["object_relink_proof"]["passed"])
        self.assertEqual(record["reused_from"]["numerical_objects"],
                         {"SubProcesses/P0_example/loop.o": bench.digest(self.sub / "loop.o")})
        self.assertEqual(bench.digest(self.target / "SubProcesses/P0_example/loop.o"),
                         bench.digest(self.sub / "loop.o"))
        # The inherited compilation binding intentionally stays old; build()
        # cannot use its cached-executable shortcut in the new namespace.
        copied = json.loads((self.target / "compilation.json").read_text())
        self.assertNotEqual(copied.get("campaign_binding"), bench.campaign_binding())
        (self.target / "Cards/MadLoopParams.dat").write_text("new private reducer settings")
        after = {str(path.relative_to(self.original)): bench.digest(path)
                 for path in self.original.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        with self.assertRaisesRegex(RuntimeError, "never overwrites"):
            bench.reuse_baseline(self.case)

    def test_rejects_active_or_unbound_baseline_without_copy(self):
        fifo = self.sub / "worker-protocol.fifo"
        fifo.touch()
        with self.assertRaisesRegex(RuntimeError, "worker is active"):
            bench.reuse_baseline(self.case)
        self.assertFalse(self.target.exists())
        fifo.unlink()
        compilation = json.loads((self.original / "compilation.json").read_text())
        compilation["numerical_build_from_sources"] = False
        bench.dump(self.original / "compilation.json", compilation)
        with self.assertRaisesRegex(RuntimeError, "frozen provenance"):
            bench.reuse_baseline(self.case)
        self.assertFalse(self.target.exists())

    def test_rejects_changed_native_dependencies(self):
        with patch.object(bench, "dependency_inputs", return_value={"archives": {"old.a": "changed"}, "metadata": {}}):
            with self.assertRaisesRegex(RuntimeError, "link dependencies changed"):
                bench.reuse_baseline(self.case)
        self.assertFalse(self.target.exists())

    def test_unproven_process_objects_are_not_copied(self):
        self.proof.side_effect = ValueError("Historical objects do not reproduce the executable")
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            bench.reuse_baseline(self.case)
        self.assertFalse(self.target.exists())


class TestObjectRelink(unittest.TestCase):
    def test_recipe_rejects_compilation_and_shell_steps(self):
        self.assertEqual(relink._link_recipe("gfortran -o check check_sa.o loop.o -lm\n")[0], "gfortran")
        for recipe in ("gfortran -c check_sa.f -o check\n", "touch x\ngfortran -o check check_sa.o\n",
                       "gfortran -o check check_sa.o && echo done\n"):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                relink._link_recipe(recipe)

    def test_replay_proves_used_objects_without_original_mutation(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "output/SubProcesses/P0_example"
            directory.mkdir(parents=True)
            (root / "output/lib").mkdir()
            for name in ("check_sa.o", "loop.o"):
                (directory / name).write_text(name)
            (directory / "check").write_text("old executable")
            expected = bench.digest(directory / "check")
            def execute(command, **kwargs):
                if command[0] == "make":
                    self.assertTrue(kwargs["cwd"] == directory)
                    self.assertIn("--old-file=loop.o", command)
                    return SimpleNamespace(stdout="gfortran -o check check_sa.o loop.o\n", stderr="")
                Path(command[command.index("-o") + 1]).write_text("old executable")
                return SimpleNamespace(returncode=0)
            with patch.object(relink.subprocess, "run", side_effect=execute):
                record = relink.prove_object_relink(directory, expected, root / "proof", {})
            self.assertTrue(record["passed"])
            self.assertEqual(bench.digest(directory / "check"), expected)
            self.assertEqual(record["original_link_command"], ["gfortran", "-o", "check", "check_sa.o", "loop.o"])
            self.assertTrue((root / "proof/proof.json").is_file())

    def test_changed_object_binary_is_retained_and_rejected(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "output/SubProcesses/P0_example"
            directory.mkdir(parents=True)
            (root / "output/lib").mkdir()
            (directory / "check_sa.o").write_text("changed object")
            (directory / "check").write_text("old executable")
            def execute(command, **kwargs):
                if command[0] == "make":
                    return SimpleNamespace(stdout="gfortran -o check check_sa.o\n", stderr="")
                Path(command[command.index("-o") + 1]).write_text("changed executable")
                return SimpleNamespace(returncode=0)
            with patch.object(relink.subprocess, "run", side_effect=execute):
                with self.assertRaisesRegex(ValueError, "full source rebuild"):
                    relink.prove_object_relink(directory, bench.digest(directory / "check"), root / "proof", {})
            self.assertFalse(json.loads((root / "proof/proof.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
