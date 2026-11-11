"""Synthetic filesystem tests only; never import the production benchmark."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("pure_preparation", Path(__file__).with_name("preparation.py"))
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)


def forbidden(*args, **kwargs):
    raise AssertionError("Native/compiler/loader/source-query path was called")


class Fixture:
    def __init__(self, root):
        self.root = root
        report, work = root / "report", root / "original-work"
        self.bench = SimpleNamespace(ROOT=root, REPORT=report, WORK=work, MODEL_ROOT=work / "models", digest=digest,
            runtime_identity=forbidden, environment=forbidden, source_identity=forbidden, campaign_guard=forbidden)
        self.bench.generated_sources = lambda output: {
            path.relative_to(output).as_posix(): digest(path) for scope in ("Source", "SubProcesses")
            for path in sorted((output / scope).rglob("*")) if path.is_file() and path.suffix in (".f", ".f90")}
        self.bench.numerical_card_inputs = lambda output: {
            path.relative_to(output).as_posix(): digest(path) for path in
            [output / "Cards/MadLoopParams.dat", *(output / "SubProcesses").rglob("MadLoopParams.dat")] if path.is_file()}
        self.bench.output_dir = lambda source, case: work / "outputs" / source / case["id"]
        self.bench.resolved_parameter_card = lambda directory: directory.parents[1] / "Cards/param_card.dat"
        self.bench.dependency_inputs = self.dependencies
        self.bench.verify_worker_template = self.verify_template
        self.cases = [{"id": case, "nfinal": 1 if "n1" in case else 2, "model": "loop_sm"} for case in prep.CASES]
        self.config = {"cases": self.cases, "points_per_case": 100}
        self.bench.load_config = lambda: self.config
        write(root / "generator.py", "frozen generator\n")
        write(root / "tools/tirust_benchmark_worker.f", "frozen worker\n")
        write(work / "models" / prep.MODEL / "particles.py", "frozen model\n")
        source = {"files": {"generator.py": digest(root / "generator.py"), "absent-generator.py": None}}
        source["files_sha256"] = hashlib.sha256(json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.manifest = {"candidate": {"exporter_identity": source}, "shared_inputs": {"files": {},
            "model_tree": {"particles.py": digest(work / "models" / prep.MODEL / "particles.py")}}}
        for case in self.cases:
            points = report / "points" / (case["id"] + ".dat")
            write(points, "100\nsynthetic saved stream\n")
            write(points.with_suffix(".json"), {"sha256": digest(points),
                  "inputs": {"count": 100, "incoming_masses": [0, 0], "masses": [1] * case["nfinal"]}})
            for path in (points, points.with_suffix(".json")):
                self.manifest["shared_inputs"]["files"][path.relative_to(report).as_posix()] = digest(path)
        write(report / "manifest.json", self.manifest)
        self.binding = {"id": "synthetic", "manifest_sha256": digest(report / "manifest.json")}
        self.bench.CAMPAIGN_MANIFEST = self.manifest
        self.bench.campaign_binding = lambda: self.binding
        for folder, value in (("native/one", "first loader"), ("native/two", "second loader")):
            write(root / folder / "ld-linux.so.2", value)
        write(root / "native/libreducer.a", "synthetic archive")
        write(root / "native/tirust-config.json", {"synthetic": True})
        for case in self.cases:
            output = self.bench.output_dir("candidate", case)
            directory = output / "SubProcesses/P0_fake"
            write(output / "Cards/param_card.dat", "physical card\n")
            write(output / "Cards/MadLoopParams.dat", "backend9\n")
            write(output / "Source/model.f", "model source\n")
            write(output / "Source/model.o", "model object\n")
            write(directory / "check_sa.f", "frozen worker\n")
            write(directory / "check_sa.o", "worker object\n")
            write(directory / "check", "SYNTHETIC NOT EXECUTABLE\n")
            write(directory / "warmup.dat", "old scratch preserved until worker starts\n")
            resources = output / "SubProcesses/MadLoop5_resources"
            write(resources / "P0_HelFilter.dat", "original filter\n")
            (directory / "MadLoopParams.dat").symlink_to("../../Cards/MadLoopParams.dat")
            (directory / "MadLoop5_resources").symlink_to("../MadLoop5_resources", target_is_directory=True)
            (output / "lib").mkdir()
            for name in ("libreducer.a", "tirust-config.json"):
                (output / "lib" / name).symlink_to(root / "native" / name)
            generation = {"source": "candidate", "case": case, "campaign_binding": self.binding,
                          "source_identity": source, "parameter_card_sha256": digest(output / "Cards/param_card.dat")}
            dependencies = self.dependencies(directory)
            compilation = {"campaign_binding": self.binding, "sources": self.bench.generated_sources(output),
                "numerical_build_from_sources": True, "executable_sha256": digest(directory / "check"),
                "worker_template_sha256": digest(root / "tools/tirust_benchmark_worker.f"),
                "masses": [0, 0] + [1] * case["nfinal"], "link_inputs_before": dependencies,
                "link_inputs_after": dependencies, "effective_compiler_flags": {"recorded_only": True}}
            write(output / "generation.json", generation)
            write(output / "compilation.json", compilation)
            libraries = {str(root / folder / "ld-linux.so.2"): digest(root / folder / "ld-linux.so.2")
                         for folder in ("native/one", "native/two")}
            runtime = {"libraries": libraries, **dependencies, "ldd": "historical loader text with old executable alias\n"}
            points = report / "points" / (case["id"] + ".dat")
            inputs = {"campaign_binding": self.binding, "generation": generation, "compilation": compilation,
                "runtime": runtime, "param_card_sha256": generation["parameter_card_sha256"],
                "points_sha256": digest(points), "points_metadata_sha256": digest(points.with_suffix(".json")),
                "numerical_cards": self.bench.numerical_card_inputs(output)}
            after = {"runtime": runtime, "executable_sha256": compilation["executable_sha256"],
                     "points_sha256": digest(points), "numerical_cards": inputs["numerical_cards"]}
            closed = report / "experiments/numerical" / ("full100-warmup-" + case["id"] + "-1") / "candidate-9"
            for name, value in (("candidate-inputs.json", inputs), ("inputs-after.json", after),
                                ("result.json", {"status": "complete", "source": "candidate", "backend": 9})):
                write(closed / name, value)

    def dependencies(self, directory):
        library = directory.parents[1] / "lib"
        return {"archives": {str((library / "libreducer.a").resolve()): digest(library / "libreducer.a")},
                "metadata": {str((library / "tirust-config.json").resolve()): digest(library / "tirust-config.json")}}

    @staticmethod
    def verify_template(compilation, template):
        if digest(template) != compilation["worker_template_sha256"]:
            raise ValueError("worker changed")


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture = Fixture(self.root)
        self.bench = self.fixture.bench

    def prepare(self):
        return prep.prepare(self.bench, self.root / "private", self.root / "records")

    def test_all_fifteen_clones_are_private_and_pure(self):
        original = prep.collect_originals(self.bench)
        result = self.prepare()
        self.assertEqual(result["clone_count"], 15)
        self.assertFalse(result["native_queries_performed"])
        self.assertEqual(original, prep.collect_originals(self.bench))
        for key, row in result["clones"].items():
            output = Path(row["private_output"])
            self.assertTrue((output / "lib").is_symlink())
            self.assertEqual((output / "SubProcesses/P0_fake/MadLoopParams.dat").readlink(), Path("../../Cards/MadLoopParams.dat"))
            self.assertTrue((output / "SubProcesses/P0_fake/warmup.dat").is_file())
            self.assertTrue((output / "SubProcesses/MadLoop5_resources/P0_HelFilter.dat").is_file())
            self.assertEqual(prep.verify_clone(self.bench, output, row)["status"], "verified")
        libraries = original[prep.CASES[0]]["runtime_files"]["libraries"]
        self.assertEqual(len(libraries), 2)  # Same basename, different paths/bytes.

    def test_private_configure_does_not_mutate_original_card(self):
        result = self.prepare()
        row = result["clones"][prep.CASES[0] + "/1"]
        output = Path(row["private_output"])
        (output / "Cards/MadLoopParams.dat").write_text("backend1\n")
        prep.verify_clone(self.bench, output, row)
        self.assertEqual(prep.collect_originals(self.bench), result["originals"])

    def test_original_card_mutation_changes_guard_inventory(self):
        before = prep.collect_originals(self.bench)
        (Path(before[prep.CASES[0]]["original_output"]) / "Cards/MadLoopParams.dat").write_text("changed\n")
        self.assertNotEqual(before, prep.collect_originals(self.bench))

    def test_runtime_change_rejected_without_loader(self):
        (self.root / "native/one/ld-linux.so.2").write_text("changed")
        with self.assertRaisesRegex(ValueError, "runtime file changed"):
            prep.collect_originals(self.bench)

    def test_expected_absent_source_rejects_new_file(self):
        prep.shared_bindings(self.bench)
        write(self.root / "absent-generator.py", "new unexpected file")
        with self.assertRaisesRegex(ValueError, "Registered current source changed"):
            prep.shared_bindings(self.bench)

    def test_private_object_change_rejected(self):
        row = self.prepare()["clones"][prep.CASES[0] + "/9"]
        output = Path(row["private_output"])
        (output / "Source/model.o").write_text("changed object")
        with self.assertRaisesRegex(ValueError, "objects changed"):
            prep.verify_clone(self.bench, output, row)

    def test_external_mutable_link_fails_and_retains_attempt(self):
        original = self.bench.output_dir("candidate", self.fixture.cases[0])
        target = original / "SubProcesses/P0_fake/MadLoopParams.dat"
        target.unlink()
        target.symlink_to(original / "Cards/MadLoopParams.dat")
        with self.assertRaisesRegex(ValueError, "Mutable card/resource escapes"):
            self.prepare()
        self.assertTrue((self.root / "records/preparation-failure.json").is_file())
        self.assertFalse((self.root / "records/preparation.json").exists())

    def test_active_original_is_rejected_before_copy(self):
        output = self.bench.output_dir("candidate", self.fixture.cases[0])
        os.mkfifo(output / "SubProcesses/P0_fake/worker-protocol.fifo")
        with self.assertRaisesRegex(ValueError, "worker is active"):
            self.prepare()
        self.assertFalse((self.root / "private").exists())

    def test_append_once_and_partial_failure_preserved(self):
        self.prepare()
        before = (self.root / "records/preparation.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()
        self.assertEqual(before, (self.root / "records/preparation.json").read_bytes())

    def test_closed_runtime_must_match_original_provenance(self):
        path = self.bench.REPORT / "experiments/numerical" / ("full100-warmup-" + prep.CASES[0] + "-1") / "candidate-9/candidate-inputs.json"
        value = prep.read(path)
        value["compilation"]["executable_sha256"] = "0" * 64
        write(path, value)
        with self.assertRaisesRegex(ValueError, "Closed runtime observation does not bind"):
            prep.collect_originals(self.bench)


if __name__ == "__main__":
    unittest.main()
