"""Closed handoff bindings and real CLI parsing; every local action is mocked.

Reading campaign and historical metadata supplies no fresh numerical, runtime,
native-build or local-install result. This test never launches those actions.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex
import sys
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CAMPAIGN_ID = "entry-latch-cache-guard-1"
CAMPAIGN = ROOT / "docs/benchmarkfolder/campaigns" / CAMPAIGN_ID
INSTALLATION = "hs-entry-context-native-1"
MEASUREMENT = "hs-entry-context-local-1"
MANIFEST_SHA = "4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076"
INITIAL_SUMMARY_SHA = "7939fc1e617965e01249cb2921e6f8f2e7f76681328afa192ec0c068afdc54fe"
SOURCE_REVISION = "0fcd2cd873d61d134837e1d9e87afb13a47824fd"
SOURCE_FILES = "afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6"
NUMERICAL_IDENTITY = "3c13ad88b825d9d19b1030b62b317c0f31759e002d8dfdb9050142301a7cae24"
NATIVE_LIBRARY = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
PRODUCTION_TEMPLATES = {
    "madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc",
    "madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commands():
    text = (HERE / "LOCAL_TIMING_TODO.md").read_text()
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    return [shlex.split(line) for block in blocks
            for line in block.replace("\\\n", " ").splitlines()
            if line.startswith("python3 -B ")]


def source_module(name, path):
    previous_path, previous_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(ROOT / "tools"))
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode


class TestHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local = source_module("handoff_entry_context_local", ROOT / "tools/tirust_benchmark_local.py")

    def test_exact_new_campaign_source_native_and_pending_summary(self):
        binding = read(HERE / "bindings.json")
        for name, expected in binding["canonical_files"].items():
            self.assertEqual(digest(ROOT / name), expected, name)
        self.assertEqual(digest(CAMPAIGN / "manifest.json"), MANIFEST_SHA)
        manifest = read(CAMPAIGN / "manifest.json")
        initial = binding["initial_pending_summary"]
        self.assertEqual(initial["path"],
            "docs/benchmarkfolder/campaigns/" + CAMPAIGN_ID + "/timing-handoff-1/initial-summary.json")
        self.assertEqual(initial["origin_path"], "docs/benchmarkfolder/campaigns/" + CAMPAIGN_ID + "/summary.json")
        self.assertEqual(initial["sha256"], INITIAL_SUMMARY_SHA)
        self.assertEqual(digest(ROOT / initial["path"]), INITIAL_SUMMARY_SHA)
        self.assertNotIn(initial["origin_path"], binding["canonical_files"])
        summary = read(ROOT / initial["path"])
        self.assertEqual(summary["campaign_binding"], {"id": CAMPAIGN_ID, "manifest_sha256": MANIFEST_SHA})
        self.assertEqual(manifest["id"], CAMPAIGN_ID)
        identity = manifest["candidate"]["exporter_identity"]
        self.assertEqual(identity["revision"], SOURCE_REVISION)
        self.assertEqual(identity["revision"], binding["source_revision"])
        self.assertEqual(identity["files_sha256"], SOURCE_FILES)
        self.assertEqual(identity["files_sha256"], binding["source_files_sha256"])
        self.assertEqual(manifest["candidate"]["numerical_identity_sha256"], NUMERICAL_IDENTITY)
        self.assertEqual(manifest["candidate"]["numerical_identity_sha256"], binding["numerical_identity_sha256"])
        self.assertEqual(manifest["candidate"]["native"]["files"], binding["native_files"])
        self.assertEqual(binding["native_library_sha256"], NATIVE_LIBRARY)
        self.assertEqual(manifest["candidate"]["native"]["files"]["lib/libmadloop_tirust.so"], NATIVE_LIBRARY)
        self.assertFalse(summary["gate"]["complete"])
        self.assertFalse(summary["gate"]["passed"])
        self.assertEqual(summary["candidate_source_hashes"], [])
        self.assertEqual(len(summary["cases"]), 11)
        self.assertEqual(len(summary["gate"]["pending_cases"]), 8)
        self.assertTrue(all(case["status"] == "pending" for case in summary["cases"]))

    def test_two_production_templates_bind_raw_and_numerical_identities(self):
        binding = read(HERE / "bindings.json")
        manifest = read(CAMPAIGN / "manifest.json")
        self.assertEqual(set(binding["production_templates"]), PRODUCTION_TEMPLATES)
        for name, pins in binding["production_templates"].items():
            with self.subTest(template=name):
                self.assertEqual(digest(ROOT / name), pins["sha256"])
                self.assertEqual(manifest["candidate"]["exporter_identity"]["files"][name], pins["sha256"])
                self.assertEqual(manifest["candidate"]["numerical_sources"][name], pins)

    def test_fresh_local_pending_qualification_is_separate_from_shared_runtime_and_ancestry(self):
        binding = read(HERE / "bindings.json")
        expected = {"cases": 11, "lanes_per_case": 6, "total_lanes": 66, "points_per_lane": 100,
            "status": "pending", "observations": 0, "baseline": {"source": "baseline", "backend": 1},
            "candidate_backends": [1, 6, 7, 8, 9]}
        self.assertEqual(binding["qualification_requirements"], expected)
        self.assertEqual(expected["total_lanes"], expected["cases"] * expected["lanes_per_case"])
        self.assertEqual(len(read(ROOT / binding["initial_pending_summary"]["path"])["gate"]["pending_cases"]), 8)
        self.assertFalse(binding["actual_local_install_performed"])
        self.assertFalse(binding["actual_local_registry_created"])

    def test_declared_card_ancestry_only_retains_original_bytes(self):
        binding = read(HERE / "bindings.json")
        manifest = read(CAMPAIGN / "manifest.json")
        value = self.local.campaign.verify_physical_card_origins(self.local.bench, manifest)
        expected = binding["physical_input_ancestry"]
        self.assertEqual(value["scope"], "physical-input-ancestry-only")
        self.assertEqual(value["scope"], expected["scope"])
        self.assertEqual(value["origin"], expected["origin"])
        self.assertEqual(value["origin"]["id"], "invariant-symmetry-1")
        self.assertEqual(len(value["cards"]), 11)
        self.assertEqual(len(value["observations"]), expected["observations"])
        self.assertEqual(expected["observations"], 66)
        self.assertEqual(sum(len(item["files"]) for item in value["cards"].values()), expected["card_files"])
        self.assertEqual(expected["card_files"], 22)
        self.assertEqual(expected["total_bytes"], expected["observation_bytes"] + expected["card_bytes"])

    def test_legacy_failure_stays_in_recursive_predecessor_metadata(self):
        node = read(CAMPAIGN / "manifest.json")["predecessor"]
        identifiers, failures = [], []
        while node is not None:
            identifiers.append(node["id"])
            failures.extend((node["id"], row) for row in node["qualified_failures"])
            node = node["ancestor"]
        self.assertEqual(identifiers, ["invariant-symmetry-1", "split-dp-terminal-1", "legacy"])
        self.assertEqual(len(failures), 1)
        ancestor, failure = failures[0]
        self.assertEqual(ancestor, "legacy")
        self.assertEqual((failure["case"], failure["label"]), ("qcd_n1_uux_z", "tirust-first"))
        self.assertEqual(failure["directory"], "runs/qcd_n1_uux_z/tirust-first-attempt-3")
        self.assertEqual(len(failure["files"]), 101)

    def test_no_source_override_and_current_protocol_inventory_matches(self):
        binding = read(HERE / "bindings.json")
        identity = self.local.bench.source_identity(ROOT, persist=False)
        self.assertEqual(identity["files_sha256"], binding["source_files_sha256"])
        for name, expected in binding["protocol_files"].items():
            self.assertEqual(digest(ROOT / name), expected, name)

    def test_all_documented_launcher_commands_parse_without_actions(self):
        selected = [cmd for cmd in commands() if cmd[2] == "tools/tirust_benchmark_local.py"]
        self.assertTrue(selected)
        for command in selected:
            with self.subTest(command=command), contextlib.ExitStack() as stack:
                mocks = {name: stack.enter_context(patch.object(self.local, name, return_value={"mock": True}))
                    for name in ("install", "prepare", "verify", "forward")}
                argv = [argument.replace("$TIRUST_LOCAL_CPU", "9") for argument in command[2:]]
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if "--help" in argv:
                    with self.assertRaises(SystemExit) as error: self.local.main()
                    self.assertEqual(error.exception.code, 0)
                    self.assertTrue(all(mock.call_count == 0 for mock in mocks.values()))
                else:
                    self.local.main()
                    self.assertEqual(mocks[argv[1]].call_count, 1)
                    self.assertEqual(sum(mock.call_count for mock in mocks.values()), 1)
                    def option(name, default=None):
                        return argv[argv.index(name) + 1] if name in argv else default
                    if argv[1] == "install":
                        mocks["install"].assert_called_once_with(CAMPAIGN_ID, INSTALLATION,
                            Path(option("--tirust-prefix")), int(option("--jobs", "4")))
                    elif argv[1] == "prepare":
                        mocks["prepare"].assert_called_once_with(MEASUREMENT, CAMPAIGN_ID,
                            Path(option("--tirust-prefix")), 9, Path(option("--install-receipt")))
                    elif argv[1] == "verify":
                        mocks["verify"].assert_called_once_with(MEASUREMENT)
                    else:
                        stages = [value for value in argv[2:] if value in ("init", "model", "build", "numerical", "campaign", "summarize")]
                        self.assertEqual(len(stages), 1)
                        mocks["forward"].assert_called_once_with(MEASUREMENT, stages[0], int(option("--jobs", "4")))

    def test_install_receipt_and_fresh_stages_precede_local_timing(self):
        selected = [cmd for cmd in commands() if cmd[2] == "tools/tirust_benchmark_local.py" and "--help" not in cmd]
        self.assertEqual([cmd[3] for cmd in selected],
            ["install", "prepare", "verify", "forward", "forward", "forward", "forward", "verify", "forward", "forward"])
        self.assertEqual([cmd[cmd.index("--measurement") + 2] for cmd in selected if cmd[3] == "forward"],
            ["init", "model", "build", "numerical", "campaign", "summarize"])
        install, prepare = selected[:2]
        self.assertEqual(install[install.index("--installation") + 1], INSTALLATION)
        self.assertEqual(prepare[prepare.index("--install-receipt") + 1],
            "docs/benchmarkfolder/local-installations/" + INSTALLATION + "/receipt.json")
        self.assertEqual(install[install.index("--tirust-prefix") + 1], prepare[prepare.index("--tirust-prefix") + 1])
        for command in selected[1:]: self.assertEqual(command[command.index("--measurement") + 1], MEASUREMENT)
        self.assertFalse(any("tirust_benchmark_provisional.py" in " ".join(cmd) for cmd in commands()))

    def test_renderer_cli_uses_only_new_local_registry(self):
        renderer = source_module("handoff_entry_context_renderer", ROOT / "docs/benchmarkfolder/build_local_report.py")
        command, = [cmd for cmd in commands() if cmd[2] == "docs/benchmarkfolder/build_local_report.py"]
        with patch.object(renderer, "build", return_value={"mock": True}) as build, \
                patch.object(sys, "argv", command[2:]), contextlib.redirect_stdout(io.StringIO()):
            renderer.main()
        build.assert_called_once_with(MEASUREMENT, "render-1", False)

    def test_fresh_full_hundred_point_policy_six_lanes_and_seven_pairs(self):
        config = read(CAMPAIGN / "cases.json")
        self.assertEqual(len(config["cases"]), 11)
        self.assertEqual(config["backends"], [1, 6, 7, 8, 9])
        self.assertEqual(config["points_per_case"], 100)
        self.assertGreaterEqual(config["repetitions"], 7)
        self.assertEqual(config["minimum_batch_seconds"], 1.0)
        self.assertEqual(config["performance_upper_ratio"], 1.05)
        self.assertEqual(config["relative_tolerance"], 3e-6)
        self.assertEqual(config["madloop_parameters"]["MLStabThres"], 1e-6)
        text = (HERE / "LOCAL_TIMING_TODO.md").read_text()
        self.assertIn("0.95-second observed batch floor", text)
        self.assertIn("all eleven cases", text)

    def test_prior_handoffs_unchanged_and_links_resolve(self):
        binding = read(HERE / "bindings.json")
        self.assertTrue(binding["preserved_handoffs"])
        for name, expected in binding["preserved_handoffs"].items():
            self.assertEqual(digest(ROOT / name), expected, name)
        published = binding["published_report"]
        self.assertRegex(published["commit"], r"^[0-9a-f]{40}$")
        for name in ("pdf_sha256", "ledger_sha256", "dossier_sha256"):
            self.assertRegex(published[name], r"^[0-9a-f]{64}$")
        self.assertEqual(published["dossier_path"],
            "campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-1.json")
        publication = "https://github.com/MadGraphTeam/MadGraph7/blob/" + published["commit"] + "/docs/benchmarkfolder/benchmark.pdf"
        links = re.findall(r"\]\(([^)]+)\)", (HERE / "LOCAL_TIMING_TODO.md").read_text())
        self.assertIn(publication, links)
        for link in links:
            if link.startswith("https://"):
                self.assertEqual(link, publication)
            else:
                self.assertTrue((HERE / link.split("#", 1)[0]).resolve().is_file(), link)

    def test_pinned_shell_activation_and_real_entrypoint_are_explicit(self):
        text = (HERE / "LOCAL_TIMING_TODO.md").read_text()
        self.assertIn("nix develop\nsource tools/tirust_env.sh", text)
        self.assertIn("`flake.nix`/`flake.lock`", text)
        self.assertIn("`x86_64-linux`", text)
        self.assertIn("it does not install missing dependencies", text)
        entrypoint = read(HERE / "bindings.json")["entrypoint"]
        self.assertEqual(entrypoint["path"], "bin/madgraph")
        launcher = ROOT / entrypoint["path"]
        self.assertTrue(launcher.is_file())
        self.assertFalse(launcher.is_symlink())
        self.assertEqual(digest(launcher), entrypoint["sha256"])
        self.assertEqual(read(CAMPAIGN / "manifest.json")["candidate"]["exporter_identity"]["files"][entrypoint["path"]],
            entrypoint["sha256"])


if __name__ == "__main__":
    unittest.main()
