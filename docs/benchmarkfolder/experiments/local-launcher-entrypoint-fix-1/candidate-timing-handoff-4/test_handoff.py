"""Read-only handoff binding and actual CLI parser checks; no native actions."""
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
CAMPAIGN = ROOT / "docs/benchmarkfolder/campaigns/invariant-symmetry-1"
TARGET = CAMPAIGN / "timing-handoff-4"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark_local as local


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


class TestHandoff(unittest.TestCase):
    def test_exact_current_campaign_and_pending_summary(self):
        binding = read(HERE / "bindings.json")
        for name, expected in binding["canonical_files"].items():
            self.assertEqual(digest(ROOT / name), expected)
        manifest = read(CAMPAIGN / "manifest.json")
        summary = read(CAMPAIGN / "summary.json")
        self.assertEqual(manifest["id"], "invariant-symmetry-1")
        self.assertEqual(manifest["candidate"]["exporter_identity"]["revision"], binding["source_revision"])
        self.assertEqual(manifest["candidate"]["exporter_identity"]["files_sha256"], binding["source_files_sha256"])
        self.assertEqual(manifest["candidate"]["numerical_identity_sha256"], binding["numerical_identity_sha256"])
        self.assertFalse(summary["gate"]["complete"])
        self.assertFalse(summary["gate"]["passed"])
        self.assertEqual(summary["candidate_source_hashes"], [])
        self.assertEqual(len(summary["cases"]), 11)
        self.assertEqual(len(summary["gate"]["pending_cases"]), 8)
        self.assertTrue(all(case["status"] == "pending" for case in summary["cases"]))

    def test_declared_actual_card_ancestry_only(self):
        value = local.campaign.verify_physical_card_origins(local.bench, read(CAMPAIGN / "manifest.json"))
        self.assertEqual(value["scope"], "physical-input-ancestry-only")
        self.assertEqual(value["origin"]["id"], "split-dp-terminal-1")
        self.assertEqual(len(value["cards"]), 11)
        self.assertEqual(len(value["observations"]), 66)
        self.assertEqual(sum(len(item["files"]) for item in value["cards"].values()), 22)
        binding = read(HERE / "bindings.json")
        self.assertFalse(binding["actual_local_install_performed"])
        self.assertFalse(binding["actual_local_registry_created"])

    def test_no_source_override_and_current_inventory_matches(self):
        binding = read(HERE / "bindings.json")
        identity = local.bench.source_identity(ROOT, persist=False)
        self.assertEqual(identity["files_sha256"], binding["source_files_sha256"])
        for name, expected in binding["protocol_files"].items():
            self.assertEqual(digest(ROOT / name), expected, name)

    def test_all_documented_launcher_commands_parse_without_execution(self):
        selected = [cmd for cmd in commands() if cmd[2] == "tools/tirust_benchmark_local.py"]
        for command in selected:
            with self.subTest(command=command), contextlib.ExitStack() as stack:
                mocks = {name: stack.enter_context(patch.object(local, name, return_value={"mock": True}))
                         for name in ("install", "prepare", "verify", "forward")}
                argv = [argument.replace("$TIRUST_LOCAL_CPU", "9") for argument in command[2:]]
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if "--help" in argv:
                    with self.assertRaises(SystemExit) as error:
                        local.main()
                    self.assertEqual(error.exception.code, 0)
                    self.assertTrue(all(mock.call_count == 0 for mock in mocks.values()))
                else:
                    local.main()
                    self.assertEqual(mocks[argv[1]].call_count, 1)
                    self.assertEqual(sum(mock.call_count for mock in mocks.values()), 1)
                    if argv[1] in ("install", "prepare"):
                        self.assertEqual(mocks[argv[1]].call_args.args[1 if argv[1] == "prepare" else 0], "invariant-symmetry-1")

    def test_install_receipt_and_all_fresh_stages_precede_timing(self):
        selected = [cmd for cmd in commands() if cmd[2] == "tools/tirust_benchmark_local.py" and "--help" not in cmd]
        self.assertEqual([cmd[3] for cmd in selected],
                         ["install", "prepare", "verify", "forward", "forward", "forward", "forward", "verify", "forward", "forward"])
        self.assertEqual([cmd[cmd.index("--measurement") + 2] for cmd in selected if cmd[3] == "forward"],
                         ["init", "model", "build", "numerical", "campaign", "summarize"])
        install, prepare = selected[:2]
        self.assertEqual(install[install.index("--installation") + 1], "hs-symmetry-native-1")
        self.assertEqual(prepare[prepare.index("--install-receipt") + 1],
                         "docs/benchmarkfolder/local-installations/hs-symmetry-native-1/receipt.json")
        self.assertEqual(install[install.index("--tirust-prefix") + 1], prepare[prepare.index("--tirust-prefix") + 1])
        for command in selected[1:]:
            self.assertEqual(command[command.index("--measurement") + 1], "hs-symmetry-local-1")
        self.assertFalse(any("tirust_benchmark_provisional.py" in " ".join(cmd) for cmd in commands()))

    def test_renderer_cli_uses_new_local_registry_only(self):
        spec = importlib.util.spec_from_file_location("handoff_local_renderer", ROOT / "docs/benchmarkfolder/build_local_report.py")
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        command, = [cmd for cmd in commands() if cmd[2] == "docs/benchmarkfolder/build_local_report.py"]
        with patch.object(renderer, "build", return_value={"mock": True}) as build, \
             patch.object(sys, "argv", command[2:]), contextlib.redirect_stdout(io.StringIO()):
            renderer.main()
        build.assert_called_once_with("hs-symmetry-local-1", "render-1", False)

    def test_policy_is_fresh_full100_six_lanes_seven_pairs(self):
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
        self.assertIn("baseline CutTools and candidate reducers\n1/6/7/8/9", text)
        self.assertIn("0.95-second observed batch floor", text)
        self.assertIn("all eleven cases", text)

    def test_handoff2_unchanged_and_document_links_resolve(self):
        binding = read(HERE / "bindings.json")
        for name, checksum in binding["preserved_handoff2"].items():
            self.assertEqual(digest(ROOT / name), checksum)
        for link in re.findall(r"\]\(([^)]+)\)", (HERE / "LOCAL_TIMING_TODO.md").read_text()):
            self.assertTrue((TARGET / link.split("#", 1)[0]).resolve().is_file(), link)

    def test_pinned_nix_shell_activation_is_explicit(self):
        text = (HERE / "LOCAL_TIMING_TODO.md").read_text()
        self.assertIn("nix develop\nsource tools/tirust_env.sh", text)
        self.assertIn("`flake.nix`/`flake.lock`", text)
        self.assertIn("`x86_64-linux`", text)
        self.assertIn("it does not install missing dependencies", text)

    def test_handoff3_preserved_and_real_entrypoint_hash_bound(self):
        binding = read(HERE / "bindings.json")
        for name, checksum in binding["preserved_handoff3"].items():
            self.assertEqual(digest(ROOT / name), checksum)
        entrypoint = binding["entrypoint"]
        launcher = ROOT / entrypoint["path"]
        self.assertEqual(entrypoint["path"], "bin/madgraph")
        self.assertTrue(launcher.is_file())
        self.assertFalse(launcher.is_symlink())
        self.assertEqual(digest(launcher), entrypoint["sha256"])
        manifest = read(CAMPAIGN / "manifest.json")
        self.assertEqual(manifest["candidate"]["exporter_identity"]["files"][entrypoint["path"]],
                         entrypoint["sha256"])


if __name__ == "__main__":
    unittest.main()
