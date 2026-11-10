"""Pure continuation checks; no CLI, compiler, reducer or native child."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("remaining_supervisor_test", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class ContinuationTests(unittest.TestCase):
    def test_exact_remaining_order(self):
        self.assertEqual([case["id"] for case in run.cases()], run.ORDER)
        self.assertEqual(len(set(run.ORDER)), 9)
        self.assertFalse(set(run.ORDER) & {"qcd_n1_uux_z", "ew_n1_udx_wp"})

    def test_every_case_uses_existing_four_stage_cli(self):
        for case in run.cases():
            with self.subTest(case=case["id"]):
                rows = run.schedule(case)
                self.assertEqual([r["id"] for r in rows[:3]], ["reuse-baseline", "generate-candidate", "build-both"])
                for row in rows[:3]:
                    self.assertEqual(row["command"][:3], ["taskset", "-c", "12-15"])
                    i = row["command"].index("--cases")
                    self.assertEqual(row["command"][i+1:i+3], [case["id"], "--cpu"])
                command = rows[-1]["command"]
                i = command.index("--backends")
                self.assertEqual(command[i+1:i+6], ["1", "6", "7", "8", "9"])
                self.assertIn("--warmup-only", command)
                self.assertNotIn("--timeout", command)
                self.assertEqual(command[-2:], ["--cpu", "12"])
                self.assertEqual(rows[-1]["case"], case["id"])

    def test_caps_do_not_change_registered_worker_policy(self):
        manifest = run.read(run.REPORT / "manifest.json")
        for case in run.cases():
            n = case["nfinal"]
            expected = [600., 900., 1200., 7200.] if n <= 2 else [600., 1800., 3600., 7200.] if n == 3 else [600., 3600., 7200., 28800.]
            self.assertEqual([row["cap_seconds"] for row in run.schedule(case)], expected)
            self.assertEqual(manifest["administrative_worker_timeout_seconds"][case["id"]], 14400. if n == 4 else 3600.)

    def test_exact_reviewed_stage_cleanup_and_replay_remain_in_use(self):
        self.assertEqual(run.digest(run.ADAPTER), run.ADAPTER_SHA)
        self.assertEqual(run.adapter.run_stage.__module__, "remaining_n1_support")
        cleanup, preparation, numerical = run.adapter.load_support()
        self.assertIs(preparation.OwnedChild, cleanup.OwnedChild)
        self.assertEqual(len(numerical.EXPECTED_LANES), 6)

    def test_default_never_spawns_or_activates(self):
        with patch.object(sys, "argv", ["run.py"]), patch.object(run, "verify_seal"), \
             patch.object(run.campaign, "activate_campaign") as activate, \
             patch.object(run.adapter.subprocess, "Popen") as child, redirect_stdout(io.StringIO()) as stream:
            run.main()
        activate.assert_not_called(); child.assert_not_called()
        value = json.loads(stream.getvalue())
        self.assertEqual(len(value["cases"]), 9)
        self.assertFalse(value["performance_acceptance_eligible"])

    def test_per_case_chain_is_preserved_and_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); output = folder / "outputs/candidate/li_gg_h"
            for source in ("baseline", "candidate"):
                target = folder / "outputs" / source / "li_gg_h"
                (target / "Cards").mkdir(parents=True)
                (target / "Cards/param_card.dat").write_text("physical")
                run.dump(target / "generation.json", {"campaign_binding": {"id": "test"},
                    "parameter_card_sha256": run.digest(target / "Cards/param_card.dat"), "default_chain": "7|6|1"})
            run.dump(folder / "preflight.json", {"preflight": {"baseline_records": {"li_gg_h": {
                "param_card_sha256": run.digest(output / "Cards/param_card.dat"), "default_chain": [7, 6, 1]}}}})
            with patch.object(run, "HERE", folder), patch.object(run.adapter, "CASES", ["li_gg_h"]), \
                 patch.object(run.bench, "WORK", folder), patch.object(run.bench, "campaign_binding", return_value={"id": "test"}):
                self.assertEqual(len(run.declared_cards()), 2)
                value = run.read(output / "generation.json");value["default_chain"]="8|6|7|1"
                run.dump(output / "generation.json", value)
                with self.assertRaisesRegex(ValueError, "default chain"):
                    run.declared_cards()


if __name__ == "__main__":
    unittest.main()
