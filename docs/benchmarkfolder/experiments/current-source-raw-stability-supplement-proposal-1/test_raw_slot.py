"""Actual retained replay plus explicitly synthetic admission/error fixtures."""
import copy
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from prepare_preview import BUILD, DOCS, HELPER, HERE, DOSSIER, SELECTION, prepare


class TestRawSlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = json.loads((HERE / "preview.json").read_text())
        cls.typst = json.loads((HERE / "preview-receipt.json").read_text())["typst_inputs"]
        cls.inputs = BUILD.ReportInputs(DOCS)
        manifest = cls.inputs.json("campaigns/entry-latch-cache-guard-1/manifest.json", HELPER.MANIFEST)
        with patch.object(subprocess, "Popen", side_effect=AssertionError("process launch")), \
                patch.object(ctypes, "CDLL", side_effect=AssertionError("foreign call")), \
                patch.object(ctypes, "PyDLL", side_effect=AssertionError("foreign call")):
            cls.raw = HELPER.raw_stability(cls.inputs, HELPER.RAW_STABILITY, manifest)
        if cls.raw != cls.context["source_supplement"]["raw_stability"]:
            raise AssertionError("Actual retained raw replay differs from the prepared context")

    def test_actual_context_changes_only_the_optional_raw_slot(self):
        old = json.loads((DOCS / "campaigns/entry-latch-cache-guard-1/report-supplements/progress-1/report-context.json").read_text())
        current = copy.deepcopy(self.context)
        current["source_supplement"].pop("raw_stability")
        self.assertEqual(current, old)
        self.assertEqual(self.typst["context"], "campaigns/entry-latch-cache-guard-1/report-supplements/progress-2/report-context.json")
        self.assertFalse(self.context["delivery_complete"])

    def test_all_five_fixed_populations_and_failure_evidence_retained(self):
        rows = self.raw["summary"]["rows"]
        self.assertEqual([row["backend"] for row in rows], [1, 6, 7, 8, 9])
        self.assertEqual([row["available_points"] for row in rows], [100, 100, 100, 100, 95])
        self.assertEqual((self.raw["primary_calls"], self.raw["bootstrap_calls"]), (1500, 25))
        self.assertTrue(all(row["return_code_counts"] == {"110": 300} for row in rows))
        tirust = rows[-1]
        self.assertEqual(tirust["state"], "complete")
        self.assertEqual(tirust["primary_outcome_counts"], {"diagnostic_available": 285, "rejected": 15})
        self.assertEqual(tirust["active_group_failure_observations"], 27)
        self.assertEqual(tirust["qualified_incoming_clear_calls"], 5)
        self.assertEqual(tirust["legacy_extra_ineligible_calls"], 5)
        self.assertEqual([row["point"] for row in self.raw["provenance"]["unavailable_reasons"]["qcd_n3_uux_zgg-backend9"]], [25, 34, 45, 70, 84])
        self.assertEqual((tirust["tail_at_1e3"]["lower"], tirust["tail_at_1e3"]["upper"]), (0, .05))
        self.assertIsNone(tirust["tail_at_1e3"]["fraction"])

    def test_every_declared_package_payload_is_bound(self):
        for pin in HELPER.RAW_STABILITY.values():
            self.assertEqual(self.inputs.files[pin["path"]]["sha256"], pin["sha256"])
            index = json.loads((DOCS / pin["path"]).read_text())
            prefix = str(Path(pin["path"]).parent)
            for name, identity in index["files"].items():
                self.assertEqual(self.inputs.files[prefix + "/" + name], identity)

    def test_raw_source_is_current_and_distinct_from_controller_revision(self):
        p = self.raw["provenance"]
        self.assertEqual(p["campaign_manifest_sha256"], HELPER.MANIFEST)
        self.assertEqual(p["source_revision"], self.context["source_supplement"]["current"]["revision"])
        self.assertNotEqual(p["source_revision"], p["generation_receipt_revision"])
        self.assertFalse(self.raw["summary"]["numerical_acceptance_claimed"])
        self.assertFalse(self.raw["summary"]["hardware_ieee_state_observed"])
        self.assertFalse(self.raw["summary"]["performance_acceptance_claimed"])

    def test_absent_optional_role_does_not_probe_raw_data(self):
        old = json.loads((DOCS / "campaigns/entry-latch-cache-guard-1/report-supplements/progress-1/selection.json").read_text())
        with patch.object(HELPER, "raw_stability", side_effect=AssertionError("unexpected raw read")), \
                patch.object(subprocess, "Popen", side_effect=AssertionError("process launch")), \
                patch.object(ctypes, "CDLL", side_effect=AssertionError("foreign call")):
            view, _ = HELPER.validate(BUILD.ReportInputs(DOCS),
                "campaigns/entry-latch-cache-guard-1/report-supplements/progress-1/selection.json", HELPER.HISTORICAL, DOSSIER)
        self.assertNotIn("raw_stability", old)
        self.assertNotIn("raw_stability", view)
        wanted = copy.deepcopy(self.context["source_supplement"])
        wanted.pop("raw_stability")
        self.assertEqual(view, wanted)

    def test_unadmitted_role_rejected_before_any_payload_reads(self):
        for role in (None, {}, {**HELPER.RAW_STABILITY, "other": {}},
                     {**HELPER.RAW_STABILITY, "data_index": {"path": "active/index.json", "sha256": "0" * 64}}):
            with self.subTest(role=role), patch.object(HELPER, "bind_index", side_effect=AssertionError("payload read")):
                with self.assertRaisesRegex(ValueError, "exact admitted"):
                    HELPER.raw_stability(None, role, {})

    def synthetic(self, mutation):
        """Synthetic summary metadata only; both pure verifier calls are mocked."""
        raw = copy.deepcopy(self.raw)
        mutation(raw)
        manifest = json.loads((DOCS / "campaigns/entry-latch-cache-guard-1/manifest.json").read_text())
        dataset = {"provenance": raw["provenance"]}
        class Inputs:
            def json(self, name, expected=None):
                return manifest if name.endswith("manifest.json") else dataset
        response = {"verified": True, "index_sha256": HELPER.RAW_STABILITY["data_index"]["sha256"], "summary": raw["summary"]}
        with patch.object(HELPER, "bind_index", return_value=({"files": {"synthetic": {}}, "caption": "SYNTHETIC"}, {"synthetic"})), \
                patch.object(HELPER, "replay", return_value=response):
            return HELPER.raw_stability(Inputs(), HELPER.RAW_STABILITY, manifest)

    def test_synthetic_wrong_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "source or population"):
            self.synthetic(lambda raw: raw["provenance"].update(source_revision="wrong"))

    def test_synthetic_lost_unknowns_or_measured_failure_claim_rejected(self):
        mutations = [lambda raw: raw["summary"]["rows"].pop(),
            lambda raw: raw["summary"]["rows"][-1].update(available_points=100),
            lambda raw: raw["summary"]["rows"][-1]["tail_at_1e3"].update(fraction=.05),
            lambda raw: raw["summary"]["rows"][-1]["tail_at_1e3"].update(n=95),
            lambda raw: raw["summary"].update(performance_acceptance_claimed=True)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.synthetic(mutation)

    def test_corrupt_indexed_payload_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-corrupt-raw-") as directory:
            root = Path(directory)
            pin = {"path": "index.json", "sha256": "0" * 64}
            (root / "index.json").write_bytes(b"SYNTHETIC corrupt index")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                HELPER.bind_index(BUILD.ReportInputs(root), pin)

    def test_typst_append_only_preserves_reviewed_layout_and_palette(self):
        old = (HERE / "baseline-current-source-supplement.typ").read_text()
        new = (HERE / "proposed/current-source-supplement.typ").read_text()
        hook = '  if s.at("raw_stability", default: none) != none {\n    show-current-source-raw(s)\n  }\n'
        start = new.index("#let show-current-source-raw(s) = {")
        end = new.index("#let show-current-source(s) = {", start)
        self.assertEqual((new[:start] + new[end:]).replace(hook, "", 1), old)
        self.assertIn('table.cell(fill: ratio-fill(row.ratio)', new)
        self.assertIn('pagebreak()\n  heading(level: 1)[Raw stability pilot', new)
        self.assertIn('within 0–5%', new)


if __name__ == "__main__":
    unittest.main()
