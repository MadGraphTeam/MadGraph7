"""Actual retained preview checks and synthetic whitelist refusal cases."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from prepare_preview import BUILD, DOCS, HELPER, HERE

NEW_CASES = {
    "qcd_n4_uux_zzgg": {
        "counts": [(88, 12), (88, 12), (97, 3), (100, 0), (96, 4), (14, 86)],
        "misses": {6: 88, 7: 80, 8: 80, 9: 78},
        "columns": {6: [0, 88, 0, 0], 7: [0, 80, 0, 0], 8: [0, 80, 0, 0], 9: [0, 78, 0, 0]},
    },
    "ew_n4_epem_mupmumaa": {
        "counts": [(93, 7), (93, 7), (100, 0), (100, 0), (100, 0), (40, 60)],
        "misses": {6: 12, 7: 10, 8: 14, 9: 8},
        "columns": {6: [0, 10, 2, 0], 7: [0, 8, 0, 2], 8: [0, 10, 2, 2], 9: [0, 8, 0, 0]},
    },
}

class TestCampaignExtension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preview = json.loads((HERE / "preview.json").read_text())
        cls.old = json.loads((DOCS / "campaigns/entry-latch-cache-guard-1/report-supplements/progress-2/report-context.json").read_text())
        cls.selection = json.loads((HERE / "selection.json").read_text())
        cls.numerical = cls.preview["source_supplement"]["numerical"]

    def test_all_previous_context_outside_numerical_is_identical(self):
        before, after = copy.deepcopy(self.old), copy.deepcopy(self.preview)
        before["source_supplement"].pop("numerical")
        after["source_supplement"].pop("numerical")
        self.assertEqual(before, after)
        self.assertFalse(after["delivery_complete"])
        self.assertEqual(after["source_supplement"]["local"]["status"], "pending")

    def test_previous_closed_cases_are_identical(self):
        now = {row["id"]: row for row in self.numerical["cases"]}
        for row in self.old["source_supplement"]["numerical"]["cases"]:
            if row["status"] != "pending":
                self.assertEqual(row, now[row["id"]])
        self.assertEqual(self.numerical["closed_cases"], 9 + len(NEW_CASES))
        self.assertEqual(self.numerical["closed_lanes"], 54 + 6 * len(NEW_CASES))
        self.assertEqual(self.numerical["qualified_cases"], self.numerical["closed_cases"])

    def test_new_closed_lanes_keep_precision_and_target_misses(self):
        by_case = {row["id"]: row for row in self.numerical["cases"]}
        for case, expected in NEW_CASES.items():
            row = by_case[case]
            lanes = row["evidence"]["lanes"]
            self.assertTrue(row["regression_pass"])
            self.assertEqual(len(lanes), 6)
            self.assertEqual([(v["source"], v["backend"]) for v in lanes], list(HELPER.LANES))
            self.assertEqual([(v["accepted_backend_precision_counts"].get(str(v["backend"])+":DP",0),
                               v["accepted_backend_precision_counts"].get(str(v["backend"])+":QP",0))
                              for v in lanes], expected["counts"])
            misses = [v for v in self.numerical["target_misses"] if v["case"] == case]
            self.assertEqual({v["backend"]: v["scalar_misses"] for v in misses}, expected["misses"])
            self.assertEqual({v["backend"]: [v["columns"][str(i)] for i in range(4)] for v in misses}, expected["columns"])
            self.assertTrue(all(sum(v["columns"].values()) == v["scalar_misses"] for v in misses))
            self.assertTrue(all(v["matched_saved_points"] == 100 for v in lanes))
            self.assertTrue(all(v["performance_acceptance_eligible"] is False for v in lanes))

    def test_new_checkpoint_and_every_payload_are_bound(self):
        bound = json.loads((HERE / "preview-inputs.json").read_text())
        for selected in self.selection["numerical_checkpoints"]:
            pin = selected["closure_index"]
            self.assertEqual(bound[pin["path"]]["sha256"], pin["sha256"])
            closure = json.loads((DOCS / pin["path"]).read_text())
            for name, identity in closure["files"].items():
                self.assertEqual(bound[HELPER.COMBINED_PROTOCOL_ALIASES.get(name, name.removeprefix("docs/benchmarkfolder/"))], identity)

    def test_old_selections_remain_admitted(self):
        for progress in (1,2):
            old = json.loads((DOCS / f"campaigns/entry-latch-cache-guard-1/report-supplements/progress-{progress}/selection.json").read_text())
            HELPER.admit_numerical(old["numerical_checkpoints"])
        self.assertEqual(self.selection["raw_stability"], HELPER.RAW_STABILITY)
        self.assertEqual(self.selection["context_path"], "campaigns/entry-latch-cache-guard-1/report-supplements/progress-3/report-context.json")

    def test_unadmitted_hash_and_duplicate_rejected_before_reads(self):
        bad = copy.deepcopy(self.selection["numerical_checkpoints"][-1])
        bad["closure_index"]["sha256"] = "0" * 64
        duplicate = self.selection["numerical_checkpoints"] + self.selection["numerical_checkpoints"][-1:]
        with patch.object(HELPER, "bind_index", side_effect=AssertionError("unexpected payload read")):
            for selected in ([], [bad], duplicate):
                with self.subTest(selected=selected), self.assertRaises(ValueError):
                    HELPER.numerical(None, selected, None, None)

    def test_canonical_typst_raw_figure_and_baseline_files_unchanged(self):
        baseline = json.loads((HERE / "baselines.json").read_text())
        for path, wanted in baseline.items():
            actual = (DOCS / path).read_bytes()
            self.assertEqual({"sha256": hashlib.sha256(actual).hexdigest(), "bytes":len(actual)}, wanted)
        self.assertEqual(self.preview["source_supplement"]["raw_stability"], self.old["source_supplement"]["raw_stability"])

    def test_baseline_build_caveat_is_retained_and_extended(self):
        notes = self.numerical["build_scope_notes"]
        self.assertEqual(notes[0], self.old["source_supplement"]["numerical"]["build_scope_notes"][0])
        self.assertTrue(any("QCD n4" in note and "155 unchanged inherited baseline objects" in note
                            and "original baseline full-build receipt was not located" in note for note in notes))

    def test_combined_attempt_note_and_actual_outcomes_are_preserved(self):
        row = next(row for row in self.numerical["cases"] if row["id"] == "ew_n4_epem_mupmumaa")
        self.assertEqual((row["accepted_dp"], row["accepted_qp"]), (526, 74))
        self.assertTrue(any("saved point 74" in note and "No partial streams were concatenated" in note
                            for note in self.numerical["build_scope_notes"]))
        self.assertTrue(any("320 unchanged inherited baseline objects, 371 generated source pins" in note
                            for note in self.numerical["build_scope_notes"]))
        lanes = [lane for case in self.numerical["cases"] for lane in case["evidence"]["lanes"]]
        self.assertEqual(sum(lane["ordinary_target_passed"] for lane in lanes), 54)
        self.assertEqual(sum(row["accepted_dp"] for row in self.numerical["cases"]), 6396)
        self.assertEqual(sum(row["accepted_qp"] for row in self.numerical["cases"]), 204)

    def test_combined_protocol_aliases_require_identical_indexed_copies(self):
        selected = self.selection["numerical_checkpoints"][-1]
        closure = json.loads((DOCS / selected["closure_index"]["path"]).read_text())
        self.assertEqual(set(HELPER.COMBINED_PROTOCOL_ALIASES),
                         {name for name in closure["files"] if not name.startswith("docs/benchmarkfolder/")})
        for original, alias in HELPER.COMBINED_PROTOCOL_ALIASES.items():
            self.assertEqual(closure["files"][original], closure["files"]["docs/benchmarkfolder/" + alias])
        first = next(iter(HELPER.COMBINED_PROTOCOL_ALIASES.values()))
        for mutation in ("wrong", "missing"):
            bad = copy.deepcopy(closure)
            if mutation == "wrong":
                bad["files"]["docs/benchmarkfolder/" + first]["sha256"] = "0" * 64
            else:
                del bad["files"]["docs/benchmarkfolder/" + first]
            with self.subTest(mutation=mutation), patch.object(HELPER, "pin_json", return_value=bad):
                with self.assertRaisesRegex(ValueError, "alias identity differs"):
                    HELPER.bind_index(None, selected["closure_index"], repository_paths=True,
                                      repository_aliases=HELPER.COMBINED_PROTOCOL_ALIASES)

    def test_only_laurent_miss_table_changes_typst(self):
        current = (DOCS / "current-source-supplement.typ").read_text()
        proposed = (HERE / "proposed/current-source-supplement.typ").read_text()
        start = current.index("  if s.numerical.target_misses.len() > 0 {")
        end = current.index("  for note-text in s.numerical.build_scope_notes {")
        pstart = proposed.index("  if s.numerical.target_misses.len() > 0 {")
        pend = proposed.index("  for note-text in s.numerical.build_scope_notes {")
        self.assertEqual(current[:start], proposed[:pstart])
        self.assertEqual(current[end:], proposed[pend:])
        for column in range(4):
            self.assertIn('row.columns.at("' + str(column) + '")', proposed[pstart:pend])

if __name__ == "__main__":
    unittest.main()
