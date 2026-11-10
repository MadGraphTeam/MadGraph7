"""Pure source and synthetic-journal tests; never loads or executes a DSO."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import driver_patch
import protocol

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CLOSED = HERE.parent / "closed-2"
BASE = HERE.parent / "source-proposal-1/replay.c"


def fixture():
    slots = json.loads((CLOSED / "summary.json").read_text())["slots"]
    inputs = []
    for slot in slots:
        if slot["input_state"] != "captured":
            continue
        key = slot["input_key"]
        name = f"tirust-input-p{key['point']:03d}-m{key['method']}-g{key['group']:03d}.dat"
        raw = (CLOSED / "capture/qcd_n3_uux_zgg/outputs" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == slot["input_sha256"]
        inputs.append(dict(protocol.input_record(raw), basename=name, sha256=slot["input_sha256"],
                           expected_status=slot["c_statuses"][0], expected_error=slot["c_errors"][0]["error"]))
    return inputs


def snapshot():
    return "\n".join(("private_dp_diagnostics_v1", "reduce_batch_executed=true", "precision_bits=53",
        "embedder_options=Some(EmbedderOptions { method_choice: true, partial_rescue: false, precision_tiers: false, gate: None, method_choice_gate: None, master_policy: Default })",
        "reduce_batch=Ok", "workspace_metadata_fresh=true", "lane_statuses=[Unstable(\"primary Gram\")]",
        "fallback_used=[true]", "analytic_used=[false]", "last_lane_routes=[EmbeddedMethodChoice]",
        "engine_execution=Embedded { direct: 1 }", "last_diagnostics=[LaneDiagnostic { message: \"fallback condition\" }]",
        "truncated=false", ""))


def journal(inputs):
    rows = []
    for i, value in enumerate(inputs):
        for j, stage in enumerate(("create", "register", "evaluate", "evaluate", "destroy")):
            evaluate = stage == "evaluate"
            row = dict(record_type=stage, input_key=value["input_key"], input_path="/private/"+value["basename"],
                       repetition=(1, 1, 1, 2, 2)[j], status=1 if evaluate else 0,
                       error_encoding="latin1-bytes", error_truncated=False,
                       error=value["expected_error"] if evaluate else "", context_present=j != 4,
                       evaluation_count=(2*i, 2*i, 2*i+1, 2*i+2, 2*i+2)[j])
            row["error_required"] = len(row["error"].encode("latin1")) + 1
            if evaluate:
                row.update(private_diagnostic=snapshot(), private_diagnostic_required=len(snapshot().encode())+1,
                           private_diagnostic_encoding="latin1-bytes", private_diagnostic_copy_truncated=False)
            if stage == "destroy":row["destroy_succeeded"] = True
            rows.append(row)
    return rows


def serialized(rows):
    return "".join(json.dumps(row)+"\n" for row in rows)


class DriverProtocolTests(unittest.TestCase):
    def setUp(self):
        self.inputs = fixture()
        self.rows = journal(self.inputs)

    def reject(self, rows):
        with self.assertRaises((ValueError, KeyError, UnicodeError)):
            protocol.replay_summary(serialized(rows), self.inputs)

    def test_derived_c_is_exact_and_numerically_unchanged(self):
        base = BASE.read_bytes()
        candidate = driver_patch.prepare(base).decode()
        self.assertEqual(hashlib.sha256(base).hexdigest(), driver_patch.BASE_SHA256)
        self.assertEqual(candidate.count("status = ml_tirust_evaluate_dp("), 1)
        self.assertIn("MAX_REQUESTS = 6, MAX_EVALUATIONS = 12", candidate)
        self.assertIn("if (*count != 6) return 0;", candidate)
        self.assertIn("r->case_id != 1 ||", candidate)
        self.assertIn("Exactly the six captured Zgg", candidate)
        self.assertNotIn("An empty manifest", candidate)
        self.assertEqual(candidate.count("for (repetition = 1; repetition <= 2; ++repetition)"), 1)
        self.assertLess(candidate.index("copy_error(&error);", candidate.index("status = ml_tirust_evaluate_dp(")),
                        candidate.index('record(out, r, "evaluate"'))
        self.assertIn('strcmp(stage, "evaluate") == 0 && status == TIRUST_SUCCESS', candidate)
        with self.assertRaises(ValueError):driver_patch.prepare(base+b"\n")

    def test_exact_six_inputs_parse_without_floating_point(self):
        self.assertEqual(len(self.inputs), 6)
        self.assertTrue(all(len(x["shift_words"]) == 20 and len(x["invariant_words"]) == 25 and
                            len(x["mass_words"]) == 10 for x in self.inputs))
        for replacement in (b"MLTCAP_INPUT_V2", b"MLTCAP_INPUT_V1\n2 1 25 1 59 5 4 20 25 5 280 0 1 0 0\n"):
            with self.assertRaises((ValueError, IndexError)):protocol.input_record(replacement)

    def test_synthetic_complete_protocol(self):
        result = protocol.replay_summary(serialized(self.rows), self.inputs)
        self.assertEqual(result["evaluations"], 12)
        self.assertEqual(result["failed_tensor_outputs_read"], 0)
        self.assertFalse(result["inner_variant_or_operand_observer"])
        self.assertFalse(result["numerical_acceptance_claimed"])

    def test_failed_output_rejected(self):
        self.rows[2]["output_bits"] = []
        self.reject(self.rows)

    def test_context_presence_destruction_counts_repetitions(self):
        for index, key, value in ((0, "context_present", False), (1, "context_present", False),
                                  (4, "context_present", True), (4, "destroy_succeeded", False),
                                  (0, "evaluation_count", 1), (4, "evaluation_count", 1),
                                  (1, "repetition", 2), (4, "repetition", 1)):
            rows = copy.deepcopy(self.rows); rows[index][key] = value; self.reject(rows)

    def test_wrong_order_duplicate_missing_and_extra_inputs(self):
        for rows in (self.rows[:-1], self.rows+[self.rows[-1]], self.rows[1:]+self.rows[:1]):self.reject(rows)
        for inputs in (self.inputs[:-1], self.inputs[:5]+[self.inputs[0]]):
            with self.assertRaises(ValueError):protocol.replay_summary(serialized(self.rows), inputs)

    def test_last_error_and_status_must_match_unchanged_eb3(self):
        for key, value in (("status", 0), ("error", "changed"), ("error_required", 1),
                           ("error_truncated", True), ("error_encoding", "utf8")):
            rows = copy.deepcopy(self.rows); rows[2][key] = value; self.reject(rows)

    def test_incomplete_stale_changed_or_wrong_precision_snapshot_rejected(self):
        for before, after in (("truncated=false", "truncated=true"), ("precision_bits=53", "precision_bits=113"),
                              ("workspace_metadata_fresh=true", "workspace_metadata_fresh=unproven_partial_or_previous"),
                              ("method_choice: true", "method_choice: false"), ("gate: None", "gate: Some(0.0)"),
                              ("reduce_batch=Ok", "reduce_batch=Err(InvalidInput)")):
            rows = copy.deepcopy(self.rows)
            rows[2]["private_diagnostic"] = snapshot().replace(before, after)
            rows[2]["private_diagnostic_required"] = len(rows[2]["private_diagnostic"].encode())+1
            self.reject(rows)

    def test_snapshot_copy_bounds_encoding_and_duplicate_fields(self):
        for key, value in (("private_diagnostic_required", 65537), ("private_diagnostic_copy_truncated", True),
                           ("private_diagnostic_encoding", "utf8"), ("private_diagnostic", "")):
            rows = copy.deepcopy(self.rows); rows[2][key] = value; self.reject(rows)
        rows = copy.deepcopy(self.rows)
        rows[2]["private_diagnostic"] = snapshot().replace("lane_statuses=", "lane_statuses=[]\nlane_statuses=", 1)
        rows[2]["private_diagnostic_required"] = len(rows[2]["private_diagnostic"].encode())+1
        self.reject(rows)


if __name__ == "__main__":unittest.main()
