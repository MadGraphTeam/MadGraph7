"""Pure tests of the TIRust capture/replay tooling (no bridge, compiler or physics)."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools" / "tirust_capture"


def load(name):
    spec = importlib.util.spec_from_file_location("tirust_capture_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace = load("tirust_trace")
prepare = load("prepare_replay")
gate = load("dd_gate")


def dp_record(n, rank, status=0, check=0, inputs=True, output=None):
    data = bytes([1]) + struct.pack("<7id", n, rank, 0, 1, check, 1, status, 1e-6)
    if not inputs:
        return data + bytes([0])
    data += bytes([1])
    data += struct.pack(f"<{4 * n}d", *range(4 * n))
    data += struct.pack(f"<{n * n}d", *([0.5] * n * n))
    data += struct.pack(f"<{2 * n}d", *([1.0, -0.0] * n))
    data += struct.pack("<d", 91.0)
    olen = 0 if status else 4 * trace.coefficient_count(rank)
    data += struct.pack("<Q", olen)
    values = output if output is not None else [float(k) for k in range(2 * olen)]
    return data + struct.pack(f"<{2 * olen}d", *values)


def qp_record(n, rank, status=0, with_inputs=False):
    data = bytes([5 if with_inputs else 2]) + struct.pack("<7id", n, rank, 0, 1, 0, 1, status, 2e-6)
    if with_inputs:
        records = [b"a" * 64 * 4 * n, b"b" * 64 * n * n, b"c" * 128 * n, b"d" * 64]
        data += struct.pack("<4Q", *map(len, records)) + b"".join(records)
        out = b"" if status else b"e" * 128
        data += struct.pack("<Q", len(out)) + out
    return data


def register(n, rank):
    return bytes([3]) + struct.pack("<3id", n, rank, 0, 1e-3)


def clear():
    return bytes([4]) + struct.pack("<d", 1e-7)


class TraceFormatTests(unittest.TestCase):
    def test_coefficient_counts_match_madloop(self):
        self.assertEqual([trace.coefficient_count(r) for r in range(7)], [1, 5, 15, 35, 70, 126, 210])

    def test_round_trip_all_record_kinds(self):
        data = (register(3, 2) + dp_record(3, 2) + dp_record(4, 3, status=1) + qp_record(2, 1)
                + qp_record(2, 1, with_inputs=True) + clear() + dp_record(2, 0, check=1, inputs=False) + clear())
        records = trace.parse_bytes(data)
        self.assertEqual([r["kind"] for r in records], [3, 1, 1, 2, 5, 4, 1, 4])
        first = records[1]
        self.assertEqual(first["shifts"], tuple(float(k) for k in range(12)))
        self.assertEqual(first["mu2"], 91.0)
        self.assertEqual(first["output_length"], 4 * 15)
        self.assertEqual(len(first["output"]), 2 * 4 * 15)
        self.assertEqual(records[2]["output_length"], 0)
        self.assertEqual(records[4]["scale"], b"d" * 64)
        self.assertFalse(records[6]["has_inputs"])

    def test_summary_counts_segments_and_failures(self):
        data = (register(3, 2) + dp_record(3, 2) + dp_record(3, 2, status=1) + clear()
                + dp_record(3, 2, check=1) + qp_record(3, 2) + clear() + clear())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.1"
            path.write_bytes(data)
            summary = trace.summarize(path)
        self.assertEqual(summary["segments"], 2)
        self.assertEqual(summary["totals"]["dp"], 3)
        self.assertEqual(summary["totals"]["dp_failed"], 1)
        self.assertEqual(summary["segments_with_dp_failures"], 1)
        self.assertEqual(summary["segments_with_qp_calls"], 1)
        self.assertEqual(summary["by_key"]["dp:n3:r2:check1"]["calls"], 1)
        self.assertEqual(summary["registrations"], [[3, 2, 0]])

    def test_truncated_and_inconsistent_records_are_rejected(self):
        whole = dp_record(3, 2)
        with self.assertRaises(trace.TraceError):
            trace.parse_bytes(whole[:-8])
        with self.assertRaises(trace.TraceError):
            trace.parse_bytes(bytes([9]))
        bad = bytearray(dp_record(3, 2, status=1))
        bad[-8:] = struct.pack("<Q", 60)
        with self.assertRaises(trace.TraceError):
            trace.parse_bytes(bytes(bad))


class ReplayPreparationTests(unittest.TestCase):
    SOURCE = "EmbedderOptions { precision_tiers: true, tier_margin_bits: 0, double_double_headroom_bits: 26, .. }"

    def test_headroom_override_changes_only_that_setting(self):
        text, previous = prepare.set_headroom(self.SOURCE, 40)
        self.assertEqual(previous, 26)
        self.assertEqual(text, self.SOURCE.replace("bits: 26", "bits: 40"))
        self.assertEqual(prepare.production_headroom(self.SOURCE), 26)

    def test_headroom_override_is_strict(self):
        with self.assertRaises(ValueError):
            prepare.set_headroom(self.SOURCE + self.SOURCE, 40)
        with self.assertRaises(ValueError):
            prepare.set_headroom("no setting", 40)
        with self.assertRaises(ValueError):
            prepare.set_headroom(self.SOURCE, 106)

    def test_vendor_bridge_has_one_production_headroom(self):
        self.assertIsInstance(prepare.production_headroom((ROOT / "vendor/tirust/src/lib.rs").read_text()), int)

    def test_manifest_points_at_local_sources_and_pinned_symbolica(self):
        revision = prepare.symbolica_revision()
        manifest = prepare.cargo_manifest("/w/bridge-h40", "/w/sources/tirust", revision)
        self.assertIn('madloop-tirust = { path = "/w/bridge-h40", features = ["poles-only"] }', manifest)
        self.assertIn('iregi-rust = { path = "/w/sources/tirust/crates/iregi-rust" }', manifest)
        self.assertEqual(manifest.count(revision), 3)
        self.assertIn("[profile.replay]", manifest)


def call(i, delta, *, seg=0, n=4, rank=3, status=0, rec_status=0, identical=True, error=0.0):
    return {"i": i, "seg": seg, "n": n, "rank": rank, "check": 0, "rec_status": rec_status, "status": status,
            "delta": delta, "identical": identical, "max_rel": error, "max_comp_rel": error}


DIRECT = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DD = [0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
T256 = [0, 1, 0, 1, 0, 0, 0, 0, 1, 0]


class DoubleDoubleGateTests(unittest.TestCase):
    def write(self, directory, name, rows):
        path = Path(directory) / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_reference_errors_on_dd_calls_decide_the_gate(self):
        with tempfile.TemporaryDirectory() as d:
            production = self.write(d, "p", [call(0, DIRECT), call(1, DD), call(2, DD, seg=1)])
            reference = self.write(d, "r", [call(0, DIRECT), call(1, T256, identical=False, error=3e-12),
                                            call(2, T256, seg=1)])
            result = gate.analyse(production, [("h105", reference)])
        self.assertTrue(result["qualified"])
        report = result["references"]["h105"]
        self.assertEqual(result["dd_calls"], 2)
        self.assertEqual(report["dd_calls_compared"], 2)
        self.assertEqual(report["bit_identical"], 1)
        self.assertEqual(report["above_flag"], 1)
        self.assertEqual(report["above_fail"], 0)
        self.assertEqual(report["flagged"][0]["i"], 1)

    def test_large_error_or_status_change_fails(self):
        with tempfile.TemporaryDirectory() as d:
            production = self.write(d, "p", [call(0, DD), call(1, DD)])
            reference = self.write(d, "r", [call(0, T256, identical=False, error=2e-9),
                                            call(1, T256, status=1, identical=False, error=None)])
            result = gate.analyse(production, [("h40", reference)])
        self.assertFalse(result["qualified"])
        self.assertEqual(result["references"]["h40"]["above_fail"], 2)

    def test_nondeterministic_production_replay_is_not_qualified(self):
        with tempfile.TemporaryDirectory() as d:
            production = self.write(d, "p", [call(0, DD, identical=False, error=1e-16)])
            reference = self.write(d, "r", [call(0, T256)])
            result = gate.analyse(production, [("h40", reference)])
        self.assertFalse(result["qualified"])

    def test_reference_without_higher_tier_is_counted_not_compared(self):
        with tempfile.TemporaryDirectory() as d:
            production = self.write(d, "p", [call(0, DD)])
            reference = self.write(d, "r", [call(0, DD)])
            report = gate.analyse(production, [("h40", reference)])["references"]["h40"]
        self.assertEqual(report["dd_calls_compared"], 0)
        self.assertEqual(report["dd_calls_still_dd_in_reference"], 1)
        self.assertEqual(report["dd_calls_without_higher_tier_in_reference"], 1)

    def test_misaligned_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            production = self.write(d, "p", [call(0, DD)])
            reference = self.write(d, "r", [call(0, T256, seg=3)])
            with self.assertRaises(ValueError):
                gate.analyse(production, [("h40", reference)])


if __name__ == "__main__":
    unittest.main()
