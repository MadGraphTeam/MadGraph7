#!/usr/bin/env python3
"""Synthetic-only parser tests. No compiler, process generator or reducer runs.

The small amplitude text frame tests syntax, not physics. The in-memory trace
contains fabricated integer cache/lifecycle observations solely to exercise
accept/reject branches; it is not a fake numerical benchmark campaign.
"""
import copy
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lifecycle_retained_analysis", HERE / "analyze.py")
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


def word(value):
    return struct.pack(">d", value).hex().upper()


def amplitude_text():
    zero = word(0.0)
    model = f"{word(91.1876)} {word(91.1876)}"
    lines = ["AMP 301 1 1 219 1 1 1 1", f"MODELI 301 {model}",
             f"MODEL 301 {model}", "FLAGS 301 " + " ".join(["F"] * 10)]
    for leg in range(1, 4):
        for tag in ("PI", "P"):
            lines.append(f"{tag} 301 {leg} " + " ".join([zero] * 4))
    for column in (0, 1):
        lines.append(f"V 301 {column} " + " ".join(map(word, (1.0, 2.0, -3.0, -4.0, 1e-7))))
    return "\n".join(lines) + "\n"


def record(action, process, kind, token, **changes):
    row = dict.fromkeys(ANALYZE.TRACE_NAMES, ANALYZE.NA)
    row.update(sequence=0, action=action, process=process, kind=kind,
               token_before=token, token_after=token, status=0, version=0,
               before=[0] * 15, after=[0] * 15)
    row.update({f"{field}{index}": 0 for field in ("f", "r") for index in range(4)})
    if kind in (5, 6, 7):
        row.update(n=3, rank=2)
    if kind in (6, 7):
        row.update(poles=0, finite=1, check=0, cache=1)
    if kind == 4:
        row.update({name: 0 for name in ANALYZE.TRACE_NAMES[14:18]})
    row.update(changes)
    return row


def trace_line(row):
    values = [row[name] for name in ANALYZE.TRACE_NAMES] + row["before"] + row["after"]
    return "R " + " ".join(map(str, values))


def lifecycle_fixture(cache=-1):
    """Parsed integer structures only; no synthetic amplitude coefficients."""
    groups, amplitudes, keys = {}, {}, {}
    order = [101, 201, *range(301, 316), 350, 316, 360, 317, 318, 319, 900,
             *ANALYZE.KEYS, 1900]
    token = -1
    for action in order:
        process = 2 if action == 201 or action in (302, 305, 308, 311, 314, 317, 319) else 1
        if action == 350:
            process = 0
        if action in ANALYZE.KEYS:
            key = action % 100
            process = 1 + key % 2
        before_token = token
        rows = [record(action, process, 1, token)]
        if action in (101, 317):
            token = 1 if action == 101 else 2
            rows.append(record(action, process, 2, -1, token_after=token, version=1))
        if action in (360, 1900):
            rows.append(record(action, process, 3, token, token_after=-1))
            token = -1
        elif action == 350:
            rows.append(record(action, process, 6, token, n=1, rank=0))
        elif action == 900:
            rows.append(record(action, process, 4, token))
        elif action in ANALYZE.KEYS:
            precision = (action - 1000) // 100
            n, rank = (1, 0) if key < 8 else (3, 2)
            rows.append(record(action, process, 5 + precision, token, n=n, rank=rank,
                               cache=int(cache == -1)))
            keys[action] = dict(process=process, precision=precision, n=n, rank=rank)
        else:
            entry = record(action, process, 4, token)
            if action == 316:
                entry["before"][6:8] = [1, 64]
            rows.append(entry)
            if action in (101, 201, 317, 318):
                rows.append(record(action, process, 5, token))
            rows.append(record(action, process, 6, token, cache=int(cache == -1)))
            final = record(action, process, 4, token)
            if cache == -1:
                final["before"][6:8] = [1, 64]
            rows.append(final)
            amplitudes[action] = dict(process=process, generation_before=before_token,
                                      generation_after=token)
        groups[action] = rows
    records, flushes = [], []
    for action in order:
        rows = groups[action]
        for row in rows:
            row["sequence"] = len(records) + 1
            records.append(row)
        flushes.append(dict(serial=len(flushes) + 1, token=rows[-1]["token_after"], records=rows))
    flushes.append(dict(serial=len(flushes) + 1, token=-1, records=[]))
    data = dict(cache=cache, order=order, amplitudes=amplitudes, keys=keys)
    trace = dict(observed=True, order=order, groups=groups, records=records, flushes=flushes)
    return data, trace


class TemporaryEvidence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tirust-lifecycle-parser-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "synthetic.txt"

    def write(self, text):
        self.path.write_text(text, encoding="ascii")
        return self.path


class AmplitudeRecordTests(TemporaryEvidence):
    def parse(self, text):
        reader = ANALYZE.Reader(self.write(text))
        result = ANALYZE.parse_amp(reader, [1, 1, 1, 1])
        reader.done()
        return result

    def test_complete_small_frame(self):
        frame = self.parse(amplitude_text())
        self.assertEqual(frame["action"], 301)
        self.assertEqual(frame["momenta"], frame["momenta_before"])
        self.assertEqual(len(frame["values"]), 2)

    def test_missing_duplicate_extra_and_reordered_rows(self):
        lines = amplitude_text().splitlines()
        mutations = {
            "missing MODELI": lines[:1] + lines[2:],
            "duplicate PI": lines[:5] + [lines[4]] + lines[5:],
            "missing last coefficient": lines[:-1],
            "extra coefficient": lines + [lines[-1]],
            "wrong leg": [line.replace("P 301 2 ", "P 301 3 ") for line in lines],
            "wrong column": [line.replace("V 301 1 ", "V 301 2 ") for line in lines],
        }
        for name, changed in mutations.items():
            with self.subTest(name=name), self.assertRaises(ANALYZE.EvidenceError):
                self.parse("\n".join(changed) + "\n")

    def test_malformed_scalar_fields_and_incomplete_stream(self):
        original = amplitude_text()
        for changed in (original.replace("AMP 301", "AMP 3.01", 1),
                        original.replace("FLAGS 301 F", "FLAGS 301 0", 1),
                        original.replace("0000000000000000", "000000000000000G", 1),
                        original.rstrip("\n"), original + "\n"):
            with self.subTest(prefix=changed[:50]), self.assertRaises(ANALYZE.EvidenceError):
                self.parse(changed)

    def test_campaign_reader_rejects_missing_or_extra_protocol_records(self):
        for text in ("", "HEADER 0 1 1 1 1\n", "HEADER 0 1 1 1 1\nCAP 15\n",
                     "HEADER 0 1 1 1 1\nCAP 15\nCAP 15\n",
                     "HEADER 0 1 1 1 1\nCAP 15\nWARM 0 0\n"):
            with self.subTest(text=text), self.assertRaises(ANALYZE.EvidenceError):
                ANALYZE.parse_records(self.write(text))


class TraceRecordTests(TemporaryEvidence):
    def baseline(self):
        row = record(301, 1, 1, -1, sequence=1)
        return "H 1 0 32768 360\n" + trace_line(row) + "\nF 1 1 1 0 -1\nF 2 0 1 0 -1\n"

    def test_small_plain_trace(self):
        result = ANALYZE.parse_trace(self.write(self.baseline()))
        self.assertFalse(result["observed"])
        self.assertEqual(result["order"], [301])

    def test_missing_extra_duplicate_and_ragged_trace_records(self):
        lines = self.baseline().splitlines()
        row_fields = lines[1].split()
        mutations = {
            "missing header": lines[1:],
            "missing close flush": lines[:-1],
            "extra trailing record": lines + ["UNEXPECTED 1"],
            "duplicate row": lines[:2] + [lines[1]] + lines[2:],
            "short row": [lines[0], " ".join(row_fields[:-1]), *lines[2:]],
            "long row": [lines[0], lines[1] + " 0", *lines[2:]],
            "wrong flush count": [*lines[:2], "F 1 2 1 0 -1", lines[3]],
            "missing flush": lines[:2],
        }
        for name, changed in mutations.items():
            with self.subTest(name=name), self.assertRaises(ANALYZE.EvidenceError):
                ANALYZE.parse_trace(self.write("\n".join(changed) + "\n"))

    def test_health_overflow_and_negative_counters_fail_closed(self):
        original = self.baseline()
        for flag in (1, 2, 4, 8, 16, 32):
            with self.subTest(health=flag), self.assertRaises(ANALYZE.EvidenceError):
                ANALYZE.parse_trace(self.write(original.replace("F 1 1 1 0 -1", f"F 1 1 1 {flag} -1")))
        row = record(301, 1, 1, -1, sequence=1)
        row["before"][0] = -1
        text = "H 1 0 32768 360\n" + trace_line(row) + "\nF 1 1 1 0 -1\nF 2 0 1 0 -1\n"
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "uint64"):
            ANALYZE.parse_trace(self.write(text))


class TraceSemanticTests(unittest.TestCase):
    def setUp(self):
        self.data, self.trace = lifecycle_fixture()

    def failures(self, trace=None):
        checks = ANALYZE.Checks()
        label = "observed-cache0" if self.data["cache"] == 0 else "observed-cache-all"
        ANALYZE.check_trace(self.data, self.trace if trace is None else trace,
                            checks, label)
        return checks.failures

    def evaluation(self, action=301):
        return next(row for row in self.trace["groups"][action] if row["kind"] == 6)

    def test_fabricated_integer_lifecycle_control_is_accepted(self):
        self.assertEqual(self.failures(), [])

    def test_observer_only_ieee_or_rounding_mutation_is_rejected(self):
        for field in ("f1", "f3", "r1", "r3"):
            changed = copy.deepcopy(self.trace)
            row = next(row for row in changed["groups"][301] if row["kind"] == 6)
            row[field] = 16
            with self.subTest(field=field):
                self.assertTrue(any("observer altered IEEE" in failure for failure in self.failures(changed)))

    def test_real_numerical_ieee_effect_is_not_attributed_to_observer(self):
        self.evaluation()["f2"] = self.evaluation()["f3"] = 16
        self.assertEqual(self.failures(), [])

    def test_wrong_context_and_missing_reregistration_are_rejected(self):
        self.evaluation()["token_after"] = 99
        self.assertTrue(any("context-token mismatch" in failure for failure in self.failures()))
        self.data, self.trace = lifecycle_fixture()
        registration = next(row for row in self.trace["groups"][317] if row["kind"] == 5)
        self.trace["groups"][317].remove(registration)
        self.trace["records"].remove(registration)
        failures = self.failures()
        self.assertTrue(any("prefix 2 did not repeat" in failure for failure in failures))
        self.assertTrue(any("prefix 2 did not re-register" in failure for failure in failures))

    def test_failed_statistics_and_nonempty_clear_are_rejected(self):
        for field in ("stats_pre_aggregate", "stats_pre_qp", "stats_post_aggregate", "stats_post_qp"):
            changed = copy.deepcopy(self.trace)
            clear = next(row for row in changed["groups"][301] if row["kind"] == 4)
            clear[field] = 2
            with self.subTest(field=field):
                self.assertTrue(any("clear statistics failed" in failure for failure in self.failures(changed)))
        clear = self.trace["groups"][301][-1]
        clear["after"][6] = 1
        self.assertTrue(any("clear did not reset" in failure for failure in self.failures()))

    def test_disabled_cache_seed_exemption_does_not_cover_final_clear(self):
        self.data, self.trace = lifecycle_fixture(cache=0)
        # The forced-cache seed at action350 is allowed only at the first real
        # entry clear of action316; the untouched disabled fixture must pass.
        self.assertEqual(self.failures(), [])
        for index in (6, 7, 11, 12):
            changed = copy.deepcopy(self.trace)
            final_clear = changed["groups"][316][-1]
            self.assertEqual(final_clear["kind"], 4)
            final_clear["before"][index] = 1
            with self.subTest(counter=index):
                self.assertTrue(any("disabled event retained entries at 316" in failure
                                    for failure in self.failures(changed)))

    def test_missing_entry_or_final_clear_is_rejected(self):
        for index, message in ((1, "lacks automatic entry clear"), (-1, "lacks final automatic native clear")):
            changed = copy.deepcopy(self.trace)
            row = changed["groups"][301].pop(index)
            self.assertEqual(row["kind"], 4)
            changed["records"].remove(row)
            with self.subTest(index=index):
                self.assertTrue(any(message in failure for failure in self.failures(changed)))

    def test_recreation_does_not_replace_required_real_entry_clear(self):
        rows = self.trace["groups"][317]
        clear = next(row for row in rows if row["kind"] == 4)
        rows.remove(clear)
        self.trace["records"].remove(clear)
        self.assertTrue(any("physical 317 lacks automatic entry clear" in failure for failure in self.failures()))

    def test_statistics_outside_clear_and_hidden_key_clear_are_rejected(self):
        self.evaluation()["stats_pre_aggregate"] = 0
        self.assertTrue(any("statistics queried outside clear" in failure for failure in self.failures()))
        self.data, self.trace = lifecycle_fixture()
        rows = self.trace["groups"][1102]
        clear = record(1102, 1, 4, 2, sequence=999)
        rows.insert(1, clear)
        self.trace["records"].insert(self.trace["records"].index(rows[0]) + 1, clear)
        self.assertTrue(any("hidden clear" in failure for failure in self.failures()))


class ExactPromotionTests(unittest.TestCase):
    def test_known_binary64_to_binary128_patterns(self):
        # Independently specified IEEE interchange encodings, not values
        # obtained by calling the analyzer's own conversion implementation.
        cases = {
            "0000000000000000": ["0000000000000000", "0000000000000000"],
            "8000000000000000": ["0000000000000000", "8000000000000000"],
            "3FF0000000000000": ["0000000000000000", "3FFF000000000000"],
            "C004000000000000": ["0000000000000000", "C000400000000000"],
            "3FF0000000000001": ["1000000000000000", "3FFF000000000000"],
            "0000000000000001": ["0000000000000000", "3BCD000000000000"],
            "0010000000000000": ["0000000000000000", "3C01000000000000"],
            "7FEFFFFFFFFFFFFF": ["F000000000000000", "43FEFFFFFFFFFFFF"],
        }
        for source, target in cases.items():
            with self.subTest(source=source):
                self.assertEqual(ANALYZE.promoted128(source), target)

    def test_nonfinite_and_malformed_promotions_are_rejected(self):
        for source in ("7FF0000000000000", "FFF0000000000000", "7FF8000000000001",
                       "3ff0000000000000", "3FF000000000000", "GFF0000000000000"):
            with self.subTest(source=source), self.assertRaises(ANALYZE.EvidenceError):
                ANALYZE.promoted128(source)

    def test_binary128_finite_classifier(self):
        self.assertTrue(ANALYZE.finite128(["FFFFFFFFFFFFFFFF", "7FFEFFFFFFFFFFFF"]))
        self.assertFalse(ANALYZE.finite128(["0000000000000000", "7FFF000000000000"]))
        self.assertFalse(ANALYZE.finite128(["0000000000000001", "7FFF000000000000"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
