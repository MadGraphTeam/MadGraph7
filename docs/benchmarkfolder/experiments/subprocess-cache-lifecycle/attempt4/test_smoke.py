#!/usr/bin/env python3
"""Synthetic-only checks of the card-reader evidence parser.

All records and traces below are fabricated temporary text. Importing resume
and calling check_smoke does not freeze inputs, compile, load a reducer, run a
model reader, or provide numerical/installation qualification evidence.
"""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lifecycle_attempt4_smoke", HERE / "resume.py")
RESUME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESUME)

NA = -(1 << 31)
TRACE_FIELDS = (
    "sequence", "action", "process", "kind", "token_before", "token_after",
    "status", "n", "rank", "poles", "finite", "check", "cache", "version",
    "stats_pre_aggregate", "stats_pre_qp", "stats_post_aggregate", "stats_post_qp",
    "f0", "f1", "f2", "f3", "r0", "r1", "r2", "r3",
)


def records_text():
    return "\n".join((
        "HEADER 1",
        "INITIAL -1 -1 0",
        "CARD 1 nominal/param_card.dat 4056CC083126E979 -1 -1 -1 -1 0",
        "CARD 2 alternate/param_card.dat 4057000000000000 -1 -1 -1 -1 0",
        "CARD 3 nominal/param_card.dat 4056CC083126E979 -1 -1 -1 -1 0",
        "DONE 3 -1 -1 0",
        "CLOSED -1 -1 0",
    )) + "\n"


def begin(action):
    row = dict.fromkeys(TRACE_FIELDS, NA)
    row.update(sequence=0, action=action, process=0, kind=1,
               token_before=-1, token_after=-1, status=0, version=0,
               before=[0] * 15, after=[0] * 15)
    row.update({field + str(i): 0 for field in ("f", "r") for i in range(4)})
    return row


def groups():
    return [[begin(action)] for action in (1, 2, 3)]


def trace_text(rows=None, observed=1, health=0, token=-1):
    """Serialize internally consistent counts, so semantic mutations stay targeted."""
    rows = groups() if rows is None else rows
    lines = [f"H 1 {observed} 32768 360"]
    count = 0
    for serial, group in enumerate(rows, 1):
        for row in group:
            count += 1
            fields = [count if name == "sequence" else row[name] for name in TRACE_FIELDS]
            fields += row["before"] + row["after"]
            lines.append("R " + " ".join(map(str, fields)))
        lines.append(f"F {serial} {len(group)} {count} {health} {token}")
    lines.append(f"F {len(rows) + 1} 0 {count} {health} {token}")
    return "\n".join(lines) + "\n"


class CardSmokeParserTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="tirust-card-smoke-synthetic-")
        self.addCleanup(directory.cleanup)
        self.records = Path(directory.name) / "records.txt"
        self.trace = Path(directory.name) / "trace.txt"

    def check(self, records=None, trace=None):
        self.records.write_text(records_text() if records is None else records, encoding="ascii")
        self.trace.write_text(trace_text() if trace is None else trace, encoding="ascii")
        return RESUME.check_smoke(self.records, self.trace)

    def reject(self, records=None, trace=None):
        with self.assertRaises((RuntimeError, ValueError)):
            self.check(records, trace)

    def test_complete_synthetic_protocol_is_accepted(self):
        result = self.check()
        self.assertIs(result["passed"], True)
        self.assertEqual(result["native_exports_observed"], 0)
        self.assertEqual(result["trace_records"], 3)

    def test_missing_closed_bare_stop_and_truncated_records_are_rejected(self):
        lines = records_text().splitlines()
        for changed in ("", "STOP\n", "STOP 0\n", "\n".join(lines[:-1]) + "\n",
                        "\n".join(lines[:-2]) + "\n", records_text() + "STOP 0\n"):
            with self.subTest(records=changed):
                self.reject(records=changed)

    def test_wrong_initial_alternate_and_restored_mass_are_rejected(self):
        lines = records_text().splitlines()
        for index in (2, 3, 4):
            changed = lines.copy()
            tokens = changed[index].split()
            tokens[3] = "4057000000000000" if index != 3 else "4056CC083126E979"
            changed[index] = " ".join(tokens)
            with self.subTest(card=index - 1):
                self.reject(records="\n".join(changed) + "\n")

    def test_wrong_paths_order_ordinals_and_extra_records_are_rejected(self):
        original = records_text()
        lines = original.splitlines()
        swapped = lines.copy()
        swapped[2], swapped[3] = swapped[3], swapped[2]
        cases = (original.replace("nominal/param_card.dat", "param_card.dat", 1),
                 original.replace("alternate/param_card.dat", "nominal/param_card.dat"),
                 original.replace("CARD 2", "CARD 1"), original.replace("DONE 3", "DONE 2"),
                 original + "CLOSED -1 -1 0\n", "\n".join(swapped) + "\n")
        for index, changed in enumerate(cases):
            with self.subTest(case=index):
                self.reject(records=changed)

    def test_record_generation_or_health_changes_are_rejected(self):
        original = [line.split() for line in records_text().splitlines()]
        for line, positions in ((1, (1, 2, 3)), (2, (4, 5, 6, 7, 8)),
                                (3, (4, 5, 6, 7, 8)), (4, (4, 5, 6, 7, 8)),
                                (5, (2, 3, 4)), (6, (1, 2, 3))):
            for column in positions:
                changed = copy.deepcopy(original)
                changed[line][column] = "1"
                with self.subTest(line=line, column=column):
                    self.reject(records="\n".join(" ".join(row) for row in changed) + "\n")

    def test_extra_native_export_and_unobserved_trace_are_rejected(self):
        changed = groups()
        native = begin(1)
        native.update(kind=6, n=1, rank=0, poles=0, finite=1, check=0, cache=0)
        changed[0].append(native)
        self.reject(trace=trace_text(changed))
        self.reject(trace=trace_text(observed=0))

    def test_dirty_begin_metadata_and_counters_are_rejected(self):
        for field in TRACE_FIELDS[2:18]:
            if field == "kind":
                continue
            changed = groups()
            changed[1][0][field] = 1
            with self.subTest(field=field):
                self.reject(trace=trace_text(changed))
        for field in ("before", "after"):
            for index in range(15):
                changed = groups()
                changed[1][0][field][index] = 1
                with self.subTest(field=field, counter=index):
                    self.reject(trace=trace_text(changed))

    def test_observer_ieee_flag_or_rounding_perturbation_is_rejected(self):
        for field in ("f", "r"):
            for indices in ((0,), (1,), (2,), (3,), (0, 1), (2, 3)):
                changed = groups()
                for index in indices:
                    changed[1][0][field + str(index)] = 16 if field == "f" else 1024
                with self.subTest(field=field, indices=indices):
                    self.reject(trace=trace_text(changed))

    def test_unchanged_nonzero_ieee_state_is_accepted(self):
        changed = groups()
        for row in changed[1]:
            for index in range(4):
                row["f" + str(index)] = 16
                row["r" + str(index)] = 1024
        self.assertIs(self.check(trace=trace_text(changed))["passed"], True)

    def test_wrong_action_count_order_and_process_are_rejected(self):
        cases = (groups()[:2], groups() + [[begin(4)]],
                 [[begin(2)], [begin(1)], [begin(3)]],
                 [[begin(1)], [begin(2)], [begin(4)]])
        for index, changed in enumerate(cases):
            with self.subTest(case=index):
                self.reject(trace=trace_text(changed))
        changed = groups()
        changed[0][0]["process"] = 1
        self.reject(trace=trace_text(changed))

    def test_flush_health_generation_and_missing_close_are_rejected(self):
        self.reject(trace=trace_text(health=8))
        self.reject(trace=trace_text(token=1))
        lines = trace_text().splitlines()
        self.reject(trace="\n".join(lines[:-1]) + "\n")
        self.reject(trace=trace_text() + "UNEXPECTED 0\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
