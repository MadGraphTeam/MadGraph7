from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from tirust_benchmark_cache import actions, fixture_stream, parse_results, DEFAULT_FIXTURE


class TestCacheDiagnostics(unittest.TestCase):
    def setUp(self):
        self.fixture = {"n": 2, "rank": 0}

    def record(self, identifier=1, **kwargs):
        record = dict(id=identifier, name="test", ctmode=1, mode=0, finite=1,
                      cache=0, budget=10000, control=0, expectation="bypass",
                      reference=1, is_reference=identifier == 1)
        record.update(kwargs)
        return record

    def frame(self, record, before=None, after=None, values="8 0 2 0 1 0", generations=(1, 1)):
        before = [0]*15 if before is None else before
        after = list(before) if after is None else after
        metadata = [record[k] for k in ("id", "ctmode", "mode", "finite", "cache", "budget", "control")]
        identifier = record["id"]
        return ("F " + " ".join(map(str, metadata)) + f" 1 T {generations[0]} {generations[1]}\n"
                + f"B {identifier} " + " ".join(map(str, before)) + "\n"
                + f"A {identifier} " + " ".join(map(str, after)) + "\n"
                + f"C {identifier} 0 {values}\n")

    def test_stream_uses_separate_reversal_references_and_rank_guard(self):
        stream, records = fixture_stream(DEFAULT_FIXTURE, 15)
        self.assertIn("3 2 7", stream.splitlines()[0])
        self.assertGreater(len(records), 50)
        for record in records:
            reference = records[record["reference"]-1]
            self.assertEqual(tuple(record[k] for k in ("ctmode", "mode", "finite")),
                             tuple(reference[k] for k in ("ctmode", "mode", "finite")))
        with self.assertRaises(ValueError):
            fixture_stream(DEFAULT_FIXTURE, 1)

    def test_cached_values_must_match_uncached_reference(self):
        reference = self.record()
        hit = self.record(2, cache=-1, expectation="dp_hit")
        before = [0]*15
        before[6:8] = [1, 100]
        after = list(before)
        after[4] = 1
        text = self.frame(reference) + self.frame(hit, before, after)
        results = parse_results(text, self.fixture, [reference, hit])
        self.assertTrue(all(r["status"] == "passed" for r in results))
        text = self.frame(reference) + self.frame(hit, before, after, values="9 0 2 0 1 0")
        result = parse_results(text, self.fixture, [reference, hit])[1]
        self.assertIn("cached tensor differs", " ".join(result["failures"]))

    def test_qp_mode_collision_is_rejected_even_with_equal_values(self):
        reference = self.record(ctmode=4)
        ref_after = [0]*15
        ref_after[14] = 1
        cold = self.record(2, ctmode=4, cache=-1, expectation="qp_miss")
        after = [0]*15
        after[4] = after[9] = 1
        result = parse_results(self.frame(reference, after=ref_after) + self.frame(cold, after=after),
                               self.fixture, [reference, cold])[1]
        self.assertIn("distinct tensor entry", " ".join(result["failures"]))

    def test_partial_result_requires_nan_plus_zero_imaginary_mask(self):
        record = self.record(finite=0)
        result = parse_results(self.frame(record, values="NaN 0 2 0 1 0"), self.fixture, [record])[0]
        self.assertEqual(result["status"], "passed")
        for values in ("0 0 2 0 1 0", "NaN 1 2 0 1 0", "NaN 0 NaN 0 1 0"):
            result = parse_results(self.frame(record, values=values), self.fixture, [record])[0]
            self.assertEqual(result["status"], "failed")

    def test_qp_budget_refusal_does_not_change_dp_store(self):
        record = self.record(ctmode=4, cache=-1, budget=64, expectation="qp_refused")
        after = [0]*15
        after[5] = after[8] = after[10] = after[13] = 1
        self.assertEqual(parse_results(self.frame(record, after=after), self.fixture, [record])[0]["status"], "passed")
        after[0] = 1
        result = parse_results(self.frame(record, after=after), self.fixture, [record])[0]
        self.assertIn("QP operation changed DP caches", result["failures"])

    def test_clear_and_finalize_require_empty_statistics_and_new_generation(self):
        reference = self.record()
        cold = self.record(2, cache=-1, control=2, expectation="dp_miss")
        after = [0]*15
        after[1] = after[6] = 1
        after[7] = 100
        text = self.frame(reference) + self.frame(cold, after=after, generations=(-1, 2))
        self.assertEqual(parse_results(text, self.fixture, [reference, cold])[1]["status"], "passed")
        text = self.frame(reference) + self.frame(cold, after=after, generations=(-1, 1))
        result = parse_results(text, self.fixture, [reference, cold])[1]
        self.assertIn("finalize did not reinitialize", " ".join(result["failures"]))

    def test_missing_and_duplicate_records_fail_closed(self):
        record = self.record()
        with self.assertRaisesRegex(ValueError, "Missing cache frame"):
            parse_results("", self.fixture, [record])
        with self.assertRaisesRegex(ValueError, "duplicate cache frame"):
            parse_results(self.frame(record)*2, self.fixture, [record])


if __name__ == "__main__":
    unittest.main()
