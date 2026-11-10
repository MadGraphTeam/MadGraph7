"""Offline checks of the retained scalar-input extraction; no compiler or FFI."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cms_retained_extract", HERE / "extract.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


class RetainedInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads((HERE / "point20-inputs.json").read_text())
        cls.text = (extract.SOURCE / "output.log").read_text()
        cls.records = extract.parse(cls.text)

    def test_source_hashes_and_counts(self):
        for name, expected in self.saved["source_files"].items():
            self.assertEqual(extract.sha(extract.ROOT / name), expected)
        self.assertEqual(len(self.records), 840)
        self.assertEqual(sum(len(r["inputs"]) for r in self.records), 2708)
        self.assertEqual(sum(k.startswith("p") for r in self.records
                             for k, _ in r["inputs"]), 1076)

    def test_representatives_are_exact_source_records(self):
        self.assertEqual(len(self.saved["selected"]), 18)
        for selected in self.saved["selected"]:
            record = self.records[selected["ordinal"] - 1]
            self.assertEqual({k: selected[k] for k in record}, record)
            self.assertEqual(selected["features"], extract.features(record))

    def test_all_four_families_and_both_small_input_causes(self):
        selected = self.saved["selected"]
        self.assertEqual({r["family"] for r in selected}, {"a0", "b11", "c0", "d0"})
        self.assertTrue(any(r["features"]["has_physical_small_transfer"] for r in selected))
        self.assertTrue(any(r["features"]["has_small_external_residual"] for r in selected))
        self.assertTrue(any(r["features"]["has_complex_mass"] for r in selected))
        self.assertTrue(any(r["features"]["all_masses_exact_zero"] for r in selected))

    def test_incomplete_or_misordered_records_are_rejected(self):
        good = """onshell: 1e-6
muscale: 1e3
pp: 1e-7
m1: (1,2)
m2: (0,0)
b11(2): (0,0)
"""
        self.assertEqual(len(extract.parse(good)), 1)
        with self.assertRaises(AssertionError):
            extract.parse(good.replace("m2: (0,0)\n", ""))
        with self.assertRaises(AssertionError):
            extract.parse(good.replace("m1: (1,2)\nm2: (0,0)",
                                       "m2: (0,0)\nm1: (1,2)"))
        with self.assertRaises(AssertionError):
            extract.parse(good.replace("pp: 1e-7", "pp: NaN"))


if __name__ == "__main__":
    unittest.main()
