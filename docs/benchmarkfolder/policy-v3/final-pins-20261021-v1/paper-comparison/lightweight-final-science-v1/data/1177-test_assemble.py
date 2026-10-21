"""Small raw-data diagnostics controls; no numerical engines or compilers."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

ROOT=Path.cwd()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
adapter=load("private_random_assembler",Path(__file__).with_name("assemble.py"))
fixture=load("public_dp_test_fixtures",ROOT/"tests/unit_tests/loop/test_ninjars_dp_stability.py")

class CrossBackendDiagnostic(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def data(self,left=1.,right=1.,rejected=0,qualified=True):
        runs=[]
        for tool,value in ((6,left),(8,right)):
            raw=self.root/(str(tool)+".raw");raw.write_text(fixture.raw_text(tool=tool,values=[value]*3,rejected=rejected if tool==8 else 0))
            runs.append({"case":"fixture","tool":tool,"N":1,"nso":1,"helicity":-1,
                "artifacts":{"points":{"path":"shared-point-stream","sha256":"same"}},
                "raw":{"path":str(raw),"sha256":adapter.dp.digest(raw)}})
        summary={"cases":{"fixture":{"tools":{str(tool):{"physical_qualified":qualified,
            "initialization":{"safe":qualified}} for tool in (6,8)}}}}
        return summary,runs
    def result(self,**kwargs):
        summary,runs=self.data(**kwargs);return adapter.discrepancy(summary,runs)["cases"]["fixture"]
    def test_true_point_scale_and_small_finite_discrepancy(self):
        result=self.result(left=2.,right=2.+1e-12)
        point=result["points_data"][0]
        self.assertEqual(point["scale"],2.+1e-12)
        self.assertEqual(result["finite_coefficient_tolerance_failures"],0)
        finite=next(c for c in point["coefficients"] if c["coefficient"]=="finite")
        self.assertAlmostEqual(finite["absolute_difference"],(2.+1e-12)-2.)
        self.assertTrue(finite["passes_tolerance"])
    def test_rejected_equal_zeros_are_never_success_conditioned(self):
        result=self.result(left=0.,right=0.,rejected=1)
        self.assertEqual(result["point_failures"],1)
        self.assertEqual(result["points_data"][0]["failure_categories"]["8"],"reduction_rejected")
        self.assertTrue(all(c["passes_tolerance"] is None for c in result["points_data"][0]["coefficients"]))
    def test_nonfinite_point_has_no_invented_common_scale(self):
        result=self.result(left=1.,right=float("nan"))
        self.assertIsNone(result["points_data"][0]["scale"])
        self.assertEqual(result["point_failures"],1)
    def test_opposite_large_finite_values_do_not_overflow_json_diagnostic(self):
        result=self.result(left=1e308,right=-1e308)
        finite=next(c for c in result["points_data"][0]["coefficients"] if c["coefficient"]=="finite")
        self.assertTrue(finite["absolute_difference_overflow"])
        self.assertIsNone(finite["absolute_difference"])
        self.assertEqual(finite["relative_difference"],2.)
        self.assertFalse(finite["passes_tolerance"])
    def test_unsafe_warmup_is_separate_from_point_failure_denominator(self):
        result=self.result(qualified=False)
        self.assertEqual(result["point_failures"],0)
        self.assertFalse(result["initialization"]["8"]["safe"])
        self.assertFalse(result["points_data"][0]["physical_qualified"])
        self.assertTrue(all(c["passes_tolerance"] is None for c in result["points_data"][0]["coefficients"]))
    def test_missing_tool_pending_and_mismatched_stream_rejected(self):
        summary,runs=self.data()
        self.assertEqual(adapter.discrepancy(summary,runs[:1])["cases"]["fixture"]["status"],"pending_matched_tools")
        runs[1]["artifacts"]["points"]["sha256"]="different"
        with self.assertRaises(ValueError):adapter.discrepancy(summary,runs)

if __name__=="__main__":unittest.main(verbosity=2)
