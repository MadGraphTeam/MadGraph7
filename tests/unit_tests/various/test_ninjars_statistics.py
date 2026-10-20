"""Legacy and contextual MadLoop backend statistics remain distinguishable."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from xml.dom.minidom import parseString

from madgraph.madevent.sum_html import RunStatistics
from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd


class TestNinjaRSStatistics(unittest.TestCase):
    @staticmethod
    def xml(extra=""):
        return parseString("""<statistics>
          <u_return_code>0,0,7,0,0,0,0,0,11,0</u_return_code>
          <t_return_code>0,10,8,0,0</t_return_code>
          <h_return_code>0,0,10,8,0</h_return_code>
          <average_time>0.001</average_time>
          <cumulated_time>0.018</cumulated_time>
          <max_prec>1e-8</max_prec><min_prec>1e-12</min_prec>
          <n_evals>18</n_evals>%s</statistics>""" % extra).documentElement

    def test_old_xml_is_unchanged(self):
        stats = RunStatistics()
        stats.load_statistics(self.xml())
        self.assertEqual(stats["PJFry_usage"], 7)
        self.assertEqual(stats["Ninja_QP_usage"], 11)
        self.assertEqual(stats["NinjaRS_DP_usage"], 0)
        self.assertEqual(stats["NinjaRS_QP_usage"], 0)

    def test_mixed_xml_and_aggregation(self):
        stats = RunStatistics()
        stats.load_statistics(self.xml(
            "<ninjars_dp_usage>9</ninjars_dp_usage>"
            "<ninjars_qp_usage>5</ninjars_qp_usage>"))
        self.assertEqual(stats["PJFry_usage"], 2)
        self.assertEqual(stats["Ninja_QP_usage"], 2)
        self.assertEqual(stats["NinjaRS_DP_usage"], 9)
        self.assertEqual(stats["NinjaRS_QP_usage"], 5)
        self.assertIn("NinjaRS_DP", stats.nice_output("test"))
        old = RunStatistics()
        old.load_statistics(self.xml())
        stats.aggregate_statistics(old)
        self.assertEqual(stats["PJFry_usage"], 9)
        self.assertEqual(stats["Ninja_QP_usage"], 13)
        self.assertEqual(stats["NinjaRS_DP_usage"], 9)

    def test_nlo_logs_old_and_mixed(self):
        log = """Satistics from MadLoop:
 Total points tried: 18
 Stability unknown: 0
 Stable PS point: 18
 Unstable PS point (and rescued): 0
 Exceptional PS point (unstable and not rescued): 0
 Double precision used: 10
 Quadruple precision used: 8
 Initialization phase-space points: 0
 Unknown return code (100): 0
 Unknown return code (10): 0
 #Unit 2 = 7
 #Unit 8 = 11
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            for extra, expected in [
                ("", False),
                ("#NinjaRS DP = 9\n#NinjaRS QP = 5\n", True),
            ]:
                path.write_text(log + extra)
                _, report = aMCatNLOCmd.compile_advanced_stats(
                    SimpleNamespace(me_dir=directory),
                    [str(path)], [], "")
                self.assertEqual("NinjaRS (double precision)" in report, expected)
                self.assertEqual("NinjaRS (quadruple precision)" in report, expected)
                if expected:
                    self.assertRegex(report, r"PJFry\+\+\s+2(?:\n|$)")
                    self.assertRegex(report, r"Ninja \(quadruple precision\)\s+2(?:\n|$)")


if __name__ == "__main__":
    unittest.main()
