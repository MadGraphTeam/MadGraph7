"""Stress driver and classification: refusals are recorded, wrong accepted values block."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_stress as stress  # noqa: E402

STUB = """
module stub_state
 integer*8, save :: counters(20)=0
end module
subroutine SETPARA(card)
 character(*) card
end subroutine
subroutine UPDATE_AS_PARAM()
end subroutine
subroutine STUB_GET_ANSWER_DIMENSION(n)
 integer n
 n=1
end subroutine
subroutine STUB_GET_NSQSO_LOOP(n)
 integer n
 n=1
end subroutine
subroutine STUB_FORCE_STABILITY_CHECK(force)
 logical force
end subroutine
subroutine STUB_EXIT_MADLOOP()
end subroutine
subroutine STUB_GET_TIRUST_DP_ROUTE_STATISTICS(values,status)
 use stub_state
 integer*8 values(20)
 integer status
 values=counters
 status=0
end subroutine
subroutine STUB_SLOOPMATRIX_THRES(p,ans,threshold,acc,code)
 use stub_state
 real*8 p(0:3,3),ans(0:3,0:1),threshold,acc(0:1)
 integer code,point
 logical FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
 common /STUB_FPE_IN_REDUCTION/ FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
 integer,save::calls=0
 calls=calls+1
 point=nint(p(0,3))
 FPE_IN_DP_REDUCTION=.false.
 FPE_IN_QP_REDUCTION=.false.
 ans=dble(point)
 acc=1d-12
 code=219
 if(calls.le.2) code=139
 counters(1)=counters(1)+1
 if(point.eq.20)then
  counters(3)=counters(3)+1
  counters(8)=counters(8)+1
 endif
 if(point.eq.30)then
  counters(6)=counters(6)+1
  code=429
  acc=1d-2
 endif
end subroutine
"""


@unittest.skipUnless(shutil.which("gfortran"), "GNU Fortran required")
class TestStressDriver(unittest.TestCase):
    def test_refusals_are_recorded_and_warm_up_stops_when_steady(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            params = {"prefix": "STUB_", "nexternal": 3, "nborn": 1}
            (d / "driver.f").write_text(stress.render(stress.DRIVER.read_text(), params, STUB, routes=True))
            (d / "coupl.inc").write_text("      REAL*8 MU_R\n      COMMON /SCALE/ MU_R\n")
            (d / "stub.f90").write_text(STUB)
            subprocess.run([shutil.which("gfortran"), "-fcheck=all", "-o", "check", "stub.f90", "driver.f"],
                           cwd=d, check=True, capture_output=True)
            point = lambda e: f"1 0 0 1\n1 0 0 -1\n{e} 0 0 0\n"
            (d / "warm.in").write_text("3\n" + point(1) + point(2) + point(3))
            (d / "stress.in").write_text("2\n" + point(20) + point(30))
            completed = subprocess.run(["./check", "warm.in", "stress.in", "checked"], cwd=d,
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("STRESS-DONE", completed.stdout)
            warm = (d / "warm.dat").read_text().splitlines()
            self.assertEqual(len(warm), 5)  # two initialization calls, then three steady calls
            rows = stress.parse_values(d / "stress-values.dat", 1)
            self.assertEqual([(r["point"], r["status"]) for r in rows], [(1, 219), (1, 219), (2, 429), (2, 429)])
            routes = stress.parse_routes(d / "stress-routes.dat")
            self.assertEqual(routes[1]["delta"][:10], [1, 0, 1, 0, 0, 0, 0, 1, 0, 0])
            self.assertEqual(routes[2]["delta"][5], 1)
            self.assertIsNone(stress.accepted_backend(stress.point_rows({"rows": rows}, 2), [9]))
            self.assertEqual(stress.accepted_backend(stress.point_rows({"rows": rows}, 1), [9]), (9, "DP"))

    def test_driver_without_route_hooks_keeps_comment_anchors(self):
        text = stress.render(stress.DRIVER.read_text(), {"prefix": "STUB_", "nexternal": 3, "nborn": 1}, STUB)
        self.assertIn("C@STRESS_ROUTE_BEFORE@", text)
        self.assertNotIn("GET_TIRUST_DP_ROUTE_STATISTICS", text)
        with self.assertRaisesRegex(ValueError, "FPE flag layout"):
            stress.render(stress.DRIVER.read_text(), {"prefix": "STUB_", "nexternal": 3, "nborn": 1}, "")


class TestPointFiles(unittest.TestCase):
    def test_points_split_into_per_point_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "p.dat"
            path.write_text("2\n1 0 0 1\n1 0 0 -1\n2 0 0 0\n3 0 0 1\n3 0 0 -1\n6 0 0 0\n")
            blocks = stress.split_points(path)
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[1], ["3 0 0 1", "3 0 0 -1", "6 0 0 0"])
            path.write_text("2\n1 0 0 1\n1 0 0 -1\n2 0 0 0\n")
            with self.assertRaises(ValueError):
                stress.split_points(path)


class TestPartialOutputs(unittest.TestCase):
    def test_incomplete_trailing_point_is_dropped_only_when_partial(self):
        row = lambda p, j: f"{p} {j} 219 1.0 2.0 3.0 4.0 1e-12 1 0 0\n"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stress-values.dat"
            path.write_text(row(1, 0) + row(1, 1) + row(2, 0))
            self.assertEqual({r["point"] for r in stress.parse_values(path, 1, partial=True)}, {1})
            with self.assertRaises(ValueError):
                stress.parse_values(path, 1)


class TestStressClassification(unittest.TestCase):
    def lane(self, accepted, passed=None, status="complete"):
        lane = {"status": status, "accepted": accepted}
        if passed is not None:
            lane["vs_reference"] = {"passed": passed}
        return lane

    def test_only_wrong_accepted_values_block(self):
        reference = [{"point": 1, "order": 0}]
        self.assertEqual(stress.classify(self.lane(None), [9], reference), "refused")
        self.assertEqual(stress.classify(self.lane([9, "DP"], True), [9], reference), "accepted-agrees")
        self.assertEqual(stress.classify(self.lane([9, "QP"], False), [9], reference), "accepted-WRONG")
        self.assertEqual(stress.classify(self.lane([6, "QP"], True), [9, 6, 7, 1], reference), "rescued-by-6-agrees")
        self.assertEqual(stress.classify(self.lane([1, "DP"], False), [9, 6, 7, 1], reference), "rescued-by-1-WRONG")
        self.assertEqual(stress.classify(self.lane(None), [9, 6, 7, 1], reference), "unrescued")
        self.assertEqual(stress.classify(self.lane([9, "DP"]), [9], None), "accepted-unreferenced")
        self.assertEqual(stress.classify(self.lane(None, status="timeout"), [9], None), "driver-timeout")
        stopped = dict(self.lane(None), stopped="STOP")
        self.assertEqual(stress.classify(stopped, [9, 6, 7, 1], None), "unrescued (process STOP)")
        self.assertEqual(stress.classify(dict(stopped, stopped="hang"), [9], None), "refused (process hang)")

    def test_nonfinite_or_flagged_results_are_never_accepted(self):
        row = {"point": 1, "order": 0, "status": 219, "finite": True, "fpe_dp": False, "fpe_qp": False,
               "coefficients": [1.0, 2.0, 3.0, 4.0]}
        self.assertEqual(stress.accepted_backend([row], [9]), (9, "DP"))
        self.assertIsNone(stress.accepted_backend([dict(row, finite=False)], [9]))
        self.assertIsNone(stress.accepted_backend([dict(row, fpe_qp=True)], [9]))
        self.assertIsNone(stress.accepted_backend([dict(row, status=429)], [9]))
        self.assertIsNone(stress.accepted_backend([dict(row, status=216)], [9]))
        self.assertEqual(stress.accepted_backend([dict(row, status=328)], [9, 6, 7, 1]), (6, "QP"))  # Ninja QP alias


if __name__ == "__main__":
    unittest.main()
