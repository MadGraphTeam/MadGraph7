"""Compile the real timing driver against deterministic recording stubs.

These tests deliberately cannot reach the calibration target. They qualify
protocol, refusal handling, generated array stride and bridge rank-six capacity,
not numerical accuracy or timing performance.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("gfortran"), "GNU Fortran is needed for the driver ABI fixture")
class TestPoleTimingDriver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        p = cls.directory
        (p / "coupl.inc").write_text("      REAL*8 MU_R\n      COMMON /SCALE_TEST/ MU_R\n")
        (p / "MadLoopParams.inc").write_text(
            "      INTEGER MLReductionLib(10),TIRustGlobalCache,LoopPoleMode\n"
            "      LOGICAL TIRustComputeFinite\n"
            "      COMMON /PARAM_TEST/ MLReductionLib,TIRustGlobalCache,LoopPoleMode,TIRustComputeFinite\n")
        (p / "loop_max_coefs.inc").write_text("      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=15)\n")
        module = """module mtirust
  implicit none
  integer, parameter :: tirust_success=0
  interface tirust_tensor_evaluate
    module procedure tensor_dp,tensor_qp
  end interface
contains
  subroutine tirust_register_plan(n,r,status)
    integer :: n,r,status
    status=0
  end subroutine
  subroutine tirust_clear_cache(status)
    integer :: status
    status=0
  end subroutine
"""
        for name, kind in (("dp", 8), ("qp", 16)):
            module += f"""  subroutine tensor_{name}(n,r,p,s,m,mu2,c,status,mode,finite,check,cache)
    integer :: n,r,status,mode
    real(kind={kind}) :: p(0:,:),s(:,:),mu2
    complex(kind={kind}) :: m(:),c(:,:)
    logical :: finite,check,cache
    if(mode/=0.or.check.or.cache) stop 81
    if(size(c,1)/=(r+1)*(r+2)*(r+3)*(r+4)/24) stop 82
    c=cmplx(0,0,kind={kind})
    c(:,1)=cmplx(2,0,kind={kind})
    c(:,2)=cmplx(3,0,kind={kind})
    c(:,3)=cmplx(4,0,kind={kind})
    if(finite) c(:,4)=cmplx(8,0,kind={kind})
    status=0
  end subroutine
"""
        (p / "mtirust.f90").write_text(module + "end module\n")
        stub = """      SUBROUTINE SETPARA(NAME)
      CHARACTER*(*) NAME
      END
      SUBROUTINE DEFAULTPARAM()
      END
      SUBROUTINE UPDATE_AS_PARAM()
      END
      SUBROUTINE TEST_FINALIZE_TIRUST()
      END
"""
        for prefix, kind in (("", 8), ("MP_", 16)):
            stub += f"""      SUBROUTINE TEST_{prefix}TIRUST_TENSOR_DIAGNOSTIC(CTMODE,N,R,PL,MASSES,COEFS,STABLE)
      USE, INTRINSIC :: IEEE_ARITHMETIC
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      INTEGER CTMODE,N,R
      REAL(KIND={kind}) PL(0:3,N)
      COMPLEX*16 MASSES(N)
      COMPLEX(KIND={kind}) COEFS(0:14,3)
      LOGICAL STABLE
      IF(TIRustGlobalCache.NE.0.OR.LoopPoleMode.NE.0) STOP 83
      STABLE=.FALSE.
      IF(N.EQ.6) RETURN
      IF(N.EQ.2.AND.PL(1,1).NE.1) STOP 84
      IF(REAL(MASSES(1)).NE.9) STOP 85
      COEFS=CMPLX(0,0,KIND={kind})
      COEFS(:,1)=CMPLX(8,0,KIND={kind})
      COEFS(:,2)=CMPLX(2,0,KIND={kind})
      COEFS(:,3)=CMPLX(1,0,KIND={kind})
      IF(.NOT.TIRustComputeFinite) COEFS(:,1)=CMPLX(IEEE_VALUE(REAL(0,KIND={kind}),IEEE_QUIET_NAN),0,KIND={kind})
      PL=123
      MASSES=456
      STABLE=.TRUE.
      END
"""
        (p / "stub.f").write_text(stub)
        driver = (ROOT / "tools/tirust_benchmark_pole_timing.f").read_text().replace("@PREFIX@", "TEST_")
        (p / "driver.f").write_text(driver)
        subprocess.run(["gfortran", "-ffixed-line-length-none", "-fcheck=all", "mtirust.f90",
                        "driver.f", "stub.f", "-o", "driver"], cwd=p, check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def execute(self, scope, n, rank, precision):
        lines = [f"{scope} {n} {rank} 7 {precision} 7 3600 1 1"]
        lines += ["0 0 0 0"] + ["0 1 0 0"]*(n-1)
        lines += ["9 0"]*n
        (self.directory / "input.dat").write_text("\n".join(lines)+"\n")
        subprocess.run(["./driver", "input.dat"], cwd=self.directory, check=True, capture_output=True)
        return [line.split() for line in (self.directory / "pole-timing-values.dat").read_text().splitlines()]

    def test_generated_stride_mask_and_bounded_calibration(self):
        for precision in (1, 4):
            with self.subTest(precision=precision):
                rows = self.execute(0, 2, 2, precision)
                frames = [row for row in rows if row[0] == "F"]
                self.assertEqual([(int(row[1]),int(row[3])) for row in frames], [(0,1),(0,0),(1,1),(1,0)])
                self.assertTrue(all(row[4:7] == ["1","T","15"] for row in frames))
                self.assertEqual(rows[-1], ["E","0"])
                coefficients = [row for row in rows if row[0] == "C"]
                self.assertEqual(len(coefficients), 4*15)
                for row in coefficients:
                    self.assertEqual(list(map(float,row[6:])), [0,2,0,1,0])
                    self.assertEqual(row[5].lower() == "nan", row[3] == "0")

    def test_bridge_rank_six_exceeds_adapter_capacity_without_fabricated_nan(self):
        for precision in (1, 4):
            with self.subTest(precision=precision):
                rows = self.execute(1, 1, 6, precision)
                self.assertEqual(rows[0][1:6], ["1",str(precision),"1","6","210"])
                coefficients = [row for row in rows if row[0] == "C"]
                self.assertEqual(len(coefficients), 4*210)
                self.assertEqual(coefficients[-1][4], "209")
                for row in coefficients:
                    self.assertEqual(list(map(float,row[5:])), [8 if row[3] == "1" else 0,0,5,0,4,0])
                self.assertEqual(rows[-1], ["E","0"])

    def test_refusal_keeps_both_preflights_and_never_times(self):
        rows = self.execute(0, 6, 2, 1)
        frames = [row for row in rows if row[0] == "F"]
        self.assertEqual([(row[1],row[3],row[4],row[5]) for row in frames],
                         [("0","1","0","F"),("0","0","0","F")])
        self.assertFalse(any(row[0] == "C" for row in rows))
        self.assertEqual(rows[-1], ["E","0"])


if __name__ == "__main__":
    unittest.main()
