import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from tirust_benchmark_poles import fixture_stream, parse_results, pole_split_checks


class TestPoleDiagnostics(unittest.TestCase):
    def fixture(self):
        return {"name": "test-tadpole", "n": 1, "rank": 0, "mu2": 7,
                "shifts": [[0, 0, 0, 0]], "mass2": [[4, 0]]}

    def test_stream_expands_precision_and_separate_poles(self):
        stream, selected, omitted = fixture_stream([self.fixture()], 1, 2, "both")
        self.assertEqual(len(selected), 6)
        self.assertEqual(stream.splitlines()[0], "6")
        self.assertFalse(omitted)
        self.assertEqual({(item["pole_mode"], item["ctmode"]) for item in selected},
                         {(mode, precision) for mode in (0, 1, 2) for precision in (1, 4)})

    def test_high_rank_omissions_are_explicit(self):
        fixture = dict(self.fixture(), rank=6)
        stream, selected, omitted = fixture_stream([fixture], 15, 1, "dp")
        self.assertFalse(selected)
        self.assertEqual(len(omitted), 1)
        self.assertIn("rank", omitted[0]["reason"])

    def text_and_cases(self, partial_pole="2", partial_finite="NaN"):
        fixture = dict(self.fixture(), ctmode=1, pole_mode=0, loops=1)
        text = ("F 1 1 0 1 1 T F 1 0.1\n"
                "C 1 1 0 1 0 8 0 2 0 0 0\n"
                "F 1 1 0 0 1 T T 1 0.05\n"
                f"C 1 1 0 0 0 {partial_finite} 0 {partial_pole} 0 0 0\n")
        return text, [fixture]

    def test_partial_result_requires_nan_finite_and_equal_poles(self):
        text, cases = self.text_and_cases()
        self.assertEqual(parse_results(text, cases)[0]["status"], "passed")
        text, cases = self.text_and_cases(partial_pole="2.1")
        self.assertEqual(parse_results(text, cases)[0]["status"], "failed")
        text, cases = self.text_and_cases(partial_finite="0")
        self.assertEqual(parse_results(text, cases)[0]["status"], "failed")

    def test_nonfinite_evaluated_pole_and_missing_frame_rejected(self):
        text, cases = self.text_and_cases(partial_pole="NaN")
        with self.assertRaisesRegex(ValueError, "Nonfinite evaluated pole"):
            parse_results(text, cases)
        text, cases = self.text_and_cases()
        with self.assertRaises(KeyError):
            parse_results("\n".join(text.splitlines()[:2]), cases)

    def test_split_poles_recombine_and_finite_term_is_mode_independent(self):
        _, selected, _ = fixture_stream([self.fixture()], 1, 1, "dp")
        records = []
        for identifier, fixture in enumerate(selected, 1):
            mode = fixture["pole_mode"]
            for finite in (1, 0):
                value = "8" if finite else "NaN"
                pole = 2 if mode == 0 else 1
                records.append(f"F {identifier} 1 {mode} {finite} 1 T {'F' if finite else 'T'} 1 0.1")
                records.append(f"C {identifier} 1 {mode} {finite} 0 {value} 0 {pole} 0 0 0")
        text = "\n".join(records)
        self.assertTrue(all(check["status"] == "passed" for check in pole_split_checks(text, selected)))
        wrong = text.replace("C 2 1 1 1 0 8 0 1", "C 2 1 1 1 0 9 0 1")
        self.assertEqual(pole_split_checks(wrong, selected)[0]["status"], "failed")
        unavailable = text.replace("F 2 1 1 0 1 T", "F 2 1 1 0 1 F")
        self.assertEqual(pole_split_checks(unavailable, selected)[1]["status"], "unavailable")

    def test_partial_nan_mask_rejects_nonfinite_imaginary_part(self):
        text, selected = self.text_and_cases()
        text = text.replace("NaN 0 2", "NaN NaN 2")
        self.assertEqual(parse_results(text, selected)[0]["status"], "failed")

    @unittest.skipUnless(shutil.which("gfortran"), "GNU Fortran is needed for the driver ABI fixture")
    def test_compiled_driver_dp_qp_and_nan_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "coupl.inc").write_text("      REAL*8 MU_R,NF\n      PARAMETER(NF=5D0)\n      COMMON /SCALE_TEST/ MU_R\n")
            (directory / "MadLoopParams.inc").write_text(
                "      INTEGER MLReductionLib(10),TIRustGlobalCache,LoopPoleMode\n"
                "      LOGICAL TIRustComputeFinite\n"
                "      COMMON /PARAM_TEST/ MLReductionLib,TIRustGlobalCache,LoopPoleMode,TIRustComputeFinite\n")
            (directory / "loop_max_coefs.inc").write_text("      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=1)\n")
            driver = (ROOT / "tools/tirust_benchmark_poles.f").read_text().replace("@PREFIX@", "TEST_")
            (directory / "driver.f").write_text(driver)
            stub = """      SUBROUTINE SETPARA(NAME)
      CHARACTER*(*) NAME
      END
      SUBROUTINE DEFAULTPARAM()
      END
      SUBROUTINE UPDATE_AS_PARAM()
      END
      SUBROUTINE TEST_CLEAR_TIRUST_CACHE()
      END
      SUBROUTINE TEST_FINALIZE_TIRUST()
      END
"""
            for prefix, kind in (("", "8"), ("MP_", "16")):
                stub += f"""      SUBROUTINE TEST_{prefix}TIRUST_TENSOR_DIAGNOSTIC(CTMODE,N,R,PL,MASSES,COEFS,STABLE)
      USE, INTRINSIC :: IEEE_ARITHMETIC
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      INTEGER CTMODE,N,R
      REAL(KIND={kind}) PL(0:3,N)
      COMPLEX*16 MASSES(N)
      COMPLEX(KIND={kind}) COEFS(0:0,3)
      LOGICAL STABLE
      COEFS=CMPLX(0,0,KIND={kind})
      COEFS(0,1)=CMPLX(8,0,KIND={kind})
      COEFS(0,2)=CMPLX(2,0,KIND={kind})
      IF(.NOT.TIRustComputeFinite) COEFS(0,1)=CMPLX(IEEE_VALUE(REAL(0,KIND={kind}),IEEE_QUIET_NAN),0,KIND={kind})
      STABLE=.TRUE.
      END
"""
            (directory / "stub.f").write_text(stub)
            stream, selected, _ = fixture_stream([self.fixture()], 1, 2, "both")
            (directory / "fixtures.dat").write_text(stream)
            subprocess.run(["gfortran", "-ffixed-line-length-none", "driver.f", "stub.f", "-o", "driver"],
                           cwd=directory, check=True, capture_output=True)
            subprocess.run(["./driver", "fixtures.dat"], cwd=directory, check=True, capture_output=True)
            results = parse_results((directory / "pole-values.dat").read_text(), selected)
            self.assertEqual(len(results), 6)
            self.assertTrue(all(result["status"] == "passed" for result in results))


if __name__ == "__main__":
    unittest.main()
