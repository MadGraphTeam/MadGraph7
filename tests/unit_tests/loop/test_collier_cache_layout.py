"""Exercise the rendered COLLIER cache clear against its consumer's layout.

The byte canary belongs to explicit oversized backing storage for the same
named COMMON. It does not depend on unrelated globals being adjacent. An old
declaration mutation demonstrates the resulting flag-offset mismatch and write
past the consumer's storage extent, even when its visible flags become false.
"""
from math import comb
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "madgraph/iolibs/template_files/loop_optimized/COLLIER_interface.inc"


def cache_declarations(source):
    """Keep the actual cache and status declarations from one template routine."""
    block = re.search(
        r"(?im)^\s*logical LOOP_ID_DONE\(NLOOPGROUPS\).*?"
        r"^\s*COMMON/%\(proc_prefix\)sCOLLIER_CACHE_STATUS/[^\n]*\n",
        source, re.DOTALL,
    )
    if block is None:
        raise AssertionError("COLLIER cache declarations were not found")
    # Explicit fixture seeding replaces DATA. This keeps every COMMON tentative,
    # allowing its largest declaration (the byte backing) to own the storage.
    return "\n".join(line for line in block.group().splitlines()
                     if not line.strip().upper().startswith("DATA ")) + "\n"


def fixture_sources(rank, groups, old_error_declaration=False):
    template = TEMPLATE.read_text()
    consumer = template.split("END SUBROUTINE %(proc_prefix)sCOLLIERLOOP", 1)[0]
    match = re.search(
        r"(?im)^\s*SUBROUTINE %\(proc_prefix\)sCLEAR_COLLIER_CACHE\(\).*?"
        r"^\s*END\s*$", template, re.DOTALL,
    )
    if match is None:
        raise AssertionError("COLLIER cache clear routine was not found")
    clear = match.group() + "\n"
    if old_error_declaration:
        clear, count = re.subn(
            r"%\((?:real|complex)_dp_format\)s(?= TIR_COEFS_ERROR\()",
            "%(complex_dp_format)s", clear,
        )
        if count != 1:
            raise AssertionError("Expected exactly one error-array declaration")
    substitutions = dict(proc_prefix="TEST_", maxrank=rank,
                         nloop_groups=groups, complex_dp_format="COMPLEX*16",
                         real_dp_format="REAL*8")
    consumer_declarations = cache_declarations(consumer) % substitutions
    clear_declarations = cache_declarations(clear) % substitutions
    ncoef = comb(rank + 4, 4)
    # Enough space for either layout plus a tail, including the malformed
    # complex-error case. No tested routine can run past this backing array.
    backing_size = ncoef * 3 * groups * 16 + (rank + 1) * groups * 16 + groups * 4 + 128
    constants = f"""      INTEGER NLOOPGROUPS,MAXRANK
      PARAMETER(NLOOPGROUPS={groups},MAXRANK={rank})
      INCLUDE 'loop_max_coefs.inc'
"""
    layout = """      LAYOUT(1)=LOC(TIR_COEFS_ERROR(0,1))-
     $ LOC(TIR_COEFS_DIRECT_MODE(0,1,1))
      LAYOUT(2)=LOC(LOOP_ID_DONE(1))-
     $ LOC(TIR_COEFS_DIRECT_MODE(0,1,1))
      LAYOUT(3)=LOC(LOOP_ID_DONE(NLOOPGROUPS))+
     $ STORAGE_SIZE(LOOP_ID_DONE(1))/8-
     $ LOC(TIR_COEFS_DIRECT_MODE(0,1,1))
      LAYOUT(4)=STORAGE_SIZE(TIR_COEFS_ERROR(0,1))/8
      LAYOUT(5)=LOC(TIR_COEFS_DIRECT_MODE(0,1,1))
"""
    probes = """      SUBROUTINE CONSUMER(ACTION,LAYOUT,COUNTS)
      USE ISO_C_BINDING, ONLY: C_INTPTR_T
      IMPLICIT NONE
      INTEGER ACTION,COUNTS(5),I,J,K
      INTEGER(C_INTPTR_T) LAYOUT(5)
""" + constants + consumer_declarations + layout + """      COUNTS=0
      IF(ACTION.EQ.1) THEN
        DO I=1,NLOOPGROUPS
          LOOP_ID_DONE(I)=.TRUE.
          DO J=0,LOOPMAXCOEFS-1
            DO K=1,3
              TIR_COEFS_DIRECT_MODE(J,K,I)=
     $         DCMPLX(1000*I+10*J+K,-1000*I-10*J-K)
            ENDDO
          ENDDO
          DO J=0,MAXRANK
            TIR_COEFS_ERROR(J,I)=100*I+J+0.5D0
          ENDDO
        ENDDO
        MUST_INIT_EVENT=.FALSE.
        COLLIER_CACHE_ACTIVE=17
        NCALLS_IN_CACHE=(/11,12,13,14/)
        NCOLLIERCALLS=(/21,22,23,24/)
      ELSEIF(ACTION.EQ.2) THEN
        COUNTS(1)=COUNT(TIR_COEFS_DIRECT_MODE.NE.(0D0,0D0))
        COUNTS(2)=COUNT(TIR_COEFS_ERROR.NE.0D0)
        COUNTS(3)=COUNT(LOOP_ID_DONE)
        IF(MUST_INIT_EVENT) COUNTS(4)=1
        IF(COLLIER_CACHE_ACTIVE.NE.17) COUNTS(5)=1
        COUNTS(5)=COUNTS(5)+COUNT(NCALLS_IN_CACHE.NE.
     $   (/11,12,13,14/))+COUNT(NCOLLIERCALLS.NE.(/21,22,23,24/))
      ENDIF
      END

      SUBROUTINE CLEAR_LAYOUT(LAYOUT)
      USE ISO_C_BINDING, ONLY: C_INTPTR_T
      IMPLICIT NONE
      INTEGER(C_INTPTR_T) LAYOUT(5)
""" + constants + clear_declarations + layout + """      END
"""
    driver = f"""      PROGRAM CHECK_LAYOUT
      USE ISO_C_BINDING, ONLY: C_INT8_T,C_INTPTR_T
      IMPLICIT NONE
      INTEGER(C_INT8_T) BACKING({backing_size})
      COMMON/TEST_COLLIER_TIR_COEFS/BACKING
      INTEGER(C_INTPTR_T) EXPECTED(5),ACTUAL(5)
      INTEGER COUNTS(5),PASS,CHANGED
      BACKING=90_C_INT8_T
      CALL CONSUMER(0,EXPECTED,COUNTS)
      CALL CLEAR_LAYOUT(ACTUAL)
      WRITE(*,*) 'CONSUMER',EXPECTED(1:4)
      WRITE(*,*) 'CLEAR',ACTUAL(1:4)
      WRITE(*,*) 'BASE',EXPECTED(5).EQ.LOC(BACKING(1)),
     $ ACTUAL(5).EQ.LOC(BACKING(1))
      CALL CONSUMER(1,EXPECTED,COUNTS)
      DO PASS=1,2
        CALL TEST_CLEAR_COLLIER_CACHE()
        CALL CONSUMER(2,EXPECTED,COUNTS)
        CHANGED=COUNT(BACKING(EXPECTED(3)+1:).NE.90_C_INT8_T)
        WRITE(*,*) 'AFTER',PASS,COUNTS,CHANGED
      ENDDO
      END
"""
    return {
        "collier.f90": "module collier\n  implicit none\nend module\n",
        "clear.f": clear % substitutions,
        "probes.f": probes,
        "driver.f": driver,
        "MadLoopParams.inc": "",
        "loop_max_coefs.inc": ("      INTEGER LOOPMAXCOEFS\n"
                               f"      PARAMETER(LOOPMAXCOEFS={ncoef})\n"),
    }


@unittest.skipUnless(shutil.which("gfortran"), "GNU Fortran is required for the COMMON ABI fixture")
class TestCollierCacheLayout(unittest.TestCase):
    def execute(self, rank, groups, old_error_declaration=False):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            for name, text in fixture_sources(rank, groups, old_error_declaration).items():
                (work / name).write_text(text)
            built = subprocess.run(
                [shutil.which("gfortran"), "-O0", "-ffixed-line-length-none",
                 "-fcheck=all", "collier.f90", "clear.f", "probes.f", "driver.f",
                 "-o", "check"], cwd=work, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(work / "check")], cwd=work,
                                 capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            rows = [line.split() for line in run.stdout.splitlines()]
            self.assertEqual([row[0] for row in rows],
                             ["CONSUMER", "CLEAR", "BASE", "AFTER", "AFTER"])
            return rows

    def test_rendered_clear_matches_consumer_layout_and_preserves_guards(self):
        for rank, groups in ((0, 3), (2, 3), (4, 2)):
            with self.subTest(rank=rank, groups=groups):
                rows = self.execute(rank, groups)
                coefficient_bytes = comb(rank + 4, 4) * 3 * groups * 16
                error_bytes = (rank + 1) * groups * 8
                expected = [coefficient_bytes, coefficient_bytes + error_bytes,
                            coefficient_bytes + error_bytes + groups * 4, 8]
                self.assertEqual(list(map(int, rows[0][1:])), expected)
                self.assertEqual(list(map(int, rows[1][1:])), expected)
                self.assertEqual(rows[2][1:], ["T", "T"])
                for repetition, row in enumerate(rows[3:], 1):
                    # All coefficients, errors and flags cleared; event reset;
                    # status counters and explicit backing canary untouched.
                    self.assertEqual(list(map(int, row[1:])),
                                     [repetition, 0, 0, 0, 1, 0, 0])

    def test_old_complex_error_declaration_shifts_flags_and_overwrites_canary(self):
        rank, groups = 2, 3
        rows = self.execute(rank, groups, old_error_declaration=True)
        consumer = list(map(int, rows[0][1:]))
        clear = list(map(int, rows[1][1:]))
        self.assertEqual(rows[2][1:], ["T", "T"])
        self.assertEqual(clear[0], consumer[0])
        self.assertEqual(clear[1] - consumer[1], (rank + 1) * groups * 8)
        self.assertEqual(clear[2] - consumer[2], (rank + 1) * groups * 8)
        self.assertEqual((consumer[3], clear[3]), (8, 16))
        for row in rows[3:]:
            self.assertGreater(int(row[-1]), 0, "Old layout must overwrite explicit tail canary")


if __name__ == "__main__":
    unittest.main()
