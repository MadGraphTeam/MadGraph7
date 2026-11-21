"""Compile the real cache-index branch with the documented zero-size setting."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs.file_writers import FortranWriter


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "madgraph/iolibs/template_files/loop_optimized/TIR_interface.inc"


def cache_index_source(size, old_divisor=False):
    source = TEMPLATE.read_text()
    match = re.search(
        r"(?ims)^\s*if \(TIR_CACHE_SIZE\.eq\.0\) THEN.*?^\s*ENDIF\s*$",
        source,
    )
    if match is None:
        raise AssertionError("The generated cache-index branch was not found")
    branch = match.group() % {"proc_prefix": "TEST_"}
    if old_divisor:
        branch, count = re.subn(r"MAX\(TIR_CACHE_SIZE,1\)", "TIR_CACHE_SIZE", branch)
        if count != 1:
            raise AssertionError("Expected one protected cache divisor")
    return f"""      PROGRAM CHECK_CACHE_INDEX
      IMPLICIT NONE
      INTEGER TIR_CACHE_SIZE
      PARAMETER(TIR_CACHE_SIZE={size})
      INTEGER CTMODE,TIR_CACHE_INDEX,TEST_TIRCACHE_INDEX
      DO CTMODE=1,8
{branch}
        WRITE(*,*) TIR_CACHE_INDEX
      ENDDO
      END
      INTEGER FUNCTION TEST_TIRCACHE_INDEX(CTMODE)
      INTEGER CTMODE
      TEST_TIRCACHE_INDEX=CTMODE
      END
"""


@unittest.skipUnless(shutil.which("gfortran"), "GNU Fortran is required")
class TestZeroTirCache(unittest.TestCase):
    def compile_and_run(self, size, old_divisor=False):
        with tempfile.TemporaryDirectory(prefix="tir-cache-zero-") as temporary:
            directory = Path(temporary)
            with FortranWriter(str(directory / "index.f")) as writer:
                writer.writelines(cache_index_source(size, old_divisor))
            built = subprocess.run(
                [shutil.which("gfortran"), "-O0", "-ffixed-line-length-132",
                 "-fcheck=all", "index.f", "-o", "check"],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
            if built.returncode:
                return built, None
            run = subprocess.run([str(directory / "check")], cwd=directory,
                                 capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            return built, [int(value) for value in run.stdout.split()]

    def test_disabled_cache_compiles_and_uses_only_dummy_slot(self):
        built, indices = self.compile_and_run(0)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        self.assertEqual(indices, [0] * 8)

    def test_existing_exported_cache_sizes_keep_their_wrapping(self):
        for size, expected in ((1, [1] * 8), (2, [1, 2, 1, 2, 1, 2, 1, 2])):
            with self.subTest(size=size):
                built, indices = self.compile_and_run(size)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                self.assertEqual(indices, expected)

    def test_old_expression_is_rejected_even_in_unreached_branch(self):
        built, indices = self.compile_and_run(0, old_divisor=True)
        self.assertNotEqual(built.returncode, 0)
        self.assertIsNone(indices)
        self.assertIn("zero", built.stderr.lower())


if __name__ == "__main__":
    unittest.main()
