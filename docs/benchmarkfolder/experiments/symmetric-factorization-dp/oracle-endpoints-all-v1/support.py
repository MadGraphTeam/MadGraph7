#!/usr/bin/env python3
"""Build helpers for the isolated binary64 experiment, never production files."""
import sys
sys.dont_write_bytecode = True

import hashlib
from pathlib import Path
import re
import subprocess
import time

import factor

ROOT = factor.ROOT
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines

FLAGS = {
    "standalone": ["-O3", "-w", "-fPIC", "-ffixed-line-length-132"],
    "nlo": ["-O3", "-w", "-fPIC", "-ffixed-line-length-132", "-fno-automatic"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instrument(source):
    source = source.replace("  implicit none", "  implicit none\n"
        "  INTEGER FAST,FALLBACK\n  COMMON/FACTOR_TRACE/FAST,FALLBACK", 1)
    assert source.count("C FACTOR_FAST_PATH") == 1
    source = source.replace("C FACTOR_FAST_PATH", "C FACTOR_FAST_PATH\nFAST=FAST+1")
    source, count = re.subn(
        r"(?m)^(\s*CALL TEST_UPDATE_WL_\d_1\(.*)$",
        r"  FALLBACK=FALLBACK+1\n\1", source)
    assert count >= 5
    return source


def write_sources(work, rank, vertex):
    work.mkdir(parents=True, exist_ok=True)
    (work / "coef_specs.inc").write_text(
        "      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n"
        "      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n")
    (work / "loop_max_coefs.inc").write_text(
        "      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=210)\n")
    generator = FortranPolynomialRoutines(
        6, updater_max_rank=2, proc_prefix="TEST_", sub_prefix="TEST_",
        coef_format="complex*16")
    sources = {
        "constants.f": generator.write_polynomial_constant_module(),
        "original.f": generator.write_compact_wl_updater(rank, 1),
        "factor.f": instrument(factor.generate(generator, rank, vertex)),
    }
    for name, source in sources.items():
        with FortranWriter(str(work / name)) as writer:
            writer.writelines(source)
    return sources


def run(command, work, log):
    start = time.monotonic()
    result = subprocess.run(command, cwd=work, capture_output=True, text=True)
    (work / log).write_text(result.stdout + result.stderr)
    record = dict(command=command, returncode=result.returncode,
                  seconds=time.monotonic() - start,
                  output=result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{work / log}: {record['output']}")
    return record
