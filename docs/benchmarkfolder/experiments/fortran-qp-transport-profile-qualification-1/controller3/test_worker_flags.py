"""Pure source checks: no compiler, process launch, or numerical execution."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("private_worker_flags", HERE / "worker_flags.py")
flags = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flags)
BASE = (HERE.with_name("fortran-qp-transport-profile-1") / "source-preview" / "check_sa.f").read_text()


def test_sealed_input_and_deterministic_transform():
    assert hashlib.sha256(BASE.encode()).hexdigest() == flags.BASE_WORKER_SHA256
    assert flags.instrument_worker(BASE) == flags.instrument_worker(BASE)


@pytest.mark.parametrize("altered", [BASE + "\n", BASE.replace("NEXTERNAL=5", "NEXTERNAL=6"), ""])
def test_refuses_unpinned_source(altered):
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        flags.instrument_worker(altered)


def test_refuses_double_instrumentation():
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        flags.instrument_worker(flags.instrument_worker(BASE))


def test_exact_reversible_additions_leave_original_program_intact():
    patched = flags.instrument_worker(BASE)
    assert patched.endswith(flags.CAPTURE)
    recovered = patched[:-len(flags.CAPTURE)]
    recovered = recovered.replace(flags.DECLARATIONS, "", 1)
    recovered = recovered.replace(flags.OPEN_FLAGS, "", 1)
    recovered = recovered.replace(flags.OBSERVED_ANCHOR, flags.ANCHOR, 1)
    recovered = recovered.replace("         CLOSE(77)\n", "", 1)
    assert recovered == BASE


def test_only_profile_anchor_is_observed_and_post_capture_precedes_loggers():
    patched = flags.instrument_worker(BASE)
    body = patched.split("      SUBROUTINE QP_CAPTURE_FP", 1)[0]
    assert body.count("CALL QP_CAPTURE_FP") == 2
    assert body.count("CALL ML5_0_SLOOPMATRIX_THRES") == BASE.count("CALL ML5_0_SLOOPMATRIX_THRES")
    anchor = body.index(flags.OBSERVED_ANCHOR)
    logger = body.index("            CALL BENCHMARK_TRACE_WARMUP(76,")
    assert anchor + len(flags.OBSERVED_ANCHOR) == logger
    assert "QP_CAPTURE_FP" not in body.split(" 100  CONTINUE", 1)[0]
    assert "IF(NLOOPS.LT.0.OR.NLOOPS.GT.2) GOTO 800" in body


def test_capture_is_query_only_and_unknown_is_not_false():
    calls = [line.strip().split("(", 1)[0] for line in flags.CAPTURE.splitlines()
             if line.strip().startswith("CALL ")]
    assert calls == ["CALL IEEE_GET_FLAG", "CALL IEEE_GET_HALTING_MODE", "CALL IEEE_GET_ROUNDING_MODE"]
    assert "IEEE_SET" not in flags.CAPTURE
    assert "FLAGS=-1" in flags.CAPTURE and "HALTING=-1" in flags.CAPTURE
    assert "ROUNDING=-1" in flags.CAPTURE
    assert "IEEE_SUPPORT_FLAG" in flags.CAPTURE
    assert "IEEE_SUPPORT_HALTING" in flags.CAPTURE
    assert "IEEE_SUPPORT_ROUNDING" in flags.CAPTURE
    assert not any(word in flags.CAPTURE for word in ("WRITE(", "READ(", "SQRT(", "REAL(", "DBLE("))


def test_schema_and_exclusive_output():
    assert len(flags.FLAGS_COLUMNS) == 14
    assert flags.FLAGS_COLUMNS[:4] == ("mode", "point", "phase", "rounding")
    assert "(14I12)" in flags.OBSERVED_ANCHOR
    assert "FILE='ieee-flags.dat',STATUS='NEW'" in flags.OPEN_FLAGS
    assert "DO QP_FP_PHASE=1,2" in flags.OBSERVED_ANCHOR
    assert "sticky flags are not cleared" in flags.OPEN_FLAGS


def test_added_fixed_form_lines_fit_72_columns():
    for addition in (flags.DECLARATIONS, flags.OPEN_FLAGS, flags.OBSERVED_ANCHOR, flags.CAPTURE):
        for line in addition.splitlines():
            assert len(line) <= 72, line
