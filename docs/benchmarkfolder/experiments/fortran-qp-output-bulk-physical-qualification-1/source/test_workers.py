"""Source-only checks; none compile Fortran or execute a reducer."""
import hashlib
import importlib.util
from pathlib import Path
import re

import pytest

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
BASE = ROOT / (
    ".benchmark-work/tirust/campaigns/entry-latch-cache-guard-1/"
    "outputs/candidate/qcd_n3_uux_zgg/SubProcesses/P0_uux_zgg/check_sa.f"
)
spec = importlib.util.spec_from_file_location("physical_worker_sources", HERE / "workers.py")
workers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workers)


@pytest.fixture
def base():
    text = BASE.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == workers.BASE_SHA256
    return text


def test_template_expansion_is_exact_frozen_base(base):
    template = (ROOT / "tools/tirust_benchmark_worker.f").read_text()
    assert hashlib.sha256(template.encode()).hexdigest() == workers.TEMPLATE_SHA256
    assert template.replace("@PREFIX@", "ML5_0_").replace(
        "@NEXTERNAL@", "5").replace("@NBORN@", "1") == base


@pytest.mark.parametrize("transform", [workers.parity_driver, workers.timings_driver])
def test_exact_base_hash_required(base, transform):
    for changed in [base + "\n", base.replace("NEXTERNAL=5", "NEXTERNAL=6")]:
        with pytest.raises(ValueError, match="Frozen generated worker"):
            transform(changed)
    with pytest.raises(TypeError):
        transform(base.encode())


@pytest.mark.parametrize("transform", [workers.parity_driver, workers.timings_driver])
def test_full_original_warmup_and_run_body_preserved(base, transform):
    result = transform(base)
    warmup = base.split("      CALL SETPARA", 1)[1].split(
        "      PREVIOUS_SEQUENCE=0\n", 1)[0]
    assert "      CALL SETPARA" + warmup in result
    original_run = base.split("      IF(TRIM(COMMAND).NE.'RUN')", 1)[1]
    assert "      IF(TRIM(COMMAND).NE.'RUN')" + original_run in result
    assert result.count("CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)") == 1
    assert "DO K=1,1000+NPOINTS" in result
    assert "IF(WARM_POINTS.GE.NPOINTS) EXIT" in result


@pytest.mark.parametrize("transform", [workers.parity_driver, workers.timings_driver])
def test_fixed_form_columns_and_unique_unit_declarations(base, transform):
    result = transform(base)
    assert all(len(line) <= 72 for line in result.splitlines()
               if line and line[0] not in "Cc*!")
    assert result.count("PROGRAM BENCHMARK_MADLOOP_WORKER") == 1
    assert result.count("SUBROUTINE BENCHMARK_BULK_WORDS(") == 1
    assert result.count("SUBROUTINE BENCHMARK_BULK_PARSE(") == 1
    assert "QP_WIRE" not in result
    assert "@PREFIX@" not in result


def test_parity_exact102_occurrences_and_unique_wrapper_markers(base):
    result = workers.parity_driver(base)
    assert result.count("C     BULK_PARITY_BEGIN\n") == 1
    assert result.count("C     BULK_PARITY_END\n") == 1
    body = result.split("C     BULK_PARITY_BEGIN\n")[1].split(
        "C     BULK_PARITY_END\n")[0]
    assert "DO BULK_OCCURRENCE=1,NPOINTS+2" in body
    assert "BULK_POINT=1\n" in body and "BULK_POINT=16\n" in body
    assert "BULK_POINT=BULK_OCCURRENCE-2" in body
    assert body.count("CALL ML5_0_SLOOPMATRIX_THRES(") == 1
    assert "BULK_PHASE=1" in body and "BULK_PHASE=2" in body and "BULK_PHASE=3" in body
    assert "IF(BULK_IOS.NE.0.OR.NPOINTS.NE.100) GOTO 800" in result
    assert "'PARITYDONE ',SEQUENCE" in result
    assert "FILE='parity-words.dat'" in result
    assert "FILE='parity-ieee.dat'" in result


def test_parity_ieee_queries_are_read_only_and_immediately_bracket_call(base):
    result = workers.parity_driver(base)
    body = result.split("C     BULK_PARITY_BEGIN\n")[1].split(
        "C     BULK_PARITY_END\n")[0]
    assert """            CALL BENCHMARK_BULK_IEEE(BULK_PRE_FLAGS,
     $           BULK_PRE_HALTS,BULK_PRE_ROUND)
            CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
            CALL BENCHMARK_BULK_IEEE(BULK_POST_FLAGS,
     $           BULK_POST_HALTS,BULK_POST_ROUND)
""" in body
    assert not re.search(r"IEEE_(SET|CLEAR)", result)
    for name in ["IEEE_INVALID", "IEEE_DIVIDE_BY_ZERO", "IEEE_OVERFLOW",
                 "IEEE_UNDERFLOW", "IEEE_INEXACT", "IEEE_GET_FLAG",
                 "IEEE_GET_HALTING_MODE", "IEEE_GET_ROUNDING_MODE"]:
        assert name in result
    assert "'(A,16I12)'" in result


def test_words_use_only_integer_hex_transfer_and_actual_accuracy_bounds(base):
    result = workers.parity_driver(base)
    helper = result.split("      SUBROUTINE BENCHMARK_BULK_WORDS", 1)[1].split(
        "      SUBROUTINE BENCHMARK_BULK_IEEE", 1)[0]
    assert "DO J=0,NDIM" in helper and "DO I=0,3" in helper
    assert "DO J=0,NSO" in helper
    assert "WORD=TRANSFER(ANS(I,J),WORD)" in helper
    assert "WORD=TRANSFER(ACC(J),WORD)" in helper
    assert "Z16.16" in helper and "ES26" not in helper


def test_timing_clock_encloses_only_loop_no_io_or_ieee_observers(base):
    result = workers.timings_driver(base)
    block = result.split("C     BULK_ANCHOR_CLOCK_BEGIN\n")[1].split(
        "C     BULK_ANCHOR_CLOCK_END\n")[0]
    assert block.count("CALL SYSTEM_CLOCK(") == 2
    assert "DO L=1,NLOOPS" in block
    assert block.count("CALL ML5_0_SLOOPMATRIX_THRES(") == 1
    assert "CALL UPDATE_AS_PARAM()" in block
    assert "TIMED_CODES(CODE)=TIMED_CODES(CODE)+1" in block
    assert "IF(.NOT.BENCHMARK_WARM_STATUS(CODE)) STOP 4" in block
    assert "IF(TENS.NE.1.AND.TENS.NE.2) STOP 4" in block
    assert not re.search(r"\b(READ|WRITE|OPEN|CLOSE|FLUSH)\s*\(", block)
    assert "IEEE" not in result and "QP_WIRE" not in result
    assert "BENCHMARK_BULK_WORDS" not in block
    for name in ["anchor-timing.dat", "anchor-words.dat", "anchor-status.dat"]:
        assert name in result
    assert "BULK_POINT,NLOOPS,T0,T1,RATE" in result
    assert "'ANCHORDONE ',SEQUENCE" in result


@pytest.mark.parametrize("transform", [workers.parity_driver, workers.timings_driver])
def test_new_commands_strict_integer_parser_and_sequence_bounds(base, transform):
    result = transform(base)
    assert "IF(PREVIOUS_SEQUENCE.GE.2000000000) GOTO 800" in result
    assert "SEQUENCE.NE.PREVIOUS_SEQUENCE+1) GOTO 800" in result
    assert "IF(DIGIT.LT.0.OR.DIGIT.GT.9) RETURN" in result
    assert "IF(VALUE.GT.(2000000000-DIGIT)/10) RETURN" in result
    assert "IF(NTOKENS.NE.NFIELDS+1) RETURN" in result
    assert "IF(I-FIRST.GT.64) RETURN" in result
    assert "READ(LINE" not in result


def test_timing_point_and_loop_bounds(base):
    result = workers.timings_driver(base)
    assert "IF(BULK_POINT.LT.1.OR.BULK_POINT.GT.NPOINTS) GOTO 800" in result
    assert "IF(NLOOPS.LT.1.OR.NLOOPS.GT.1000000) GOTO 800" in result
    assert "IF(T1.LT.T0.OR.RATE.LE.0) STOP 5" in result
