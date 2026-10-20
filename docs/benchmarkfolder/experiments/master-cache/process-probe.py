#!/usr/bin/env python3
"""Isolated qcd3 cache on/off diagnostic; never alters core generated outputs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/master-cache"
OUTPUT = WORK / "qcd3"
PREFIX = WORK / "install"
sys.path.insert(0, str(ROOT))
from madgraph.various.banner import MadLoopParam

DRIVER = """      PROGRAM CACHE_PROBE
      USE MNINJARS, ONLY: NINJARS_SET_CACHE_ENABLED,
     $ NINJARS_GET_CACHE_STATISTICS,NINJARS_FINALIZE
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER(NEXTERNAL=5)
      INCLUDE 'coupl.inc'
      REAL*8 POINTS(0:3,NEXTERNAL,100),P(0:3,NEXTERNAL)
      REAL*8 TOTAL(0:3),ELAPSED
      REAL*8,ALLOCATABLE :: ANS(:,:),ACC(:)
      INTEGER NDIM,NSO,N,I,J,K,B,MODE,CODE,STATUS,TENS
      INTEGER SELECTED(4),MODES(4)
      INTEGER*8 T0,T1,RATE,D0(8),D1(8),M0(8),M1(8)
      CHARACTER*1024 FILE
      DATA SELECTED/1,44,67,80/
      DATA MODES/0,1,1,0/
      CALL SETPARA('param_card.dat')
      CALL @PREFIX@GET_ANSWER_DIMENSION(NDIM)
      CALL @PREFIX@GET_NSQSO_LOOP(NSO)
      CALL GET_COMMAND_ARGUMENT(1,FILE)
      OPEN(71,FILE=TRIM(FILE),STATUS='OLD')
      READ(71,*) N
      IF(N.NE.100) STOP 11
      DO K=1,N
         DO I=1,NEXTERNAL
            READ(71,*) POINTS(:,I,K)
         ENDDO
      ENDDO
      CLOSE(71)
      ALLOCATE(ANS(0:3,0:NDIM),ACC(0:NSO))
      CALL NINJARS_SET_CACHE_ENABLED(.FALSE.,STATUS)
      IF(STATUS.NE.0) STOP 12
      CALL @PREFIX@FORCE_STABILITY_CHECK(.TRUE.)
      P=POINTS(:,:,1)
      TOTAL=P(:,1)+P(:,2)
      MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
      CALL UPDATE_AS_PARAM()
      DO I=1,1000
         CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
         TENS=MOD(CODE,100)/10
         IF(TENS.EQ.1.OR.TENS.EQ.2) EXIT
      ENDDO
      IF(I.GT.1000) STOP 13
C     Warm every selected reduction, including genuine MP, outside timing.
      DO I=1,4
         K=SELECTED(I)
         P=POINTS(:,:,K)
         TOTAL=P(:,1)+P(:,2)
         MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
         CALL UPDATE_AS_PARAM()
         CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
         IF(CODE/100.NE.2.AND.CODE/100.NE.3) STOP 14
      ENDDO
      OPEN(75,FILE='probe-values.dat',STATUS='REPLACE')
      OPEN(76,FILE='probe-cache.dat',STATUS='REPLACE')
      CALL SYSTEM_CLOCK(T0,RATE)
      DO B=1,4
         MODE=MODES(B)
         CALL NINJARS_SET_CACHE_ENABLED(MODE.EQ.1,STATUS)
         IF(STATUS.NE.0) STOP 15
         DO I=1,4
            K=SELECTED(I)
            P=POINTS(:,:,K)
            TOTAL=P(:,1)+P(:,2)
            MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
            CALL UPDATE_AS_PARAM()
            CALL NINJARS_GET_CACHE_STATISTICS(0,D0,STATUS)
            IF(STATUS.NE.0.OR.D0(8).NE.0) STOP 16
            CALL NINJARS_GET_CACHE_STATISTICS(1,M0,STATUS)
            IF(STATUS.NE.0.OR.M0(8).NE.0) STOP 17
            CALL SYSTEM_CLOCK(T0)
            CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
            CALL SYSTEM_CLOCK(T1)
            ELAPSED=DBLE(T1-T0)/DBLE(RATE)
            CALL NINJARS_GET_CACHE_STATISTICS(0,D1,STATUS)
            IF(STATUS.NE.0.OR.D1(8).NE.0) STOP 18
            CALL NINJARS_GET_CACHE_STATISTICS(1,M1,STATUS)
            IF(STATUS.NE.0.OR.M1(8).NE.0) STOP 19
            DO J=0,NSO
               WRITE(75,'(5I8,6ES26.17E3)') B,MODE,K,J,CODE,
     $              ANS(:,J),ACC(J),ELAPSED
            ENDDO
            WRITE(76,'(3I8,16I20)') B,MODE,K,D1-D0,M1-M0
            FLUSH(75)
            FLUSH(76)
            IF(CODE/100.NE.2.AND.CODE/100.NE.3) STOP 20
         ENDDO
      ENDDO
      CLOSE(75)
      CLOSE(76)
      CALL NINJARS_FINALIZE()
      END
"""

WRAPPER = """      SUBROUTINE @PREFIX@SLOOPMATRIX_THRES(P,ANS,PREC_ASKED,
     $ PREC_FOUND,RET_CODE)
      USE MNINJARS, ONLY: NINJARS_BEGIN_POINT,NINJARS_END_POINT
      IMPLICIT NONE
      INCLUDE 'nsqso_born.inc'
      INTEGER NEXTERNAL,NSQUAREDSO,ANS_DIMENSION
      PARAMETER(NEXTERNAL=5,NSQUAREDSO=1)
      PARAMETER(ANS_DIMENSION=MAX(NSQSO_BORN,NSQUAREDSO))
      REAL*8 P(0:3,NEXTERNAL),ANS(0:3,0:ANS_DIMENSION)
      REAL*8 PREC_ASKED,PREC_FOUND(0:NSQUAREDSO)
      INTEGER RET_CODE,STATUS
      CALL NINJARS_BEGIN_POINT(STATUS)
      IF(STATUS.NE.0) THEN
         ANS=0D0
         PREC_FOUND=-1D0
         RET_CODE=420
         RETURN
      ENDIF
      CALL @PREFIX@SLOOPMATRIX_THRES_PRIVATE(P,ANS,PREC_ASKED,
     $ PREC_FOUND,RET_CODE)
      CALL NINJARS_END_POINT(STATUS)
      IF(STATUS.NE.0) RET_CODE=420
      END

"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory():
    return next(OUTPUT.glob("SubProcesses/P*_*"))


def prepare():
    parent = ROOT / ".benchmark-work/outputs/candidate/qcd-3"
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUTPUT}")
    if not (PREFIX / "lib/mninjars.mod").exists():
        raise RuntimeError("Build/install the isolated ABI3 bridge first")
    shutil.copytree(parent, OUTPUT, symlinks=True)
    proc = directory()
    loop = proc / "loop_matrix.f"
    text = loop.read_text()
    match = re.search(r"SUBROUTINE (\w*)SLOOPMATRIX_THRES\(", text)
    prefix = match.group(1)
    text = text[:match.start()] + text[match.start():].replace(
        prefix+"SLOOPMATRIX_THRES(", prefix+"SLOOPMATRIX_THRES_PRIVATE(", 1)
    text += "\n" + WRAPPER.replace("@PREFIX@", prefix)
    clear = "      SUBROUTINE " + prefix + "CLEAR_CACHES()\n"
    assert text.count(clear) == 1
    text = text.replace(clear, clear +
        "      USE MNINJARS, ONLY: NINJARS_CLEAR_INTEGRAL_CACHE\n")
    needle = "      CALL " + prefix + "CLEAR_TIR_CACHE()\n"
    assert text.count(needle) == 1
    text = text.replace(needle, "      CALL NINJARS_CLEAR_INTEGRAL_CACHE()\n" + needle)
    loop.write_text(text)
    (proc / "check_sa.f").write_text(DRIVER.replace("@PREFIX@", prefix))
    definitions = OUTPUT / "SubProcesses/MadLoop_makefile_definitions"
    old = str(ROOT / "HEPTools/ninjars/lib")
    text = definitions.read_text()
    assert old in text
    definitions.write_text(text.replace(old, str(PREFIX / "lib")))
    for name in ("ninjars_lib", "ninjars_include"):
        link = OUTPUT / "lib" / name
        if link.is_symlink():
            link.unlink()
            link.symlink_to(PREFIX / "lib")
    # Local linker symlinks must refer to the copied output, not its parent.
    for link in proc.iterdir():
        if link.is_symlink():
            target = os.readlink(link)
            if target.startswith(str(parent)):
                link.unlink()
                link.symlink_to(str(OUTPUT) + target[len(str(parent)):])
    card = MadLoopParam(str(OUTPUT / "Cards/MadLoopParams.dat"))
    card["MLReductionLib"] = "8"
    card["CTModeRun"] = -1
    card["MLStabThres"] = 1e-9
    card["CTModeInit"] = 1
    card["UseLoopFilter"] = False
    for path in (OUTPUT / "Cards/MadLoopParams.dat",
                 *OUTPUT.glob("SubProcesses/**/MadLoopParams.dat")):
        if not path.resolve().is_relative_to(OUTPUT):
            raise RuntimeError(f"Refusing to write a card outside the isolated output: {path}")
        card.write(str(path), template=str(
            ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    with (WORK / "process-build.log").open("w") as log:
        env = os.environ.copy()
        env["MAKEFLAGS"] = "SHELL=" + shutil.which("bash")
        subprocess.run(["make", "-j2", "check"], cwd=proc, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    report = HERE / "qcd3"
    report.mkdir(exist_ok=True)
    for path in (proc / "check_sa.f", loop, OUTPUT / "Cards/MadLoopParams.dat",
                 OUTPUT / "Cards/param_card.dat", definitions):
        shutil.copy2(path, report / path.name)
    metadata = {"parent": str(parent.relative_to(ROOT)),
                "parent_compilation": json.loads((parent / "compilation.json").read_text()),
                "executable_sha256": digest(proc / "check"),
                "driver_sha256": digest(proc / "check_sa.f"),
                "loop_matrix_sha256": digest(loop),
                "prototype_library_sha256": digest(PREFIX / "lib/libninjars.a"),
                "parameter_card_sha256": digest(OUTPUT / "Cards/param_card.dat"),
                "madloop_card_sha256": digest(OUTPUT / "Cards/MadLoopParams.dat"),
                "note": "Isolated diagnostic; cache scope is each full amplitude; "
                        "cache entries must be zero before and after every call."}
    (report / "build.json").write_text(json.dumps(metadata, indent=2)+"\n")


def run(cpu):
    proc = directory()
    points = ROOT / "docs/benchmarkfolder/points/qcd-3.dat"
    command = ["taskset", "-c", str(cpu), "./check", str(points)]
    env = os.environ.copy()
    env.setdefault("SYMBOLICA_LOCK_DIR", str(ROOT / ".ninjars-build/symbolica-locks"))
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([
        str(ROOT / ".benchmark-work/original/HEPTools/ninja/lib"),
        str(ROOT / ".benchmark-work/original/HEPTools/collier"),
        env.get("LD_LIBRARY_PATH", "")])
    report = HERE / "qcd3"
    with (WORK / "process-run.log").open("w") as log:
        subprocess.run(command, cwd=proc, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    for name in ("probe-values.dat", "probe-cache.dat"):
        shutil.copy2(proc / name, report / name)
    shutil.copy2(points, report / "points.dat")
    (report / "run.json").write_text(json.dumps({
        "command": command, "cpu": cpu, "points_sha256": digest(points),
        "selected_point_indices": [1, 44, 67, 80], "cache_mode_order": [0, 1, 1, 0],
        "executable_sha256": digest(proc / "check"),
        "purpose": "diagnostic cache disabled/enabled parity and timing, not acceptance"
    }, indent=2)+"\n")
    rows = []
    reference = {}
    for line in (report / "probe-values.dat").read_text().splitlines():
        words = line.split()
        block, mode, point, order, status = map(int, words[:5])
        values = list(map(float, words[5:10]))
        seconds = float(words[10])
        if any(not math.isfinite(x) for x in values+[seconds]) or seconds <= 0:
            raise RuntimeError("Nonfinite value or invalid elapsed time")
        identity = (point, order)
        answer = (status, words[5:10])
        if identity in reference and answer != reference[identity]:
            raise RuntimeError(f"Cache changed printed coefficients/accuracy/status: {identity}")
        reference.setdefault(identity, answer)
        if status // 100 not in (2, 3):
            raise RuntimeError(f"Unsuccessful evaluation: {point}, {status}")
        if point != 1 and (status // 100 != 3 or status % 10 != 2):
            raise RuntimeError(f"Expected genuine NinjaRS rescue: {point}, {status}")
        rows.append(dict(block=block, enabled=bool(mode), point=point, order=order,
                         status=status, coefficients=values[:4], accuracy=values[4],
                         seconds=seconds))
    if len(rows) != 32:
        raise RuntimeError(f"Expected 4 blocks × 4 points × 2 split-order rows, got {len(rows)}")
    stats = []
    for line in (report / "probe-cache.dat").read_text().splitlines():
        values = list(map(int, line.split()))
        block, mode, point = values[:3]
        dp, mp = values[3:11], values[11:]
        if len(mp) != 8 or dp[7] or mp[7] or min(dp+mp) < 0:
            raise RuntimeError("Invalid counters or cache entries survived the full evaluation")
        stats.append(dict(block=block, enabled=bool(mode), point=point, dp=dp, mp=mp))
    if len(stats) != 16 or not any(r["enabled"] and r["mp"][1] for r in stats):
        raise RuntimeError("Expected genuine MP cache hits in the enabled blocks")
    summary = {"printed_results_identical": True, "scope_empty_after_each_call": True,
               "rows": rows, "cache_stats": stats,
               "timing_note": "Diagnostic four-block comparison; no acceptance confidence interval."}
    (report / "results.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.cpu)
