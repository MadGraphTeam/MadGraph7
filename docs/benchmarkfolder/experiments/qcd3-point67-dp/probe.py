#!/usr/bin/env python3
"""Compare original Ninja and NinjaRS on identical point67 DP reduction inputs."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/qcd3-point67-dp"
OUTPUT = WORK / "qcd3"
sys.path.insert(0, str(ROOT))
from madgraph.various.banner import MadLoopParam

DRIVER = """      PROGRAM PAIRED_DP
      USE MNINJARS, ONLY: NINJARS_FINALIZE
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER(NEXTERNAL=5)
      INCLUDE 'coupl.inc'
      REAL*8 POINTS(0:3,NEXTERNAL,100),P(0:3,NEXTERNAL)
      REAL*8 TOTAL(0:3)
      REAL*8,ALLOCATABLE :: ANS(:,:),ACC(:)
      INTEGER NDIM,NSO,N,I,K,CODE,TENS
      LOGICAL TRACE_DP
      COMMON/PAIRED_DP_TRACE/TRACE_DP
      CHARACTER*1024 FILE
      TRACE_DP=.FALSE.
      CALL SETPARA('param_card.dat')
      CALL ML5_0_GET_ANSWER_DIMENSION(NDIM)
      CALL ML5_0_GET_NSQSO_LOOP(NSO)
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
      CALL ML5_0_FORCE_STABILITY_CHECK(.FALSE.)
      P=POINTS(:,:,1)
      TOTAL=P(:,1)+P(:,2)
      MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
      CALL UPDATE_AS_PARAM()
      DO I=1,1000
         CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
         TENS=MOD(CODE,100)/10
         IF(TENS.EQ.1.OR.TENS.EQ.2) EXIT
      ENDDO
      IF(I.GT.1000) STOP 13
      OPEN(77,FILE='paired-reductions.dat',STATUS='REPLACE')
      TRACE_DP=.TRUE.
      P=POINTS(:,:,67)
      TOTAL=P(:,1)+P(:,2)
      MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
      CALL UPDATE_AS_PARAM()
      CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
      CLOSE(77)
      OPEN(78,FILE='point-result.dat',STATUS='REPLACE')
      DO I=0,NSO
         WRITE(78,'(2I8,5ES26.17E3)') I,CODE,ANS(:,I),ACC(I)
      ENDDO
      CLOSE(78)
      CALL NINJARS_FINALIZE()
      END
"""

TRACE = """      IF (TRACE_DP) THEN
        IF (.NOT.PAIRED_INIT) THEN
          CALL ML5_0_INITCT()
          CALL ML5_0_INITNINJA()
          PAIRED_INIT=.TRUE.
        ENDIF
        PAIRED_COEFF=TENSORCOEFS
        PAIRED_RES=(991D0,991D0)
        PAIRED_R1=(991D0,991D0)
        CALL NINJA_TENSOR_EVALUATE(
     $   PAIRED_COEFF(0:CURR_MAXCOEF-1),NLOOPLINE,RANK,
     $   REAL_S_MAT,P_NRS,MASSES,NRS_SCALE,
     $   PAIRED_RES,PAIRED_R1,PAIRED_STATUS)
        WRITE(77,'(6I8,ES26.17E3)') ID,SQSOINDEX,CTMODE,
     $    NLOOPLINE,RANK,CURR_MAXCOEF,NRS_SCALE
        WRITE(77,'(2I8)') NRS_STATUS,PAIRED_STATUS
        WRITE(77,'(2ES26.17E3)') TENSORCOEFS(0:CURR_MAXCOEF-1)
        WRITE(77,'(ES26.17E3)') REAL_S_MAT
        WRITE(77,'(ES26.17E3)') P_NRS
        WRITE(77,'(2ES26.17E3)') MASSES
        WRITE(77,'(2ES26.17E3)') NRS_RES,PAIRED_RES
      ENDIF
"""

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def directory():
    return next(OUTPUT.glob("SubProcesses/P*_*"))

def prepare():
    parent = ROOT / ".benchmark-work/outputs/candidate/qcd-3"
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUTPUT}")
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copytree(parent, OUTPUT, symlinks=True)
    proc = directory()
    for link in proc.iterdir():
        if link.is_symlink():
            target = os.readlink(link)
            if target.startswith(str(parent)):
                link.unlink()
                link.symlink_to(str(OUTPUT) + target[len(str(parent)):])
    interface = proc / "CT_interface.f"
    text = interface.read_text()
    start = text.index("      SUBROUTINE ML5_0_NINJARS_LOOP(")
    end = text.index("      SUBROUTINE ML5_0_MP_NINJARS_LOOP(", start)
    block = text[start:end]
    block = block.replace("      USE MNINJARS\n", "      USE MNINJARS\n"
                          "      USE MNINJA, ONLY: NINJA_TENSOR_EVALUATE\n", 1)
    block = block.replace("      STABLE=.FALSE.\n", """      LOGICAL TRACE_DP,PAIRED_INIT
      COMMON/PAIRED_DP_TRACE/TRACE_DP
      SAVE PAIRED_INIT
      DATA PAIRED_INIT/.FALSE./
      INTEGER PAIRED_STATUS
      COMPLEX*16 PAIRED_COEFF(0:LOOPMAXCOEFS-1)
      COMPLEX*16 PAIRED_RES(0:2),PAIRED_R1

      NRS_RES=(991D0,991D0)
      R1=(991D0,991D0)
      STABLE=.FALSE.
""", 1)
    block = block.replace("      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS\n",
                          TRACE + "      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS\n")
    interface.write_text(text[:start] + block + text[end:])
    (proc / "check_sa.f").write_text(DRIVER)
    env = os.environ.copy()
    env["MAKEFLAGS"] = "SHELL=" + shutil.which("bash")
    with (WORK / "build.log").open("w") as log:
        subprocess.run(["make", "-j2", "check"], cwd=proc, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    for path in (interface, proc / "check_sa.f", OUTPUT / "Cards/param_card.dat"):
        shutil.copy2(path, HERE / path.name)
    (HERE / "build.json").write_text(json.dumps({
        "parent": str(parent.relative_to(ROOT)),
        "parent_compilation": json.loads((parent / "compilation.json").read_text()),
        "executable_sha256": digest(proc / "check"),
        "driver_sha256": digest(proc / "check_sa.f"),
        "interface_sha256": digest(interface),
        "parameter_card_sha256": digest(OUTPUT / "Cards/param_card.dat"),
        "purpose": "DP-only paired same-input diagnosis, no threshold change",
    }, indent=2) + "\n")

def parse_trace(path):
    rows = []
    with path.open() as stream:
        while header := stream.readline():
            words = header.split()
            loop, order, direction, n, rank, count = map(int, words[:6])
            status = list(map(int, next(stream).split()))
            pairs = lambda count: [list(map(float, next(stream).split())) for _ in range(count)]
            coefficients = pairs(count)
            s = [float(next(stream)) for _ in range(n*n)]
            momenta = [float(next(stream)) for _ in range(4*n)]
            masses = pairs(n)
            values = pairs(6)
            rows.append(dict(sequence=len(rows), loop=loop, split_order=order,
                direction=direction, n=n, rank=rank, scale_squared=float(words[6]),
                status_ninjars=status[0], status_ninja=status[1],
                coefficients=coefficients, s_matrix_column_major=s,
                momenta_column_major=momenta, masses_squared=masses,
                ninjars=values[:3], ninja=values[3:]))
    return rows

def run(cpu):
    proc = directory()
    points = ROOT / "docs/benchmarkfolder/points/qcd-3.dat"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([
        str(ROOT / ".benchmark-work/original/HEPTools/ninja/lib"),
        str(ROOT / ".benchmark-work/original/HEPTools/collier"),
        env.get("LD_LIBRARY_PATH", "")])
    summaries = []
    for direction in (1, 2):
        card = MadLoopParam(str(OUTPUT / "Cards/MadLoopParams.dat"))
        for key, value in dict(MLReductionLib="8", CTModeRun=direction,
                              MLStabThres=1e-9, CTModeInit=1, UseLoopFilter=False).items():
            card[key] = value
        for path in (OUTPUT / "Cards/MadLoopParams.dat", *OUTPUT.glob("SubProcesses/**/MadLoopParams.dat")):
            if not path.resolve().is_relative_to(OUTPUT):
                raise RuntimeError(f"Refusing external card write: {path}")
            card.write(str(path), template=str(ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
        command = ["taskset", "-c", str(cpu), "./check", str(points)]
        with (WORK / f"direction-{direction}.log").open("w") as log:
            subprocess.run(command, cwd=proc, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = parse_trace(proc / "paired-reductions.dat")
        for row in rows:
            row["finite_real_difference"] = row["ninjars"][0][0] - row["ninja"][0][0]
        report = HERE / f"direction-{direction}"
        report.mkdir(exist_ok=True)
        with gzip.open(report / "reductions.json.gz", "wt") as out:
            json.dump(rows, out)
        for name in ("point-result.dat", "MadLoopParams.dat"):
            shutil.copy2(proc / name if name != "MadLoopParams.dat" else OUTPUT / "Cards" / name, report / name)
        failed = [r for r in rows if r["status_ninjars"] != 0 or r["status_ninja"] != 0]
        summary = dict(direction=direction, command=command, reductions=len(rows),
            failed=failed, largest_differences=sorted(rows,
            key=lambda row: abs(row["finite_real_difference"]), reverse=True)[:12])
        (report / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        summaries.append(dict(direction=direction, reductions=len(rows),
            failed_ninjars=sum(r["status_ninjars"] != 0 for r in rows),
            failed_ninja=sum(r["status_ninja"] != 0 for r in rows),
            largest_loop=summary["largest_differences"][0]["loop"],
            largest_absolute_finite_difference=abs(summary["largest_differences"][0]["finite_real_difference"])))
    shutil.copy2(points, HERE / "points.dat")
    (HERE / "summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    print(json.dumps(summaries, indent=2))

def normal(cpu):
    """Stop after the normal DP decision, before any possible MP call."""
    proc = directory()
    loop = proc / "loop_matrix.f"
    text = loop.read_text()
    text = text.replace("      IMPLICIT NONE\n", """      IMPLICIT NONE
      LOGICAL TRACE_DP
      COMMON/PAIRED_DP_TRACE/TRACE_DP
""", 1)
    needle = "          IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN\n"
    position = text.index(needle, text.index("        ELSEIF(.NOT.DOING_QP_EVALS)THEN"))
    text = text[:position] + text[position:].replace(needle, """          IF(TRACE_DP) THEN
            OPEN(79,FILE='normal-dp-decision.dat',STATUS='REPLACE')
            WRITE(79,*) FPE_IN_DP_REDUCTION,N_DP_EVAL,MLSTABTHRES
            WRITE(79,'(ES26.17E3)') ACC
            DO J=0,NSQUAREDSO
              DO I=1,N_DP_EVAL
                WRITE(79,'(2I8,3ES26.17E3)') J,I,DP_RES(:,J,I)
              ENDDO
            ENDDO
            CLOSE(79)
            CLOSE(77)
            STOP
          ENDIF
""" + needle, 1)
    loop.write_text(text)
    env = os.environ.copy()
    env["MAKEFLAGS"] = "SHELL=" + shutil.which("bash")
    with (WORK / "normal-build.log").open("w") as log:
        subprocess.run(["make", "-j2", "check"], cwd=proc, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    card = MadLoopParam(str(OUTPUT / "Cards/MadLoopParams.dat"))
    card["CTModeRun"] = -1
    for path in (OUTPUT / "Cards/MadLoopParams.dat", *OUTPUT.glob("SubProcesses/**/MadLoopParams.dat")):
        if not path.resolve().is_relative_to(OUTPUT):
            raise RuntimeError(f"Refusing external card write: {path}")
        card.write(str(path), template=str(ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([
        str(ROOT / ".benchmark-work/original/HEPTools/ninja/lib"),
        str(ROOT / ".benchmark-work/original/HEPTools/collier"),
        env.get("LD_LIBRARY_PATH", "")])
    points = ROOT / "docs/benchmarkfolder/points/qcd-3.dat"
    command = ["taskset", "-c", str(cpu), "./check", str(points)]
    report = HERE / "normal"
    report.mkdir()
    (report / "build.json").write_text(json.dumps({
        "executable_sha256": digest(proc / "check"),
        "loop_matrix_sha256": digest(loop), "command": command,
        "purpose": "Terminate after normal DP stability decision and before MP",
    }, indent=2)+"\n")
    shutil.copy2(loop, report / loop.name)
    with (WORK / "normal-run.log").open("w") as log:
        subprocess.run(command, cwd=proc, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(proc / "normal-dp-decision.dat", report / "normal-dp-decision.dat")
    shutil.copy2(OUTPUT / "Cards/MadLoopParams.dat", report / "MadLoopParams.dat")
    rows = parse_trace(proc / "paired-reductions.dat")
    with gzip.open(report / "reductions.json.gz", "wt") as stream:
        json.dump(rows, stream)
    print((report / "normal-dp-decision.dat").read_text())
    print("retained normal reductions:", len(rows))

def reference(cpu):
    """Use the already built normal-decision driver with original Ninja."""
    proc = directory()
    before = json.loads((HERE / "normal/build.json").read_text())
    assert digest(proc / "check") == before["executable_sha256"]
    card = MadLoopParam(str(OUTPUT / "Cards/MadLoopParams.dat"))
    card["MLReductionLib"] = "6"
    card["CTModeRun"] = -1
    for path in (OUTPUT / "Cards/MadLoopParams.dat", *OUTPUT.glob("SubProcesses/**/MadLoopParams.dat")):
        if not path.resolve().is_relative_to(OUTPUT):
            raise RuntimeError(f"Refusing external card write: {path}")
        card.write(str(path), template=str(ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([
        str(ROOT / ".benchmark-work/original/HEPTools/ninja/lib"),
        str(ROOT / ".benchmark-work/original/HEPTools/collier"),
        env.get("LD_LIBRARY_PATH", "")])
    report = HERE / "normal-ninja"
    report.mkdir()
    with (WORK / "normal-ninja.log").open("w") as log:
        subprocess.run(before["command"], cwd=proc, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(proc / "normal-dp-decision.dat", report / "normal-dp-decision.dat")
    shutil.copy2(OUTPUT / "Cards/MadLoopParams.dat", report / "MadLoopParams.dat")
    (report / "run.json").write_text(json.dumps(before, indent=2)+"\n")
    print((report / "normal-dp-decision.dat").read_text())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "normal", "reference"))
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run(args.cpu)
    elif args.action == "normal":
        normal(args.cpu)
    else:
        reference(args.cpu)
