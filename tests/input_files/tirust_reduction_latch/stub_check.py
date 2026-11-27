#!/usr/bin/env python3
"""Bounded source-bound software-latch STUB check; no physical/reducer qualification.

Only --execute compiles/runs. Four compiler commands and 32 fresh stub processes
are the absolute maximum, without retries. All generated files live in /tmp.
The actual public wrapper, independent-call initialization and automatic-rescue
control block are extracted from the two adjacent frozen templates. Physics,
accuracy estimates, rotations and reduction calls are explicitly synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
INSERT = "FPE_IN_DP_REDUCTION=.FALSE.\nFPE_IN_QP_REDUCTION=.FALSE.\n"
INIT = "MP_DONE=.FALSE.\nMP_DONE_ONCE=.FALSE.\nMP_PS_SET=.FALSE.\nSTAB_INDEX=0\nDOING_QP_EVALS=.FALSE.\n"
AUTO = "IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.EQ.-1)) THEN"
SCOPE = "Compiled source-bound software-latch stubs only; no complete host, physical amplitude, reducer, accuracy or performance qualification."


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def templates(directory=HERE):
    before = (directory / "template.before.inc").read_text()
    candidate = (directory / "template.candidate.inc").read_text()
    require(before.count(INIT) == candidate.count(INIT) == 1, "Independent-call initialization anchor differs")
    require(candidate == before.replace(INIT, INSERT + INIT, 1), "Proposal is not exactly the two entry assignments")
    return before, candidate


def fragments(source):
    begin = source.index("## if(ninjars_available) {")
    end = source.index("C  \n%(info_lines)s", begin)
    wrapper = source[begin:end]
    start = source.index(INIT)
    if source[max(0, start-len(INSERT)):start] == INSERT:
        start -= len(INSERT)
    end = source.index("\n## if(LoopInduced)", start)
    init = source[start:end]
    start = source.index("      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION")
    end = source.index("\n\n", start)
    common = source[start:end]
    start = source.index(AUTO)
    end = source.index("C Finally for the summed result in ANS", start)
    policy = source[start:end]
    require(policy.count("FPE_IN_DP_REDUCTION=.False.") == 3, "Production transition resets differ")
    require(policy.count("FPE_IN_QP_REDUCTION=.False.") == 3, "Production transition resets differ")
    require("IF(FPE_IN_DP_REDUCTION) THEN" in policy and "IF(FPE_IN_QP_REDUCTION) THEN" in policy,
            "Actual DP/QP accuracy masks are missing")
    return {"public_wrapper": wrapper, "call_initialization": init,
            "latch_common": common, "automatic_policy": policy}


def conditional(source, available):
    """Render only the actual NinjaRS availability conditional, fail closed."""
    result, stack, active = [], [], True
    for line in source.splitlines():
        if line.startswith("## if(ninjars_available)"):
            stack.append((active, available))
            active = active and available
        elif line == "## } else {":
            require(bool(stack), "Unmatched template else")
            parent, selected = stack[-1]
            active = parent and not selected
        elif line == "## }":
            require(bool(stack), "Unmatched template end")
            active = stack.pop()[0]
        elif line.startswith("##"):
            raise ValueError("Unexpected conditional in extracted source: " + line)
        elif active:
            result.append(line)
    require(not stack, "Unclosed template conditional")
    return "\n".join(result) + "\n"


def fixed(source):
    """Ordinary fixed-form indentation; retain executable text and labels."""
    lines = []
    for raw in source.expandtabs(8).splitlines():
        if (not raw or raw[0] in "Cc*!" or raw.startswith("      ")
                or raw.startswith(("     $", "     &"))):
            lines.append(raw)
        elif re.match(r"^\s*\d+\s", raw):
            match = re.match(r"^\s*(\d+)\s+(.*)$", raw)
            lines.append(match[1].rjust(5) + " " + match[2])
        else:
            lines.append("      " + raw)
    return "\n".join(lines) + "\n"


DECLARATIONS = """
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
      INTEGER NEXTERNAL,NSQUAREDSO,NLOOPLIB,QP_NLOOPLIB
      PARAMETER(NEXTERNAL=2,NSQUAREDSO=1,NLOOPLIB=9,QP_NLOOPLIB=1)
      REAL*8 P_USER(0:3,NEXTERNAL),ANS(0:3,0:NSQUAREDSO)
      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED
      INTEGER NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL
      INTEGER NRS_RETRY_HARD_SKIP
      REAL*8 NRS_NATIVE_ACC,NRS_COMP_ACC
      COMMON/P_NRS_RETRY_FLAGS/NRS_RETRY_MODE,NRS_RETRY_USED
      COMMON/P_NRS_RETRY_STATS/NRS_RETRY_COUNT,NRS_RETRY_PASS,
     $ NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP
      COMMON/P_NRS_RETRY_ACC/NRS_NATIVE_ACC,NRS_COMP_ACC
      LOGICAL MP_DONE,MP_DONE_ONCE,MP_PS_SET,DOING_QP_EVALS
      LOGICAL EVAL_DONE(4),CHECKPHASE,HELDOUBLECHECKED
      LOGICAL LOOPLIBS_QPAVAILABLE(0:9),LOOPLIBS_DIRECTEST(9)
      LOGICAL QP_TOOLS_AVAILABLE
      INTEGER STAB_INDEX,I,J,K,I_LIB,I_QP_LIB,BASIC_CT_MODE
      INTEGER MAXSTABILITYLENGTH,N_DP_EVAL,N_QP_EVAL,NEPS
      PARAMETER(MAXSTABILITYLENGTH=4)
      INTEGER INDEX_QP_TOOLS(2),MLReductionLib(10),CTMODE,CTMODERUN
      INTEGER NRotations_DP,NRotations_QP,RET_CODE_H,RET_CODE_T
      INTEGER RET_CODE_U,SET_RET_CODE_U,GROUP
      REAL*8 QP_RES(3,0:1,4),DP_RES(3,0:1,4),ACC(0:1)
      REAL*8 ACCURACY(0:1),TEMP1(0:1),TEMP(0:3,0:1),MLSTABTHRES
      REAL*8 PS(0:3,2),P(0:3,2),MU_R
      REAL*16 MP_PS(0:3,2),MP_P(0:3,2)
      INTEGER CASE_ID,SEQUENCE,FAIL_KIND,REQUEST_MODE,HAS_QP
      COMMON/STUB_REQUEST/CASE_ID,SEQUENCE,FAIL_KIND,REQUEST_MODE,HAS_QP
      INTEGER OUT_CODE,ENTRY_DP,ENTRY_QP,EVALUATIONS,N200,N300
      REAL*8 OUT_ACC
      COMMON/STUB_RESULT/OUT_CODE,ENTRY_DP,ENTRY_QP,EVALUATIONS,
     $ N200,N300,OUT_ACC
"""

BODY = """
C     Synthetic state; only the extracted initialization/policy is production.
      ENTRY_DP=MERGE(1,0,FPE_IN_DP_REDUCTION)
      ENTRY_QP=MERGE(1,0,FPE_IN_QP_REDUCTION)
      EVALUATIONS=0
      N200=0
      N300=0
      ANS=0D0
      DP_RES=0D0
      QP_RES=0D0
      ACCURACY=-1D0
      ACC=0D0
      PS=0D0
      P=0D0
      MP_PS=0
      MP_P=0
      MU_R=1D0
      CHECKPHASE=.FALSE.
      HELDOUBLECHECKED=.TRUE.
      CTMODERUN=REQUEST_MODE
      CTMODE=MAX(1,REQUEST_MODE)
      N_DP_EVAL=3
      N_QP_EVAL=3
      NRotations_DP=1
      NRotations_QP=1
      MLSTABTHRES=1D-6
      MLReductionLib=0
      MLReductionLib(1)=9
      LOOPLIBS_QPAVAILABLE=.FALSE.
      LOOPLIBS_QPAVAILABLE(9)=HAS_QP.EQ.1
      LOOPLIBS_DIRECTEST=.TRUE.
      QP_TOOLS_AVAILABLE=HAS_QP.EQ.1
      INDEX_QP_TOOLS=0
      IF(QP_TOOLS_AVAILABLE) INDEX_QP_TOOLS(1)=1
      I_LIB=1
      I_QP_LIB=1
      NEPS=11
      NRS_RETRY_MODE=.FALSE.
      NRS_RETRY_USED=.FALSE.
      NRS_RETRY_COUNT=0
      NRS_RETRY_PASS=0
      NRS_RETRY_FAIL=0
      NRS_RETRY_HARD_SKIP=0
      GOTO 208
  200 CONTINUE
      N200=N200+1
      GOTO 208
  300 CONTINUE
      N300=N300+1
  208 CONTINUE
      EVALUATIONS=EVALUATIONS+1
      IF(EVALUATIONS.GT.8) STOP 70
      IF(CTMODE.GE.4) MP_DONE_ONCE=.TRUE.
      DO GROUP=1,2
C       The simulated reducer only sets flags; successful groups never clear.
        IF(GROUP.EQ.1.AND.FAIL_KIND.NE.0.AND.
     $     (CASE_ID.NE.5.OR.EVALUATIONS.EQ.1)) THEN
          IF(CTMODE.LT.4) THEN
            FPE_IN_DP_REDUCTION=.TRUE.
          ELSEIF(FAIL_KIND.EQ.2) THEN
            FPE_IN_QP_REDUCTION=.TRUE.
          ENDIF
        ENDIF
        WRITE(*,'(A,6I8)') 'LATCH_GROUP ',SEQUENCE,EVALUATIONS,
     $       GROUP,CTMODE,MERGE(1,0,FPE_IN_DP_REDUCTION),
     $       MERGE(1,0,FPE_IN_QP_REDUCTION)
      ENDDO
      DO I=0,1
        ANS(:,I)=(/7D0,11D0,13D0,17D0/)
      ENDDO
C     Case 5 exercises internal jumps without re-entering the public routine.
      IF(CASE_ID.EQ.5.AND.EVALUATIONS.EQ.1) THEN
        CTMODE=2
        GOTO 300
      ENDIF
      IF(CASE_ID.EQ.5.AND.EVALUATIONS.EQ.2) THEN
        CTMODE=1
        GOTO 200
      ENDIF
@POLICY@
      OUT_CODE=100*RET_CODE_H+10*RET_CODE_T+RET_CODE_U
      OUT_ACC=ACCURACY(0)
      END
      SUBROUTINE P_COMPUTE_ACCURACY(VALUES,LENGTH,ACC,ANS)
      IMPLICIT NONE
      INTEGER LENGTH,I
      REAL*8 VALUES(3,0:1,LENGTH),ACC(0:1),ANS(0:3,0:1)
C     Identical synthetic estimates; only production FPE masks force rescue.
      ACC=0D0
      DO I=0,1
        ANS(1:3,I)=VALUES(:,I,1)
      ENDDO
      END
      SUBROUTINE P_ROTATE_PS(PS,P,NUMBER)
      INTEGER NUMBER
      REAL*8 PS(0:3,2),P(0:3,2)
      P=PS
      END
      SUBROUTINE P_MP_ROTATE_PS(PS,P,NUMBER)
      INTEGER NUMBER
      REAL*16 PS(0:3,2),P(0:3,2)
      P=PS
      END
      INTEGER FUNCTION SET_RET_CODE_U(LIB,QP,STABLE)
      INTEGER LIB
      LOGICAL QP,STABLE
C     Synthetic return-code callback implementing backend 9's documented IDs.
      SET_RET_CODE_U=0
      IF(STABLE.AND..NOT.QP) SET_RET_CODE_U=9
      IF(STABLE.AND.QP) SET_RET_CODE_U=3
      END
"""

MAIN = """
      PROGRAM LATCH_STUB
      IMPLICIT NONE
      INTEGER CASE_ID,SEQUENCE,FAIL_KIND,REQUEST_MODE,HAS_QP
      COMMON/STUB_REQUEST/CASE_ID,SEQUENCE,FAIL_KIND,REQUEST_MODE,HAS_QP
      INTEGER OUT_CODE,ENTRY_DP,ENTRY_QP,EVALUATIONS,N200,N300
      REAL*8 OUT_ACC,P(0:3,2),ANS(0:3,0:1)
      COMMON/STUB_RESULT/OUT_CODE,ENTRY_DP,ENTRY_QP,EVALUATIONS,
     $ N200,N300,OUT_ACC
      LOGICAL DP,QP
      COMMON/P_FPE_IN_REDUCTION/DP,QP
      INTEGER CALLS,MODES(5)
      CHARACTER*8 ARG
      CALL GET_COMMAND_ARGUMENT(1,ARG)
      READ(ARG,*) CASE_ID
      IF(CASE_ID.LT.1.OR.CASE_ID.GT.8) STOP 71
      MODES=(/1,2,4,5,-1/)
      CALLS=2
      IF(CASE_ID.EQ.5) CALLS=1
      IF(CASE_ID.EQ.6) CALLS=3
      IF(CASE_ID.EQ.8) CALLS=5
      P=0D0
      DO SEQUENCE=1,CALLS
        HAS_QP=1
        REQUEST_MODE=1
        FAIL_KIND=0
        IF(CASE_ID.LE.4) REQUEST_MODE=MODES(CASE_ID)
        IF(CASE_ID.LE.5.AND.SEQUENCE.EQ.1) FAIL_KIND=2
        IF(CASE_ID.EQ.6) THEN
          REQUEST_MODE=-1
          IF(SEQUENCE.EQ.1) FAIL_KIND=1
          IF(SEQUENCE.EQ.2) FAIL_KIND=2
        ENDIF
        IF(CASE_ID.EQ.7) THEN
          REQUEST_MODE=-1
          HAS_QP=0
          IF(SEQUENCE.EQ.1) FAIL_KIND=1
        ENDIF
        IF(CASE_ID.EQ.8) REQUEST_MODE=MODES(SEQUENCE)
        CALL P_SLOOPMATRIX(P,ANS)
        WRITE(*,'(A,9I8,5ES24.15E3)') 'LATCH_ROW ',SEQUENCE,
     $    OUT_CODE,MERGE(1,0,DP),MERGE(1,0,QP),ENTRY_DP,ENTRY_QP,
     $    EVALUATIONS,N200,N300,ANS(:,0),OUT_ACC
      ENDDO
      END
"""


def render(source, available):
    parts = fragments(source)
    substitutions = {"proc_prefix": "P_", "nexternal": 2, "real_dp_format": "REAL*8"}
    wrapper = conditional(parts["public_wrapper"], available) % substitutions
    common = parts["latch_common"] % substitutions
    policy = conditional(parts["automatic_policy"], available) % substitutions
    return (MAIN + fixed(wrapper) + DECLARATIONS + fixed(common)
            + fixed(parts["call_initialization"]) + BODY.replace("@POLICY@", fixed(policy)))


def parse(stdout):
    rows, groups = [], []
    for line in stdout.splitlines():
        fields = line.split()
        if fields[:1] == ["LATCH_ROW"]:
            require(len(fields) == 15, "Malformed stub row")
            rows.append({"integers": list(map(int, fields[1:10])),
                         "values": list(map(float, fields[10:]))})
        elif fields[:1] == ["LATCH_GROUP"]:
            require(len(fields) == 7, "Malformed stub group")
            groups.append(list(map(int, fields[1:])))
    require(bool(rows) and bool(groups), "Stub emitted no observations")
    require([r["integers"][0] for r in rows] == list(range(1, len(rows)+1)), "Missing stub calls")
    return {"rows": rows, "groups": groups}


def adjudicate(observed):
    for available in (False, True):
        def case(version, number):
            return observed[(version, available, number)]
        for number in (1, 2, 3, 4):
            original, candidate = case("before", number), case("candidate", number)
            require(original["rows"][0] == candidate["rows"][0], "First failure changed")
            slot = 2 if number <= 2 else 3
            require(candidate["rows"][0]["integers"][slot] == 1, "First failure was hidden")
            require(original["rows"][1]["integers"][slot] == 1, "Original sticky defect was not reproduced")
            require(candidate["rows"][1]["integers"][2:6] == [0,0,0,0], "Candidate independent-call flags did not clear")
            require(candidate["rows"][1]["values"] == original["rows"][1]["values"], "Healthy synthetic answer changed")
        for version in ("before", "candidate"):
            within = case(version, 5)
            require(within["rows"][0]["integers"][2] == 1, "Within-call failure lost")
            require(within["rows"][0]["integers"][6:] == [3,1,1], "Direction/rotation jump coverage missing")
            require(len(within["groups"]) == 6 and all(g[4:] == [1,0] for g in within["groups"]), "Group/jump lost latch")
            automatic = case(version, 6)
            require([r["integers"][1] for r in automatic["rows"]] == [323,420,219], "Automatic rescue code sequence changed")
            require(automatic["rows"][0]["integers"][2:4] == [0,0], "DP-to-QP transition did not clear flags")
            require(automatic["rows"][1]["integers"][3] == 1, "Terminal QP failure hidden")
            require([g[3] for g in automatic["groups"] if g[0] == 1 and g[2] == 1] == [1,2,1,4,5,4], "Automatic direction/rotation/QP modes differ")
        require(case("candidate", 6)["rows"][2]["integers"][2:6] == [0,0,0,0], "Healthy call after terminal QP failure contaminated")
        require(case("before", 6)["rows"][2]["integers"][3] == 1, "Original QP sticky defect missing")
        require([r["integers"][1] for r in case("candidate", 7)["rows"]] == [410,219], "No-QP recovery failed")
        require([r["integers"][1] for r in case("before", 7)["rows"]] == [410,410], "Original no-QP contamination missing")
        require(case("candidate", 7)["rows"][1]["integers"][2:6] == [0,0,0,0], "No-QP healthy flags remain set")
        require(case("before", 8) == case("candidate", 8), "Healthy parity changed")
        require([r["integers"][1] for r in case("candidate", 8)["rows"]] == [110,110,120,120,219], "Healthy forced/automatic modes differ")
    for version in ("before", "candidate"):
        for number in range(1,9):
            require(observed[(version,False,number)] == observed[(version,True,number)], "NinjaRS availability changes stub outcomes")
    return {"passed": True, "fresh_process_cases": 32,
            "claims": ["original sticky defect reproduced", "first-call failure retained", "healthy parity retained",
                       "DP/QP independent-call reset", "within-call groups and jumps retain failure",
                       "automatic DP-to-QP rescue and terminal failures retained"], "scope": SCOPE}


def clean_environment(tmp):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LD_", "DYLD_"))}
    env.update(TMPDIR=str(tmp), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               LANG="C", LC_ALL="C", PYTHONDONTWRITEBYTECODE="1")
    return env


def execute():
    start = time.monotonic()
    compiler = shutil.which("gfortran")
    require(compiler is not None, "GNU Fortran is required; no fallback compiler")
    before, candidate = templates()
    receipt = {"schema": "madloop-reduction-latch-stub-v1", "scope": SCOPE,
               "compiler": {"invoked": compiler, "resolved": str(Path(compiler).resolve()),
                            "sha256": sha(Path(compiler).resolve().read_bytes())},
               "template_sha256": {"before": sha(before.encode()), "candidate": sha(candidate.encode())},
               "stub_check_sha256": sha(Path(__file__).read_bytes()),
               "fragment_sha256": {version: {name: sha(raw.encode()) for name,raw in fragments(source).items()}
                                   for version,source in (("before",before),("candidate",candidate))},
               "caps": {"compiler_commands": 4, "compiler_seconds_each": 30,
                        "stub_executions": 32, "stub_seconds_each": 5, "total_seconds": 300},
               "commands": [], "observations": []}
    observed = {}
    def command(argv, directory, limit, kind):
        remaining = 300-(time.monotonic()-start)
        require(remaining > 0, "Inclusive stub deadline exceeded")
        tick = time.monotonic()
        result = subprocess.run(argv, cwd=directory, env=clean_environment(directory),
                                capture_output=True, text=True, timeout=min(limit,remaining), check=False)
        record = {"kind": kind, "argv": list(map(str,argv)), "returncode": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr, "seconds": time.monotonic()-tick}
        receipt["commands"].append(record)
        if result.returncode != 0:
            error = ValueError("Stub command failed: " + json.dumps(record))
            error.receipt = receipt
            raise error
        return result.stdout
    with tempfile.TemporaryDirectory(prefix="reduction-latch-stub-", dir="/tmp") as temporary:
        work = Path(temporary)
        for version, source in (("before",before),("candidate",candidate)):
            for available in (False,True):
                directory = work / (version + ("-nrs" if available else "-plain"))
                directory.mkdir()
                rendered = render(source,available)
                (directory / "stub.f").write_text(rendered)
                command([compiler,"-O0","-fcheck=all","-ffixed-line-length-none",
                         "-fno-fast-math","stub.f","-o","check"],directory,30,"compiler")
                for number in range(1,9):
                    stdout = command([str(directory / "check"),str(number)],directory,5,"stub")
                    value = parse(stdout)
                    observed[(version,available,number)] = value
                    receipt["observations"].append({"version":version,"ninjars_available":available,
                                                   "case":number,"rendered_sha256":sha(rendered.encode()),**value})
        receipt["adjudication"] = adjudicate(observed)
    receipt["elapsed_seconds_including_cleanup"] = time.monotonic()-start
    require(receipt["elapsed_seconds_including_cleanup"] <= 300, "Cleanup exceeded inclusive deadline")
    require(sum(row["kind"] == "compiler" for row in receipt["commands"]) == 4, "Compiler count differs")
    require(sum(row["kind"] == "stub" for row in receipt["commands"]) == 32, "Stub process count differs")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true",help="Opt in to the bounded synthetic compiler/stub checks")
    args = parser.parse_args()
    if not args.execute:
        parser.error("No execution is implicit; use pure test_stub.py checks or explicit --execute")
    try:
        receipt = execute()
    except Exception as error:
        receipt = getattr(error,"receipt",{"schema":"madloop-reduction-latch-stub-v1","scope":SCOPE})
        receipt["error"] = repr(error)
        print(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False))
        raise SystemExit(1)
    print(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
