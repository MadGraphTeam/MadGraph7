#!/usr/bin/env python3
"""Exercise the real generated retry state machine with mocked reduction values.

This is strictly control-flow evidence: mocked QP is not a numerical rescue.
No NinjaRS DP/MP C evaluation is invoked by either mocked wrapper.
"""
import argparse,hashlib,importlib.util,json,os,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];DOC=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("retry_process",DOC/"process.py")
process=importlib.util.module_from_spec(spec);spec.loader.exec_module(process)
WORK=process.WORK/"mock-flow";OUTPUT=WORK/"qcd-1";REPORT=DOC/"mock-flow"
COMMON="""      INTEGER TEST_SCENARIO,TEST_REBUILDS,TEST_DP,TEST_QP
      COMMON/ML5_0_RETRY_TEST/TEST_SCENARIO,TEST_REBUILDS,
     $ TEST_DP,TEST_QP
"""
def save(p,v):p.write_text(json.dumps(v,indent=2)+"\n")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    source=process.WORK/"qcd-1"
    if OUTPUT.exists():raise RuntimeError(f"Refusing overwrite {OUTPUT}")
    WORK.mkdir(parents=True);REPORT.mkdir()
    subprocess.run(["cp","-a","--reflink=auto",str(source),str(OUTPUT)],check=True)
    for p in OUTPUT.rglob("*"):
        if p.is_symlink()and p.resolve().is_relative_to(source):
            target=OUTPUT/p.resolve().relative_to(source);p.unlink();p.symlink_to(os.path.relpath(target,p.parent))
    proc=next(OUTPUT.glob("SubProcesses/P*_*"))
    matrix=(proc/"loop_matrix.f").read_text()
    # Count explicit label-200 rebuilds; initial construction enters label 208.
    end=matrix.index("      SUBROUTINE ML5_0_CLEAR_CACHES()")
    main=matrix[:end].replace("      IMPLICIT NONE\n","      IMPLICIT NONE\n"+COMMON)
    main=main.replace("      NRS_RETRY_MODE=.FALSE.\n", "      TEST_REBUILDS=0\n"
        "      TEST_DP=0\n      TEST_QP=0\n      NRS_RETRY_MODE=.FALSE.\n",1)
    pattern=r"(?m)^(\s*200\s+CONTINUE)\s*$"
    main,count=re.subn(pattern,r"\1\n      TEST_REBUILDS=TEST_REBUILDS+1",main);assert count==1
    pattern=r"(?m)^(\s*300\s+CONTINUE)\s*$"
    main,count=re.subn(pattern,r"\1\n      IF(CTMODE.LT.4) THEN\n"
        "        TEST_DP=TEST_DP+1\n      ELSE\n        TEST_QP=TEST_QP+1\n      ENDIF",main);assert count==1
    (proc/"loop_matrix.f").write_text(main+matrix[end:])
    text=(proc/"CT_interface.f").read_text()
    start=text.index("      SUBROUTINE ML5_0_NINJARS_LOOP(")
    mp=text.index("      SUBROUTINE ML5_0_MP_NINJARS_LOOP(",start)
    block=text[start:mp].replace("      IMPLICIT NONE\n","      IMPLICIT NONE\n"+COMMON,1)
    a=block.index("      IF(NRS_RETRY_MODE) THEN");b=block.index("      STABLE=NRS_STATUS",a)
    fake="""C     Test-only fake reduction values; no C bridge callback is made.
      NRS_STATUS=NINJARS_SUCCESS
      NRS_RES=CMPLX(1500D0,0D0,KIND=KI_NRS)
      IF(TEST_SCENARIO.NE.0) THEN
        IF((.NOT.NRS_RETRY_MODE).OR.TEST_SCENARIO.EQ.3) THEN
          NRS_RES=CMPLX(1000D0*DBLE(CTMODE),0D0,KIND=KI_NRS)
        ENDIF
      ENDIF
      IF(TEST_SCENARIO.EQ.2.AND..NOT.NRS_RETRY_MODE) THEN
        NRS_STATUS=NINJARS_ERROR
        NRS_RES=CMPLX(987D0,654D0,KIND=KI_NRS)
      ENDIF
"""
    block=block[:a]+fake+block[b:]
    tail=text[mp:];end=tail.index("\n      END\n")+len("\n      END\n")
    fake_mp="""      SUBROUTINE ML5_0_MP_NINJARS_LOOP(NLOOPLINE,PL,M2L,
     $ RANK,RES,STABLE)
      IMPLICIT NONE
      INTEGER NLOOPLINE,RANK
      REAL*16 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE),RES(3)
      LOGICAL STABLE
C     Test-only QP stub. This is not genuine MP numerical evidence.
      RES=(0.0D0,0.0D0)
      STABLE=.TRUE.
      END
"""
    (proc/"CT_interface.f").write_text(text[:start]+block+fake_mp+tail[end:])
    driver=(proc/"check_sa.f").read_text()
    driver=driver[:driver.index("C     Save numerical outputs outside the timing interval.")]
    driver=driver.replace("      IMPLICIT NONE\n","      IMPLICIT NONE\n"+COMMON,1)
    driver=driver.replace("      CHARACTER*1024 ARG\n","      CHARACTER*1024 ARG\n"
        "      INTEGER SCENARIOS(6),IDX\n      DATA SCENARIOS/1,0,3,0,2,0/\n")
    driver=driver.replace("      CALL SETPARA","      TEST_SCENARIO=0\n      CALL SETPARA",1)
    driver+="""      OPEN(74,FILE='mock-flow.dat',STATUS='REPLACE')
      DO IDX=1,6
        TEST_SCENARIO=SCENARIOS(IDX)
        P=POINTS(:,:,1)
        TOTAL=P(:,1)+P(:,2)
        MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
        CALL UPDATE_AS_PARAM()
        CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
        WRITE(74,'(9I8,2L3,2ES26.17E3)') TEST_SCENARIO,CODE,
     $   NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL,
     $   NRS_RETRY_HARD_SKIP,TEST_REBUILDS,TEST_DP,TEST_QP,
     $   NRS_RETRY_MODE,NRS_RETRY_USED,NRS_NATIVE_ACC,NRS_COMP_ACC
        IF(NRS_RETRY_MODE) STOP 91
      ENDDO
      CLOSE(74)
      END
"""
    (proc/"check_sa.f").write_text(driver)
    (proc/"check").unlink(missing_ok=True)
    with (REPORT/"build.log").open("w")as log:
        subprocess.run(["make","-j2","check"],cwd=proc,env=process.environment(),
            stdout=log,stderr=subprocess.STDOUT,check=True)
    for name in ["loop_matrix.f","CT_interface.f","check_sa.f"]:shutil.copy2(proc/name,REPORT/name)
    save(REPORT/"build.json",{"executable_sha256":sha(proc/"check"),
        "sources":{name:sha(proc/name)for name in ["loop_matrix.f","CT_interface.f","check_sa.f"]},
        "note":"Mock reductions only. Both NinjaRS C callbacks are absent from the mocked wrapper bodies."})
def run(cpu):
    proc=next(OUTPUT.glob("SubProcesses/P*_*"));results=[]
    for rotations,backend in [(0,"8"),(2,"8"),(0,"8|6"),(2,"8|6")]:
        name=f"rotations-{rotations}-backends-{backend.replace('|','-')}"
        report=REPORT/name;report.mkdir()
        card=process.MadLoopParam(str(OUTPUT/"Cards/MadLoopParams.dat"))
        for k,v in {"NRotations_DP":rotations,"NRotations_QP":0,
            "MLReductionLib":backend,"MLStabThres":1e-9}.items():card[k]=v
        for p in [OUTPUT/"Cards/MadLoopParams.dat",*(OUTPUT/"SubProcesses").rglob("MadLoopParams.dat")]:
            assert p.resolve().is_relative_to(OUTPUT)
            card.write(str(p),template=str(ROOT/"Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
        (proc/"mock-flow.dat").unlink(missing_ok=True)
        command=["taskset","-c",str(cpu),"./check",str(ROOT/"docs/benchmarkfolder/points/qcd-1.dat"),"1"]
        with (report/"run.log").open("w")as log:
            subprocess.run(command,cwd=proc,env=process.environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
        shutil.copy2(proc/"mock-flow.dat",report/"mock-flow.dat")
        shutil.copy2(OUTPUT/"Cards/MadLoopParams.dat",report/"MadLoopParams.dat")
        rows=[]
        for line in (proc/"mock-flow.dat").read_text().splitlines():
            t=line.split();ints=list(map(int,t[:9]));case,code,count,passed,failed,hard,rebuilds,dp,qp=ints
            assert t[9]=="F","retry mode leaked"
            if case==0:
                assert(count,passed,failed,hard,code//100,code%10)==(0,0,0,0,2,8)
                assert t[10]=="F";assert dp==2+rotations and qp==0
            elif case==1:
                assert(count,passed,failed,hard,code//100,code%10)==(1,1,0,0,2,8)
                assert dp==2*(2+rotations) and qp==0
                # Initial construction bypasses 200; each rotation enters it.
                # A rotated retry also rebuilds once in the original frame.
                assert rebuilds==(2*rotations+1 if rotations else 0)
            else:
                assert(count,passed,failed,hard)==((0,0,0,1)if case==2 else(1,0,1,0))
                if backend=="8":
                    assert(code//100,code%10)==(3,2);assert qp==2
                else:
                    assert(code//100,code%10)==(2,6);assert qp==0
            rows.append({"scenario":case,"status":code,"retry_count":count,"passed":passed,
                "failed":failed,"hard_skip":hard,"rebuilds":rebuilds,"dp_phases":dp,
                "mock_qp_phases":qp,"mode_after_return":t[9],"used":t[10],
                "native_accuracy":float(t[11]),"compensated_accuracy":float(t[12])})
        assert [row["scenario"] for row in rows]==[1,0,3,0,2,0],\
            "missing, duplicated or reordered mocked scenarios"
        results.append({"case":name,"command":command,"rows":rows,"passed":True})
        print(name,"PASS",flush=True)
    save(REPORT/"results.json",{"control_flow_only":True,"genuine_mp":False,"passed":True,"cases":results})
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","run"]);p.add_argument("--cpu",type=int,default=5)
    a=p.parse_args();prepare()if a.action=="prepare"else run(a.cpu)
