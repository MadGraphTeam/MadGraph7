#!/usr/bin/env python3
"""Clone corrected generated source and add one explicit, checked DP retry."""
import argparse,difflib,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];DOC=Path(__file__).resolve().parent
WORK=ROOT/".benchmark-work/experiments/t3-dp-retry"
SOURCE=ROOT/".benchmark-work/policy-v2/generation/outputs/candidate"
BRIDGE=WORK/"bridge"
sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+"\n")
COMMON="""      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED
      INTEGER NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL
      INTEGER NRS_RETRY_HARD_SKIP
      REAL*8 NRS_NATIVE_ACC,NRS_COMP_ACC
      COMMON/ML5_0_NRS_RETRY_FLAGS/NRS_RETRY_MODE,NRS_RETRY_USED
      COMMON/ML5_0_NRS_RETRY_STATS/NRS_RETRY_COUNT,NRS_RETRY_PASS,
     $ NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP
      COMMON/ML5_0_NRS_RETRY_ACC/NRS_NATIVE_ACC,NRS_COMP_ACC
"""
WRAPPER="""      SUBROUTINE ML5_0_SLOOPMATRIX(P_USER,ANS)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      REAL*8 P_USER(0:3,NEXTERNAL),ANS(0:3,*)
"""+COMMON+"""      NRS_RETRY_MODE=.FALSE.
      NRS_RETRY_USED=.FALSE.
      NRS_RETRY_COUNT=0
      NRS_RETRY_PASS=0
      NRS_RETRY_FAIL=0
      NRS_RETRY_HARD_SKIP=0
      NRS_NATIVE_ACC=-1D0
      NRS_COMP_ACC=-1D0
      CALL ML5_0_SLOOPMATRIX_RETRY_IMPL(P_USER,ANS)
C     Guaranteed reset after every ordinary return from the implementation.
      NRS_RETRY_MODE=.FALSE.
      END

"""
RETRY="""C         Experimental explicit DP retry, never an override of stability.
          IF(NRS_RETRY_MODE) THEN
            NRS_COMP_ACC=MAXVAL(ACC)
            NRS_RETRY_MODE=.FALSE.
            IF(MAXVAL(ACC).LT.MLSTABTHRES.AND.
     $         .NOT.FPE_IN_DP_REDUCTION) THEN
              NRS_RETRY_PASS=NRS_RETRY_PASS+1
            ELSE
              NRS_RETRY_FAIL=NRS_RETRY_FAIL+1
            ENDIF
          ENDIF
          IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
            IF(MLREDUCTIONLIB(I_LIB).EQ.8.AND.
     $         .NOT.NRS_RETRY_USED) THEN
              NRS_NATIVE_ACC=MAXVAL(ACC)
              IF(.NOT.FPE_IN_DP_REDUCTION.AND.
     $           ALL(IEEE_IS_FINITE(ACC)).AND.
     $           ALL(IEEE_IS_FINITE(DP_RES(:,:,1:N_DP_EVAL)))) THEN
                NRS_RETRY_USED=.TRUE.
                NRS_RETRY_MODE=.TRUE.
                NRS_RETRY_COUNT=NRS_RETRY_COUNT+1
                EVAL_DONE(1)=.TRUE.
                DO I=2,MAXSTABILITYLENGTH
                  EVAL_DONE(I)=.FALSE.
                ENDDO
                STAB_INDEX=0
                CTMODE=1
                DP_RES=0D0
                IF(NROTATIONS_DP.GE.1) THEN
                  GOTO 200
                ELSE
                  GOTO 300
                ENDIF
              ELSE
                NRS_RETRY_HARD_SKIP=NRS_RETRY_HARD_SKIP+1
              ENDIF
            ENDIF
C           Mode is off before advancing a backend or entering genuine QP.
            NRS_RETRY_MODE=.FALSE.
"""
def environment():
    e=os.environ.copy();e.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        MAKEFLAGS="SHELL="+shutil.which("bash"),
        SYMBOLICA_LOCK_DIR=str(ROOT/".ninjars-build/symbolica-locks"),
        LD_LIBRARY_PATH=":".join(str(ROOT/".benchmark-work/original/HEPTools"/p)
            for p in ["ninja/lib","collier"]))
    return e
def prepare(case):
    source=SOURCE/case;output=WORK/case;report=DOC/case
    if output.exists():raise RuntimeError(f"Refusing overwrite {output}")
    report.mkdir(parents=True,exist_ok=True)
    subprocess.run(["cp","-a","--reflink=auto",str(source),str(output)],check=True)
    for p in output.rglob("*"):
        if p.is_symlink()and p.resolve().is_relative_to(source):
            target=output/p.resolve().relative_to(source);p.unlink();p.symlink_to(os.path.relpath(target,p.parent))
    proc=next(output.glob("SubProcesses/P*_*"))
    original={}
    for name in ["loop_matrix.f","CT_interface.f","check_sa.f"]:
        original[name]=(proc/name).read_text()
    matrix=original["loop_matrix.f"]
    first=matrix.index("      SUBROUTINE ML5_0_SLOOPMATRIX(P_USER,ANS)")
    end=matrix.index("      SUBROUTINE ML5_0_CLEAR_CACHES()",first)
    block=matrix[first:end].replace("SUBROUTINE ML5_0_SLOOPMATRIX(P_USER,ANS)",
        "SUBROUTINE ML5_0_SLOOPMATRIX_RETRY_IMPL(P_USER,ANS)",1)
    block=block.replace("      IMPLICIT NONE\n","      USE, INTRINSIC :: IEEE_ARITHMETIC,\n"
        "     $ ONLY: IEEE_IS_FINITE\n      IMPLICIT NONE\n"+COMMON,1)
    branch=block.index("        ELSEIF(.NOT.DOING_QP_EVALS)THEN")
    needle="          IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN\n"
    pos=block.index(needle,branch);block=block[:pos]+block[pos:].replace(needle,RETRY,1)
    matrix=matrix[:first]+WRAPPER+block+matrix[end:];(proc/"loop_matrix.f").write_text(matrix)
    interface=original["CT_interface.f"];first=interface.index("      SUBROUTINE ML5_0_NINJARS_LOOP(")
    end=interface.index("      SUBROUTINE ML5_0_MP_NINJARS_LOOP(",first)
    block=interface[first:end]
    block=block.replace("      IMPLICIT NONE\n","      IMPLICIT NONE\n"+COMMON,1)
    i=block.index("      CALL NINJARS_TENSOR_EVALUATE(");j=block.index("      STABLE=NRS_STATUS",i)
    call=block[i:j]
    corrected="""      CALL NINJARS_TENSOR_EVALUATE_COMPENSATED(
     $ TENSORCOEFS(0:CURR_MAXCOEF-1),NLOOPLINE,RANK,
     $ REAL_S_MAT,P_NRS,MASSES,NRS_SCALE,NRS_RES,R1,NRS_STATUS)
"""
    block=block[:i]+"      IF(NRS_RETRY_MODE) THEN\n"+corrected+"      ELSE\n"+call+"      ENDIF\n"+block[j:]
    interface=interface[:first]+block+interface[end:];(proc/"CT_interface.f").write_text(interface)
    driver=original["check_sa.f"].replace("      IMPLICIT NONE\n","      IMPLICIT NONE\n"+COMMON,1)
    driver=driver.replace("C     Save numerical outputs outside the timing interval.\n",
        "C     Save numerical outputs outside the timing interval.\n"
        "      OPEN(74,FILE='retry-stats.dat',STATUS='REPLACE')\n",1)
    driver=driver.replace("         VALUES(:,:,K)=ANS\n",
        "         WRITE(74,'(6I8,2L3,2ES26.17E3)') K,CODE,\n"
        "     $    NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL,\n"
        "     $    NRS_RETRY_HARD_SKIP,NRS_RETRY_MODE,NRS_RETRY_USED,\n"
        "     $    NRS_NATIVE_ACC,NRS_COMP_ACC\n"
        "         IF(NRS_RETRY_MODE) STOP 91\n"
        "         VALUES(:,:,K)=ANS\n",1)
    driver=driver.replace("C     Suppress any repeated Fortran diagnostics inside the timed region.",
        "      CLOSE(74)\nC     Suppress any repeated Fortran diagnostics inside the timed region.",1)
    (proc/"check_sa.f").write_text(driver)
    definitions=output/"SubProcesses/MadLoop_makefile_definitions"
    old=definitions.read_text();definitions.write_text(old.replace(str(ROOT/"HEPTools/ninjars/lib"),str(BRIDGE)))
    card=MadLoopParam(str(output/"Cards/MadLoopParams.dat"))
    for k,v in {"MLReductionLib":"8","CTModeRun":-1,"MLStabThres":1e-9,
        "COLLIERRequiredAccuracy":1e-12}.items():card[k]=v
    for p in [output/"Cards/MadLoopParams.dat",*(output/"SubProcesses").rglob("MadLoopParams.dat")]:
        assert p.resolve().is_relative_to(output)
        card.write(str(p),template=str(ROOT/"Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    (proc/"check").unlink(missing_ok=True)
    with (report/"build.log").open("w")as log:
        subprocess.run(["make","-j2","check"],cwd=proc,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    patch=""
    for name,old in original.items():
        new=(proc/name).read_text()
        patch+="".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
            fromfile="a/"+name,tofile="b/"+name))
        shutil.copy2(proc/name,report/name)
    (report/"process.patch").write_text(patch)
    manifest=bench.generated_source_manifest(output);save(report/"generated-sources.json",manifest)
    for name in ["param_card.dat","MadLoopParams.dat"]:shutil.copy2(output/"Cards"/name,report/name)
    save(report/"build.json",{"base_output":str(source.relative_to(ROOT)),
        "source_compilation_sha256":sha(source/"compilation.json"),
        "source_files":{k:hashlib.sha256(v.encode()).hexdigest()for k,v in original.items()},
        "executable_sha256":sha(proc/"check"),"archive_sha256":sha(BRIDGE/"libninjars.a"),
        "generated_sources_sha256":manifest["files_sha256"],"patch_sha256":sha(report/"process.patch")})
def run(case,cpu,points,loops,label,rotations=None):
    output=WORK/case;proc=next(output.glob("SubProcesses/P*_*"));report=DOC/case/"runs"/label
    if rotations is not None:
        assert rotations>=0
        card=MadLoopParam(str(output/"Cards/MadLoopParams.dat"))
        card["NRotations_DP"]=rotations
        for path in [output/"Cards/MadLoopParams.dat",*(output/"SubProcesses").rglob("MadLoopParams.dat")]:
            assert path.resolve().is_relative_to(output)
            card.write(str(path),template=str(ROOT/"Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    build=json.loads((DOC/case/"build.json").read_text())
    assert sha(proc/"check")==build["executable_sha256"],"executable changed after build"
    assert sha(BRIDGE/"libninjars.a")==build["archive_sha256"],"archive changed after build"
    manifest=bench.generated_source_manifest(output)
    assert manifest["files_sha256"]==build["generated_sources_sha256"],"generated source changed after build"
    candidates=[proc/"param_card.dat"]+[
        proc.joinpath(*([".."]*level),"Cards/param_card.dat") for level in range(6)]
    actual_param=next((path for path in candidates if path.is_file()),None)
    assert actual_param and actual_param.read_bytes()==(output/"Cards/param_card.dat").read_bytes(),\
        "SETPARA reads a different parameter card"
    report.mkdir(parents=True)
    pointpath=ROOT/"docs/benchmarkfolder/points"/f"{case}.dat"
    if points:
        text=pointpath.read_text().splitlines();nexternal=(len(text)-1)//int(text[0])
        indices=[int(x)for x in points.split(",")]
        assert all(1<=i<=int(text[0]) for i in indices)
        subset=[str(len(indices))]
        for i in indices:subset.extend(text[1+(i-1)*nexternal:1+i*nexternal])
        pointpath=report/"selected-points.dat";pointpath.write_text("\n".join(subset)+"\n")
        save(report/"point-map.json",indices)
    else:
        shutil.copy2(pointpath,report/"points.dat")
    for name in ["MadLoopParams.dat","param_card.dat"]:
        shutil.copy2(output/"Cards"/name,report/name)
    artifacts=["values.dat","retry-stats.dat","timed_status.dat","timing.dat","dimensions.dat"]
    for name in artifacts:(proc/name).unlink(missing_ok=True)
    command=["taskset","-c",str(cpu),"./check",str(pointpath),str(loops)]
    metadata={"command":command,"working_directory":str(proc.relative_to(ROOT)),
        "executable_sha256":sha(proc/"check"),"archive_sha256":sha(BRIDGE/"libninjars.a"),
        "generated_sources_sha256":manifest["files_sha256"],"points_sha256":sha(pointpath),
        "card_sha256":sha(output/"Cards/MadLoopParams.dat"),
        "parameter_card_sha256":sha(output/"Cards/param_card.dat"),
        "resolved_parameter_card":str(actual_param.resolve().relative_to(ROOT)),
        "note":"Exploratory correctness/retry controls only. No acceptance performance claim."}
    save(report/"run.json",metadata)
    with (report/"run.log").open("w")as log:
        result=subprocess.run(command,cwd=proc,env=environment(),stdout=log,stderr=subprocess.STDOUT)
    for name in artifacts:
        if (proc/name).exists():shutil.copy2(proc/name,report/name)
    metadata.update(returncode=result.returncode,artifacts={
        name:sha(report/name) for name in artifacts if (report/name).exists()})
    save(report/"run.json",metadata)
    result.check_returncode()
    assert len(metadata["artifacts"])==len(artifacts),"missing output"
    print((report/"retry-stats.dat").read_text())
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","run"]);p.add_argument("--case",default="qcd-3")
    p.add_argument("--cpu",type=int,default=4);p.add_argument("--points");p.add_argument("--loops",type=int,default=1)
    p.add_argument("--label",default="full-100")
    p.add_argument("--rotations",type=int)
    a=p.parse_args();prepare(a.case) if a.action=="prepare" else run(a.case,a.cpu,a.points,a.loops,a.label,a.rotations)
