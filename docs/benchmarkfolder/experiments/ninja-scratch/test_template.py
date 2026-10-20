#!/usr/bin/env python3
"""Compare rendered original/optimized Ninja payloads without applying the patch."""
from pathlib import Path
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = ROOT / ".benchmark-work/experiments/ninja-scratch"
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from madgraph.iolibs.file_writers import FortranWriter

TEMPLATE = ROOT / "madgraph/iolibs/template_files/loop_optimized"
PARAMS = ROOT / "Template/loop_material/StandAlone/SubProcesses"
old = (TEMPLATE / "CT_interface.inc").read_text()
new = (HERE / "CT_interface.optimized.inc").read_text()
helpers = old[old.index("\t  SUBROUTINE %(proc_prefix)sBUILD_KINEMATIC_MATRIX"):]
helpers = helpers[:helpers.index("## if(samurai_available)")]
switch = (TEMPLATE / "TIR_interface.inc").read_text()
switch = switch[switch.index("      SUBROUTINE %(proc_prefix)sSWITCH_ORDER"):]
switch = switch[:switch.index("\t  SUBROUTINE %(proc_prefix)sINITTIR")]
replacements = {
    "proc_prefix": "TEST_", "nloop_groups": 1, "nSquaredSO": 1, "nexternal": 4,
    "real_dp_format": "REAL*8", "complex_dp_format": "COMPLEX*16",
    "mass_dp_format": "COMPLEX*16", "real_mp_format": "REAL*16",
    "complex_mp_format": "COMPLEX*32", "include_vector": "",
    "info_lines": "", "process_lines": "",
}
module = """module TEST_POLYNOMIAL_CONSTANTS
integer :: NCOEF_R(0:3)=[1,5,15,35]
integer :: COEFTORANK_MAP(0:34)=[0,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
end module
module mninja
integer,parameter :: KI_NIN=kind(1d0),KI_QNIN=selected_real_kind(33,4931),NINJA_UNSTABLE_KINEMATICS=1
interface ninja_tensor_evaluate
module procedure eval_dp,eval_mp
end interface
contains
subroutine ninja_set_integral_library(i)
integer :: i
end subroutine
"""
for kind in ("KI_NIN", "KI_QNIN"):
    lane = "dp" if kind == "KI_NIN" else "mp"
    module += f"""subroutine eval_{lane}(c,n,r,s,p,m,mu,res,r1,status)
integer :: n,r,status,i
complex({kind}) :: c(:),m(n),res(0:2),r1
real({kind}) :: s(n,n),p(0:3,n),mu
if (size(c)/=(r+1)*(r+2)*(r+3)*(r+4)/24) stop 31
write(*,'(2I4)') n,r
write(*,'(ES48.37E4)') c,s,p,m,mu
status=0
r1=0
do i=0,2
res(i)=(i+1)*sum(c)*cmplx(1,1,kind={kind})
enddo
end subroutine
"""
module += "end module\n"
helper_stubs = """subroutine TEST_INITCT()
end subroutine
subroutine cite(a,b)
character(*) a,b
end subroutine
"""
for prefix, kind in (("", 8), ("MP_", 16)):
    helper_stubs += f"""subroutine {prefix}TEST_INVERT_MOMENTA_IN_POLYNOMIAL(n,c)
use TEST_POLYNOMIAL_CONSTANTS
integer :: n,i
complex({kind}) :: c(0:n-1)
do i=0,n-1
if(mod(COEFTORANK_MAP(i),2)==1) c(i)=-c(i)
enddo
end subroutine
"""

for amplitude in (False, True):
    outputs = []
    for name, source in (("original", old), ("scratch", new)):
        work = HERE / "template-test" / str(amplitude) / name
        work.mkdir(parents=True, exist_ok=True)
        body = source[source.index("## if(ninja_available){"):]
        with FortranWriter(str(work / "wrapper.f")) as writer:
            writer.writelines((body + helpers + switch) % replacements,
                              context={"ninja_available": True,
                                       "ninja_supports_quad_prec": True,
                                       "AmplitudeReduction": amplitude})
        (work / "modules.f90").write_text(module)
        (work / "stubs.f90").write_text(helper_stubs)
        (work / "loop_max_coefs.inc").write_text(
            "      INTEGER LOOPMAXCOEFS\n      PARAMETER (LOOPMAXCOEFS=35)\n")
        (work / "coupl.inc").write_text("      REAL*8 MU_R\n      COMMON/TEST_COUPL/MU_R\n")
        shape = "(0:34,1)" if amplitude else "(0:34,1,1)"
        (work / "driver.f90").write_text(f"""program check
implicit none
integer :: id,sqso,rank,mode,i,j,k,rs(5)=[2,0,3,1,0],ms(4)=[1,2,4,5]
common/TEST_LOOP/id,sqso,rank
real(8) :: p(0:3,3),mu_r,lscale
real(16) :: mp(0:3,3)
common/TEST_COUPL/mu_r
common/TEST_CT/lscale,mode
complex(8) :: coef{shape},m(3),res(3)
complex(16) :: mpcoef{shape}
common/TEST_LCOEFS/coef
common/TEST_MP_LCOEFS/mpcoef
logical :: stable,flags(6),fdp,fqp
common/REDUCTIONCODEINIT/flags
common/TEST_FPE_IN_REDUCTION/fdp,fqp
call DEFAULTPARAM()
id=1
sqso=1
mu_r=7
flags=.false.
fdp=.false.
fqp=.false.
coef=reshape([(cmplx(i+1,-i,kind=8),i=0,34)],shape(coef))
mpcoef=cmplx(coef,kind=16)+cmplx(2.0_16**(-100),0.0_16,kind=16)
do j=1,5
rank=rs(j)
do k=1,4
mode=ms(k)
p(:,1)=[4d0,1d0,2d0,3d0]
p(:,2)=[3d0,-1d0,1d0,-2d0]
p(:,3)=-p(:,1)-p(:,2)
m=[cmplx(2d0,-0.2d0,kind=8),cmplx(3d0,-0.1d0,kind=8),cmplx(4d0,0d0,kind=8)]
mp=real(p,kind=16)
call TEST_NINJA_LOOP(3,p,m,rank,res,stable)
write(*,'(ES48.37E4)') res
call TEST_MP_NINJA_LOOP(3,mp,m,rank,res,stable)
write(*,'(ES48.37E4)') res
enddo
enddo
end program
""")
        command = ["gfortran", "-ffixed-line-length-132", "-fcheck=all", "-I", str(PARAMS),
                   "modules.f90", "stubs.f90", "wrapper.f", "driver.f90",
                   str(PARAMS / "MadLoopParamReader.f"), "-o", "check"]
        subprocess.run(command, cwd=work, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outputs.append(subprocess.check_output(["./check"], cwd=work))
    if outputs[0] != outputs[1]:
        raise RuntimeError(f"Rendered template changes payload or result, amplitude={amplitude}")
    print(f"PASS amplitude={amplitude}: DP/QP ranks2,0,3,1,0; directions1,2,4,5; identical payload/results")
