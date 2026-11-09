#!/usr/bin/env python3
"""Bounded isolated compiler/IEEE/consumer/timing checks, pinned to CPU 10."""
import hashlib,json,os,resource,shutil,subprocess,time
from pathlib import Path
from madgraph.iolibs.file_writers import FortranWriter
from tests.unit_tests.various.test_dp_terminal_compiled import guard_driver,frozen,_CALLER,_CONSUMER_DRIVER
ROOT=Path.cwd(); HERE=ROOT/'docs/benchmarkfolder/experiments/dp-higher-rank/kernels'; WORK=ROOT/'.tirust-build/dp-higher-rank/kernels'
RESULT={'scope':'synthetic private kernels, no acceptance timing; numerical results require independent COMB and consumer proof','commands':[],'checks':[],'timings':[]}
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b,1)
def fixed(path,s,context=None):
    with FortranWriter(str(path)) as w:w.downcase=False;w.writelines(s,context=context or {})
def no_core():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def run(command,cwd,tag,timeout=90):
    started=time.time();p=subprocess.run(command,cwd=cwd,text=True,capture_output=True,timeout=timeout,preexec_fn=no_core,env=dict(os.environ,GFORTRAN_ERROR_BACKTRACE='0'))
    r=dict(command=list(map(str,command)),cwd=str(cwd),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,elapsed=time.time()-started)
    (HERE/(tag+'.log')).write_text(json.dumps(r,indent=2)+'\n');RESULT['commands'].append(r)
    (HERE/'results-in-progress.json').write_text(json.dumps(RESULT,indent=2)+'\n')
    return r
def good(r):assert r['returncode']==0,r
def build(w,flags,extra,out,tag):
    good(run(['gfortran',*flags,'-fcheck=all','-ffixed-line-length-132','-ffree-line-length-none','constants.f','public.f',*extra,'-o',out],w,tag))
for rank in (2,3):
    w=WORK/('r%d'%rank);w.mkdir(parents=True,exist_ok=True)
    for name in ('constants.f','public.f','coef_specs.inc','loop_max_coefs.inc','merge.f'):
        shutil.copy2(HERE/name,w/name)
    private=(HERE/('private-r%d.txt'%rank)).read_text()
    (w/'private.f').write_text((HERE/('private-r%d.f'%rank)).read_text())
    traced=once(private,'OUT_SIZE,OUT,WL_ENTRY_INEXACT)','OUT_SIZE,OUT,WL_ENTRY_INEXACT,TEST_FAST)')
    traced=once(traced,'  LOGICAL WL_ENTRY_INEXACT,','  LOGICAL TEST_FAST\n  LOGICAL WL_ENTRY_INEXACT,')
    traced=once(traced,'IF(.NOT.WL_ENTRY_INEXACT)','TEST_FAST=.FALSE.\nIF(.NOT.WL_ENTRY_INEXACT)')
    traced=once(traced,'C DP_TERMINAL_FAST_PATH','C DP_TERMINAL_FAST_PATH\nTEST_FAST=.TRUE.')
    fixed(w/'traced.f',traced)
    gd=guard_driver().replace('TEST_DP_TERMINAL_WL_1_1','TEST_DP_TERMINAL_WL_%d_1'%rank)
    gd=gd.replace('0:14','0:209').replace('0,14','0,209').replace('4*4*15*2','4*4*210*2').replace('15*2','210*2').replace(',new_bits,30)',',new_bits,420)')
    (HERE/('guards-r%d.f90'%rank)).write_text(gd);(w/'guards.f90').write_text(gd)
    nleft={2:15,3:35}[rank]
    oracle='''subroutine DP_DENSE(a,ls,b,ins,outs,out)
use TEST_POLYNOMIAL_CONSTANTS
implicit none
integer :: ls,ins,outs,j,m,k,l,i,pos
complex(8) :: a(4,0:209,4),b(4,0:4,4),out(4,0:209,4),u
out=cmplx(0.0_8,0.0_8,8)
do j=1,outs
 do m=0,4
  do k=1,ins
   u=b(j,m,k)
   if(u==cmplx(0.0_8,0.0_8,8)) cycle
   do l=0,NLEFT
    pos=COMB_COEF_POS(l,m)
    do i=1,ls
     out(j,pos,i)=out(j,pos,i)+a(k,l,i)*u
    enddo
   enddo
  enddo
 enddo
enddo
end subroutine
'''.replace('NLEFT',str(nleft-1))
    (HERE/('oracle-r%d.f90'%rank)).write_text(oracle);(w/'oracle.f90').write_text(oracle)
    for label,flags in [('o3',['-O3','-frounding-math']),('static',['-O3','-frounding-math','-fno-automatic'])]:
        build(w,flags,['oracle.f90','traced.f','guards.f90'],'guards','r%d-%s-build'%(rank,label))
        r=run([str(w/'guards')],w,'r%d-%s-guards'%(rank,label));good(r);assert '384000' in r['stdout'];RESULT['checks'].append({'rank':rank,'mode':label,'guard':r['stdout']})
    caller=_CALLER.replace('TEST_UPDATE_WL_1_1','TEST_UPDATE_WL_%d_1'%rank).replace('),2,4,2,3,-2,2,2)', '),%d,4,2,3,-2,2,2)'%(rank+1))
    reference=caller.replace('CHECK_CANDIDATE','CHECK_REFERENCE').replace('TEST_UPDATE_WL_%d_1'%rank,'DP_DENSE')
    caller=once(caller,'IMPLICIT NONE','USE, INTRINSIC :: IEEE_EXCEPTIONS, ONLY: IEEE_GET_FLAG,IEEE_INEXACT\nIMPLICIT NONE\nLOGICAL WL_ENTRY_INEXACT')
    call='CALL TEST_UPDATE_WL_%d_1(WL(1,0,1,1),4,COEFS,4,4,WL(1,0,1,2))'%rank
    caller=once(caller,call,'CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n'+call.replace('UPDATE_WL','DP_TERMINAL_WL')[:-1]+',WL_ENTRY_INEXACT)')
    fixed(w/'caller.f',caller);fixed(w/'reference.f',reference)
    driver=_CONSUMER_DRIVER.replace('0:14','0:%d'%({2:34,3:69}[rank])).replace('30)',str({2:70,3:140}[rank])+')')
    # Compact kernels clear every coefficient, including the unused tail.
    driver=driver.replace('if(any(wl(:,15:,:,2)/=sentinel)) stop 49','if(any(wl(:,%d:,:,2)/=cmplx(0.0_8,0.0_8,8))) stop 49'%({2:35,3:70}[rank]))
    (w/'consumer_driver.f90').write_text(driver)
    template=(ROOT/'madgraph/iolibs/template_files/loop_optimized/polynomial.inc').read_text()
    rep=dict(mp_prefix='',proc_prefix='TEST_',nbornamps=2,real_format='REAL*8',complex_format='COMPLEX*16',kind=8,zero_def='0.0D0',one_def='1.0D0',nloopamps=2,nloop_groups=2,ncomb=2,nSO=1,nSquaredSO=3,nAmpSO=2,color_matrix_size='2')
    fixed(w/'consumer.f',template%rep,{'AmplitudeReduction':False,'LoopInduced':False,'ninja_available':False})
    good(run(['gfortran','-O0','-ffixed-line-length-132','-c','constants.f','caller.f','reference.f'],w,'r%d-caller-build'%rank))
    build(w,['-O3','-frounding-math'],['oracle.f90','private.f','merge.f','consumer.f','caller.o','reference.o','consumer_driver.f90'],'consumer','r%d-consumer-build'%rank)
    r=run([str(w/'consumer')],w,'r%d-consumer'%rank);good(r);RESULT['checks'].append({'rank':rank,'consumer':r['stdout']})
    for name in ('traced.f','caller.f','reference.f','consumer.f','consumer_driver.f90'):
        shutil.copy2(w/name,HERE/('r%d-'%rank+name))
    bench=(HERE/'timing-template.f90').read_text().replace('RANK_ID',str(rank))
    (w/'timing.f90').write_text(bench);shutil.copy2(w/'timing.f90',HERE/('timing-r%d.f90'%rank))
    build(w,['-O3'],['private.f','timing.f90'],'timing','r%d-timing-build'%rank)
    for pair in range(7):
        for method in ([0,1] if pair%2==0 else [1,0]):
            r=run([str(w/'timing'),str(method)],w,'r%d-pair%d-method%d'%(rank,pair,method));good(r)
            fields=r['stdout'].split();assert fields[0]=='TIMING'
            row={'rank':rank,'pair':pair,'method':method,'iterations':int(fields[1]),'seconds':float(fields[2]),'checksum_real':float(fields[3]),'checksum_imag':float(fields[4])}
            assert row['seconds']<=.1,row
            RESULT['timings'].append(row)
    print('RANK_PASS',rank,flush=True)
RESULT['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.f','.f90','.inc','.txt')}
RESULT['passed']=True
(HERE/'results.json').write_text(json.dumps(RESULT,indent=2)+'\n')
print('ALL_PASS',flush=True)
