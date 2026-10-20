#!/usr/bin/env python3
"""Fixed-size four-arm binary64 kernel experiment; never a process benchmark."""
import sys
sys.dont_write_bytecode=True
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0,str(BASE/'oracle'))
import oracle

FLAGS={'standalone':['-O3','-w','-fPIC','-ffixed-line-length-132'],
       'nlo':['-O3','-w','-fPIC','-ffixed-line-length-132','-fno-automatic']}
ARMS=['original','guarded-original','factor-only','A-zero-only']
SCHEDULE=[[0,1,2,3],[3,2,1,0],[1,2,3,0],[0,3,2,1],
          [2,3,0,1],[1,0,3,2],[3,0,1,2],[2,1,0,3]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def command(args,cwd,log):
    started=time.monotonic()
    result=subprocess.run(args,cwd=cwd,capture_output=True,text=True)
    (cwd/log).write_text(result.stdout+result.stderr)
    record=dict(command=args,returncode=result.returncode,elapsed=time.monotonic()-started)
    if result.returncode:
        raise RuntimeError(f'{cwd/log}: {result.returncode}')
    return record


def source_work(flags,vertex,rank):
    run='oracle-sa-rank234-v1' if flags=='standalone' else ('oracle-nlo-rank2-v1' if rank==2 else 'oracle-nlo-rank34-v1')
    return BASE/run/f'{flags}-{vertex}-rank{rank}-full'


def variants(work,source,rank):
    original=(source/'original.f').read_text()
    instrumented=(source/'factor.f').read_text()
    factor,removed=re.subn(r'(?m)^\s*(?:INTEGER FAST,FALLBACK|COMMON/FACTOR_TRACE/FAST,FALLBACK|FAST=FAST\+1|FALLBACK=FALLBACK\+1)\n','',instrumented)
    assert removed>=8 and 'FACTOR_TRACE' not in factor
    marker='C     FACTOR_FAST_PATH\n'
    prefix,rest=factor.split(marker,1)
    _,internal=rest.split('      CONTAINS\n',1)
    body=original[original.index('      OUT(:,:,:)'):]
    assert body.endswith('      END\n')
    body=body[:-len('      END\n')]
    zero=body.replace('                NEW_POSITION = COMB_COEF_POS(L,M)',
                      '                IF(A(K,L,I).EQ.(0.0D0,0.0D0)) CYCLE\n'
                      '                NEW_POSITION = COMB_COEF_POS(L,M)')
    assert zero!=body
    name=f'TEST_UPDATE_WL_{rank}_1_FACTOR'
    guarded=(prefix+marker+body+'      CONTAINS\n'+internal).replace(name,f'TEST_UPDATE_WL_{rank}_1_GUARDED')
    zero=(prefix+marker+zero+'      CONTAINS\n'+internal).replace(name,f'TEST_UPDATE_WL_{rank}_1_AZERO')
    for filename,text in [('original.f',original),('factor.f',factor),('guarded.f',guarded),('azero.f',zero)]:
        (work/filename).write_text(text)
    for filename in ('constants.f','coef_specs.inc','loop_max_coefs.inc'):
        (work/filename).write_bytes((source/filename).read_bytes())
    return dict(trace_lines_removed=removed,source_original_sha256=sha(source/'original.f'),
                source_instrumented_factor_sha256=sha(source/'factor.f'),
                zero_only_change='same admission prefix as factor; original body plus A==0 cycle; no reassociation')


def driver(rank,loops,pairs):
    name=f'TEST_UPDATE_WL_{rank}_1'
    calls=f"""select case(arm)
case(0)
 call {name}(A,4,B,4,4,out)
case(1)
 call {name}_GUARDED(A,4,B,4,4,out,.true.)
case(2)
 call {name}_FACTOR(A,4,B,4,4,out,.true.)
case(3)
 call {name}_AZERO(A,4,B,4,4,out,.true.)
end select"""
    order=','.join(str(v) for row in SCHEDULE for v in row)
    return f"""program kernel
use,intrinsic::ieee_arithmetic
use,intrinsic::ieee_exceptions
implicit none
complex*16 A(4,0:209,4),B(4,0:14,4),out(4,0:209,4),reference(4,0:209,4,0:1)
integer*8 bits(2),abits(2*4*210*4),bbits(2*4*15*4),outbits(2*4*210*4),refbits(2*4*210*4)
integer*8 wall0,wall1,rate
integer arm,it,pair,position,j,l,i,ios,expected,zero_changes
integer,parameter::schedule(4,8)=reshape((/{order}/),(/4,8/))
real*8 t0,t1,cpu,wall,checksum
logical flags(size(ieee_all)),baseline_flags(size(ieee_all))
character(8) tag
character(16) hex1,hex2
character(4096) filename
call get_command_argument(1,filename)
if(storage_size(1.0_8)/=64.or.digits(1.0_8)/=53) stop 1
call ieee_set_rounding_mode(ieee_nearest)
do i=1,size(ieee_all)
 if(.not.ieee_support_halting(ieee_all(i))) stop 2
 call ieee_set_halting_mode(ieee_all(i),.false.)
enddo
A=0
B=0
reference=0
open(77,file=trim(filename),status='old',action='read')
do
 read(77,*,iostat=ios) tag,j,l,i,hex1,hex2
 if(ios<0) exit
 if(ios/=0) stop 3
 read(hex1,'(Z16)') bits(1)
 read(hex2,'(Z16)') bits(2)
 select case(trim(tag))
 case('A')
  A(j,l,i)=transfer(bits,A(j,l,i))
 case('B')
  B(j,l,i)=transfer(bits,B(j,l,i))
 case('OLD')
  reference(j,l,i,0)=transfer(bits,out(j,l,i))
 case('NEW')
  reference(j,l,i,1)=transfer(bits,out(j,l,i))
 case default
  stop 4
 end select
enddo
close(77)
abits=transfer(A,abits)
bbits=transfer(B,bbits)
call system_clock(count_rate=rate)
do arm=0,3
 call ieee_set_flag(ieee_all,.false.)
 call ieee_set_flag(ieee_inexact,.true.)
 {calls}
 call ieee_get_flag(ieee_all,flags)
 if(arm==0) baseline_flags=flags
 if(any(flags.neqv.baseline_flags)) stop 5
 call verify_output()
 write(*,*) 'VALIDATED',arm,zero_changes
enddo
do arm=0,3
 do it=1,100
  {calls}
 enddo
enddo
do pair=1,{pairs}
 do position=1,4
  arm=schedule(position,mod(pair-1,8)+1)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,.true.)
  call system_clock(wall0)
  call cpu_time(t0)
  do it=1,{loops}
   {calls}
  enddo
  call cpu_time(t1)
  call system_clock(wall1)
  cpu=t1-t0
  wall=real(wall1-wall0,8)/real(rate,8)
  call ieee_get_flag(ieee_all,flags)
  if(any(flags.neqv.baseline_flags)) stop 6
  call verify_output()
  checksum=real(sum(out),8)
  write(*,'(A,3(1X,I0),3(1X,ES25.17))') 'SAMPLE',pair,position,arm,cpu,wall,checksum
 enddo
enddo
write(*,*) 'PASS', {pairs}, {loops}
contains
subroutine verify_output()
 expected=merge(1,0,arm==2)
 outbits=transfer(out,outbits)
 refbits=transfer(reference(:,:,:,expected),refbits)
 zero_changes=count(outbits/=refbits)
 if(arm/=3.and.zero_changes/=0) stop 7
 if(arm==3.and.any(out/=reference(:,:,:,expected))) stop 8
 if(any(abits/=transfer(A,abits)).or.any(bbits/=transfer(B,bbits))) stop 9
end subroutine
end program
"""


def fixture(work,case):
    path=work/f'fixture-{case["number"]}.dat'
    with path.open('x') as output:
        for tag in ('A','B','OLD','NEW'):
            for key,value in sorted(case['bits'][tag].items()):
                output.write(tag+' '+' '.join(map(str,key))+f' {value[0]:016X} {value[1]:016X}\n')
    return dict(case=case['number'],file=path.name,sha256=sha(path),
                A_zero=sum(v==oracle.ZERO for v in case['arrays']['A'].values()),
                A_total=len(case['arrays']['A']),B_zero=sum(v==oracle.ZERO for v in case['arrays']['B'].values()),
                B_total=len(case['arrays']['B']),fixture_kind={1:'source-derived dense phase-related rows',2:'source-derived sparse phase-related rows',7:'source-derived dense independently rounded actual ALOHA expressions'}[case['number']])


def prepare(report,args):
    report.mkdir(exist_ok=False)
    (report/'run.py').write_bytes(Path(__file__).read_bytes())
    seal=BASE/'correctness-seal.json'
    command([sys.executable,'-B',str(BASE/'verify-correctness.py')],report,'correctness-seal.log')
    plan=dict(schema='fixed-dp-kernel-four-arm-v1',complete=False,cpu=args.cpu,pairs=21,loops=5000,
              warmup_calls_per_arm=100,arms=ARMS,schedule=SCHEDULE,case_ids=[1,2,7],flags=FLAGS,
              correctness_seal_sha256=sha(seal),runner_sha256=sha(Path(__file__)),
              scope='Source-derived exact-validated pure-Fortran DP kernels; no captured physical call trace, Symbolica, generated process or acceptance claim.',
              compiler=command(['gfortran','--version'],report,'compiler.log'),configurations=[])
    write_json(report/'plan.json',plan)
    try:
      for flags in FLAGS:
       for vertex in ('FFV1L1_2','FFV1L2_1'):
        for rank in (2,3,4):
          key=f'{flags}-{vertex}-rank{rank}'
          work=report/key
          work.mkdir()
          source=source_work(flags,vertex,rank)
          result=json.loads((source/'exact-oracle.json').read_text())
          assert result['passed'] and result['dump_sha256']==sha(source/'dump.txt.gz')
          sources=variants(work,source,rank)
          (work/'kernel.f90').write_text(driver(rank,plan['loops'],plan['pairs']))
          build=command(['gfortran',*FLAGS[flags],'constants.f','original.f','factor.f','guarded.f','azero.f','kernel.f90','-o','kernel'],work,'build.log')
          cases=oracle.read_cases(source/'dump.txt.gz')
          fixtures=[]
          for number in plan['case_ids']:
            assert result['cases'][number-1]['fast']==1
            fixtures.append(fixture(work,cases[number-1]))
          plan['configurations'].append(dict(key=key,rank=rank,vertex=vertex,flags=flags,build=build,variants=sources,
              source_dump=str(source.relative_to(BASE)/'dump.txt.gz'),source_dump_sha256=sha(source/'dump.txt.gz'),
              oracle_result_sha256=sha(source/'exact-oracle.json'),fixtures=fixtures,executable_sha256=sha(work/'kernel'),
              generated_sources={p.name:sha(p) for p in work.iterdir() if p.suffix in ('.f','.f90','.inc')}))
          write_json(report/'plan.json',plan)
          print(key,'prepared',flush=True)
      plan['complete']=True
    except Exception as error:
      plan['failure']=repr(error)
      raise
    finally:
      write_json(report/'plan.json',plan)


def cpu_stats():
    return {row.split()[0]:list(map(int,row.split()[1:])) for row in Path('/proc/stat').read_text().splitlines() if re.fullmatch(r'cpu\d+',row.split()[0])}


def host_preflight(cpu):
    sibling_text=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip()
    siblings=[]
    for item in sibling_text.split(','):
        pair=item.split('-')
        siblings.extend(range(int(pair[0]),int(pair[-1])+1))
    snapshots=[]
    for _ in range(6):
        snapshots.append(dict(monotonic=time.monotonic(),loadavg=Path('/proc/loadavg').read_text().strip(),stats=cpu_stats()))
        if len(snapshots)<6:
            time.sleep(1)
    result=dict(cpu=cpu,siblings=siblings,loadavg=[s['loadavg'] for s in snapshots],cpu_idle={})
    for current in siblings:
        a=snapshots[0]['stats'][f'cpu{current}']; b=snapshots[-1]['stats'][f'cpu{current}']
        delta=[y-x for x,y in zip(a,b)]
        result['cpu_idle'][str(current)]=(delta[3]+delta[4])/sum(delta)
    result['passed']=result['cpu_idle'][str(cpu)]>=.90 and all(v>=.80 for v in result['cpu_idle'].values())
    result['raw_cpu_snapshots']=[{str(c):s['stats'][f'cpu{c}'] for c in siblings} for s in snapshots]
    return result


def measure(report):
    plan=json.loads((report/'plan.json').read_text())
    assert plan['complete'] and plan['pairs']==21 and plan['loops']==5000
    assert not (report/'measurements.json').exists()
    os.sched_setaffinity(0,{plan['cpu']})
    preflight=host_preflight(plan['cpu'])
    write_json(report/'preflight.json',preflight)
    if not preflight['passed']:
        raise RuntimeError('CPU/sibling idle preflight failed; no measurement launched')
    data=dict(complete=False,plan_sha256=sha(report/'plan.json'),preflight_sha256=sha(report/'preflight.json'),all_samples_retained=True,results=[])
    write_json(report/'measurements.json',data)
    try:
      for configuration in plan['configurations']:
        work=report/configuration['key']
        assert sha(work/'kernel')==configuration['executable_sha256']
        for fixture in configuration['fixtures']:
          assert sha(work/fixture['file'])==fixture['sha256']
          log=f'case-{fixture["case"]}.log'
          ran=command([str(work/'kernel'),fixture['file']],work,log)
          lines=[row.split() for row in (work/log).read_text().splitlines()]
          assert len(lines)==4+21*4+1 and [row[:2] for row in lines[:4]]==[['VALIDATED',str(i)] for i in range(4)]
          assert lines[-1]==['PASS','21','5000']
          samples=[]
          for row in lines[4:-1]:
            assert row[0]=='SAMPLE' and len(row)==7
            pair,position,arm=map(int,row[1:4])
            assert arm==SCHEDULE[(pair-1)%8][position-1]
            cpu,wall,checksum=map(float,row[4:])
            assert cpu>0 and wall>0
            samples.append(dict(pair=pair,position=position,arm=arm,cpu_seconds=cpu,wall_seconds=wall,checksum=checksum))
          assert [(r['pair'],r['position']) for r in samples]==[(pair,position) for pair in range(1,22) for position in range(1,5)]
          data['results'].append(dict(key=configuration['key'],case=fixture['case'],run=ran,samples=samples,log_sha256=sha(work/log)))
          write_json(report/'measurements.json',data)
          print(configuration['key'],fixture['case'],'21 blocks complete',flush=True)
      data['complete']=True
    except Exception as error:
      data['failure']=repr(error)
      raise
    finally:
      write_json(report/'measurements.json',data)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','measure'])
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--cpu',type=int,default=4)
    args=parser.parse_args()
    report=BASE/args.run_id
    if args.action=='prepare':
        prepare(report,args)
    else:
        measure(report)
