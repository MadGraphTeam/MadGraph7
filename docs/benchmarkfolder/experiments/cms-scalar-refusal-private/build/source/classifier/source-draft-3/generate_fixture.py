#!/usr/bin/env python3
"""Write meaningful Fortran mock fixtures and exact future compile recipes only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path.cwd().resolve();HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    spec=importlib.util.spec_from_file_location('cms_source_generator',HERE/'generate.py')
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    metadata=json.loads((HERE/'classification.json').read_text())
    assert sha(g.SOURCE)==metadata['source_sha256']
    source=g.SOURCE.read_text();out=HERE/'fixture';out.mkdir(exist_ok=False)
    tables=[];modules=[];checks=[]
    for precision,kind in [('dp',8),('qp',16)]:
        for topology in ('tri','box'):
            header='module avh_olo_forCutTools_'+precision+'_'+topology
            start=source.index(header+'\n');end=source.index('\ncontains',start)
            block=source[start:end]
            constants=block[block.index('  integer ,parameter ::'):]
            tables.append(header+'\n  implicit none\n'+constants+'\nend module\n')
        prefix=f'module cms_mock_{precision}\n  use cms_scalar_observer\n'
        prefix+='  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_support_denormal\n  implicit none\n'
        prefix+=f'  integer,parameter :: kindr2={kind}\n  real(kindr2),parameter :: RZRO=0\n'
        prefix+='  integer :: wunit=0,prcpar=0,mock_calls=0,mock_nargs=0\n'
        prefix+='  real(kindr2) :: onshellthrs=1e-6_kindr2,muscale=1000\n'
        prefix+='  logical :: nonzerothrs=.true.,initz=.false.,mock_nan=.false.\n'
        prefix+='  complex(kindr2) :: mock_last_args(10)=0\ncontains\n'
        prefix+='  function areal(x) result(y)\n    complex(kindr2),intent(in)::x\n    real(kindr2)::y\n    y=real(x,kindr2)\n  end function\n'
        prefix+='  function acmplx(x,y) result(z)\n    real(kindr2),intent(in)::x,y\n    complex(kindr2)::z\n    z=cmplx(x,y,kind=kindr2)\n  end function\n'
        prefix+='  function neglig(i) result(y)\n    integer,intent(in)::i\n    real(kindr2)::y\n    y=epsilon(0._kindr2)\n  end function\n'
        stubs=[];routines=[];tests=[]
        for number,name in enumerate(g.NAMES,1):
            record=g.generate_routine(source,precision,name,number)
            original=record['original'];sig=re.match(r'  subroutine \w+\s*\(([^\n]+)\)',original)[1]
            args=[s.strip() for s in sig.split(',')];n=record['topology'];np={1:0,2:1,3:3,4:6}[n]
            outputs=args[:4] if n==2 else args[:1];inputs=args[len(outputs):]
            ps=inputs[:-n];ms=inputs[-n:];complex_mass=record['complex_masses']
            stub='  subroutine '+name+'('+','.join(args)+')\n'
            stub+='    complex(kindr2),intent(out) :: '+','.join(a+'(0:2)' for a in outputs)+'\n'
            if ps:stub+='    real(kindr2),intent(in) :: '+','.join(ps)+'\n'
            stub+=('    complex(kindr2)' if complex_mass else '    real(kindr2)')+',intent(in) :: '+','.join(ms)+'\n'
            stub+='    mock_calls=mock_calls+1\n    mock_last_args=0\n'
            stub+=f'    mock_nargs={len(inputs)}\n'
            for i,a in enumerate(inputs,1):stub+=f'    mock_last_args({i})={a}\n'
            for a in outputs:
                stub+=f'    {a}=cmplx(991,0,kind=kindr2)\n'
                stub+=f'    if (mock_nan) {a}=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)\n'
            stub+='  end subroutine\n';stubs.append(stub)
            routines+=[record['preflight'],record['checked']]
            callargs=[f'r(:,{i})' for i in range(1,len(outputs)+1)]+[f'p({i})' for i in range(1,np+1)]+[
                f'cm({i})' if complex_mass else f'rm({i})' for i in range(1,n+1)]+['report']
            call='    call cms_checked_'+name+'('+','.join(callargs)+')\n'
            tests.append(f'    ! {precision} {name}: accepted callback receives original inputs.\n')
            tests.append('    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel\n    before=mock_calls\n')
            tests.append(call)
            tests.append('    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)\n')
            tests.append('    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)\n')
            expected=[f'cmplx(p({i}),kind=kindr2)' for i in range(1,np+1)]+[
                f'cm({i})' if complex_mass else f'cmplx(rm({i}),kind=kindr2)' for i in range(1,n+1)]
            tests.append(f'    call insist(mock_nargs.eq.{len(inputs)},103)\n')
            for i,a in enumerate(expected,1):tests.append(f'    call insist(mock_last_args({i}).eq.{a},104)\n')
            tests.append('    do i=1,6\n      p(i)=real(i,kindr2)*onshellthrs/20\n    enddo\n')
            tests.append('    do i=1,4\n      rm(i)=real(i+6,kindr2)*onshellthrs/20\n      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)\n    enddo\n')
            tests.append('    r=sentinel;before=mock_calls\n    p_before=p;rm_before=rm;cm_before=cm\n')
            tests.append('    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)\n    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)\n')
            tests.append(call)
            tests.append(f'    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.{record["max_events"]},105)\n')
            tests.append('    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)\n')
            tests.append('    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)\n')
            tests.append('    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)\n')
            tests.append('    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)\n')
            tests.append('    call insist(report%dropped.eq.0,108)\n    do i=1,report%count\n')
            tests.append('      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)\n')
            tests.append(f'      call insist(report%events(i)%precision.eq.{kind}.and.report%events(i)%topology.eq.{n},110)\n')
            tests.append('      call insist(report%events(i)%source_line.gt.0,111)\n    enddo\n')
        extra='''    ! Strict inequality, exact zeros and real/complex norm distinctions.
    r=sentinel;before=mock_calls
    call cms_checked_a0_r(r(:,1),onshellthrs,report)
    call insist(report%status.eq.CMS_OK.and.mock_calls.eq.before+1,120)
    call cms_checked_a0_r(r(:,1),nearest(onshellthrs,-1._kindr2),report)
    call insist(report%status.eq.CMS_REFUSED.and..not.report%native_called,137)
    call cms_checked_a0_r(r(:,1),nearest(onshellthrs,1._kindr2),report)
    call insist(report%status.eq.CMS_OK.and.report%native_called,138)
    call cms_checked_a0_r(r(:,1),0._kindr2,report)
    call insist(report%status.eq.CMS_OK.and.report%count.eq.0,121)
    cm(1)=cmplx(0.6_kindr2*onshellthrs,-0.6_kindr2*onshellthrs,kind=kindr2)
    call cms_checked_a0_c(r(:,1),cm(1),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%events(1)%metric_norm.eq.CMS_ABS_COMPLEX,122)
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),-10._kindr2,cm(1),cmplx(10,0,kindr2),report)
    call insist(report%status.eq.CMS_OK,123)
    call cms_checked_a0_c(r(:,1),cmplx(0,-onshellthrs/2,kindr2),report)
    call insist(report%status.eq.CMS_REFUSED.and.aimag(report%events(1)%original_value).ne.0,139)
    ! The actual C difference metric is modulus; D uses Cartesian L1.
    p=-10;rm=(/1._kindr2,2._kindr2,3._kindr2,4._kindr2/);cm=cmplx(rm,kind=kindr2)
    p(1)=rm(2)+0.6_kindr2*onshellthrs;cm(2)=cmplx(rm(2),-0.6_kindr2*onshellthrs,kindr2)
    call cms_checked_c0rc(r(:,1),p(1),p(2),p(3),cm(1),cm(2),cm(3),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.1,140)
    call insist(report%events(1)%metric_kind.eq.CMS_DIFFERENCE.and.report%events(1)%metric_norm.eq.CMS_ABS_COMPLEX,141)
    call cms_checked_d0rc(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),cm(1),cm(2),cm(3),cm(4),report)
    call insist(report%status.eq.CMS_OK.and.report%count.eq.0,142)
    if (ieee_support_denormal(0._kindr2)) then
      cm(1)=cmplx(nearest(0._kindr2,1._kindr2),0,kindr2)
      call cms_checked_a0_c(r(:,1),cm(1),report)
      call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.1,143)
      call insist(real(report%events(1)%classified_value,16).ne.0,144)
    else
      print *, 'SUBNORMAL_FIXTURE_UNSUPPORTED',kindr2
    endif
    ! Force the erasure-observer boundary with an underflowed zero metric:
    ! this verifies a nonzero actual complex value cannot be silently skipped.
    call cms_reset(report,kindr2,1,1)
    call cms_record(report,1,CMS_MASS,CMS_ABS_COMPLEX,0,1,'am',0._kindr2, &
      cmplx(tiny(0._kindr2),0,kindr2),cmplx(tiny(0._kindr2),0,kindr2),onshellthrs,1._kindr2)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.1.and.report%events(1)%metric_underflowed,145)
    ! All-zero triangle/box are explicit unsupported results; A/B exact-zero remains delegated.
    p=0;rm=0;r=sentinel;before=mock_calls
    call cms_checked_c0rr(r(:,1),p(1),p(2),p(3),rm(1),rm(2),rm(3),report)
    call insist(report%status.eq.CMS_ZERO_TOPOLOGY_UNSUPPORTED.and.mock_calls.eq.before,124)
    call cms_checked_d0rr(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4),report)
    call insist(report%status.eq.CMS_ZERO_TOPOLOGY_UNSUPPORTED.and.all(r.eq.sentinel),125)
    nonzerothrs=.false.
    call cms_checked_a0_r(r(:,1),1._kindr2,report)
    call insist(report%status.eq.CMS_DEFAULT_UNSUPPORTED.and..not.report%native_called,126)
    nonzerothrs=.true.;initz=.true.
    call cms_checked_a0_r(r(:,1),1._kindr2,report)
    call insist(report%status.eq.CMS_UNINITIALIZED.and..not.report%native_called,127)
    initz=.false.
    ! Opposite mass positions exercise the exact triangle permutation table.
    p=-10;rm=0;rm(1)=20
    call cms_preflight_c0rr(report,p(1),p(2),p(3),rm(1),rm(2),rm(3))
    call insist(all(report%p_sources(1:3).eq.(/2,3,1/)),128)
    call insist(all(report%mass_sources(1:3).eq.(/2,3,1/)),129)
    ! Soft box routing jj=2 when the 1/3 channel dominates 2/4 and 5/6.
    p=(/10._kindr2,2._kindr2,10._kindr2,2._kindr2,0._kindr2,0._kindr2/);rm=20
    call cms_preflight_d0rr(report,p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4))
    call insist(report%soft_permutation.eq.2,130)
    call insist(all(report%p_sources.eq.(/5,2,6,4,1,3/)),131)
    ! A forced observer overflow is still a refusal, never a success.
    call cms_reset(report,kindr2,4,7)
    do i=1,CMS_MAX_EVENTS+1
      call cms_record(report,1,CMS_INPUT,CMS_ABS_REAL,1,0,'forced',onshellthrs/2, &
        cmplx(onshellthrs/2,0,kindr2),cmplx(onshellthrs/2,0,kindr2),onshellthrs,1._kindr2)
    enddo
    call insist(report%status.eq.CMS_EVENT_OVERFLOW.and.report%count.eq.CMS_MAX_EVENTS.and.report%dropped.eq.1,132)
    call insist(.not.report%result_valid.and..not.report%native_called,133)
    ! Native callbacks with nonfinite output must not publish caller buffers.
    r=sentinel;before=mock_calls;mock_nan=.true.
    call cms_checked_a0_r(r(:,1),20._kindr2,report)
    call insist(report%status.eq.CMS_NONFINITE_RESULT.and.report%native_called,134)
    call insist(.not.report%result_valid.and.all(r.eq.sentinel).and.mock_calls.eq.before+1,135)
    mock_nan=.false.
    ! Finite input whose native L1 norm overflows is refused before callback.
    cm(1)=cmplx(huge(0._kindr2),-huge(0._kindr2),kind=kindr2);before=mock_calls
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),-10._kindr2,cm(1),cmplx(10,0,kindr2),report)
    call insist(report%status.eq.CMS_BAD_INPUT.and..not.report%native_called.and.mock_calls.eq.before,136)
'''
        test='  subroutine fixture_check()\n    type(cms_scalar_report)::report\n'
        test+='    real(kindr2)::p(6),rm(4),p_before(6),rm_before(4)\n'
        test+='    complex(kindr2)::cm(4),r(0:2,4),sentinel,cm_before(4)\n    integer::i,before\n'
        test+='    integer(8)::p_words(6*kindr2/8),rm_words(4*kindr2/8),cm_words(8*kindr2/8),r_words(24*kindr2/8)\n'
        test+='    sentinel=cmplx(1234,5678,kind=kindr2)\n'+''.join(tests)+extra
        if kind==16:
            test+='    ! Exact retained physical-transfer and external-rounding literals (not assumed preflight outcomes).\n'
            for token in ('-2.499999999999791305007018193257151140E-7','-2.643850145484542582671511495203734939E-12'):
                test+='    p(1)='+token+'_16\n'
                test+='    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),p(1),cmplx(20,0,16),cmplx(30,0,16),report)\n'
                test+='    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.1,146)\n'
                test+='    call insist(real(report%events(1)%original_value,16).eq.p(1),147)\n'
        test+=f"    write(*,*) 'CLASSIFIER_MOCK_COMPLETE',kindr2,8\n  end subroutine\n"
        test+='  subroutine insist(ok,code)\n    logical,intent(in)::ok\n    integer,intent(in)::code\n'
        test+="    if (.not.ok) then\n      write(*,*) 'MOCK_FAILURE',kindr2,code\n      error stop 1\n    endif\n  end subroutine\n"
        modules.append(prefix+'\n'.join(stubs+routines)+test+'end module\n')
    (out/'tables.f90').write_text('\n'.join(tables))
    (out/'mock_hosts.f90').write_text('\n'.join(modules))
    driver='program classifier_mock\n  use cms_mock_dp,only:check_dp=>fixture_check\n  use cms_mock_qp,only:check_qp=>fixture_check\n'
    driver+='  implicit none\n  if (precision(0._16).lt.33) error stop 2\n  call check_dp()\n  call check_qp()\n'
    driver+="  print *, 'COMPLETE_CLASSIFIER_MOCK_ONLY'\nend program\n"
    (out/'driver.f90').write_text(driver)
    compiler='/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran'
    build=HERE/'build';build.mkdir(exist_ok=False)
    actual=HERE/'actual-build';actual.mkdir(exist_ok=False)
    jobs=[]
    for name,path in [('observer',HERE/'cms_scalar_observer.f90'),('tables',out/'tables.f90'),('mock_hosts',out/'mock_hosts.f90'),('driver',out/'driver.f90')]:
        jobs.append({'name':'compile-'+name,'argv':[compiler,'-O0','-g','-fcheck=all','-ffree-line-length-none',
            '-J'+str(build),'-I'+str(build),'-c',str(path),'-o',str(build/(name+'.o'))],'cwd':str(ROOT)})
    jobs.append({'name':'link-mock','argv':[compiler,*[str(build/(name+'.o')) for name in ['observer','tables','mock_hosts','driver']],
                 '-o',str(build/'classifier_mock')],'cwd':str(ROOT)})
    jobs.append({'name':'compile-real-AVH-additions','argv':[compiler,'-O0','-ffree-line-length-none','-J'+str(actual),
        '-I'+str(actual),'-I'+str(build),'-I'+str(ROOT/'vendor/CutTools/includects'),'-c',str(HERE/'generated/avh_olo_checked.f90'),
        '-o',str(actual/'avh_olo_checked.o')],'cwd':str(ROOT)})
    plan={'schema':'cms-classifier-source-only-compile-plan-v1','state':'not_authorized_or_executed','jobs':jobs,
        'global_compile_link_seconds_max':120,'cpu':8,'jobs_concurrency':1,'automatic_retry':False,
        'mock_execution_requires_separate_authorization':True,
        'mock_execution_argv':[str(build/'classifier_mock')],'mock_execution_seconds_max':15,
        'actual_AVH_native_execution_allowed':False,
        'compiler':{'path':compiler,'sha256':sha(compiler)},
        'dependency_sha256':{str(ROOT/'vendor/CutTools/includects/mpmodule.mod'):sha(ROOT/'vendor/CutTools/includects/mpmodule.mod')},
        'fixture_sha256':{str(p):sha(p) for p in sorted(out.iterdir())}}
    (HERE/'compile-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'compile_jobs':5,'link_jobs':1,'mock_source_overloads':16,'compiler_or_native_started':False}))


if __name__=='__main__':main()
