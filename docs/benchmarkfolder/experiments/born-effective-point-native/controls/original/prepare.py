#!/usr/bin/env python3
"""Prepare exact generated Fortran control fixtures; never compile or run them."""
from pathlib import Path
import difflib
import hashlib
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
IMPLEMENTATION=ROOT/'.tirust-build/born-effective-implementation'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def annotation(name,body):
    return 'C RUNTIME-OBSERVER-BEGIN '+name+'\n'+body+'C RUNTIME-OBSERVER-END '+name+'\n'


def observe(aggregate,prefix,once):
    """Read-only fixture ENTRY; no assignment into original learning state."""
    decl=annotation('DECLARATION',
        '      INTEGER FX_OBS_NTRY(NFLAV)\n'
        '      LOGICAL FX_OBS_GOODHEL(NCOMB,NFLAV),FX_OBS_HELRESET\n')
    body=annotation('ENTRY',
        '      RETURN\n'
        '      ENTRY '+prefix+'OBSERVE_LEARNING(FX_OBS_NTRY,\n'
        '     $ FX_OBS_GOODHEL,FX_OBS_HELRESET)\n'
        '      FX_OBS_NTRY=NTRY\n'
        '      FX_OBS_GOODHEL=GOODHEL\n'
        '      FX_OBS_HELRESET=HELRESET\n'
        '      RETURN\n')
    result=once(aggregate,'      INTEGER '+prefix+'BROKEN_SYM\n',
                '      INTEGER '+prefix+'BROKEN_SYM\n'+decl)
    result=once(result,'      END\n',body+'      END\n')
    assert result.replace(decl,'').replace(body,'')==aggregate
    return result


def born_stubs(n,split):
    code='''module fx_control
  implicit none
  integer :: fx_calls=0,fx_symmetry=1
  integer :: fx_sequence(32)=0
  real(8) :: fx_values(3,4)=0d0
  logical :: fx_selected(4)=.true.
end module
'''
    for prefix in ['TX_','LEG_']:
        if split:
            code+='''subroutine @MATRIX(p,nhel,jc,flav_idx,t)
  use fx_control
  implicit none
  real(8) :: p(0:3,N),t(3)
  integer :: nhel(N),jc(N),flav_idx,h
  h=nhel(N)
  fx_calls=fx_calls+1
  fx_sequence(fx_calls)=h
  t=fx_values(:,h)
end subroutine
'''.replace('@',prefix).replace('N',str(n))
        else:
            code+='''real(8) function @MATRIX(p,nhel,jc,flav_idx)
  use fx_control
  implicit none
  real(8) :: p(0:3,N)
  integer :: nhel(N),jc(N),flav_idx,h
  h=nhel(N)
  fx_calls=fx_calls+1
  fx_sequence(fx_calls)=h
  @MATRIX=fx_values(1,h)
end function
'''.replace('N',str(n)).replace('@',prefix)
        code+='''subroutine @GET_FLAVOR(flav_idx,flavor)
  implicit none
  integer :: flav_idx,flavor(N)
  flavor=flav_idx
end subroutine
logical function @IS_BORN_HEL_SELECTED(h)
  use fx_control
  implicit none
  integer :: h
  @IS_BORN_HEL_SELECTED=fx_selected(h)
end function
integer function @BROKEN_SYM(flavor)
  use fx_control
  implicit none
  integer :: flavor(N)
  @BROKEN_SYM=fx_symmetry
end function
'''.replace('(N)', '('+str(n)+')').replace('@',prefix)
    return code


def main():
    transaction=load('frozen_learning_transaction',IMPLEMENTATION/'learning-transaction/generate.py')
    repair=load('frozen_checked_repair',IMPLEMENTATION/'checked-repair/generate.py')
    assert transaction.REVISION==2
    sys.path.insert(0,str(ROOT))
    from madgraph.iolibs.file_writers import FortranWriter
    sys.path.pop(0)
    inputs={}
    for part in ['learning-transaction','checked-repair']:
        directory=IMPLEMENTATION/part
        manifest=json.loads((directory/'artifacts/manifest.json').read_text())
        assert sha((directory/'generate.py').read_bytes())==manifest['generator_sha256']
        for name in ['generate.py','inputs/source.json','artifacts/manifest.json']:
            path=directory/name
            inputs[str(path.relative_to(ROOT))]=sha(path.read_bytes())
    for path in [ROOT/'madgraph/iolibs/file_writers.py',HERE/'born_driver.f90.in',
                 HERE/'repair_driver.f90',HERE/'repair_stubs.f90',
                 HERE/'README.md',HERE/'validate_sources.py']:
        inputs[str(path.relative_to(ROOT))]=sha(path.read_bytes())
    outputs={};proof=[];jobs=[];expected={}
    _,sources=transaction.source_inputs()
    case_names=['finite_same_bits_and_learning','invalid_flavor_zero',
        'fixed_helicity_success','raw_nan','raw_infinity','excluded_helicity_nan',
        'beam_weight_overflow','partial_sum_overflow','final_normalization_overflow',
        'genuine_zero']
    for split in [False,True]:
        source=sources[transaction.NAMES[split]]
        checked=transaction.transform(source,split)
        aggregate=transaction.extract_aggregate(checked,split)
        original=transaction.extract_aggregate(source,split)
        old_wrappers=source[:source.index(original)]
        for n in [3,5]:
            variant=('split' if split else 'scalar')+'_n'+str(n)
            params=transaction.parameters(n,split)
            params['proc_prefix']='TX_'
            raw=aggregate%params
            observed=observe(raw,'TX_',transaction.once)
            params['proc_prefix']='LEG_'
            legacy=original%params
            observed_legacy=observe(legacy,'LEG_',transaction.once)
            generated=observed+'\n'+transaction.wrapper(split)%dict(params,proc_prefix='TX_')
            generated+=observed_legacy+'\n'+old_wrappers%params
            outputs[variant+'/aggregate.raw.f']=generated
            proof.append({'variant':variant,'checked_before_observer_sha256':sha(raw.encode()),
                'legacy_before_observer_sha256':sha(legacy.encode()),
                'checked_before_observer_bytes':len(raw.encode()),
                'legacy_before_observer_bytes':len(legacy.encode()),
                'observer_is_only_addition_and_strips_exactly':True})
            outputs[variant+'/observer.patch']=''.join(difflib.unified_diff(
                raw.splitlines(True),observed.splitlines(True),
                fromfile='checked-aggregate.f',tofile='observed-checked-aggregate.f'))
            outputs[variant+'/stubs.f90']=born_stubs(n,split)
            replacements={'@N@':str(n),'@NS@':'3' if split else '1',
                '@SPLIT@':'.true.' if split else '.false.',
                '@FIXED_EXPECTED@':'[8d0,2d0,4d0,6d0]' if split else '[2d0,2d0]',
                '@FINITE_EXPECTED@':'[10d0,2.5d0,5d0,7.5d0]' if split else '[2.5d0,2.5d0]',
                '@LEGACY_CALL@':('call LEG_SMATRIXHEL_SPLITORDERS(p,-1,1,ans_old)' if split else
                    'call LEG_SMATRIXHEL(p,-1,1,ans_old(1))\n  ans_old(0)=ans_old(1)'),
                '@EXTRA_BUFFER_CASE@':('''call prepare_rejection()
  fx_values=0d0
  fx_values(1:2,1)=huge(1d0)*0.75d0
  call reject_case('learning_buffer_overflow',1)''' if split else '')}
            driver=(HERE/'born_driver.f90.in').read_text()
            for key,value in replacements.items():driver=driver.replace(key,value)
            assert '@' not in driver
            outputs[variant+'/driver.f90']=driver
            expected[variant]=case_names[:-1]+(['learning_buffer_overflow'] if split else [])+case_names[-1:]
    _,repair_sources=repair.source_inputs()
    old,checked=repair.checked_wrapper(repair_sources['improve_ps.inc'])
    params=repair.fixture_parameters('runtime',['ZERO']*5)
    params['proc_prefix']='RX_'
    checked=checked%params
    # Include only exact checked wrapper and exact PREPARE host. All numerical
    # repair routines are explicit opaque stubs, not extracted real algorithms.
    prepare=repair.HOST.split('C OPTIONAL ALGORITHM-ONLY SETTER.',1)[0]%params
    outputs['repair/wrapper.raw.f']=checked+'\n'+prepare
    outputs['repair/MadLoopParams.inc']='      INTEGER ImprovePSPoint\n      COMMON/FX_REPAIR_MODE/ImprovePSPoint\n'
    outputs['repair/stubs.f90']=(HERE/'repair_stubs.f90').read_text()
    outputs['repair/driver.f90']=(HERE/'repair_driver.f90').read_text()
    expected['repair']=['psmc_primary','orig_primary','mode_zero','mode_negative',
        'psmc_to_orig','orig_to_psmc','both_failed','nonphysical','unsupported']
    proof.append({'variant':'repair','checked_wrapper_sha256':sha(checked.encode()),
                  'prepare_host_sha256':sha(prepare.encode()),'wrapper_and_prepare_exact':True})
    for name,text in list(outputs.items()):
        path=HERE/'sources'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)
        if name.endswith('.raw.f'):
            formatted=path.with_name(path.name.replace('.raw.f','.f'))
            with FortranWriter(str(formatted)) as writer:writer.writelines(text)
            outputs[str(formatted.relative_to(HERE/'sources'))]=formatted.read_text()
    for variant in expected:
        work=HERE/'work'/variant
        source=HERE/'sources'/variant
        aggregate='wrapper.f' if variant=='repair' else 'aggregate.f'
        executable=work/'control.exe'
        compile_argv=['timeout','30s','taskset','-c','14','gfortran','-O0','-g',
            '-fcheck=all','-fbacktrace','-fno-fast-math','-ffixed-line-length-none',
            '-ffree-line-length-none','-J',str(work),'-I',str(work),'-I',str(source),
            str(source/'stubs.f90'),str(source/aggregate),str(source/'driver.f90'),'-o',str(executable)]
        jobs.append({'name':variant,'working_directory':str(ROOT),
            'create_directories':[str(work)],'compile_argv':compile_argv,
            'run_argv':['timeout','15s','taskset','-c','14',str(executable)],
            'compile_log':str(work/'compile.log'),'run_log':str(work/'run.log'),
            'expected_exit_code':0,'scope':'controlled standalone Fortran, no reducers'})
    (HERE/'expected.json').write_text(json.dumps({'schema':1,'cases':expected,
        'finite_sum_values':{'scalar':[2.5,2.5],'split':[10.0,2.5,5.0,7.5]},
        'fixed_helicity_two_values':{'scalar':[2.0,2.0],'split':[8.0,2.0,4.0,6.0]},
        'repair_published_values':{'PSMC_success':101,'ORIG_success':202,'unadopted_C_sentinel':-99},
        'stdout_rule':'Each listed case must appear exactly once as PASS <name>; each job must finish with ALL PASS and exit 0. Repair warnings are allowed. Any FAIL, missing/duplicate PASS, nonzero/timeout is failure.',
        'status':'prepared_not_executed'},indent=2)+'\n')
    (HERE/'compiler-jobs.json').write_text(json.dumps({'schema':1,
        'authorized_execution_owner':'parent complex_stress; shared 600-second CPU14 budget',
        'prepared_only':True,'total_per_job_timeout_caps_seconds':225,
        'no_reducer_dependencies':True,'jobs':jobs},indent=2)+'\n')
    (HERE/'manifest.json').write_text(json.dumps({'schema':1,'prepared_only':True,
        'generator_sha256':sha(Path(__file__).read_bytes()),'inputs':inputs,'proof':proof,
        'files':{str((Path('sources')/name)):sha(text.encode()) for name,text in sorted(outputs.items())},
        'expected_sha256':sha((HERE/'expected.json').read_bytes()),
        'compiler_jobs_sha256':sha((HERE/'compiler-jobs.json').read_bytes())},indent=2)+'\n')
    print('Prepared five exact Fortran control fixtures; no compiler or native execution.')


if __name__=='__main__':main()
