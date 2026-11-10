#!/usr/bin/env python3
"""Generate private caller-boundary sources and commands, without native work."""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COMBINED = HERE.parent
ROOT = COMBINED.parents[2]
SOURCE = COMBINED/'src'
sys.path.insert(0, str(ROOT))
import aloha
from madgraph.core import base_objects, helas_objects
from madgraph.loop import loop_helas_objects
from madgraph.iolibs import file_writers

COMPILER = '/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran'
PYTHON = '/nix/store/463ik27ppxhpbvw7c50s7qcfc7mgfa5x-python3-3.12.12-env/bin/python3.12'
WRITER = Path('madgraph/iolibs/helas_call_writers.py')
TEMPLATE = Path('madgraph/iolibs/template_files/loop_optimized/helas_calls_split.inc')
LEGACY_SOURCE = COMBINED.parent/'attempt1/base'/WRITER
MANIFEST_SHA = '276f562eeaf0ffb150ae864e9ee7a5a281f56b18c8a48038afc05ef012c61292'
REFRESH=False


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not REFRESH:
        assert path.read_text() == text, 'Refusing to overwrite changed generated source: '+str(path)
    else:
        path.write_text(text)


def render(text, context=None):
    with tempfile.TemporaryDirectory(prefix='render-',dir=HERE) as directory:
        output=Path(directory)/'render.f'
        with file_writers.FortranWriter(str(output)) as writer:
            writer.writelines(text,context=context or {})
        return output.read_text()


def amplitude(masses, widths=None, pairing=None, number=1, group=0):
    widths = widths or ['ZERO']*len(masses)
    pairing = pairing or [1]*len(masses)
    seed = helas_objects.HelasWavefunction({'number': 1, 'is_loop': True})
    lines = [helas_objects.HelasWavefunction({
        'particle': base_objects.Particle({'mass': mass, 'width': width}),
        'is_loop': True, 'mothers': helas_objects.HelasWavefunctionList([seed])})
        for mass,width in zip(masses,widths,strict=True)]
    return loop_helas_objects.LoopHelasAmplitude({
        'wavefunctions': helas_objects.HelasWavefunctionList([seed,seed]+lines),
        'mothers': helas_objects.HelasWavefunctionList([
            helas_objects.HelasWavefunction({'number': 11+i}) for i in range(sum(pairing))]),
        'amplitudes': helas_objects.HelasAmplitudeList([
            helas_objects.HelasAmplitude({'number': number})]),
        'pairing': pairing, 'number': number, 'loop_group_id': group})


def generate_calls(module, route):
    a = amplitude(['ZERO','MDL_REAL'],number=7,group=3)
    duplicate = amplitude(['ZERO','MDL_REAL'],number=9,group=3)
    b = amplitude(['MDL_REAL','MDL_DERIVED','MDL_CMS'],
                  widths=['ZERO','ZERO','MDL_WIDTH'],number=11,group=7)
    c = amplitude(['MDL_DERIVED','MDL_CMS'],widths=['ZERO','MDL_WIDTH'],
                  pairing=[1,2],number=13,group=11)
    amps = [a,duplicate,b,c]
    diagram = loop_helas_objects.LoopHelasDiagram({
        'amplitudes': helas_objects.HelasAmplitudeList(amps)})
    matrix = loop_helas_objects.LoopHelasMatrixElement({
        'processes': base_objects.ProcessList([
            base_objects.Process({'has_born': route!='loop_induced'})]),
        'diagrams': helas_objects.HelasDiagramList([diagram])})
    matrix['loop_groups'] = [((),[a,duplicate]),((),[b]),((),[c])]
    writer = module.FortranUFOHelasCallWriterOptimized(base_objects.Model())
    before = [a.get('loop_group_id') for a in amps]
    with mock.patch.object(aloha,'complex_mass',True), mock.patch.object(
            loop_helas_objects.LoopHelasAmplitude,'get_analytic_info',return_value=2):
        calls = writer.get_loop_CT_calls(matrix,group_loops=route!='ungrouped')
    assert before == [a.get('loop_group_id') for a in amps]
    return calls


def metadata(call):
    assert call.startswith('CALL FX_LOOP_')
    name,args = call[5:].split('(',1)
    # Nested DCMPLX/CMPLX comma arguments are parsed without evaluating expressions.
    tokens=[];start=0;depth=0
    args=args[:-1]
    for i,c in enumerate(args):
        if c=='(': depth+=1
        if c==')': depth-=1
        if c==',' and depth==0:
            tokens.append(args[start:i]);start=i+1
    tokens.append(args[start:])
    family=name.removeprefix('FX_LOOP_').split('_')
    n=int(family[0]);nmother=int(family[1]) if len(family)>1 else n
    offset=n if n!=nmother else 0
    pairing=list(map(int,tokens[:n])) if offset else [1]*n
    mothers=list(map(int,tokens[offset:offset+nmother]))
    masses=[s.removeprefix('DCMPLX(').removesuffix(')')
            for s in tokens[offset+nmother:offset+nmother+n]]
    rank,sqso,group=(1 if x=='I_SO' else int(x) for x in tokens[-3:])
    return dict(helper=name,n=n,npair=n,nmother=nmother,pairing=pairing,
                mothers=mothers,masses=masses,rank=rank,sqso=sqso,group=group,
                legacy_call=call)


def helper(family, paired):
    n,nmother=family
    name='FX_LOOP_'+str(n)+(('_'+str(nmother)) if nmother!=n else '')
    if paired: name+='_QPMASS'
    pairs=['P%d'%i for i in range(1,n+1)] if nmother!=n else []
    mothers=['I%d'%i for i in range(1,nmother+1)]
    masses=['M%d'%i for i in range(1,n+1)]
    mpmasses=['MP_M%d'%i for i in range(1,n+1)] if paired else []
    tail=['RANK','SQUAREDSOINDEX','LOOPNUM']
    mass_actuals=[v for pair in zip(masses,mpmasses,strict=True) for v in pair] if paired else masses
    text='      SUBROUTINE '+name+'('+','.join(pairs+mothers+mass_actuals+tail)+')\n'
    text+='      USE FIXTURE_RECORDING, ONLY: RECORD_DP,RECORD_QP\n      IMPLICIT NONE\n'
    text+='      INTEGER '+','.join(pairs+mothers+tail)+'\n'
    text+='      COMPLEX*16 '+','.join(masses)+'\n'
    if paired: text+='      COMPLEX*32 '+','.join(mpmasses)+'\n'
    args=[str(n),str(n),str(nmother),'(/'+','.join(pairs or ['1']*n)+'/)',
          '(/'+','.join(mothers)+'/)','RANK','SQUAREDSOINDEX','LOOPNUM',
          '(/'+','.join(masses)+'/)']
    if paired: args.append('(/'+','.join(mpmasses)+'/)')
    text+='      CALL '+('RECORD_QP' if paired else 'RECORD_DP')+'('+','.join(args)+')\n      END\n'
    return text


def main():
    assert Path.cwd().resolve()==ROOT
    assert sha(COMBINED/'source-manifest.json')==MANIFEST_SHA
    pinned=json.loads((COMBINED/'source-manifest.json').read_text())
    for path,item in pinned['sources'].items(): assert sha(SOURCE/path)==item['sha256']
    assert sha(LEGACY_SOURCE)=='7a267fa78b75fa5b8efc4fc2f595769f938ee59919f55c6fffc3ab86ec5707ea'
    bindings={rel(COMBINED/'source-manifest.json'):MANIFEST_SHA,
              rel(LEGACY_SOURCE):sha(LEGACY_SOURCE)}
    bindings.update({rel(SOURCE/path):item['sha256'] for path,item in pinned['sources'].items()})
    legacy=load('fixture_legacy_writer',LEGACY_SOURCE)
    guarded=load('fixture_guarded_writer',SOURCE/WRITER)
    exporter_module=load('fixture_composed_exporter',SOURCE/'madgraph/loop/loop_exporters.py')
    exporter=object.__new__(exporter_module.LoopProcessOptimizedExporterFortranSA)
    exporter.template_dir=str(SOURCE/TEMPLATE.parent)
    template=(SOURCE/TEMPLATE).read_text()
    mapping={key:1 if fmt=='d' else '' for key,fmt in re.findall(r'%\((\w+)\)([ds])',template)}
    mapping.update(proc_prefix='FX_',real_dp_format='REAL*8',real_mp_format='REAL*16',
                   complex_dp_format='COMPLEX*16',complex_mp_format='COMPLEX*32',
                   nexternal=3,ncomb=1,nbornamps=1,nloops=13,nloop_groups=13,
                   nctamps=13,nloopamps=13,nwavefuncs=20,nloopwavefuncs=1,
                   nSO=1,nSquaredSO=1,nAmpSO=1)
    src=HERE/'src';src.mkdir(exist_ok=True)
    dispatch=[];matrix={};families=set();rendered=[]
    for arm,module in [('legacy',legacy),('guarded',guarded)]:
        dispatch.append(('    if' if arm=='legacy' else '    else if')+"(trim(arm)=='%s') then"%arm)
        dispatch.append('      select case(trim(route))')
        for route in ('grouped','ungrouped','loop_induced'):
            calls=[s%mapping for s in generate_calls(module,route)]
            old=[s%mapping for s in generate_calls(legacy,route)]
            expected=[]
            for a,b in zip(calls,old,strict=True):
                if b.startswith('CALL '):
                    if arm=='guarded':
                        lines=a.splitlines();assert len(lines)==5
                        assert lines[0]=='IF(CTMODE.LT.4) THEN' and lines[1]==b
                        assert lines[2]=='ELSE' and lines[4]=='ENDIF'
                        assert '_QPMASS(' in lines[3] and 'KIND=16' in lines[3]
                    item=metadata(b);expected.append(item);families.add((item['n'],item['nmother']))
                else: assert a==b
            matrix[arm+'_'+route]=dict(items=calls,expected=expected)
            rep=dict(mapping)
            context={'LoopInduced':route=='loop_induced','AmplitudeReduction':route=='loop_induced'}
            text=exporter.split_HELASCALLS(None,rep,TEMPLATE.name,'',calls,'loop_CT_calls',
                arm+'_'+route,n_helas=2,context=context)
            path=src/(arm+'_'+route+'.f')
            result=render(text,context)
            assert all(len(line)<=132 for line in result.splitlines())
            write(path,result);rendered.append(path)
            dispatch.append("      case('%s')"%route)
            dispatch.extend('        '+s for s in rep['loop_CT_calls'].splitlines())
        dispatch.append('      end select')
    dispatch.append('    endif')
    assert [x['group'] for x in matrix['legacy_grouped']['expected']]==[4,8,12]
    for route in ('ungrouped','loop_induced'):
        assert [x['group'] for x in matrix['legacy_'+route]['expected']]==[7,9,11,13]
    driver=(HERE/'driver-template.f90').read_text().replace('!@GENERATED_DISPATCH@','\n'.join(dispatch))
    assert '!@GENERATED_DISPATCH@' not in driver
    write(src/'driver.f90',driver)
    helper_text='\n'.join(helper(family,paired) for family in sorted(families) for paired in (False,True))
    write(src/'helpers.f',render(helper_text))
    write(src/'recording.f90',(HERE/'recording.f90').read_text())
    aloha_path=ROOT/'aloha/template_files/aloha_functions_loop.f'
    bindings[rel(aloha_path)]=sha(aloha_path)
    aloha_text=aloha_path.read_text();end=re.search(r'(?im)^\s*end module ALOHA_OBJECT\s*$',aloha_text)
    assert end
    write(src/'aloha_object.f',aloha_text[:end.end()]+'\n')
    write(src/'polynomial_constants.f90','module FX_POLYNOMIAL_CONSTANTS\n  implicit none\n  integer,parameter :: MAXLWFSIZE=4,VERTEXMAXCOEFS=5,LOOPMAXCOEFS=35\nend module\n')
    write(src/'coupl.inc','      REAL*8 MDL_REAL,MDL_DERIVED,MDL_CMS,MDL_WIDTH\n      COMMON/FX_MODEL_DP/MDL_REAL,MDL_DERIVED,MDL_CMS,MDL_WIDTH\n      COMPLEX*16 CMASS_MDL_CMS\n      COMMON/FX_MODEL_CMS/CMASS_MDL_CMS\n')
    write(src/'mp_coupl.inc','      REAL*16 MP__MDL_REAL,MP__MDL_DERIVED\n      COMMON/FX_MODEL_MP/MP__MDL_REAL,MP__MDL_DERIVED\n      COMPLEX*32 MP__CMASS_MDL_CMS\n      COMMON/FX_MODEL_MP_CMS/MP__CMASS_MDL_CMS\n')
    write(HERE/'argument-matrix.json',json.dumps(matrix,indent=2,sort_keys=True)+'\n')
    build=HERE/'build';build.mkdir(exist_ok=True)
    sources=[src/'aloha_object.f',src/'polynomial_constants.f90',src/'recording.f90',*rendered,src/'helpers.f',src/'driver.f90']
    jobs=[];outputs=[]
    common_flags=['-O0','-fPIC','-fno-fast-math','-fsignaling-nans','-ffixed-line-length-132','-ffree-line-length-none',
                  '-I'+str(src),'-I'+str(build),'-J'+str(build)]
    for source in sources:
        obj=build/(source.stem+'.o');outputs.append(rel(obj))
        jobs.append(dict(name='fixture-'+source.stem,argv=[COMPILER,*common_flags,'-c',str(source),'-o',str(obj)],cwd=rel(build)))
    exe=build/'caller_fixture'
    jobs.append(dict(name='fixture-link',argv=[COMPILER,*[str(ROOT/p) for p in outputs],'-o',str(exe)],cwd=rel(build)))
    outputs.append(rel(exe))
    runs=[]
    def add(arm,mode,payload,trap,route):
        name='fixture-%s-m%s-%s-t%d-%s'%(arm,str(mode).replace('-','minus'),payload,trap,route)
        directory=HERE/'runs'/name;directory.mkdir(parents=True,exist_ok=True)
        case=dict(arm=arm,mode=mode,payload=payload,trap=trap,route=route)
        runs.append(dict(name=name,argv=[str(exe),arm,str(mode),payload,str(trap),route],
                         cwd=rel(directory),timeout_seconds=3,expected_outputs=['caller.hex','stdout.log'],
                         scientific_outputs=['caller.hex'],controller_outputs=['stdout.log'],
                         raw_stdout=True,log_file=rel(directory/'stdout.log'),
                         expected_exit=-8 if arm=='trap_control' and trap else 0,case=case))
    for mode in (-1,0,1,2,3,4,5,6):
        for payload in (('finite_poison','qnan','snan') if mode<4 else ('valid',)):
            for arm in ('legacy','guarded'):
                for trap in (0,1):
                    for route in ('grouped','ungrouped','loop_induced'): add(arm,mode,payload,trap,route)
    for trap in (0,1):add('trap_control',0,'finite_poison',trap,'grouped')
    assert len(runs)==218
    bindings.update({rel(p):sha(p) for p in src.iterdir() if p.is_file()})
    for p in [Path(__file__),HERE/'recording.f90',HERE/'driver-template.f90',HERE/'argument-matrix.json',HERE/'validate.py']:
        bindings[rel(p)]=sha(p)
    package=dict(schema='qp-mass-caller-fixture-v1',scope='Source-only recording-stub caller boundary; no reducer or physical amplitude qualification',
        generated_by_actual_writers=True,native_compile_performed=False,native_execution_performed=False,
        inputs=bindings,compiler=COMPILER,python=PYTHON,build_jobs=jobs,build_outputs=outputs,runs=runs,
        module_outputs=[rel(build/p) for p in ['aloha_object.mod','fx_polynomial_constants.mod','fixture_recording.mod']],
        build_seconds_max_shared_with_ew=180,execution_seconds_max=60,individual_timeout_seconds=3,
        cpu=8,retries=0,required_reducer_libraries=[],
        object_dependencies={'recording.o':[], 'helpers.o':['recording.o'], 'driver.o':['aloha_object.o','recording.o'],
          **{p.stem+'.o':['aloha_object.o','polynomial_constants.o'] for p in rendered}},
        validation=dict(argv=[PYTHON,'-B',str(HERE/'validate.py'),'--events',str(COMBINED/'run.json')],
                        cwd=rel(HERE),timeout_seconds=20),
        interpretation={'results':'Synthetic recording-stub outputs only','status':'Synthetic zero stub status only',
          'missing_mp_initialization':'Explicitly poisoned binary128 storage, not undefined memory',
          'cms':'Synthetic genuine complex mass symbol; not a full physical CMS export'})
    write(HERE/'commands.json',json.dumps(package,indent=2,sort_keys=True)+'\n')
    assert all(sha(ROOT/p)==digest for p,digest in bindings.items())
    print(json.dumps(dict(source_only=True,run_count=len(runs),compile_jobs=len(jobs),
                         commands_sha256=sha(HERE/'commands.json'))))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--refresh-generated',action='store_true',help='Explicitly rewrite only this owned fixture generated files before freezing')
    REFRESH=cli.parse_args().refresh_generated
    with mock.patch.object(subprocess,'Popen',side_effect=AssertionError('Native launch forbidden during fixture preparation')):
        main()
