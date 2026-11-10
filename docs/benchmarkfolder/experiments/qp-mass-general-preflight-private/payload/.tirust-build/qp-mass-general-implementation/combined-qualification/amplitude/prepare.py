#!/usr/bin/env python3
"""Append-only source preparation. No compiler, linker, subprocess or native call."""
import hashlib
import importlib.util
import json
import re
import shutil
import struct
import sys
from pathlib import Path
from unittest import mock

if sys.flags.optimize:
    raise RuntimeError('Source checks require assertions')
sys.dont_write_bytecode = True
ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
COMBINED = HERE.parent
HISTORY = ROOT / 'docs/benchmarkfolder/experiments/ew-qp-mass-prototype'
OBS = HISTORY / 'observer-native-design'
WORK = ROOT / '.tirust-build/ew-qp-mass-prototype/observer-native-design/attempt2'
SUB = WORK / 'original/SubProcesses/P0_epem_ttxa'
TEMPLATE = Path('madgraph/iolibs/template_files/loop_optimized')
sys.path.insert(0, str(ROOT))
import aloha
from madgraph.core import base_objects, helas_objects
from madgraph.loop import loop_helas_objects
from madgraph.iolibs import file_writers


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def amplitude(masses, pairing, number):
    seed = helas_objects.HelasWavefunction({'number': 1, 'is_loop': True})
    lines = [helas_objects.HelasWavefunction({
        'particle': base_objects.Particle({'mass': m, 'width': 'ZERO'}),
        'is_loop': True, 'mothers': helas_objects.HelasWavefunctionList([seed])}) for m in masses]
    return loop_helas_objects.LoopHelasAmplitude({
        'wavefunctions': helas_objects.HelasWavefunctionList([seed, seed] + lines),
        'mothers': helas_objects.HelasWavefunctionList([
            helas_objects.HelasWavefunction({'number': i+11}) for i in range(sum(pairing))]),
        'amplitudes': helas_objects.HelasAmplitudeList([helas_objects.HelasAmplitude({'number': number})]),
        'pairing': pairing, 'number': number, 'loop_group_id': number-1})


def main():
    assert ROOT.name == 'madgraph7_madloop_tirust'
    assert not (HERE / 'prepared.json').exists(), 'Do not overwrite a prepared package'
    manifest = COMBINED / 'source-manifest.json'
    assert sha(manifest) == '276f562eeaf0ffb150ae864e9ee7a5a281f56b18c8a48038afc05ef012c61292'
    source_manifest = json.loads(manifest.read_text())
    external = {str(manifest): sha(manifest)}
    def bind(path):
        path = Path(path)
        assert path.is_file(), str(path)
        external[str(path)] = sha(path)
        return path
    for name, entry in source_manifest['sources'].items():
        assert sha(bind(COMBINED / 'src' / name)) == entry['sha256']
    for name in ('src', 'templates', 'build', 'runs', 'inputs'):
        (HERE/name).mkdir(exist_ok=False)
    templates = HERE/'templates'
    for path in (ROOT/TEMPLATE).iterdir():
        if path.is_file():
            shutil.copyfile(bind(path), templates/path.name)
    for path in (COMBINED/'src'/TEMPLATE).iterdir():
        shutil.copyfile(bind(path), templates/path.name)
    exporter_module = load('combined_amplitude_exporter', bind(COMBINED/'src/madgraph/loop/loop_exporters.py'))
    writer_module = load('combined_amplitude_writer', bind(COMBINED/'src/madgraph/iolibs/helas_call_writers.py'))
    instrument = load('frozen_amplitude_instrument', bind(OBS/'instrument.py'))
    bind(HISTORY/'transform.py')
    transform = instrument.BASE
    groups = json.loads(bind(HISTORY/'group-mapping.json').read_text())['groups']
    assert len(groups) == 200
    families = sorted({tuple(map(int, g['entrypoint'].split('_LOOP_')[1].split('_'))) for g in groups})
    families = [(v[0],v[1] if len(v)>1 else v[0]) for v in families]
    assert len(families) == 7
    all_text = ''.join(p.read_text() for p in templates.iterdir() if p.is_file())
    mapping = {key: 2 if fmt == 'd' else '' for key,fmt in re.findall(r'%\((\w+)\)([ds])',all_text)}
    mapping.update(proc_prefix='ML5_0_',real_dp_format='REAL*8',real_mp_format='REAL*16',
        complex_dp_format='COMPLEX*16',complex_mp_format='COMPLEX*32',mass_dp_format='COMPLEX*16',
        mass_mp_format='COMPLEX*32',maxrank=3,nexternal=5,ncomb=32,nbornamps=8,nloops=921,
        nloop_groups=200,nctamps=354,nloopamps=1275,nwavefuncs=35,nloopwavefuncs=1362,
        nSO=1,nSquaredSO=1,nAmpSO=2,include_vector='')
    exporter = object.__new__(exporter_module.LoopProcessOptimizedExporterFortranSA)
    exporter.template_dir = str(templates)
    exporter.opt = {'vector_size': 0}
    matrix = mock.Mock()
    matrix.rep_dict = dict(mapping)
    matrix.get.return_value = [base_objects.Process({'has_born': True})]
    matrix.get_used_helas_loop_amps.return_value = families
    matrix.get_loop_diagrams.return_value = []
    rendered = exporter_module.LoopProcessExporterFortranSA.write_CT_interface(exporter,None,matrix,optimized_output=True)
    rendered += '\n' + exporter.write_TIRUST_interface(None,matrix)
    context = dict(AmplitudeReduction=False,LoopInduced=False,samurai_available=False,ninja_available=True,
        ninja_supports_quad_prec=True,ninjars_available=True,ninjars_supports_quad_prec=True,
        tirust_available=True,tirust_supports_quad_prec=True)
    full = HERE/'src/generated_all.f'
    with file_writers.FortranWriter(str(full)) as writer:
        writer.writelines(rendered,context=context)
    routines = transform.routines(full.read_text())
    helper_names = [n for n in routines if re.fullmatch(r'ML5_0_LOOP_\d+(?:_\d+)?_QPMASS',n)]
    adapter_names = [n for n in routines if n.startswith('ML5_0_MP_') and n.endswith('_QPMASS')]
    assert len(helper_names)==7 and set(adapter_names)=={
        'ML5_0_MP_NINJARS_LOOP_QPMASS','ML5_0_MP_TIRUST_LOOP_QPMASS','ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC_QPMASS'}
    helper_texts=[]
    for name in helper_names:
        original=routines[name]['text']
        text=instrument.helper(original,True)
        capture=instrument.call('C32','QP_M2L','NLOOPLINE','MP_M2L(1)')
        assert text.count(capture)==1
        text=text.replace(capture,'')
        n=int(name.split('_LOOP_')[1].split('_')[0])
        last=f'MP_M2L({n})=MP_M{n-1 if n>1 else 1}**2'
        text=instrument.around(text,lambda s:s.replace(' ','').upper()==last,after=capture)
        assert text.index("'QP_M2L'")>text.index(last)
        helper_texts.append(text)
    (HERE/'src/helpers.f').write_text('\n\n'.join(helper_texts)+'\n')
    adapters=[]
    for name in adapter_names:
        text=routines[name]['text']
        if 'NINJARS' in name:text=instrument.adapter(text,True,8)
        elif 'TENSOR_DIAGNOSTIC' in name:text=instrument.adapter(text,True,9)
        adapters.append(text)
    (HERE/'src/adapters.f').write_text('\n\n'.join(adapters)+'\n')
    # Generate each actual retained group with the proposed HELAS call writer.
    writer=writer_module.FortranUFOHelasCallWriterOptimized(base_objects.Model())
    generated_calls=[]
    for g in groups:
        pieces=g['entrypoint'].split('_LOOP_')[1].split('_')
        n=int(pieces[0]);m=int(pieces[1]) if len(pieces)>1 else n
        args=g['original_actuals'];pairing=list(map(int,args[:n])) if n!=m else [1]*n
        offset=n if n!=m else 0
        amp=amplitude([v['model_symbol'] for v in g['mass_pairs']],pairing,g['loop_group'])
        for mother,number in zip(amp.get('mothers'),args[offset:offset+m],strict=True):mother.set('number',int(number))
        with mock.patch.object(aloha,'complex_mass',False),mock.patch.object(
            loop_helas_objects.LoopHelasAmplitude,'get_analytic_info',return_value=int(args[-3])):
            call=writer.get_amplitude_call(amp)%{'proc_prefix':'ML5_0_'}
        expected='IF(CTMODE.LT.4) THEN\nCALL '+g['entrypoint']+'('+','.join(args)+')\nELSE\nCALL '+g['new_entrypoint']+'('+','.join(g['new_actuals'])+')\nENDIF'
        assert call.replace(' ','')==expected.replace(' ',''),(g['loop_group'],call)
        generated_calls.append(call)
    old_calls=bind(SUB/'loop_CT_calls_1.f').read_text()
    statements=[s for a,b,s in transform.logical_blocks(old_calls) if s]
    old_science=[s for s in statements if s.upper().startswith('CALL ML5_0_LOOP_')]
    assert len(old_science)==200
    assert [s.replace(' ','') for s in old_science]==[
        ('CALL '+g['entrypoint']+'('+','.join(g['original_actuals'])+')').replace(' ','') for g in groups]
    mapping.update(bunch_name='LOOP_CT_CALLS',bunch_number=1,required_so_broadcaster='CTCALL_REQ_SO_DONE',
                   continue_label=5000,helas_calls='\n'.join(generated_calls)+
                   '\nIF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 5000')
    with file_writers.FortranWriter(str(HERE/'src/loop_CT_calls_1.f')) as out:
        out.writelines((templates/'helas_calls_split.inc').read_text()%mapping,context=context)
    # DP control differs only in observation count, warmup point cycling and
    # a transfer-only model snapshot. Both DP arms use this identical driver.
    driver=bind(OBS/'attempt2/check_observed.f').read_text()
    dp=driver.replace('fixed-QP','fixed-DP').replace('EW_QP_MASS_OBSERVED','EW_DP_MASS_OBSERVED')
    dp=dp.replace('ONLY: OBS_CLOSE','ONLY: OBS_CLOSE, OBS_R8, OBS_R16, OBS_C16, OBS_C32')
    dp=dp.replace("      INCLUDE 'coupl.inc'","      INCLUDE 'coupl.inc'\n      INCLUDE 'mp_coupl.inc'")
    dp=dp.replace('POINTS(0:3,5,2)','POINTS(0:3,5,1)').replace('NPOINTS.NE.2','NPOINTS.NE.1')
    dp=dp.replace('DO K=1,2','DO K=1,1').replace('1+MOD(K-1,2)','1')
    dp=dp.replace('FIXED_QP_TRANSPORT_ONLY_UNCHECKED','FIXED_DP_OUTPUT_ONLY_UNCHECKED')
    dp=dp.replace('over only two points.','on the ordinary point only.')
    snapshot=''
    for mass in ('MT','MZ','MW','MH'):
        snapshot+=instrument.call('R8','MODEL_MDL_'+mass,1,'(/MDL_'+mass+'/)')
        snapshot+=instrument.call('R16','MODEL_MP__MDL_'+mass,1,'(/MP__MDL_'+mass+'/)')
    snapshot+=instrument.call('R8','MODEL_MU_R',1,'(/MU_R/)')+instrument.call('C32','MODEL_MP__MU_R',1,'(/MP__MU_R/)')
    snapshot+=instrument.call('C16','MODEL_GAL',2,'GAL(1)')+instrument.call('C32','MODEL_MP__GAL',2,'MP__GAL(1)')
    dp=instrument.once(dp,'        CALL EW_MASS_POINT_BEGIN(K,P)\n','        CALL EW_MASS_POINT_BEGIN(K,P)\n'+snapshot)
    (HERE/'src/check_dp_observed.f').write_text(dp)
    # Preserve source-only evidence of the exact observer hook additions.
    import difflib
    for name,names in [('helpers',helper_names),('adapters',adapter_names)]:
        before='\n\n'.join(routines[n]['text'] for n in names)+'\n'
        (HERE/'src'/f'{name}.unobserved.f').write_text(before)
        after=(HERE/'src'/f'{name}.f').read_text()
        (HERE/'src'/f'{name}.observer.patch').write_text(''.join(difflib.unified_diff(
            before.splitlines(True),after.splitlines(True),fromfile=name+'.unobserved.f',tofile=name+'.f')))
    (HERE/'src/driver-dp.patch').write_text(''.join(difflib.unified_diff(driver.splitlines(True),dp.splitlines(True),fromfile='frozen-check_observed.f',tofile='check_dp_observed.f')))
    for path in (HERE/'src').glob('*.f'):
        for line in path.read_text().splitlines():
            if line.strip() and line[0] not in 'Cc*!#':
                assert len(line)<=132 and all(c in ' 0123456789' for c in line[:5]),(path,line)
    historical_build_path=bind(OBS/'attempt2/execution-1/build.json')
    old_build=json.loads(historical_build_path.read_text())
    old_prepared=json.loads(bind(OBS/'attempt2/prepared.json').read_text())
    old_run=json.loads(bind(OBS/'attempt2/execution-1/run.json').read_text())
    compile_recipe=old_build['commands'][0]['command']
    link_recipe=old_build['commands'][1]['command']
    compiler=bind(compile_recipe[3]);taskset=bind(compile_recipe[0])
    includes=[SUB]+[WORK/'includes'/str(i) for i in range(6)]
    for directory in includes:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in ('.inc','.mod'):bind(p)
    for p in (SUB/'CT_interface.f',SUB/'TIRUST_interface.f'):
        assert '_QPMASS' not in bind(p).read_text()
    jobs=[];outputs=[]
    for name in ('helpers','adapters','loop_CT_calls_1','check_dp_observed'):
        source=HERE/'src'/f'{name}.f';obj=HERE/'build'/f'{name}.o'
        argv=[str(taskset),'-c','8',str(compiler),'-w','-fPIC','-ffixed-line-length-132',
              '-J'+str(HERE/'build'),'-I'+str(HERE/'build'),*['-I'+str(p) for p in includes],
              '-c',str(source),'-o',str(obj)]
        jobs.append({'name':'compile-'+name,'argv':argv,'cwd':str(ROOT)})
        outputs.append(str(obj))
    old_objects=[arg for arg in link_recipe if arg.endswith('.o')]
    assert len(old_objects)==24 and str(SUB/'check_sa.o') in old_objects
    for arg in link_recipe:
        if arg.startswith('/') and Path(arg).is_file():bind(arg)
    exes={}
    for name in ('legacy-dp','guarded-dp','guarded-qp'):
        exe=HERE/'build'/name;exes[name]=exe
        argv=list(link_recipe)
        outindex=argv.index('-o')+1;argv[outindex]=str(exe)
        if name!='guarded-qp':argv[argv.index(str(SUB/'check_sa.o'))]=str(HERE/'build/check_dp_observed.o')
        if name!='legacy-dp':
            argv[argv.index(str(SUB/'loop_CT_calls_1.o'))]=str(HERE/'build/loop_CT_calls_1.o')
            # Only additive new QPMASS symbols; inherited legacy objects remain.
            insert=argv.index(str(SUB/'CT_interface.o'))
            argv[insert:insert]=[str(HERE/'build/helpers.o'),str(HERE/'build/adapters.o')]
        jobs.append({'name':'link-'+name,'argv':argv,'cwd':str(ROOT)})
        outputs.append(str(exe))
    historical_runtime={}
    for case in old_run['cases']:
        historical_runtime=case['loaded_libraries'];break
    assert historical_runtime
    for entry in historical_runtime.values():
        assert sha(bind(entry['path']))==entry['sha256']
    elf=bind(SUB/'check_observed').read_bytes()
    assert elf[:6]==b'\x7fELF\x02\x01'
    phoff=struct.unpack_from('<Q',elf,32)[0]
    phsize,phnum=struct.unpack_from('<HH',elf,54)
    interp=[]
    for index in range(phnum):
        header=struct.unpack_from('<IIQQQQQQ',elf,phoff+index*phsize)
        if header[0]==3:interp.append(elf[header[2]:header[2]+header[5]].rstrip(b'\0').decode())
    assert len(interp)==1
    loader=bind(interp[0])
    runtime={k:v for k,v in historical_runtime.items() if not k.startswith('/')}
    runtime[str(loader)]={'path':str(loader),'sha256':sha(loader)}
    points=bind(SUB/'selected.dat').read_text().splitlines(keepends=True)
    assert points[0].strip()=='2' and len([s for s in points if s.strip()])==11
    assert sha(bind(SUB/'param_card.dat'))==sha(bind(SUB/'MadLoop5_resources/param_card.dat'))
    runs=[]
    for arm in ('legacy-dp','guarded-dp','guarded-qp'):
        for backend in (8,9):
            name=f'{arm}-{backend}';directory=HERE/'runs'/name;directory.mkdir()
            resources=directory/'MadLoop5_resources';resources.mkdir()
            for source in (SUB/'MadLoop5_resources').rglob('*'):
                assert not source.is_symlink()
                target=resources/source.relative_to(SUB/'MadLoop5_resources')
                if source.is_dir():target.mkdir()
                else:shutil.copyfile(bind(source),target)
            shutil.copyfile(SUB/'param_card.dat',directory/'param_card.dat')
            (directory/'selected.dat').write_text(''.join(points) if arm=='guarded-qp' else '1\n'+''.join(points[1:6]))
            card=resources/'MadLoopParams.dat';lines=card.read_text().splitlines(keepends=True)
            values={'MLReductionLib':str(backend),'CTModeRun':'4' if arm=='guarded-qp' else '1','CTModeInit':'1',
                'UseQPIntegrandForNinjaRS':'.TRUE.','UseQPIntegrandForTIRust':'.TRUE.','TIRustComputeFinite':'.TRUE.'}
            changes={}
            for key,value in values.items():
                indices=[i for i,line in enumerate(lines) if line.strip().lower()=='#'+key.lower()]
                assert len(indices)==1
                i=indices[0]+1;changes[key]={'old':lines[i].strip(),'new':value};lines[i]=value+'\n'
            card.write_text(''.join(lines))
            assert re.search(r'#LoopPoleMode\s+0\s',card.read_text())
            run={'name':name,'argv':[str(exes[arm]),str(directory/'selected.dat')],'cwd':str(directory),
                 'timeout_seconds':120,'expected_outputs':['mass-observer.hex','observer-run.log','param.log'],
                 'arm':arm,'backend':backend,'mode':4 if arm=='guarded-qp' else 1,
                 'point_count':2 if arm=='guarded-qp' else 1,'card_changes':changes,
                 'input_sha256':{str(p):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()},
                 'runtime_mutable_inputs':[], 'controller_outputs':['stdout.log','loader-list.txt'],
                 'runtime_check_argv':[str(taskset),'-c','8',str(loader),'--list',str(exes[arm])],
                 'allowed_extra_outputs':[]}
            runs.append(run)
    oracles={}
    for backend in (8,9):
        path=bind(OBS/f'attempt2/execution-1/raw/candidate-{backend}/mass-observer.hex')
        oracles[str(backend)]={'path':str(path),'sha256':sha(path)}
    parser=bind(OBS/'analyze.py')
    bind(OBS/'attempt2/execution-1/evidence-index.json')
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename)
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(HERE):bind(path)
    prior_artifacts=dict(old_prepared['artifacts'],**old_build['outputs'])
    for path,checksum in external.items():
        if Path(path).is_relative_to(WORK):
            assert prior_artifacts.get(str(Path(path).relative_to(ROOT)))==checksum,('Historical input drift',path)
    prepared={'schema':'combined-general-qp-mass-amplitude-v1','state':'source_prepared_not_executed',
        'compiler_or_native_launched':False,'build_jobs':jobs,'build_outputs':outputs,'runs':runs,
        'compiler':{'path':str(compiler),'sha256':sha(compiler)},'runtime_expected':runtime,
        'historical_ldd_runtime':historical_runtime,
        'loader':{'path':str(loader),'sha256':sha(loader),'source':'PT_INTERP of bound historical executable; same pinned compiler/link recipe',
                  'historical_ldd_interpreter_alias_is_not_direct_loader':True},
        'historical_environment':old_build['environment'],'readonly_sha256':external,
        'copied_sha256':{str(p):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()},
        'oracles':oracles,'frozen_qp_parser':{'path':str(parser),'sha256':sha(parser)},
        'resources':{'cpu':8,'jobs':1,'combined_build_global_seconds':180,'run_seconds_each':120,
                     'automatic_retry':False,'shared_numerical_lock_required':True},
        'source_proof':{'actual_families':families,'actual_group_count':200,'maxrank':3,
            'guarded_calls_from_actual_helas_writer':200,'dp_actuals_exact_legacy':True,
            'qp_actuals_exact_frozen_candidate':True,'no_original_objects_modified':True,
            'observer_change':'QP_M2L capture moved after the final guarded square, inside the selected-8/9 guard; intermediate sticky-flag timing is not compared',
            'not_full_cli_regeneration':'Retained 200 actual group mappings reconstructed as HELAS amplitude objects and emitted through the real proposed writer/exporter/templates',
            'dp_transport':'No QP groups expected; DP model snapshots and output/status/final flags only'},
        'scope':'Fixed-mode diagnostics only; no stability, physics acceptance, all-backend or performance claim.'}
    save(HERE/'prepared.json',prepared)
    print(json.dumps({'prepared_sha256':sha(HERE/'prepared.json'),'build_jobs':len(jobs),'runs':len(runs),'compiler_or_native_launched':False}))


if __name__=='__main__':
    main()
