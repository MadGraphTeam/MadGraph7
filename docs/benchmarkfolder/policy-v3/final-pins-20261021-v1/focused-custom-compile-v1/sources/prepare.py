"""Compile-only private focused drivers; no executable or reducer invocation."""
import argparse,contextlib,difflib,hashlib,json,math,os,re,shutil,signal,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'tools'),str(ROOT)]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_references_v2 as adapter
import ninjars_benchmark_workers as workers
import controls_driver as controls
FINAL=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1'
REPORT=FINAL/'focused-custom-compile-v1'
WORK=ROOT/'.benchmark-work/policy-v3/final-pins-20261021-v1/focused-custom-compile-v1'
RUNTIME='a7577a9a33292ffb5d675ccf1b9c3667db3ab3613d77ccd270f1efc9e9802008'
GENERATOR='77b16df0251af4104a2c541bab913c11f4e67496a290495ce1a1c5fc526dfe3c'
VARIANTS=(
('control-rotation','control-gg-hh','control-rotation',None,False),
('control-rebinding','control-gg-hh','control-rebinding','state',False),
('control-rebinding-fresh','control-gg-hh','control-rebinding-fresh','state',True),
('control-direction','control-gg-hh','control-gg-hh','direct',False),
('control-rebinding-qp','control-rebinding','control-rebinding','direct',False),
('control-rebinding-fresh-qp','control-rebinding-fresh','control-rebinding-fresh','direct',False),
('control-complex-qp','control-complex','control-complex','direct',False))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def bound(path):return {'path':policy.relative(path),'sha256':digest(path),'bytes':Path(path).stat().st_size}
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def writable_clone(output):
    """Reject external aliases before the reused build/configuration mutates."""
    output = output.resolve()
    paths = [output / name for name in (
        "Source", "SubProcesses", "Cards", "compilation.json",
        "compiled-sources.json", "compilation-history", "Cards/MadLoopParams.dat")]
    for directory in output.glob("SubProcesses/P*_*"):
        paths.extend(directory / name for name in (
            "check", "check_sa.f", "check_sa.original", "dimensions.dat", "masses.dat",
            "timing.dat", "values.dat", "timed_status.dat"))
        # Imported module symlinks (e.g. CutTools mpmodule.mod) are read-only
        # inputs. Only locally declared MODULEs can be written by this build.
        local_modules = {
            name.lower() for source in directory.iterdir()
            if source.is_file() and source.suffix.lower() in (".f", ".f90", ".inc")
            for name in re.findall(
                r"(?im)^\s*MODULE\s+(?!PROCEDURE\b|FUNCTION\b|SUBROUTINE\b)(\w+)",
                source.read_text())}
        paths.extend(p for p in directory.rglob("*") if
                     p.suffix in (".o", ".smod") or
                     (p.suffix == ".mod" and
                      (not p.is_symlink() or p.stem.lower() in local_modules)))
    paths.extend((output / "SubProcesses").rglob("MadLoopParams.dat"))
    paths.extend((output / "compilation-history").rglob("*"))
    paths.extend((output / "SubProcesses").rglob("*HelFilter.dat"))
    paths.extend((output / "SubProcesses").rglob("*LoopFilter.dat"))
    if any(not path.resolve().is_relative_to(output) for path in paths):
        raise RuntimeError("Writable generated path escapes its isolated clone")


def parse_direct(directory,expected_dimensions,mode):
    """Strict diagnostic110/120 reader, with real generated Born dimensions."""
    if mode not in (1,2,4,5):raise ValueError('Unknown public direct mode')
    dimensions=list(map(int,(directory/'dimensions.dat').read_text().split()))
    if dimensions!=list(expected_dimensions):raise ValueError('Direct result changed generated answer dimensions')
    allocated,loop,born=dimensions
    if allocated!=max(loop,born) or min(dimensions)<0 or born>loop:raise ValueError('Invalid direct answer dimensions')
    expected_status=110 if mode in (1,2) else 120
    rows=[]
    for line in (directory/'values.dat').read_text().splitlines():
        fields=line.split()
        if len(fields)!=8:raise ValueError('Malformed direct row')
        row={'point':int(fields[0]),'order':int(fields[1]),'status':int(fields[2]),'coefficients':list(map(float,fields[3:7])),'accuracy':float(fields[7])}
        if row['status']!=expected_status or row['accuracy']!=-1 or not all(math.isfinite(x) for x in row['coefficients']):raise ValueError('Wrong direct code/accuracy or nonfinite coefficients')
        rows.append(row)
    if [(r['point'],r['order']) for r in rows]!=[(p,o) for p in range(1,11) for o in range(loop+1)]:raise ValueError('Incomplete/reordered direct physical rows')
    histogram=[list(map(int,line.split())) for line in (directory/'timed_status.dat').read_text().splitlines()]
    if histogram!=[[expected_status,10]]:raise ValueError('Wrong direct evaluation histogram')
    return {'rows':rows,'status_counts':{str(expected_status):10},'timed_status_counts':{str(expected_status):10},'answer_dimensions':{'allocated':allocated,'loop_orders':loop,'born_orders':born},'direct_mode':mode,'stability_measurement':False,'automatic_rescue':False,'performance_acceptance_claim':False}


def parent_record(name,ready):
    item=ready['fresh_outputs'][name];output=policy.path(item['output'])
    for field in ('generation','compilation','compiled_sources','param_card'):policy.check_file(item[field])
    manifest=bench.generated_source_manifest(output)
    if manifest['files_sha256']!=item['generated_sources_sha256'] or pair.actual_runtime(output,manifest,8)['identity_sha256']!=RUNTIME:raise ValueError('Final parent source/full runtime changed')
    directory=next(output.glob('SubProcesses/P*_*'))
    if digest(directory/'check')!=item['executable_sha256']:raise ValueError('Final parent executable changed')
    return {'output':policy.relative(output),'sources':manifest,'libraries':policy.library_hashes(output),'runtime':pair.actual_runtime(output,manifest,8),'parameter_card_sha256':digest(output/'Cards/param_card.dat'),'executable_sha256':digest(directory/'check'),'dimensions':item['dimensions'],'definition':item['definition']}


@contextlib.contextmanager
def owned_build():
    def interrupted(signum,_frame):raise SystemExit('Compile preparation interrupted by signal '+str(signum))
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGHUP)}
    original_adapter=adapter.subprocess
    try:
        for sig in previous:signal.signal(sig,interrupted)
        with pair.owned_preparation(REPORT,label='focused-compile'):
            adapter.subprocess=workers.subprocess
            original_run=bench.run
            def serial_make(command,*args,**kwargs):
                if command==['make','-j4','check']:command=['make','-j1','check']
                return original_run(command,*args,**kwargs)
            bench.run=serial_make
            yield
    finally:
        adapter.subprocess=original_adapter
        for sig,handler in previous.items():signal.signal(sig,handler)


def prepare():
    if os.sched_getaffinity(0)!={6} or any(os.environ.get(k)!='1' for k in pair.THREADS):raise ValueError('Compile-only preparation requires CPU6 and four thread settings1')
    if REPORT.exists() or WORK.exists():raise ValueError('Preserve previous preparation; fresh namespace required')
    recipe=read(HERE/'custom-controls-held.json');ready=read(FINAL/'focused-readiness-ninja-nrs-v1/readiness.json')
    if digest(HERE/'custom-controls-held.json')!='0e81b9f49b355217022da681ae50d14791e9e2ec416f3df1970ed44a46bd0370' or digest(HERE/'controls_driver.py')!='93e41b821058e89c04d01f72070406ede09a55cb37021b5995cb27635897ff9c':raise ValueError('Bound recipe/state helper changed')
    if digest(FINAL/'focused-readiness-ninja-nrs-v1/readiness.json')!='b5d9e652d38ad4707c0e7b06e28a9df155f319802bcca2b6b1a32e6f176e076c' or any(ready['fresh_outputs'][name]!=record for name,record in recipe['parents'].items()):raise ValueError('Reviewed focused readiness or exact recipe parents changed')
    if bench.content_digest(bench.generator_tree_state())!=GENERATOR:raise ValueError('Live generator is not the final frozen generator')
    parents={name:parent_record(name,ready) for name in ('control-gg-hh','control-complex')}
    REPORT.mkdir();WORK.mkdir(parents=True);(REPORT/'points').mkdir()
    bench.WORK,bench.REPORT=WORK,REPORT
    prepared={'scope':'Compile-only private focused drivers; no executable/SETPARA/reducer calls','ready_for_numerical_execution':False,'native_evaluation_calls':0,'script':bound(Path(__file__)),'recipe':bound(HERE/'custom-controls-held.json'),'control_helper':bound(HERE/'controls_driver.py'),'parents':parents,'variants':{},'actual_commands':'Only registered cp and make -j1 check'}
    with owned_build():
        for name,parent_name,points_name,transformation,fresh in VARIANTS:
            if parent_name in parents:origin=parents[parent_name]
            else:origin=prepared['variants'][parent_name]
            parent=policy.path(origin['output']);output=adapter.clone_output(parent,WORK/'outputs/candidate'/name);writable_clone(output)
            original_parent='control-complex' if name=='control-complex-qp' else 'control-gg-hh'
            base=parents[original_parent];case={**base['definition'],'id':name,'parent_control':original_parent,'madloop_parameters':{'NRotations_DP':2,'NRotations_QP':2}}
            directory=next(output.glob('SubProcesses/P*_*'));driver=directory/'check_sa.f';before=driver.read_text()
            prefix=re.search(r'CALL\s+(\w*)GET_ANSWER_DIMENSION',before,re.I).group(1)
            if int(re.search(r'NSQSO_BORN\s*=\s*(\d+)',before,re.I).group(1))!=base['dimensions'][2]:raise ValueError('Driver Born dimension differs from final parent')
            if transformation=='state':
                if before.count('CALL UPDATE_AS_PARAM()')!=4:raise ValueError('Expected all4 state update sites')
                controls.build_rebinding_driver(case,fresh)
                after=driver.read_text()
                if after.count('CALL CONTROL_STATE(')!=4 or 'MP__MDL_MT=REAL(MDL_MT,KIND=16)' not in after or 'MP__MU_R=CMPLX(REAL(MU_R,KIND=16),0.0_16,KIND=16)' not in after:raise ValueError('Missing exact both-precision state update')
            elif transformation=='direct':
                old='CALL '+prefix+'FORCE_STABILITY_CHECK(.TRUE.)';new='CALL '+prefix+'FORCE_STABILITY_CHECK(.FALSE.)'
                if before.count(old)!=1:raise ValueError('Expected exactly one derived-prefix public force switch')
                driver.unlink();driver.write_text(before.replace(old,new));driver.with_suffix('.o').unlink(missing_ok=True);(directory/'check').unlink(missing_ok=True)
                bench.run(['make','-j1','check'],directory,WORK/'logs'/('compile-'+name+'.log'))
            manifest=bench.generated_source_manifest(output);allowed={str(driver.relative_to(output))}
            if not policy.delta(base['sources']['files'],manifest['files']).issubset(allowed):raise ValueError('Focused clone changed numerical source outside the driver')
            if policy.library_hashes(output)!=base['libraries'] or pair.actual_runtime(output,manifest,8)!=base['runtime'] or digest(output/'Cards/param_card.dat')!=base['parameter_card_sha256']:raise ValueError('Focused clone changed physics card/full runtime')
            writable_clone(output)
            metadata=read(output/'compilation.json');metadata.update(driver_sha256=digest(driver),executable_sha256=digest(directory/'check'),generated_sources_sha256=manifest['files_sha256'],source_manifest_recording='recorded-after-compile-only-focused-driver')
            bench.dump(output/'compilation.json',metadata);bench.dump(output/'compiled-sources.json',manifest)
            points=ROOT/'docs/benchmarkfolder/points'/f'{points_name}.dat';shutil.copy2(points,REPORT/'points'/f'{name}.dat')
            if int(points.read_text().splitlines()[0])!=10:raise ValueError('Focused stream must retain10 points')
            retained=REPORT/'drivers'/f'{name}.f';retained.parent.mkdir(exist_ok=True);shutil.copy2(driver,retained)
            write(REPORT/'drivers'/f'{name}.patch',{'diff':''.join(difflib.unified_diff(before.splitlines(True),driver.read_text().splitlines(True),fromfile='clone-parent.f',tofile=name+'.f'))})
            prepared['variants'][name]={'case':case,'output':policy.relative(output),'sources':manifest,'libraries':base['libraries'],'runtime':base['runtime'],'parameter_card_sha256':base['parameter_card_sha256'],'executable_sha256':digest(directory/'check'),'driver':bound(retained),'points':bound(REPORT/'points'/f'{name}.dat'),'dimensions':base['dimensions'],'prefix':prefix,'direct_mode_driver':transformation=='direct','compilation':metadata}
            print(name+' compile-only prepared',flush=True)
    for name,record in parents.items():
        if parent_record(name,ready)!=record:raise ValueError('Final parent changed during private preparation')
    if bench.content_digest(bench.generator_tree_state())!=GENERATOR:raise ValueError('Generator changed during private preparation')
    write(REPORT/'prepared.json',policy.seal(prepared))
    print('Compile-only preparation complete; no custom numerical execution is authorized.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-compile-only',action='store_true');args=parser.parse_args()
    if not args.prepare_compile_only:parser.error('No action without explicit compile-only flag')
    prepare()
