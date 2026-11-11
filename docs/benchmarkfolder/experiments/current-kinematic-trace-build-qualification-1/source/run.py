#!/usr/bin/env python3
"""Private staged trace executor. Default is a read-only plan.

Generation requires a reviewed source index and the shared lock. The generated
CMS source and derived insertions receive a second reviewed index before build.
No stage is implicitly continued, and no failed stage is retried in-place.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time

from admission import (HERE, OLD, ROOT, WORK, RESULTS, helpers, initial_admission,
                       load, read_json, record, require, save, sealed, sha, verify_record)
from trace_current import derive

SUPERVISOR = ROOT/'docs/benchmarkfolder/experiments/current-cms-replay-supervisor-1'
CONTEXT = json.loads(verify_record(OLD/'context-bindings.json',{
    'bytes':2756,'sha256':'082f5df9baa6973940914ce9d0d8b89cf14ae31c8cc5b2cb4340f3a71d227366'}))
SCIENCE = json.loads(verify_record(OLD/'input-bindings.json',{
    'bytes':3603,'sha256':'6f801d535de58b77ae8158315d0e686ea8575bfdc61c668181b4a7cb3d703a4c'}))
STAGES = ('generate','derive','build','ordinary','stress')
CAPS = {'generate':900,'derive':120,'build':3000,'ordinary':2400,'stress':4800}
CASES = {'cms':('cms_ew_n2_epem_mupmum','P0_epem_mupmum',4),
         'ew':('ew_n3_epem_ttxa','P0_epem_ttxa',5)}
L = PREREQUISITES = None


def setup_owned_helpers():
    global L, PREREQUISITES
    helpers()
    L = load('ktrace_owned',SUPERVISOR/'ownership/run.py')
    L.ROOT,L.WORK,L.REPORT = ROOT,WORK,RESULTS
    L.q.ROOT,L.q.WORK,L.q.REPORT = ROOT,WORK,RESULTS
    L.q.capture = L.capture
    L.STAGES = STAGES
    PREREQUISITES = load('ktrace_dependencies',SUPERVISOR/'prerequisites.py')
    # The old helper's imported DATA is not the admitted dependency contract.
    PREREQUISITES.DATA = read_json(HERE/'dependencies.json')
    def environment(cwd):
        env = L.q.env()
        for key in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT'):
            env.pop(key,None)
        env.update(PWD=str(cwd),RAYON_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   XDG_STATE_HOME=str(WORK/'xdg/state'),
                   XDG_DATA_HOME=str(WORK/'xdg/data'),
                   MAKEFLAGS='-j2 SHELL='+str(Path(shutil.which('bash')).resolve()))
        return env
    L.environment = environment


def command(argv,label,cwd=ROOT,seconds=600):
    # The inherited owner supervises descendants and preserves failed commands.
    return L.command(argv,label,cwd,seconds)


def verify_context():
    initial_admission()
    require(Path(sys.executable).resolve()==Path(CONTEXT['generation_python']).resolve(),
            'Pinned tools/tirust_env.sh Python required')
    manifest = json.loads(verify_record(ROOT/CONTEXT['campaign_manifest']['path'],CONTEXT['campaign_manifest']))
    for name, expected in manifest['candidate']['exporter_identity']['files'].items():
        p = ROOT/name
        require((sha(p.read_bytes()) if p.is_file() else None)==expected,
                'Current source changed: '+name)
    for case in ('cms','ew'):
        for key in ('points','physical_card','numerical_card'):
            pin = SCIENCE[case][key]
            verify_record(ROOT/pin['path'],pin)
    return manifest


def source_inventory(directory):
    """Use the retained benchmark's complete generated-source/build-input scope."""
    suffixes={'.f','.f90','.for','.inc','.c','.cc','.h'}
    return {str(p.relative_to(ROOT)):record(p)
            for subtree in ('Source','SubProcesses')
            for p in sorted((Path(directory)/subtree).rglob('*'))
            if p.is_file() and (p.suffix.lower() in suffixes or p.name.startswith('makefile')
                or p.name in ('make_opts','MadLoop_makefile_definitions'))}


def verify_inventory(files):
    for name,pin in files.items():
        verify_record(ROOT/name,pin)


def exact_inventory(directory,files):
    require(source_inventory(directory)==files,
            'Generated source/build-input inventory added, removed or changed entries')


def dependency_snapshot():
    PREREQUISITES.check_donors()
    data = PREREQUISITES.DATA
    paths = dict(data['external'])
    for pin in data['sources']['candidate']['copies'].values():
        paths[pin['donor']] = pin
    return {str(Path(name).resolve()):record(name) for name in paths}


def private_clone(source,target,keep_objects=False):
    """Rewrite internal links to the clone; external links need admitted bytes."""
    source,target = Path(source),Path(target)
    require(not target.exists(), 'Fresh clone required')
    allowed = dependency_snapshot()
    directory_aliases=read_json(HERE/'external-directory-bindings.json')
    private_deps=PREREQUISITES.DATA['sources']['candidate']['copies']
    for name,pin in private_deps.items():
        local=WORK/'mg-current'/name
        if local.is_file():
            require(sha(local.read_bytes())==pin['sha256'],'Private generated dependency differs')
            allowed[str(local.resolve())]=record(local)
    def ignore(directory,names):
        result=[]
        for name in names:
            p=Path(directory)/name
            if (name=='__pycache__' or name.endswith('.pyc') or name.endswith('.fifo') or
                name.endswith(('HelFilter.dat','LoopFilter.dat')) or
                (not keep_objects and (p.suffix=='.o' or name=='check'))):
                result.append(name)
        return result
    shutil.copytree(source,target,symlinks=True,ignore=ignore)
    for p in target.rglob('*'):
        if not p.is_symlink():
            continue
        original=source/p.relative_to(target)
        resolved=original.resolve(strict=True)
        if resolved.is_relative_to(source.resolve()):
            replacement=target/resolved.relative_to(source.resolve())
            p.unlink();p.symlink_to(os.path.relpath(replacement,p.parent))
        else:
            if resolved.is_dir():
                alias=directory_aliases.get(p.name)
                require(alias is not None and str(resolved)==alias['resolved_target'],
                        'Unreviewed external directory alias: '+str(original))
                require(sorted(x.name for x in resolved.iterdir() if x.is_file())==alias['direct_file_names'],
                        'External include/library directory entries changed')
                for name,pin in alias['required_files'].items():verify_record(name,pin)
                continue
            require(resolved.is_file() and str(resolved) in allowed,
                    'Unbound external clone link: '+str(original))
            verify_record(resolved,allowed[str(resolved)])
    return target


def generate():
    manifest=verify_context()
    require(not WORK.exists(), 'Fresh private work root required')
    WORK.mkdir(parents=True)
    before=dependency_snapshot()
    configuration={str(p):record(p) for p in (ROOT/'input').glob('*configuration*.txt') if p.is_file()}
    mg=WORK/'mg-current';mg.mkdir()
    archive=WORK/'current-source.tar'
    command(['git','archive','--output='+str(archive),CONTEXT['production_anchor']],
            'current-source-archive',seconds=120)
    with tarfile.open(archive) as stream:
        stream.extractall(mg,filter='data')
    for name,expected in manifest['candidate']['exporter_identity']['files'].items():
        p=mg/name
        require((sha(p.read_bytes()) if p.is_file() else None)==expected,
                'Archived current source differs: '+name)
    PREREQUISITES.copy_private('candidate',mg)
    model=WORK/'models/loop_qcd_qed_sm_Gmu';model.mkdir(parents=True)
    for name,expected in manifest['shared_inputs']['model_tree'].items():
        donor=ROOT/CONTEXT['model_donor']/name
        require(sha(donor.read_bytes())==expected,'Model donor changed')
        target=model/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(donor,target)
    output=WORK/'exports/cms';output.parent.mkdir(parents=True)
    script=(OLD/'candidate-cms.mg5').read_text()
    script=script.replace(str(OLD.parent/'work/models/loop_qcd_qed_sm_Gmu'),str(model))
    script=script.replace(str(OLD.parent/'work/exports/candidate/cms_ew_n2_epem_mupmum'),str(output))
    script_path=WORK/'current-cms.mg5';script_path.write_text(script)
    require(sha((mg/'bin/madgraph').read_bytes())==CONTEXT['cli']['sha256'],
            'CLI entrypoint changed')
    raw=command([sys.executable,'-B','./bin/madgraph','--plain',script_path],
                'current-cms-generation',mg,600)
    require(not any(s in raw for s in ('Error detected','interrupted in sub-command',
            'Traceback (most recent call last)','Compiling CutTools','Compiling IREGI',
            'Re-compiling it now')), 'CLI failure/dependency rebuild marker')
    PREREQUISITES.check_private('candidate',mg)
    PREREQUISITES.check_generated('candidate',output,'SubProcesses/P0_epem_mupmum')
    require(before==dependency_snapshot(),'Dependency mutated during generation')
    for name,pin in configuration.items():verify_record(name,pin)
    verify_context()
    files=source_inventory(output)
    require(files,'CMS export sources missing')
    save(RESULTS/'generation.json',{'source_anchor':CONTEXT['production_anchor'],
        'source_inventory':CONTEXT['exporter_inventory'],'archive':record(archive),
        'command':record(script_path),'raw_sources':files,
        'dependencies_before_after':before,'shared_configuration':configuration,
        'generation_only':True,'amplitude_evaluations':0})


def render_driver(nlegs,nborn,instrumented):
    text=(HERE/'driver.f.in').read_text()
    control=('      CALL ML5_0_PRIVATE_KTRACE_SET(LUN,POINT,ENABLED)' if instrumented
             else '      CONTINUE')
    for key,value in {'PREFIX':'ML5_0_','NEXTERNAL':str(nlegs),'NBORN':str(nborn),
                      'CONTROL_CALL':control}.items():
        text=text.replace('@'+key+'@',value)
    require(not re.search(r'@[A-Z_]+@',text),'Unexpanded driver placeholder')
    require(all(len(line)<=72 for line in text.splitlines() if not line.startswith(('C','c','*'))),
            'Rendered driver fixed-form line too long')
    return text


def derive_sources():
    verify_context()
    generated=read_json(RESULTS/'generation.json')
    exact_inventory(WORK/'exports/cms',generated['raw_sources'])
    dest=RESULTS/'derived';require(not dest.exists(),'Fresh derived namespace required')
    dest.mkdir()
    output=WORK/'exports/cms';sub=str((output/'SubProcesses/P0_epem_mupmum').relative_to(ROOT))
    binding={'schema':'private-saved-routing-inputs-v1','files':generated['raw_sources'],
        'cases':{'cms_ew_n2_epem_mupmum':{'subprocess':sub,
        'dhelas':str((output/'Source/DHELAS').relative_to(ROOT)), 'nexternal':4,
        'loop_call_sources':[sub+'/loop_CT_calls_1.f']}}}
    binding_path=dest/'cms-source-bindings.json';save(binding_path,binding)
    # The binding is written and hashed before source parsing; independent
    # reviewed derived index admission remains mandatory before compilation.
    derive(binding_path,sha(binding_path.read_bytes()),[3,4,8,12,24],dest/'cms')
    ew=HERE/'ew-source-bindings.json'
    derive(ew,sha(ew.read_bytes()),[5],dest/'ew')
    for key,(_,_,nlegs) in CASES.items():
        for variant in ('unmodified','instrumented'):
            (dest/key/('driver-'+variant+'.f')).write_text(render_driver(nlegs,1,variant=='instrumented'))
    save(dest/'index.json',{'schema':'current-generated-trace-derived-index-v1',
        'files':{str(p.relative_to(dest)):record(p) for p in sorted(dest.rglob('*')) if p.is_file()},
        'generated_source_receipt':record(RESULTS/'generation.json'),
        'compiled':False,'native_evaluations':0})


def acquire(requests):
    handle=(ROOT/'.tirust-build/numerical-tests.lock').open('a')
    start=time.monotonic()
    while True:
        requests.check()
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic()-start>=3600:
                handle.close();raise TimeoutError('Shared-lock admission expired')
            time.sleep(.1)


class BudgetRequests:
    """Reuse the owner's polling boundary for bounded diagnostic log files."""
    def __init__(self,requests):self.requests=requests
    def check(self):
        self.requests.check()
        for path in (RESULTS/'execution').glob('*.txt'):
            require(path.stat().st_size<=33554432,'Owned command log byte limit exceeded')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',choices=STAGES)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--reviewed-source-index')
    parser.add_argument('--reviewed-derived-index')
    args=parser.parse_args()
    require(Path.cwd()==ROOT,'Run from repository root')
    if not args.execute:
        print(json.dumps({'execution':False,'stages':CAPS,'work':str(WORK),
            'results':str(RESULTS),'workers':30,'outer_entry_cap':60,
            'native_modes':[1,4],'deferred_modes':[2,5]},indent=2))
        return
    require(args.stage and args.reviewed_source_index,'Explicit stage and reviewed source index required')
    sealed(HERE/'index.json',args.reviewed_source_index)
    verify_context();setup_owned_helpers()
    if args.stage in ('build','ordinary','stress'):
        require(args.reviewed_derived_index,'Reviewed generated-source index required')
        sealed(RESULTS/'derived/index.json',args.reviewed_derived_index)
    require({12,13}<=set(os.sched_getaffinity(0)),'Proposed nonexclusive CPUs unavailable')
    RESULTS.mkdir(parents=True,exist_ok=True)
    marker=RESULTS/('stage-'+args.stage+'.json');require(not marker.exists(),'Append-only stage; no retry')
    L.CURRENT_STAGE=args.stage;L.CPU='12,13' if args.stage in ('generate','build') else 12
    failure=None;lock=None;start=time.monotonic();active=None
    with L.cleanup.SignalRequests() as requests:
        L.REQUESTS=BudgetRequests(requests)
        try:
            if args.stage!='derive':lock=acquire(requests)
            active=time.monotonic()
            # Inherited owner reserves 10s; subtract another50s for the agreed60s.
            with L.Deadline(CAPS[args.stage]-50):
                actions={'generate':generate,'derive':derive_sources}
                if args.stage in ('build','ordinary','stress'):
                    from stages_native import build,run_matrix
                    actions.update(build=lambda:build(sys.modules[__name__]),
                                   ordinary=lambda:run_matrix(sys.modules[__name__],'ordinary'),
                                   stress=lambda:run_matrix(sys.modules[__name__],'stress'))
                actions[args.stage]()
        except BaseException as error:
            failure=error
        finally:
            if lock is not None:lock.close()
            save(marker,{'stage':args.stage,'passed':failure is None,
                'failure':None if failure is None else repr(failure),
                'elapsed_with_lock_wait':time.monotonic()-start,
                'elapsed_after_lock':None if active is None else time.monotonic()-active,
                'source_index_sha256':args.reviewed_source_index,
                'derived_index_sha256':args.reviewed_derived_index,
                'performance_measurement':False})
    if failure is not None:raise failure


if __name__=='__main__':main()
