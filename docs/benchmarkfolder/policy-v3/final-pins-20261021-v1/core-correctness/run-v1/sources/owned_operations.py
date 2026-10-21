#!/usr/bin/env python3
"""Collect fixed stress raw streams before the real core dataset is available.

Operational glue only. Existing workers, numerical settings, scale adapter,
source binder and final readers remain unchanged. No core identity is invented.
"""
import argparse
from contextlib import contextmanager, nullcontext
import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents
            if (path/'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'docs/benchmarkfolder/policy-v2/final-stress-execution-v1'))
import early_build
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers
import ninjars_stress_v2 as stress
import ninjars_supplement_v2 as supplement

HERE = Path(__file__).resolve().parent
SCHEMA = 'stress-early-raw-spec-v1'
CPU = 190
SOURCES = ('candidate',)
BACKENDS = (1, 6, 7, 8)
ROLES = {'candidate':'candidate_corrected'}


def read(path):
    return json.loads(Path(path).read_text())


def exclusive_json(path, value):
    path = policy.path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def frozen_sources():
    paths = {name:ROOT/'tools'/name for name in (*workers.SOURCE_NAMES,
        'ninjars_stress_v2.py','ninjars_supplement_v2.py','ninjars_benchmark_policy_v2.py',
        'ninjars_precision_v2.py')}
    paths['early_build.py'] = Path(early_build.__file__).resolve()
    paths['collect.py'] = Path(__file__).resolve()
    return {name:supplement.bound(path) for name,path in paths.items()}


def actual_runtime(output, manifest):
    return policy.reducer_signature({'backend':8,'libraries':policy.library_hashes(output),
        'runtime_metadata':{name:supplement.bound(path)
            for name,path in policy.runtime_metadata_files(output).items()},
        'compiled_manifest':manifest})


def cohort(early_result):
    """Check the real early assets without fabricating a final expected map."""
    filename = policy.path(early_result)
    result = policy.verified(read(filename), 'stress-early-build-result-v1')
    registration = policy.verified(read(supplement.checked(result['registration'])), early_build.SCHEMA)
    status = read(supplement.checked(result['status']))
    if status['status'] != 'complete' or status['amplitude_evaluations'] != 0:
        raise ValueError('The early CLI/build/point cohort is not complete and unevaluated')
    if ('expected' in registration or 'core_dataset_identity_sha256' in registration
            or registration['loop_amplitude_evaluations'] != 0
            or registration['initialized_symbolica_workers'] != 0):
        raise ValueError('Early assets must not assert a future core dataset or prior reduction')
    signature = early_build.runtime(registration)
    inventory = early_build.sources(registration['numerical_revision'])
    if inventory['production'] != registration['numerical_sources_sha256']['production']:
        raise ValueError('The actual numerical source differs from the early freeze')
    if bench.content_digest(bench.generator_tree_state()) != registration['generator_identity_sha256']:
        raise ValueError('The actual generator differs from the early source cohort')
    for variant in stress.VARIANTS:
        if read(supplement.checked(result['configurations'][variant])) != stress.configuration(variant):
            raise ValueError('The fixed stress configuration changed')
    if supplement.checked(result['scale_schedule']).read_text() != ''.join(
            f'{factor:.17e}\n' for factor in stress.scale_schedule()):
        raise ValueError('The fixed scale schedule changed')
    work = policy.path(registration['work'])
    ordinary_config = stress.configuration('ordinary')
    for index,case in enumerate(ordinary_config['cases']):
        cid = case['id']
        proof = policy.verified(read(supplement.checked(result['source_proofs'][cid])),
                                'stress-generation-source-v1')
        output = work/'outputs/candidate'/cid
        manifest = bench.generated_source_manifest(output)
        identity = proof['generation']['source_identity']
        if (proof['case'] != case or manifest != proof['compiled_manifest']
                or identity['revision'] != registration['generation_revision']
                or identity['tree_state_sha256'] != registration['generator_identity_sha256']
                or identity.get('dirty_patch_sha256') is not None or identity['tree_state']['untracked']
                or actual_runtime(output,manifest) != signature):
            raise ValueError('A built stress input differs from its actual source/full runtime freeze')
        physical = supplement.checked(registration['physical_cards'][case['model']])
        if (output/'Cards/param_card.dat').read_bytes() != physical.read_bytes():
            raise ValueError('A physical parameter card changed')
        ordinary = supplement.checked(result['point_files'][cid])
        events = stress.parse_points(ordinary)
        metadata = read(supplement.checked(result['ordinary_point_metadata'][cid]))
        masses = status['cases'][cid]['masses']
        if (len(events) != 20 or metadata['points'] != 20
                or metadata['seed'] != ordinary_config['seed']+index
                or metadata['energy_gev'] != 1000. or metadata['masses'] != masses
                or metadata['sha256'] != supplement.digest(ordinary)):
            raise ValueError('The ordinary stress stream changed')
        supplement.point_metadata(ordinary,masses,4)
        for variant,values in (('rotation',stress.rotated(events)),('scale',stress.scale_events(events))):
            points = supplement.checked(result['point_files'][cid+'-'+variant])
            if points.read_text() != stress.serialize(values):
                raise ValueError('A fixed stress transformation changed')
            supplement.point_metadata(points,masses,4)
    return result,registration,signature


def write_spec(args):
    result,registration,signature = cohort(args.early_result)
    if args.variant not in stress.VARIANTS:
        raise ValueError('Unknown stress variant')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    relative = policy.relative(Path(__file__))
    tracked = subprocess.run(['git','ls-files','--error-unmatch','--',relative],cwd=ROOT,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0
    spec = policy.seal({'schema':SCHEMA,'early_result':supplement.bound(args.early_result),
        'variant':args.variant,'cpu':CPU,'report':policy.relative(args.report),
        'work':policy.relative(args.work),'run_id':'early-stress-'+args.variant+'-v1',
        'generation_revision':registration['generation_revision'],
        'numerical_revision':registration['numerical_revision'],
        'generator_identity_sha256':registration['generator_identity_sha256'],
        'installed_archive_sha256':registration['archive_sha256'],
        'runtime_signature':signature,'sources':frozen_sources(),
        'operational_source_state':{'actual_git_head':head,'wrapper_path':relative,
            'wrapper_tracked':tracked,'wrapper_bytes_bound_in_sources':True,
            'uncommitted_docs_source_permitted':True,
            'production_equivalence_checked_against':registration['numerical_revision']},
        'core_dataset_link':'not-yet-available; no core identity asserted',
        'configuration':stress.configuration(args.variant),
        'performance_acceptance_gate':False,'final_dataset_claim':False})
    exclusive_json(args.output,spec)
    return {'specification':supplement.bound(args.output),'numerical_evaluations':0,
        'compiler_calls':0,'core_dataset_identity_asserted':False}


def check_orchestration_head(spec):
    actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if actual != spec['operational_source_state']['actual_git_head']:
        raise ValueError('Orchestration HEAD changed during the declared raw collection')
    return actual


def process_identity(process):
    base = Path('/proc')/str(process.pid)
    if not base.exists():
        return {'pid':process.pid,'exited':True,'returncode':process.poll()}
    fields = (base/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':process.pid,'ppid':int(fields[1]),'state':fields[0],
        'start_ticks':int(fields[19]),'pgid':os.getpgid(process.pid),
        'command':(base/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),
        'affinity':sorted(os.sched_getaffinity(process.pid))}


def ledger(report, event, **fields):
    value = {'event':event,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields}
    with (report/'process-ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def terminate_group(process):
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: pass
    try: process.wait(timeout=10.)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()
    # The group may retain a child after the direct make/copy process exits.
    try: os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError: pass


@contextmanager
def owned_processes(work,report):
    """Operational subprocess ownership only; inherited Worker math is intact."""
    original_run, original_worker, original_subprocess = bench.run,workers.Worker,workers.subprocess
    serial = [0]

    def run(command,cwd,log,**kwargs):
        timeout = kwargs.pop('timeout',21600.)
        if kwargs or timeout > 21600.:
            raise ValueError('Unexpected build subprocess override')
        log.parent.mkdir(parents=True,exist_ok=True)
        start = time.perf_counter()
        with log.open('x') as stream:
            process = subprocess.Popen(command,cwd=cwd,env=workers.environment(bench.WORK),
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                ledger(report,'build-start',identity=process_identity(process),command=command,
                       cwd=str(cwd),log=policy.relative(log),timeout_seconds=timeout)
                code = process.wait(timeout=timeout)
            finally:
                terminate_group(process)
                ledger(report,'build-exit',pid=process.pid,returncode=process.returncode)
        if code:
            raise RuntimeError(f'{command[0]} failed ({code}); see {log}')
        return time.perf_counter()-start

    class OwnedWorker(original_worker):
        def __setattr__(self,name,value):
            super().__setattr__(name,value)
            if name == 'process' and value is not None:
                ledger(report,'worker-start',identity=process_identity(value),
                       directory=str(self.directory),log=policy.relative(self.log),
                       protocol_timeout_seconds=self.timeout)
        def close(self,*args,**kwargs):
            try: return super().close(*args,**kwargs)
            finally:
                if self.process is not None:
                    ledger(report,'worker-close',pid=self.process.pid,
                           returncode=self.process.poll(),closed=self.closed)

    class SubprocessProxy:
        def __getattr__(self,name): return getattr(original_subprocess,name)
        def run(self,command,**kwargs):
            # workers.prepare's only active subprocess.run in this fixed
            # fresh_parity=False setup is the private reflink/copy operation.
            if command[:3] != ['cp','-a','--reflink=auto'] or kwargs != {'check':True}:
                raise ValueError('Unexpected frozen worker subprocess.run call')
            serial[0] += 1
            log = report/'logs'/f'copy-{serial[0]:02d}.log'
            run(command,ROOT,log)
            return subprocess.CompletedProcess(command,0)

    bench.run,workers.Worker,workers.subprocess = run,OwnedWorker,SubprocessProxy()
    try: yield
    finally: bench.run,workers.Worker,workers.subprocess = original_run,original_worker,original_subprocess


def collect(args):
    specification = policy.path(args.specification)
    spec = policy.verified(read(specification),SCHEMA)
    if not args.execute_numerical:
        raise ValueError('Root must explicitly hand off the serialized numerical slot')
    if 'expected' in spec or 'core_dataset_identity_sha256' in spec:
        raise ValueError('Early raw collection cannot assert a future core dataset')
    if spec['cpu'] != CPU or CPU not in os.sched_getaffinity(0):
        raise ValueError('All three stress variants require the fixed available CPU190')
    handoff = read(args.handoff)
    if (handoff.get('root_authorized') is not True
            or handoff.get('phase') != 'early-stress-raw-collection'
            or handoff.get('cpu') != CPU or handoff.get('variant') != spec['variant']
            or handoff.get('specification_sha256') != supplement.digest(specification)):
        raise ValueError('Missing or mismatched explicit root numerical handoff')
    for record in spec['sources'].values(): supplement.checked(record)
    orchestration_revision = check_orchestration_head(spec)
    result,registration,signature = cohort(supplement.checked(spec['early_result']))
    for key,actual in (('generation_revision',registration['generation_revision']),
            ('numerical_revision',registration['numerical_revision']),
            ('generator_identity_sha256',registration['generator_identity_sha256']),
            ('installed_archive_sha256',registration['archive_sha256']),('runtime_signature',signature)):
        if spec[key] != actual: raise ValueError('Raw specification belongs to another actual cohort')
    variant = spec['variant']
    config = stress.configuration(variant)
    if spec['configuration'] != config:
        raise ValueError('Fixed stress configuration changed')
    report,work = policy.path(spec['report']),policy.path(spec['work'])
    if report.exists() or work.exists():
        raise ValueError('Retain this collection attempt and choose a new namespace')
    report.mkdir(parents=True);work.mkdir(parents=True)
    for name in ('sources','points'): (report/name).mkdir()
    for name,record in spec['sources'].items(): shutil.copy2(supplement.checked(record),report/'sources'/name)
    shutil.copy2(specification,report/'input-spec.json')
    shutil.copy2(args.handoff,report/'root-handoff.json')
    for case in config['cases']:
        shutil.copy2(supplement.checked(result['point_files'][case['id']]),report/'points'/(case['id']+'.dat'))
    descriptor = workers.protocol_descriptor()
    bench.dump(report/'protocol.json',{'schema':workers.SCHEME,'configuration':config,
        'protocol_descriptor':descriptor,'protocol_identity_sha256':bench.content_digest(descriptor),
        'cpu':CPU,'repetitions':2,'sources':'candidate','backends':list(BACKENDS),
        'fresh_parity':False,'acceptance_eligible':False,'performance_acceptance_gate':False,
        'stress_variant':variant,'role_map':ROLES,
        'orchestration_revision':orchestration_revision,
        'actual_cli_generation_revision':registration['generation_revision'],
        'input_adapter':supplement.bound(report/'sources/collect.py'),
        'input_spec':supplement.bound(report/'input-spec.json'),
        'core_dataset_link':'not-yet-available; no core identity asserted'})
    status = {'status':'incomplete','requested_cases':[case['id'] for case in config['cases']],
        'completed_cases':[],'core_dataset_link':'pending','final_dataset_claim':False}
    bench.dump(report/'run-status.json',status)
    aliases = {case['id']:case['id'][:-len('-'+variant)] if variant != 'ordinary' else case['id']
               for case in config['cases']}
    roots = {'candidate':policy.relative(policy.path(registration['work'])/'outputs/candidate')}
    proofs = {'candidate':result['source_proofs']}
    worker_args = SimpleNamespace(cpu=CPU,repetitions=2,minimum_seconds=1e-12,
        timeout=86400.,fresh_parity=False,clean_build=True)
    sentinel = work/'input-root-sentinel'
    adapter = stress.scale_adapter(supplement.checked(result['scale_schedule']),report) if variant == 'scale' else nullcontext()
    try:
        with supplement.harness(sentinel,report), supplement.input_roots(roots,sentinel,aliases), \
                owned_processes(work,report), adapter:
            for case in config['cases']:
                workers.run_case(case,SOURCES,BACKENDS,config,worker_args,work/'workers',report,spec['run_id'])
                check_orchestration_head(spec)
                status['completed_cases'].append(case['id'])
                bench.dump(report/'run-status.json',status)
        cohort(supplement.checked(spec['early_result']))
        check_orchestration_head(spec)
        for record in spec['sources'].values(): supplement.checked(record)
        status['status'] = 'complete'
        bench.dump(report/'run-status.json',status)
        workers.checked_series(report,minimum=2)
        binding = supplement.bind_collected(report,work,ROLES,config,
            {'output_roots':roots,'source_proofs':proofs,'case_aliases':aliases})
        pending = policy.seal({'schema':'stress-early-raw-result-v1',
            'status':'complete-raw-only-awaiting-real-core-link',
            'early_result':spec['early_result'],'input_specification':supplement.bound(report/'input-spec.json'),
            'protocol':supplement.bound(report/'protocol.json'),
            'run_status':supplement.bound(report/'run-status.json'),
            'bindings':supplement.bound(report/'source-bindings.json'),
            'process_ledger':supplement.bound(report/'process-ledger.jsonl'),
            'variant':variant,'cpu':CPU,'runtime_signature':signature,
            'final_dataset_claim':False,'performance_acceptance_gate':False})
        exclusive_json(report/'early-raw-result.json',pending)
        return pending
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error))
        bench.dump(report/'run-status.json',status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    create = commands.add_parser('write-spec',help='Bind completed early assets only; never evaluate')
    create.add_argument('--early-result',required=True,type=Path)
    create.add_argument('--variant',required=True,choices=tuple(stress.VARIANTS))
    create.add_argument('--report',required=True,type=Path)
    create.add_argument('--work',required=True,type=Path)
    create.add_argument('--output',required=True,type=Path)
    run = commands.add_parser('collect',help='Requires the later explicit serialized engine handoff')
    run.add_argument('specification',type=Path)
    run.add_argument('--execute-numerical',action='store_true')
    run.add_argument('--handoff',required=True,type=Path)
    args = parser.parse_args()
    result = write_spec(args) if args.action == 'write-spec' else collect(args)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    def interrupted(signum,_frame):
        raise SystemExit(f'Controller interrupted by signal {signum}; owned cleanup follows')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGHUP,interrupted)
    main()
