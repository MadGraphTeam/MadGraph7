#!/usr/bin/env python3
"""Final pinned-release NinjaRS core correctness, never a timing gate.

Reuse the unchanged persistent worker and actual eight CLI exports. No source
generation, legacy backend evaluation, or 100-point fresh-parity claim occurs.
"""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'docs/benchmarkfolder/policy-v2/final-stress-raw-collection-v1'))
import collect as operations
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_workers as workers
import ninjars_supplement_v2 as supplement

HERE = Path(__file__).resolve().parent
CORE = HERE.parent/'generation'
CORE_WORK = ROOT/'.benchmark-work/policy-v3/final-pins-20261021-v1/generation'
ORDER = ('qcd-4','ew-4','qcd-1','ew-1','qcd-2','ew-2','qcd-3','ew-3')
CPU = 190
THREAD_ENV = {name:'1' for name in
              ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS')}
SCHEMA = 'final-pins-core-correctness-spec-v1'
USER_QUOTE = ('Continue as planned, but focus first on getting a validated MadLoop+NinjaRS version. '
              'Then if you uncovered issues with other reduction tools (e.g. COLLIER), '
              'then just post an issue with an MRE to the corresponding github')


def read(path):
    return json.loads(Path(path).read_text())


def configuration():
    original = read(ROOT/'docs/benchmarkfolder/policy-v2/cases.json')
    value = copy.deepcopy(original)
    cases = {case['id']:case for case in value['cases']}
    value['cases'] = [cases[cid] for cid in ORDER]
    value['repetitions'] = 2
    if (value['points_per_case'] != 100 or value['seed'] != 410083 or
            value['madloop_parameters'] != {'MLStabThres':1e-9,'COLLIERRequiredAccuracy':1e-12}):
        raise ValueError('The original core physics configuration changed')
    return value


def frozen_sources():
    paths = {name:policy.path(item['path']) for name,item in operations.frozen_sources().items()}
    # These additional imported adapters are bound by their real current bytes.
    paths.update({'core_collect.py':Path(__file__), 'launch.py':HERE/'launch.py',
                  'owned_operations.py':Path(operations.__file__),
                  'ninjars_benchmark_pair_v3.py':Path(pair.__file__),
                  'ninjars_benchmark_references_v2.py':ROOT/'tools/ninjars_benchmark_references_v2.py'})
    return {name:supplement.bound(path) for name,path in paths.items()}


def numerical_state(numerical, generator):
    paths = (*bench.GENERATOR_PATHS,'vendor')
    for revision in ('HEAD', numerical):
        subprocess.run(['git','diff','--exit-code',revision,'--',*paths],cwd=ROOT,check=True)
    additions = set(subprocess.check_output(
        ['git','ls-files','--others','--exclude-standard','--',*paths],cwd=ROOT,text=True).splitlines())
    if additions - {'vendor/StdHEP/fail'}:
        raise ValueError('Uncommitted numerical/generator sources are present')
    state = bench.generator_tree_state()
    if bench.content_digest(state) != generator:
        raise ValueError('Actual generator differs from the eight CLI exports')
    return {'orchestration_revision':subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'generator_identity_sha256':generator,'numerical_revision':numerical,
            'retained_untracked_diagnostics':sorted(additions)}


def cohort():
    """Read-only byte checks; no configuration, compiler, or amplitude call."""
    status = read(CORE/'status.json')
    freeze = read(CORE/'executed-freeze.json')
    generation = status['git_revision']
    generator = status['generator_tree_state_sha256']
    numerical = freeze['committed_revision']
    archive = freeze['installation_files_sha256']['lib/libninjars.a']
    if generation != numerical:
        raise ValueError('Generation differs from the committed final source freeze')
    state = numerical_state(numerical, generator)
    inventory = policy.verified(read(CORE/'runtime-inventory.json'),'fresh-generated-runtime-inventory-v2')
    if (not all(status.get(key) is True for key in ('complete','preserved_existing_inputs',
            'dependencies_unchanged','original_archive_unchanged')) or set(status['cases']) != set(ORDER)):
        raise ValueError('The actual eight CLI exports are incomplete or from another source')
    if read(supplement.checked(inventory['status'])) != status:
        raise ValueError('Runtime inventory and actual generation status differ')
    groups = inventory['runtime_groups']['8']
    if len(groups) != 1:
        raise ValueError('Every new runtime group requires its own representative proof')
    runtime, group = next(iter(groups.items()))
    signature = policy.verified(group['signature'],'reducer-runtime-signature-v2')
    if (signature['identity_sha256'] != runtime or archive not in signature['libraries'].values()
            or set(group['cases']) != set(ORDER)):
        raise ValueError('The final archive/full runtime inventory differs')
    original_config = read(ROOT/'docs/benchmarkfolder/policy-v2/cases.json')
    config = configuration()
    cases = {}
    from madgraph.various.banner import MadLoopParam
    for case in config['cases']:
        cid = case['id']; actual = status['cases'][cid]
        output = CORE_WORK/'outputs/candidate'/cid
        generation_record = read(output/'generation.json')
        compilation = read(output/'compilation.json')
        manifest = bench.generated_source_manifest(output)
        identity = generation_record['source_identity']
        if (actual['status'] != 'complete' or actual['case'] != case or
                identity['revision'] != generation or identity['tree_state_sha256'] != generator or
                identity.get('dirty_patch_sha256') is not None or identity['tree_state']['untracked'] or
                manifest != read(CORE/f'inputs/candidate/{cid}/compiled-sources.json') or
                manifest['files_sha256'] != compilation['generated_sources_sha256'] or
                operations.actual_runtime(output,manifest) != signature or
                inventory['cases'][cid]['signatures']['8'] != runtime):
            raise ValueError('An actual core export/source/full runtime changed: '+cid)
        executable = next(output.glob('SubProcesses/P*_*/check'))
        if supplement.digest(executable) != compilation['executable_sha256']:
            raise ValueError('An original core executable changed: '+cid)
        physical = CORE/f'inputs/candidate/{cid}/param_card.dat'
        if (output/'Cards/param_card.dat').read_bytes() != physical.read_bytes():
            raise ValueError('A physical parameter card changed: '+cid)
        if not MadLoopParam(str(output/'Cards/MadLoopParams.dat'))['UseQPIntegrandForNinjaRS']:
            raise ValueError('The actual inherited NinjaRS QP integrand option is disabled')
        points = CORE/f'points/{cid}.dat'; metadata = read(CORE/f'points/{cid}.json')
        index = next(i for i,c in enumerate(original_config['cases']) if c['id']==cid)
        if (metadata['points'] != 100 or metadata['seed'] != original_config['seed']+index or
                metadata['sha256'] != supplement.digest(points) or metadata['masses'] != actual['masses'] or
                metadata['energy_gev'] != 1000.):
            raise ValueError('Core point identity/seed/masses changed: '+cid)
        supplement.point_metadata(points, actual['masses'], case['nfinal'])
        cases[cid] = {'case':case,'output':policy.relative(output),
            'generation':supplement.bound(output/'generation.json'),
            'compilation':supplement.bound(output/'compilation.json'),
            'compiled_manifest':supplement.bound(CORE/f'inputs/candidate/{cid}/compiled-sources.json'),
            'physical_card':supplement.bound(physical),'madloop_card':supplement.bound(output/'Cards/MadLoopParams.dat'),
            'points':supplement.bound(points),'point_metadata':supplement.bound(CORE/f'points/{cid}.json'),
            'full_runtime_identity_sha256':runtime,'executable_sha256':compilation['executable_sha256']}
    representative_path = HERE.parent/'representative/representative-proof.json'
    representative = policy.verified(read(representative_path),'representative-policy-validation-v2')
    if runtime not in representative['reducer_runtimes']['candidate_corrected/8']:
        raise ValueError('The new runtime lacks its own fresh/QP representative proof')
    return {'source_state':state,'runtime_signature':signature,'cases':cases,
            'candidate_cohort':{'numerical_revision':numerical,'generation_revision':generation,
                'generator_identity_sha256':generator,'runtime_identity_sha256':runtime,
                'archive_sha256':archive},
            'executed_freeze':supplement.bound(CORE/'executed-freeze.json'),
            'generation_status':supplement.bound(CORE/'status.json'),
            'runtime_inventory':supplement.bound(CORE/'runtime-inventory.json'),
            'representative_proof':supplement.bound(representative_path),
            'representative_identity_sha256':representative['identity_sha256']}


def write_spec(args):
    checked = cohort()
    value = policy.seal({'schema':SCHEMA,'status':'held-unattempted','user_priority_amendment':USER_QUOTE,
        'configuration':configuration(),'original_policy':supplement.bound(ROOT/'docs/benchmarkfolder/policy-v2/cases.json'),
        'cpu':CPU,'controller_allowed_cpus':[6,7,190],'evaluation_thread_environment':THREAD_ENV,
        'sources':frozen_sources(),'cohort':checked,
        'source_roles':{'candidate':'candidate_corrected'},'scheduled_sources':['candidate'],'backends':[8],
        'report':policy.relative(args.report),'work':policy.relative(args.work),
        'fresh_parity':False,'fresh_parity_scope':'New-runtime representative proof only; no100-point fresh-process claim',
        'repetitions':2,'minimum_seconds':1e-12,'worker_timeout_seconds':86400.,'build_timeout_seconds':21600.,
        'strict_statuses':[218,228,322],'acceptance_eligible':False,'performance_acceptance_gate':False,
        'seven_repetition_timing_gate':False,'final_dataset_claim':False})
    operations.exclusive_json(args.output,value)
    return {'specification':supplement.bound(args.output),'numerical_evaluations':0,'compiler_calls':0}


def check_sources(spec):
    for item in spec['sources'].values(): supplement.checked(item)
    supplement.checked(spec['original_policy'])
    actual = cohort()
    # Orchestration-only commits are recorded accurately; measured bytes remain exact.
    before = copy.deepcopy(spec['cohort']); after = copy.deepcopy(actual)
    for value in (before,after): value['source_state'].pop('orchestration_revision')
    if before != after:
        raise ValueError('Actual source/runtime/physical-input cohort changed')
    return actual['source_state']


def strict_records(report,cid):
    records = list((report/'data').glob(cid+'-candidate-8-*.json'))
    if len(records) != 3:
        raise ValueError('Expected exactly calibration plus two retained traversals')
    for path in records:
        item = read(path)
        settings = item['shared_numerical_settings']
        required = {'mlstabthres':1e-9,'collierrequiredaccuracy':1e-12,
                    'colliercomputeuvpoles':True,'colliercomputeirpoles':True,
                    'selected_backend_qp_integrand':True,'useloopfilter':False}
        if any(settings.get(k) != v for k,v in required.items()):
            raise ValueError('A fixed numerical setting changed: '+str(path))
        if (item['backend'] != 8 or item['cpu'] != CPU or item['loops'] != 1 or
                any(row['status'] not in (218,228,322) for row in item['rows']) or
                any(int(code) not in (218,228,322) for code in item['timed_status_counts'])):
            raise ValueError('NinjaRS-only strict status/point stream gate failed: '+str(path))


def collect(args):
    spec = policy.verified(read(args.specification),SCHEMA)
    handoff = read(args.handoff)
    if (not args.execute_numerical or handoff.get('root_authorized') is not True or
            handoff.get('phase') != 'final-pins-core-correctness' or handoff.get('cpu') != CPU or
            handoff.get('specification_sha256') != supplement.digest(args.specification)):
        raise ValueError('Explicit matching root numerical handoff is required')
    if CPU not in os.sched_getaffinity(0):
        raise ValueError('Controller must includeCPU190; all numerical workers are pinned190')
    if any(os.environ.get(name)!=value for name,value in THREAD_ENV.items()):
        raise ValueError('Use the retained launcher with the fixed single-thread environment')
    if spec['configuration'] != configuration() or spec['backends'] != [8] or spec['scheduled_sources'] != ['candidate']:
        raise ValueError('The fixed NinjaRS-only schedule changed')
    before = check_sources(spec)
    report,work = policy.path(spec['report']),policy.path(spec['work'])
    if report.exists() or work.exists(): raise ValueError('Preserve this attempt; destination already exists')
    report.mkdir(parents=True);work.mkdir(parents=True)
    for name in ('sources','points','source-proofs'): (report/name).mkdir()
    for name,item in spec['sources'].items(): shutil.copy2(supplement.checked(item),report/'sources'/name)
    shutil.copy2(args.specification,report/'input-spec.json');shutil.copy2(args.handoff,report/'root-handoff.json')
    config = spec['configuration'];proofs={}
    for case in config['cases']:
        cid=case['id'];item=spec['cohort']['cases'][cid]
        shutil.copy2(supplement.checked(item['points']),report/'points'/(cid+'.dat'))
        proof=report/'source-proofs'/(cid+'.json');bench.dump(proof,item);proofs[cid]=supplement.bound(proof)
    descriptor=workers.protocol_descriptor()
    bench.dump(report/'protocol.json',{'schema':workers.SCHEME,'configuration':config,
        'protocol_descriptor':descriptor,'protocol_identity_sha256':bench.content_digest(descriptor),
        'cpu':CPU,'repetitions':2,'sources':'candidate','backends':[8],'fresh_parity':False,
        'acceptance_eligible':False,'performance_acceptance_gate':False,'input_adapter':supplement.bound(report/'sources/core_collect.py'),
        'input_spec':supplement.bound(report/'input-spec.json'),'user_priority_amendment':USER_QUOTE,
        'actual_cli_generation_revision':spec['cohort']['candidate_cohort']['generation_revision'],
        'numerical_revision':spec['cohort']['candidate_cohort']['numerical_revision'],
        'candidate_cohort':spec['cohort']['candidate_cohort'],
        'fresh_parity_scope':spec['fresh_parity_scope']})
    status={'status':'incomplete','requested_cases':[case['id'] for case in config['cases']],'completed_cases':[],'source_checkpoints':[before],
            'final_dataset_claim':False,'performance_acceptance_gate':False}
    bench.dump(report/'run-status.json',status)
    # The baseline resolver is available to shared helpers but never scheduled.
    roots={'candidate':policy.relative(CORE_WORK/'outputs/candidate'),
           'baseline':'.benchmark-work/outputs/baseline'}
    args_worker=SimpleNamespace(cpu=CPU,repetitions=2,minimum_seconds=1e-12,timeout=86400.,fresh_parity=False,clean_build=True)
    sentinel=work/'input-root-sentinel'
    try:
        with supplement.harness(sentinel,report),supplement.input_roots(roots,sentinel):
            for case in config['cases']:
                with pair.owned_preparation(report,label=case['id']):
                    workers.run_case(case,('candidate',),(8,),config,args_worker,work/'workers',report,'final-pins-core-correctness-v1')
                strict_records(report,case['id'])
                status['source_checkpoints'].append(check_sources(spec))
                status['completed_cases'].append(case['id']);bench.dump(report/'run-status.json',status)
        status['status']='complete'
        bench.dump(report/'run-status.json',status)
        workers.checked_series(report,minimum=2)
        binding=supplement.bind_collected(report,work,{'candidate':'candidate_corrected'},config,
            {'output_roots':roots,'source_proofs':{'candidate':proofs}})
        if any(item['reducer_runtime']['identity_sha256'] != spec['cohort']['candidate_cohort']['runtime_identity_sha256']
               for item in binding['targets'].values()):
            raise ValueError('A collected worker used another full runtime')
        status['source_checkpoints'].append(check_sources(spec));status['status']='complete'
        bench.dump(report/'run-status.json',status)
        result=policy.seal({'schema':'final-pins-core-correctness-result-v1','status':status['status'],
            'input_specification':supplement.bound(report/'input-spec.json'),'run_status':supplement.bound(report/'run-status.json'),
            'protocol':supplement.bound(report/'protocol.json'),
            'bindings':supplement.bound(report/'source-bindings.json'),'process_ledger':supplement.bound(report/'process-ledger.jsonl'),
            'performance_acceptance_gate':False,'seven_repetition_timing_gate':False,'final_dataset_claim':False})
        operations.exclusive_json(report/'raw-result.json',result)
        return result
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error))
        bench.dump(report/'run-status.json',status)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='action',required=True)
    prepare=commands.add_parser('write-spec');prepare.add_argument('--report',type=Path,required=True)
    prepare.add_argument('--work',type=Path,required=True);prepare.add_argument('--output',type=Path,required=True)
    run=commands.add_parser('collect');run.add_argument('specification',type=Path)
    run.add_argument('--handoff',type=Path,required=True);run.add_argument('--execute-numerical',action='store_true')
    args=parser.parse_args();result=write_spec(args) if args.action=='write-spec' else collect(args)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    def interrupted(signum,_frame):
        raise SystemExit('Controller interrupted bysignal'+str(signum)+'; owned cleanup follows')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    main()
