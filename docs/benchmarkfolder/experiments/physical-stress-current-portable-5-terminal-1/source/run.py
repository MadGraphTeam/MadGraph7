#!/usr/bin/env python3
"""Stress suffix controller: inherited setup, fresh20 neverlaunched requests."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
SOURCE4 = HERE.with_name('physical-stress-current-supervisor-4')
OLD_WORK = ROOT / '.tirust-build/physical-stress-current-4'
OLD_REPORT = HERE.with_name('physical-stress-current-qualification-4')
WORK = ROOT / '.tirust-build/physical-stress-current-5'
REPORT = HERE.with_name('physical-stress-current-qualification-5')
STAGES = ('inherit', 'run')
CAPS = {'inherit': 600, 'run': 7600}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load('stress5_original_controller', SOURCE4 / 'run.py')
links = load('stress5_sealed_link_helper', HERE.with_name('current-cms-replay-supervisor-2') / 'link_closure.py')
policy = load('stress5_conditional_policy', HERE / 'policy.py')
PLAN = copy.deepcopy(base.PLAN)
L, q, sha, save, read, require = base.L, base.q, base.sha, base.save, base.read, base.require
L.WORK, L.REPORT, L.STAGES = WORK, REPORT, STAGES
q.WORK, q.REPORT = WORK, REPORT


def environment(cwd):
    env = base.environment(cwd)
    for name, value in list(env.items()):
        if isinstance(value, str) and value.startswith(str(OLD_WORK) + '/'):
            env[name] = str(WORK) + value[len(str(OLD_WORK)):]
    require(not any(str(OLD_WORK) in env.get(name, '') for name in
                   ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_STATE_HOME')),
            'Writable environment points at inherited work')
    return env


L.environment = environment


def seal():
    index = read(HERE / 'source-index.json')
    require(index.get('sealed_for_execution') is True, 'Source-only proposal: execution binding is absent')
    paths = list(HERE.rglob('*'))
    require({p.relative_to(HERE).as_posix() for p in paths if p.is_file()} == set(index['files']) | {'source-index.json'}
            and not any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths), 'Source inventory differs')
    for name, item in index['files'].items():
        require(sha(HERE / name) == item['sha256'], 'Source changed: ' + name)
    binding = read(HERE / 'binding.json')
    require(binding['sealed_for_execution'] is True, 'Unsealed continuation binding')
    for name in ('ancestor_index_sha256', 'ancestor_verifier_sha256',
                 'closed_master_audit_sha256', 'root_terminal_confirmation_sha256'):
        require(isinstance(binding[name], str) and re.fullmatch('[0-9a-f]{64}', binding[name]), 'Missing actual binding: ' + name)
    template = read(HERE / 'binding.template.json')
    for name in ('original_source_index_sha256', 'preparation_index_sha256',
                 'terminal_retention_design_index_sha256', 'mechanical_link_helper_sha256'):
        require(binding[name] == template[name], 'Original scientific/mechanical binding differs')
    require(sha(SOURCE4 / 'source-index.json') == binding['original_source_index_sha256'], 'Source4 index differs')
    require(sha(Path(links.__file__)) == binding['mechanical_link_helper_sha256'], 'Mechanical helper differs')
    require(sha(HERE.with_name('physical-stress-current-terminal-retention-proposal-1') / 'source-index.json') ==
            binding['terminal_retention_design_index_sha256'], 'Terminal retention design differs')
    proposal = HERE.with_name('physical-stress-current-continuation-proposal-1')
    require(sha(proposal / 'source-index.json') == binding['source_proposal_index_sha256'], 'Original continuation proposal differs')
    for name, item in read(proposal / 'source-index.json')['files'].items():
        require(sha(proposal / name) == item['sha256'], 'Original continuation proposal changed: ' + name)
    require(sha(proposal / 'closed-input-observation.json') == binding['original_observation_sha256'], 'Original closed observation differs')
    base.seal()
    return binding


def launch_inventory():
    """After closure, derive launched identities from authenticated report paths."""
    ids = []
    for request in PLAN['requests']:
        rid = policy.key(request)
        evidence = OLD_REPORT / 'runs' / request['case'] / request['label']
        execution = OLD_REPORT / 'execution' / ('request-' + request['case'] + '-' + request['label'] + '.json')
        if evidence.exists() or execution.exists():
            require((evidence / 'specification.json').is_file() and execution.is_file(), 'Partial/unexpected launch inventory')
            specification = read(evidence / 'specification.json')
            require(specification['request'] == request, 'Launched specification identity differs')
            ids.append(rid)
    return ids


def verify_closed_report(retention, snapshot):
    """Cross-bind the post-closure guard to the capsule's complete inventory."""
    require(retention['original_closed_root'] == str(OLD_REPORT), 'Retention donor report root differs')
    entries = snapshot['entries']
    files = {name: item for name, item in entries.items() if item['kind'] == 'regular_file'}
    directories = {name for name, item in entries.items() if item['kind'] == 'directory'}
    require(set(files) == set(retention['closed_files']) and directories | {'.'} == set(retention['closed_directories'])
            and len(files) + len(directories) == len(entries), 'Closed report capsule/audit inventory differs')
    for name, item in retention['closed_files'].items():
        require(files[name]['sha256'] == item['sha256'] and files[name]['size'] == item['size'],
                'Closed report capsule/audit byte binding differs: ' + name)


def verify_inherited_inputs():
    """Read-only, admitted closed roots only; no subprocess or loader probes."""
    binding = seal()
    capsule = HERE.with_name(binding['ancestor_capsule'])
    require(capsule.name == 'physical-stress-current-portable-4-terminal-2', 'Unexpected ancestor namespace')
    require(sha(capsule / 'evidence-index.json') == binding['ancestor_index_sha256'] and
            sha(capsule / 'verify_evidence.py') == binding['ancestor_verifier_sha256'], 'Ancestor capsule differs')
    verified = load('stress5_terminal4_verifier', capsule / 'verify_evidence.py').verify(capsule)
    require(verified['verified'] is True and verified['index_sha256'] == binding['ancestor_index_sha256']
            and verified['preparation_index_sha256'] == binding['preparation_index_sha256'], 'Ancestor proof binding differs')
    require(sha(capsule / 'qualification-summary.json') == binding['ancestor_summary_sha256'] and
            read(capsule / 'qualification-summary.json') == verified['summary'], 'Ancestor summary binding differs')
    audit_path = HERE / 'closed-master-audit.json'
    confirmation_path = HERE / 'root-terminal-confirmation.json'
    require(sha(audit_path) == binding['closed_master_audit_sha256'] and
            sha(confirmation_path) == binding['root_terminal_confirmation_sha256'], 'Closed audit/confirmation differs')
    confirmation = read(confirmation_path)
    require(confirmation == {'observed_terminal': True, 'session_id': 29078, 'exit_code': 1,
                            'owned_cleanup_complete': True, 'ancestor_index_sha256': binding['ancestor_index_sha256']},
            'Missing genuine terminal4 confirmation')
    audit = read(audit_path)
    require(audit['schema'] == 'physical-stress-closed-input-audit-v1'
            and audit['observed_only_after_root_terminal'] is True
            and audit['ancestor_index_sha256'] == binding['ancestor_index_sha256'], 'Unbound post-closure audit')
    original = read(HERE.with_name('physical-stress-current-continuation-proposal-1') / 'closed-input-observation.json')
    expected_audit = {**original, 'ancestor_index_sha256': binding['ancestor_index_sha256'],
                      'original_observation_sha256': binding['original_observation_sha256'], 'root_admitted_terminal_capsule': True}
    require(audit == expected_audit, 'Closed audit is not the exact admitted observation with ancestry metadata')
    require(audit['closed_report']['root'] == str(OLD_REPORT), 'Unexpected donor report root')
    # Terminal retention must inventory the complete closed original report,
    # including preparation bytes referenced from the pinned sibling capsule.
    verify_closed_report(read(capsule / 'retention.json'), audit['closed_report'])
    links.verify_snapshot(audit['closed_report'])
    expected_masters = {source + '/' + entry['case']['id'] for entry in PLAN['cases'] for source in ('baseline', 'candidate')}
    require(set(audit['masters']) == expected_masters, 'Expected exactly eight inherited masters')
    for name, snapshot in audit['masters'].items():
        require(snapshot['root'] == str(OLD_WORK / 'exports' / name), 'Unexpected inherited master root')
        links.verify_snapshot(snapshot)
    links.verify_external_graph(audit['external_graph'])
    selected = policy.select(PLAN, verified['summary'], read(OLD_REPORT / 'stage-run.json'),
                             read(OLD_REPORT / 'request-accounting.json'), read(OLD_REPORT / 'aborted-request.json'), launch_inventory())
    base.frozen()
    builds = {}
    for entry in PLAN['cases']:
        for source in ('baseline', 'candidate'):
            name = source + '/' + entry['case']['id']; b = base.build_record(source, entry)
            prefix = OLD_REPORT / 'builds' / (source + '-' + entry['case']['id'])
            pre = read(prefix / 'preflight.json')
            require(pre['build_sha256'] == sha(prefix / 'build.json'), 'Inherited build/preflight mismatch')
            base.check_hashes(pre['runtime']['files'])
            # Every historically used leaf is also an explicit current read-only
            # audit leaf. Ambient directory membership is a new observation,
            # never retrospectively claimed to have been frozen at build time.
            for path, checksum in {**b['bound_inputs_before_after'], **pre['runtime']['files']}.items():
                leaf = audit['external_graph']['historical_leaf_bindings'].get(str(Path(path).resolve()), {})
                require(leaf.get('observed_sha256') == checksum, 'Historically used leaf omitted/changed')
            builds[name] = {'build_sha256': sha(prefix / 'build.json'), 'preflight_sha256': sha(prefix / 'preflight.json'),
                            'executable_sha256': b['executable_sha256'], 'runtime': pre['runtime'], 'masses': pre['masses']}
    return {'ancestor_index_sha256': binding['ancestor_index_sha256'], 'audit_sha256': sha(audit_path),
            'source4_index_sha256': binding['original_source_index_sha256'],
            'source_freeze_sha256': sha(OLD_REPORT / 'source-freeze.json'), 'builds': builds, 'requests': selected}


def inherit():
    require(not REPORT.exists() and not WORK.exists(), 'Fresh successor namespaces required')
    REPORT.mkdir(); WORK.mkdir()
    before = verify_inherited_inputs(); after = verify_inherited_inputs()
    require(before == after, 'Inherited checkpoint changed')
    save(REPORT / 'inherited-checkpoint.json', {'kind': 'read-only-inherited-stress-preparation', 'checkpoint': before,
         'inherited_stages': {'prepare': {'session_id': 25421, 'exit_code': 0}, 'preflight': {'session_id': 95469, 'exit_code': 0}},
         'new_setup_launches': 0, 'new_mass_probe_launches': 0, 'inherited_completed_requests': 4,
         'continuation_requests': 20, 'amplitudes_executed_in_this_stage': 0, 'performance_eligible': False})


def frozen():
    state = read(REPORT / 'inherited-checkpoint.json')
    require(state['checkpoint'] == verify_inherited_inputs(), 'Inherited checkpoint changed')
    return state


def private_destinations(output, directory):
    paths = [output / 'Cards' / name for name in ('param_card.dat', 'MadLoopParams.dat')]
    paths += [directory / name for name in ('MadLoop5_resources', 'worker-raw', 'worker-protocol.fifo',
              'dimensions.dat', 'warmup.dat', 'timing.dat', 'values.dat', 'timed_status.dat')]
    paths += list((output / 'SubProcesses').rglob('MadLoopParams.dat'))
    paths += list((output / 'SubProcesses').rglob('*HelFilter.dat')) + list((output / 'SubProcesses').rglob('*LoopFilter.dat'))
    for path in paths: links.assert_private_path(output, path)


def allowed_worker_changes(output, directory, context):
    prefix = directory.relative_to(output).as_posix() + '/'
    changed = {'Cards/MadLoopParams.dat'} | {name for name in context['expected_worker']['entries'] if name.endswith('/MadLoopParams.dat')}
    additions = {}
    outputs = {prefix + n for n in ('dimensions.dat', 'warmup.dat', 'timing.dat', 'values.dat', 'timed_status.dat')}
    for path in output.rglob('*'):
        name = path.relative_to(output).as_posix()
        if name in context['expected_worker']['entries']: continue
        raw = name.removeprefix(prefix + 'worker-raw/')
        known_raw = name == prefix + 'worker-raw' or (name.startswith(prefix + 'worker-raw/') and (
            re.fullmatch(r'session-[a-z0-9_]{8}(?:/(?:startup|previous-outputs|[0-9]{6})(?:/(?:timing|values|timed_status|dimensions|warmup)\.dat)?)?', raw)
            or re.fullmatch(r'session-[a-z0-9_]{8}/(?:startup/startup\.json|[0-9]{6}/command\.json)', raw)))
        permitted = name in outputs or re.fullmatch(r'SubProcesses/MadLoop5_resources/ML5_\d+_(?:Hel|Loop)Filter\.dat', name) or known_raw
        kind = 'regular_file' if name.endswith(('.dat', '.json')) else 'directory'
        require(permitted and not path.is_symlink() and (path.is_file() if kind == 'regular_file' else path.is_dir()), 'Unexpected worker output: ' + name)
        links.assert_private_path(output, path); additions[name] = kind
    return changed, additions


def run_request(request):
    entry = next(e for e in PLAN['cases'] if e['case']['id'] == request['case'])
    source, cid, label = request['source'], request['case'], request['label']
    evidence = REPORT / 'runs' / cid / label; evidence.mkdir(parents=True)
    before = verify_inherited_inputs(); save(evidence / 'inherited-before.json', before)
    output = WORK / 'workers' / cid / label; context = None; directory = None
    try:
        b = base.build_record(source, entry); pre = read(OLD_REPORT / 'builds' / (source + '-' + cid) / 'preflight.json')
        audit = read(HERE / 'closed-master-audit.json')
        context = links.prepare_worker(OLD_WORK / 'exports' / source / cid, output, audit['masters'][source + '/' + cid], audit['external_graph'])
        directory = output / entry['process_subdirectory']; private_destinations(output, directory)
        settings = base.configure_worker(output, request); private_destinations(output, directory)
        require(sha(directory / 'check') == b['executable_sha256'], 'Private worker executable differs')
        sources = base.scientific_sources(output); cardsha = sha(output / 'Cards/param_card.dat'); numerical = base.bench.numerical_card_inputs(output)
        actual = base.bench.resolved_parameter_card(directory); require(actual is not None, 'Missing physical card')
        links.assert_private_path(output, actual)
        require(sha(actual) == cardsha == entry['physical_card_sha256'], 'Shadowed physical card')
        require(sha(request['points']) == entry['point_sha256'], 'Frozen point stream changed')
        rt = base.runtime(directory / 'check'); require(rt['files'] == pre['runtime']['files'], 'Relocated runtime differs')
        specification = {'request': request, 'backend': request['backend'], 'cwd': str(directory),
            'argv_without_fifo': ['taskset', '-c', '12', './check', request['points']],
            'expected_runtime': {**rt['files'], str((directory / 'check').resolve()): b['executable_sha256']},
            'settings': settings, 'actual_stdin_after_READY': 'RUN 1 1\nSTOP 0 0\n', 'points_sha256': sha(request['points']),
            'physical_card_sha256': cardsha, 'scientific_sources': sources, 'numerical_cards': numerical, 'build': b, 'runtime': rt,
            'inherited_checkpoint_sha256': sha(REPORT / 'inherited-checkpoint.json'), 'ancestor_index_sha256': before['ancestor_index_sha256']}
        save(evidence / 'specification.json', specification)
        shutil.copy2(output / 'Cards/MadLoopParams.dat', evidence / 'MadLoopParams.dat'); shutil.copy2(actual, evidence / 'param_card.dat')
        failure = error_object = None
        try: base.cmd([sys.executable, '-B', SOURCE4 / 'native_job.py', evidence / 'specification.json'], 'request-' + cid + '-' + label, directory, 370)
        except Exception as error: failure = repr(error); error_object = error
        finally:
            for name in ('warmup.dat', 'dimensions.dat', 'values.dat', 'timed_status.dat', 'timing.dat'):
                if (directory / name).is_file() and not (evidence / name).exists(): shutil.copy2(directory / name, evidence / name)
            require(base.scientific_sources(output) == sources and base.bench.numerical_card_inputs(output) == numerical
                    and sha(actual) == cardsha and sha(directory / 'check') == b['executable_sha256'], 'Worker input changed')
        result = read(evidence / 'result.json') if (evidence / 'result.json').exists() else {
            'orchestration_complete': False, 'numerically_qualified': False, 'failure': failure, 'runtime_identity_failure': False}
        execution_path = REPORT / 'execution' / ('request-' + cid + '-' + label + '.json')
        execution = read(execution_path) if execution_path.exists() else None
        result, stop = base.admit_completion(result, error_object, execution)
        require(not stop, 'Fatal cleanup/runtime/controller failure: ' + str(stop))
    finally:
        try:
            try:
                if context is not None:
                    changed, added = allowed_worker_changes(output, directory, context)
                    links.assert_worker_inputs(output, context, allow_file_changes=changed, allow_new_paths=added)
            finally:
                after = verify_inherited_inputs(); save(evidence / 'inherited-after.json', after)
                require(after == before, 'Inherited input changed during request')
        except BaseException as error:
            save(evidence / 'completion.json', {'result': {'numerically_qualified': False}, 'outer_failure': repr(error),
                 'inputs_unchanged': False, 'fatal_infrastructure_failure': 'post-request input guard', 'performance_eligible': False})
            raise
    save(evidence / 'completion.json', {'result': result, 'outer_failure': failure, 'inputs_unchanged': True,
         'fatal_infrastructure_failure': stop, 'performance_eligible': False})
    return result


def run():
    requests = frozen()['checkpoint']['requests']; results = []; total = 0.
    for request in requests:
        lock = None; wait = time.monotonic(); start = None
        try:
            lock = base.acquire(request); wait = time.monotonic() - wait; start = time.monotonic()
            with L.Deadline(min(380, CAPS['run'] - total)): result = run_request(request)
            elapsed = time.monotonic() - start; total += elapsed
            results.append({'request': request, 'lock_wait_seconds': wait, 'elapsed_seconds': elapsed, 'result': result})
            save(REPORT / 'request-completions' / ('%02d.json' % len(results)), results[-1])
        except BaseException as error:
            save(REPORT / 'aborted-request.json', {'request': request, 'state': 'deferred-before-launch' if start is None else 'aborted-during-request',
                 'exception': repr(error), 'numerically_qualified': False}); raise
        finally:
            if lock is not None: lock.close()
    save(REPORT / 'summary.json', {'requests': results, 'request_count': 20, 'native_elapsed_seconds_excluding_lock_wait': total,
         'complete_checked_streams': sum(r['result']['numerically_qualified'] for r in results),
         'full24_coverage_requires_two_independently_verified_closed_proofs': True,
         'full24_coefficient_agreement': None, 'performance_eligible': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', choices=STAGES); parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    require(Path.cwd() == ROOT, 'Run from repository root')
    if not args.execute:
        print(json.dumps({'source_only': True, 'sealed_for_execution': True, 'commands_executed': 0,
              'conditional_requests': 20, 'previously_completed_requests_never_rerun': 4, 'stages': CAPS,
              'work': str(WORK), 'report': str(REPORT),
              'actual_closed_ancestor_binding': '6fc2035649ba0387f0d4457784ae883592f13bb7e3b6f9c33b99e038b81f7146',
              'native_stage_admission': 'root must separately admit inherit and run'}, indent=2)); return
    require(args.stage is not None, 'Explicit stage required'); seal()
    L.CURRENT_STAGE = args.stage; L.CPU = 12
    if args.stage == 'run':
        predecessor = read(REPORT / 'stage-inherit.json')
        require(predecessor['status'] == 'passed' and predecessor['source_index_sha256'] == sha(HERE / 'source-index.json')
                and predecessor['inherited_checkpoint_sha256'] == sha(REPORT / 'inherited-checkpoint.json'), 'Inheritance did not pass')
    marker = REPORT / ('stage-' + args.stage + '.json'); require(not marker.exists(), 'Append-only stage; no retry')
    start = time.monotonic(); failure = None
    with L.cleanup.SignalRequests() as requests:
        L.REQUESTS = requests
        try:
            if args.stage == 'run': run(); frozen()
            else:
                with L.Deadline(600): inherit(); frozen()
        except BaseException as error: failure = error
        finally:
            if REPORT.exists() and args.stage == 'run':
                selected = read(REPORT / 'inherited-checkpoint.json')['checkpoint']['requests']
                completed = {policy.key(read(p)['request']) for p in (REPORT / 'request-completions').glob('*.json')}
                save(REPORT / 'request-accounting.json', {'requested': 20, 'completed': len(completed),
                     'not_completed': [r for r in selected if policy.key(r) not in completed], 'native_stage_failure': None if failure is None else repr(failure)})
            if REPORT.exists():
                save(marker, {'stage': args.stage, 'status': 'passed' if failure is None else 'failed', 'exception': None if failure is None else repr(failure),
                     'elapsed_seconds_including_wait': time.monotonic() - start, 'cap_seconds_excluding_lock_wait': CAPS[args.stage],
                     'source_index_sha256': sha(HERE / 'source-index.json'), 'native_execution': args.stage == 'run',
                     'inherited_checkpoint_sha256': sha(REPORT / 'inherited-checkpoint.json') if (REPORT / 'inherited-checkpoint.json').is_file() else None,
                     'performance_eligible': False})
    if failure is not None: raise failure


if __name__ == '__main__': main()
