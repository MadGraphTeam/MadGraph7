#!/usr/bin/env python3
"""Controller-only CMS successor: read-only inheritance, then admitted native run."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
SOURCE1 = HERE.with_name('current-cms-replay-supervisor-1')
WORK = ROOT / '.tirust-build/current-cms-replay-2'
REPORT = HERE.with_name('current-cms-replay-qualification-2')
OLD_WORK = ROOT / '.tirust-build/current-cms-replay-1'
OLD_REPORT = HERE.with_name('current-cms-replay-qualification-1')
STAGES = ('inherit', 'run')
CAPS = {'inherit': 600, 'run': 3720}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# This sealed controller supplies unchanged science, comparison, ownership,
# native-driver and prerequisite mechanisms. Its donor read roots stay at 1.
base = load('cms2_sealed_controller1', SOURCE1 / 'run.py')
links = load('cms2_link_closure', HERE / 'link_closure.py')
PLAN, L, q = copy.deepcopy(base.PLAN), base.L, base.q
for request in PLAN['requests']:
    for key in ('cwd', 'fifo_argument_appended_by_worker'):
        old = request[key]
        if old.startswith(str(OLD_WORK) + '/'):
            request[key] = str(WORK) + old[len(str(OLD_WORK)):]
P, sha, save, require, read = base.P, base.sha, base.save, base.require, base.read
BINDING = read(HERE / 'binding.json')
CAPSULE = HERE.with_name(BINDING['ancestor_retention'])
AUDIT_PATH = HERE.parent / BINDING['audit_path']
L.REPORT, L.WORK, L.STAGES = REPORT, WORK, STAGES
q.REPORT, q.WORK = REPORT, WORK


def environment(cwd):
    env = base.environment(cwd)
    # These are writable caches/state, not scientific inputs. Never retain the
    # donor namespace's XDG or matplotlib destination in a successor process.
    for key, value in list(env.items()):
        if isinstance(value, str) and value.startswith(str(OLD_WORK) + '/'):
            env[key] = str(WORK) + value[len(str(OLD_WORK)):]
    require(not any(str(OLD_WORK) in env.get(key, '') for key in
                    ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME',
                     'XDG_STATE_HOME', 'XDG_DATA_HOME', 'XDG_RUNTIME_DIR')),
            'Writable environment still refers to donor work')
    return env


L.environment = environment


def seal():
    index = read(HERE / 'source-index.json')
    expected = set(index['files']) | {'source-index.json'}
    paths = list(HERE.rglob('*'))
    actual = {p.relative_to(HERE).as_posix() for p in paths if p.is_file()}
    require(actual == expected and not any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths),
            'Successor source inventory differs')
    for name, item in index['files'].items():
        require(sha(HERE / name) == item['sha256'], 'Successor source changed: ' + name)
    require(sha(SOURCE1 / 'source-index.json') == BINDING['source1_index_sha256'], 'Donor source index changed')
    require(sha(P / 'source-index.json') == BINDING['proposal1_index_sha256'], 'Donor proposal index changed')
    require(sha(SOURCE1 / 'run.py') == BINDING['inherited_controller_sha256'], 'Donor controller changed')
    require(sha(SOURCE1 / 'native_job.py') == BINDING['native_job_sha256'], 'Native protocol changed')
    require(sha(SOURCE1 / 'prerequisites.py') == BINDING['prerequisites_sha256'] and
            sha(SOURCE1 / 'prerequisites.json') == BINDING['prerequisites_manifest_sha256'], 'Prerequisite helper changed')
    require(sha(AUDIT_PATH) == BINDING['audit_sha256'], 'Closed link audit changed')
    require(sha(AUDIT_PATH.parent / 'source-index.json') == BINDING['inheritance_proposal_index_sha256'],
            'Closed inheritance proposal index changed')
    for name, item in read(AUDIT_PATH.parent / 'source-index.json')['files'].items():
        require(sha(AUDIT_PATH.parent / name) == item['sha256'], 'Closed inheritance proposal changed: ' + name)
    require(sha(CAPSULE / 'evidence-index.json') == BINDING['ancestor_index_sha256'] and
            sha(CAPSULE / 'verify_evidence.py') == BINDING['ancestor_verifier_sha256'], 'Failed ancestor capsule changed')
    base.seal()


def verify_original_evidence(retention):
    """Check donor report files, never an active campaign or a new native run."""
    require(Path(retention['original_closed_root']) == OLD_REPORT, 'Unexpected donor report root')
    paths = list(OLD_REPORT.rglob('*'))
    require(not any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths), 'Unexpected donor evidence object')
    files = {p.relative_to(OLD_REPORT).as_posix() for p in paths if p.is_file()}
    directories = {p.relative_to(OLD_REPORT).as_posix() for p in paths if p.is_dir()}
    require(files == set(retention['closed_files']) and directories == set(retention['closed_directories']),
            'Closed donor evidence inventory changed')
    for name, item in retention['closed_files'].items():
        path = OLD_REPORT / name
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Closed donor evidence changed: ' + name)


def verify_inherited_inputs():
    """Read-only guard; never invokes CLI, compiler, loader, model or worker."""
    seal()
    verifier = load('cms2_failed_ancestor_verifier', CAPSULE / 'verify_evidence.py')
    ancestor = verifier.verify(CAPSULE)
    require(ancestor['index_sha256'] == BINDING['ancestor_index_sha256'] and
            ancestor['summary']['overall_status'] == 'failed' and
            ancestor['summary']['amplitude_launch_count'] == 0, 'Failed ancestor identity/accounting differs')
    verify_original_evidence(read(CAPSULE / 'retention.json'))
    audit = read(AUDIT_PATH)
    require(audit['schema'] == 'cms-replay-link-closure-audit-v1', 'Unexpected link audit schema')
    for key in ('baseline_export', 'candidate_export', 'sealed_supervisor1', 'sealed_proposal1'):
        snapshot = audit['source1_master_guard_inventories'][key]
        links.verify_tree(snapshot['root'], snapshot)
    links.verify_external_graph(audit)
    base.frozen()
    builds = {}
    for source in ('baseline', 'candidate'):
        b = base.build_record(source, PLAN['cases'][0])
        pre, pre_sha = base.preflight_record(source, PLAN['case']['id'])
        base.check_hashes(pre['runtime']['files'])
        builds[source] = {'build_sha256': sha(OLD_REPORT / 'builds' / (source + '-' + PLAN['case']['id']) / 'build.json'),
                          'preflight_sha256': pre_sha, 'executable_sha256': b['executable_sha256'],
                          'probe_sha256': b['probe']['sha256'], 'runtime': pre['runtime'],
                          'probe_runtime': pre['probe_runtime'], 'complex_model': pre['complex_model']}
    return {'ancestor_index_sha256': ancestor['index_sha256'], 'audit_sha256': sha(AUDIT_PATH),
            'donor_source_index_sha256': BINDING['source1_index_sha256'],
            'source_freeze_sha256': sha(OLD_REPORT / 'source-freeze.json'), 'builds': builds}


def inherit():
    require(not REPORT.exists() and not WORK.exists(), 'Fresh successor namespaces required')
    REPORT.mkdir(); WORK.mkdir()
    before = verify_inherited_inputs()
    after = verify_inherited_inputs()
    require(before == after, 'Donor checkpoint changed during read-only inheritance')
    save(REPORT / 'inherited-checkpoint.json', {
        'schema_version': 1, 'kind': 'read-only-inherited-CMS-preparation',
        'ancestor_failed_native_attempt': BINDING['ancestor_index_sha256'],
        'inherited_stages': {'prepare': {'source': 'attempt1', 'exit_code': 0, 'session_id': 85858},
                             'preflight': {'source': 'attempt1', 'exit_code': 0, 'session_id': 11912}},
        'source2_executed_setup': False, 'subprocess_launch_count': 0,
        'mass_probe_launch_count': 0, 'CMS_model_probe_launch_count': 0,
        'new_ambient_directory_snapshot': BINDING['audit_sha256'], 'checkpoint': before,
        'qualification': 'inherited preparation only; no amplitudes', 'performance_eligible': False})


def frozen():
    checkpoint = read(REPORT / 'inherited-checkpoint.json')
    require(checkpoint['checkpoint'] == verify_inherited_inputs(), 'Inherited checkpoint changed')
    return checkpoint


def private_destinations(output, directory):
    """Prove all known configure/Fortran/FIFO/resource destinations private."""
    paths = [output / 'Cards/param_card.dat', output / 'Cards/MadLoopParams.dat',
             directory, directory / 'MadLoop5_resources', directory / 'worker-raw',
             directory / 'worker-protocol.fifo']
    paths += [directory / name for name in ('dimensions.dat', 'warmup.dat', 'timing.dat', 'values.dat', 'timed_status.dat')]
    paths += list((output / 'SubProcesses').rglob('MadLoopParams.dat'))
    resources = output / 'SubProcesses/MadLoop5_resources'
    paths += [resources / ('ML5_0_' + name) for name in ('HelFilter.dat', 'LoopFilter.dat')]
    for path in paths:
        links.assert_private_path(output, path)


def allowed_worker_changes(output, context):
    """Only private cards, known Fortran outputs/filters and raw archives vary."""
    prefix = 'SubProcesses/P0_epem_mupmum/'
    changed = {'Cards/MadLoopParams.dat'}
    changed.update(name for name in context['expected_worker']['entries'] if name.endswith('/MadLoopParams.dat'))
    allowed = {}
    for path in output.rglob('*'):
        name = path.relative_to(output).as_posix()
        if name in context['expected_worker']['entries']:
            continue
        raw = name.removeprefix(prefix + 'worker-raw/')
        permitted_raw = (name == prefix + 'worker-raw' or
                         (name.startswith(prefix + 'worker-raw/') and
                          re.fullmatch(r'session-[a-z0-9_]{8}(?:/(?:startup|previous-outputs|[0-9]{6})(?:/(?:timing|values|timed_status|dimensions|warmup)\.dat)?)?', raw)))
        permitted_raw = permitted_raw or (name.startswith(prefix + 'worker-raw/') and
                          re.fullmatch(r'session-[a-z0-9_]{8}/(?:startup/startup\.json|[0-9]{6}/command\.json)', raw))
        permitted = (name in {prefix + n for n in ('dimensions.dat', 'warmup.dat', 'timing.dat', 'values.dat', 'timed_status.dat')} or
                     name in {'SubProcesses/MadLoop5_resources/ML5_0_HelFilter.dat', 'SubProcesses/MadLoop5_resources/ML5_0_LoopFilter.dat'} or permitted_raw)
        expected_kind = 'regular_file' if name.endswith(('.dat', '.json')) else 'directory'
        require(permitted and not path.is_symlink() and
                (path.is_file() if expected_kind == 'regular_file' else path.is_dir()),
                'Unexpected worker output name/type: ' + name)
        links.assert_private_path(output, path)
        allowed[name] = expected_kind
    return changed, allowed


def run_request(request):
    entry = PLAN['cases'][0]; source = request['source']; cid = request['case']; label = request['label']
    evidence = REPORT / 'runs' / cid / label; evidence.mkdir(parents=True)
    before = verify_inherited_inputs(); save(evidence / 'inherited-before.json', before)
    output = WORK / 'workers' / label / cid
    context = None
    try:
        b = base.build_record(source, entry); pre, preflight_sha = base.preflight_record(source, cid)
        audit = read(AUDIT_PATH)
        expected = audit['source1_master_guard_inventories'][source + '_export']
        context = links.prepare_worker(OLD_WORK / 'exports' / source / cid, output, expected, audit)
        directory = output / entry['process_subdirectory']
        private_destinations(output, directory)
        settings = base.configure_worker(output, request)
        private_destinations(output, directory)
        require(sha(directory / 'check') == b['executable_sha256'], 'Worker executable differs')
        sources = base.scientific_sources(output); cardsha = sha(output / 'Cards/param_card.dat')
        numerical = base.bench.numerical_card_inputs(output)
        actual = base.bench.resolved_parameter_card(directory)
        require(actual is not None, 'Missing physical card')
        links.assert_private_path(output, actual)
        require(sha(actual) == cardsha == entry['physical_card_sha256'], 'Shadowed physical card')
        rt = base.runtime(directory / 'check')
        require(rt['files'] == pre['runtime']['files'], 'Relocated runtime differs')
        require(sha(request['points']) == PLAN['points_sha256'], 'Frozen points changed')
        specification = {'request': request, 'backend': request['backend'], 'cwd': str(directory),
            'argv_without_fifo': ['taskset', '-c', '12', './check', request['points']],
            'expected_runtime': {**rt['files'], str((directory / 'check').resolve()): b['executable_sha256']},
            'settings': settings, 'actual_stdin_after_READY': 'RUN 1 1\nSTOP 0 0\n',
            'points_sha256': sha(request['points']), 'physical_card_sha256': cardsha,
            'scientific_sources': sources, 'numerical_cards': numerical, 'build': b, 'runtime': rt,
            'preflight_sha256': preflight_sha, 'complex_model': pre['complex_model'],
            'phase_timeout_seconds': 300, 'points_count': 21, 'host_accuracy_limit': 1e-3,
            'expected_dimensions': PLAN['expected_observation']['answer_dimensions'],
            'inherited_checkpoint_sha256': sha(REPORT / 'inherited-checkpoint.json'),
            'ancestor_failed_native_attempt': BINDING['ancestor_index_sha256']}
        save(evidence / 'specification.json', specification)
        shutil.copy2(output / 'Cards/MadLoopParams.dat', evidence / 'MadLoopParams.dat')
        shutil.copy2(actual, evidence / 'param_card.dat')
        failure = error_object = None
        try:
            base.cmd([sys.executable, '-B', SOURCE1 / 'native_job.py', evidence / 'specification.json'], 'request-' + cid + '-' + label, directory, 610)
        except Exception as error:
            failure = repr(error); error_object = error
        finally:
            for name in ('warmup.dat', 'dimensions.dat', 'values.dat', 'timed_status.dat', 'timing.dat'):
                if (directory / name).is_file() and not (evidence / name).exists(): shutil.copy2(directory / name, evidence / name)
            require(base.scientific_sources(output) == sources and base.bench.numerical_card_inputs(output) == numerical and
                    sha(actual) == cardsha and sha(directory / 'check') == b['executable_sha256'], 'Worker scientific inputs changed')
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
                    changed, added = allowed_worker_changes(output, context)
                    links.assert_worker_inputs(output, context, allow_file_changes=changed, allow_new_paths=added)
            finally:
                # Recheck inherited masters even if the private worker check
                # itself failed. Neither failure may admit a qualified result.
                after = verify_inherited_inputs(); save(evidence / 'inherited-after.json', after)
                require(after == before, 'Inherited inputs changed during request')
        except BaseException as error:
            save(evidence / 'completion.json', {'result': {'numerically_qualified': False},
                 'outer_failure': repr(error), 'inputs_unchanged': False,
                 'fatal_infrastructure_failure': 'post-request input guard', 'performance_eligible': False})
            raise
    save(evidence / 'completion.json', {'result': result, 'outer_failure': failure, 'inputs_unchanged': True,
                                     'fatal_infrastructure_failure': stop, 'performance_eligible': False})
    return result


def run():
    frozen(); results = []; total = 0.
    for request in PLAN['requests']:
        lock = None; wait = time.monotonic(); start = None
        try:
            lock = base.acquire(request); wait = time.monotonic() - wait; start = time.monotonic()
            with L.Deadline(min(620, 3720 - total)):
                result = run_request(request)
            elapsed = time.monotonic() - start; total += elapsed
            results.append({'request': request, 'lock_wait_seconds': wait, 'elapsed_seconds': elapsed, 'result': result})
            save(REPORT / 'request-completions' / ('%02d.json' % len(results)), results[-1])
        except BaseException as error:
            save(REPORT / 'aborted-request.json', {'request': request, 'state': 'deferred-before-launch' if start is None else 'aborted-during-request',
                                                'exception': repr(error), 'numerically_qualified': False})
            raise
        finally:
            if lock is not None: lock.close()
    summary = base.summarize_results(results, total)
    summary.update(ancestor_failed_native_attempt=BINDING['ancestor_index_sha256'],
                   inherited_checkpoint_sha256=sha(REPORT / 'inherited-checkpoint.json'))
    save(REPORT / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', choices=STAGES); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'Run from repository root')
    if not args.execute:
        print(json.dumps({'source_only': True, 'commands_executed': 0, 'stages': CAPS, 'requests': 6,
                          'work': str(WORK), 'report': str(REPORT), 'inherited_from_failed_attempt': BINDING['ancestor_index_sha256'],
                          'commands': [[sys.executable, '-B', str(HERE / 'run.py'), stage, '--execute'] for stage in STAGES]}, indent=2))
        return
    require(args.stage is not None, 'Explicit stage required'); seal()
    L.CURRENT_STAGE = args.stage; L.CPU = 12
    if args.stage == 'run':
        predecessor = read(REPORT / 'stage-inherit.json')
        require(predecessor['status'] == 'passed' and predecessor['source_index_sha256'] == sha(HERE / 'source-index.json') and
                predecessor['inherited_checkpoint_sha256'] == sha(REPORT / 'inherited-checkpoint.json'), 'Read-only inheritance did not pass')
    marker = REPORT / ('stage-' + args.stage + '.json'); require(not marker.exists(), 'Append-only stage; no retry')
    start = time.monotonic(); failure = None
    with L.cleanup.SignalRequests() as requests:
        L.REQUESTS = requests
        try:
            if args.stage == 'run':
                run()
                frozen()
            else:
                with L.Deadline(600):
                    inherit()
                    frozen()
        except BaseException as error: failure = error
        finally:
            if REPORT.exists() and args.stage == 'run':
                completed = {read(p)['request']['id'] for p in (REPORT / 'request-completions').glob('*.json')}
                save(REPORT / 'request-accounting.json', {'requested': 6, 'completed': len(completed),
                     'not_completed': [r for r in PLAN['requests'] if r['id'] not in completed],
                     'native_stage_failure': None if failure is None else repr(failure)})
            if REPORT.exists():
                save(marker, {'stage': args.stage, 'status': 'passed' if failure is None else 'failed',
                    'exception': None if failure is None else repr(failure), 'elapsed_seconds_including_wait': time.monotonic() - start,
                    'cap_seconds_excluding_lock_wait': CAPS[args.stage], 'source_index_sha256': sha(HERE / 'source-index.json'),
                    'campaign_manifest_sha256': PLAN['manifest_sha256'], 'ancestor_failed_native_attempt': BINDING['ancestor_index_sha256'],
                    'inherited_checkpoint_sha256': sha(REPORT / 'inherited-checkpoint.json') if (REPORT / 'inherited-checkpoint.json').is_file() else None,
                    'native_execution': args.stage == 'run', 'performance_eligible': False})
    if failure is not None: raise failure


if __name__ == '__main__': main()
