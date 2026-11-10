#!/usr/bin/env python3
"""Verify the closed text snapshot and replay the frozen 218-case parser offline.

No native artifacts are needed. No controller or scientific worker is imported.
The only imported payload source is the frozen stdlib-only fixture validator.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = Path('.tirust-build/qp-mass-general-implementation/combined-qualification')
PINNED_RECORDS = {
    'run.json': 'a5d7541c314367999c11dd4335a15f86b9bcf7d44294849b10d6b25558c33b12',
    'loader-correction/run.json': 'f741701d2980d6c6c0d9587cb992504e5356f6a66ee147e34d62df74f4e2c2b6',
    'loader-correction/fixture-offline-validation.json': '35a0584f529f641804a345c171f9232a6ef4d04291154ff9c40634fcbb187b28',
    'fixture/validate.py': 'c0bf0e8278ee4d873f885cd79f2795463dfc4284cacf142880eca41a090626f1',
    'fixture/commands.json': '138b8de86543c74b8310fd6c89ae37023795e87a841740ebda11f7e43a2d1709',
}


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def closed_inventory(root, expected):
    actual = {}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Symlink in portable package: '+str(path))
        if path.is_file() and path != root/'index.json':
            actual[str(path.relative_to(root))] = sha(path)
    require(set(actual) == set(expected), 'Package inventory differs: missing='+str(sorted(set(expected)-set(actual)))+
            ' extra='+str(sorted(set(actual)-set(expected))))
    for name, value in expected.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe inventory path')
        require(actual[name] == value, 'Changed package file: '+name)
    return len(actual)


def load_frozen_validator(package):
    root = package/'payload'
    fixture = root/BASE/'fixture'
    require(sha(fixture/'validate.py') == PINNED_RECORDS['fixture/validate.py'], 'Changed scientific validator')
    spec = importlib.util.spec_from_file_location('frozen_qpmass_fixture_offline', fixture/'validate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Transport relocation only. Source bytes, numerical expectations, absolute
    # historical argv strings and case definitions remain unchanged.
    module.ROOT = root
    module.HERE = fixture
    return module


def check_failures(package):
    root = package/'payload'
    base = root/BASE
    original, corrected = read(base/'run.json'), read(base/'loader-correction/run.json')
    commands = read(base/'fixture/commands.json')
    spec = read(base/'prepared.json')
    require(original['state'] == corrected['state'] == 'failed', 'Failures were relabeled')
    require(original['automatic_retry'] is False and corrected['automatic_retry'] is False, 'Retry semantics changed')
    require(original['input_hashes_after'] == corrected['input_hashes_after'] == 'unchanged', 'Changed original inputs')
    require(original['error'] == {'message': 'Fixture core runtime missing', 'type': 'ValueError'}, 'Changed loader failure')
    require(len(original['jobs']) == 1 and original['jobs'][0]['name'] == 'loader-fixture', 'Unexpected initial execution')
    require(original['jobs'][0]['returncode'] == 0, 'Initial loader process itself failed')
    require('libquadmath.so.0' in spec['runtime_checks'][0]['required'], 'Missing false-required quadmath evidence')
    require('libquadmath' not in (root/original['jobs'][0]['log']).read_text(), 'Unexpected fixture quadmath load')
    loader_names = [x['job']['name'] for x in spec['runtime_checks']]
    require(len(loader_names) == 7, 'Wrong loader case count')
    fixture_names = [x['name'] for x in commands['runs']]
    require([x['name'] for x in corrected['jobs']] == loader_names+fixture_names+['validate-fixture'],
            'Unexpected execution stage, missing fixture, or amplitude worker in captured run')
    require(len(corrected['jobs']) == 226, 'Wrong corrected event count')
    for event in corrected['jobs'][:7]:
        require(event['returncode'] == 0 and event['timed_out'] is False and not event.get('error'), 'Loader execution failed')
    require(corrected['error'] == {'message': 'Job failed; no retry: validate-fixture', 'type': 'ValueError'},
            'Changed stale-parser failure')
    event = corrected['jobs'][-1]
    require(event['returncode'] == 1 and event['timed_out'] is False, 'Changed stale-parser process result')
    original_events = read(package/'manifest.json')['repository_root_original']+'/'+str(BASE/'run.json')
    require(event['command'][-2:] == ['--events', original_events], 'Stale events path no longer captured exactly')
    require('Missing or extra fixture execution events' in (root/event['log']).read_text(), 'Missing original parser traceback')
    correction = read(base/'loader-correction/prepared.json')
    expected_closure = {'ld-linux-x86-64.so.2', 'libc.so.6', 'libgcc_s.so.1', 'libgfortran.so.5', 'libm.so.6'}
    require(set(correction['fixture_expected']) == expected_closure, 'Changed corrected fixture closure')
    require(corrected['runtime_checks'][0]['allow_system_subset'] is False, 'Permissive corrected loader check')
    require(corrected['runtime_checks'][0]['libraries'] == correction['fixture_expected'], 'Runtime closure mismatch')
    for left, right in zip(corrected['runtime_checks'], loader_names, strict=True):
        require(left['allow_system_subset'] is False, 'Incomplete runtime closure check: '+right)
    # Retain and recheck every event output hash, including all stdout/stderr and
    # failed logs, beyond the frozen validator's scientific caller.hex checks.
    for record in [original, corrected]:
        for event in record['jobs']:
            require(sha(root/event['log']) == event['log_sha256'], 'Changed captured log')
            for name, value in event.get('output_files', {}).items():
                require(sha(root/event['cwd']/name) == value, 'Changed captured native output')
    return dict(initial_failed_loader_jobs=1, corrected_loader_jobs=7, completed_fixture_jobs=218,
                failed_parser_jobs=1, amplitude_worker_jobs=0)


def verify(package=HERE, check_inventory=True):
    package = Path(package).resolve()
    manifest = read(package/'manifest.json')
    index = read(package/'index.json') if check_inventory else None
    count = closed_inventory(package, index['files']) if check_inventory else None
    root = package/'payload'
    actual_payload = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual_payload == set(manifest['text_files']), 'Payload not closed')
    for name, item in manifest['text_files'].items():
        path = root/name
        require(not path.is_symlink() and sha(path) == item['sha256'] and path.stat().st_size == item['bytes'],
                'Changed mapped text: '+name)
    for name in manifest['native_and_system_hash_only']:
        require(name not in actual_payload, 'Native or system identity copied as payload: '+name)
    for name, value in PINNED_RECORDS.items():
        require(sha(root/BASE/name) == value, 'Changed frozen historical record: '+name)
    for name in manifest['forbidden_outputs']:
        require(not (root/BASE/name).exists(), 'Forbidden future output included')
    failures = check_failures(package)
    validator = load_frozen_validator(package)
    commands_path = root/BASE/'fixture/commands.json'
    events_path = root/BASE/'loader-correction/run.json'
    result = validator.validate(events_path, commands_path=commands_path)
    retained = read(root/BASE/'loader-correction/fixture-offline-validation.json')
    require(result == retained['validation'], 'Portable replay differs from retained offline PASS')
    require(retained['native_executed'] is False and retained['validator_sha256'] == PINNED_RECORDS['fixture/validate.py'],
            'Offline receipt provenance changed')
    try:
        validator.validate(root/BASE/'run.json', commands_path=commands_path)
    except ValueError as error:
        require(str(error) == 'Missing or extra fixture execution events', 'Stale path failed for another reason')
    else:
        raise ValueError('Stale event-path failure not reproduced')
    return dict(schema='qp-mass-portable-offline-verification-v1', passed=True,
                indexed_files=count, text_payload_files=len(actual_payload),
                hash_only_artifact_identities=len(manifest['native_and_system_hash_only']),
                captured_events=failures, normal_cases=result['normal_cases'],
                positive_invalid_trap_controls=result['positive_invalid_trap_controls'],
                dp_exact_pairs=result['dp_exact_pairs'], qp_transport_pairs=result['qp_transport_pairs'],
                unchanged_validator_sha256=retained['validator_sha256'],
                replay_matches_retained_pass=True, stale_path_failure_reproduced=True,
                compiler_or_native_launched=False, local_native_dependencies_required=False,
                qualification=result['interpretation'])


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True))
