#!/usr/bin/env python3
"""Portable evidence/hash/completeness verifier; does not execute native code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CMS_PREFIX = '.tirust-build/cms-scalar-refusal-prototype/'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE/name).read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    index = read('source-index.json')
    require(index['schema'] == 'cms-portable-milestone-file-index-v1', 'Wrong index')
    paths = list(HERE.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Symlink in package')
    actual = {str(p.relative_to(HERE)) for p in paths if p.is_file()}
    require(actual == set(index['files']) | {'source-index.json','SHA256SUMS'}, 'Incomplete/extra file coverage')
    for name, digest in index['files'].items():
        path = HERE/name
        require(path.resolve().is_relative_to(HERE), 'Nonportable index path')
        require(sha(path) == digest, 'Retained file changed: '+name)
        raw = path.read_bytes()
        require(not raw.startswith((b'\x7fELF', b'!<arch>')), 'Bundled native artifact')
        raw.decode('utf-8')
    sums = ''.join(sha(HERE/name)+'  '+name+'\n' for name in sorted([*index['files'],'source-index.json']))
    require((HERE/'SHA256SUMS').read_text() == sums, 'Checksum list coverage differs')
    manifest = read('manifest.json')
    require(manifest['retained_files'] == {k:v for k,v in index['files'].items() if k != 'manifest.json'},
            'Manifest/index retained coverage differs')
    origins = manifest['source_bindings']
    native = manifest['native_artifacts']
    for name, info in origins.items():
        require(manifest['retained_files'].get(info['file']) == info['sha256'], 'Unbound original source: '+name)

    def original(name):
        require(name in origins, 'Missing original text evidence: '+name)
        return HERE/origins[name]['file']

    def covered(name, digest):
        value = origins[name]['sha256'] if name in origins else native.get(name)
        require(value == digest, 'Missing/inconsistent protected input: '+name)

    build_manifest = read('build/manifest.json')
    require(sha(HERE/'build/manifest.json') == manifest['prior_build_manifest_sha256'], 'Build parent identity')
    for name, digest in build_manifest['retained_files'].items():
        require(manifest['retained_files'].get('build/'+name) == digest, 'Build package coverage')
    for name, digest in build_manifest['native_artifacts'].items():
        require(native.get(name) == digest, 'Native build identity')
    source_index = read('build/source-index.json')
    for name, digest in source_index['package_files'].items():
        covered(CMS_PREFIX+name, digest)
    build = read('build/build.json')
    require(build['state'] == 'complete' and build['input_hashes_after'] == 'unchanged', 'Build incomplete')
    require(len(build['jobs']) == 17 and build['finished_unix']-build['started_unix'] <= 180, 'Build scope')
    for event in build['jobs']:
        require(event['returncode'] == 0 and event['timed_out'] is False and not event.get('error'), 'Compiler failure')
        require(sha(original(event['log'])) == event['log_sha256'], 'Compiler raw log binding')
    for name, digest in {**build['outputs'],**build['module_outputs']}.items():
        require(native.get(name) == digest, 'Unbound compiled artifact')

    for attempt in ('attempt1','attempt2'):
        prefix = 'mock/'+attempt+'/'
        spec = read(prefix+'prepared.json')
        run = read(prefix+'run.json')
        require(run['prepared_sha256'] == sha(HERE/prefix/'prepared.json'), 'Run/preparation binding')
        require(run['controller_sha256'] == sha(HERE/prefix/'control.py'), 'Run/controller binding')
        for name, digest in spec['inputs'].items():
            covered(name, digest)
        for name, digest in spec['system_files'].items():
            require(manifest['system_artifacts_hash_only'].get(name) == digest, 'System artifact identity')
        for event in run['jobs']:
            require(sha(original(event['log_file'])) == event['log_sha256'], 'Mock raw log binding')
            for name, digest in event['output_files'].items():
                require(sha(original(event['cwd']+'/'+name)) == digest, 'Mock output binding')
        require(run['protected_inputs_after'] == 'unchanged', 'Protected input drift')
    failed = read('mock/attempt1/run.json')
    require(sha(HERE/'mock/attempt1/run.json') == manifest['failed_mock_run_sha256'], 'Failed attempt identity')
    require(failed['state'] == 'failed' and len(failed['jobs']) == 1, 'Prior failure missing')
    require(failed['jobs'][0]['name'] == 'classifier-loader' and failed['jobs'][0]['returncode'] == 127,
            'Original failure scope differs')
    require('GLIBC_PRIVATE' in original(failed['jobs'][0]['log_file']).read_text(), 'Missing original failure cause')
    spec = read('mock/attempt2/prepared.json')
    run = read('mock/attempt2/run.json')
    require(sha(HERE/'mock/attempt2/run.json') == manifest['successful_mock_run_sha256'], 'Successful run identity')
    require(run['state'] == 'complete' and len(run['jobs']) == 4, 'Replay incomplete')
    require(run['prior_failed_run_sha256'] == manifest['failed_mock_run_sha256'], 'Failure/replay chain missing')
    require(run['source_index_sha256'] == sha(HERE/'mock/attempt2/source-index.json'), 'Replay source index identity')
    require(run['runtime_environment']['LD_LIBRARY_PATH'] is None, 'Outer loader path override')
    require(run['finished_unix']-run['started_unix'] <= 35, 'Replay exceeded cap')
    jobs = [c['job'] for c in spec['loader_checks']] + spec['mock_jobs']
    for event, job in zip(run['jobs'], jobs, strict=True):
        require(event['name'] == job['name'], 'Job ordering differs')
        require(event['command'] == [spec['taskset'],'-c','8',*job['argv']], 'Unbound job command')
        require(event['returncode'] == 0 and event['timed_out'] is False and not event.get('error'), 'Replay process failure')
        require(event['output_closure_matches'] is True and set(event['output_files']) == set(job['expected_outputs']),
                'Output closure incomplete')
        require(0 < event['timeout_seconds'] <= job['seconds_max'], 'Process cap drift')
    for event, check in zip(run['jobs'][:2], spec['loader_checks'], strict=True):
        require(event['resolved_runtime'] == check['expected'], 'Recorded loader identities differ')
    classifier = load('portable_classifier_gate', original(spec['classifier_validator']))
    actual_classifier = classifier.validate_text(original(run['jobs'][2]['log_file']).read_text())
    require(actual_classifier == run['validations']['classifier'], 'Classifier assertion evidence differs')
    require(original(run['jobs'][3]['log_file']).read_bytes() == b'', 'Unexpected caller process output')
    cases = json.loads(original(spec['caller_cases']).read_text())
    caller = load('portable_caller_gate', original(spec['caller_validator']))
    journal = original(run['jobs'][3]['cwd']+'/caller-mock.log').read_text()
    require(caller.validate_text(journal,cases) == run['validations']['caller'], 'Caller journal mismatch')
    require(len(cases['cases']) == 91 and len(cases['models']) == 2 and len(journal.splitlines()) == 730,
            'Caller coverage differs')
    require(manifest['caller_mock_reduction_calls'] == 3*91*2 == 546, 'Mock call count differs')
    for name, digest in manifest['source_parent_hashes'].items():
        covered(name, digest)
    print(json.dumps({'retained_files_verified':len(manifest['retained_files']),
                      'native_artifacts_hash_only':len(native), 'build_commands':17,
                      'failed_attempts_retained':1, 'loader_checks':2, 'classifier_precision_overloads':16,
                      'caller_primary_cases':182, 'caller_mock_reduction_calls':546, 'caller_journal_rows':730,
                      'compiler_loader_or_native_launched':False}, sort_keys=True))


if __name__ == '__main__':
    main()
