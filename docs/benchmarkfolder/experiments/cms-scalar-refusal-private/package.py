#!/usr/bin/env python3
"""Assemble portable text evidence without compiling or running native code."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CMS = HERE.parent
ROOT = CMS.parents[1]
BUILD_PACKAGE_SHA = '77e0fad4de57ac35ab90db2c76c7b97ad9eca9e7200cc2413c8309d7ddc01c69'
FAILED_RUN_SHA = 'c79ab580f2799f0bcebbd11b12b8ca88c5ac2d47f35b35595cac0a47a91312ab'
SUCCESS_RUN_SHA = 'e287551a802dd0bb839a225e98d650b972c2787b8cdcf8ebc2d5b42618dfe7e4'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def main():
    require(Path.cwd().resolve() == ROOT, 'Run from MadGraph root')
    require(not (HERE/'manifest.json').exists(), 'Already packaged')
    build_package = CMS/'source-build-evidence'
    attempt1 = CMS/'mock-qualification'
    attempt2 = attempt1/'attempt2'
    require(sha(build_package/'manifest.json') == BUILD_PACKAGE_SHA, 'Build package changed')
    require(sha(attempt1/'run.json') == FAILED_RUN_SHA, 'Failed run changed')
    require(sha(attempt2/'run.json') == SUCCESS_RUN_SHA, 'Successful run changed')
    build_manifest = json.loads((build_package/'manifest.json').read_text())
    native = dict(build_manifest['native_artifacts'])
    origins = {}

    def bind(source, target, digest=None):
        digest = digest or sha(source)
        require(sha(source) == digest and not source.is_symlink(), 'Changed/symlink source: '+str(source))
        target_path = HERE/target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        require(not target_path.exists(), 'Duplicate destination: '+target)
        raw = source.read_bytes()
        require(not raw.startswith((b'\x7fELF', b'!<arch>')), 'Native artifact cannot be bundled')
        raw.decode('utf-8')
        shutil.copyfile(source, target_path)
        origins[rel(source)] = {'file':target, 'sha256':digest}

    for name, digest in build_manifest['retained_files'].items():
        bind(build_package/name, 'build/'+name, digest)
    bind(build_package/'manifest.json', 'build/manifest.json', BUILD_PACKAGE_SHA)
    original = json.loads((CMS/'source-index.json').read_text())
    for name, digest in original['package_files'].items():
        source = CMS/name
        if rel(source) not in native:
            require(sha(HERE/'build/source'/name) == digest, 'Incomplete copied build source')
            origins[rel(source)] = {'file':'build/source/'+name, 'sha256':digest}
    origins[rel(CMS/'source-index.json')] = {'file':'build/source-index.json', 'sha256':sha(CMS/'source-index.json')}
    origins[rel(CMS/'build.json')] = {'file':'build/build.json', 'sha256':sha(CMS/'build.json')}
    build = json.loads((CMS/'build.json').read_text())
    for event in build['jobs']:
        origins[event['log']] = {'file':'build/logs/'+Path(event['log']).name, 'sha256':event['log_sha256']}

    for directory, target in ((attempt1, 'mock/attempt1'), (attempt2, 'mock/attempt2'),
                               (CMS/'classifier/mock-qualification', 'classifier-validator')):
        for source in sorted(directory.rglob('*')):
            require(not source.is_symlink(), 'Unexpected symlink')
            if directory == attempt1 and source.is_relative_to(attempt2):
                continue
            if source.is_file():
                bind(source, target+'/'+str(source.relative_to(directory)))
    spec = json.loads((attempt2/'prepared.json').read_text())
    for name, digest in spec['inputs'].items():
        require(sha(ROOT/name) == digest, 'Protected input changed: '+name)
        if name in origins:
            require(origins[name]['sha256'] == digest, 'Origin hash mismatch')
        elif name in native:
            require(native[name] == digest, 'Native hash mismatch')
        else:
            source = ROOT/name
            if source.suffix.lower() in ('.mod', '.smod', '.o', '.a', '.so') or source.read_bytes().startswith(b'\x7fELF'):
                native[name] = digest
            else:
                bind(source, 'protected/'+name, digest)
    parents = json.loads((CMS/'parents.json').read_text())['parent_files']
    for name, digest in parents.items():
        require(sha(ROOT/name) == digest, 'Production/source parent changed: '+name)
        if name not in origins:
            bind(ROOT/name, 'parents/'+name, digest)
    for name, digest in native.items():
        require(sha(ROOT/name) == digest, 'Native artifact changed: '+name)
    for name, digest in spec['system_files'].items():
        require(sha(name) == digest, 'Pinned system artifact changed')
    run = json.loads((attempt2/'run.json').read_text())
    require(run['state'] == 'complete' and len(run['jobs']) == 4, 'Incomplete replay')
    retained = {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
    manifest = {
        'schema':'cms-source-build-mock-portable-milestone-v1',
        'retained_files':retained, 'source_bindings':dict(sorted(origins.items())),
        'native_artifacts':dict(sorted(native.items())), 'system_artifacts_hash_only':spec['system_files'],
        'source_parent_hashes':parents, 'source_parent_state_at_packaging':'unchanged',
        'prior_build_manifest_sha256':BUILD_PACKAGE_SHA, 'failed_mock_run_sha256':FAILED_RUN_SHA,
        'successful_mock_run_sha256':SUCCESS_RUN_SHA,
        'compiler_commands_passed':17, 'mock_processes_passed':2, 'loader_checks_passed':2,
        'classifier_precision_kinds':[8,16], 'classifier_families_per_precision':8,
        'caller_primary_cases':182, 'caller_mock_reduction_calls':546, 'caller_journal_rows':730,
        'actual_analytic_execution_qualified':False, 'physical_accuracy_qualified':False,
        'full_host_rescue_qualified':False, 'hardware_ieee_traps_rounding_qualified':False,
        'performance_qualified':False, 'packaging_launched_native':False,
    }
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    index = {'schema':'cms-portable-milestone-file-index-v1',
             'files':{**retained,'manifest.json':sha(HERE/'manifest.json')}}
    (HERE/'source-index.json').write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
    (HERE/'SHA256SUMS').write_text(''.join(sha(HERE/name)+'  '+name+'\n' for name in sorted([*index['files'],'source-index.json'])))
    print(json.dumps({'manifest_sha256':sha(HERE/'manifest.json'), 'source_index_sha256':sha(HERE/'source-index.json'),
                      'retained_files':len(retained), 'native_hash_only':len(native), 'native_launched':False}))


if __name__ == '__main__':
    main()
