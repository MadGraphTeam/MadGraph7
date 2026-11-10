#!/usr/bin/env python3
"""Append-only mechanical launcher correction; file reads/writes only."""
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent
CMS = PRIOR.parent
ROOT = CMS.parents[1]
PRIOR_HASHES = {
    'control.py': 'fe6fa15cb041bb2a1112bac67a7192bd8431e44f0d16344230ed3eeff07b69ad',
    'prepared.json': 'd9eea934283b646bf5ad2bca4b38711908011ea2839a64f6254ec983ba6efb25',
    'source-index.json': '0b1d997d1b855ca483b5575d88b7b54a965d33fa086d1c386b3e844c02c39b19',
    'run.json': 'c79ab580f2799f0bcebbd11b12b8ca88c5ac2d47f35b35595cac0a47a91312ab',
    'classifier-loader.log': '99a94e8d134ee33585a52cc6df4969580c590cee6ce3c7918ede673e13718ee2',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def proposal(old):
    spec = copy.deepcopy(old)
    spec['schema'] = 'cms-scalar-refusal-mock-attempt2-proposal-v1'
    spec['prior_failed_run_sha256'] = PRIOR_HASHES['run.json']
    spec['correction'] = 'Use explicit target loader --library-path; never export target libraries to outer taskset'
    library_path = ':'.join(spec['runtime_directories'])
    require('LD_LIBRARY_PATH' not in spec['compiler_environment'], 'Historical normal environment unexpectedly has LD_LIBRARY_PATH')
    by_kind = {}
    for check in spec['loader_checks']:
        binary = str(ROOT / check['executable'])
        require(check['job']['argv'] == [check['loader'], '--list', binary], 'Unexpected historical loader recipe')
        check['job']['argv'] = [check['loader'], '--library-path', library_path, '--list', binary]
        by_kind[check['job']['name'].removesuffix('-loader')] = check
    for job in spec['mock_jobs']:
        check = by_kind[job['validator']]
        binary = str(ROOT / check['executable'])
        require(job['argv'] == [binary], 'Unexpected historical mock recipe')
        job['argv'] = [check['loader'], '--library-path', library_path, binary]
    return spec


def main():
    require(Path.cwd().resolve() == ROOT, 'Run from MadGraph repository root')
    require(not (HERE/'prepared.json').exists() and not (HERE/'source-index.json').exists(), 'Already prepared')
    for name, digest in PRIOR_HASHES.items():
        require(sha(PRIOR/name) == digest, 'Prior evidence changed: '+name)
    failed = json.loads((PRIOR/'run.json').read_text())
    require(failed['state'] == 'failed' and len(failed['jobs']) == 1, 'Unexpected failed-attempt scope')
    require(failed['jobs'][0]['name'] == 'classifier-loader' and failed['jobs'][0]['returncode'] == 127,
            'Unexpected original failure')
    build = json.loads((CMS/'build.json').read_text())
    require(build['state'] == 'complete' and len(build['jobs']) == 17, 'Exact successful build required')
    old = json.loads((PRIOR/'prepared.json').read_text())
    spec = proposal(old)
    for name, digest in PRIOR_HASHES.items():
        spec['inputs'][relative(PRIOR/name)] = digest
    for name in ('prepare.py', 'control.py', 'control.patch', 'README.md', 'test_source.py'):
        spec['inputs'][relative(HERE/name)] = sha(HERE/name)
    for name, digest in spec['inputs'].items():
        require(sha(ROOT/name) == digest, 'Protected input changed: '+name)
    for name, digest in spec['system_files'].items():
        require(sha(name) == digest, 'System tool/runtime changed: '+name)
    for job in [c['job'] for c in spec['loader_checks']] + spec['mock_jobs']:
        directory = HERE/job['directory']
        directory.mkdir()
        require(not any(directory.iterdir()), 'Nonempty new job directory')
    (HERE/'prepared.json').write_text(json.dumps(spec, indent=2, sort_keys=True)+'\n')
    files = {name:sha(HERE/name) for name in ('prepare.py', 'control.py', 'control.patch', 'README.md', 'test_source.py', 'prepared.json')}
    index = {'schema':'cms-scalar-refusal-mock-attempt2-source-index-v1', 'files':files,
             'prior_evidence':PRIOR_HASHES, 'parent_build_sha256':sha(CMS/'build.json'),
             'numerical_sources_binaries_runtime_and_expected_outputs_unchanged':True,
             'compiler_loader_or_native_started':False}
    (HERE/'source-index.json').write_text(json.dumps(index, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'prepared_sha256':sha(HERE/'prepared.json'),
                      'source_index_sha256':sha(HERE/'source-index.json'),
                      'compiler_loader_or_native_started':False}, sort_keys=True))


if __name__ == '__main__':
    main()
