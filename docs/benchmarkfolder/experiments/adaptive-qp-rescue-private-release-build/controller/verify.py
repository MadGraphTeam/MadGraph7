#!/usr/bin/env python3
"""Read-only source/hash verification; no compiler, process or DSO invocation."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(directory, index_name, expected=None):
    index = directory / index_name
    if expected:
        assert sha(index) == expected
    files = json.loads(index.read_text())['files']
    paths = list(directory.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(directory)) for p in paths if p.is_file()} == set(files) | {index_name}
    for name, entry in files.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p = directory / name
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
    return files


def main():
    files = inventory(HERE, 'source-index.json')
    ast.parse((HERE / 'run.py').read_text())
    source = ROOT / '.tirust-build/adaptive-qp-rescue-prototype3'
    inventory(source, 'source-index.json', 'f5550b8719e76e177ad6b73d456cc655815770eb7092a8fabdc776c415c03213')
    pins = json.loads((source / 'source-bindings.json').read_text())['files']
    assert all(sha(ROOT / name) == digest for name, digest in pins.items())
    runner = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2'
    inventory(runner, 'source-index.json', '69f76225a439dd325c4c08369a46ae5654102f1a95c19d6c3e7ca83682f0a3f4')
    build = ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-rescue-private-build'
    inventory(build, 'evidence-index.json', '1a4af7f8f8be9ce8880ebc9ce466f66afec961be4c24b6978cdd8fdfa28fc8d8')
    prior = json.loads((build / 'attempt-3/execution.json').read_text())
    test = prior['test_executable_not_run']
    assert sha(test['cargo_record']['executable']) == test['sha256'] == 'd22953ed3b955adf7dcce3831259060cec9e974816d432a4a86fd000b1afe8b7'
    debug = ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-rescue-private-cdylib-build'
    inventory(debug, 'evidence-index.json', 'd3e1442560aa572c6a3dd2964a9685ee423e58fbc275e0027e9b2c0f2693f03b')
    library = json.loads((debug / 'attempt1/execution.json').read_text())['private_library_not_loaded']
    assert sha(library['path']) == sha(library['isolated_copy']) == library['sha256'] == '63298bdd242b99dd46d654debe6304f5b4fdff955d5082d8c50c3b2ff01ae9ea'
    cache = json.loads((HERE / 'cache-inspection.json').read_text())
    assert not cache['compiler_executed'] and not cache['native_executed'] and not cache['exact_reuse_guaranteed']
    assert (cache['proposed_cap_seconds'], cache['jobs'], cache['affinity_override'], cache['cpu_reservation']) == (900, 1, None, False)
    print(json.dumps({'verification': 'pass', 'controller_files': len(files), 'source_bindings': len(pins),
                      'cached_release_rlibs_observed': len(cache['cache']), 'compiler_executed': False,
                      'native_executed': False, 'source_index_sha256': sha(HERE / 'source-index.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
