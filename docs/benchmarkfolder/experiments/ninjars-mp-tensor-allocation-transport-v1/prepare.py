#!/usr/bin/env python3
"""Snapshot the committed parser and exact frozen allocation patch; no build."""
from pathlib import Path
import difflib
import gzip
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-transport-v1'
OLD = ROOT / 'docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-v1'
REVISION = '9f13d10cc'
MANIFEST_SHA = 'ad8c34734d4764b7ccdbd5ed98d77d961ba1b9c5cf701d6bc4ccd1b35fed1d71'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def binding(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size}


def main():
    assert Path.cwd() == ROOT
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'madloop_ninjars'
    assert sha(OLD / 'manifest.json') == MANIFEST_SHA
    frozen = json.loads((OLD / 'manifest.json').read_text())
    for name, record in frozen['files'].items():
        path = OLD / name
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], name
    revision = subprocess.check_output(['git', 'rev-parse', REVISION], text=True).strip()
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, '--', 'vendor/ninjars'], text=True).splitlines()
    source = {}
    for name in tracked:
        data = subprocess.check_output(['git', 'show', f'{revision}:{name}'])
        assert (ROOT / name).read_bytes() == data, f'live source differs from committed transport revision: {name}'
        source[str(Path(name).relative_to('vendor/ninjars'))] = data
    assert hashlib.sha256(source['src/tensor.rs']).hexdigest() == 'f5b153e179ac56e4a833f8d83b011a667e61cff1bcb567c2f1508a1d94013e55'
    assert b'input.set_prec(precision);' in source['src/lib.rs']
    WORK.mkdir(parents=True, exist_ok=False)
    bridge = WORK / 'bridge'
    for name, data in source.items():
        for base in [bridge, HERE / 'original']:
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
    shutil.copyfile(bridge / 'src/tensor.rs', bridge / 'src/tensor_dense_reference.rs')
    reused = {'src/tensor.rs': 'tensor.rs', 'src/tensor_allocation_tests.rs': 'tensor_allocation_tests.rs',
              'src/tensor_capture_tests.rs': 'tensor_capture_tests.rs'}
    for target, name in reused.items():
        shutil.copyfile(OLD / name, bridge / target)
        retained = HERE / 'reused' / name
        retained.parent.mkdir(exist_ok=True)
        shutil.copyfile(OLD / name, retained)
    before = source['src/tensor.rs'].decode()
    after = (bridge / 'src/tensor.rs').read_text()
    with (HERE / 'tensor.patch').open('x') as stream:
        stream.write(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='corrected-production/src/tensor.rs', tofile='combined/src/tensor.rs')))
    capture = ROOT / 'docs/benchmarkfolder/experiments/generic-tensor-plan/capture/actual-mp-requests.jsonl.gz'
    expected = json.loads((OLD / 'provenance.json').read_text())['capture']
    assert sha(capture) == expected['sha256']
    with gzip.open(capture, 'rb') as compressed, (WORK / 'actual-mp-requests.jsonl').open('xb') as output:
        shutil.copyfileobj(compressed, output)
    assert sha(WORK / 'actual-mp-requests.jsonl') == expected['uncompressed_sha256']
    rows = [json.loads(line) for line in (WORK / 'actual-mp-requests.jsonl').read_text().splitlines()]
    assert len(rows) == 1392 and all(row['status'] == 0 and len(row['output']) == 8 for row in rows)
    original = {name: hashlib.sha256(data).hexdigest() for name, data in source.items()}
    actual = {str(path.relative_to(bridge)): sha(path) for path in sorted(bridge.rglob('*')) if path.is_file()}
    changed = [name for name, digest in original.items() if actual[name] != digest]
    added = sorted(set(actual) - set(original))
    assert changed == ['src/tensor.rs']
    assert added == ['src/tensor_allocation_tests.rs', 'src/tensor_capture_tests.rs', 'src/tensor_dense_reference.rs']
    write(HERE / 'provenance.json', {'schema': 'corrected-transport-allocation-preparation-v1',
        'production_revision': revision, 'working_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'original_files': original, 'combined_files': actual, 'changed': changed, 'added': added,
        'reused': {target: binding(OLD / name) for target, name in reused.items()},
        'prior_manifest': binding(OLD / 'manifest.json'), 'capture': expected,
        'plan': binding(HERE / 'PLAN.md'), 'prepare_controller': binding(Path(__file__).resolve()),
        'compilation_status': 'HELD', 'mp_execution_status': 'HELD', 'production_changed': False,
        'acceptance_evidence': False})
    print(json.dumps({'status': 'prepared-not-compiled', 'work': str(WORK.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
