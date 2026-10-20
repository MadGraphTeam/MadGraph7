#!/usr/bin/env python3
"""Read-only evidence validation; no compiler, installation, or engine call."""
import difflib
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    manifest = read('manifest.json')
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.name != 'manifest.json'}
    assert actual == set(manifest['files'])
    for name, record in manifest['files'].items():
        path = HERE / name
        assert not path.is_symlink()
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], name
    scalar = read('scalar-result.json')
    assert scalar['complete'] and scalar['engine_calls'] == 0 and not scalar['production_modified']
    assert sha(HERE / 'scalar-build.json') == scalar['build_sha256']
    assert [(row['arm'], row['step'], row['returncode']) for row in scalar['runs']] == [
        ('old', 1, 1), ('old', 2, 1), ('proposed', 1, 0), ('proposed', 2, 0)]
    assert (HERE / 'scalar-old/1.stdout').read_text().splitlines()[-1] == (
        'SUMMARY finite_cases=28 value_fail=0 sign_fail=2 precision_fail=0 format_fail=2 branch_fail=2 reject_fail=0')
    assert (HERE / 'scalar-proposed/1.stdout').read_text().splitlines()[-1] == (
        'SUMMARY finite_cases=28 value_fail=0 sign_fail=0 precision_fail=0 format_fail=0 branch_fail=0 reject_fail=0')
    assert (HERE / 'scalar-proposed/2.stdout').read_text().splitlines()[-1].split() == ['COMPLETE', '14']
    prepared = read('bridge-prepared.json')
    for name, digest in prepared['original_files'].items():
        assert sha(HERE / 'original' / name) == digest
    for name, digest in prepared['proposed_files'].items():
        assert sha(HERE / 'proposed' / name) == digest
    changed = {name for name, digest in prepared['original_files'].items()
               if prepared['proposed_files'][name] != digest}
    assert changed == {'src/lib.rs'}
    patches = []
    for name in ('src/lib.rs', 'src/transport_contract_tests.rs', 'src/transport_replay_tests.rs'):
        old, new = HERE / 'original' / name, HERE / 'proposed' / name
        patches.append(''.join(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],
            new.read_text().splitlines(True), fromfile='a/vendor/ninjars/' + name if old.exists() else '/dev/null',
            tofile='b/vendor/ninjars/' + name)))
    assert ''.join(patches) == (HERE / 'production-ready.patch').read_text()
    build, run = read('bridge-build.json'), read('bridge-result.json')
    assert build['complete'] and not build['executed']
    assert run['complete'] and not run['production_modified'] and not run['allocation_optimization_present']
    assert run['build_sha256'] == sha(HERE / 'bridge-build.json')
    assert build['preparation_sha256'] == sha(HERE / 'bridge-prepared.json')
    assert all(row['returncode'] == 0 for row in run['runs'])
    assert '18 passed; 0 failed; 1 ignored;' in (HERE / 'unit.stdout').read_text()
    for name, digest in run['files'].items():
        assert sha(HERE / name) == digest
    capture = prepared['capture']
    assert sha(ROOT / capture['path']) == capture['sha256']
    with gzip.open(ROOT / capture['path'], 'rb') as stream:
        contents = stream.read()
    assert hashlib.sha256(contents).hexdigest() == capture['uncompressed_sha256']
    source = [json.loads(line) for line in contents.splitlines()]
    replay = [json.loads(line) for line in (HERE / 'actual-replay.jsonl').read_text().splitlines()]
    assert len(source) == 1392 and len(replay) == 1393
    for index, (expected, observed) in enumerate(zip(source, replay[:-1])):
        assert observed['record'] == index
        assert expected['n'] == observed['n'] and expected['rank'] == observed['rank']
        assert expected['status'] == observed['status'] == 0
        assert expected['output'] == observed['expected'] == observed['actual']
        assert len(observed['actual']) == 8 and observed['changed_fields'] == []
    assert replay[-1] == run['replay_summary'] == {
        'complete': True, 'records': 1392, 'changed_records': 0, 'changed_fields': 0, 'failures': 0}
    print(json.dumps({'passed': True, 'files': len(actual), 'ordinary_tests': 18,
                      'replayed_requests': 1392, 'unchanged_output_fields': 11136,
                      'production_unchanged_during_validation': True, 'engine_calls': 0}))


if __name__ == '__main__':
    main()
