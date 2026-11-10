"""Package or verify the closed current-library stability replay without execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.tirust-build/default-stability-current-1'
INDEX = HERE / 'evidence-index.json'
TEXT_SUFFIXES = {'.f', '.f90', '.inc', '.json', '.log', '.txt'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': digest(data)}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def copy_text(source, destination, bindings):
    assert source.is_file() and not source.is_symlink(), source
    data = source.read_bytes()
    data.decode('utf-8')
    assert b'\0' not in data, source
    assert not destination.exists(), destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    bindings[str(destination.relative_to(HERE))] = {
        'path': str(source.relative_to(ROOT)), **record(source)}


def capture():
    assert not INDEX.exists(), 'A closed package must not be overwritten'
    result = json.loads((WORK / 'execution-1/execution.json').read_text())
    assert result['complete'] and result['exit_code'] == 0
    bindings = {}
    for source in sorted(WORK.rglob('*')):
        relative = source.relative_to(WORK)
        if source.is_file() and source.suffix in TEXT_SUFFIXES and relative.parts[0] != 'analyzer-smoke':
            copy_text(source, HERE / 'evidence' / relative, bindings)
    build = json.loads((WORK / 'build.json').read_text())
    artifacts = {}
    input_pins = {}
    for metadata in build['build_metadata'].values():
        input_pins.update(metadata.get('input_sha256', metadata.get('source_hashes')))
    for relative, expected in sorted(input_pins.items()):
        source = ROOT / relative
        assert digest(source.read_bytes()) == expected, source
        if source.suffix in TEXT_SUFFIXES or source.suffix == '.py':
            copy_text(source, HERE / 'inputs' / relative, bindings)
        else:
            artifacts[relative] = record(source)
    for relative in [
        'docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/run_host_rescue.py',
        'docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/adjudicate_host.py',
        'docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/host-dispatch-release.json',
        'docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/release-raw-and-native.json',
        'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py',
    ]:
        if relative not in input_pins:
            copy_text(ROOT / relative, HERE / 'inputs' / relative, bindings)
    save(HERE / 'bindings.json', {'copies': bindings, 'native_artifacts_hash_only': artifacts})
    files = {}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file():
            continue
        assert not path.is_symlink()
        path.read_bytes().decode('utf-8')
        assert b'\0' not in path.read_bytes()
        files[str(path.relative_to(HERE))] = record(path)
    save(INDEX, {'schema': 1, 'files': files})


def verify(live):
    index = json.loads(INDEX.read_text())
    assert index['schema'] == 1
    for relative, expected in index['files'].items():
        path = HERE / relative
        assert path.is_file() and not path.is_symlink()
        assert record(path) == expected, relative
    bindings = json.loads((HERE / 'bindings.json').read_text())
    for relative, source in bindings['copies'].items():
        expected = {key: source[key] for key in ('bytes', 'sha256')}
        assert record(HERE / relative) == expected, relative
        if live:
            assert record(ROOT / source['path']) == expected, source['path']
    if live:
        for relative, expected in bindings['native_artifacts_hash_only'].items():
            assert record(ROOT / relative) == expected, relative
    execution = json.loads((HERE / 'evidence/execution-1/execution.json').read_text())
    rows = json.loads((HERE / 'evidence/execution-1/rows.json').read_text())
    comparison = json.loads((HERE / 'evidence/execution-1/historical-comparison.json').read_text())
    assert execution['complete'] and execution['exit_code'] == 0
    assert rows['state'] == 'complete' and len(rows['rows']) == 8
    assert rows['all_default_kept_dp'] and rows['all_diagnostic_rescued_qp']
    assert len(comparison['rows']) == 8
    for row in comparison['rows']:
        assert all(row['historical_result_bit_equal'])
        assert row['historical_accuracy_bit_equal']
        assert all(math.isfinite(value) for value in row['truth_symmetric_errors'])
        if row['case'][0] == 1e-6:
            assert max(row['truth_symmetric_errors']) <= 3e-6
    print(json.dumps({'verification': 'pass', 'files': len(index['files']),
                      'synthetic_cases': 8, 'physical_amplitude_qualified': False,
                      'native_execution': False, 'live_bindings_checked': live}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if args.capture:
        capture()
    verify(args.live)
