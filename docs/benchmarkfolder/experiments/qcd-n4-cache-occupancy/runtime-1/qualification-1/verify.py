"""Portable verification of the retained failed cache diagnostic; no native calls."""
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert __debug__, 'Run without Python optimization'
    index = json.loads((HERE / 'index.json').read_text())
    for name, entry in index['files'].items():
        path = HERE / name
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'] and digest(raw) == entry['sha256'], name
    copies = json.loads((HERE / 'copies.json').read_text())
    decoded = {}
    for name, entry in copies.items():
        raw = gzip.decompress((HERE / name).read_bytes())
        raw.decode('utf-8')
        assert len(raw) == entry['raw_bytes'] and digest(raw) == entry['raw_sha256'], name
        decoded[name.removeprefix('raw/').removesuffix('.gz')] = raw
    execution = json.loads(decoded['execution.json'])
    assert execution['complete'] is False
    assert execution['protected_before'] == execution['protected_after']
    assert set(execution['arms']) == {'original'}
    arm = execution['arms']['original']
    assert arm['process_spawned'] and arm['process_reaped'] and arm['returncode'] == -9
    assert arm['copied_inputs_unchanged_after_failure'] and arm['complete'] is False
    assert arm['seconds_cap'] == 180 and execution['automatic_retry_count'] == 0
    assert decoded['original/stdout.log'] == b''
    assert 'TimeoutError' in execution['error']
    summary = json.loads((HERE / 'summary.json').read_text())
    assert summary['runtime_sha256'] == digest(decoded['execution.json'])
    assert summary['state'] == 'incomplete' and summary['unstarted_arms'] == ['off', 'on']
    assert not any(summary[k] for k in ('complete', 'cache_occupancy_measured', 'parity_qualified', 'performance_eligible'))
    print('PASS: retained incomplete attempt, owned worker reaped; no cache or performance qualification')


if __name__ == '__main__':
    main()
