#!/usr/bin/env python3
"""Verify retained unit results offline; do not execute tests or load a DSO."""
from pathlib import Path
import argparse
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()
BUILD = HERE.with_name('adaptive-qp-rescue-private-build')
CONTROLLER = HERE.with_name('adaptive-qp-rescue-private-test-controller')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(folder, expected=None):
    index = folder / 'evidence-index.json'
    if expected:
        assert sha(index) == expected
    entries = json.loads(index.read_text())['files']
    paths = list(folder.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(folder)) for p in paths if p.is_file()} == set(entries) | {'evidence-index.json'}
    for name, entry in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p = folder / name
        raw = p.read_bytes()
        assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'], name
        assert not raw.startswith(b'\x7fELF') and p.suffix not in ('.so', '.a', '.o', '.mod', '.pyc'), name
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    entries = inventory(HERE)
    inventory(BUILD, '1a4af7f8f8be9ce8880ebc9ce466f66afec961be4c24b6978cdd8fdfa28fc8d8')
    inventory(CONTROLLER, '615bc8f52e4128fd04aa507ae7171c74a52c82b826148de243e29c5b72036e4d')
    record = json.loads((HERE / 'attempt1/execution.json').read_text())
    build = json.loads((BUILD / 'attempt-3/execution.json').read_text())
    assert record['build_index_sha256'] == sha(BUILD / 'evidence-index.json')
    assert record['source_index_sha256'] == build['source_index_sha256'] == 'f5550b8719e76e177ad6b73d456cc655815770eb7092a8fabdc776c415c03213'
    assert record['test_executable_sha256'] == build['test_executable_not_run']['sha256'] == 'd22953ed3b955adf7dcce3831259060cec9e974816d432a4a86fd000b1afe8b7'
    assert record['complete'] and record['returncode'] == 0
    assert not record['physical_amplitude_qualified'] and not record['production_promoted']
    assert (record['cpu'], record['threads'], record['inclusive_deadline_seconds'], record['automatic_retry_count']) == (11, 1, 180, 0)
    assert record['command'][1:3] == ['-c', '11'] and record['command'][4:] == ['--test-threads=1', '--nocapture']
    assert record['command'][3] == build['test_executable_not_run']['cargo_record']['executable']
    assert record['environment'] == build['environment']
    assert record['process']['process_spawned'] and record['process']['process_reaped']
    assert record['process']['returncode'] == 0 and record['process']['process_wall_seconds'] <= 180
    assert record['protected_inputs_before'] == record['protected_inputs_after']
    pins = record['protected_inputs_before']
    assert all(pins[k] == v for k, v in build['protected_inputs_after'].items())
    assert pins[str(ROOT / '.tirust-build/adaptive-qp-rescue-test-controller/source-index.json')] == '74058446fda67d455c78bf1d3404a10d9bf9a5cdb4a66f6a9ca47d16c4ac9603'
    if args.live:
        assert all(sha(Path(name)) == digest for name, digest in pins.items())
    log = HERE / 'attempt1/tests.log'
    assert sha(log) == record['log_sha256']
    text = log.read_text()
    results = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;', text)
    assert results == [('20', '0', '0', '0', '0')]
    names = re.findall(r'^test ([A-Za-z0-9_:]+) \.\.\. (?:ok|.*)$', text, re.M)
    assert names == record['test_names'] and len(names) == len(set(names)) == 20
    assert record['test_result'] == {'passed': 20, 'failed': 0, 'ignored': 0}
    print(json.dumps({'verification': 'pass', 'portable_files': len(entries), 'passed': 20,
                      'live_pins_checked': len(pins) if args.live else 0,
                      'native_executed_by_verifier': False, 'physical_amplitude_qualified': False}, sort_keys=True))


if __name__ == '__main__':
    main()
