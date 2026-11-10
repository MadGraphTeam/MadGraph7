#!/usr/bin/env python3
"""Verify the source-only observer inventory and immutable source routing inputs."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import runpy

ROOT = Path.cwd().resolve()
HERE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inventory = json.loads((HERE / 'evidence-index.json').read_text())
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
    assert actual == set(inventory['files']) | {'evidence-index.json'}, actual ^ set(inventory['files'])
    for name, entry in inventory['files'].items():
        path = HERE / name
        assert not path.is_symlink(), name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
        assert path.suffix not in ('.pyc', '.o', '.mod', '.a', '.so'), name
        assert not path.read_bytes().startswith(b'\x7fELF'), name
    report = json.loads((HERE / 'prepared.json').read_text())
    assert not report['native_executed'] and not report['production_modified']
    assert sha(HERE / 'prepare.py') == report['prepare_sha256']
    for name, entry in report['protected_source_inputs'].items():
        path = ROOT / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    for key in ('protected_historical_inputs', 'frozen_evidence_indices'):
        for name, digest in report[key].items():
            assert sha(ROOT / name) == digest, name
    count = 0
    ids = []
    for entry in report['source_transformations']:
        old_path = ROOT / entry['original_path']
        new_path = ROOT / entry['observed_path']
        old, new = old_path.read_text(), new_path.read_text()
        assert sha(old_path) == entry['original_sha256']
        assert sha(new_path) == entry['observed_sha256']
        assert (new_path.parent / (new_path.name + '.original')).read_text() == old
        matches = list(re.finditer(r'(?ms)^C QOBS_BEGIN (\d+)\n.*?^C QOBS_END \1\n', new))
        stripped = re.sub(r'(?ms)^C QOBS_BEGIN (\d+)\n.*?^C QOBS_END \1\n', '', new)
        assert stripped == old and 'QOBS_BEGIN' not in stripped
        ids.extend(int(m[1]) for m in matches)
        count += len(matches)
        patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                          fromfile=new_path.name + '.original', tofile=new_path.name))
        assert (new_path.parent / (new_path.name + '.patch')).read_text() == patch
    assert count == report['insertion_blocks'] and sorted(ids) == list(range(1, count + 1))
    assert 'CALL QO_END_POINT(CODE)\n        CALL QO_HEALTH_REPORT(I)' in (HERE / 'sources/amplitude_driver.f').read_text()
    routing_namespace = runpy.run_path(str(HERE / 'derive_routing.py'))
    routing = routing_namespace['derive']()
    assert routing == json.loads((HERE / 'routing-modes.json').read_text())
    print(json.dumps({'files': len(inventory['files']), 'source_count': len(report['source_transformations']),
                      'insertion_blocks': count, 'native_executed': False,
                      'routing_summary': routing['summary'],
                      'source_and_inventory_verification': 'pass'}, sort_keys=True))


if __name__ == '__main__':
    main()
