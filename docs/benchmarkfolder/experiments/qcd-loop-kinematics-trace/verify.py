#!/usr/bin/env python3
"""Verify the complete trace-design/Born-consistency package, read-only."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    index_path = HERE / 'evidence-index.json'
    index = json.loads(index_path.read_text())
    found = set()
    for path in HERE.rglob('*'):
        assert not path.is_symlink(), str(path)
        if not path.is_file() or path == index_path:
            continue
        name = str(path.relative_to(HERE))
        found.add(name)
        data = path.read_bytes()
        assert path.suffix not in ('.pyc', '.o', '.a', '.so', '.dylib'), name
        assert data[:4] != b'\x7fELF', name
        entry = index['files'][name]
        assert len(data) == entry['bytes'] and sha(path) == entry['sha256'], name
    assert found == set(index['files']), 'Missing/unindexed portable files'
    design = json.loads((HERE / 'design/source-bindings.json').read_text())
    for name, entry in design['source_inputs'].items():
        assert sha(ROOT / name) == entry['sha256'], name
    execution = json.loads((HERE / 'born-consistency/native-1/execution.json').read_text())
    assert execution['complete'] and execution['protected_inputs_unchanged']
    assert execution['protected_inputs_before'] == execution['protected_inputs_after']
    for name, expected in execution['protected_inputs_before'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in execution['logs_sha256'].items():
        assert sha(HERE / 'born-consistency/native-1' / name) == expected, name
    for file, field in [('born-consistency/native-1/analysis.json', 'source_bindings'),
                        ('design-amendment-1/targets.json', 'bindings')]:
        data = json.loads((HERE / file).read_text())
        for name, expected in data[field].items():
            assert sha(ROOT / name) == expected, name
    print(json.dumps({'verified_unique_files': len(found), 'native_executed': False,
                      'production_promoted': False, 'index_sha256': sha(index_path)}, sort_keys=True))


if __name__ == '__main__':
    main()
