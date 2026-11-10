#!/usr/bin/env python3
"""Verify all-component source proposal without any fixture integration."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p/'madgraph').is_dir() and (p/'tools').is_dir())


def main():
    index = json.loads((HERE/'source-index.json').read_text())
    for record in index['files']:
        path = HERE/record['path']
        assert path.resolve().is_relative_to(HERE.resolve())
        raw = path.read_bytes()
        assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], str(path)
    for record in index['inputs']:
        path = REPO/record['path']
        assert path.resolve().is_relative_to(REPO.resolve())
        assert path.stat().st_size == record['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], str(path)
    spec = importlib.util.spec_from_file_location('all_components_source_verify', HERE/'all_components.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.numpy = lambda: (_ for _ in ()).throw(AssertionError('source verification must not integrate'))
    proposal = json.loads(json.dumps(module.proposal()))
    assert proposal == json.loads((HERE/'proposal.json').read_text())
    assert index['actual_fixture_vectorized_evaluations'] == index['native_reducer_calls'] == 0
    print(json.dumps({'passed': True, 'owned_files': len(index['files']),
                      'bound_inputs': len(index['inputs']), 'components': 210,
                      'primitive_moments': 296, 'actual_fixture_evaluations': 0,
                      'native_reducer_calls': 0}))


if __name__ == '__main__':
    main()
