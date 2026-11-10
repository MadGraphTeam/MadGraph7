#!/usr/bin/env python3
"""Verify the offline-parser packet, with no integration or reducer call."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p/'madgraph').is_dir() and (p/'tools').is_dir())


def check(path, record):
    data = path.read_bytes()
    assert len(data) == record['bytes'], str(path)
    assert hashlib.sha256(data).hexdigest() == record['sha256'], str(path)


def main():
    index = json.loads((HERE/'source-index.json').read_text())
    for record in index['files']:
        path = HERE/record['path']
        assert path.resolve().is_relative_to(HERE.resolve())
        check(path, record)
    for record in index['inputs']:
        path = REPO/record['path']
        assert path.resolve().is_relative_to(REPO.resolve())
        check(path, record)
    spec = importlib.util.spec_from_file_location('rank6_fp_compare_verify', HERE/'compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.retained_snapshot()
    assert index['native_reducer_calls'] == index['physics_quadrature_calls'] == 0
    print(json.dumps({'passed': True, 'owned_files': len(index['files']),
                      'bound_inputs': len(index['inputs']), 'physics_quadrature_calls': 0,
                      'native_reducer_calls': 0}))


if __name__ == '__main__':
    main()
