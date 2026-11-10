#!/usr/bin/env python3
"""Verify this source-only packet without integrating the physics fixture."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p/'madgraph').is_dir() and (p/'tools').is_dir())


def check(path, record):
    raw = path.read_bytes()
    assert len(raw) == record['bytes'], str(path)
    assert hashlib.sha256(raw).hexdigest() == record['sha256'], str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='also check optional original reference-source files')
    args = parser.parse_args()
    index = json.loads((HERE/'source-index.json').read_text())
    for record in index['files']:
        path = HERE/record['path']
        assert path.resolve().is_relative_to(HERE.resolve())
        check(path, record)
    for record in index['inputs']:
        path = REPO/record['path']
        assert path.resolve().is_relative_to(REPO.resolve())
        check(path, record)
    if args.live:
        for record in index['optional_live_inputs']:
            check(REPO/record['path'], record)
    spec = importlib.util.spec_from_file_location('rank6_fp_packet_verify', HERE/'quadrature.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.gaussian_rule = lambda *args: (_ for _ in ()).throw(AssertionError('verification must not generate quadrature'))
    assert json.loads(json.dumps(module.proposal())) == json.loads((HERE/'proposal.json').read_text())
    assert index['scope'] == 'source-only; no actual fixture quadrature, reducer execution, build or timing'
    print(json.dumps({'passed': True, 'owned_files': len(index['files']),
                      'bound_repo_inputs': len(index['inputs']),
                      'physics_evaluations': 0, 'native_reducer_calls': 0,
                      'arithmetic_total_error_qualification': 'pending'}))


if __name__ == '__main__':
    main()
