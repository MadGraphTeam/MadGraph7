#!/usr/bin/env python3
"""Verify retained allocation evidence without disposable builds or execution."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def files():
    return {str(p.relative_to(HERE)): {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
            for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    summary = runpy.run_path(str(HERE/'summarize.py'))['summarize']()
    assert json.loads((HERE/'summary.json').read_text()) == summary
    result = {'schema': 'isolated-mp-tensor-allocation-retention-v1', 'files': files(),
              'passed_correctness': True, 'acceptance_evidence': False,
              'performance_measured': False, 'production_changed': False,
              'excluded': ['Disposable build/cache/executable files remain inside .benchmark-work; original build products are represented by hashes only.', '__pycache__ files are not evidence.']}
    result['identity'] = sha(canonical(result))
    manifest = HERE/'manifest.json'
    if args.seal:
        with manifest.open('x') as stream:
            stream.write(json.dumps(result, indent=2)+'\n')
    else:
        assert json.loads(manifest.read_text()) == result
    print(json.dumps({'verified': True, 'files':len(result['files']), 'identity':result['identity']}))


if __name__ == '__main__':
    main()
