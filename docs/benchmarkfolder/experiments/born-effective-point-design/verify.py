#!/usr/bin/env python3
"""Verify frozen design; optional source checks never load native libraries."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INDEX = HERE/'evidence-index.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--sources', action='store_true')
    parser.add_argument('--current-sources', action='store_true')
    args = parser.parse_args()
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p != INDEX)
    assert all('__pycache__' not in p.parts and p.suffix != '.pyc' for p in paths)
    files = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
              'sha256': sha(p)} for p in paths]
    if args.write:
        INDEX.write_text(json.dumps({'schema': 1, 'source_only': True,
            'native_executed': False, 'implementation_qualified': False,
            'files': files}, indent=2, sort_keys=True)+'\n')
    else:
        assert json.loads(INDEX.read_text())['files'] == files
    tests = json.loads((HERE/'tests.json').read_text())
    for name, digest in tests['sources_sha256'].items():
        assert sha(HERE/name) == digest, name
    assert tests['returncode'] == 0
    if args.sources or args.current_sources:
        binding = json.loads((HERE/'source-bindings.json').read_text())
        sources = {}
        for name, row in binding['source_sha256'].items():
            committed = subprocess.check_output(['git', 'cat-file', 'blob', row['git_blob_at_recorded_head']], cwd=ROOT)
            assert hashlib.sha256(committed).hexdigest() == row['git_blob_sha256'], name
            data = gzip.decompress(base64.b64decode(row['working_source_gzip_base64'])) if row['working_differs_from_head'] else committed
            assert hashlib.sha256(data).hexdigest() == row['sha256'] and len(data) == row['bytes'], name
            assert (data != committed) == row['working_differs_from_head'], name
            sources[name] = data.decode()
            if args.current_sources:
                assert sha(ROOT/name) == row['sha256'], name
        for name, digest in binding['evidence_sha256'].items():
            assert sha(ROOT/name) == digest, name
        hooks = json.loads((HERE/'hook-plan.json').read_text())
        for row in hooks['hooks']:
            assert sources[row['source']].count(row['anchor']) == row['expected_occurrences'], row['name']
        print(json.dumps({'sources': len(binding['source_sha256']), 'hook_anchors': len(hooks['hooks']), 'verified': True}))
    print(json.dumps({'files': len(files), 'index_sha256': sha(INDEX)}))


if __name__ == '__main__':
    main()
