#!/usr/bin/env python3
"""Portable inventory check for the research-only QP input policy."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INDEX = HERE / 'evidence-index.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--sources', action='store_true',
        help='Also reconstruct audited source bytes from snapshots or recorded Git blobs.')
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p != INDEX)
    assert all('__pycache__' not in p.parts and p.suffix != '.pyc' for p in paths)
    files = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    if args.write:
        INDEX.write_text(json.dumps({'schema': 1, 'native_execution': False,
            'production_change': False, 'qualification': False, 'files': files},
            indent=2, sort_keys=True)+'\n')
    else:
        assert json.loads(INDEX.read_text())['files'] == files
    if args.sources:
        route = json.loads((HERE / 'source-routing.json').read_text())
        assert hashlib.sha256((HERE / 'source-routing.md').read_bytes()).hexdigest() == route['source_routing_md_sha256']
        for record in route['sources']:
            if 'full_source_gzip_base64' in record:
                data = gzip.decompress(base64.b64decode(record['full_source_gzip_base64']))
            else:
                data = subprocess.check_output(['git', 'show', record['git_blob_at_recorded_head']], cwd=ROOT)
            assert len(data) == record['bytes'], record['path']
            assert hashlib.sha256(data).hexdigest() == record['sha256'], record['path']
            lines = data.decode().splitlines()
            assert len(lines) == record['line_count'], record['path']
            for excerpt in record['excerpts']:
                selected = excerpt['numbered_text'].splitlines()
                assert len(selected) == excerpt['end_line']-excerpt['start_line']+1
                for number, line in zip(range(excerpt['start_line'], excerpt['end_line']+1), selected):
                    assert int(line[:6]) == number
                    assert line[6:] in (' '+lines[number-1], '  '+lines[number-1]), (record['path'], number)
            if 'working_tree_unified_diff' in record:
                assert hashlib.sha256(record['working_tree_unified_diff'].encode()).hexdigest() == record['working_tree_diff_sha256']
        for record in route['prior_manifest_context']:
            assert hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() == record['sha256']
        print(json.dumps({'reconstructed_sources': len(route['sources']), 'source_snapshots_and_excerpts': 'verified'}))
    print(json.dumps({'files': len(files), 'index_sha256': hashlib.sha256(INDEX.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
