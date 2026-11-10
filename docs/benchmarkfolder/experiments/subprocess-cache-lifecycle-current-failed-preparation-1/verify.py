#!/usr/bin/env python3
"""Offline portable verification; recorded machine paths are never opened."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root):
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()
    summary = json.loads((root / 'summary.json').read_text())
    packed = (root / 'payload.json.gz').read_bytes()
    assert sha(packed) == summary['payload_sha256']
    data = json.loads(gzip.decompress(packed))
    records, blobs = data['records'], data['blobs']
    for name, record in records.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and str(path) == name
        raw = blobs[record['sha256']].encode('utf-8')
        assert b'\0' not in raw and len(raw) == record['bytes'] and sha(raw) == record['sha256']
    def raw(name):
        return blobs[records[name]['sha256']].encode('utf-8')
    def load(name):
        return json.loads(raw(name))
    source = raw('protocol/source-index.json')
    assert sha(source) == summary['source_index_sha256']
    for name, record in json.loads(source)['files'].items():
        observed = raw('protocol/' + name)
        assert sha(observed) == record['sha256'] and len(observed) == record['size']
    names = ('freeze', 'materialize', 'generate', 'prepare-build', 'build')
    stages = {name: load('report/stage-' + name + '.json') for name in names}
    assert stages == summary['stages']
    assert [s['status'] for s in stages.values()] == ['passed'] * 4 + ['failed']
    assert all(not s['native_execution'] for s in stages.values())
    assert 'Actual linker input absent before link' in stages['build']['exception']
    assert not any('native-' in name or 'stage-run' in name or 'stage-smoke' in name for name in records)
    for name, record in records.items():
        if name.startswith('report/execution/') and name.endswith('.json'):
            item = load(name)
            if 'log_sha256' in item:
                assert sha(raw(name[:-5] + '.txt')) == item['log_sha256']
    for name, item in data['generated_inventory'].items():
        if item.get('retained_utf8'):
            assert sha(raw(name)) == item['sha256']
    assert len(records) == summary['retained_paths'] and len(blobs) == summary['unique_utf8_contents']
    assert not summary['native_execution_performed'] and not summary['binary_contents_retained']
    return {'passed': True, 'retained_paths': len(records), 'source_index_sha256': sha(source),
            'native_execution_performed': False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().root), sort_keys=True, indent=2))
