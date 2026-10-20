#!/usr/bin/env python3
"""Verify the retained regression-evidence file set and content."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((HERE/'manifest.json').read_text())
    files = manifest['files']
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if identity != manifest['identity_sha256']:
        raise RuntimeError('Regression evidence manifest identity mismatch')
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file() and path.name != 'manifest.json'}
    if actual != set(files):
        raise RuntimeError('Regression evidence file set changed')
    for relative, item in files.items():
        path = HERE/relative
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError(f'Regression evidence changed: {relative}')
    return {'files': len(files), 'bytes': sum(item['bytes'] for item in files.values()),
            'identity_sha256': identity, 'status': 'verified'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
