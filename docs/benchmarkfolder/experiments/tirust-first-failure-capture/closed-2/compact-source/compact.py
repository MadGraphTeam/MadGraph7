#!/usr/bin/env python3
"""Losslessly compact only the two closed diagnostic GNU link maps."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL_INDEX = 'd85a79ce185660512334c2f51e01cfa1734304149a29188fd62c782625a7092e'
MAPS = ('private-sources/qcd_n3_uux_zgg/capture-link.map',
        'private-sources/qcd_n4_uux_zzgg/capture-link.map')


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def compact(original, destination):
    require(not destination.exists() and not destination.is_symlink(), 'Fresh compact destination required')
    require(destination.parent.resolve() == destination.parent.absolute(), 'Destination parent must not traverse aliases')
    require(not destination.absolute().is_relative_to(original.resolve()) and
            not destination.absolute().is_relative_to(HERE.resolve()),
            'Compaction must not add files to immutable input/source namespaces')
    index_blob = (original / 'index.json').read_bytes()
    require(sha(index_blob) == ORIGINAL_INDEX, 'Original closed inventory differs')
    index = json.loads(index_blob)
    source = json.loads((HERE / 'source-index.json').read_bytes())
    for name, expected in source['files'].items():
        require(Path(name).name == name, 'Unsafe compact source name')
        blob = (HERE / name).read_bytes()
        require({'sha256': sha(blob), 'bytes': len(blob)} == expected, 'Compact source differs')
    payloads = {'original-closed-1-index.json': index_blob}
    compression = {}
    for name, expected in index['files'].items():
        path = original / name
        require(not path.is_symlink() and path.is_file(), 'Expected regular original payload')
        blob = path.read_bytes()
        require({'sha256': sha(blob), 'bytes': len(blob)} == expected, 'Original payload differs: ' + name)
        if name in MAPS:
            compressed_name = name + '.gz'
            payloads[compressed_name] = gzip.compress(blob, compresslevel=9, mtime=0)
            compression[name] = {'path': compressed_name, 'sha256': sha(blob), 'bytes': len(blob)}
        else:
            payloads[name] = blob
    require(set(compression) == set(MAPS), 'Expected both complete diagnostic link maps')
    payloads['COMPACT_README.md'] = (HERE / 'CLOSED_README.md').read_bytes()
    for name in (*source['files'], 'source-index.json'):
        payloads['compact-source/' + name] = (HERE / name).read_bytes()
    compact_index = {'schema': 'madloop-tirust-first-failure-capture-compact-v1',
                     'files': {name: {'sha256': sha(blob), 'bytes': len(blob)} for name, blob in sorted(payloads.items())},
                     'compression': compression, 'original_index_sha256': ORIGINAL_INDEX,
                     'compact_source_index_sha256': sha((HERE / 'source-index.json').read_bytes()),
                     'native_evaluations': 0, 'logical_data_changes': 0}
    destination.mkdir(parents=True)
    for name, blob in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(blob)
    with (destination / 'index.json').open('xb') as stream:
        stream.write(encode(compact_index))
    verifier_blob = (HERE / 'verify.py').read_bytes()
    require(sha(verifier_blob) == source['files']['verify.py']['sha256'], 'Compact verifier changed')
    import types
    verifier = types.ModuleType('compact_closed_verifier')
    verifier.__file__ = str(HERE / 'verify.py')
    exec(compile(verifier_blob, verifier.__file__, 'exec'), verifier.__dict__)
    result = verifier.verify(destination)
    print(json.dumps({'verified': result['verified'], 'index_sha256': result['index_sha256'],
                      'original_index_sha256': ORIGINAL_INDEX, 'payloads': len(payloads),
                      'bytes': sum(map(len, payloads.values()))}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    compact(args.original, args.destination)
