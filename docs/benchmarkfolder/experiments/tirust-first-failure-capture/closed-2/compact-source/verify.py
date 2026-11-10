"""Losslessly reconstruct the original closed reader, then verify it in memory.

No original host path is opened. Only two indexed GNU link maps are compressed.
The inherited verifier is executed from exact checked bytes, without importing
its file or creating a reconstructed tree on disk.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import types
import zlib

ORIGINAL_INDEX_SHA256 = 'd85a79ce185660512334c2f51e01cfa1734304149a29188fd62c782625a7092e'
INHERITED_VERIFIER_SHA256 = '9592792bd86d22cc251b3060235804cec17039c82e22b6caf0a184f654186cd3'
MAPS = ('private-sources/qcd_n3_uux_zgg/capture-link.map',
        'private-sources/qcd_n4_uux_zzgg/capture-link.map')
MAX_MAP_BYTES = 40 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def json_value(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=constant)


def logical_path(name):
    require(isinstance(name, str) and name and '\\' not in name and
            not PurePosixPath(name).is_absolute() and
            all(part not in ('', '.', '..') for part in name.split('/')),
            'Unsafe compact payload path')
    return name


def decode_gzip(blob, size, checksum):
    """One bounded gzip member; no flush, concatenation or trailing bytes."""
    require(type(size) is int and 0 <= size <= MAX_MAP_BYTES, 'Invalid bounded map size')
    require(isinstance(blob, bytes) and len(blob) >= 18 and blob[:3] == b'\x1f\x8b\x08'
            and blob[3] == 0 and blob[4:8] == b'\x00' * 4,
            'Expected a deterministic gzip member without optional headers')
    try:
        decoder = zlib.decompressobj(31)
        raw = decoder.decompress(blob, size + 1)
    except zlib.error as error:
        raise ValueError('Invalid gzip stream') from error
    require(len(raw) == size and decoder.eof and not decoder.unconsumed_tail
            and not decoder.unused_data, 'Truncated, oversized, concatenated or trailing gzip data')
    require(sha(raw) == checksum, 'Decompressed map hash differs')
    return raw


def physical_payloads(index, reader):
    require(index.get('schema') == 'madloop-tirust-first-failure-capture-compact-v1', 'Wrong compact schema')
    require(index.get('original_index_sha256') == ORIGINAL_INDEX_SHA256 and
            type(index.get('native_evaluations')) is int and index['native_evaluations'] == 0 and
            type(index.get('logical_data_changes')) is int and index['logical_data_changes'] == 0,
            'Compaction must preserve the exact original without numerical work')
    files = index.get('files')
    require(isinstance(files, dict) and files and 'index.json' not in files, 'Missing compact physical inventory')
    if hasattr(reader, 'paths'):
        require(set(reader.paths) == set(files) | {'index.json'}, 'Missing or unindexed compact payloads')
    payloads = {}
    for name, expected in files.items():
        logical_path(name)
        blob = reader(name)
        require(isinstance(blob, bytes) and {'sha256': sha(blob), 'bytes': len(blob)} == expected,
                'Compact payload differs: ' + name)
        payloads[name] = blob
    return payloads


def reconstruct(index, payloads):
    original_blob = payloads['original-closed-1-index.json']
    require(sha(original_blob) == ORIGINAL_INDEX_SHA256, 'Original closed inventory differs')
    original = json_value(original_blob)
    require(original.get('schema') == 'madloop-tirust-first-failure-capture-closed-v1', 'Wrong original closed schema')
    compression = index.get('compression')
    require(isinstance(compression, dict) and set(compression) == set(MAPS),
            'Only the exact two complete GNU link maps may be compressed')
    logical = {'index.json': original_blob}
    for name, expected in original['files'].items():
        logical_path(name)
        if name in compression:
            item = compression[name]
            require(item == {'path': name + '.gz', 'sha256': expected['sha256'], 'bytes': expected['bytes']},
                    'Compressed map metadata does not join the original inventory')
            require(name not in payloads, 'Original large map was duplicated physically')
            logical[name] = decode_gzip(payloads[item['path']], expected['bytes'], expected['sha256'])
        else:
            require(name in payloads and {'sha256': sha(payloads[name]), 'bytes': len(payloads[name])} == expected,
                    'Uncompressed logical payload differs from the original')
            logical[name] = payloads[name]
    source_blob = payloads['compact-source/source-index.json']
    require(sha(source_blob) == index.get('compact_source_index_sha256'), 'Compact source seal differs')
    source = json_value(source_blob)
    source_names = set()
    for name, expected in source['files'].items():
        name = 'compact-source/' + logical_path(name)
        require(name in payloads and {'sha256': sha(payloads[name]), 'bytes': len(payloads[name])} == expected,
                'Compact source payload differs')
        source_names.add(name)
    expected_physical = ((set(original['files']) - set(MAPS)) |
                         {name + '.gz' for name in MAPS} | source_names |
                         {'original-closed-1-index.json', 'COMPACT_README.md', 'compact-source/source-index.json'})
    require(set(payloads) == expected_physical, 'Physical inventory contains unexpected compaction changes')
    return logical


def inherited_verifier(blob):
    require(sha(blob) == INHERITED_VERIFIER_SHA256, 'Inherited verifier bytes differ')
    module = types.ModuleType('checked_original_capture_verifier')
    module.__file__ = 'retained-logical/retention-source/verify.py'
    exec(compile(blob, module.__file__, 'exec'), module.__dict__)
    return module


def verify(root=None, *, read_bytes=None):
    if read_bytes is None:
        base = Path(root or Path(__file__).resolve().parent.parent)
        require(base.is_dir() and not base.is_symlink(), 'Expected a regular compact directory')
        def read_bytes(name):
            logical_path(name)
            path = base
            for part in name.split('/'):
                path = path / part
                require(not path.is_symlink(), 'Compact paths cannot follow symlinks')
            require(path.is_file(), 'Missing compact payload: ' + name)
            return path.read_bytes()
        read_bytes.paths = {str(path.relative_to(base)) for path in base.rglob('*')
                            if path.is_file() or path.is_symlink()}
    index_blob = read_bytes('index.json')
    index = json_value(index_blob)
    payloads = physical_payloads(index, read_bytes)
    logical = reconstruct(index, payloads)
    def logical_reader(name):
        logical_path(name)
        require(name in logical, 'Inherited verifier requested an unindexed logical payload')
        return logical[name]
    logical_reader.paths = set(logical)
    original = inherited_verifier(logical['retention-source/verify.py']).verify(read_bytes=logical_reader)
    require(original.get('verified') is True and original.get('index_sha256') == ORIGINAL_INDEX_SHA256,
            'Original logical verification failed')
    return {'verified': True, 'index_sha256': sha(index_blob),
            'original_index_sha256': ORIGINAL_INDEX_SHA256,
            'compact_source_index_sha256': index['compact_source_index_sha256'],
            'compressed_members': len(MAPS), 'logical_payloads': len(logical) - 1,
            'physical_payloads': len(payloads), 'logical_data_changes': 0, 'native_evaluations': 0,
            'summary': original['summary']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, allow_nan=False))
