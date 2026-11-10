#!/usr/bin/env python3
"""Restore captured UTF-8 text only; never execute or reconstruct native files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def contents():
    combined = {}
    for filename in ('sources.json.gz', 'prior-build-history.json.gz'):
        bundle = json.loads(gzip.decompress((HERE / filename).read_bytes()))
        for name, meta in bundle['files'].items():
            name = name if filename == 'sources.json.gz' else bundle['base'] + '/' + name
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts, name
            digest = meta['sha256']
            data = bundle['blobs'][digest].encode('utf-8')
            assert len(data) == meta['bytes'] and hashlib.sha256(data).hexdigest() == digest
            assert b'\0' not in data and not data.startswith((b'\x7fELF', b'!<arch>\n'))
            if name in combined:
                assert combined[name] == data, name
            combined[name] = data
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    assert args.list or args.destination, 'Select --list or --destination'
    files = {k: v for k, v in contents().items() if k.startswith(args.prefix)}
    if args.list:
        for name in sorted(files):
            print(name)
    if args.destination:
        root = args.destination.resolve()
        for name, data in sorted(files.items()):
            dest = root / name
            assert dest.resolve().is_relative_to(root) and not dest.is_symlink(), name
            if dest.exists():
                assert dest.is_file() and dest.read_bytes() == data, name
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(data)
        print(json.dumps({'restored_text_files': len(files), 'destination': str(root)}))


if __name__ == '__main__':
    main()
