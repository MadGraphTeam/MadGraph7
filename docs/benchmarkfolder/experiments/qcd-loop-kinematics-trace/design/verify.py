#!/usr/bin/env python3
"""Check portable design files and read-only original source bindings."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def check(path, entry):
    assert not path.is_symlink(), str(path)
    data = path.read_bytes()
    assert len(data) == entry['bytes'], str(path)
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], str(path)


def main():
    index_path = HERE / 'evidence-index.json'
    index = json.loads(index_path.read_text())
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*')
              if p.is_file() and p != index_path}
    assert actual == set(index['files']), 'Portable inventory differs'
    for name, entry in index['files'].items():
        check(HERE / name, entry)
    bindings = json.loads((HERE / 'source-bindings.json').read_text())
    for name, entry in bindings['source_inputs'].items():
        check(ROOT / name, entry)
    for anchor in bindings['source_anchors']:
        lines = (ROOT / anchor['path']).read_text().splitlines()
        assert anchor['text'] in lines[anchor['line'] - 1], anchor
    print(json.dumps({'portable_files_verified': len(actual),
                      'source_files_verified': len(bindings['source_inputs']),
                      'native_executed': False,
                      'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest()},
                     sort_keys=True))


if __name__ == '__main__':
    main()
