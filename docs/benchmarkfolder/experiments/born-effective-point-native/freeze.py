#!/usr/bin/env python3
"""Create the outer immutable text-evidence index; never run scientific code."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    target = HERE / 'evidence-index.json'
    assert not target.exists(), 'Frozen index already exists; preserve this milestone.'
    files = {}
    for path in sorted(HERE.rglob('*')):
        assert not path.is_symlink(), path
        if not path.is_file():
            continue
        assert '__pycache__' not in path.parts
        assert path.suffix.lower() not in ('.pyc', '.pkl', '.pickle', '.o', '.a', '.so', '.mod', '.smod', '.exe')
        data = path.read_bytes()
        assert not data.startswith((b'\x7fELF', b'!<arch>\n')), path
        if path.suffix != '.gz':
            data.decode('utf-8')
            assert b'\0' not in data
        files[str(path.relative_to(HERE))] = {
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    result = {'schema': 'private-effective-born-source-build-legacy-v1',
              'scope': 'Complete private source/build/control/legacy milestone; fixed CT4 unchecked. No backend8/9 or retained stress amplitude qualification.',
              'files': files}
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'indexed_files': len(files), 'physical_files_with_index': len(files) + 1,
                      'index_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    main()
