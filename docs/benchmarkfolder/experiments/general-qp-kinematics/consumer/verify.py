"""Verify frozen private source evidence and source inputs without executing it."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    index = json.loads((HERE / 'source-index.json').read_text())
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*')
              if p.is_file() and p.name != 'source-index.json'}
    assert actual == set(index['files']), 'Inventory changed'
    for name, expected in index['files'].items():
        assert sha(HERE / name) == expected['sha256'], name
    for name, expected in index['protected_inputs'].items():
        assert sha(ROOT / name) == expected, name
    print(json.dumps({'verified_files': len(actual),
        'protected_inputs': len(index['protected_inputs']),
        'source_only': True, 'source_index_sha256': sha(HERE / 'source-index.json')}))


if __name__ == '__main__':
    main()
