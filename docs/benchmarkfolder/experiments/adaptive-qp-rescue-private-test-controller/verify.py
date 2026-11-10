#!/usr/bin/env python3
"""Read-only verification; no compiler, test process, or library loading."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    entries = json.loads((HERE / 'evidence-index.json').read_text())['files']
    paths = list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()} == set(entries) | {'evidence-index.json'}
    for name, entry in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path = HERE / name
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
        assert not path.read_bytes().startswith(b'\x7fELF') and path.suffix not in ('.o', '.a', '.so', '.mod', '.pyc')
    binding = json.loads((HERE / 'bindings.json').read_text())
    assert not binding['native_executed'] and not binding['slot_admitted']
    assert binding['source_index_sha256'] == '74058446fda67d455c78bf1d3404a10d9bf9a5cdb4a66f6a9ca47d16c4ac9603'
    assert sha(HERE / 'source/source-index.json') == binding['source_index_sha256']
    source_index = json.loads((HERE / 'source/source-index.json').read_text())['files']
    assert {p.name for p in (HERE / 'source').iterdir()} == set(source_index) | {'source-index.json'}
    for name, entry in binding['source_copies'].items():
        path = HERE / 'source' / name
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
        if '--live' in sys.argv:
            assert sha(ROOT / entry['original_path']) == entry['sha256']
    for field in ('build_index', 'build_execution'):
        assert sha(ROOT / binding[field]['path']) == binding[field]['sha256']
    build = json.loads((ROOT / binding['build_execution']['path']).read_text())
    executable = binding['test_executable']
    assert not executable['copied'] and build['complete'] and not build['native_tests_executed']
    assert executable['sha256'] == build['test_executable_not_run']['sha256']
    assert executable['path'] == build['test_executable_not_run']['cargo_record']['executable']
    if '--live' in sys.argv:
        assert sha(Path(executable['path'])) == executable['sha256']
    print(json.dumps({'verification': 'pass', 'portable_files': len(entries),
                      'source_index': binding['source_index_sha256'],
                      'native_executed': False, 'slot_admitted_at_freeze': False}, sort_keys=True))


if __name__ == '__main__':
    main()
