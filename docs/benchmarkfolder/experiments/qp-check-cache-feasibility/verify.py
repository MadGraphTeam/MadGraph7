#!/usr/bin/env python3
"""Read-only integrity and exact-count verification; no native library loading."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    entries = json.loads((HERE / 'evidence-index.json').read_text())['files']
    files = list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in files)
    assert {str(p.relative_to(HERE)) for p in files if p.is_file()} == set(entries) | {'evidence-index.json'}
    for name, entry in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path = HERE / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        assert not path.read_bytes().startswith(b'\x7fELF') and path.suffix not in ('.so', '.a', '.o', '.mod', '.pyc')
    notes = json.loads((HERE / 'source-notes.json').read_text())
    for name, entry in notes['sources'].items():
        path = ROOT / name
        # Live source files are optional for portable report replay. The exact
        # copied excerpts remain authenticated by this package in either case.
        if '--live' in sys.argv:
            assert sha(path) == entry['sha256'], name
            text = path.read_text()
            assert all(a['text'] in text for a in entry['anchors'])
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location('verified_check_cache_analysis', HERE / 'analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.compute()
    assert result == json.loads((HERE / 'counts.json').read_text())
    print(json.dumps({'verification': 'pass', 'portable_files': len(entries),
                      'native_executed': False, 'cache_implemented': False,
                      'check_successful_repeats': {r['arm']: r['candidate_successful_repeats']
                        for r in result['rows'] if r['ctmode'] == 5 and r['calls']}}, sort_keys=True))


if __name__ == '__main__':
    main()
