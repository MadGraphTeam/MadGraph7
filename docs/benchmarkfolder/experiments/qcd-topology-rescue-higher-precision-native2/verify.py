#!/usr/bin/env python3
"""Check the completed native2 package without launching any process."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    entries = json.loads((HERE / 'evidence-index.json').read_text())['files']
    paths = list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()} == set(entries) | {'evidence-index.json'}
    for name, entry in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p = HERE / name
        raw = p.read_bytes()
        assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'], name
        assert not raw.startswith(b'\x7fELF') and p.suffix not in ('.pyc', '.o', '.mod', '.a', '.so'), name
    spec = importlib.util.spec_from_file_location('authenticated_native2_audit', HERE / 'audit_result.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    result = audit.compute(HERE, args.live)
    recorded = json.loads((HERE / 'audit-result.json').read_text())
    observed = dict(result)
    observed['live_hashes_checked'] = recorded['live_hashes_checked']
    assert observed == recorded
    result['portable_files'] = len(entries)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
