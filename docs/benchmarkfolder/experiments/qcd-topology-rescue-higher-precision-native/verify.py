#!/usr/bin/env python3
"""Verify the immutable failed-attempt package; no native process is launched."""
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def main():
    entries=json.loads((HERE/'evidence-index.json').read_text())['files']
    paths=list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()}==set(entries)|{'evidence-index.json'}
    for name,entry in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=HERE/name;raw=p.read_bytes()
        assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
        assert not raw.startswith(b'\x7fELF') and p.suffix not in ('.so','.a','.o','.mod','.pyc'),name
    from audit_failure import compute
    assert compute()==json.loads((HERE/'failure-audit.json').read_text())
    print(json.dumps({'verification':'pass','portable_files':len(entries),
                     'capture_complete':False,'native_executed_by_verifier':False,'physics_accepted':False},sort_keys=True))


if __name__=='__main__':
    main()
