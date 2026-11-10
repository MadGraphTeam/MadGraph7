#!/usr/bin/env python3
"""Verify the frozen proposal by reading files only; never execute a library."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def main():
    from prepare import derive, require
    index=json.loads((HERE/'evidence-index.json').read_text())
    paths=list(HERE.rglob('*'))
    require(not any(p.is_symlink() for p in paths),'Symlink in proposal.')
    actual={str(p.relative_to(HERE)) for p in paths if p.is_file()}
    require(actual==set(index['files'])|{'evidence-index.json'},'Proposal inventory differs.')
    for name,entry in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe proposal path.')
        data=(HERE/name).read_bytes()
        require(len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],'Proposal bytes differ: '+name)
        require(not data.startswith(b'\x7fELF') and Path(name).suffix not in ('.pyc','.o','.so','.a','.mod'),'Native artifact in proposal.')
    frozen=json.loads((HERE/'proposal.json').read_text())
    require(derive()==frozen,'Source/ELF/input derivation differs.')
    print(json.dumps({'verification':'pass','portable_files':len(index['files']),
                      'protected_inputs':len(frozen['protected_inputs']),
                      'native_executed':False,'compiler_required':False},sort_keys=True))


if __name__=='__main__':
    main()
