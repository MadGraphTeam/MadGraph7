#!/usr/bin/env python3
"""Read-only source/proposal verification; does not run controller.main."""
from pathlib import Path
import ast
import hashlib
import json
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def main():
    entries=json.loads((HERE/'source-index.json').read_text())['files']
    paths=list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()}==set(entries)|{'source-index.json'}
    for name,entry in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        raw=(HERE/name).read_bytes()
        assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
    for name in ('controller.py','supervise.py','verify_source.py'):
        ast.parse((HERE/name).read_text(),filename=name)
    from controller import preflight,require,sha
    proposal,pins,_,_,_=preflight()
    require(not proposal['native_authorized'] and not proposal['native_executed'],'Proposal execution scope changed.')
    require(proposal['runtime']['actual_process_maps_required'],'Missing actual runtime requirement.')
    from check_snapshots import compute
    assert compute()==json.loads((HERE/'snapshot-checks.json').read_text())
    print(json.dumps({'verification':'pass','source_files':len(entries),
                      'source_index_sha256':sha(HERE/'source-index.json'),
                      'protected_inputs':len(pins),'compiler_executed':False,'native_executed':False},sort_keys=True))


if __name__=='__main__':
    main()
