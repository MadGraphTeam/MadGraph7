#!/usr/bin/env python3
"""Read-only source/proposal verification; does not run controller.main."""
from pathlib import Path
import ast
import json
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def main():
    for name in ('controller.py','supervise.py','verify_source.py'):
        ast.parse((HERE/name).read_text(),filename=name)
    from controller import preflight,require,sha
    proposal,pins,_,_,_=preflight()
    require(not proposal['native_authorized'] and not proposal['native_executed'],'Proposal execution scope changed.')
    require(proposal['runtime']['actual_process_maps_required'],'Missing actual runtime requirement.')
    entries=json.loads((HERE/'source-index.json').read_text())['files']
    print(json.dumps({'verification':'pass','source_files':len(entries),
                      'source_index_sha256':sha(HERE/'source-index.json'),
                      'protected_inputs':len(pins),'compiler_executed':False,'native_executed':False},sort_keys=True))


if __name__=='__main__':
    main()
