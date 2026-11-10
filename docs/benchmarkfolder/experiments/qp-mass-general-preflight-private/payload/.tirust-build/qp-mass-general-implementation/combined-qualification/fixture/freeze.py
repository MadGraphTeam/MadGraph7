#!/usr/bin/env python3
"""Freeze/check only fixture source files; exclude future native artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--check',action='store_true');args=cli.parse_args()
    sources=[p for p in HERE.rglob('*') if p.is_file() and
             p.relative_to(HERE).parts[0] not in ('runs','build') and p.name!='index.json']
    for p in sources:
        assert not p.is_symlink() and p.suffix not in ('.pyc','.o','.mod','.so','.a'),str(p)
    record=dict(schema='qp-mass-caller-fixture-source-index-v1',native_compile_performed=False,
                native_execution_performed=False,excluded_future_artifact_trees=['runs','build'],
                files={str(p.relative_to(HERE)):sha(p) for p in sorted(sources)})
    output=HERE/'index.json';text=json.dumps(record,indent=2,sort_keys=True)+'\n'
    if args.check:assert output.read_text()==text,'Source index mismatch'
    else:
        with output.open('x') as stream:stream.write(text)
    print(json.dumps(dict(files=len(sources),source_only=True,index_sha256=sha(output),check=args.check)))


if __name__=='__main__':main()
