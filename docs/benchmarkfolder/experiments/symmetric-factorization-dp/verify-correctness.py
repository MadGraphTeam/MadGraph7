#!/usr/bin/env python3
"""Read-only checksum verification of the frozen DP correctness scope."""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recompute',action='store_true',help='also replay exact stored-input oracle; no floating-point program runs')
    args=parser.parse_args()
    seal=json.loads((HERE/'correctness-seal.json').read_text())
    assert seal['schema']=='dp-factor-correctness-seal-v1'
    assert seal['complete'] and seal['file_count']==len(seal['files'])
    for name,record in seal['files'].items():
        relative=Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path=HERE/relative
        assert path.is_file() and not path.is_symlink(),name
        data=path.read_bytes()
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
    if args.recompute:
        subprocess.run([sys.executable,'-B',str(HERE/'oracle/verify.py'),'--recompute'],check=True)
    print(json.dumps(dict(passed=True,bound_files=seal['file_count'],
                         bound_bytes=sum(r['bytes'] for r in seal['files'].values()),
                         exact_recomputed=args.recompute)))


if __name__=='__main__':
    main()
