#!/usr/bin/env python3
"""Pack/verify/materialize large evidence without changing frozen analyzers."""
import sys
sys.dont_write_bytecode = True
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
INDEX=HERE/'compressed-artifacts.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
PATHS=('run-replay/raw.jsonl','run-convergence/raw.jsonl',
       'replay-analysis.json','convergence-analysis.json',
       'supplement/run-screening/raw.jsonl','screening-analysis.json')


def decompressed_record(path):
    digest=hashlib.sha256();size=0
    with gzip.open(path,'rb') as stream:
        while data:=stream.read(1<<20):digest.update(data);size+=len(data)
    return {'bytes':size,'sha256':digest.hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('pack','verify','materialize'))
    parser.add_argument('--correctness-only',action='store_true',
        help='Archive the completed numerical gates before the separately held screen')
    args=parser.parse_args()
    index_path=HERE/'compressed-correctness-artifacts.json' if args.correctness_only else INDEX
    paths=PATHS[:4] if args.correctness_only else PATHS
    if args.action=='pack':
        assert not index_path.exists()
        stages=('run-replay','run-convergence') if args.correctness_only else ('run-replay','run-convergence','supplement/run-screening')
        for stage in stages:
            result=json.loads((HERE/stage/'result.json').read_text())
            assert 'returncode' in result,'Never archive a live or unfinished attempt'
        entries={}
        for name in paths:
            plain=HERE/name
            if not plain.exists():continue
            assert plain.is_file() and not plain.is_symlink()
            compressed=plain.with_name(plain.name+'.gz')
            if not compressed.exists():
                with compressed.open('xb') as raw:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=9,mtime=0) as stream:
                        with plain.open('rb') as source:shutil.copyfileobj(source,stream,1<<20)
            original={'bytes':plain.stat().st_size,'sha256':sha(plain)}
            assert decompressed_record(compressed)==original
            entries[name]={'uncompressed':original,'compressed':{
                'path':str(compressed.relative_to(HERE)),'bytes':compressed.stat().st_size,'sha256':sha(compressed)}}
        index={'complete':True,'packer_sha256':sha(Path(__file__)),
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'completed-correctness-only' if args.correctness_only else 'correctness-and-completed-screen',
            'codec':'gzip level9, mtime0, empty filename; original bytes unchanged',
            'plain_files_are_materialized_working_data_not_git_payload':True,'files':entries}
        index_path.write_text(json.dumps(index,indent=2)+'\n')
    else:index=json.loads(index_path.read_text())
    assert index['complete']
    for name,entry in index['files'].items():
        compressed=HERE/entry['compressed']['path']
        assert sha(compressed)==entry['compressed']['sha256']
        assert compressed.stat().st_size==entry['compressed']['bytes']
        assert decompressed_record(compressed)==entry['uncompressed']
        plain=HERE/name
        if args.action=='materialize' and not plain.exists():
            temporary=plain.with_name('.'+plain.name+'.materializing')
            assert not temporary.exists()
            try:
                with gzip.open(compressed,'rb') as source,temporary.open('xb') as target:
                    shutil.copyfileobj(source,target,1<<20)
                assert sha(temporary)==entry['uncompressed']['sha256']
                assert temporary.stat().st_size==entry['uncompressed']['bytes']
                temporary.rename(plain)
            finally:
                if temporary.exists():temporary.unlink()
        if plain.exists():
            assert plain.is_file() and not plain.is_symlink()
            assert sha(plain)==entry['uncompressed']['sha256'] and plain.stat().st_size==entry['uncompressed']['bytes']
    print(json.dumps({'complete':True,'action':args.action,'artifacts':len(index['files']),
        'engine_calls':0,'index_path':str(index_path.relative_to(HERE)),
        'index_sha256':sha(index_path)}))


if __name__=='__main__':main()
