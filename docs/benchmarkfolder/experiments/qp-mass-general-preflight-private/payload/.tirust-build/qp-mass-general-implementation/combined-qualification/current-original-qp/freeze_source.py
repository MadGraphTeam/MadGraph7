#!/usr/bin/env python3
"""Freeze/verify the additive source plan; never link or execute."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parents[2]
INDEX=HERE/'evidence-index.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--check',action='store_true');args=cli.parse_args()
    spec=json.loads((HERE/'prepared.json').read_text())
    for name,digest in spec['inputs'].items():assert sha(ROOT/name)==digest,name
    for name,digest in spec['system_files'].items():assert sha(name)==digest,name
    if args.check:index=json.loads(INDEX.read_text())
    else:
        assert not INDEX.exists() and not (HERE/'build.json').exists() and not (HERE/'run.json').exists()
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
        index={'schema':'additive-current-original-qp-source-index-v1','state':'prepared-not-linked-or-executed',
            'source_files':files,'source_tests':{'scientific_parser':13,'controller_prerequisites':5},
            'parent_source_index_sha256':sha(HERE.parent/'evidence-index.json'),
            'parent_build_evidence_sha256':sha(HERE.parent/'build-evidence/index.json'),
            'bounds':{'cpu':8,'jobs':1,'link_jobs':1,'link_seconds':30,'new_fortran_compiles':0,
                      'loader_jobs':2,'loader_seconds':10,'native_jobs':2,'native_seconds_each':120,
                      'parser_seconds':30,'lock_admission_seconds':7200},
            'scope':'Same-current-runtime original versus mass-guarded fixed4 EW transport only',
            'not_qualified':['checked QP rescue','complete threshold amplitude','all backends','performance']}
        INDEX.write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
    for name,digest in index['source_files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps({'index_sha256':sha(INDEX),'manifest_sha256':sha(HERE/'prepared.json'),
        'files':len(index['source_files']),'bound_inputs':len(spec['inputs']),'system_files':len(spec['system_files']),
        'compiler_launched':False,'native_launched':False}))

if __name__=='__main__':main()
