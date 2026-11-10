#!/usr/bin/env python3
"""Portable source inventory/reproduction; optional read-only staged hashes."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
from instrument import instrument

HERE=Path(__file__).resolve().parent


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--working-root',type=Path);args=parser.parse_args()
    index=json.loads((HERE/'evidence-index.json').read_text())
    for name,h in index['files'].items():assert digest(HERE/name)==h,name
    parent=HERE.parent;prior=json.loads((parent/'prepared.json').read_text())
    assert digest(parent/'evidence-index.json')==index['parent_index_sha256']
    parent_index=json.loads((parent/'evidence-index.json').read_text())
    for name,h in parent_index['files'].items():assert digest(parent/name)==h,name
    ref=prior['reference_sources'][prior['origin']+'/SubProcesses/P0_epem_ttxa/mp_compute_loop_coefs.f']['snapshot']
    for arm in ('original','candidate'):
        inputs={name:(parent/'sources'/arm/name).read_text() for name in prior['changed_generated_sources']}
        inputs['mp_compute_loop_coefs.f']=(parent/ref).read_text()
        outputs=instrument(inputs,arm=='candidate')
        for name,text in outputs.items():
            assert text==(HERE/'sources'/arm/name).read_text(),(arm,name)
            patch=''.join(difflib.unified_diff(inputs[name].splitlines(keepends=True),text.splitlines(keepends=True),fromfile='frozen/'+name,tofile='observed/'+name))
            assert patch==(HERE/'sources'/arm/(name+'.patch')).read_text()
    prepared=json.loads((HERE/'prepared.json').read_text())
    if args.working_root:
        for name,h in prepared['artifacts'].items():assert digest(args.working_root/name)==h,name
    print(json.dumps({'state':'source_only_verified','indexed_files':len(index['files']),
                      'matched_source_reproduction':True,'parent_package_unchanged':True,
                      'staged_hashes_checked':bool(args.working_root),'compiler_invocations':0,'native_reducer_calls':0},indent=2))


if __name__=='__main__':main()
