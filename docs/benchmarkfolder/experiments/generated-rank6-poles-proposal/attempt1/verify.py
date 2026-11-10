#!/usr/bin/env python3
"""Read-only verify of review package. --sources also checks bound build inputs."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((HERE/'review-index.json').read_text())
    for path,expected in manifest['files'].items():
        if sha(HERE/path)!=expected:raise SystemExit('Changed package file: '+path)
    proposal=json.loads((HERE.parent/'index.json').read_text())
    for path,expected in proposal['files'].items():
        if sha(HERE.parent/path)!=expected:raise SystemExit('Changed proposal: '+path)
    if args.sources:
        build=json.loads((HERE/'build-plan.json').read_text())
        for path,expected in build['dependencies'].items():
            if sha(ROOT/path)!=expected:raise SystemExit('Changed build dependency: '+path)
        for path,expected in build['resolved_library_files'].items():
            if sha(Path(path))!=expected:raise SystemExit('Changed resolved library: '+path)
        preparation=json.loads((HERE/'runtime-preparation.json').read_text())
        for records in preparation['models'].values():
            for item in records:
                if sha(Path(item['path']))!=item['sha256']:raise SystemExit('Changed private runtime input: '+item['path'])
        inventory=json.loads((HERE.parent/'inventory.json').read_text())
        bindings=inventory['shared_bindings']+[b for p in inventory['processes'] for b in p['bindings']]
        for item in bindings:
            if sha(ROOT/item['path'])!=item['sha256']:raise SystemExit('Changed source binding: '+item['path'])
        for tool,record in manifest['tools'].items():
            if sha(Path(record['path']))!=record['sha256']:raise SystemExit('Changed pinned tool: '+tool)
    print('PASS: reviewed source-only package'+(' and bound build inputs' if args.sources else '')+'; no compiler/native execution')


if __name__=='__main__':main()
