#!/usr/bin/env python3
"""Reuse attempt4 executables; shorten only the alternate fixture number."""
import argparse
from decimal import Decimal
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

ROOT=Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
BASE=ROOT/'docs/benchmarkfolder/experiments/subprocess-cache-lifecycle'
PRIOR=BASE/'attempt4'
REPORT=BASE/'attempt5'
WORK=ROOT/'.tirust-build/subprocess-cache-lifecycle/attempt5'
INDEX=REPORT/'source-freeze.json'
ORIGINAL_TOKEN=b'9.2000000000000000e+01'
NEW_TOKEN=b'9.200000e+01'
GUARD=None


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True); stream.write('\n')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def previous():
    if digest(PRIOR/'source-freeze.json')!='3307827092d738f99c1b12e4eaa8796aaf9e0992896b4f65fb3d98a00e113eb7':
        raise RuntimeError('Attempt4 freeze changed')
    if digest(PRIOR/'resume.py')!='8c824f3acd8b0dd479a6f709d917e2f82896154fae3a952f65ceb8d6254ba0f5':
        raise RuntimeError('Attempt4 runner changed')
    module=load(PRIOR/'resume.py','attempt4_reused')
    module.GUARD=GUARD
    module.verified()
    return module


def numeric_prevalidation(text):
    """Check all numeric card tokens against the actual reader's width20."""
    numeric=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?\Z')
    tokens=[]
    for line_number,line in enumerate(text.splitlines(),1):
        for token in line.split('#',1)[0].split():
            if numeric.fullmatch(token):
                value=Decimal(token.lower().replace('d','e'))
                if len(token)>20 or not value.is_finite():
                    raise RuntimeError(f'Unsupported numeric token at line{line_number}: {token}')
                tokens.append({'line':line_number,'token':token,'width':len(token)})
    return {'reader_character_width':20,'numeric_tokens':tokens,
            'maximum_width':max((item['width'] for item in tokens),default=0)}


def freeze():
    module=previous()
    old=(PRIOR/'runs/card-smoke/alternate/param_card.dat').read_bytes()
    if old.count(ORIGINAL_TOKEN)!=1: raise RuntimeError('Alternate fixture is not the retained failed card')
    corrected=old.replace(ORIGINAL_TOKEN,NEW_TOKEN)
    if Decimal(ORIGINAL_TOKEN.decode())!=Decimal(NEW_TOKEN.decode()) or Decimal(NEW_TOKEN.decode())!=92:
        raise RuntimeError('Fixture numeric value changed')
    nominal=(PRIOR/'runs/card-smoke/nominal/param_card.dat').read_bytes()
    ident=(PRIOR/'runs/card-smoke/nominal/ident_card.dat').read_bytes()
    validation={name:numeric_prevalidation(raw.decode()) for name,raw in
        [('nominal/param_card.dat',nominal),('alternate/param_card.dat',corrected),
         ('nominal/ident_card.dat',ident),('alternate/ident_card.dat',ident)]}
    WORK.mkdir(exist_ok=False)
    with (REPORT/'alternate-param-card.dat').open('xb') as stream: stream.write(corrected)
    patch=''.join(difflib.unified_diff(old.decode().splitlines(True),corrected.decode().splitlines(True),
        fromfile='../attempt4/runs/card-smoke/alternate/param_card.dat',tofile='alternate-param-card.dat'))
    with (REPORT/'alternate-card.patch').open('x') as stream: stream.write(patch)
    save(REPORT/'numeric-prevalidation.json',validation)
    shutil.copyfile(PRIOR/'build.json',REPORT/'build.json')
    old_files=[PRIOR/'source-freeze.json',PRIOR/'build.json',PRIOR/'build-evidence.json',
        PRIOR/'native-failure.json',PRIOR/'smoke-failure.json',PRIOR/'execution/native-card-smoke.json',
        PRIOR/'execution/native-card-smoke.txt',PRIOR/'runs/card-smoke/inputs.json',
        PRIOR/'runs/card-smoke/outputs.json',PRIOR/'runs/card-smoke/records.txt',
        PRIOR/'runs/card-smoke/trace.txt',PRIOR/'runs/card-smoke/param.log',
        PRIOR/'runs/card-smoke/alternate/param_card.dat',PRIOR/'runs/card-smoke/nominal/param_card.dat',
        PRIOR/'runs/card-smoke/nominal/ident_card.dat']
    save(INDEX,{'schema_version':1,'source_sha256':digest(REPORT/'resume.py'),
        'amendment_sha256':digest(REPORT/'AMENDMENT.md'),
        'prior_inputs':{str(path):digest(path) for path in old_files},
        'fixture_files':{name:digest(REPORT/name) for name in ('alternate-param-card.dat',
            'alternate-card.patch','numeric-prevalidation.json','build.json')},
        'unchanged_binary_build_sha256':digest(PRIOR/'build.json'),'compile_or_link_allowed':False,
        'cpu':10,'smoke_cap_seconds':30,'worker_cap_seconds':180,'native_execution_authorized':False})


def adapted():
    state=json.loads(INDEX.read_text())
    if digest(REPORT/'resume.py')!=state['source_sha256'] or digest(REPORT/'AMENDMENT.md')!=state['amendment_sha256']:
        raise RuntimeError('Attempt5 source changed')
    module=previous()
    original_verify=module.verified
    original_layout=module.input_layout
    def verify():
        if (digest(REPORT/'resume.py')!=state['source_sha256']
                or digest(REPORT/'AMENDMENT.md')!=state['amendment_sha256']):
            raise RuntimeError('Attempt5 source changed during qualification')
        module.REPORT=PRIOR; module.WORK=ROOT/'.tirust-build/subprocess-cache-lifecycle/attempt4'
        try: old,q=original_verify()
        finally: module.REPORT=REPORT; module.WORK=WORK
        q.REPORT=REPORT
        q.verify_hashes(state['prior_inputs'])
        q.verify_hashes({str(REPORT/name):value for name,value in state['fixture_files'].items()})
        return old,q
    def layout(directory,cache,q):
        original_layout(directory,cache,q)
        card=directory/'alternate/param_card.dat'
        if card.read_bytes().count(ORIGINAL_TOKEN)!=1: raise RuntimeError('Unexpected alternate source')
        corrected=card.read_bytes().replace(ORIGINAL_TOKEN,NEW_TOKEN)
        if corrected!=(REPORT/'alternate-param-card.dat').read_bytes(): raise RuntimeError('Fixture correction changed')
        card.write_bytes(corrected)
        for folder in ('nominal','alternate'):
            for name in ('param_card.dat','ident_card.dat'):
                numeric_prevalidation((directory/folder/name).read_text())
    module.REPORT=REPORT; module.WORK=WORK
    module.verified=verify; module.input_layout=layout
    module.verified()
    return module


def main():
    global GUARD
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','smoke','run'))
    args=parser.parse_args()
    if Path.cwd()!=ROOT: parser.error('Run from repository root only')
    # This unchanged helper is bound by the prior source freeze.
    previous()
    guard=load(PRIOR/'process_guard.py','attempt5_prior_guard')
    with guard.ProcessGuard() as GUARD:
        try:
            if args.stage=='freeze': freeze()
            else: adapted().run(args.stage)
        except BaseException as error:
            save(REPORT/(args.stage+'-failure.json'),{'stage':args.stage,'exception':repr(error),
                'note':'Stopped; no automatic retry or continuation.'})
            raise


if __name__=='__main__': main()
