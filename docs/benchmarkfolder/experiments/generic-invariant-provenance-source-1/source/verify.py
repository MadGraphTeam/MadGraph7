"""Read-only source/table replay. No generation, subprocess, loader or native work."""
from pathlib import Path
import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import sys

HERE=Path(__file__).resolve().parent

def require(ok,reason):
    if not ok:raise ValueError(reason)


def pin(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def verify(root=HERE,expected_index_sha256=None,check_live=False):
    root=Path(root).resolve()
    raw=(root/'source-index.json').read_bytes()
    if expected_index_sha256:require(pin(raw)['sha256']==expected_index_sha256,'index hash')
    index=json.loads(raw);cache={}
    for name,item in index['files'].items():
        rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'unsafe path')
        current=root
        for part in rel.parts:
            current=current/part;require(not current.is_symlink(),'symlink payload')
        data=current.read_bytes();require(pin(data)==item,'payload '+name);cache[name]=data
    # Exact patch reconstruction from bound base/src, not an external patch tool.
    patch=[]
    for name in sorted(n for n in cache if n.startswith('src/')):
        path=name[4:];base=cache.get('base/'+path)
        patch.extend(difflib.unified_diff(base.decode().splitlines(True) if base is not None else [],
            cache[name].decode().splitlines(True),fromfile='a/'+path if base is not None else '/dev/null',tofile='b/'+path))
    require(''.join(patch).encode()==cache['01-metadata-caller.patch'],'patch replay')
    old=cache['qualified/caller_bounds.f90'].decode();new=cache['src/madgraph/iolibs/template_files/loop_optimized/invariant_bounds.f90'].decode()
    diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='qualified/caller_bounds.f90',tofile='proposal/invariant_bounds.f90'))
    require(diff.encode()==cache['factory-formula.patch'],'isolated factory delta')
    # Import only the small pure authored definitions with private module names.
    import types
    package=types.ModuleType('madgraph');package.__path__=[]
    loop=types.ModuleType('madgraph.loop');loop.__path__=[]
    saved={n:sys.modules.get(n) for n in ('madgraph','madgraph.loop','madgraph.loop.invariant_source')}
    sys.modules['madgraph']=package;sys.modules['madgraph.loop']=loop
    loaded=[]
    try:
        def module(name,path):
            spec=importlib.util.spec_from_loader(name,loader=None)
            m=importlib.util.module_from_spec(spec);sys.modules[name]=m;loaded.append(name)
            exec(compile(cache[path],path,'exec'),m.__dict__);return m
        module('madgraph.loop.invariant_source','src/madgraph/loop/invariant_source.py')
        prov=module('private_source_prov','src/madgraph/loop/invariant_provenance.py')
        render=module('private_source_render','src/madgraph/loop/invariant_render.py')
        for n,fn in [('loop_matrix_standalone.inc',render.main_template),('mp_compute_loop_coefs.inc',render.mp_template),('helas_calls_split.inc',render.batch_template)]:
            before=cache['base/madgraph/iolibs/template_files/loop_optimized/'+n].decode()
            require(fn(before).encode()==cache['rendered-templates/'+n],'template replay '+n)
        receipt=json.loads(cache['generation-results.json']);routes=0;mode_rows=0
        require(len(receipt)==7,'actual generation case count')
        for row in receipt:
            case=row['case'];folder='generation-evidence/'+case+'/'
            data=json.loads(cache[folder+'invariant_provenance.json'])
            require(prov.emit_tables(data).encode()==cache[folder+'invariant_bounds_data.inc'],'table replay '+case)
            require(len(data['routes'])==row['routes'],'route count '+case)
            require(not row['numerical_execution'] and not row['compiler_execution'],'source scope')
            require(data['environment_contract']==row['environment_contract'] and not data['environment_contract']['complete'],'model-conversion qualification boundary')
            routes+=len(data['routes'])
            for r in data['routes']:
                for m in prov.MODES:
                    require(r['modes'][str(m)]['supported'],'expected source geometry')
                    mode_rows+=1
            for n in ('invariant_provenance.json','invariant_bounds_data.inc','invariant_bounds.f90','invariant_environment.c','loop_CT_calls_1.f'):
                require(pin(cache[folder+n])==row['generated']['SubProcesses/P0/'+n],'generated receipt '+case+'/'+n)
            require((row['subprocess_prefix']+'kinematic_environment_token').encode() in cache[folder+'invariant_environment.c'],'namespace '+case)
        require(routes==29 and mode_rows==116,'complete generated route total')
    finally:
        for name in loaded:sys.modules.pop(name,None)
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    for name,count in [('protocol-tests.log',17),('generation-tests.log',7)]:
        text=cache[name].decode();require(re.search(r'Ran '+str(count)+r' tests in [0-9.]+s\n\nOK\s*$',text),'final tests '+name)
    if check_live:
        repo=HERE.parents[1]
        for name,expected in index['production_sources'].items():require(pin((repo/name).read_bytes())==expected,'live source '+name)
    return dict(passed=True,index_sha256=pin(raw)['sha256'],payloads=len(cache),
                pure_tests=17,actual_generation_tests=7,actual_routes=29,mode_rows=116,
                metadata_only=True,scientific_dispatch='legacy pass-through',
                environment_path_qualified=False,native_or_compiler_executions=0,
                current_source_pins_checked=bool(check_live))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index-sha');parser.add_argument('--live',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(expected_index_sha256=args.index_sha,check_live=args.live),sort_keys=True,indent=2))
