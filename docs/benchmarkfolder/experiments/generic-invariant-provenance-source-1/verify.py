"""Authenticate and replay this source-only packet without a MadGraph checkout."""
import sys
sys.dont_write_bytecode = True
import argparse
import contextlib
import copy
import dataclasses
import difflib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import types
import unittest

HERE=Path(__file__).resolve().parent
SOURCE_INDEX_SHA='72c5f2729b0b176bc9ed71779dbf03d25189ac0ba5e176f7eb76cdec86de05f7'


def require(ok,reason):
    if not ok:raise ValueError(reason)


def pin(data):return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def read_regular(root,name):
    relative=Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative)==name,'unsafe payload path')
    path=root
    for part in relative.parts:
        path=path/part
        require(not path.is_symlink(),'symlink payload '+name)
    require(path.is_file(),'nonregular payload '+name)
    return path.read_bytes()


@contextlib.contextmanager
def read_only_replay(root):
    """Only package/stdlib reads; no writes, process creation or foreign loads.

    Hooks are permanent Python objects but inert outside this lexical scope.
    They do not replace or repair any source algorithm or test assertion.
    """
    state={'active':True}
    allowed=(root.resolve(),Path(sys.base_prefix).resolve(),Path(sys.prefix).resolve())
    writes={'os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.chown',
            'os.link','os.symlink','os.truncate','os.utime'}
    launches={'subprocess.Popen','os.system','os.exec','os.posix_spawn','ctypes.dlopen'}
    def path_ok(value):
        if isinstance(value,int):return
        require(isinstance(value,(str,bytes,os.PathLike)),'opaque file path')
        p=Path(os.fsdecode(value)).resolve()
        require(any(p.is_relative_to(a) for a in allowed),'read outside package/stdlib: '+str(p))
    def audit(event,args):
        if not state['active']:return
        require(event not in writes,'write forbidden: '+event)
        require(event not in launches,'process/foreign load forbidden: '+event)
        if event=='open':
            path,mode,flags=args
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            require(not writing,'write forbidden during replay')
            path_ok(path)
        elif event in ('os.listdir','os.scandir'):path_ok(args[0])
    sys.addaudithook(audit)
    try:yield
    finally:state['active']=False


def load_definition(name,source,path):
    spec=importlib.util.spec_from_loader(name,loader=None)
    module=importlib.util.module_from_spec(spec)
    module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def run_protocol(source_root,source_bytes):
    """Run the exact sealed tests with package facades, not a workspace import."""
    names=('madgraph','madgraph.loop','madgraph.iolibs')
    saved={n:m for n,m in sys.modules.items() if n=='madgraph' or n.startswith('madgraph.')}
    old_path=list(sys.path)
    added=[]
    try:
        for name in list(saved):sys.modules.pop(name,None)
        for name in names:
            module=types.ModuleType(name);module.__path__=[]
            sys.modules[name]=module;added.append(name)
            if '.' in name:
                parent,child=name.rsplit('.',1);setattr(sys.modules[parent],child,module)
        for suffix in ('source','provenance','render','finalize'):
            name='madgraph.loop.invariant_'+suffix
            path='src/madgraph/loop/invariant_'+suffix+'.py'
            module=load_definition(name,source_bytes[path],source_root/path)
            setattr(sys.modules['madgraph.loop'],'invariant_'+suffix,module)
            added.append(name)
        module=load_definition('_retained_invariant_protocol_tests',source_bytes['test_protocol.py'],source_root/'test_protocol.py')
        added.append(module.__name__)
        suite=unittest.defaultTestLoader.loadTestsFromModule(module)
        stream=io.StringIO()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        require(result.testsRun==17 and result.wasSuccessful(),stream.getvalue())
        return dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                    skipped=len(result.skipped),source_unchanged=True)
    finally:
        sys.path[:]=old_path
        for name in added:sys.modules.pop(name,None)
        sys.modules.update(saved)


def verify(root=HERE,expected_index_sha256=None,repo_root=None):
    root=Path(root).resolve()
    raw=read_regular(root,'evidence-index.json')
    if expected_index_sha256:require(pin(raw)['sha256']==expected_index_sha256,'public index hash')
    index=json.loads(raw)
    require(index['schema']=='generic-invariant-provenance-public-source-v1','public schema')
    payload={name:read_regular(root,name) for name in index['files']}
    for name,data in payload.items():require(pin(data)==index['files'][name],'public payload '+name)
    source_raw=payload['source/source-index.json']
    require(pin(source_raw)['sha256']==SOURCE_INDEX_SHA,'original source index')
    source_index=json.loads(source_raw)
    source_bytes={name:payload['source/'+name] for name in source_index['files']}
    require(len(source_bytes)==71,'original payload count')
    require({n for n in payload if n.startswith('source/')}==
            {'source/'+n for n in source_bytes}|{'source/source-index.json'},'exact original payload selection')
    for name,data in source_bytes.items():require(pin(data)==source_index['files'][name],'original payload '+name)
    live=None
    if repo_root is not None:
        # Explicit opt-in only, no inferred checkout or historical root paths.
        repo=Path(repo_root).resolve()
        for name,expected in source_index['production_sources'].items():
            require(pin(read_regular(repo,name))==expected,'explicit repository pin '+name)
        live=dict(passed=True,files=len(source_index['production_sources']),repo_root=str(repo))
    with read_only_replay(root):
        name='_retained_invariant_source_verifier'
        try:
            module=load_definition(name,source_bytes['verify.py'],root/'source/verify.py')
            replay=module.verify(root=root/'source',expected_index_sha256=SOURCE_INDEX_SHA,check_live=False)
        finally:sys.modules.pop(name,None)
        protocol=run_protocol(root/'source',source_bytes)
    # Recheck the fixed source data gates; no future author can upgrade this
    # packet's scope merely by changing a report field.
    for name,data in source_bytes.items():
        if name.startswith('generation-evidence/') and name.endswith('/invariant_bounds_data.inc'):
            require(b'source_environment_supported=.false.' in data,'scientific environment admission')
    return dict(passed=True,public_index_sha256=pin(raw)['sha256'],
        original_source_index_sha256=SOURCE_INDEX_SHA,public_payloads=len(payload),
        exact_original_payloads=71,patch_template_table_replay=replay,
        rerun_portable_protocol_tests=protocol,historical_mg_generation_tests=7,
        mg_generation_rerun=False,full_model_closure_retained=False,
        scientific_environment_admission=False,scientific_dispatch='legacy pass-through',
        compiler_loader_reducer_amplitude_executions=0,explicit_live_pins=live)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-sha',help='Expected public evidence-index.json SHA256')
    parser.add_argument('--repo-root',type=Path,help='Optional explicit repository for read-only production-pin checks; does not generate or compile')
    args=parser.parse_args()
    print(json.dumps(verify(expected_index_sha256=args.index_sha,repo_root=args.repo_root),sort_keys=True,indent=2))
