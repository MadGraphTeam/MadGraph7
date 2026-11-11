"""Replay the source2 capsule and explicit source1 ancestor without a checkout."""
import sys
sys.dont_write_bytecode=True
import argparse,difflib,hashlib,importlib.util,io,json,os,re,types,unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE_SHA='0e93995f751c7a0f3dfae1e39748714afe5a194ed6dbc1c9e873ca9d89117685'
ANCESTOR_SHA='0d2bbdda8c09fc51dc2cffad348794f3a184cdd273bc1bc602ce616475f7848f'
ANCESTOR_SOURCE_SHA='72c5f2729b0b176bc9ed71779dbf03d25189ac0ba5e176f7eb76cdec86de05f7'
REVIEW_SHA='d0718cf538b5991f7889e06166c3b334d858dd6525c71a59e9bd0854763f2765'

def require(ok,why):
    if not ok:raise ValueError(why)
def pin(data):return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def read(root,name):
    rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts and str(rel)==name,'unsafe payload path')
    p=root
    for part in rel.parts:p=p/part;require(not p.is_symlink(),'symlink payload '+name)
    require(p.is_file(),'nonregular payload '+name);return p.read_bytes()
def payload(root,expected=None):
    raw=read(root,'evidence-index.json')
    if expected:require(pin(raw)['sha256']==expected,'public index hash')
    idx=json.loads(raw);data={n:read(root,n) for n in idx['files']}
    for n,b in data.items():require(pin(b)==idx['files'][n],'payload mismatch '+n)
    return raw,idx,data

def load(name,code,path):
    spec=importlib.util.spec_from_loader(name,loader=None);module=importlib.util.module_from_spec(spec)
    module.__file__=str(path);sys.modules[name]=module;exec(compile(code,str(path),'exec'),module.__dict__);return module


def pure_tests(root,data,helper):
    saved={n:m for n,m in sys.modules.items() if n=='madgraph' or n.startswith('madgraph.') or n=='invariant_environment_source'}
    old_path=list(sys.path);added=[]
    try:
        for n in saved:sys.modules.pop(n,None)
        for n in ('madgraph','madgraph.loop'):
            m=types.ModuleType(n);m.__path__=[];sys.modules[n]=m;added.append(n)
        sys.modules['madgraph'].loop=sys.modules['madgraph.loop']
        for suffix in ('environment_source','window'):
            n='madgraph.loop.invariant_'+suffix;path='src/madgraph/loop/invariant_'+suffix+'.py'
            m=load(n,data[path],root/'source'/path);setattr(sys.modules['madgraph.loop'],'invariant_'+suffix,m);added.append(n)
            if suffix=='environment_source':sys.modules['invariant_environment_source']=m;added.append('invariant_environment_source')
        suite=unittest.TestSuite()
        for f in ('test_environment.py','test_window.py'):
            n='_portable_source2_'+f[:-3];m=load(n,data[f],root/'source'/f);added.append(n)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))
        stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        require(result.testsRun==40 and result.wasSuccessful(),stream.getvalue())
        return {'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'source_unchanged':True}
    finally:
        sys.path[:]=old_path
        for n in added:sys.modules.pop(n,None)
        sys.modules.update(saved)


def record_checks(source,ancestor):
    inputs=json.loads(source['input-bindings.json']);require(len(inputs)==470,'historical input count')
    closed=json.loads(source['closed-results.json']);require(set(closed)=={'gmu','loop_sm'},'closed cases')
    prefixes={
        'gmu':'.tirust-build/kinematic-provenance-current-executor-1/work/exports/cms/',
        'loop_sm':'.benchmark-work/tirust/campaigns/entry-latch-cache-guard-1/outputs/candidate/qcd_n3_uux_zgg/'}
    counts={'gmu':(238,141,18,32),'loop_sm':(164,165,16,30)}
    def closure(c,bindings):
        require(c['scientific_admission'] is False,'scientific admission in closure')
        require(type(c['reference_complete']) is bool and type(c['complete']) is bool,'typed closure status')
        require(c['reference_complete']==(not c['unknown']),'reference closure status')
        require(c['complete']==(c['reference_complete'] and not c['runtime_limitations']),'source effect status')
        for n,p in c['procedures'].items():
            require(p['file'] in bindings and type(p['line']) is int and p['line']>0,'procedure/source coordinate')
            require(p['kind'] in ('subroutine','function'),'procedure kind')
            for callee in p['edges']:
                require(callee in c['procedures'] or any(u.get('procedure')==callee for u in c['unknown']),'unrecorded reference')
    def window(w,bindings):
        require(w['scientific_admission'] is False and type(w['complete']) is bool,'window admission')
        closure(w['prefix'],bindings);closure(w['suffix'],bindings)
        last=w['last_producer'];require(last['file'] in bindings and type(last['line']) is int and last['line']>0,'last producer binding')
        if w['complete']:
            require(w['prefix']['complete'] and w['suffix']['reference_complete'] and not w['later_writes'] and w['reason'] is None,'positive window requirements')
    summaries={}
    for case,r in closed.items():
        expected=counts[case];bindings=r['source_files'];require(len(bindings)==expected[0],'closed source selection')
        for n,m in bindings.items():require(inputs.get(prefixes[case]+n)==m,'closed source/historical input join '+n)
        closure(r['producer'],bindings)
        require(r['producer']['reference_complete'] and not r['producer']['complete'],'whole-producer limitation')
        require(len(r['producer']['procedures'])==expected[1],'closed reference count')
        require(set(r['momentum_windows'])=={'64','128'},'closed precision set')
        for w in r['momentum_windows'].values():window(w,bindings);require(w['complete'],'closed source window')
        for c in list(r['model_functions'].values())+list(r['initialization'].values()):closure(c,bindings)
        supported=sum(c['complete'] for c in r['model_functions'].values())
        require((supported,len(r['model_functions']))==expected[2:],'model function scope')
        require(all(not c['complete'] for c in r['initialization'].values()),'initialization scope')
        summaries[case]={'selected_source_files':expected[0],'reference_graph_procedures':expected[1],
            'complete_whole_producer':False,'supported_momentum_windows':2,'model_functions_supported':supported,
            'model_functions_total':expected[3],'initialization_qualified':False}
    generated=json.loads(source['generation-results.json']);old={x['case']:x for x in json.loads(ancestor['generation-results.json'])}
    require(len(generated)==7 and {x['case'] for x in generated}==set(old),'generation case set')
    fields=('case','has_born','group_loops','routes','external_slot_permuted','loopnums','source_mode_supported','subprocess_prefix','template_capabilities')
    supported=0;unknown=0
    for r in generated:
        require(all(r[k]==old[r['case']][k] for k in fields),'parent graph/routing summary changed')
        require(r['compiler_execution'] is False and r['numerical_execution'] is False,'generation scope')
        closure(r['environment_contract'],r['generated'])
        require(set(r['momentum_environment_contract'])=={'64','128'},'generated precision set')
        for w in r['momentum_environment_contract'].values():
            window(w,r['generated']);require(w['prefix']['complete'],'generated construction prefix')
            if r['has_born']:require(w['complete'],'Born window scope');supported+=1
            else:
                require(not w['complete'] and any('CALL arity' in u['reason'] for u in w['suffix']['unknown']),'loop-induced argument-contract unknown');unknown+=1
    require((supported,unknown)==(10,4),'generated scope totals')
    return {'closed':summaries,'historical_generation_cases':7,'generated_born_windows':10,
            'generated_loop_induced_unknown_windows':4,'ancestor_graph_summaries_identical':True,
            'closed_source_reparse':False,'mg_generation_rerun':False,'historical_input_bytes_opened':False}


def verify(root=HERE,expected_index_sha256=None,ancestor_root=None):
    root=Path(root).resolve();ancestor_root=Path(ancestor_root).resolve() if ancestor_root else root.parent/'generic-invariant-provenance-source-1'
    raw,idx,p=payload(root,expected_index_sha256)
    require(idx['schema']=='generic-invariant-provenance-public-source-v2','public schema')
    require(idx['ancestor']=={'directory':'generic-invariant-provenance-source-1','evidence_index_sha256':ANCESTOR_SHA,'source_index_sha256':ANCESTOR_SOURCE_SHA},'explicit ancestor')
    source_raw=p['source/source-index.json'];require(pin(source_raw)['sha256']==SOURCE_SHA,'source2 original index')
    source_index=json.loads(source_raw);data={n:p['source/'+n] for n in source_index['files']}
    require(len(data)==21,'source2 payload count')
    require({n for n in p if n.startswith('source/')}=={'source/'+n for n in data}|{'source/source-index.json'},'exact source2 selection')
    for n,b in data.items():require(pin(b)==source_index['files'][n],'original source2 payload '+n)
    require(source_index['source_environment_table_admission'] is False and source_index['production_changes'] is False and source_index['compiler_or_scientific_native_executions']==0,'original scope')
    ar,ai,ap=payload(ancestor_root,ANCESTOR_SHA)
    ancestor={n[7:]:b for n,b in ap.items() if n.startswith('source/') and n!='source/source-index.json'}
    require(pin(ap['source/source-index.json'])['sha256']==ANCESTOR_SOURCE_SHA,'original ancestor')
    helper=load('_source2_portable_ancestor_verifier',ap['verify.py'],ancestor_root/'verify.py')
    try:
        ancestor_result=helper.verify(root=ancestor_root,expected_index_sha256=ANCESTOR_SHA)
        with helper.read_only_replay(root):
            patch=[]
            for name in ('invariant_environment_source.py','invariant_window.py','invariant_finalize.py'):
                path='src/madgraph/loop/'+name;previous=ancestor.get(path)
                patch.extend(difflib.unified_diff(previous.decode().splitlines(True) if previous is not None else [],data[path].decode().splitlines(True),fromfile='a/madgraph/loop/'+name if previous is not None else '/dev/null',tofile='b/madgraph/loop/'+name))
            require(''.join(patch).encode()==data['source2-closure.patch'],'exact source2 patch derivation')
            require(b'source_environment_supported=.false.' in ancestor['src/madgraph/loop/invariant_provenance.py'],'ancestor scientific gate')
            require(b'from madgraph.loop.invariant_provenance import derive,emit_tables,routing_contract' in data['src/madgraph/loop/invariant_finalize.py'],'unchanged table producer binding')
            tests=pure_tests(root,data,helper);records=record_checks(data,ancestor)
            review_raw=p['root-review-receipt.json'];require(pin(review_raw)['sha256']==REVIEW_SHA,'root pure-review receipt')
            review=json.loads(review_raw)
            require(review['source_index_sha256']==SOURCE_SHA and review['passed'] is True and review['pure_tests']['tests']==40 and review['scientific_admission'] is False and review['native_execution'] is False,'root receipt scope')
    finally:sys.modules.pop('_source2_portable_ancestor_verifier',None)
    return {'passed':True,'public_index_sha256':pin(raw)['sha256'],'original_source_index_sha256':SOURCE_SHA,
        'public_payloads':len(p),'exact_original_payloads':21,'explicit_ancestor_index_sha256':ANCESTOR_SHA,
        'ancestor_portable_replay_passed':ancestor_result['passed'],'ancestor_pure_tests':ancestor_result['rerun_portable_protocol_tests']['tests'],
        'patch_derivation_passed':True,'rerun_pure_tests':tests,'record_validation':records,
        'scientific_environment_admission':False,'scientific_dispatch':'legacy pass-through',
        'compiler_loader_reducer_amplitude_executions':0,'full_model_closure_retained':False,
        'root_review_receipt_scope':'retained pure verification result; historical live pins, no added source/native qualification'}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--index-sha');ap.add_argument('--ancestor-root',type=Path)
    a=ap.parse_args();print(json.dumps(verify(expected_index_sha256=a.index_sha,ancestor_root=a.ancestor_root),sort_keys=True,indent=2))
