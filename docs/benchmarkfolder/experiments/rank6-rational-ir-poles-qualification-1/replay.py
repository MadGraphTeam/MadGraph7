"""Portable closed-evidence replay: package reads and exact Python algebra only.

Historical absolute paths are dictionary keys/provenance, never opened here.
No controller, compiler, loader, reducer, subprocess or ctypes API is imported.
"""
import argparse,gzip,hashlib,importlib.util,json,posixpath,re,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
PINS={'scientific':'4f496ebf3071e8d405e1e2e72b4bcff3ab7b210ef2338740cdbe314b0cfcb2b0','validation':'73f311c041c176fe80c00b53f51935d8283521672717247ee73d26c855ee0640','run':'1d1642c77d698afd8f0a4f8fa3c0db361733fa6ca6d9b1c1e1e44cbdd8be30cf','oracle':'9f866cc001a47cda221d6be4183ed135cb61af2ad6b48536012bce00b6e7587d'}
CLOSED={'failed-build':'e1d442dbc86c04606a5aba1ee5f83614b00c64d6ac9ef0a07558277fc5f2ad77','validation':'9c6d48c1db7e725db8a7579d9d6dc1ea623c5f22f4603232ebf3540042f4715c','run':'72d946f7cb97ad5f7a707724560255863bbda422812d972fd0a91c5d6c68e998'}
EXTRA='/nix/store/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib/libquadmath.so.0.0.0'
EXTRA_SHA='78d384074a271a26fd8bd466aedb4f285e56547daf596aa898c3bfff4bcaa6b5'
def require(x,m):
    if not x:raise ValueError(m)
def identity(b):return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def safe(rel):
    p=Path(rel);require(not p.is_absolute() and '..' not in p.parts,'Unsafe retained path');return HERE/p

def retained(meta):
    b=safe(meta['stored']).read_bytes();require(identity(b)==meta['stored_identity'],'Stored text/gzip differs')
    require(meta['encoding'] in ('identity','gzip'),'Unknown retention encoding')
    b=gzip.decompress(b) if meta['encoding']=='gzip' else b
    require(identity(b)=={k:meta[k] for k in ('bytes','sha256')},'Decoded retained bytes differ')
    b.decode('utf8');require(b'\x00' not in b,'Nontext retained payload');return b

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
class MemoryFile:
    def __init__(self,name,data):self.name=name;self.data=data
    def read_text(self,*unused,**kwargs):return self.data.decode('utf8')
    def __str__(self):return 'sealed-RUN/'+self.name
class MemoryRun:
    def __init__(self,data):self.data=data
    def __truediv__(self,name):return MemoryFile(str(name),self.data[str(name)])
    def __str__(self):return 'sealed-RUN'

def source_check(retention):
    roots={}
    for label,pin in PINS.items():
        info=retention['sources'][label];root=safe(info['root']);raw=(root/'source-index.json').read_bytes()
        require(identity(raw)['sha256']==pin==info['index_sha256'],'Source seal differs '+label)
        index=json.loads(raw)
        actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        require(actual==set(index['files'])|{'source-index.json'},'Source payload inventory differs')
        for rel,w in index['files'].items():
            p=Path(rel);require(not p.is_absolute() and '..' not in p.parts,'Unsafe ancestor path')
            require(identity((root/p).read_bytes())==w,'Ancestor source bytes differ')
        roots[label]=root
    require((roots['scientific']/'oracle.py').read_bytes()==(roots['oracle']/'oracle.py').read_bytes(),'Fraction oracle is not source9f')
    require((roots['scientific']/'future-triangle-fixture.json').read_bytes()==(roots['oracle']/'future-triangle-fixture.json').read_bytes(),'Triangle math fixture is not source9f')
    return roots

def replay(expected_index=None,authenticate=True):
    if authenticate:
        raw=(HERE/'index.json').read_bytes()
        if expected_index:require(identity(raw)['sha256']==expected_index,'Unexpected public capsule index')
        index=json.loads(raw);require(index['schema']=='rank6-rational-ir-poles-closed-capsule-v1','Capsule schema')
        actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
        require(actual==set(index['files'])|{'index.json'},'Capsule inventory differs')
        for rel,w in index['files'].items():
            p=safe(rel);require(p.is_file() and not p.is_symlink() and identity(p.read_bytes())==w,'Capsule payload differs '+rel)
    retention=read(HERE/'retention.json');roots=source_check(retention)
    datasets={group:{rel:retained(meta) for rel,meta in info['files'].items()} for group,info in retention['datasets'].items()}
    context={rel:retained(meta) for rel,meta in retention['context'].items()}
    audit={rel:retained(meta) for rel,meta in retention['independent_audit'].items()}
    get=lambda group,rel:json.loads(datasets[group][rel])
    for group,pin in CLOSED.items():require(identity(datasets[group]['terminal.json'])['sha256']==pin,'Closed stage seal differs')
    sci_pins=read(roots['scientific']/'input-pins.json');vpins=read(roots['validation']/'input-pins.json');rpins=read(roots['run']/'input-pins.json')
    failed=get('failed-build','terminal.json');require(failed['status']=='failed' and failed['returncode']==1 and not failed['cleanup']['survivors'],'Original failed BUILD changed')
    require('result.json' not in datasets['failed-build'],'Original failed BUILD gained a success result')
    require(b'Expected runtime provider was not prebound' in datasets['failed-build']['console.log'],'Failed guard cause differs')
    require(len(datasets['failed-build'])==170 and len(retention['datasets']['failed-build']['excluded_binaries'])==2,'Failed BUILD retention differs')
    for rel,w in vpins['failed_build_files'].items():
        if rel in datasets['failed-build']:require(identity(datasets['failed-build'][rel])==w,'Original failed BUILD text differs')
        else:require(retention['datasets']['failed-build']['excluded_binaries'][rel]==w,'Excluded binary identity differs')
    labels={rel.removeprefix('commands/').removesuffix('.terminal.json') for rel in datasets['failed-build'] if rel.startswith('commands/') and rel.endswith('.terminal.json')}
    require(labels=={'compiler-version','driver-compile','link'}|{'prelink-query-'+str(i).zfill(3) for i in range(51)},'Original54 tool coverage differs')
    for label in labels:
        cmd=get('failed-build','commands/'+label+'.command.json');term=get('failed-build','commands/'+label+'.terminal.json')
        require(term['returncode']==0 and cmd['argv']==term['argv'],'Original tool failure/argv differs')
        require(identity(datasets['failed-build']['commands/'+label+'.stdout.txt'])['sha256']==term['stdout_sha256'],'Original stdout seal differs')
    for i,label in enumerate(('driver-compile','link')):require(get('failed-build','commands/'+label+'.command.json')['argv']==sci_pins['build_commands'][i],'Compiled driver/link recipe differs')
    amended=get('validation','amended-prebound.json');old=get('failed-build','prelink.json')
    require(amended['record_created_after_failed_build'] and amended['original_helper_bound_unchanged'],'Amendment provenance lost')
    require(identity(datasets['failed-build']['prelink.json'])['sha256']==amended['original_helper_prelink_sha256'],'Original helper map differs')
    require(EXTRA not in old['bound'] and amended['added_bindings']=={EXTRA:EXTRA_SHA} and amended['bound']=={**old['bound'],EXTRA:EXTRA_SHA},'Amendment is not the single scoped provider')
    require(sci_pins['files'][EXTRA]==EXTRA_SHA,'Extra provider was not original-source protected')
    validation=get('validation','result.json');vterminal=get('validation','terminal.json')
    require(vterminal['status']=='passed' and vterminal['returncode']==0 and not vterminal['cleanup']['survivors'],'Validation not closed')
    require(identity(datasets['validation']['result.json'])['sha256']==vterminal['result_sha256'],'Validation result seal')
    require(validation['runtime']['files']==sci_pins['expected_runtime'] and validation['compiler_calls']==validation['linker_calls']==validation['native_tensor_calls']==validation['numerical_rows_executed']==0,'Validation scope/runtime differs')
    validation_labels={rel.removeprefix('commands/').removesuffix('.terminal.json') for rel in datasets['validation'] if rel.startswith('commands/') and rel.endswith('.terminal.json')}
    require(validation_labels=={'actual-pt-interp-runtime','symbols'},'Validation command set differs')
    for label in validation_labels:require(get('validation','commands/'+label+'.terminal.json')['returncode']==0,'Validation command failed')
    names=re.findall(r'^[0-9a-fA-F]+\s+[A-Za-z]\s+(\S+)$',datasets['validation']['commands/symbols.stdout.txt'].decode(),re.M)
    for name in ('cap6_tirust_tensor_diagnostic_','cap6_mp_tirust_tensor_diagnostic_','__wrap_ml_tirust_evaluate_dp','__wrap_ml_tirust_evaluate_mp','__mtirust_MOD_tensor_mp'):require(names.count(name)==1,'Compiled diagnostic/module/observer identity differs')
    for path,w in retention['excluded_binaries'].items():
        ident={k:w[k] for k in ('bytes','sha256')};require(vpins['protected_files'][path]==rpins['protected_files'][path]==ident,'Compiled binary receipt differs')
        if w['relative']=='check':require(ident['sha256']==validation['executable_sha256'],'Compiled ELF hash differs')
        if w['relative']=='driver.o':require(ident['sha256']==validation['driver_object_sha256'],'Driver object hash differs')
    # Historical materialization/build joins authenticate copied EW source only.
    materialized=json.loads(context['materialized-index.json']);historical_build=json.loads(context['build1/build.json'])
    for rel,data in context.items():
        suffix='/docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/'+rel
        pinned=[h for path,h in sci_pins['files'].items() if path.endswith(suffix)]
        if pinned:require(len(pinned)==1 and pinned[0]==identity(data)['sha256'],'Historical context does not join SOURCE1 guard')
    require(historical_build['complete'] and historical_build['sources_and_dependencies_unchanged'],'Historical build receipt differs')
    require(historical_build['build_plan_sha256']==identity(context['build-plan.json'])['sha256'],'Historical build plan seal differs')
    for rel,data in context.items():
        if rel.startswith('materialized/ew/'):require(materialized['files'][rel]==identity(data)['sha256'],'Historical CAP6 source/mapping differs')
    for name in ('cap_tirust.o','cap_helpers.o','cap_polynomial.o','observer.o'):
        candidates=[h for p,h in sci_pins['old_link_inputs'].items() if p.endswith('/ew/'+name)]
        require(candidates==[historical_build['artifacts']['ew/'+name]['sha256']],'Historical source/object identity join differs')
    result=get('run','result.json');terminal=get('run','terminal.json');analysis=get('run','analysis.json')
    require(terminal['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors'],'Numerical RUN not closed')
    require(identity(datasets['run']['result.json'])['sha256']==terminal['result_sha256'] and identity(datasets['run']['analysis.json'])['sha256']==result['analysis_sha256'],'RUN result/analysis seal differs')
    require(result['source_index_sha256']==PINS['run'] and result['reviewed_validation_sha256']==CLOSED['validation'] and result['returncode']==0,'RUN owner/validation identity differs')
    require(result['runtime']['files']==validation['runtime']['files'] and result['runtime']['files']==result['mapping']['mapped_files_sha256_after_exit'],'Recorded runtime identities differ')
    mapping=result['mapping'];require(mapping['selected_library']==sci_pins['selected_library'],'Native library selection differs')
    require(mapping['expected_runtime_sha256']==result['runtime']['files'] and set(mapping['mapped_elf_paths'])==set(result['runtime']['files'])|{mapping['executable']},'Recorded ELF namespace differs')
    maps=datasets['run']['maps.snapshots/complete.txt'];require(maps.decode()==mapping['maps_text'] and identity(maps)['sha256']==mapping['maps_sha256'],'Actual mapping snapshot seal differs')
    paths={line.split(None,5)[5] for line in maps.decode().splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
    require(paths==set(mapping['mapped_paths']) and not any(p.endswith(' (deleted)') for p in paths),'Raw mapped path namespace differs')
    # Only the sealed scientific parser/oracle is imported, never a controller.
    sys.path.insert(0,str(roots['scientific']))
    oracle=load_module('oracle',roots['scientific']/'oracle.py');lp=load_module('legacy_parser',roots['scientific']/'legacy_parser.py')
    scientific=load_module('retained_rank6_ir_analysis',roots['scientific']/'analyze.py')
    normal_path=scientific.Path;scientific.Path=lambda p:p if isinstance(p,MemoryRun) else normal_path(p)
    replayed=scientific.analyze(MemoryRun(datasets['run']))
    require(json.loads(json.dumps(replayed))==analysis,'Raw protocol oracle replay differs from retained analysis')
    require(analysis['status']=='passed' and analysis['cases']==24 and not analysis['open_items'],'Numerical oracle qualification is not passing')
    require(len(analysis['full_partial'])==12 and all(x['status']=='passed' and x['all_pole_words_equal'] for x in analysis['full_partial']),'Full/partial exact pole equality differs')
    schedule=read(roots['scientific']/'schedule.json');lines=datasets['run']['records.txt'].decode().splitlines()
    class Records:
        def read_text(self):return '\n'.join(lines[:1]+lines[3:])+'\n'
        def __str__(self):return 'sealed-RUN/records.txt'
    from decimal import localcontext
    with localcontext() as c:
        c.prec=100
        captures,cases=lp.parse_records(Records(),'ew',schedule)
        frames=lp.parse_observer(MemoryFile('observer.txt',datasets['run']['observer.txt']),[r['action'] for r in cases])
    require(not captures and len(cases)==24 and Counter(f['kind'] for f in frames.values())=={1:12,2:12},'Actual native precision coverage differs')
    require(all(f['present']==1 and f['status']==f['cache']==f['check']==0 for f in frames.values()),'Actual native requests/refusals differ')
    counts=Counter(line.split()[0] for line in lines if line.split());require(counts['V']==15120 and counts['CASE']==counts['STATE']==counts['END']==24,'Direct raw row coverage differs')
    join=load_module('retained_native_host_join',HERE/'native_host_join.py').verify(cases,frames,context['materialized/ew/cap_tirust.f'].decode(),oracle.ieee_fraction)
    receipt=json.loads(audit['receipt.json']);require(identity(audit['receipt.json'])['sha256']=='bc771c770aaf3e0280727fd4e8ffd4930ae0a923788b229768f6f180b1095171','Independent audit receipt differs')
    require(receipt['status']=='pass' and receipt['raw_replay_exact'] and receipt['forbidden_api_attempts']==receipt['audit_native_executions']==receipt['audit_binary_reads']==0,'Independent blocked-API audit scope differs')
    require(receipt['independent_direct_counts']==dict(counts) and receipt['rational_oracle_passes']==24 and receipt['refusals']==receipt['open_items']==0,'Independent raw audit does not join')
    require(set(receipt['source_indices'].values())=={PINS[k] for k in ('scientific','validation','run')},'Independent source audit seal set differs')
    for historical,want in receipt['pinned_closures'].items():
        joined=False
        for group,info in retention['datasets'].items():
            prefix=info['original_root']+'/'
            if historical.startswith(prefix):
                require(identity(datasets[group][historical[len(prefix):]])==want,'Independent closed-stage data seal differs');joined=True;break
        require(joined,'Independent closure has no retained dataset')
    # The earlier massive-bubble UV corroboration remains a separate
    # retrospective result with its original inherited-IEEE limitation.
    earlier=load_module('retained_rank6_earlier_UV_analysis',roots['oracle']/'analyze.py').analyze()
    require(earlier==read(roots['oracle']/'retrospective-analysis.json') and earlier['all_coefficient_oracle_gates_pass'],'Earlier exact UV source9f replay differs')
    return {'schema':'portable-rank6-rational-ir-poles-replay-v1','verified':True,'cases':24,'components_per_case':210,'actual_DP_rows':12,'actual_QP_rows':12,
      'rational_oracle_passes':24,'full_partial_pairs_with_exact_pole_words':12,'retrospective_source9f_UV_replayed':True,'retrospective_IEEE_failure_unchanged':True,'native_to_host_transport':join,'initial_IEEE':analysis['initial_IEEE'],'model_IEEE':analysis['model_IEEE'],
      'retained_runtime_providers':len(result['runtime']['files']),'original_failed_BUILD_preserved':True,'validation_separate_and_preserved':True,'independent_blocked_API_audit_joined':True,
      'new_native_executions':0,'compiler_calls':0,'loader_queries':0,'live_binary_reads':0,'physical_amplitude_qualified':False,'amplitude_rescue_qualified':False,
      'physical_stress_qualified':False,'trap_safety_qualified':False,'cache_enabled':False,'performance_acceptance':False,'runtime_identity_scope':'Retained actual listing/maps/hash receipts only; no fresh provider hash or binary execution.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index-sha256');a=p.parse_args()
    result=replay(a.index_sha256)
    require(result==read(HERE/'summary.json'),'Derived portable summary differs')
    print(json.dumps(result,indent=2,sort_keys=True))
