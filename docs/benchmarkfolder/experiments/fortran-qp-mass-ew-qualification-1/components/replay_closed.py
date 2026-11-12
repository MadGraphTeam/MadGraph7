"""File-only EW closed-chain draft replay; no numerical RUN or final capsule seal."""
import argparse
import builtins
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import types

sys.dont_write_bytecode=True

def require(value,message):
    if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()


@contextmanager
def blocked_apis(memory):
    """Only preloaded logical text reads are available during semantic replay."""
    saved=[];attempts=[]
    def deny(*args,**kwargs):
        attempts.append(repr(args[:1]));raise RuntimeError('Forbidden API during portable replay')
    def memory_bytes(path,*args,**kwargs):
        if str(path) not in memory:return deny(str(path))
        return memory[str(path)]
    def memory_text(path,encoding=None,errors=None,*args,**kwargs):
        return memory_bytes(path).decode(encoding or 'utf-8',errors or 'strict')
    def patch(obj,name,value=deny):
        if hasattr(obj,name):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    try:
        patch(Path,'read_bytes',memory_bytes);patch(Path,'read_text',memory_text)
        for name in ('open','stat','lstat','resolve','exists','is_file','is_dir','is_symlink','iterdir','glob','rglob',
                     'write_bytes','write_text','mkdir','unlink','rename','replace','touch','chmod','symlink_to','hardlink_to'):
            patch(Path,name)
        for name in ('open',):patch(builtins,name)
        for name in ('open','open_code'):patch(io,name)
        for name in ('open','stat','lstat','access','listdir','scandir','readlink','system','popen','posix_spawn','posix_spawnp',
                     'execv','execve','execvp','execvpe','spawnv','spawnve','spawnvp','spawnvpe','fork','forkpty',
                     'mkdir','makedirs','unlink','remove','rename','replace','rmdir','chmod','chown','truncate','utime',
                     'link','symlink','kill','killpg'):
            patch(os,name)
        for name in ('Popen','run','call','check_call','check_output'):patch(subprocess,name)
        for name in ('CDLL','PyDLL','_dlopen'):patch(ctypes,name)
        if '_ctypes' in sys.modules:patch(sys.modules['_ctypes'],'dlopen')
        for name in ('socket','create_connection'):patch(socket,name)
        for name in ('signal','setitimer','alarm','pthread_kill'):patch(signal,name)
        yield attempts
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)


def load_source(name,raw,path):
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__);return module



GEN_SOURCE='6fb69ba7da6fe07e702047c211c04329be2003ea577f6bf1574eeb1dd210f917'
HELPER='6d9407f27372ed45e5dc9b9a2ab672f638d273ea85ab1cc17ba85eb70924025e'
BUILD_SOURCE='f245a10696fea62dad8a7bb116675ea43e09d5e3376686d3433bbc3f47d24f2f'
BUILD=('f7cac8b007f41e09964b95c9a0fb6172cb0f4eaa249ed3192188f9fa1af46b22','60a99e609b0aae3576c31f3f49cf6a250f21dfa1e7dbc670e95f4c49ac72d603')
GEN={'original':('026f54e26f163c6cacec3262ae9b6a27273e59447e9302038a12979be18761d9','f0c85c95311e1f73c0782a5f9e2ecec6d67a7861d704d27b4d7d4c0334750555'),
 'candidate':('99085edcf65ba41297eb4f393efcb3070a6c10adff573dcfd65d59a0b2e4959d','2ed7b8ddb064c85163224ce0f3b41e17bca796ec2077ad6a13da398706a9f2ad')}

def validate_future_numerical_closure(value):
    """Shape only: this dormant gate never opens a supplied path or admits work."""
    required={'run_source_sha256','terminal_sha256','result_sha256','review_sha256','root_admission_sha256','review_path','root_admission_path'}
    require(isinstance(value,dict) and set(value)==required,'Exact later root closure and independent review required')
    for name in required-{'review_path','root_admission_path'}:require(isinstance(value[name],str) and re.fullmatch('[0-9a-f]{64}',value[name]),'Complete numerical closure hash required')
    for name in ('review_path','root_admission_path'):
        path=Path(value[name]);require(not path.is_absolute() and '..' not in path.parts and str(path).startswith('.tirust-build/ew-qp-mass-'),'Later closure path must remain in the EW task namespace')
    return {'closure_shape_valid':True,'files_read':0,'RUN_collected':False,'RUN_admitted':False,'numerical_outcome_generated':False,'capsule_sealed':False}

def verify(root,expected_state_sha256=None):
    import ast
    from collections import Counter
    import difflib
    import posixpath
    from pathlib import PurePosixPath
    root=Path(root).resolve();raw=(root/'collection-state.json').read_bytes()
    if expected_state_sha256:require(sha(raw)==expected_state_sha256,'Closed collection state changed')
    state=json.loads(raw);require(state['schema']=='EW-QP-mass-closed-chain-draft-v1' and not state['numerical_RUN_collected'] and state['numerical_outcome'] is None and not state['final_capsule_sealed'],'Draft must not invent a numerical outcome or final seal')
    require(not state['public_writes'] and state['new_native_executions']==0,'Retention action scope changed')
    actual=set()
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Portable collection symlink refused')
        if path.is_file():actual.add(str(path.relative_to(root)))
    require(actual=={p['path'] for p in state['logical_files'].values()}|{'collection-state.json'},'Exact closed collection inventory changed')
    blobs={}
    for name,pin in state['logical_files'].items():
        path=Path(name);physical=Path(pin['path']);require(not path.is_absolute() and '..' not in path.parts and not physical.is_absolute() and '..' not in physical.parts,'Unsafe contained text path')
        value=(root/physical).read_bytes()
        if pin['compression']=='gzip':require(value[:3]==b'\x1f\x8b\x08' and value[4:8]==b'\0'*4,'Nondeterministic gzip');value=gzip.decompress(value)
        else:require(pin['compression']=='none','Unknown text compression')
        require(sha(value)==pin['raw_sha256'] and len(value)==pin['raw_bytes'],'Changed closed logical text: '+name);value.decode('utf-8')
        require(not value.startswith(b'\x7fELF'),'Native text payload refused');blobs[name]=value
    def data(name,wanted=None):
        require(name in blobs,'Missing closed text: '+name);value=blobs[name]
        if wanted:require(sha(value)==wanted,'Frozen chain changed: '+name)
        return value
    def record(name,wanted=None):return json.loads(data(name,wanted))
    expected={'gen-source':GEN_SOURCE,'helper-source1':'4c7ea15770fa256346e99b7e5a38d83c4924a993374f3082ab58a02b5790366c',
        'helper-source2':HELPER,'build-source1':'97283fce7f418824f2a13c512cbaa386f9dc5ef9f77bd8b6d5cfb015171c5471','build-source2':BUILD_SOURCE}
    for prefix,wanted in expected.items():
        index=record(prefix+'/source-index.json',wanted)
        require({n[len(prefix)+1:] for n in blobs if n.startswith(prefix+'/')}==set(index['files'])|{'source-index.json'},'Exact frozen source namespace changed')
        for name,pin in index['files'].items():require(len(data(prefix+'/'+name,pin['sha256']))==pin['bytes'],'Frozen source size changed')
    memory={str(root/name):value for name,value in blobs.items()}
    def definitions(name,globals_,excluded=(),constants=()):
        tree=ast.parse(data(name));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name not in excluded or
            isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)]
        namespace=dict(globals_);exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),namespace);return types.SimpleNamespace(**namespace)
    with blocked_apis(memory) as attempts:
        fixed=definitions('helper-source2/fixed_source.py',{'re':re})
        scope=definitions('helper-source2/scope.py',{'Path':PurePosixPath,'hashlib':hashlib,'posixpath':posixpath,'re':re})
        helper=definitions('helper-source2/review.py',{'Counter':Counter,'difflib':difflib,'hashlib':hashlib,'json':json,'Path':Path,'re':re,'fixed':fixed,'scope_api':scope},excluded=('read_sources',),constants=('SUFFIXES',))
        bpins=record('build-source2/input-pins.json');generated={}
        for role,(terminal_sha,result_sha) in GEN.items():
            term=record('gen/'+role+'/terminal.json',terminal_sha);result=record('gen/'+role+'/result.json',result_sha)
            require(term['status']=='passed' and term['returncode']==0 and not term['cleanup']['survivors'] and term['result_sha256']==result_sha,'GEN terminal closure changed')
            require(result['status']=='generated_pending_source_review' and result['source_index_sha256']==GEN_SOURCE and result['actual_CLI_invocations']==1 and
                result['amplitude_evaluations']==0 and not result['BUILD_admitted'] and not result['RUN_admitted'],'Original GEN source-review qualifier changed')
            generated[role]={name:data('generated/'+role+'/'+name,pin['sha256']) for name,pin in result['source_inventory'].items()}
            require(len(generated[role])==427 and result['stock_Model_DHELAS_build_objects']==344,'Actual GEN source/support counts changed')
            for name,pin in bpins['roles'][role]['data_files'].items():require(len(data('generated-data/'+role+'/'+name,pin['sha256']))==pin['bytes'],'Generated data/card identity changed')
        correspondence=helper.analyze(generated['original'],generated['candidate'],'SubProcesses/P0_epem_ttxa')
        correspondence=json.loads(json.dumps(correspondence,sort_keys=True))
        expected_report=record('reviews/ew-qp-mass-actual-source-correspondence-review-1/correspondence.json','64a5fa25c3387504de2f8b907feb8e1e12d806d9cd713f9ed21dd6f21f8d0356')
        require(correspondence==expected_report and correspondence['narrow_checks_passed'] and not correspondence['source_qualified'] and not correspondence['BUILD_admitted'] and not correspondence['semantic_equivalence_proven'],'Frozen narrow source correspondence replay changed')
        require(correspondence['actual_public_units']==25 and correspondence['actual_private_helpers']==9 and correspondence['verified_guarded_calls']==200 and
            sum(correspondence['verified_DP_MP_mass_pair_counts'].values())==626 and len(correspondence['raw_changed_files'])==6,'Actual source correspondence counts changed')
        actual_review=record('reviews/ew-qp-mass-actual-source-correspondence-review-1/receipt.json','5e9ba7d71757d08a58f3e9a8834e26cb8e51667163162e25f7d9c171892a1570')
        require(actual_review['raw_identical_bindings']==421 and actual_review['DP_calls_logical_label_tokens_exact']==200 and actual_review['DP_calls_raw_bytes_exact']==0 and not actual_review['source_qualified'],'DP logical versus raw-layout qualifier changed')
        data('reviews/ew-qp-mass-actual-source-correspondence-review-1/raw-delta-classification.json','83aaf4cee2b35c1e395cb2073d4f2608158bcc9877fbde9fe8ae32dd5cbafa11')
        # Raw-delta classification is authenticated independent evidence. This
        # helper replay does not pretend to regenerate its later report.
        blocked=record('reviews/ew-qp-mass-generated-source-review-independent-1/receipt.json','7f260554eaa94017a852e2af173803cef5e04a4173cba03a1de4eb7170e5f442')
        require(any(r.get('blocking') for r in blocked['findings']),'Helper1 blocked ancestry changed')
        record('reviews/ew-qp-mass-generated-source-review-independent-2/receipt.json','9f63c632639f32a6713c4a6b614367beab42a0d2772f698c4b532148f2fffda9')
        gate=record('closure/ew-qp-mass-root-reviewed-source-1.json','b8e6be67dc0b38da9b15a0df05c5ea1fb4bdf0a692970f6857c80881e71d18a9')
        require(gate['generated_text_accepted'] and not gate['source_qualified'],'Root text-only gate changed')
        terminal=record('build/terminal.json',BUILD[0]);built=record('build/result.json',BUILD[1])
        require(terminal['status']==built['status']=='passed' and terminal['returncode']==0 and terminal['result_sha256']==BUILD[1] and not terminal['cleanup']['survivors'] and built['source_index_sha256']==BUILD_SOURCE,'Closed BUILD source/result chain changed')
        require(built['source_review_admission_sha256']=='b8e6be67dc0b38da9b15a0df05c5ea1fb4bdf0a692970f6857c80881e71d18a9' and not built['RUN_admitted'] and built['native_amplitude_calls']==built['model_probes']==built['reducer_calls']==0 and not built['performance_acceptance'] and not built['candidate_promoted'],'BUILD-only scope changed')
        raw_inventory=record('closure/ew-qp-mass-build-raw-inventory-2.json','c1e46e69465817ec0e43b6df490eeac8ec5d0d7fd96548ea4b36baf71edec82a')
        require(len(raw_inventory['files'])==571,'Closed BUILD raw inventory count changed')
        for name,pin in raw_inventory['files'].items():require(len(data('build/'+name,pin['sha256']))==pin['bytes'],'Actual BUILD raw member changed')
        commands=[name for name in blobs if name.startswith('build/') and '/commands/' in name and name.endswith('.command.json')];require(len(commands)==168,'Actual BUILD command count changed')
        compiles=0;links=0
        for name in commands:
            stem=name[:-len('.command.json')];row=record(stem+'.terminal.json');argv=record(name)['argv']
            require(row['returncode']==0 and row['stdout_sha256']==sha(data(stem+'.stdout.txt')),'Actual BUILD command/stdout closure changed')
            compiles+=int('-c' in argv);links+=int(any(a.startswith('-Wl,-Map,') for a in argv) and '-c' not in argv)
        require(compiles==42 and links==2,'Actual fresh compile/link count changed')
        excluded=state['excluded_binaries'];expected_native=set()
        for role,row in built['variants'].items():
            require(len(row['sources'])==427 and len(row['stock_objects'])==344 and len(row['process_objects'])==21 and len(row['actual_link_loads'])==54 and len(row['runtime']['files'])==15 and
                len(row['MTIRUST_definitions'])==23 and len(row['private_definitions'])==(9 if role=='candidate' else 0) and row['native_main_calls']==0,'Closed BUILD symbol/source/provider counts changed')
            data('build/'+role+'/private-link.map',row['link_map_sha256']);pre=record('build/'+role+'/prelink.json',row['prelink_sha256'])
            require(all(pre['bound'].get(n)==h for n,h in {**row['actual_link_loads'],**row['runtime']['files']}.items()),'Prebound LOAD/provider metadata changed')
            require(excluded['BUILD/'+role+'/check']['sha256']==row['executable_sha256'],'Excluded process ELF changed')
            expected_native.add('BUILD/'+role+'/check')
            for name,pin in bpins['roles'][role]['stock_objects'].items():
                key='GEN/'+role+'/'+name;expected_native.add(key);require(excluded[key]['sha256']==pin['sha256'] and excluded[key]['bytes']==pin['bytes'],'GEN support identity metadata changed')
            for kind in ('process_objects','process_modules','stock_objects','stock_modules'):
                for name,pin in row[kind].items():
                    key='BUILD/'+role+'/'+name;expected_native.add(key);require(excluded[key]['sha256']==pin['sha256'] and excluded[key]['bytes']==pin['bytes'],'Excluded BUILD object/module identity changed')
            for name,wanted in row['actual_link_loads'].items():
                key='LOAD/'+role+'/'+name;expected_native.add(key);require(excluded[key]['sha256']==wanted,'Recorded LOAD native identity changed')
            for name,wanted in row['runtime']['files'].items():
                key='runtime/'+name;expected_native.add(key);require(excluded[key]['sha256']==wanted,'Recorded runtime provider identity changed')
        for name,pin in bpins['external_compile_inputs'].items():
            if Path(name).suffix in ('.h','.f90'):require(len(data('external-text/'+name.lstrip('/'),pin['sha256']))==pin['bytes'],'External compile text identity changed')
            else:
                key='external-compile/'+name;expected_native.add(key);require(excluded[key]['sha256']==pin['sha256'] and excluded[key]['bytes']==pin['bytes'],'External binary interface/module identity changed')
        require(set(excluded)==expected_native,'Exact excluded-native identity inventory changed')
        review=record('reviews/ew-qp-mass-build-independent-review-2/receipt.json','fa3e423b60d76d840efe66376ec47d00b8ffb73642192f3504eb3bfcbbefa9be')
        require(review['raw_inventory_files']==571 and review['command_records']==168 and review['command_counts']['compile']==42 and not review['findings'],'Independent BUILD review scope changed')
        for name,pin in bpins['support_files'].items():require(len(data('parents/'+name,pin['sha256']))==pin['bytes'],'Minimal pinned support subset changed')
        require(not attempts,'Forbidden contained replay API attempted')
    return {'closed_chain_replayed':True,'GEN_per_role':1,'generated_inputs_each':427,'identical_source_bindings':421,'explained_changed_source_bindings':6,
        'public_units_raw_exact':25,'private_helpers':9,'guarded_calls':200,'mass_actual_pairs':626,'DP_calls_logical_exact':200,'DP_calls_raw_exact':0,
        'BUILD_commands':168,'actual_compiles':42,'actual_links':2,'LOAD_bindings_each':54,'runtime_providers_each':15,'MTIRUST_symbols_each':23,
        'forbidden_API_attempts':0,'numerical_RUN_collected':False,'numerical_outcome':None,'final_capsule_sealed':False,'new_native_executions':0}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--draft',type=Path,required=True);parser.add_argument('--state-sha256');args=parser.parse_args()
    print(json.dumps(verify(args.draft,args.state_sha256),sort_keys=True,indent=2))
