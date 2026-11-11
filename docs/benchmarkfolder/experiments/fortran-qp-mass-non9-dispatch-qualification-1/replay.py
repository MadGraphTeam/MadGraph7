"""Portable file-only non9 unit replay; historical controllers are never run."""
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
SOURCE='4de119d2c61ab208f6f32b950920fb1ef992bb1a69506c852fe0b836c3ba2fd1'
OLD='e5ad27c3cd698f25c74b38711f98e9e90c2c12de7da259ef414aafc8dc725f44'
PARENT='893c51f75dedbb6bbb1a70f78bed84b00d97b958ac678c194dc28cbb3aaa6b0b'
BUILD='fb947cbd00843ab18c4971a3927b6d9fe66be02f78fe949c359794920ba7c5b5'
BUILD_RESULT='4ea6b651397709ad4821b09dfca4fca5927147593537557d8ed0e441033f2e10'
RUN='c0bd0cce2541e8ba09447d9b2d4f1d95a75b2aa92afbebd1ed9ed668d1a3daba'
RUN_RESULT='bdb454efa243c617b8ec5161f5b76558f8efdd69dacfd615acd7a4153896edfd'
RAW='39ec847eff89ca9e8780085702826357c6e47922846a4d667f5427866da5f109'
# These two exact historical spellings were canonicalized in the closed BUILD
# receipt and independently reviewed then. Replay performs no live resolution.
RUNTIME_ALIASES={
 '/nix/store/cb4kg08gc8kwa81rxclkn6qqc6hxwwp5-gfortran-15.2.0-lib/lib/libgfortran.so.5':
 '/nix/store/cb4kg08gc8kwa81rxclkn6qqc6hxwwp5-gfortran-15.2.0-lib/lib/libgfortran.so.5.0.0',
 '/nix/store/xc0ga87wdclrx54qjaryahkkmkmqi9qz-gcc-15.2.0-lib/lib/libgcc_s.so.1':
 '/nix/store/b7kx9bkjsma9wslr1cg0316m5jy450l8-gcc-15.2.0-libgcc/lib/libgcc_s.so.1'}
REVIEWS={'source1':'f77fda1eb2cd9daaa29bbe002a148ebb2c01a3bd8a532283cebad0bdec0fbfa8',
 'source2':'8e7da5e9776897169841cf2518206d641591ea9ddfed4d636dfd18aaf52dc374',
 'build':'51c0e7526ec07163889cefb211c23954d31d1c757d0507d7b879edecefa545b2',
 'run':'f41ac18474f94243fa1098f0e31f76cfd73a669bc41ee691194c646419017ad4'}


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


def verify(root,expected_index_sha256=None):
    root=Path(root).resolve();index_raw=(root/'index.json').read_bytes()
    if expected_index_sha256:require(sha(index_raw)==expected_index_sha256,'Portable index seal changed')
    index=json.loads(index_raw);require(index['schema']=='non9-unit-portable-qualification-v1','Wrong capsule schema')
    actual=set()
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Portable symlinks refused')
        if path.is_file():actual.add(str(path.relative_to(root)))
    require(actual==set(index['files'])|{'index.json'},'Portable exact inventory changed')
    physical={}
    for name,row in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe physical member')
        raw=(root/name).read_bytes();require(sha(raw)==row['sha256'] and len(raw)==row['bytes'],'Changed payload: '+name)
        require(not raw.startswith(b'\x7fELF'),'Native executable retained');physical[name]=raw
    blobs={}
    for name,row in index['logical_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe logical member')
        raw=physical[row['path']]
        if row['compression']=='gzip':
            require(raw[:3]==b'\x1f\x8b\x08' and raw[4:8]==b'\0'*4,'Non-deterministic gzip')
            raw=gzip.decompress(raw)
        else:require(row['compression']=='none','Unknown compression')
        require(sha(raw)==row['raw_sha256'] and len(raw)==row['raw_bytes'],'Changed logical text: '+name)
        raw.decode('utf-8');require(not raw.startswith(b'\x7fELF'),'Native logical member');blobs[name]=raw
    def data(name,wanted=None):
        require(name in blobs,'Missing logical member: '+name)
        raw=blobs[name]
        if wanted:require(sha(raw)==wanted,'Changed closed identity: '+name)
        return raw
    def record(name,wanted=None):return json.loads(data(name,wanted))
    source=record('source/source-index.json',SOURCE)
    require({n[7:] for n in blobs if n.startswith('source/')}==set(source['files'])|{'source-index.json'},'Exact source2 inventory changed')
    for name,row in source['files'].items():require(len(data('source/'+name,row['sha256']))==row['bytes'],'Source2 size changed')
    old=record('ancestor/source1/source-index.json',OLD)
    retained={n[len('ancestor/source1/'):] for n in blobs if n.startswith('ancestor/source1/')}
    require(retained=={'source-index.json','controller.py','build-spec.json','README.md','review-diff.md'},'Wrong minimal source1 subset')
    for name in retained-{'source-index.json'}:
        require(len(data('ancestor/source1/'+name,old['files'][name]['sha256']))==old['files'][name]['bytes'],'Source1 subset changed')
    parent=record('parent-subset/index.json',PARENT)
    require(set(n[len('parent-subset/'):] for n in blobs if n.startswith('parent-subset/'))==set(index['parent_members'])|{'index.json'},'Parent subset namespace changed')
    for name in index['parent_members']:
        row=parent['logical_files'][name]
        require(len(data('parent-subset/'+name,row['raw_sha256']))==row['raw_bytes'],'Parent membership changed')
    parent_source=record('parent-subset/source-2/source-index.json','3c0f53a6e5526535b6fae98b0f055466b429a9a9cfc863fb22fb999a5c8a0815')
    for name in ('controller.py','source-preview/candidate/CT_interface.f','source-preview/original/CT_interface.f','source-preview/candidate/loop_CT_calls_1.f'):
        data('parent-subset/source-2/'+name,parent_source['files'][name]['sha256'])
    preparation=load_source('non9_portable_preparation',data('source/prepare.py'),root/'source/prepare.py')
    parser=load_source('non9_portable_parser',data('source/parser.py'),root/'source/parser.py')
    memory={str(root/name):raw for name,raw in blobs.items()}
    with blocked_apis(memory) as attempts:
        spec=record('source/cases.json');buildspec=record('source/build-spec.json')
        candidate=data('parent-subset/source-2/source-preview/candidate/CT_interface.f').decode()
        original=data('parent-subset/source-2/source-preview/original/CT_interface.f').decode()
        helpers=[]
        for family in spec['families']:
            name='ML5_0_LOOP_'+family['name'];public=preparation.routine(candidate,name)
            private=preparation.routine(candidate,name+'_QPMASS')
            require(public==preparation.routine(original,name),'Public original helper changed')
            require(sha(public.encode())==spec['helper_origins'][name]['public_sha256'] and
                    sha(private.encode())==spec['helper_origins'][name]['private_sha256'],'Helper derivation hash changed')
            helpers.extend([public.replace('ML5_0_','N9UNIT_'),private.replace('ML5_0_','N9UNIT_')])
        require('\n'.join(helpers).encode()==data('source/helpers.f'),'Fourteen helper derivation changed')
        chooser=preparation.routine(data('parent-subset/original-cli/SubProcesses/P0_uux_zgg/TIR_interface.f').decode(),'ML5_0_CHOOSE_LOOPLIB')
        chooser=chooser.replace('ML5_0_CHOOSE_LOOPLIB','N9UNIT_REAL_CHOOSE_LOOPLIB').replace('ML5_0_','N9UNIT_')
        require(chooser.encode()==data('source/chooser.f'),'Actual chooser derivation changed')
        template=data('derivation/MadLoopCommons.inc')
        dependency=buildspec['dependencies']['Template/loop_material/StandAlone/SubProcesses/MadLoopCommons.inc']
        require(sha(template)==dependency['sha256'] and len(template)==dependency['bytes'],'Detector source dependency changed')
        detectors='\n'.join(preparation.routine(template.decode(),name) for name in spec['actual_detectors'])
        require(detectors.encode()==data('source/detectors.f'),'Actual pure detector derivation changed')
        built=record('build/result.json',BUILD_RESULT);ran=record('run/result.json',RUN_RESULT)
        for phase,checksum,result in [('build',BUILD,built),('run',RUN,ran)]:
            terminal=record(phase+'/terminal.json',checksum)
            require(terminal['status']==result['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors'],'Unclosed phase')
            require(terminal['result_sha256']==sha(data(phase+'/result.json')) and result['source_index_sha256']==SOURCE,'Phase source/result binding changed')
            require(terminal['cap_seconds']==180 and terminal['cleanup_reserve_seconds']==60 and
                    terminal['original_end_monotonic']==terminal['deadline_monotonic'] and
                    abs(terminal['original_end_monotonic']-terminal['original_start_monotonic']-180)<1e-8,'Original total deadline changed')
            limits=record(phase+'/actual-limits.json')['actual']
            require(limits=={'address_space_bytes':[4294967296]*2,'file_bytes':[268435456]*2,'core_bytes':[0,0]},'Actual phase limits changed')
            admission=record(phase+'/admission.json')
            require(admission['source_index_sha256']==SOURCE and admission['cpu']==12 and not admission['cpu_reservation'],'Admission binding changed')
            for name,wanted in result['raw_files'].items():data(phase+'/'+name,wanted)
        require(ran['reviewed_build']==BUILD and record('run/admission.json')['reviewed']['build']==BUILD,'Separately reviewed BUILD changed')
        require(built['unit_executions']==built['actual_reducer_calls']==ran['actual_reducer_calls']==ran['physical_amplitudes']==0,'Native science scope inflated')
        commands=sorted(n for n in blobs if n.startswith('build/commands/') and n.endswith('.command.json'))
        require(len(commands)==30,'Actual BUILD command coverage changed')
        review_build=record('reviews/build.json',REVIEWS['build'])
        for name in commands:
            stem=name[:-len('.command.json')];terminal=record(stem+'.terminal.json');command=record(name)
            require(terminal['returncode']==0 and terminal['stdout_sha256']==sha(data(stem+'.stdout.txt')),'BUILD command/log failure')
            label=stem.split('/')[-1];joined=review_build['command_bindings'][label]
            require(command['argv']==joined['argv'] and sha(data(stem+'.terminal.json'))==joined['terminal_sha256'],'Independent command join changed')
        source_path=os.path.dirname(record('build/commands/compile-records.command.json')['argv'][-3])
        def expand(argv):return [s.replace('{source}',source_path).replace('{work}',built['output']) for s in argv]
        for row in buildspec['compile_commands']:require(record('build/commands/'+row['label']+'.command.json')['argv']==expand(row['argv']),'Compile argv differs from frozen policy')
        require(record('build/commands/link.command.json')['argv']==expand(buildspec['link_command']),'Link argv changed')
        require(buildspec['flags']==['-w','-fPIC','-ffixed-line-length-132'],'Compiler policy changed')
        data('build/unit-link.map',built['link_map_sha256']);pre=record('build/prelink.json',built['prelink_sha256'])
        require(pre['completed_before_link'] and len(built['actual_link_loads'])==23,'Recorded LOAD closure changed')
        require(all(pre['bound'].get(n)==h for n,h in {**built['actual_link_loads'],**built['runtime']['files']}.items()),'Recorded LOAD/provider not prebound')
        listing=data('build/commands/unit-actual-runtime.stdout.txt').decode()
        require(listing==built['runtime']['listing'],'Actual loader listing changed')
        provider_paths={built['runtime']['loader']}
        for line in listing.splitlines():
            line=line.strip()
            if not line or line.startswith('linux-vdso.so.1 '):continue
            match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line)
            require(match is not None,'Unknown recorded loader line')
            provider_paths.add(RUNTIME_ALIASES.get(match[1],match[1]))
        require(provider_paths==set(built['runtime']['files']) and len(provider_paths)==5 and
                built['runtime']['files']==review_build['runtime_providers'],'Five recorded providers changed')
        names=[line.split()[-1] for line in data('build/commands/symbols.stdout.txt').decode().splitlines() if line.split()]
        expected={('n9unit_loop_'+row['name']+suffix+'_').lower() for row in spec['families'] for suffix in ('','_qpmass')}
        actual={n for n in names if re.fullmatch(r'n9unit_loop_[0-9_]+(?:qpmass_)?',n)}
        require(actual==expected==set(built['helper_definitions']) and all(names.count(n)==1 for n in expected),'Fourteen unique compiled helpers changed')
        excluded=index['excluded_binaries'];require(len(excluded)==16,'Native exclusion inventory changed')
        for phase in ('build','run'):
            require(excluded[phase+'/unit_mass_check']['sha256']==built['executable_sha256'],'Excluded ELF identity changed')
            for name,wanted in built['objects'].items():require(excluded[phase+'/'+name]['sha256']==wanted,'Excluded object/module identity changed')
        raw=data('run/commands/recording-unit.stdout.txt',RAW)
        parsed=parser.parse(raw.decode());require(parsed==record('run/parsed.json')==ran['parsed'],'Actual raw/parser replay changed')
        require(len(raw)==238802 and parsed['raw_records']==6425,'Actual raw coverage changed')
        command=record('run/commands/recording-unit.command.json')
        require(command['argv']==[excluded['run/unit_mass_check']['original_path']] and not command['subprocess_new_session'],'RUN executable/session changed')
        run_command=record('run/commands/recording-unit.terminal.json')
        require(run_command['returncode']==0 and run_command['stdout_sha256']==RAW,'Actual RUN command failed')
        reviews={label:record('reviews/'+label+'.json',wanted) for label,wanted in REVIEWS.items()}
        require(reviews['source1']['status']=='blocked_pending_source_successor' and reviews['source1']['unit_executions']==0 and
                reviews['source1']['source_index_sha256']==OLD,'Pre-native rejection history changed')
        require({k:reviews[k]['status'] for k in ('source2','build','run')}==
                {'source2':'clear','build':'clear_for_separate_RUN_review','run':'clear'},'Independent review status changed')
        require(reviews['run']['reproduced_parser']==parsed and reviews['run']['closed_RUN_pins']['terminal.json']==RUN,'Independent RUN join changed')
        rows=[line.split() for line in raw.decode().splitlines()];begins={(r[1],r[2]):int(r[4]) for r in rows if r[0]=='BEGIN'}
        mode_counts={str(k):sum(v==k for v in begins.values()) for k in (1,2,4,5)}
        signatures={}
        for row in rows:
            if row[0]!='ENTRY':continue
            lib=row[4];kind=int(row[3]);target=signatures.setdefault(lib,{})
            key='QP_input_signature' if kind==2 else 'DP_input_signature';target[key]=target.get(key,0)+1
            if lib=='1' and begins[(row[1],row[2])]>=4:
                target['DP_signature_with_requested_QP_mode']=target.get('DP_signature_with_requested_QP_mode',0)+1
        require(signatures==reviews['run']['recording_target_input_signatures'] and mode_counts==reviews['run']['requested_CTMODE_counts'],'Requested mode/input signature scope changed')
        require(not attempts,'Replay attempted an external/native/mutation API')
        facts={'compiled_dispatch_input_parity':True,'helper_families':7,'helper_pairs':126,'recording_helper_calls':252,
            'actual_chooser_helper_calls':196,'controlled_selector_helper_calls':56,'chooser_only_controls':2,
            'recorded_rows':6425,'observed_IEEE_rows_zero':507,'requested_CTMODE_counts':mode_counts,
            'recording_target_input_signatures':signatures,'build_commands':30,'compiled_sources':6,
            'unique_helper_definitions':14,'recorded_LOAD_bindings':23,'recorded_runtime_providers':5,
            'excluded_native_identities':16,'native_binaries_packaged':0,'new_native_executions':0,
            'source1_pre_native_rejection_retained':True,'actual_physical_reducers':0,'physical_amplitudes':0,
            'real_COLLIER_initialization':False,'UseQPIntegrandFalse_physical_qualification':False,
            'EW_complex_mass_physical_qualification':False,'loop_induced_physical_qualification':False,
            'local_performance_acceptance':False,'production_promotion':False}
        require(index['claims']==facts,'Published claims differ from actual closed evidence')
    return {**facts,'verified':True,'forbidden_API_attempts':0,
            'runtime_scope':'Recorded BUILD listing/LOAD/provider identities and unchanged RUN ELF; no live binary reads or per-RUN maps'}


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--index-sha256');args=cli.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))
