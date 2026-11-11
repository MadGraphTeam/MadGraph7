"""Portable text-only replay. Never execute historical controllers/review scripts."""
import argparse
import builtins
from contextlib import contextmanager
import copy
from datetime import datetime
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import posixpath
import random
import re
import subprocess
import sys
import types

sys.dont_write_bytecode=True
SOURCE='4e3878690b366992d1dad8e43ae650a62362220d7bfc21e85850228d2aeef12d'
PARENT='3d2e4b775334374082771f670996e67138205b66cf7d5a7ff084013235046f4c'
HELPER='4dfedb9f940dd8e0565ff1ff34b9aa0d56635a5e5d8c52b328359fbfc754afda'
BUILD='a12fd704ab54e935028258299239beb62f6d366620f990e0130db4f9df4fa506'
BUILD_RESULT='13b896d86270ca9da75404046d0a1de655a0adaabcf8987cbf2846f54b2f5f25'
RUN='7a67bf86babae2497c65c2c421995c7038edd1e19186c1d5ed0ecc24f707f93e'
RUN_RESULT='49b2cf6c15c6cc9b4ef43519c8f21d1a6126901c28ba140199d0f9c295f843db'
RAW='a40826258c4842ac4ec21fa477a28f6b75670484387ea8c9c30ddfdafdbb4be7'
REVIEWS={'independent-build':'1524bc3595e3bff40e0b6007aa56d8a08f47b375a3df3447a89c5ddc8a0edd99',
         'independent-run':'0dd577ffc378140a9b3b37fda3c259bad5b51c68cf728ab5ce3559880448e9d7',
         'root-run':'9da499cafea01f94f77c9dc48f8a4d027e7a7054d903fb74ea92472c186a871b'}


def require(value,message):
    if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def normalized(value):return json.loads(json.dumps(value,sort_keys=True))


@contextmanager
def blocked_apis(memory):
    """Numerical replay gets only exact preloaded Path reads; no live filesystem."""
    saved=[];attempts=[]
    def deny(*args,**kwargs):
        attempts.append(repr(args[:1]));raise RuntimeError('Forbidden API during portable replay')
    def memory_bytes(path,*args,**kwargs):
        key=str(path)
        if key not in memory:return deny(key)
        return memory[key]
    def memory_text(path,encoding=None,errors=None,*args,**kwargs):
        return memory_bytes(path).decode(encoding or 'utf-8',errors or 'strict')
    def patch(obj,name,value=deny):
        if hasattr(obj,name):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    try:
        patch(Path,'read_bytes',memory_bytes);patch(Path,'read_text',memory_text)
        for name in ('open','stat','lstat','resolve','exists','is_file','is_dir','is_symlink','iterdir','glob','rglob',
                     'write_bytes','write_text','mkdir','unlink','rename','replace','touch','chmod','symlink_to','hardlink_to'):
            patch(Path,name)
        patch(builtins,'open');patch(io,'open');patch(io,'open_code')
        for name in ('open','stat','lstat','access','listdir','scandir','readlink','system','popen','posix_spawn','posix_spawnp',
                     'execv','execve','execvp','execvpe','spawnv','spawnve','spawnvp','spawnvpe','fork','forkpty',
                     'mkdir','makedirs','unlink','remove','rename','replace','rmdir','removedirs','chmod','chown',
                     'truncate','utime','link','symlink'):
            patch(os,name)
        for name in ('Popen','run','call','check_call','check_output'):patch(subprocess,name)
        if 'ctypes' in sys.modules:
            module=sys.modules['ctypes']
            for name in ('CDLL','PyDLL','_dlopen'):patch(module,name)
        if '_ctypes' in sys.modules:patch(sys.modules['_ctypes'],'dlopen')
        yield attempts
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)


def load_source(name,data,path):
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(data,str(path),'exec'),module.__dict__)
    return module


def verify(root,expected_index_sha256=None):
    root=Path(root).resolve()
    index_data=(root/'index.json').read_bytes()
    if expected_index_sha256 is not None:
        require(sha(index_data)==expected_index_sha256,'Portable index seal changed')
    index=json.loads(index_data)
    require(index['schema']=='exact-geometry-portable-qualification-v1','Wrong capsule schema')
    actual=set()
    for item in root.rglob('*'):
        require(not item.is_symlink(),'Capsule symlinks are forbidden')
        if item.is_file():actual.add(str(item.relative_to(root)))
    require(actual==set(index['files'])|{'index.json'},'Capsule exact inventory changed')
    physical={}
    for relative,row in index['files'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'Unsafe owned payload path')
        data=(root/relative).read_bytes()
        require(len(data)==row['bytes'] and sha(data)==row['sha256'],'Changed owned payload: '+relative)
        require(not data.startswith(b'\x7fELF'),'Native binary was packaged')
        physical[relative]=data
    blobs={}
    for logical,row in index['logical_files'].items():
        payload=physical[row['path']]
        require(row['compression'] in ('none','gzip'),'Unsupported retention encoding')
        if row['compression']=='gzip':
            require(payload[:3]==b'\x1f\x8b\x08' and payload[4:8]==b'\0'*4,'Non-deterministic gzip header')
            payload=gzip.decompress(payload)
        require(len(payload)==row['raw_bytes'] and sha(payload)==row['raw_sha256'],'Changed logical text: '+logical)
        payload.decode('utf-8');blobs[logical]=payload
    def data(name,checksum=None):
        raw=blobs[name]
        if checksum is not None:require(sha(raw)==checksum,'Changed frozen closure: '+name)
        return raw
    def record(name,checksum=None):return json.loads(data(name,checksum))
    for prefix,checksum in [('source',SOURCE),('generic-equality-postcondition-proposal-1',PARENT)]:
        seal=record(prefix+'/source-index.json',checksum)
        owned={n[len(prefix)+1:] for n in blobs if n.startswith(prefix+'/')}
        require(owned==set(seal['files'])|{'source-index.json'},'Nested source inventory changed')
        for name,row in seal['files'].items():
            value=data(prefix+'/'+name,row['sha256'])
            require(len(value)==row['bytes'],'Nested source payload size changed')
    data('source/exact_geometry.f90',HELPER)
    data('generic-equality-postcondition-proposal-1/exact_geometry.f90',HELPER)
    # These modules are authenticated source-only Python. Their import-time Path
    # setup stays inside this capsule; actual arithmetic below runs with APIs blocked.
    reference=load_source('portable_exact_geometry_reference',data('generic-equality-postcondition-proposal-1/reference.py'),
                          root/'generic-equality-postcondition-proposal-1/reference.py')
    corpus=load_source('portable_exact_geometry_corpus',data('source/corpus.py'),root/'source/corpus.py')
    analyzer=load_source('portable_exact_geometry_analyzer',data('source/analyze.py'),root/'source/analyze.py')
    # Guard-only I/O substitution: the whole parent source was authenticated above.
    # No Fraction, capacity, generator or parser arithmetic is replaced.
    corpus.frozen_module=lambda:reference
    memory={str(root/name):raw for name,raw in blobs.items()}
    with blocked_apis(memory) as attempts:
        cases=corpus.build();expected=corpus.expectations(cases)
        require(corpus.encode(cases).encode()==data('source/fixtures.hex'),'Frozen fixtures differ from generator')
        require(expected==record('source/expectations.json'),'Independent Fraction/capacity expectations changed')
        require(expected['case_count']==1698 and expected['counts']['expected_accept']==526 and
                expected['counts']['expected_refuse']==1172,'Wrong oracle coverage')
        require(sum(r['mathematical_equality'] is True and not r['expected_accept']
                    for r in expected['cases'].values())==10,'Capacity-refusal witnesses lost')
        build_terminal=record('build/terminal.json',BUILD)
        built=record('build/result.json',BUILD_RESULT)
        run_terminal=record('run/terminal.json',RUN)
        ran=record('run/result.json',RUN_RESULT)
        for phase,terminal,result in [('build',build_terminal,built),('run',run_terminal,ran)]:
            require(terminal['status']==result['status']=='passed' and terminal['returncode']==0 and
                    terminal['cleanup']['survivors']==[],'Unclosed native phase')
            require(terminal['result_sha256']==sha(data(phase+'/result.json')) and
                    result['source_index_sha256']==SOURCE,'Phase source/result chain differs')
            require(terminal['cap_seconds']==180 and terminal['cleanup_reserve_seconds']==60,'Phase budget changed')
            require((datetime.fromisoformat(terminal['finished_utc'])-
                     datetime.fromisoformat(terminal['started_utc'])).total_seconds()<180,'Phase exceeded budget')
            require(record(phase+'/limits.json')['actual']=={'address_space_bytes':[4294967296]*2,
                    'file_bytes':[268435456]*2,'core_bytes':[0,0]},'Phase resource limits changed')
            require(record(phase+'/admission.json')['source_index_sha256']==SOURCE,'Phase admission differs')
        require(ran['reviewed_build_sha256']==BUILD and record('run/admission.json')['reviewed_build_sha256']==BUILD,
                'Separate reviewed BUILD not bound')
        require(built['helper_sha256']==HELPER and built['helper_main_calls']==built['scientific_reducer_calls']==0,
                'BUILD scope changed')
        require(ran['scientific_reducer_calls']==0 and not ran['physical_or_performance_qualification'] and
                not ran['snapshot_ownership_qualified'],'RUN scope inflated')
        commands=sorted(n for n in blobs if n.startswith('build/commands/') and n.endswith('.command.json'))
        require(len(commands)==57,'BUILD command coverage changed')
        for name in commands:
            stem=name[:-len('.command.json')]
            result=record(stem+'.terminal.json')
            require(result['returncode']==0 and result['stdout_sha256']==sha(data(stem+'.stdout.txt')),
                    'BUILD command/log failure or mismatch')
            argv=record(name)['argv']
            require(argv[0] not in [r['path'] for r in built['variants'].values()],'BUILD executed a Fortran main')
        source_pins=record('source/input-pins.json');excluded=index['excluded_binaries']
        require(len(excluded)==9 and not any(n in blobs for n in excluded),'Binary inventory changed')
        require(excluded['build/fenv_observer.o']['sha256']==built['fenv_object_sha256'],'Observer binary identity lost')
        replayed={};serial=[]
        for variant,row in built['variants'].items():
            prefix='build/'+variant;folder='run/'+variant
            require(excluded[prefix+'/check']['sha256']==row['sha256'],'Executable binary identity lost')
            for name,checksum in row['objects'].items():
                require(excluded[prefix+'/'+name]['sha256']==checksum,'Object binary identity lost')
            prebound=record(prefix+'/prelink.json',row['prelink_sha256'])
            data(prefix+'/link.map',row['link_map_sha256'])
            static=record(prefix+'/static-runtime.json',row['static_runtime_sha256'])
            require(row['runtime']['files']==static['files']==source_pins['runtime_before_link_augmentation'],
                    'Standalone runtime identity set changed')
            require(len(static['files'])==5 and all(prebound['bound'].get(n)==h for n,h in
                    {**row['actual_link_loads'],**static['files']}.items()),'Actual recorded LOAD/provider was not prebound')
            aliases={n:n for n in static['files']}
            aliases.update({x['selected_spelling']:x['selected'] for x in static['resolutions'] if 'selected_spelling' in x})
            providers=set()
            for line in row['runtime']['listing'].splitlines():
                line=line.strip()
                if not line or line.startswith('linux-vdso.so.1 '):continue
                match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line)
                require(match is not None and match[1] in aliases,'Unrecognized recorded actual loader provider')
                providers.add(aliases[match[1]])
            providers.add(row['runtime']['loader'])
            require(providers==set(static['files']),'Actual PT_INTERP listing/provider join differs')
            require(not any(re.search(r'float|quadmath|multc|trunc|extend|feset|ieee',n,re.I)
                            for n in row['helper_undefined_symbols']),'Helper FP/environment dependency claim invalid')
            command=record(folder+'/command.json');started=record(folder+'/started.json')
            closed=record(folder+'/terminal.json');pid=started['pid']
            require(closed['returncode']==0 and not closed['stdout_tail'] and not data(folder+'/stderr.txt'),
                    'Standalone worker failure lost')
            require(sha(data(folder+'/stderr.txt'))==closed['stderr_sha256'],'Worker stderr identity changed')
            require(data(folder+'/ready.txt').decode().strip()=='READY '+str(pid) and
                    data(folder+'/done.txt').decode().strip()=='DONE '+str(pid),'Worker handshakes differ')
            require(command['new_session'] is False,'Unexpected worker session')
            serial.append((datetime.fromisoformat(command['started_utc']),datetime.fromisoformat(closed['finished_utc'])))
            for edge in ('before','after'):
                mapped=record(folder+'/maps-'+edge+'.json')
                require(mapped['ELF']=={**static['files'],row['path']:row['sha256']} and mapped['affinity']==[12] and
                        not mapped['exclusive_CPU'],'Actual worker provider/CPU set differs')
                paths={posixpath.normpath(line.split(None,5)[5]) for line in mapped['raw'].splitlines()
                       if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                require(paths==set(mapped['ELF'])|set(mapped['data']) and
                        all(source_pins['allowed_data_mappings'].get(n)==h for n,h in mapped['data'].items()),
                        'Recorded raw maps disagree with stored identities')
                require(mapped['pid']==mapped['identity']['pid']==pid,'Mapped worker PID differs')
                keys=('pid','parent_pid','process_group','session','start_time_ticks')
                require(all(mapped['identity'][k]==started['identity'][k] for k in keys),'Worker identity changed')
                require(any(all(x[k]==mapped['identity'][k] for k in ('pid','process_group','session','start_time_ticks'))
                            for x in run_terminal['cleanup']['observed_owned_processes']),'Worker absent from ownership cleanup')
            raw=data(folder+'/observations.txt',RAW)
            require(len(raw)==6698235 and closed['observations_sha256']==RAW,'Raw observation coverage changed')
            result=normalized(analyzer.analyze(raw.decode(),expected))
            require(result==record(folder+'/analysis.json')==ran['variants'][variant],
                    'Strict raw native replay differs from retained results')
            require(result['counts']=={'accepts':10520,'refusals':23440,'executed_mode_observations':33960} and
                    result['mode_status']=={str(i):0 for i in range(20)} and
                    result['native_INTEGER128_input_words_joined']==66320,'Actual cohort/answer/input coverage changed')
            replayed[variant]=result
        require(serial[0][1]<=serial[1][0],'Workers were not serial')
        require(replayed['checked']==replayed['optimized'] and
                data('run/checked/observations.txt')==data('run/optimized/observations.txt'),
                'Checked and optimized raw outcomes differ')
        for label,checksum in REVIEWS.items():
            receipt=record('reviews/'+label+'/receipt.json',checksum)
            if label!='root-run':
                data('reviews/'+label+'/review.py',receipt['audit_source_sha256'])
                require(receipt['status']=='PASS','Independent review is not passed')
            else:
                require(receipt['status']=='passed' and [r['status'] for r in receipt['replay_history']]==
                        ['failed','failed','passed'],'Honest root replay history changed')
        require(not attempts,'A forbidden API was attempted during numerical replay')
        facts={'fixture_cases':1698,'variants':2,'total_helper_calls':67920,'executed_cohorts':20,
            'skipped_cohorts':0,'accepts_per_variant':10520,'refusals_per_variant':23440,
            'INTEGER128_words_joined_total':132640,'checked_optimized_raw_identical':True,
            'observed_environment_and_restoration_exact':True,'build_commands_passed':57,
            'runtime_providers_per_variant':5,'excluded_binary_identities':9,
            'root_pure_review_history_preserved':True,'native_binaries_packaged':0,
            'new_compiler_loader_native_calls':0,'snapshot_ownership_qualified':False,
            'physical_amplitude_qualified':False,'performance_acceptance':False,
            'actual_q_at_physical_use_qualified':False}
        require(index['claims']==facts,'Published claims differ from replayed evidence')
    return {**facts,'verified':True,'forbidden_API_attempts':len(attempts),
            'runtime_identity_scope':'Recorded actual LOAD/PT_INTERP/maps receipts only; no live binary/provider opens.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index-sha256')
    args=parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))
