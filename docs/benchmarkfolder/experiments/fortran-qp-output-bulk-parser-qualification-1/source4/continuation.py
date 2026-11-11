"""Strict file-only replay of every retained case before fresh suffix processes."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
SOURCE_SHA='511225ea5bb4523cbf84012ea424a0710ef6f86b3ec6964ba8e8b41ea08a76d6'
BUILD_SHA='d4eb1386ce923a16d70666637ff3677091449b30a812b6ebcb90b421c0b3cf9f'
RUN_SHA='4e70b14200b037f0acf05723f05a29078f26e95bf2871b3be4fc97a4421a815f'


def read(path):return json.loads(Path(path).read_text())
def identity(path):
    b=Path(path).read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def require(value,message):
    if not value:raise ValueError(message)
def module(name):
    spec=importlib.util.spec_from_file_location('bulk_retained_'+name,HERE/(name+'.py'))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def verify_inventory(base,files):
    actual={str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
    require(actual==set(files),'Retained inventory changed: '+str(base))
    for rel,want in files.items():
        q=Path(rel)
        require(not q.is_absolute() and '..' not in q.parts,'Unsafe retained relative path')
        require(identity(base/q)==want,'Changed retained bytes: '+str(base/q))


def partition(manifest,corpus):
    prefix=manifest['prefix_case_ids'];pending=manifest['pending_case_ids']
    require(len(prefix)==88 and prefix[-1]=='options-f0-p3' and len(set(prefix))==88,'Wrong retained prefix')
    end=next(i for i,c in enumerate(corpus) if c['id']==prefix[-1])+1
    require(prefix==[c['id'] for c in corpus[:end] if c['execution']!='lexical-only'],'Retained cases not exact corpus prefix')
    require(pending==[c['id'] for c in corpus[end:] if c['execution']!='lexical-only'],'Pending cases not untouched corpus suffix')
    require(len(pending)==47 and len(set(pending))==47 and not set(prefix)&set(pending),'Overlapping/missing suffix')
    require(manifest['lexical_only_case_ids']==[c['id'] for c in corpus if c['execution']=='lexical-only'] and
            len(manifest['lexical_only_case_ids'])==14,'Changed lexical-only cases')
    return prefix,pending


def expected_case_bytes(case,corpus):
    fields=['id','n','rank','poles','finite','check','cache','clear_fails','rounding','sticky_mask','halt_flag']
    return (' '.join(str(case[key]) for key in fields)+'\n'+'\n'.join(corpus.records(case))+'\n').encode()


def replay(built,reviewed_run):
    require(reviewed_run==RUN_SHA,'Exact reviewed failed RUN3 required before any suffix invocation')
    manifest=read(HERE/'continuation-inputs.json');corpus=read(HERE/'corpus.json')['cases']
    prefix,pending=partition(manifest,corpus)
    source_root=ROOT/manifest['source3_root'];build_root=ROOT/manifest['build_root'];run_root=ROOT/manifest['failed_run_root']
    require(identity(source_root/'source-index.json')['sha256']==SOURCE_SHA,'Compiled source3 seal changed')
    require(identity(build_root/'terminal.json')['sha256']==BUILD_SHA,'Reviewed BUILD3 changed')
    require(identity(run_root/'terminal.json')['sha256']==RUN_SHA,'Reviewed failed RUN3 changed')
    verify_inventory(source_root,{**manifest['source3_files'],'source-index.json':manifest['source3_index']})
    verify_inventory(build_root,manifest['build_files']);verify_inventory(run_root,manifest['failed_run_files'])
    for rel,want in manifest['scientific_files'].items():
        require(identity(HERE/rel)==identity(source_root/rel)==want,'Scientific source/corpus changed: '+rel)
    terminal=read(run_root/'terminal.json')
    require(terminal['status']=='failed' and terminal['returncode']==1 and not terminal['cleanup']['survivors'],'Old failure closure differs')
    admission=read(run_root/'admission.json')
    require(admission['source_index_sha256']==SOURCE_SHA and admission['reviewed_build_sha256']==BUILD_SHA and admission['stage']=='run','Retained run admission mismatch')
    require(read(build_root/'result.json')==built and built['source_index_sha256']==SOURCE_SHA,'Retained compiled build identity mismatch')
    require({p.name for p in run_root.iterdir() if p.is_dir()}==set(prefix),'Unexpected executed/missing retained case')
    require(all(not (run_root/name).exists() for name in pending+manifest['lexical_only_case_ids']),'Untouched/lexical case already exists')
    analyzer=module('analyze_rejection');wire=module('corpus');observations=[];old_comparisons=0;transcripts={}
    byid={case['id']:case for case in corpus}
    for name in prefix:
        case=byid[name];folder=run_root/name
        require((folder/'case.txt').read_bytes()==expected_case_bytes(case,wire),'Retained input differs from exact frozen corpus')
        pair={}
        for variant in ('original','candidate','observer'):
            row=built['executables'][variant];exe=Path(row['path'])
            require(identity(exe)['sha256']==row['sha256'],'Retained compiled executable changed')
            command=read(folder/(variant+'.command.json'));end=read(folder/(variant+'.terminal.json'))
            require(command['argv']==[str(exe),str(folder/'case.txt')] and command['cwd']==str(folder),'Retained native command/input identity mismatch')
            require(command['subprocess_new_session'] is False and command['timeout_seconds']==10.,'Retained invocation bounds differ')
            require(all(end.get(key)==value for key,value in command.items()),'Retained command/terminal join differs')
            stdout=folder/(variant+'.stdout.txt')
            require(end['returncode']==0 and not end.get('timed_out') and end['stdout_sha256']==identity(stdout)['sha256'],'Retained output terminal incomplete/changed')
            pair[variant]={'text':stdout.read_text(),'terminal':end}
            transcripts[name+'/'+variant]=identity(stdout)
        observation=analyzer.compare(case,pair)
        old=folder/'comparison.json'
        if name==prefix[-1]:require(not old.exists(),'Failed source3 case was retrospectively rewritten')
        else:
            require(old.is_file() and read(old)==observation,'Corrected oracle changed an earlier passing comparison')
            old_comparisons+=1
        observations.append(observation)
    require(old_comparisons==87 and len(transcripts)==264,'Retained replay coverage differs')
    return {'schema':'madloop-tirust-bulk-unit-retained-replay-v1','status':'pass',
            'old_failed_run_status_preserved':'failed','reviewed_failed_run_sha256':RUN_SHA,
            'compiled_source_index_sha256':SOURCE_SHA,'reviewed_build_sha256':BUILD_SHA,
            'replayed_cases':observations,'replayed_case_count':88,'unchanged_old_comparisons':87,
            'corrected_native_rejection_case':'options-f0-p3','retained_transcripts':transcripts,
            'retained_executable_invocations':264,'retained_public_mock_calls':1320,
            'prefix_case_ids':prefix,'pending_case_ids':pending,'lexical_only_case_ids':manifest['lexical_only_case_ids'],
            'new_compiler_invocations':0,'new_native_executions':0,'physical_amplitude_calls':0,'actual_tirust_calls':0,
            'scope':'Retrospective source-defined native-rejection oracle correction, all other gates unchanged; no rewriting of failed RUN3'}
