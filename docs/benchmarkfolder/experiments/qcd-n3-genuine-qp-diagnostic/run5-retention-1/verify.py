#!/usr/bin/env python3
"""Portable full sixteen-arm replay; original workspace paths are never opened."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import struct
import types

PREFIX_SHA='fcd4f75340f7db9261e91d6f0c37e7a1ba85b3ff68b8b45a639ecbcc49ddf361'
SOURCE_SHA='efc8c5b5c7a775fd3b4215171bd0d989bb5d6791710ceeb65d58816ba76b03e5'


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.loads(json.dumps(value,allow_nan=False))
def module(raw,name):
    result=types.ModuleType(name)
    exec(compile(raw,'retained/controller/'+name+'.py','exec'),result.__dict__)
    return result


def replay(payload,inventory,prefix_root):
    j=lambda name:json.loads(payload[name])
    source_index=j('controller/source-index.json')
    require(digest(payload['controller/source-index.json'])==SOURCE_SHA,'Sealed source5 identity changed')
    for name,item in source_index['files'].items():
        raw=payload['controller/'+name]
        require(len(raw)==item['bytes'] and digest(raw)==item['sha256'],'Controller payload changed: '+name)
    analyzer=module(payload['controller/analyze.py'],'combined_analyzer')
    reader=module(payload['controller/prefix_reader.py'],'combined_prefix_reader')
    capsule=reader.Capsule(prefix_root)
    binding=j('controller/bindings.json');manifest=j('controller/donor-closure.json')
    fixture_sha={name:digest(payload['controller/'+name]) for name in source_index['files'] if name.startswith('fixtures/')}
    prefix=reader.validate(capsule,j('controller/closed-run4-preflight.json'),
        digest(payload['controller/analyze.py']),fixture_sha,binding)
    result=j('actual-run5/result.json')
    require(result['passed'] is True and result.get('failure') is None and result['source_index_sha256']==SOURCE_SHA,
            'Continuation protocol did not pass')
    require(result['inherited_workers']==10 and result['new_workers_native_started']==6 and
            result['new_workers_native_reaped']==6 and result['new_workers_parsed']==6 and result['workers_completed']==16,
            'Inherited/new worker counts differ')
    require(result['recompiled'] is False and result['original_donor_runtime_preserved'] is True and
            result['original_source4_attempt_remains_failed'] is True,'Original provenance policy changed')
    require(result['new_phase_budget_seconds_decimal']==str(reader.NEW_BUDGET) and
            result['inherited_elapsed_seconds']==float(reader.OLD_ELAPSED) and
            result['elapsed_seconds']<=float(reader.NEW_BUDGET) and result['combined_elapsed_seconds']<=3000 and
            result['combined_elapsed_seconds']==result['elapsed_seconds']+float(reader.OLD_ELAPSED),
            'Combined original resource budget exceeded or changed')
    new_names=[reader.worker_name(spec) for spec in reader.SUFFIX_SPECS]
    actual_dirs={name for name in inventory['actual_attempt_directories'] if name.startswith('b') and '/' not in name}
    require(actual_dirs==set(new_names) and result['new_worker_order']==new_names,'Not exactly the six never-started workers')
    inherited=j('actual-run5/inherited-prefix.json')
    require(inherited['worker_ids']==prefix['worker_ids'] and inherited['prefix_index_sha256']==PREFIX_SHA and
            inherited['preflight_sha256']==digest(payload['controller/closed-run4-preflight.json']),
            'Inherited prefix receipt differs')
    build=j('controller/inherited-build2.json')
    require(result['build_sha256']==digest(payload['controller/inherited-build2.json']),'Inherited executable proof differs')
    expected={entry['resolved_path']:entry['sha256'] for entry in manifest['entries']}
    require(len(expected)==15 and {e['source_path']:e['sha256'] for e in manifest['entries']}==binding['runtime'],
            'Original fifteen runtime providers differ')
    require(inventory['actual_attempt_symlinks']=={'closed-runtime/'+e['soname']:e['resolved_path'] for e in manifest['entries']},
            'New alias inventory differs')
    alias_dir=binding['native_library_path']
    alias_paths={alias_dir+'/'+e['soname']:e['resolved_path'] for e in manifest['entries']}
    proof_names=['actual-run5/closed-runtime-proof.json']+['actual-run5/'+name+'/'+leaf for name in new_names
                    for leaf in ('closed-runtime-pretrace.json','closed-runtime-preworker.json')]
    for name in proof_names:
        proof=j(name)
        require(proof['passed'] is True and proof['library_count']==15 and proof['alias_directory']==alias_dir,
                'New alias guard failed')
        require({e['soname']:(e['alias_target'],e['resolved_path'],e['sha256']) for e in proof['libraries']}==
                {e['soname']:(e['resolved_path'],e['resolved_path'],e['sha256']) for e in manifest['entries']},
                'New alias/hash provider identity differs')
    for root in ['actual-run5']+['actual-run5/'+name for name in new_names]:
        receipt=j(root+'/loader.json');text=payload[root+'/loader.log'].decode()
        require(receipt['native'] is False and receipt['returncode']==0 and receipt['process_reaped'] is True and
                not receipt.get('failure') and not receipt.get('timed_out') and 0<receipt['seconds_cap']<=30,
                'New loader trace did not close successfully')
        require(receipt['target_loader_environment_at_execve']=={'LD_LIBRARY_PATH':alias_dir,'LD_TRACE_LOADED_OBJECTS':'1'},
                'New trace environment differs')
        require('not found' not in text and {alias_paths.get(p,p) for p in set(re.findall(r'/[^\s()]+',text))}==set(expected),
                'Strict new trace closure differs')
    selected=j('controller/fixtures/selected-inputs.json')
    points={p:b''.join(struct.pack('<d',float.fromhex(v)) for row in
                      selected['points'][str(p)]['binary64_hex'] for v in row) for p in (1,93,39)}
    workers=[]
    for spec,pref in zip(reader.PREFIX_SPECS,prefix['workers']):
        name=reader.worker_name(spec);root='actual-run4/'+name
        parsed=analyzer.parse_worker(capsule.raw(root+'/trace.txt').decode(),capsule.raw(root+'/worker.log').decode(),*spec,points)
        require(canonical(parsed)==pref,'Full inherited raw-worker replay differs from sealed preflight')
        workers.append(parsed)
    for spec,name in zip(reader.SUFFIX_SPECS,new_names):
        root='actual-run5/'+name;r=j(root+'/worker.json');mapping=j(root+'/mapped-runtime.json')
        require(r['native'] is True and r['returncode']==0 and r['process_spawned'] is True and r['process_reaped'] is True and
                not r.get('failure') and not r.get('timed_out') and 0<r['seconds_cap']<=180,'New native worker did not close cleanly')
        require(r['target_loader_environment_at_execve']=={'LD_LIBRARY_PATH':alias_dir},'New worker environment differs')
        require(mapping['expected_runtime_sha256']==mapping['mapped_files_sha256_after_exit']==expected and
                set(mapping['mapped_elf_paths'])==set(expected)|{build['executable']} and
                mapping['executable']==build['executable'] and mapping['cpu_allowed_list']=='12','New actual ELF closure differs')
        maps=payload[root+'/maps.txt']
        require(mapping['maps_text'].encode()==maps and mapping['maps_sha256']==digest(maps)==
                r['mapping_capture']['complete_snapshot_sha256'],'Actual new map capture differs')
        for leaf,sha in j(root+'/outputs.json').items():require(digest(payload[root+'/'+leaf])==sha,'New output inventory differs')
        cwd=r['cwd'].rstrip('/')+'/'
        for original,sha in j(root+'/inputs.json').items():
            require(original.startswith(cwd) and digest(payload[root+'/'+original[len(cwd):]])==sha,'New fixture inventory differs')
        parsed=analyzer.parse_worker(payload[root+'/trace.txt'].decode(),payload[root+'/worker.log'].decode(),*spec,points)
        require(canonical(parsed)==j(root+'/parsed.json') and j(root+'/warnings.json')==[],'New parsed-worker replay differs')
        workers.append(parsed)
    reader.require_complete_matrix(workers)
    analysis=analyzer.adjudicate(workers)
    require(canonical(analysis)==j('actual-run5/analysis.json'),'Combined sixteen-worker adjudication replay differs')
    require(digest(payload['actual-run5/analysis.json'])==result['analysis_sha256'],'Combined analysis hash differs')
    require(analysis['all_shared_input_gates_pass'] is True and analysis['host_checked'] is False and
            analysis['timing_or_campaign_acceptance'] is False,'Unchecked diagnostic scope changed')
    inherited_calls=sum(w['public_calls'] for w in workers[:10]);new_calls=sum(w['public_calls'] for w in workers[10:])
    return {'passed':True,'scope':'Portable full sixteen-arm consistency replay; unchecked120 is not an accuracy result',
        'source5_index_sha256':SOURCE_SHA,'required_prefix_index_sha256':PREFIX_SHA,
        'combined_analysis_sha256':result['analysis_sha256'],'original_run4_remains_failed':True,
        'inherited_native_workers':10,'new_native_workers':6,'combined_unique_workers':16,
        'inherited_public_calls':inherited_calls,'new_public_calls':new_calls,'combined_public_calls':inherited_calls+new_calls,
        'bootstrap_and_initialization_calls':analysis['bootstrap_and_initialization_calls'],
        'diagnostic_calls':analysis['diagnostic_physical_calls'],'unchecked_answer_order_columns':analysis['answer_order_rows'],
        'applicable_pair_point_comparisons':len(analysis['comparisons']),'all_shared_input_gates_pass':True,
        'plain_observed_public_parity_pass':True,'full_raw_prefix_and_suffix_reparsed':True,
        'all_original_runtime_providers_verified':15,'new_loader_traces_passed':7,'inherited_loader_traces_passed':11,
        'warnings_W03':0,'host_checked':False,'scientific_accuracy_qualified':False,'timing_acceptance':False,
        'native_elapsed_resource_scope_only':{'run4':result['inherited_elapsed_seconds'],
            'run5':result['elapsed_seconds'],'combined':result['combined_elapsed_seconds'],'budget':3000}}


def verify(root=None,prefix_root=None):
    root=Path(root or Path(__file__).resolve().parent)
    prefix_root=Path(prefix_root or root.parent/'failed-run4-retention-1')
    index=json.loads((root/'evidence-index.json').read_bytes());observed=set();stored={}
    for p in root.rglob('*'):
        require(not p.is_symlink(),'Package symlink is forbidden')
        if p.is_file():observed.add(p.relative_to(root).as_posix())
    require(observed==set(index['files'])|{'evidence-index.json'},'Package membership differs')
    for name,item in index['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==name,'Unsafe package member')
        raw=(root/p).read_bytes();require(len(raw)==item['bytes'] and digest(raw)==item['sha256'],'Changed retained member: '+name)
        stored[name]=raw
    dependency=json.loads(stored['prefix-dependency.json'])
    require(dependency['index_sha256']==index['required_prefix_index_sha256']==PREFIX_SHA,'Fixed prefix dependency differs')
    inventory=json.loads(stored['retention-inventory.json']);payload={}
    for logical,r in inventory['original_files'].items():
        raw=stored[r['member']]
        require(len(raw)==r['stored_bytes'] and digest(raw)==r['stored_sha256'],'Stored original differs')
        require(r['codec'] in ('identity','gzip'),'Unsupported codec')
        raw=gzip.decompress(raw) if r['codec']=='gzip' else raw
        require(len(raw)==r['raw_bytes'] and digest(raw)==r['raw_sha256'],'Raw original differs')
        payload[logical]=raw
    summary=replay(payload,inventory,prefix_root)
    require(summary==json.loads(stored['summary.json']),'Summary replay differs')
    return summary


if __name__=='__main__':print(json.dumps(verify(),sort_keys=True,indent=2))
