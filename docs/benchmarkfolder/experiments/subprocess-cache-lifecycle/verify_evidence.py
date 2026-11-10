#!/usr/bin/env python3
"""Replay the portable evidence only; never execute a compiler or reducer."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import struct
from pathlib import Path, PurePosixPath
import types

ANALYZER_SHA='0014bbf833cce64344ffcf38e2cb8c9a92605eeeb697fcf260553a13e5d1d40b'
SMOKE_SOURCE_SHA='8c824f3acd8b0dd479a6f709d917e2f82896154fae3a952f65ceb8d6254ba0f5'


def verify(root=None,*,read_bytes=None,read_link=None):
    """Verify via relative-name callbacks, or a local evidence directory.

    A callback may expose an immutable report snapshot. Absolute paths stored
    in producer metadata are compared as strings; none are opened here.
    """
    def sha(raw): return hashlib.sha256(raw).hexdigest()
    def load(name): return json.loads(read_bytes(name))
    def require(condition,message):
        if not condition: raise ValueError(message)
    def safe(name):
        path=PurePosixPath(str(name))
        require(not path.is_absolute() and '..' not in path.parts,'Nonportable evidence path')
        require(str(path)==str(name),'Noncanonical evidence path')
        return str(path)
    if read_bytes is None:
        root=Path(root or Path(__file__).resolve().parent).absolute()
        require(all(not item.is_symlink() for item in (root,*root.parents)),
                'Evidence root contains a symlink ancestor')
        def filesystem_read(name):
            path=root
            for component in PurePosixPath(name).parts:
                path=path/component
                require(not path.is_symlink(),'Indexed evidence path contains a symlink')
            require(path.is_file(),'Indexed evidence is not a regular file')
            return path.read_bytes()
        read_bytes=filesystem_read
    raw_reader=read_bytes
    index=None
    def guarded_read(name):
        name=safe(name)
        if index is not None and name!='evidence-index.json':
            require(name in index['files'] and index['files'][name]['kind']=='file',
                    'Read outside indexed regular files: '+name)
        return raw_reader(name)
    read_bytes=guarded_read
    index_bytes=read_bytes('evidence-index.json')
    index=json.loads(index_bytes)
    for name,item in index['files'].items():
        safe(name)
        require(item['kind']=='file','Portable index contains a nonregular file')
        raw=read_bytes(name)
        require(len(raw)==item['size'] and sha(raw)==item['sha256'],'Changed evidence: '+name)
    for item in index['historical_symlinks']:
        # Inert provenance only: never open the historical symlink path.
        require(item['target'].startswith('../') and item['target'].count('/')==1,
                'Unexpected historical link spelling')
        target=str(PurePosixPath(item['path']).parent.parent/item['target'][3:])
        require(target==item['resolved_regular_path'] and target in index['files']
                and sha(read_bytes(safe(target)))==item['resolved_sha256'],
                'Historical link target binding differs')
    for item in index['compressed_maps']:
        packed=read_bytes(item['path'])
        raw=gzip.decompress(packed)
        require(sha(packed)==item['gzip_sha256'] and sha(raw)==item['raw_sha256'],
                'Compressed map binding failed')
        require(len(raw)==item['raw_bytes'] and b'\x00' not in raw,'Invalid text map')
        raw.decode('utf-8')

    class ReadPath:
        def __init__(self,name): self.name=safe(name)
        def __truediv__(self,name): return ReadPath(str(PurePosixPath(self.name)/str(name)))
        def __str__(self): return self.name
        def read_bytes(self): return read_bytes(self.name)
        def read_text(self,encoding='utf-8'): return self.read_bytes().decode(encoding)
        def stat(self): return types.SimpleNamespace(st_size=len(self.read_bytes()))

    source=read_bytes('analyze.py')
    require(sha(source)==ANALYZER_SHA,'Unexpected analyzer source')
    namespace={'__name__':'frozen_lifecycle_analysis','__file__':'analyze.py'}
    exec(compile(source,'analyze.py','exec'),namespace)
    namespace['Path']=ReadPath
    analysis=types.SimpleNamespace(**namespace)
    replay=json.loads(json.dumps(analysis.analyze('attempt5/runs'),allow_nan=False))
    results=load('attempt5/results.json')
    require(replay==results and results['passed'],'Frozen raw analysis failed or differs')

    smoke_source=read_bytes('attempt4/resume.py')
    require(sha(smoke_source)==SMOKE_SOURCE_SHA,'Unexpected smoke checker source')
    functions=[node for node in ast.parse(smoke_source).body
               if isinstance(node,ast.FunctionDef) and node.name=='check_smoke']
    require(len(functions)==1,'Missing unique frozen smoke checker')
    smoke_namespace={'analyzer':lambda:analysis}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'attempt4/resume.py','exec'),smoke_namespace)
    smoke=smoke_namespace['check_smoke'](ReadPath('attempt5/runs/card-smoke/records.txt'),
                                      ReadPath('attempt5/runs/card-smoke/trace.txt'))
    smoke_result=load('attempt5/card-smoke-result.json')
    require(smoke==smoke_result['analysis'] and smoke['passed'],'Card smoke replay failed')
    for name,item in smoke_result['evidence'].items():
        require(sha(read_bytes('attempt5/runs/card-smoke/'+safe(name)))==item['sha256'],
                'Smoke prerequisite evidence changed')
    require(sha(read_bytes('attempt5/execution/native-card-smoke.json'))==smoke_result['execution_metadata_sha256'],
            'Smoke execution prerequisite changed')

    build=load('attempt4/build.json')
    build_sha=sha(read_bytes('attempt4/build.json'))
    require(read_bytes('attempt5/build.json')==read_bytes('attempt4/build.json'),'Attempt5 build copy differs')
    require(build['sources_freeze_sha256']==sha(read_bytes('attempt4/source-freeze.json')),
            'Build/source freeze mismatch')
    require(smoke_result['build_sha256']==build_sha,'Smoke build prerequisite changed')
    generated=load('generated-source-freeze.json')['source_inventory']
    installed=load('source-freeze.json')['installation']
    runtime_sets=[item['runtime']['files'] for item in build['executables'].values()]
    require(all(value==runtime_sets[0] for value in runtime_sets),'Built runtime closures differ')
    for name,item in installed.items():
        if name.endswith('/libmadloop_tirust.so'):
            require(runtime_sets[0].get(name)==item['sha256'],'Runtime does not bind the qualified DSO')
    labels=[('card-smoke','card-smoke',0),('plain-cache0','plain',0),
            ('plain-cache-all','plain',-1),('observed-cache0','observed',0),
            ('observed-cache-all','observed',-1)]
    for label,variant,cache in labels:
        prefix='attempt5/runs/'+label+'/'
        inputs=load(prefix+'inputs.json');outputs=load(prefix+'outputs.json')
        command=load('attempt5/execution/native-'+label+'.json')
        executable=build['executables'][variant]
        require(inputs['build_sha256']==build_sha and inputs['executable']==executable,
                'Run executable/build identity differs: '+label)
        require((inputs['label'],inputs['variant'],inputs['cache'])==(label,variant,cache),
                'Run role/cache differs: '+label)
        require(command['argv']==['taskset','-c','10',executable['path'],str(cache)]
                and command['cwd'].endswith('/attempt5/runs/'+label), 'Run command differs: '+label)
        require(command['returncode']==0 and not command['timed_out']
                and command['interruption'] is None and command['cleanup_exception'] is None,
                'Unsuccessful native invocation: '+label)
        require(command['deadline_seconds']==(30 if label=='card-smoke' else 180),
                'Run resource cap differs')
        require(sha(read_bytes('attempt5/execution/native-'+label+'.txt'))==command['log_sha256'],
                'Execution log changed')
        require(outputs['inputs_unchanged'] and outputs['execution_exception'] is None
                and outputs['input_files_after']==inputs['input_files'],'Inputs mutated: '+label)
        require(set(outputs['declared_outputs'])=={'records.txt','trace.txt','param.log'},
                'Declared output set differs')
        for name,item in outputs['declared_outputs'].items():
            raw=read_bytes(prefix+name)
            require(item['kind']=='file' and sha(raw)==item['sha256'] and len(raw)==item['size'],
                    'Output bytes differ: '+label+'/'+name)
        for name,item in inputs['input_files'].items():
            if name=='resources/MadLoopParams.dat': raw=read_bytes(prefix+'MadLoopParams.dat')
            elif name.startswith('resources/'):
                expected=generated['SubProcesses/MadLoop5_resources/'+name[len('resources/'):]]
                require(item['sha256']==expected['sha256'] and item['size']==expected['size'],
                        'Resource differs from generated freeze: '+name)
                continue
            else: raw=read_bytes(prefix+safe(name))
            require(item['kind']=='file' and sha(raw)==item['sha256'] and len(raw)==item['size'],
                    'Input bytes differ: '+label+'/'+name)
    summary=load('attempt5/qualification-summary.json')
    require(summary['results_sha256']==sha(read_bytes('attempt5/results.json')) and summary['passed'],
            'Summary/result binding differs')
    require(summary['implementation_revision']==load('source-freeze.json')['implementation_revision'],
            'Summary implementation differs')
    artifact_summary={PurePosixPath(name).name:{'recorded_path':name,'sha256':item['sha256'],
                        'size':item['size']} for name,item in installed.items()}
    require(summary['installed_artifacts']==artifact_summary and summary['runtime_files']==runtime_sets[0],
            'Summary installed/runtime identity differs')
    references={'source-freeze.json','materialized-source.json','generated-source-freeze.json',
        'generation.json','cli-bootstrap-provenance.json','attempt4/source-freeze.json','attempt4/build.json',
        'attempt4/build-evidence.json','attempt5/source-freeze.json','attempt5/card-smoke-result.json',
        'attempt5/independent-review.json'}
    require(set(summary['source_and_build_evidence'])==references,'Summary omits source/build evidence')
    for name,value in summary['source_and_build_evidence'].items():
        require(sha(read_bytes(name))==value,'Summary source/build reference changed')
    review=load('attempt5/independent-review.json')
    require(summary['independent_review_path']=='attempt5/independent-review.json'
            and summary['independent_review_status']==review['status']=='PASS'
            and review['results_sha256']==summary['results_sha256']
            and review['analyzer_sha256']==ANALYZER_SHA,'Independent review binding differs')
    derived_workers={};parsed={};traces={}
    for label in analysis.LABELS:
        data=analysis.parse_records(ReadPath('attempt5/runs/'+label+'/records.txt'))
        trace=analysis.parse_trace(ReadPath('attempt5/runs/'+label+'/trace.txt'))
        parsed[label]=data;traces[label]=trace
        maximum=max(struct.unpack('>d',bytes.fromhex(value[-1]))[0]
                    for frame in data['amplitudes'].values() for value in frame['values'])
        derived_workers[label]={'warmup_calls':data['warm'],'measured_physical_calls':19,
            'all_physical_calls_including_warmup':len(data['amplitudes']),'tensor_key_calls':len(data['keys']),
            'status_counts_including_warmup':dict(Counter(str(frame['status']) for frame in data['amplitudes'].values())),
            'maximum_reported_accuracy':maximum,'trace_rows':len(trace['records']),
            'trace_flushes':len(trace['flushes']),
            'native_event_counts':dict(Counter(str(row['kind']) for row in trace['records'] if row['kind']!=1))}
    require(summary['workers']==derived_workers,'Summary worker counts/accuracy/status differ')
    require(summary['coverage']=={'physical_workers':4,'warmup_calls_total':8,
        'measured_physical_calls_total':76,'tensor_key_calls_total':96,
        'tensor_keys_per_worker_per_precision':12,'generated_rank_max':2,'generated_propagator_max':3},
        'Summary qualification scope differs')
    for label in ('observed-cache0','observed-cache-all'):
        evaluations=[row for action in analysis.PHYSICAL for row in traces[label]['groups'][action]
                     if row['kind'] in (6,7)]
        require(len(evaluations)==38 and all(row['kind']==6 for row in evaluations),
                'Physical native precision differs')
    require(summary['parity']=={'bitwise_coefficients_status_accuracy_inputs_ieee':True,
        'plain_observed_parity':True,'cache_enabled_disabled_parity':True,'physical_status':219,
        'physical_precision':'native double precision; no QP rescue occurred',
        'qp_scope':'Separate genuine-QP generated tensor diagnostics,12 calls per worker; mass,scale and test shift inputs are DP values promoted to QP.'},
        'Summary parity/precision scope differs')
    require(summary['processes']==[{'id':101,'process':'u u~ > z [virt=QCD]','prefix':'ML5_101_'},
            {'id':202,'process':'d d~ > z [virt=QCD]','prefix':'ML5_202_'}]
            and summary['model']=='loop_sm' and summary['backend']==9
            and summary['pole_mode']==0 and summary['compute_finite'] is True,
            'Summary process or mode scope differs')
    require(summary['card_smoke']=={'passed':True,'actual_reader_calls':3,
        'mass_sequence_GeV':[91.188,92,91.188],'context_generations':[-1,-1,-1],'native_exports':0},
        'Summary card smoke differs')
    expected_lifecycle={'context_generations_per_worker':2,'prefix_registrations_per_worker':4,
        'registered_topology_rank':[3,2],'automatic_clears_per_measured_physical_call':2,
        'evaluations_per_measured_physical_call':2,'primary_and_stability_check_stores_distinguished':True,
        'seeded_entry_before_automatic_clear':{'entries':1,'bytes':144},
        'seeded_entry_after_automatic_clear':{'entries':0,'bytes':0},
        'cache_enabled_nonempty_final_clears':19,'cache_disabled_nonempty_final_clears':0,
        'recreated_context_precedes_automatic_entry_clear':True,'key_cold_requests_per_precision':5,
        'key_repeats_per_precision':7,'disabled_qp_bypasses_per_worker':12,
        'all_native_evaluation_statuses_zero':True}
    require(summary['observed_lifecycle']==expected_lifecycle,'Summary lifecycle scope differs')
    return {'passed':True,'index_sha256':sha(index_bytes),'results_sha256':summary['results_sha256'],
            'physical_workers':4,'measured_physical_calls':76,'tensor_key_calls':96,
            'card_smoke_passed':True,'historical_symlinks_are_metadata_only':True,
            'summary':summary,'scope':'Portable retained-text/map replay; no live binary requalification.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',nargs='?',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    result=verify(args.root)
    print(json.dumps({key:value for key,value in result.items() if key!='summary'},indent=2,sort_keys=True))
