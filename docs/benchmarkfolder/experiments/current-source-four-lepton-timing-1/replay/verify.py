"""Portable pure replay of a closed descriptive timing capsule; no native code."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=previous
    return module
def inside(root,name):
    path=Path(name);require(not path.is_absolute() and '..' not in path.parts,'Unsafe payload path')
    target=root/path;require(target.resolve().is_relative_to(root.resolve()) and not target.is_symlink(),'Escaping payload')
    return target

def verify(index_path,expected_sha):
    index_path=Path(index_path).resolve();base=index_path.parent
    require(sha(index_path)==expected_sha,'Wrong closed index SHA')
    index=read(index_path);require(index['schema']=='madloop-current-five-backend-closed-v1','Wrong closed index schema')
    require({str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}==set(index['files'])|{'index.json'},'Closed inventory is not exact')
    for name,row in index['files'].items():
        path=inside(base,name);require(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Payload mismatch: '+name)
    source=read(base/'source-index.json');spec=read(base/'specification.json');prepared=read(base/'preparation.json')
    require(sha(base/'source-index.json')==index['source_sha256']==spec['source_index_sha256'],'Wrong controller source')
    require(sha(base/'specification.json')==index['prepared_sha256'],'Wrong prepared specification')
    for name,row in source['files'].items():require(sha(base/'source'/name)==row['sha256'],'Changed sealed source')
    for name,value in spec['frozen_files'].items():require(sha(inside(base,name))==value,'Prepared evidence drift')
    for name,value in source['support'].items():
        if name.startswith('tools/'):require(sha(base/'protocol'/Path(name).name)==value,'Protocol source drift')
        else:require(sha(base/'support'/name)==value,'Ownership/loader source drift')
    worker=load('timing_capsule_worker',base/'protocol/tirust_benchmark_worker.py')
    previous=sys.modules.get('tirust_benchmark_worker');sys.modules['tirust_benchmark_worker']=worker
    try:warmup=load('timing_capsule_warmup',base/'protocol/tirust_benchmark_warmup.py')
    finally:
        if previous is None:sys.modules.pop('tirust_benchmark_worker',None)
        else:sys.modules['tirust_benchmark_worker']=previous
    analyze=load('timing_capsule_analyze',base/'source/analyze.py')
    tree=ast.parse((base/'protocol/tirust_benchmark.py').read_text())
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='compare_rows']
    require(len(tree.body)==1,'Missing retained comparator');scope={};exec(compile(tree,'retained comparator','exec'),scope)
    compare=scope['compare_rows'];global_summary=read(base/'summary.json');cases=[]
    fresh_reference=spec['schema']=='madloop-current-four-lepton-five-backend-descriptive-v1'
    require(fresh_reference or spec['schema']=='madloop-current-five-backend-descriptive-v1','Unknown timing preparation scope')
    require(global_summary['source_sha256']==index['source_sha256'] and global_summary['prepared_sha256']==index['prepared_sha256'],'Summary admission differs')
    require([row['case'] for row in global_summary['cases']]==spec['case_ids'],'Missing/reordered case closure')
    require(spec['points_per_batch']==100 and spec['cycles']==analyze.schedule() and spec['cpu_reservation'] is False,'Changed timing scope')
    for completion in global_summary['cases']:
        case_id=completion['case'];folder=base/'cases'/case_id
        if completion['status']!='complete':
            cases.append({'case_id':case_id,'status':completion['status'],'timings':None});continue
        require(read(folder/'completion.json')==completion and completion['returncode']==0
                and completion['cleanup']['survivors']==[],'Unclosed controller/worker')
        require(sha(folder/'result.json')==completion['result_sha256'],'Case result mismatch')
        result=read(folder/'result.json');summary=result['summary'];original=prepared['originals'][case_id]
        require(result['status']=='complete' and result['case']==original['case'] and result['performance_acceptance'] is False,'Wrong case/result scope')
        if fresh_reference:
            origin=read(base/'origin-preparation.json');origin_root=base/'origin-preparation'
            for name,checksum in original['shared']['closed_preparation_inputs'].items():require(origin['files'].get(name)==checksum,'Closed preparation source pin missing')
            for name,checksum in origin['files'].items():require(sha(inside(origin_root,name))==checksum,'Closed preparation payload changed')
            original_output=inside(origin_root,origin['source_output'])
            generated=origin['generated_sources'];compiled=read(original_output/'compilation.json')
            for name in generated:require(sha(inside(original_output,name))==compiled['sources'][name],'Topology source not bound to compilation')
            rank=int(re.search(r'parameter\(OVERALLMAXRANK\s*=\s*(\d+)\)',(original_output/generated[0]).read_text(),re.I).group(1))
            calls=list(map(int,re.findall(r'\bCALL\s+ML5_0_LOOP_(\d+)\s*\(',(original_output/generated[1]).read_text(),re.I)))
            topology={'overall_max_rank':rank,'max_denominators':max(calls),'loop_call_groups':len(calls),'denominator_call_counts':{str(n):calls.count(n) for n in sorted(set(calls))}}
            require(topology==origin['topology'] and origin['native_executions']==0,'Topology/preparation scope mismatch')
            declaration=read(folder/'fresh-cuttools-reference.json');reference=declaration['observation']
            require(declaration['origin']=='backend1-full100-learning' and declaration['point_sha256']==original['points_sha256']
                    and declaration['executable_sha256']==original['executable_sha256'] and declaration['independent_accuracy_claim'] is False,'Fresh reference source mismatch')
            require(warmup.parse_warmup(folder/'1/candidate-initialization',[1],100,read(folder/'1/ready.json'))==reference,'Fresh CutTools raw reference differs')
        else:
            reference=read(base/'reference'/case_id/'raw/observation.json')
            reference_raw=base/'reference'/case_id/'raw'
            require(warmup.parse_warmup(reference_raw,[1],100,read(reference_raw/'ready.json'))==reference,'CutTools raw reference differs')
        cycles=analyze.schedule();cycles=[{**row,'observations':{}} for row in cycles]
        ready={};calibration={};checks={};common=None
        for backend in analyze.BACKENDS:
            key=str(backend);lane=folder/key;before=read(lane/'candidate-inputs.json');after=read(lane/'inputs-after.json')
            if fresh_reference:require(before['generation']==read(original_output/'generation.json') and before['compilation']==compiled,'New four-lepton build receipt differs')
            require(before['generation']['case']==original['case'] and before['generation']['source_identity']['files_sha256']==original['shared']['source_identity']['files_sha256'],'Lane source/case mismatch')
            compilation=before['compilation']
            require(compilation['executable_sha256']==original['executable_sha256']==after['executable_sha256']
                    and compilation['sources']==original['generated_sources'] and compilation['numerical_build_from_sources'] is True,'Lane build mismatch')
            require(before['points_sha256']==original['points_sha256']==after['points_sha256']
                    and before['points_metadata_sha256']==original['points_metadata_sha256']
                    and before['param_card_sha256']==original['physical_card_sha256']
                    and before['numerical_cards']==after['numerical_cards'],'Lane points/cards mismatch')
            require(sha(lane/'cards/param_card.dat')==before['param_card_sha256']==sha(lane/'cards/resolved-param_card.dat'),'Physical card payload mismatch')
            for name,checksum in before['numerical_cards'].items():require(sha(inside(lane/'cards/numerical',name))==checksum,'Numerical card payload mismatch')
            for field in ('libraries','archives','metadata'):
                require(before['runtime'][field]==original['runtime'][field]==after['runtime'][field],'Runtime receipt mismatch')
            settings=before['settings'];require(settings['mlreductionlib']==key and settings['looppolemode']==0 and settings['mlstabthres']==1e-6
                and settings['colliercomputeuvpoles'] is True and settings['colliercomputeirpoles'] is True and settings['collierusecacheforpoles'] is False,'Reduction policy mismatch')
            selected={k:v for k,v in settings.items() if k!='mlreductionlib'}
            if common is None:common=selected
            require(selected==common,'Cards differ beyond selected backend')
            actual=spec['actual_runtime'][case_id]
            runtime=read(lane/'actual-runtime-before.json')
            require(runtime['files']==actual['files'] and runtime['loader']==actual['actual_interpreter']
                    and runtime['loader_sha256']==actual['files'][actual['actual_interpreter']],'Actual PT provider mismatch')
            clone=prepared['clones'][case_id+'/'+key];exe=str(Path(clone['private_output'])/'SubProcesses'/clone['subprocess']/'check')
            listing=(lane/'loader-before/actual-pt-interp-runtime.txt').read_text()
            require(runtime['listing']==listing,'Raw PT listing differs from receipt')
            command=read(lane/'loader-before/actual-pt-interp-runtime-command.json')
            require(command['argv']==[actual['actual_interpreter'],'--list',exe] and command['executable_sha256']==original['executable_sha256']
                    and command['invokes_fortran_main'] is False,'Raw PT command differs')
            aliases={name:row['resolved_path'] for name,row in original['runtime_files']['libraries'].items()}
            aliases.update({name:name for name in actual['files']})
            listed=set()
            for line in listing.splitlines():
                line=line.strip()
                if not line or re.fullmatch(r'linux-vdso\.so\.1\s+\(0x[0-9a-fA-F]+\)',line):continue
                match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line)
                require(match is not None and match.group(1) in aliases,'Unbound raw PT provider path')
                listed.add(aliases[match.group(1)])
            require(listed|{actual['actual_interpreter']}==set(actual['files']),'Raw PT provider set differs')
            required={**actual['files'],exe:original['executable_sha256']};allowed={**required,**spec['allowed_extra_mappings']}
            for name in ('maps-before.json','maps-after.json'):
                record=read(lane/name);maps=record['files'];mapped=set()
                for line in record['maps'].splitlines():
                    fields=line.split(None,5)
                    if len(fields)==6 and fields[-1].startswith('/'):
                        require(not fields[-1].endswith(' (deleted)'),'Deleted raw mapping');mapped.add(fields[-1])
                require(mapped==set(maps),'Raw maps differ from file receipt')
                require(all(maps.get(p)==v for p,v in required.items()) and all(allowed.get(p)==v for p,v in maps.items()),'Actual mapped provider mismatch')
            ready[key]=read(lane/'ready.json');learning=warmup.parse_warmup(lane/'candidate-initialization',[backend],100,ready[key])
            require(learning==read(lane/'learning.json'),'Learning replay mismatch')
            checks[key]=[({'passed':True,'ordinary_target_passed':True,'role':'fresh-reference-acquisition','independent_accuracy_claim':False}
                          if fresh_reference and backend==1 else compare(reference['rows'],learning['rows'],3e-6))]
            require(checks[key][0]==read(lane/'learning-comparison.json') and checks[key][0]['passed'],'Learning comparator mismatch/failure')
            calibration[key]=read(lane/'calibration.json')
            def observation(tag,loops):
                raw=lane/'raw'/tag;value=worker.parse_outputs(raw,[backend],100,loops)
                require(value==read(raw/'observation.json'),'Timed raw replay mismatch')
                check=compare(reference['rows'],value['rows'],3e-6)
                require(check==read(raw/'comparison.json') and check['passed'],'Timed numerical gate mismatch')
                checks[key].append(check);return value
            require(observation('calibration-0',1)==calibration[key]['initial'],'First calibration differs')
            require(observation('calibration-1',calibration[key]['sustained_loops'])==calibration[key]['sustained'],'Sustained calibration differs')
            count=max(1,math.ceil(2./calibration[key]['initial']['seconds']))
            loops=max(1,math.ceil(count*2./calibration[key]['sustained']['seconds']))
            require(calibration[key]['target_seconds']==2. and calibration[key]['sustained_loops']==count
                    and calibration[key]['loops']==loops,'Calibration formula differs from admitted controller')
            for row in cycles:
                loops=calibration[key]['loops'];row['observations'][key]={'loops':loops,'observation':observation('cycle-'+str(row['cycle']),loops)}
            calibration[key].update(numerical_regression_pass=all(c['passed'] for c in checks[key]),ordinary_target_passed=all(c['ordinary_target_passed'] for c in checks[key]))
        for row in cycles:require(row==read(folder/('cycle-'+str(row['cycle'])+'.json')),'Cycle evidence differs')
        recomputed=analyze.summarize_case(case_id,cycles,ready,calibration)
        require(recomputed==summary and checks==result['numerical_checks'],'Case summary differs from raw replay')
        cases.append({'case_id':case_id,'status':'complete','summary':recomputed,
                      **({'loop_topology':topology} if fresh_reference else {}),
                      'reference_role':'fresh same-run CutTools learning; no independent accuracy claim' if fresh_reference else 'retained same-source CutTools full100 qualification'})
    return {'schema':'madloop-current-five-backend-pure-replay-v1','index_sha256':expected_sha,'payloads_verified':len(index['files']),
            'cases':cases,'native_executions':0,'performance_acceptance':False,'confidence_intervals':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('index');parser.add_argument('sha256');args=parser.parse_args()
    print(json.dumps(verify(args.index,args.sha256),indent=2,sort_keys=True,allow_nan=False))
