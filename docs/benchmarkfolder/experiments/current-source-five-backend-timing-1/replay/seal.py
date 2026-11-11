"""Seal only a terminal timing directory, preserving the admitted source bytes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

def seal(root,report,source_sha,prepared_sha):
    root=Path(root).resolve();report=Path(report).resolve();here=Path(__file__).resolve().parent
    require(report.is_relative_to(root/'docs/benchmarkfolder/experiments'),'Wrong report location')
    require(not (report/'index.json').exists(),'Closed index already exists; never replace an attempt')
    require(sha(report/'source-index.json')==source_sha and sha(report/'specification.json')==prepared_sha,'Wrong source/prepared admission')
    source=read(report/'source-index.json');spec=read(report/'specification.json');summary=read(report/'summary.json')
    require(summary['source_sha256']==source_sha and summary['prepared_sha256']==prepared_sha,'Wrong terminal summary')
    require([v['case'] for v in summary['cases']]==spec['case_ids'],'Incomplete terminal case list')
    for row in summary['cases']:
        if row['status']=='not_run_budget':continue
        folder=report/'cases'/row['case']
        require(read(folder/'completion.json')==row and row['cleanup']['survivors']==[],'Missing actual cleanup/closure')
        if row['status']=='complete':require(row['returncode']==0 and sha(folder/'result.json')==row['result_sha256'],'Incomplete successful case')
    prepared=read(report/'preparation.json')
    for case in spec['case_ids']:
        for backend in spec['backends']:
            lane=report/'cases'/case/str(backend);input_path=lane/'candidate-inputs.json'
            if not input_path.is_file():continue
            inputs=read(input_path);output=Path(prepared['clones'][case+'/'+str(backend)]['private_output'])
            for card_source,destination,checksum in [(output/'Cards/param_card.dat',lane/'cards/param_card.dat',inputs['param_card_sha256']),
                (Path(inputs['resolved_parameter_card']),lane/'cards/resolved-param_card.dat',inputs['param_card_sha256']),
                *[(output/name,lane/'cards/numerical'/name,checksum) for name,checksum in inputs['numerical_cards'].items()]]:
                require(card_source.is_file() and sha(card_source)==checksum,'Retained card no longer matches its observed hash')
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():require(sha(destination)==checksum,'Existing retained card differs')
                else:shutil.copy2(card_source,destination)
    for name,checksum in source['support'].items():
        if name.startswith('tools/'):continue
        origin=root/name;require(sha(origin)==checksum,'Support source changed')
        target=report/'support'/name;target.parent.mkdir(parents=True,exist_ok=True);require(not target.exists(),'Support payload already exists')
        shutil.copy2(origin,target)
    (report/'replay').mkdir()
    for name in ('verify.py','seal.py'):shutil.copy2(here/name,report/'replay'/name)
    files={}
    for path in sorted(report.rglob('*')):
        require(not path.is_symlink(),'No symlinks in portable capsule')
        if path.is_file():
            raw=path.read_bytes();require(not raw.startswith(b'\x7fELF') and b'\x00' not in raw,'Only typed text evidence may be published')
            raw.decode('utf-8');files[str(path.relative_to(report))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    index={'schema':'madloop-current-five-backend-closed-v1','source_sha256':source_sha,'prepared_sha256':prepared_sha,
           'case_ids':spec['case_ids'],'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),
           'native_executions_by_sealer':0,'performance_acceptance':False,'timing_scope':'descriptive shared-host; no CPU reservation or confidence interval'}
    trial=Path(tempfile.mkdtemp(prefix='timing-seal-trial-',dir=root/'.tirust-build'))/'capsule'
    shutil.copytree(report,trial)
    with (trial/'index.json').open('x') as stream:json.dump(index,stream,indent=2,sort_keys=True);stream.write('\n')
    checksum=sha(trial/'index.json');module_spec=importlib.util.spec_from_file_location('closed_timing_verify',report/'replay/verify.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    result=module.verify(trial/'index.json',checksum)
    with (report/'index.json').open('xb') as stream:stream.write((trial/'index.json').read_bytes())
    return {'index_sha256':checksum,'payloads':len(files),'bytes':index['total_bytes'],
            'cases':[{k:v for k,v in case.items() if k!='summary'} for case in result['cases']],
            'native_executions':0,'performance_acceptance':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);parser.add_argument('--source',required=True);parser.add_argument('--prepared',required=True)
    args=parser.parse_args();print(json.dumps(seal(Path.cwd(),args.report,args.source,args.prepared),indent=2,sort_keys=True))
