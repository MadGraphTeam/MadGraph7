"""Capture completed text inputs; analyze them without native execution."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    assert Path.cwd().resolve()==ROOT and not (HERE/'analysis.json').exists()
    old=ROOT/'.tirust-build/qp-mass-general-implementation/combined-qualification/amplitude/runs/guarded-qp-8'
    new=ROOT/'.tirust-build/adaptive-qp-physical-replay-proposal-1/attempt1/case'
    prepared=json.loads((old.parents[2]/'prepared.json').read_text())
    job=next(j for stage in prepared['stages'] for j in stage['jobs'] if j['name']=='guarded-qp-8')
    files=[str(Path(name).relative_to(old)) for name in job['input_sha256']]+job['expected_outputs']
    assert len(files)==12
    blobs={};inventory={}
    def retain(label,path):
        data=path.read_bytes();blobs[label]=data.decode('utf-8')
        inventory[label]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'origin':str(path.relative_to(ROOT))}
    for label,base in [('guarded8',old),('adaptive9',new)]:
        for name in files:retain(label+'/'+name,base/name)
    parser=ROOT/'docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/analyze.py'
    retain('parser.py',parser);retain('adapters.f',old.parents[1]/'src/adapters.f')
    (HERE/'text-inputs.json.gz').write_bytes(gzip.compress(json.dumps(blobs,sort_keys=True).encode(),compresslevel=9,mtime=0))
    (HERE/'text-inventory.json').write_text(json.dumps(inventory,sort_keys=True,indent=2)+'\n')
    result=load('adaptive_independent8_audit_source',HERE/'audit.py').analyze(blobs,load('adaptive_independent8_frozen_parser',parser))
    assert result['physical_input_and_Born_parity_passed']
    (HERE/'analysis.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'physical_input_and_Born_parity_passed':True,'native_input_scale_differences':200,
                      'native_executed':False,'accuracy_or_truth_qualified':False},sort_keys=True))


if __name__=='__main__':main()
