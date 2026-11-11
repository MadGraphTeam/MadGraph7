"""Authenticate retained text and replay the pure universal-pole calculation."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def require(ok,message):
    if not ok:raise ValueError(message)

def main():
    index=json.loads((HERE/'index.json').read_bytes())
    paths=list(HERE.rglob('*'))
    require(not any(p.is_symlink() for p in paths),'Symlink in packet')
    require({str(p.relative_to(HERE)) for p in paths if p.is_file()}==set(index['files'])|{'index.json'},'File set changed')
    for name,pin in index['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe name')
        raw=(HERE/p).read_bytes();require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Changed '+name)
    spec=importlib.util.spec_from_file_location('pure_universal_pole_calculation',HERE/'calculate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.calculate()
    require(result==json.loads((HERE/'result.json').read_bytes()),'Recalculation differs')
    require(result['native_executions']==0 and not result['independent_full_amplitude_accuracy_qualified'],'Scope differs')
    print(json.dumps({'passed':True,'observation_rows':len(result['comparisons']),'native_executions':0,
                     'finite_single_truth_qualified':False,'checked_stability_qualified':False},sort_keys=True))

if __name__=='__main__':main()
