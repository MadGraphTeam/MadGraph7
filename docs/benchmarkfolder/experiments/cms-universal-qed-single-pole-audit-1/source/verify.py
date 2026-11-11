"""Portable text/bit replay of the universal QED single-pole audit."""
import gzip
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
    require({str(p.relative_to(HERE)) for p in paths if p.is_file()}==set(index['files'])|{'index.json'},'Changed file set')
    for name,pin in index['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe name')
        raw=(HERE/p).read_bytes();require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Changed '+name)
        decoded=gzip.decompress(raw) if name.endswith('.gz') else raw
        decoded.decode('utf-8');require(b'\0' not in decoded,'Nontext payload')
    pins=json.loads((HERE/'source-pins.json').read_bytes())
    for entry in json.loads((HERE/'source-excerpts.json').read_bytes()).values():
        require(entry['path'] in pins,'Unpinned excerpt')
        for section in entry['ranges']:
            require(hashlib.sha256(section['text'].encode()).hexdigest()==section['sha256'],'Changed source excerpt')
    spec=importlib.util.spec_from_file_location('pure_single_pole_calculation',HERE/'calculate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.calculate()
    require(result==json.loads((HERE/'result.json').read_bytes()),'Calculation replay differs')
    require([r['trace_parent_stage_passed'] for r in result['points']]==[True,False],'Old trace failures were rewritten')
    require(not result['finite_truth_qualified'] and not result['full_amplitude_truth_qualified'] and
            not result['checked_stability_qualified'] and result['native_executions']==0,'Scope changed')
    print(json.dumps({'passed':True,'points':len(result['points']),
                     'compared_rows':sum(len(p['comparisons']) for p in result['points']),
                     'old_stress_failure_preserved':True,'native_executions':0,
                     'finite_or_full_truth_qualified':False,'checked_stability_qualified':False},sort_keys=True))

if __name__=='__main__':main()
