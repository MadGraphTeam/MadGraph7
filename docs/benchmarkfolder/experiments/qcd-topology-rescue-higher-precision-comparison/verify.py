#!/usr/bin/env python3
"""Check all provisional comparisons offline; no executable or DSO is loaded."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location('frozen_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    entries=json.loads((HERE/'evidence-index.json').read_text())['files']
    paths=list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()}==set(entries)|{'evidence-index.json'}
    for name,entry in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=HERE/name;raw=p.read_bytes()
        assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
        assert not raw.startswith(b'\x7fELF') and p.suffix not in ('.pyc','.o','.mod','.a','.so'),name
    directory=HERE.with_name('qcd-topology-rescue-higher-precision-native')/'attempt1'
    reports=[]
    for script,result in (('compare','provisional1.json'),('compare2','provisional2.json')):
        observed=load(script).compute(directory,True)
        assert observed==json.loads((HERE/result).read_text()),result
        reports.append(observed)
    assert reports[0]['all_common_inputs_and_scale_semantics_exact'] is False
    assert reports[1]['all_common_inputs_and_scale_semantics_exact'] is True
    assert reports[0]['host']==reports[1]['host']
    summary=load('summarize').compute(HERE/'provisional2.json')
    assert summary==json.loads((HERE/'provisional2-summary.json').read_text())
    assert summary['all_696_complex_slots_included'] and summary['discarded_component_count']==0
    assert len(summary['flagged_slots'])==46
    print(json.dumps({'verification':'pass','portable_files':len(entries),'complex_slots':696,
                      'discarded_components':0,'flagged_relative_slots':46,'native_executed':False,
                      'runtime_capture_complete':False,'physics_campaign_accepted':False},sort_keys=True))


if __name__=='__main__':
    main()
