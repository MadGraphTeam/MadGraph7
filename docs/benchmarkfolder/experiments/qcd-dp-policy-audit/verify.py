#!/usr/bin/env python3
"""Read-only inventory/source checks and replay of retained text classification."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    entries=json.loads((HERE/'index.json').read_text())['files'];paths=list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()}==set(entries)|{'index.json'}
    for name,entry in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=HERE/name;assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
        assert not p.read_bytes().startswith(b'\x7fELF') and p.suffix!='.pyc'
    bindings=json.loads((HERE/'source-bindings.json').read_text())
    assert not any(bindings[k] for k in ('production_changed','new_native_run','new_build'))
    if '--live' in sys.argv:
        for name,entry in bindings['sources'].items():
            p=ROOT/name;assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes'],name
    spec=importlib.util.spec_from_file_location('authenticated_qcd_dp_classification',HERE/'classify.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actual=module.derive();assert actual==json.loads((HERE/'classification.json').read_text())
    assert actual['method_choice_rows']==454 and actual['method_choice_successes']==320
    assert actual['final_refusals']==134 and actual['partial_dd_static_shape_candidate_refusals']==102
    assert actual['top_subset_excluded_refusals']==32 and actual['actual_whole_lane_tier_events']==0
    print(json.dumps({'verification':'pass','portable_files':len(entries),'source_bindings':len(bindings['sources']),
                      'method_choice_rows':454,'static_partial_candidates':102,'native_executed':False}))
if __name__=='__main__':main()
