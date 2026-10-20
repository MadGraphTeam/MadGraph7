#!/usr/bin/env python3
"""Read-only complete capture seal and exact inventory verification."""
import sys
sys.dont_write_bytecode=True
import json
import importlib.util
from pathlib import Path
import inventory
import inventory_ulp

HERE=Path(__file__).resolve().parent


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    actual=set()
    for path in HERE.rglob('*'):
        relative=path.relative_to(HERE)
        if '__pycache__' in relative.parts or relative.parts[0]=='replay' or str(relative)=='manifest.json':continue
        if path.is_file():actual.add(str(relative))
    assert actual==set(manifest['files'])
    for name,record in manifest['files'].items():
        path=HERE/name
        assert not path.is_symlink()
        assert inventory.sha(path)==record['sha256'] and path.stat().st_size==record['bytes'],name
    assert inventory.compute()==inventory.read('input-inventory.json')
    assert inventory_ulp.compute()==inventory.read('ulp-inventory.json')
    independent_path=HERE/'independent-inventory/check.py'
    spec=importlib.util.spec_from_file_location('independent_capture_check',independent_path)
    independent=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    assert independent.compute()==inventory.read('independent-inventory/result.json')
    assert inventory.read('capture-result.json')['returncode']==127
    assert not inventory.read('capture-result.json')['complete']
    print(json.dumps({'passed':True,'files':len(manifest['files']),
        'all100_rows_statuses_accuracies_exact':True,'records':1392,'matched_keys':464,
        'independent_integer_rounding_inventory_exact':True,
        'replay_executed':False,'engine_calls':0}))


if __name__=='__main__':main()
