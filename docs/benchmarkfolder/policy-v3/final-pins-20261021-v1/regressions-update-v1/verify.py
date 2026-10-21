#!/usr/bin/env python3
"""Read-only verification of fresh --update tests and retained generation proof."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def payloads(directory,manifest):
    for name,record in manifest['payloads'].items():
        path=(directory/name).resolve()
        assert path.is_relative_to(directory)
        assert path.stat().st_size==record['bytes'],name
        assert sha(path)==record['sha256'],name

def footer(log,count,skip):
    text=log.read_text()
    matches=re.findall(r'Ran (\d+) tests? in ([0-9.]+)s',text)
    assert len(matches)==1 and int(matches[0][0])==count
    expected='OK' if not skip else f'OK (skipped={skip})'
    assert expected in text.splitlines()

def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert manifest['schema']=='ninjars-update-software-manifest-v1'
    payloads(HERE,manifest)
    summary=json.loads((HERE/'summary.json').read_text())
    source=json.loads(gzip.decompress((HERE/'fresh-source-hashes.json.gz').read_bytes()))
    correction=json.loads((HERE/'cfg-test-equivalence.json').read_text())
    special='vendor/ninjars/src/upstream_master_tests.rs'
    assert correction['path']==special
    assert correction['original_inventoried_sha256']=='1addb2b3676f25035477dc3ec275abeea27f7ecbb90dcee1bc72c8b786601cf4'
    assert correction['current_sha256']=='48eb9e0d7c7a23bd0f9039be3e986ea2322a4c31d236073a322ca4d71b57f456'
    assert correction['new_47_test_execution_claimed'] is False
    assert correction['production_and_47_test_sources_unchanged']
    for name,digest in correction['preserved_original_artifacts'].items():
        assert sha(HERE/name)==digest,name
    lib=ROOT/correction['cfg_test_declaration']['path']
    assert sha(lib)==correction['cfg_test_declaration']['sha256']
    assert len(re.findall(r'#\[cfg\(test\)\]\s*mod upstream_master_tests;',lib.read_text()))==1
    for name,digest in source['repository_files'].items():
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT)
        if name==special:
            assert digest==correction['original_inventoried_sha256']
            assert sha(path)==correction['current_sha256']
        else:
            assert sha(path)==digest,name
    freeze=json.loads((HERE/'cfg-test-source-freeze-v4.json').read_text())
    assert freeze['files']['src/upstream_master_tests.rs']==correction['current_sha256']
    assert sha(HERE/'cfg-test-source-freeze-v4.json')==correction['current_source_freeze']['sha256']
    controls=correction['separate_rust_controls']
    assert [(c['exit_code'],c['passed'],c['failed'],c['ignored']) for c in controls]==[(0,24,0,7),(101,0,1,0)]
    assert all(not c['included_in_47_software_tests'] and not c['numerical_gate_passed'] for c in controls)
    assert sha(HERE/'recorded-runner.py')==source['recorded_runner_sha256']
    fresh=json.loads((HERE/'focused-result.json').read_text())
    assert fresh['passed'] and fresh['source_unchanged']
    assert not fresh['failures'] and not fresh['errors'] and not fresh['skipped']
    assert len(fresh['test_ids'])==fresh['tests']==summary['fresh_tests_run']==47
    footer(HERE/'focused.log',47,0)
    prior=HERE.parent/'regressions'
    assert sha(prior/'manifest.json')==summary['prior_capsule_manifest_sha256']
    payloads(prior,json.loads((prior/'manifest.json').read_text()))
    for stem,count,skip in (('exporters-2',2,0),('fks-1',19,1)):
        result=json.loads((prior/f'{stem}-result.json').read_text())
        assert result['suite_successful'] and result['relevant_source_unchanged']
        assert result['tests']==count and len(result['skipped'])==skip
        assert not result['failures'] and not result['errors']
        footer(prior/f'{stem}.log',count,skip)
    spec=importlib.util.spec_from_file_location('ninjars_update_reuse',HERE/'verify_reuse.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check()==json.loads((HERE/'reuse-review.json').read_text())
    assert summary['software_passed']
    cli=json.loads((HERE/'real-cli-validation.json').read_text())
    assert cli['passed'] and cli['actual_check_install_used']
    assert cli['observed_options']['options_for_HEPToolsInstaller']==['--update','--jobs=3']
    assert cli['interface_sha256']==sha(ROOT/'madgraph/interface/madgraph_interface.py')
    assert summary['fresh_tests_passed']==47
    assert (summary['reused_tests_observed'],summary['reused_tests_passed'],
            summary['reused_tests_skipped'])==(21,20,1)
    assert summary['coverage_cases']==68
    assert summary['actual_reduction_engine_evaluations']==0
    assert summary['correctness_gate_passed'] is False
    assert summary['performance_gate_passed'] is False
    print(json.dumps({'verified':True,'fresh_passed':47,'reused_passed':20,
        'reused_existing_skip':1,'actual_reduction_engine_evaluations':0},sort_keys=True))

if __name__=='__main__':
    main()
