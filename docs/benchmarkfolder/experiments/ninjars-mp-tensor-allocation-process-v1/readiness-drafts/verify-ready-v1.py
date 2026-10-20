#!/usr/bin/env python3
"""Read-only source/archive/worker consistency checks; no numerical invocation."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from pathlib import Path
import subprocess
from build import HERE, ROOT, WORK, binding, sha, sources

def main():
    assert Path.cwd() == ROOT
    package = json.loads((HERE / 'package-exact.json').read_text())
    build = json.loads((HERE / 'exact-build/result.json').read_text())
    assert package['build'] == binding(HERE / 'exact-build/result.json')
    assert build['status'] == 'compiled-not-executed'
    assert binding(ROOT / package['baseline_archive']['path']) == package['baseline_archive'] == build['baseline_archive']
    assert binding(ROOT / package['bare_rust_archive']['path']) == package['bare_rust_archive'] == build['allocation_archive']
    assert binding(ROOT / package['allocation_archive']['path']) == package['allocation_archive']
    def members(path):
        return subprocess.check_output(['ar','t',str(path)],text=True).splitlines()
    baseline,rust,allocation = [ROOT / package[n]['path'] for n in ['baseline_archive','bare_rust_archive','allocation_archive']]
    assert members(baseline).count('mninjars.o') == 1
    assert members(allocation) == members(rust)+['mninjars.o']
    original_object = subprocess.check_output(['ar','p',str(baseline),'mninjars.o'])
    assert original_object == subprocess.check_output(['ar','p',str(allocation),'mninjars.o'])
    assert original_object == (ROOT / package['wrapper_object']['path']).read_bytes()
    assert binding(ROOT / package['wrapper_object']['path']) == package['wrapper_object']
    assert sources(WORK / 'bridge') == sources(HERE / 'source') == build['allocation_sources']
    differing = [n for n in build['original_sources'] if build['original_sources'][n] != build['allocation_sources'][n]]
    assert differing == ['src/tensor.rs']
    assert build['allocation_sources']['src/tensor.rs'] == '240bbc1f9c8937393227c16a6ce4887ee68524a3f021a8a93656151159445d70'
    a,b = [json.loads((HERE / 'exact-build' / n).read_text()) for n in ['baseline-fingerprint.json','allocation-fingerprint.json']]
    assert a == b
    assert a['profile'] == 1760205775310677080 and a['features'] == '[]' and a['rustflags'] == []
    assert build['effective_flag_environment'] == {}
    spec = importlib.util.spec_from_file_location('allocation_process_control', HERE / 'run-processes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = module.assets()
    frozen = json.loads((HERE / 'frozen-bindings.json').read_text())
    assert all(actual[k] == frozen[k] for k in actual)
    for case,arms in actual['builds'].items():
        x,y = [arms[n] for n in ['original','allocation']]
        for n in ['generated_sources_sha256','runtime_metadata','header_sha256','module_source_sha256','link_inputs','shared_numerical_settings']:
            assert x[n] == y[n],(case,n)
        assert set(x['libraries']) == set(y['libraries'])
        names = [n for n in x['libraries'] if x['libraries'][n] != y['libraries'][n]]
        assert names == ['ninjars_include/libninjars.a','ninjars_lib/libninjars.a']
        for n in names:
            assert x['libraries'][n] == package['baseline_archive']['sha256']
            assert y['libraries'][n] == package['allocation_archive']['sha256']
    print(json.dumps({'passed':True,'frozen_bindings':binding(HERE / 'frozen-bindings.json'),
        'baseline':package['baseline_archive'],'allocation':package['allocation_archive'],
        'identical_compiler_fingerprints':True,'identical_fortran_wrapper_object':True,
        'numerical_execution':False}),indent=2))

if __name__ == '__main__':
    main()
