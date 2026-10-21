#!/usr/bin/env python3
"""Rehash copied workers and original build chain without running any engine."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import subprocess
from pathlib import Path
from build import HERE, ROOT, WORK, OLD, binding, sources, sha

def main():
    assert Path.cwd() == ROOT
    spec = importlib.util.spec_from_file_location('allocation_fresh_control',HERE / 'run-processes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = module.assets()
    frozen = json.loads((HERE / 'frozen-bindings.json').read_text())
    assert all(actual[k] == frozen[k] for k in actual)
    assert frozen['repetitions'] == {'qcd-3':21,'ew-2':7}
    assert frozen['loops'] == 1 and frozen['points'] == 100
    assert (HERE / 'run-processes.py').read_bytes() == (OLD / 'run-processes.py').read_bytes()
    assert (HERE / 'analyze-processes.py').read_bytes() == (OLD / 'analyze-processes.py').read_bytes()
    assert (HERE / 'comparison.py').read_bytes() == (OLD / 'comparison.py').read_bytes()
    assert (HERE / 'bound-support/ninjars_report_v2.py').read_bytes() == (OLD / 'bound-support/ninjars_report_v2.py').read_bytes()
    prep = json.loads((HERE / 'processes-final/preparation.json').read_text())
    assert prep['complete'] and not prep['amplitudes_evaluated'] and not prep['compilation_performed']
    assert binding(OLD / 'frozen-bindings.json') == prep['old_frozen_bindings']
    package = json.loads((ROOT / frozen['package']['path']).read_text())
    build = json.loads((ROOT / frozen['rust_build']['path']).read_text())
    assert sources(WORK / 'bridge') == sources(HERE / 'source') == build['allocation_sources']
    for name in ['baseline_archive','bare_rust_archive','allocation_archive','wrapper_object']:
        assert binding(ROOT / package[name]['path']) == package[name]
    baseline,rust,allocation = [ROOT / package[n]['path'] for n in ['baseline_archive','bare_rust_archive','allocation_archive']]
    members = lambda path:subprocess.check_output(['ar','t',str(path)],text=True).splitlines()
    assert members(baseline).count('mninjars.o') == 1
    assert members(allocation) == members(rust)+['mninjars.o']
    assert subprocess.check_output(['ar','p',str(baseline),'mninjars.o']) == subprocess.check_output(['ar','p',str(allocation),'mninjars.o'])
    assert (OLD / 'exact-build/baseline-fingerprint.json').read_bytes() == (OLD / 'exact-build/allocation-fingerprint.json').read_bytes()
    old_prep = json.loads((OLD / 'processes-final/preparation.json').read_text())
    for case,arms in prep['builds'].items():
        for arm,record in arms.items():
            old = old_prep['builds'][case][arm]
            assert record['executable']['sha256'] == old['executable']['sha256']
            assert record['generated_sources_sha256'] == old['generated_sources_sha256']
            assert record['libraries'] == old['libraries'] and record['runtime_metadata'] == old['runtime_metadata']
            assert record['points']['sha256'] == old['points']['sha256']
    print(json.dumps({'passed':True,'frozen_bindings':binding(HERE / 'frozen-bindings.json'),
        'copied_workers':4,'executable_bytes_identical':True,'compiler_fingerprints_identical':True,
        'controller_and_analyzer_bytes_identical':True,'old_historical_samples_reused':False,
        'numerical_execution':False},indent=2),flush=True)

if __name__ == '__main__':
    main()
