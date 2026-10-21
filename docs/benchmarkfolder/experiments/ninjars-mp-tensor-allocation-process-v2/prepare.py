#!/usr/bin/env python3
"""Copy the previously audited workers without recompilation or evaluation."""
import sys
sys.dont_write_bytecode = True
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from build import HERE, ROOT, WORK, OLD, binding, sha, sources, write
sys.path[:0] = [str(ROOT),str(ROOT / 'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers

def main():
    assert Path.cwd() == ROOT
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip() == 'madloop_ninjars'
    recovery = json.loads((HERE / 'recovery-result.json').read_text())
    assert recovery['status'] == 'retired-without-resumption'
    assert all(s in ['absent','Z'] for s in recovery['after_states'].values())
    old_freeze = OLD / 'frozen-bindings.json'
    assert sha(old_freeze) == '26b7602867394ef5f9b28a0900d8a518afd1e8e7668075e4af52f36af6a6d4c3'
    frozen = json.loads(old_freeze.read_text())
    old_prep_path = ROOT / frozen['preparation']['path']
    assert binding(old_prep_path) == frozen['preparation']
    old_prep = json.loads(old_prep_path.read_text())
    assert old_prep['complete'] and not old_prep['amplitudes_evaluated']
    assert workers.protocol_descriptor() == frozen['protocol']
    for rec in frozen['support_tools'].values():
        assert binding(ROOT / rec['path']) == rec
    original_source = OLD / 'source'
    assert sources(original_source) == frozen['bridge_sources']
    assert not WORK.exists()
    WORK.mkdir(parents=True)
    shutil.copytree(original_source,WORK / 'bridge')
    shutil.copytree(original_source,HERE / 'source')
    shutil.copytree(OLD / 'bound-support',HERE / 'bound-support')
    for name in ['comparison.py','comparison-contract.json','analyze-processes.py','run-processes.py']:
        shutil.copy2(OLD / name,HERE / name)
    report = HERE / 'processes-final'
    report.mkdir()
    result = copy.deepcopy(old_prep)
    result.update(schema='allocation-fresh-process-copy-preparation-v2',complete=False,
        amplitudes_evaluated=False,compilation_performed=False,
        controller=binding(Path(__file__)),old_frozen_bindings=binding(old_freeze),
        old_preparation=binding(old_prep_path),recovery=binding(HERE / 'recovery-result.json'),
        new_plan=binding(HERE / 'PLAN.md'),builds={})
    write(report / 'started.json',result)
    try:
        for case,arms in old_prep['builds'].items():
            result['builds'][case] = {}
            for arm,prior in arms.items():
                original = ROOT / prior['output']
                original_manifest = bench.generated_source_manifest(original)
                assert original_manifest['files_sha256'] == prior['generated_sources_sha256']
                assert binding(ROOT / prior['executable']['path']) == prior['executable']
                assert policy.library_hashes(original) == prior['libraries']
                output = WORK / 'processes-final' / arm / 'outputs/candidate' / case
                output.parent.mkdir(parents=True,exist_ok=True)
                subprocess.run(['cp','-a','--reflink=auto',str(original),str(output)],check=True)
                for path in output.rglob('*'):
                    if path.is_symlink() and path.resolve().is_relative_to(original):
                        target = output / path.resolve().relative_to(original)
                        path.unlink()
                        path.symlink_to(os.path.relpath(target,path.parent))
                sub = output / Path(prior['directory']).relative_to(Path(prior['output']))
                for name in ['worker-protocol.fifo','dimensions.dat',*workers.OUTPUTS]:
                    (sub / name).unlink(missing_ok=True)
                for pattern in ['*HelFilter.dat','*LoopFilter.dat']:
                    for path in (output / 'SubProcesses').rglob(pattern):
                        path.unlink()
                assert bench.generated_source_manifest(output) == original_manifest
                assert sha(sub / 'check') == prior['executable']['sha256']
                assert policy.library_hashes(output) == prior['libraries']
                record = copy.deepcopy(prior)
                record['output'] = str(output.relative_to(ROOT))
                record['directory'] = str(sub.relative_to(ROOT))
                for key in ['executable','driver','parameter_card','madloop_card','actual_parameter_card']:
                    old_path = ROOT / prior[key]['path']
                    assert old_path.is_relative_to(original)
                    new_path = output / old_path.relative_to(original)
                    record[key] = binding(new_path)
                    assert record[key]['sha256'] == prior[key]['sha256']
                retained = report / case / arm
                retained.mkdir(parents=True)
                point_target = report / f'{case}-points.dat'
                if not point_target.exists():
                    shutil.copy2(ROOT / prior['points']['path'],point_target)
                record['points'] = binding(point_target)
                assert record['points']['sha256'] == prior['points']['sha256']
                write(retained / 'compiled-sources.json',original_manifest)
                record['fresh_copy_origin'] = {'build':binding(OLD / 'processes-final' / case / arm / 'build.json'),
                    'executable_sha256':prior['executable']['sha256'],'recompiled':False}
                write(retained / 'build.json',record)
                result['builds'][case][arm] = record
            a,b = [result['builds'][case][arm] for arm in ['original','allocation']]
            for name in ['generated_sources_sha256','runtime_metadata','link_inputs','header_sha256','module_source_sha256','shared_numerical_settings']:
                assert a[name] == b[name]
        for name,record in json.loads((OLD / 'pause-retention.json').read_text())['files'].items():
            assert sha(OLD / name) == record['sha256'] and (OLD / name).stat().st_size == record['bytes'],name
        result['complete'] = True
    except BaseException as exc:
        result['failure'] = repr(exc)
        raise
    finally:
        write(report / 'preparation.json',result)
    print(json.dumps({'complete':True,'compiled':False,'numerical_execution':False,
        'preparation':binding(report / 'preparation.json')}),flush=True)

if __name__ == '__main__':
    main()
