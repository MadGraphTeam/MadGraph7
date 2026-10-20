#!/usr/bin/env python3
"""Prepare/build only. This script never launches an amplitude or replay."""
import sys
sys.dont_write_bytecode = True
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
sys.path[:0] = [str(HERE), str(ROOT/'tools'), str(ROOT)]
import capture_source
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy

GEN = ROOT/'docs/benchmarkfolder/policy-v2/generation-abi3-tensor-factor-v1'
REF = ROOT/'docs/benchmarkfolder/experiments/symmetric-factorization-production/dev-qcd3-v1'
SOURCE = ROOT/'.benchmark-work/policy-v2/generation-abi3-tensor-factor-v1/outputs/candidate/qcd-3'
WORK = ROOT/'.benchmark-work/experiments/ninjars-qp-numerator-dp-probe-v1'
OUTPUT = WORK/'qcd-3'
SUB = Path('SubProcesses/P0_uux_zgg')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(name, data):
    path = HERE/name
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def runtime(output):
    return {'libraries': policy.library_hashes(output),
            'metadata': {name: sha(path) for name,path in policy.runtime_metadata_files(output).items()},
            'link_inputs': {name: sha(output/name) for name in policy.RUNTIME_LINK_INPUTS}}


def main():
    os.sched_setaffinity(0, {2})
    assert not WORK.exists() and not (HERE/'prepared.json').exists()
    generation = load(GEN/'summary.json')['cases']['qcd-3']
    source_manifest = load(ROOT/generation['compiled_manifest'])
    assert bench.generated_source_manifest(SOURCE) == source_manifest
    assert sha(SOURCE/SUB/'check') == generation['executable_sha256']
    assert generation['generation_revision'] == 'f4bffc61ed67e5d083b80ff9901768d4d5779989'
    reference = load(REF/'result.json')['measurements']['8']
    assert reference['executable_sha256'] == generation['executable_sha256']
    assert reference['status_counts'] == {'218':98, '322':2}
    assert [r['point'] for r in reference['rows'] if r['order']==0 and r['status']==322] == [44,80]
    assert reference['shared_numerical_settings']['nrotations_dp'] == 0
    assert reference['shared_numerical_settings']['nrotations_qp'] == 0
    before_runtime = runtime(SOURCE)
    retained_runtime = load(REF/'runtime.json')
    assert before_runtime['libraries'] == retained_runtime['libraries']
    assert before_runtime['metadata'] == {k:v['sha256'] for k,v in retained_runtime['metadata'].items()}
    points = GEN/'points/qcd-3.dat'
    assert sha(points) == reference['points_sha256'] and int(points.read_text().split()[0]) == 100
    card = REF/'inputs/candidate/qcd-3/MadLoopParams-8.dat'
    assert sha(card) == reference['madloop_card_sha256']
    assert sha(SOURCE/'Cards/param_card.dat') == reference['parameter_card_sha256']
    binding = {'source_output':str(SOURCE.relative_to(ROOT)),
               'source_revision':generation['generation_revision'],
               'source_manifest':source_manifest,
               'original_executable_sha256':reference['executable_sha256'],
               'runtime':before_runtime,
               'reference_result':str((REF/'result.json').relative_to(ROOT)),
               'reference_result_sha256':sha(REF/'result.json'),
               'reference_rows':reference['rows'],
               'reference_status_counts':reference['status_counts'],
               'points_sha256':sha(points), 'parameter_card_sha256':reference['parameter_card_sha256'],
               'madloop_card_sha256':sha(card),
               'shared_numerical_settings':reference['shared_numerical_settings'],
               'instrumentation_sha256':sha(HERE/'capture_source.py'),
               'prepare_sha256':sha(Path(__file__)), 'plan_sha256':sha(HERE/'PLAN.md')}
    write('bindings.json', binding)
    result = {'complete':False, 'numerical_launch':False, 'acceptance_eligible':False, 'cpu':2}
    try:
        WORK.mkdir(parents=True)
        shutil.copytree(SOURCE, OUTPUT, symlinks=False)
        assert bench.generated_source_manifest(OUTPUT) == source_manifest
        assert runtime(OUTPUT) == before_runtime
        shutil.copy2(points, HERE/'points.dat')
        shutil.copy2(SOURCE/'Cards/param_card.dat', HERE/'param_card.dat')
        shutil.copy2(card, HERE/'MadLoopParams.dat')
        for p in [OUTPUT/'Cards/MadLoopParams.dat', *(OUTPUT/'SubProcesses').rglob('MadLoopParams.dat')]:
            assert p.resolve().is_relative_to(OUTPUT)
            p.write_bytes(card.read_bytes())
        for pattern in ('*HelFilter.dat','*LoopFilter.dat'):
            for p in (OUTPUT/'SubProcesses').rglob(pattern): p.unlink()
        directory = OUTPUT/SUB
        # Preserve the unchanged public driver source and exact wrapper transformation.
        edits = {'CT_interface.f':capture_source.interface, 'check_sa.f':capture_source.driver}
        patches=[]
        for name, transform in edits.items():
            p=directory/name; old=p.read_text(); new=transform(old)
            assert new != old
            for stage,text in [('original',old),('capture',new)]:
                dst=HERE/'source'/stage/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(text)
            patches.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                          fromfile='original/'+name,tofile='capture/'+name))
            p.write_text(new)
            p.with_suffix('.o').unlink(missing_ok=True)
        (HERE/'capture.patch').write_text(''.join(patches))
        observed = bench.generated_source_manifest(OUTPUT)
        delta = sorted(k for k in source_manifest['files'] if source_manifest['files'][k] != observed['files'][k])
        assert delta == [str(SUB/'CT_interface.f'),str(SUB/'check_sa.f')],delta
        assert set(observed['files']) == set(source_manifest['files'])
        before_build=observed
        command=['make','-C',str(directory),'-j1','SHELL='+shutil.which('bash'),'check']
        start=time.monotonic()
        with (HERE/'build.log').open('w') as stream:
            process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        result['build']={'command':command,'returncode':process.returncode,'seconds_diagnostic':time.monotonic()-start,
                         'log_sha256':sha(HERE/'build.log')}
        assert process.returncode == 0
        assert bench.generated_source_manifest(OUTPUT) == before_build
        assert runtime(OUTPUT) == before_runtime
        assert bench.generated_source_manifest(SOURCE) == source_manifest
        assert sha(SOURCE/SUB/'check') == binding['original_executable_sha256']
        assert runtime(SOURCE) == before_runtime
        result.update(complete=True,source_manifest=observed, runtime=runtime(OUTPUT),
                      executable_sha256=sha(directory/'check'),changed_files=delta)
    except Exception as error:
        result['error']=repr(error)
        raise
    finally:
        write('prepared.json',result)


if __name__=='__main__':
    main()
