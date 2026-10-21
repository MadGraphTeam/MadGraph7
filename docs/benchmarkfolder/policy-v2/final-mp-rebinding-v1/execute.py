#!/usr/bin/env python3
"""Direct public-mode QP check of the retained Fortran state-update drivers."""
import difflib
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
REPORT = Path(__file__).resolve().parent
ORDINARY = REPORT.parent/'final-variant-controls-v1'
WORK = ROOT/'.benchmark-work/policy-v2/final-mp-rebinding-v1'
sys.path[:0] = [str(ROOT/'tools'), str(ROOT)]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as adapter
import ninjars_benchmark_workers as workers
import controls_driver as controls

bench.WORK, bench.REPORT = WORK, REPORT
NAMES = ('control-rebinding', 'control-rebinding-fresh')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    if path.exists():
        raise RuntimeError('Preserve previous diagnostic evidence: '+str(path))
    bench.dump(path, value)


def signatures(output, manifest):
    item = dict(backend=8, libraries=policy.library_hashes(output), compiled_manifest=manifest,
                runtime_metadata={name:policy.bound_file(path) for name,path in policy.runtime_metadata_files(output).items()})
    return policy.reducer_signature(item)


def verify_output(output, record):
    if (bench.generated_source_manifest(output) != record['sources'] or
            policy.library_hashes(output) != record['libraries'] or
            signatures(output, record['sources']) != record['runtime'] or
            sha(output/'Cards/param_card.dat') != record['parameter_card_sha256'] or
            sha(next(output.glob('SubProcesses/P*_*/check'))) != record['executable_sha256']):
        raise RuntimeError('Bound diagnostic source, executable or runtime changed')


def prepare():
    if WORK.exists() or (REPORT/'prepared.json').exists():
        raise RuntimeError('New diagnostic namespace required')
    original = policy.verified(policy.read(ORDINARY/'prepared.json'), 'final-variant-controls-preparation-v1')
    for phase in ('prepare', 'run', 'compare-rotation'):
        if policy.read(ORDINARY/'ledgers'/phase/'result.json')['status'] != 'passed':
            raise RuntimeError('Ordinary control is incomplete')
    if sha(REPORT/'controls_driver.py') != sha(ORDINARY/'controls_driver.py'):
        raise RuntimeError('Corrected DP/MP state driver differs')
    config = policy.read(ORDINARY/'control-variants.json')
    cases = [case for case in config['cases'] if case['id'] in NAMES]
    if {case['id'] for case in cases} != set(NAMES):
        raise RuntimeError('Missing exact ordinary rebinding controls')
    WORK.mkdir(parents=True)
    (REPORT/'points').mkdir()
    prepared = dict(schema='final-direct-qp-rebinding-v1', ordinary_preparation=policy.bound_file(ORDINARY/'prepared.json'),
                    script_sha256=sha(Path(__file__)), controls_driver=policy.bound_file(REPORT/'controls_driver.py'),
                    forced_mode=dict(CTModeRun=4, UseQPIntegrandForNinjaRS=True,
                                     expected_status=120, expected_accuracy=-1,
                                     stability_measurement=False, automatic_rescue=False), variants={})
    for case in cases:
        name = case['id']
        parent = ROOT/'.benchmark-work/policy-v2/final-variant-controls-v1/outputs/candidate'/name
        ordinary = original['variants'][name]
        if (bench.generated_source_manifest(parent) != ordinary['sources'] or
                policy.library_hashes(parent) != ordinary['libraries'] or
                signatures(parent, ordinary['sources']) != ordinary['reducer_signatures']['8'] or
                sha(next(parent.glob('SubProcesses/P*_*/check'))) != ordinary['executable_sha256']):
            raise RuntimeError('Ordinary driver no longer matches its passed evidence')
        output = adapter.clone_output(parent, WORK/'outputs/candidate'/name)
        directory = bench.subprocess_dir('candidate', case)
        source = directory/'check_sa.f'
        text = source.read_text()
        old = 'CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)'
        new = 'CALL ML5_0_FORCE_STABILITY_CHECK(.FALSE.)'
        if text.count(old) != 1:
            raise RuntimeError('Expected exactly one public force-stability switch')
        modified = text.replace(old, new)
        source.unlink()
        source.write_text(modified)
        retained = REPORT/'drivers'/name
        retained.mkdir(parents=True)
        (retained/'ordinary.f').write_text(text)
        (retained/'forced-qp.f').write_text(modified)
        (retained/'public-mode.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True), modified.splitlines(True),
                                                       fromfile='ordinary.f', tofile='forced-qp.f')))
        # Only this public test-driver switch differs; the DP/MP model-state
        # assignments, point schedule, numerator and reduction sources are exact.
        source.with_suffix('.o').unlink(missing_ok=True)
        (directory/'check').unlink(missing_ok=True)
        seconds = bench.run(['make', '-j1', 'check'], directory, WORK/'logs'/('compile-'+name+'.log'))
        manifest = bench.generated_source_manifest(output)
        changes = policy.delta(ordinary['sources']['files'], manifest['files'])
        if changes != {str(source.relative_to(output))}:
            raise RuntimeError('Forced QP changed another numerical source')
        if (policy.library_hashes(output) != ordinary['libraries'] or
                signatures(output, manifest) != ordinary['reducer_signatures']['8']):
            raise RuntimeError('Forced-mode reduction runtime changed')
        metadata = policy.read(output/'compilation.json')
        metadata.update(seconds=seconds, driver_sha256=sha(source), executable_sha256=sha(directory/'check'),
                        generated_sources_sha256=manifest['files_sha256'],
                        source_manifest_recording='recorded-after-direct-qp-test-driver-rebuild')
        bench.dump(output/'compiled-sources.json', manifest)
        bench.dump(output/'compilation.json', metadata)
        for suffix in ('.dat', '.json'):
            shutil.copy2(ORDINARY/'points'/(name+suffix), REPORT/'points'/(name+suffix))
        case['madloop_parameters'] = {**case['madloop_parameters'], 'CTModeRun':4,
                                      'UseQPIntegrandForNinjaRS':True}
        bench.retain_build_inputs('candidate', case)
        prepared['variants'][name] = dict(case=case, output=policy.relative(output), sources=manifest,
            libraries=policy.library_hashes(output), runtime=ordinary['reducer_signatures']['8'],
            executable_sha256=sha(directory/'check'), parameter_card_sha256=sha(output/'Cards/param_card.dat'),
            ordinary_record=policy.bound_file(ORDINARY/'data'/('candidate-'+name+'-8-0.json')),
            points=policy.bound_file(REPORT/'points'/(name+'.dat')),
            ordinary_source_sha256=sha(retained/'ordinary.f'), forced_source_sha256=sha(retained/'forced-qp.f'),
            patch=policy.bound_file(retained/'public-mode.patch'), compilation=metadata)
    dump(REPORT/'config.json', {**config, 'cases':cases})
    dump(REPORT/'prepared.json', policy.seal(prepared))
    print('Prepared two isolated direct-QP state-update controls.', flush=True)


def parse_direct_qp(directory):
    dimensions = list(map(int, (directory/'dimensions.dat').read_text().split()))
    if dimensions != [1,1,0]:
        raise RuntimeError('Direct HH answer dimensions changed')
    rows = []
    for line in (directory/'values.dat').read_text().splitlines():
        fields = line.split()
        if len(fields) != 8:
            raise RuntimeError('Malformed direct QP numerical row')
        row = dict(point=int(fields[0]), order=int(fields[1]), status=int(fields[2]),
                   coefficients=list(map(float, fields[3:7])), accuracy=float(fields[7]))
        if (row['status'] != 120 or row['accuracy'] != -1 or
                not all(math.isfinite(x) for x in row['coefficients'])):
            raise RuntimeError('Direct QP must report120 with documented unavailable stability estimate -1')
        rows.append(row)
    if [(row['point'],row['order']) for row in rows] != [(p,o) for p in range(1,11) for o in (0,1)]:
        raise RuntimeError('Direct QP rows are incomplete or reordered')
    histogram = [list(map(int,line.split())) for line in (directory/'timed_status.dat').read_text().splitlines()]
    if histogram != [[120,10]]:
        raise RuntimeError('Every evaluation must use explicit QP mode')
    return dict(rows=rows, status_counts={'120':10}, timed_status_counts={'120':10},
                answer_dimensions=dict(allocated=1,loop_orders=1,born_orders=0))


def run(selected, cpu):
    if tuple(selected) != NAMES or cpu != 190:
        raise RuntimeError('Use the fixed two-control CPU190 handoff')
    prepared = policy.verified(policy.read(REPORT/'prepared.json'), 'final-direct-qp-rebinding-v1')
    if prepared['script_sha256'] != sha(Path(__file__)):
        raise RuntimeError('Diagnostic controller changed after preparation')
    policy.check_file(prepared['controls_driver'])
    config = policy.read(REPORT/'config.json')
    results = {}
    for name in NAMES:
        record = prepared['variants'][name]
        case, output = record['case'], policy.path(record['output'])
        verify_output(output, record)
        policy.check_file(record['points'])
        policy.check_file(record['ordinary_record'])
        points = policy.path(record['points']['path'])
        directory = bench.subprocess_dir('candidate', case)
        settings = bench.configure('candidate', case, 8, config=config)
        if (settings['ctmoderun'] != 4 or not settings['selected_backend_qp_integrand']):
            raise RuntimeError('Direct QP card configuration was not retained')
        workers.validate_param_card(directory, (output/'Cards/param_card.dat').read_bytes())
        raw = REPORT/'raw'/name
        raw.mkdir(parents=True, exist_ok=False)
        for filename in (*workers.OUTPUTS, 'dimensions.dat'):
            (directory/filename).unlink(missing_ok=True)
        log = WORK/'logs'/('run-'+name+'.log')
        try:
            bench.run(['taskset','-c',str(cpu),'./check',str(points),'1'], directory, log)
        finally:
            for filename in (*workers.OUTPUTS, 'dimensions.dat'):
                if (directory/filename).is_file():
                    shutil.copy2(directory/filename, raw/filename)
            if log.is_file():
                shutil.copy2(log, raw/'run.log')
        result = parse_direct_qp(raw)
        result.update(source='candidate', case=name, backend=8, repetition=0, cpu=cpu,
                      points_sha256=sha(points), parameter_card_sha256=sha(output/'Cards/param_card.dat'),
                      madloop_card_sha256=sha(output/'Cards/MadLoopParams.dat'),
                      executable_sha256=sha(directory/'check'), runtime_identity=record['runtime']['identity_sha256'],
                      direct_forced_qp=True, stability_measurement=False, automatic_rescue=False,
                      performance_acceptance_claim=False, shared_numerical_settings=settings)
        dump(REPORT/'data'/('candidate-'+name+'-8-0.json'), result)
        results[name+'-8'] = result
        ordinary = policy.read(policy.path(record['ordinary_record']['path']))
        if not controls.compare_records('direct-qp-vs-dp-'+name,
                {'direct-qp-8':result, 'ordinary-dp-8':ordinary}, config):
            raise RuntimeError('Direct QP disagrees with passed DP control')
        verify_output(output, record)
        print(name+' actualQP120 completed', flush=True)
    if not controls.compare_records('direct-qp-retained-vs-finalized', results, config):
        raise RuntimeError('Direct QP context reset disagrees')
    dump(REPORT/'result.json', dict(passed=True, direct_forced_qp=True, expected_status=120,
         observed_statuses={name:value['status_counts'] for name,value in results.items()},
         expected_accuracy=-1, stability_measurement=False, automatic_rescue=False,
         performance_acceptance_claim=False))
