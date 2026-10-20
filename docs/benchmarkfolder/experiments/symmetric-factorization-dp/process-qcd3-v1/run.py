#!/usr/bin/env python3
"""Bound, isolated actual-shape DP correctness diagnostic; no timing claim."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import process_prototype as prototype

HERE, BASE, ROOT = prototype.HERE, prototype.BASE, prototype.ROOT
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import ninjars_benchmark as bench
from ninjars_report_v2 import compare
from madgraph.various.banner import MadLoopParam

SOURCE = ROOT/'.benchmark-work/experiments/symmetric-factorization-production/dev-qcd3-v1/outputs/candidate/qcd-3'
WORK = ROOT/'.benchmark-work/experiments/symmetric-factorization-dp/process-qcd3-v1'
REFERENCE = ROOT/'docs/benchmarkfolder/experiments/symmetric-factorization-production/dev-qcd3-v1'
POINTS = ROOT/'docs/benchmarkfolder/points/qcd-3.dat'
VARIANTS = ('original', 'capture-only', 'factor-trace')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def environment():
    result = os.environ.copy()
    result.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MAKEFLAGS='SHELL='+shutil.which('bash'),
        LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE/'HEPTools'/p) for p in ('ninja/lib', 'collier')))
    return result


def command(args, cwd, log):
    started = time.monotonic()
    with log.open('x') as output:
        result = subprocess.run(args, cwd=cwd, env=environment(), stdout=output, stderr=subprocess.STDOUT)
    record = dict(command=args, cwd=str(cwd.relative_to(ROOT)), returncode=result.returncode,
                  wall_seconds=time.monotonic()-started, log_sha256=sha(log))
    return record


def directory(variant):
    return next((WORK/variant).glob('SubProcesses/P*_*'))


def prepare():
    assert not WORK.exists() and not (HERE/'preparation.json').exists()
    inventory = json.loads((BASE/'process-plan-v1/inventory-v3.json').read_text())
    for name, digest in inventory['source_files'].items():
        assert sha(ROOT/name) == digest, name
    assert sha(BASE/'process-plan-v1/inventory.py') == inventory['script_sha256']
    points = POINTS.read_bytes()
    assert int(points.splitlines()[0]) == 100
    (HERE/'points.dat').write_bytes(points)
    (HERE/'approved-plan.md').write_bytes((BASE/'process-plan-v1/PLAN.md').read_bytes())
    config = json.loads((ROOT/'docs/benchmarkfolder/policy-v2/cases.json').read_text())
    assert config['madloop_parameters'] == dict(MLStabThres=1e-9, COLLIERRequiredAccuracy=1e-12)
    (HERE/'policy-v2-cases.json').write_bytes((ROOT/'docs/benchmarkfolder/policy-v2/cases.json').read_bytes())
    result = dict(complete=False, cpu=5, acceptance_eligible=False,
        source=str(SOURCE.relative_to(ROOT)), input_inventory_sha256=sha(BASE/'process-plan-v1/inventory-v3.json'),
        points_sha256=sha(POINTS), maxcoefs=70, qp_source_policy='Byte-identical parent QP source in every clone',
        scope='Changed-DP-association correctness/IEEE-boundary/guard-count diagnostic; no speed claim',
        generator_sha256=sha(ROOT/'madgraph/various/q_polynomial.py'),
        scripts={p.name:sha(p) for p in HERE.glob('*.py')}, builds={})
    dump(HERE/'preparation.json', result)
    WORK.mkdir(parents=True)
    try:
        for variant in VARIANTS:
            output = WORK/variant
            subprocess.run(['cp', '-a', '--reflink=auto', str(SOURCE), str(output)], check=True)
            for path in output.rglob('*'):
                if path.is_symlink() and path.resolve().is_relative_to(SOURCE):
                    target = output/path.resolve().relative_to(SOURCE)
                    path.unlink()
                    path.symlink_to(os.path.relpath(target, path.parent))
            sub = directory(variant)
            changed, proof = ({}, None) if variant == 'original' else prototype.transform(sub, variant)
            report = HERE/variant
            report.mkdir()
            diffs = []
            for name, text in changed.items():
                path = sub/name
                diffs.extend(difflib.unified_diff(path.read_text().splitlines(True), text.splitlines(True),
                    fromfile='original/'+name, tofile=variant+'/'+name))
                path.write_text(text)
            (report/'patch.diff').write_text(''.join(diffs))
            if proof:
                dump(report/'transformation.json', proof)
            for path in sub.glob('mp_*.f'):
                assert path.read_bytes() == (next(SOURCE.glob('SubProcesses/P*_*'))/path.name).read_bytes()
            assert (output/'Cards/param_card.dat').read_bytes() == (SOURCE/'Cards/param_card.dat').read_bytes()
            manifest = bench.generated_source_manifest(output)
            dump(report/'compiled-sources.json', manifest)
            generated = report/'generated'
            generated.mkdir()
            for path in [*sub.glob('coef_construction_*.f'), sub/'polynomial.f', sub/'check_sa.f',
                         sub/'coef_specs.inc', sub/'loop_max_coefs.inc', sub/'makefile']:
                shutil.copy2(path, generated/path.name)
            build = command(['make', '-j2', 'check'], sub, report/'build.log')
            result['builds'][variant] = build
            dump(HERE/'preparation.json', result)
            assert build['returncode'] == 0, variant
            assert bench.generated_source_manifest(output) == manifest
            result['builds'][variant].update(executable_sha256=sha(sub/'check'),
                generated_sources_sha256=manifest['files_sha256'],
                capture_or_factor_public_qp_unchanged=True)
            command(['ldd', './check'], sub, report/'ldd.log')
            dump(HERE/'preparation.json', result)
            print(variant, 'compiled', flush=True)
        result['complete'] = True
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        dump(HERE/'preparation.json', result)


def configure(variant, backend, reference, report):
    output, sub = WORK/variant, directory(variant)
    card = MadLoopParam(str(output/'Cards/MadLoopParams.dat'))
    card['MLReductionLib'] = str(backend)
    settings = {str(k).lower():v for k,v in card.items()
                if str(k).lower() not in ('mlreductionlib', 'useqpintegrandforninjars')}
    settings['selected_backend_qp_integrand'] = card[{6:'UseQPIntegrandForNinja', 8:'UseQPIntegrandForNinjaRS'}[backend]]
    assert settings == reference['shared_numerical_settings']
    assert card['MLStabThres'] == 1e-9 and card['COLLIERRequiredAccuracy'] == 1e-12
    assert card['CTModeRun'] == -1
    template = ROOT/'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'
    for path in [output/'Cards/MadLoopParams.dat', *(output/'SubProcesses').rglob('MadLoopParams.dat')]:
        card.write(str(path), template=str(template))
    for pattern in ('*HelFilter.dat', '*LoopFilter.dat'):
        for path in (output/'SubProcesses').rglob(pattern):
            path.unlink()
    candidates = [sub/'param_card.dat'] + [sub.joinpath(*(['..']*level), 'Cards/param_card.dat') for level in range(6)]
    actual = next(p for p in candidates if p.is_file())
    assert actual.read_bytes() == (SOURCE/'Cards/param_card.dat').read_bytes()
    shutil.copy2(output/'Cards/MadLoopParams.dat', report/'MadLoopParams.dat')
    shutil.copy2(actual, report/'param_card.dat')
    return settings


def read_rows(path, backend, dims):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        assert len(fields) == 8
        row = dict(point=int(fields[0]), order=int(fields[1]), status=int(fields[2]),
                   coefficients=list(map(float, fields[3:7])), accuracy=float(fields[7]))
        assert all(math.isfinite(x) for x in row['coefficients']+[row['accuracy']])
        assert row['accuracy'] >= 0
        bench.validate_status(row['status'], backend)
        rows.append(row)
    assert len(dims) == 3 and dims[0] == max(dims[1:]) and dims[2] <= dims[1]
    assert [(r['point'],r['order']) for r in rows] == [(p,o) for p in range(1,101) for o in range(dims[1]+1)]
    return rows


def run(backend):
    prepared = json.loads((HERE/'preparation.json').read_text())
    assert prepared['complete']
    assert sha(POINTS) == prepared['points_sha256']
    cohort = HERE/f'backend-{backend}'
    cohort.mkdir(exist_ok=False)
    reference_path = REFERENCE/f'data/candidate-qcd-3-{backend}-0.json'
    reference = json.loads(reference_path.read_text())
    result = dict(complete=False, acceptance_eligible=False, backend=backend, cpu=5,
                  source_reference_sha256=sha(reference_path), points_sha256=sha(POINTS),
                  measurements={}, source_preparation_sha256=sha(HERE/'preparation.json'))
    dump(cohort/'result.json', result)
    try:
        for variant in VARIANTS:
            report = cohort/variant
            report.mkdir()
            sub = directory(variant)
            assert sha(sub/'check') == prepared['builds'][variant]['executable_sha256']
            assert bench.generated_source_manifest(WORK/variant)['files_sha256'] == prepared['builds'][variant]['generated_sources_sha256']
            settings = configure(variant, backend, reference, report)
            for name in ('values.dat','timing.dat','timed_status.dat','dimensions.dat','dp_factor_counts.dat'):
                (sub/name).unlink(missing_ok=True)
            measured = command(['./check', str(POINTS), '1'], sub, report/'run.log')
            result['measurements'][variant] = measured
            dump(cohort/'result.json', result)
            for name in ('values.dat','timing.dat','timed_status.dat','dimensions.dat','dp_factor_counts.dat'):
                if (sub/name).is_file():
                    shutil.copy2(sub/name, report/name)
            assert measured['returncode'] == 0, variant
            dims = list(map(int, (report/'dimensions.dat').read_text().split()))
            rows = read_rows(report/'values.dat', backend, dims)
            timed = {}
            for line in (report/'timed_status.dat').read_text().splitlines():
                code, count = map(int, line.split())
                bench.validate_status(code, backend)
                assert count > 0 and str(code) not in timed
                timed[str(code)] = count
            assert sum(timed.values()) == 100
            measured.update(rows=rows, dimensions=dims, timed_status_counts=timed,
                status_counts=dict(Counter(str(r['status']) for r in rows if r['order']==0)),
                shared_numerical_settings=settings, values_sha256=sha(report/'values.dat'))
            if variant == 'original':
                measured['matches_bound_original_exactly'] = rows == reference['rows']
                assert measured['matches_bound_original_exactly']
            if variant == 'capture-only':
                measured['matches_original_exactly'] = rows == result['measurements']['original']['rows']
                assert measured['matches_original_exactly']
            if variant != 'original':
                counts = [list(map(int, line.split())) for line in (report/'dp_factor_counts.dat').read_text().splitlines()]
                assert [r[0] for r in counts] == list(range(1,5))
                if variant == 'factor-trace':
                    assert all(len(r)==4 and r[1]==r[2]+r[3] and r[1]>0 for r in counts)
                else:
                    assert all(len(r)==3 and 0<=r[2]<=r[1] and r[1]>0 for r in counts)
                measured['guard_or_capture_counts'] = counts
            dump(cohort/'result.json', result)
            print('backend', backend, variant, '100 points', measured['status_counts'], flush=True)
        groups = {name:record['rows'] for name,record in result['measurements'].items()}
        result['comparison'] = compare(groups)
        original = groups['original']
        changed = groups['factor-trace']
        result['accuracy_or_status_changes'] = [dict(point=a['point'], order=a['order'],
            original_status=a['status'], factor_status=b['status'], original_accuracy=a['accuracy'], factor_accuracy=b['accuracy'])
            for a,b in zip(original,changed) if (a['status'],a['accuracy']) != (b['status'],b['accuracy'])]
        result['printed_coefficient_bit_changes'] = sum(x.hex()!=y.hex() for a,b in zip(original,changed) for x,y in zip(a['coefficients'],b['coefficients']))
        assert result['comparison']['passed'], result['comparison']['failures']
        result['complete'] = True
        print('cohort passed', result['comparison']['max_tolerance_fraction'], flush=True)
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        dump(cohort/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare','run'))
    parser.add_argument('--backend', type=int, choices=(6,8), default=6)
    args = parser.parse_args()
    os.sched_setaffinity(0, {5})
    if args.action == 'prepare':
        prepare()
    else:
        run(args.backend)
