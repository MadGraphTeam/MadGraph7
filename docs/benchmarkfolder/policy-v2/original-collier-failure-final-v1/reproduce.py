#!/usr/bin/env python3
"""Reproduce the expected original Collier failures, never scientific success."""
from pathlib import Path
from collections import Counter
import hashlib
import itertools
import json
import math

HERE = Path(__file__).resolve().parent
CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def data(name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe capsule path')
    path = HERE / path
    require(path.resolve().is_relative_to(HERE) and not path.is_symlink(), 'Input is not capsule-local')
    if name not in CACHE:
        CACHE[name] = path.read_bytes()
    return CACHE[name]


def checked(name, record):
    content = data(name)
    require(len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256'],
            'Changed capsule evidence: ' + name)
    return content


def load(name):
    return json.loads(data(name))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


# Every bound payload is checked before source/card interpretation or comparison
# arithmetic. The historical evidence seal and its relative strings are intact.
publication = load('publication-manifest.json')
require(publication['schema'] == 'original-collier-failure-capsule-v1', 'Wrong publication manifest')
for name, record in publication['files'].items():
    checked(name, record)
index = load('input-index.json')
require(index['schema'] == 'original-collier-failure-local-input-index-v1', 'Wrong local input index')
prefix = index['historical_proof_prefix'] + '/'
seal = load(prefix + 'evidence-seal.json')
root_review = load(index['historical_root_review'])
require(hashlib.sha256(data(prefix + 'evidence-seal.json')).hexdigest() ==
        root_review['evidence_seal_sha256'], 'Historical seal differs from root review')
require(canonical({k: v for k, v in seal.items() if k != 'identity_sha256'}) == seal['identity_sha256'],
        'Historical seal identity differs')
require(len(seal['files']) == seal['file_count'] == 281 and
        sum(v['bytes'] for v in seal['files'].values()) == seal['bytes'] == 5943344,
        'Historical payload inventory differs')
for name, record in seal['files'].items():
    mapped = index['original_seal_member_map'][name]
    require(mapped['sha256'] == record['sha256'] and mapped['bytes'] == record['bytes'],
            'Historical member mapping differs')
    checked(mapped['local_path'], record)
copies = load(prefix + 'copies.json')
for original, record in copies.items():
    mapped = index['historical_original_path_map'][original]
    require(mapped == {'local_path': prefix + record['retained'], 'sha256': record['sha256'],
                       'bytes': record['bytes']}, 'Historical path alias differs')
    checked(mapped['local_path'], record)


def retained(name):
    return data(prefix + name)


def obj(name):
    return json.loads(retained(name))


result = obj('result.json')
build = obj('raw/report/builds/qcd-4-baseline-7/build.json')
compiled = obj('raw/report/builds/qcd-4-baseline-7/compiled-sources.json')
require(canonical(compiled['files']) == compiled['files_sha256'] == build['generated_sources_sha256'] ==
        result['source_bindings']['compiled_manifest_sha256'], 'Compiled source binding differs')
for name, record in compiled['files'].items():
    checked(prefix + 'source/actual-compiled-tree/' + name, record)
require(build['driver_sha256'] == hashlib.sha256(retained('raw/worker/check_sa.f')).hexdigest() and
        build['executable_sha256'] == result['source_bindings']['executable_sha256'],
        'Recorded driver/executable identity differs')
require(build['parameter_card_sha256'] == hashlib.sha256(retained('raw/actual-output/Cards/param_card.dat')).hexdigest()
        and build['madloop_card_sha256'] == hashlib.sha256(retained('raw/actual-output/Cards/MadLoopParams.dat')).hexdigest(),
        'Actual card hashes differ')
require(retained('raw/actual-output/Cards/MadLoopParams.dat') == retained('raw/actual-resource-MadLoopParams.dat') ==
        retained('raw/report/builds/qcd-4-baseline-7/MadLoopParams.dat'), 'Effective MadLoop card differs')
rebuild = obj('raw/actual-output/correction-build.json')
require({k: v['sha256'] for k, v in result['source_bindings']['libraries'].items()} == rebuild['libraries_after'],
        'Historical library/rebuild hash identities differ')
correction = obj('source/model-correction.json')
require({k: v for k, v in compiled['files'].items() if k.startswith('Source/MODEL/')} ==
        {k: v for k, v in correction['cases']['qcd-4']['fresh_after_manifest']['files'].items()
         if k.startswith('Source/MODEL/')}, 'Corrected model source postimage differs')
for anchor in result['source_path'].values():
    source = retained(anchor['file'])
    require(hashlib.sha256(source).hexdigest() == anchor['sha256'], 'Stability source identity differs')
    lines = source.decode().splitlines()
    require(all(lines[item['line'] - 1] == item['text'] for item in anchor['lines']),
            'Stability source convention differs')

lines = retained('raw/actual-output/Cards/MadLoopParams.dat').decode().splitlines()
def card_value(key):
    return lines[lines.index('#' + key) + 1].strip()

require(card_value('MLReductionLib') == '7', 'Selected backend is not solely Collier')
require(float(card_value('MLStabThres').replace('d', 'e')) == result['fixed_threshold'] == 1e-9,
        'Unchanged stability threshold differs')
require(float(card_value('COLLIERRequiredAccuracy').replace('d', 'e')) == 1e-12 and
        all(card_value(k).lower() == '.true.' for k in
            ('COLLIERComputeUVpoles', 'COLLIERComputeIRpoles', 'COLLIERUseInternalStabilityTest')),
        'Collier pole/stability configuration differs')
settings = build['shared_numerical_settings']
require(settings['mlstabthres'] == 1e-9 and settings['collierrequiredaccuracy'] == 1e-12 and
        settings['colliercomputeuvpoles'] and settings['colliercomputeirpoles'] and
        settings['collieruseinternalstabilitytest'], 'Build settings differ from actual card')
log = retained('raw/report/logs/worker-qcd-4-baseline-7.log').decode(errors='replace')
require('No quadruple precision will be used.' in log, 'Historical selected-QP warning differs')

rows = []
for line in retained('raw/worker/values.dat').decode().splitlines():
    fields = line.split()
    require(len(fields) == 8, 'Malformed captured Collier row')
    rows.append(dict(point=int(fields[0]), order=int(fields[1]), status=int(fields[2]),
                     coefficients=list(map(float, fields[3:7])), accuracy=float(fields[7])))
expected_order = [(p, o) for p in range(1, 101) for o in (0, 1)]
require([(r['point'], r['order']) for r in rows] == expected_order and rows == obj('parsed-collier-rows.json'),
        'Captured Collier rows differ')
require(list(map(int, retained('raw/worker/dimensions.dat').split())) == [1, 1, 1], 'Answer dimensions differ')
counts = dict(Counter(str(r['status']) for r in rows if r['order'] == 0))
timed = dict((str(int(a)), int(b)) for a, b in
             (line.split() for line in retained('raw/worker/timed_status.dat').decode().splitlines()))
require(counts == timed == result['point_status_counts'] == root_review['collier_point_status_counts'] ==
        {'217': 98, '410': 2}, 'Expected failed status histogram differs')
failure = obj('raw/report/rejected/qcd-4-failure.json')
require(failure['stage'] == 'calibration' and failure['backend'] == 7 and failure['command_sequence'] == 1,
        'Historical failure stage differs')
for name in ('values.dat', 'timing.dat', 'timed_status.dat', 'dimensions.dat'):
    require(failure['raw_outputs'][name] == retained('raw/worker/' + name).decode(), 'Rejected raw payload differs')
exceptional = [r for r in rows if r['status'] == 410]
rejected = sorted({r['point'] for r in exceptional})
require(exceptional == result['exceptional_rows'] and rejected ==
        root_review['stability_rejected_physical_points'] == [17, 52] and
        all(r['accuracy'] > 1e-9 for r in exceptional), 'Expected stability failures did not reproduce')

calibrations = {backend: obj('raw/report/data/qcd-4-baseline-' + str(backend) + '-calibration.json')
                for backend in (1, 6)}
point_bytes = retained('raw/report/points/qcd-4.dat')
point_lines = point_bytes.decode().splitlines()
require(int(point_lines[0]) == 100 and len(point_lines[1:]) == 600,
        'Different retained physical-point inventory')
for calibration in calibrations.values():
    require(calibration['points_sha256'] == hashlib.sha256(point_bytes).hexdigest() and
            calibration['parameter_card_sha256'] == build['parameter_card_sha256'],
            'Comparison backend uses different physical inputs')
scalars = {1: calibrations[1]['rows'], 6: calibrations[6]['rows'], 7: rows}
scales = {}
for backend, entries in scalars.items():
    require([(r['point'], r['order']) for r in entries] == expected_order, 'Backend point/order rows differ')
    require(all(len(r['coefficients']) == 4 and all(math.isfinite(v) for v in r['coefficients'] + [r['accuracy']])
                for r in entries), 'Nonfinite retained calibration values')
    for row in entries:
        scales[row['point']] = max(scales.get(row['point'], 0.0), *map(abs, row['coefficients']))
checks = []
for left, right in itertools.combinations(scalars, 2):
    for a, b in zip(scalars[left], scalars[right]):
        for column, (x, y) in enumerate(zip(a['coefficients'], b['coefficients'])):
            error = abs(x - y)
            tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scales[a['point']]
            checks.append(dict(backends=[left, right], point=a['point'], order=a['order'], column=column,
                               values=[x, y], absolute_error=error, tolerance=tolerance,
                               fraction_of_tolerance=error / tolerance if tolerance else 0.0,
                               passed=error <= tolerance))
recorded = [json.loads(line) for line in retained('comparison-checks.jsonl').decode().splitlines()]
require(checks == recorded and len(checks) == root_review['component_checks_recomputed'] == 2400,
        'Full common-point-scale arithmetic differs')
failed = [check for check in checks if not check['passed']]
require(failed == result['cross_backend_value_diagnostic']['failures'] == root_review['coefficient_failure_rows'],
        'Expected coefficient failures did not reproduce')
require(len(failed) == 2 and {(r['point'], r['column'], tuple(r['backends'])) for r in failed} ==
        {(99, 1, (1, 7))} and {r['order'] for r in failed} == {0, 1}, 'Failure event/multiplicity differs')
require(max(r['fraction_of_tolerance'] for r in checks) ==
        result['cross_backend_value_diagnostic']['maximum_fraction_of_tolerance'], 'Worst discrepancy differs')
print(json.dumps(dict(reproduction_status='expected-failures-reproduced',
    scientific_agreement_passed=False, stability_acceptance_passed=False,
    component_checks_recomputed=2400, physical_points=100,
    stability_rejected_physical_points=rejected, coefficient_failure_rows=len(failed),
    coefficient_failure_physical_points=[99],
    maximum_fraction_of_tolerance=max(r['fraction_of_tolerance'] for r in checks),
    scope='Original-helper CutTools/Ninja/Collier qcd4 calibration failure only; no campaign acceptance'), indent=2))
