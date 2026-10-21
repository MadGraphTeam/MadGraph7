#!/usr/bin/env python3
"""Reproduce only the retained CutTools qcd4 leading-pole calibration locally."""
from pathlib import Path
import hashlib
import json
import math
import re

HERE = Path(__file__).resolve().parent
CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Unsafe capsule input path')
    path = HERE / relative
    require(path.resolve().is_relative_to(HERE) and not path.is_symlink(),
            'Capsule evidence must remain local')
    return path


def payload(name):
    path = local(name)
    if name not in CACHE:
        CACHE[name] = path.read_bytes()
    return CACHE[name]


def verify(name, record):
    data = payload(name)
    require(hashlib.sha256(data).hexdigest() == record['sha256'] and
            len(data) == record['bytes'], 'Changed capsule evidence: ' + name)
    return local(name)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


manifest = json.loads(payload('publication-manifest.json'))
require(manifest['schema'] == 'analytic-qcd4-calibration-capsule-v1',
        'Wrong capsule manifest')
for name, record in manifest['files'].items():
    verify(name, record)
index = json.loads(payload('input-index.json'))
require(index['schema'] == 'analytic-qcd4-local-input-index-v1', 'Wrong local input index')
for name, record in index['retained_files'].items():
    verify(name, record)


def checked(record):
    name = record['path']
    require(name in index['historical_path_to_retained_bytes'],
            'Historical input has no retained local bytes: ' + name)
    retained = index['historical_path_to_retained_bytes'][name]
    require(retained['sha256'] == record['sha256'] and
            ('bytes' not in record or retained['bytes'] == record['bytes']),
            'Historical input binding differs: ' + name)
    return verify(retained['local_path'], retained)


documents = index['historical_documents']
receipt = json.loads(payload(documents['calibration_receipt']))
derivation = json.loads(payload(documents['independent_source_derivation']))
root_review = json.loads(payload(documents['root_review']))
old_manifest = json.loads(payload(documents['calibration_manifest']))
for name, record in old_manifest['files'].items():
    verify('historical/calibration/' + name, record)
require(hashlib.sha256(payload(documents['calibration_manifest'])).hexdigest()
        == root_review['manifest_sha256'], 'Historical manifest differs from root review')
require(hashlib.sha256(payload(documents['independent_source_derivation'])).hexdigest()
        == root_review['source_derivation_sha256'], 'Independent derivation differs from root review')

# Resolve every bound path through the capsule index, including aliases for
# content-identical corrected postimages. Historical records remain unchanged.
def verify_bindings(value):
    if isinstance(value, dict):
        if {'path', 'sha256'}.issubset(value):
            checked(value)
        for child in value.values():
            verify_bindings(child)
    elif isinstance(value, list):
        for child in value:
            verify_bindings(child)


verify_bindings(receipt)
verify_bindings(derivation)
require(receipt['backend'] == 1 and receipt['case'] == 'qcd-4' and
        receipt['role'] == 'original_helper_corrected', 'Different calibration scope')
build = json.loads(checked(receipt['build']).read_text())
compiled = json.loads(checked(receipt['compiled_source_manifest']).read_text())
raw = json.loads(checked(receipt['calibration']).read_text())
require(canonical(compiled['files']) == compiled['files_sha256'] ==
        receipt['compiled_source_identity_sha256'], 'Compiled source identity differs')
require(build['generated_sources_sha256'] == raw['generated_sources_sha256'] ==
        compiled['files_sha256'], 'Calibration was built from different sources')
require(build['executable_sha256'] == raw['executable_sha256'] ==
        receipt['bound_executable_sha256'] == derivation['bound_executable_sha256'],
        'Historical executable identity differs')
require(raw['parameter_card_sha256'] == receipt['parameter_card']['sha256'] and
        raw['points_sha256'] == receipt['points']['sha256'], 'Calibration physical inputs differ')

prefix = derivation['actual_helper_output'].rstrip('/') + '/'
sources = {}
for label, entry in receipt['actual_worker_source_excerpts'].items():
    path = checked(entry['file'])
    text = path.read_text()
    sources[label] = text
    relative = entry['file']['path'].removeprefix(prefix)
    require(compiled['files'][relative] == entry['compiled_source_record'],
            'Source excerpt differs from compiled manifest: ' + label)
    lines = text.splitlines()
    for anchor in entry['relevant_lines']:
        require(lines[anchor['line'] - 1] == anchor['text'],
                'Historical source convention changed: ' + label)
for record in receipt['actual_model_postimages'].values():
    require(checked(record['actual_source']).read_bytes() ==
            checked(record['expected_postimage']).read_bytes(), 'Corrected model postimage differs')

# Reconstruct precisely the original binary64 expression from the actual card
# and generated coupling convention. No coefficient is fitted to output rows.
block, alpha = None, None
for line in checked(receipt['parameter_card']).read_text().splitlines():
    fields = line.split('#', 1)[0].split()
    if not fields:
        continue
    if fields[0].lower() == 'block':
        block = fields[1].lower()
    elif fields[0].lower() == 'decay':
        block = None
    elif block == 'sminputs' and fields[0] == '3':
        alpha = float(fields[1].replace('D', 'E').replace('d', 'e'))
require(alpha == receipt['formula']['card_alpha_s'] == 0.118, 'Actual alpha_s convention differs')
literal = re.search(r'PARAMETER\s*\(PI=([0-9.]+D[+-]?\d+)\)',
                    sources['model_couplings'], re.I).group(1)
pi = float(literal.upper().replace('D', 'E'))
require(literal == receipt['formula']['actual_DP_pi_literal'], 'Actual DP pi literal differs')
initialization = re.sub(r'\s+', '', sources['model_initialization']).upper()
require('G=2*DSQRT(AS*PI)' in initialization and 'AS=G**2/4/PI' in initialization,
        'Actual G/alpha_s initialization convention differs')
update = sources['model_couplings'].split('SUBROUTINE UPDATE_AS_PARAM()')[1].split('\n      END')[0]
require('READLHA = .FALSE.' in update and 'CALL COUP3()' in update and
        not re.search(r'\b(?:CALL\s+ALPHAS|G\s*=\s*ALPHAS)\b', update, re.I),
        'Actual worker update changes the alpha_s convention')
require('DATA IDEN/144/' in sources['born_matrix'] and
        'DATA IDEN/144/' in sources['loop_matrix'] and
        'RES(3)=NORMALIZATION*2.0D0*DBLE(RES(3))' in sources['cuttools_interface'],
        'Born/interference normalization differs')
g = 2 * math.sqrt(alpha * pi)
coefficient = -(2 * (4 / 3) + 2 * 3) * g ** 2 / (8 * pi ** 2)
require(coefficient == receipt['formula']['source_binary64_ratio'] ==
        derivation['formula']['numeric_expected_source_binary64_expression'],
        'Independent source-derived coefficient differs')

point_lines = checked(receipt['points']).read_text().splitlines()
require(int(point_lines[0]) == 100 and len(point_lines[1:]) == 600 and
        all(len(line.split()) == 4 for line in point_lines[1:]), 'Different fixed point inventory')
rows = raw['rows']
require(raw['answer_dimensions'] == {'allocated': 1, 'loop_orders': 1, 'born_orders': 1},
        'Different split-order dimensions')
require([(row['point'], row['order']) for row in rows] ==
        [(point, order) for point in range(1, 101) for order in (0, 1)], 'Different row order')
scales = {}
for row in rows:
    require(len(row['coefficients']) == 4 and
            all(math.isfinite(x) for x in row['coefficients']), 'Nonfinite calibration coefficient')
    scales[row['point']] = max(scales.get(row['point'], 0.0), *map(abs, row['coefficients']))
expected_checks = [json.loads(line) for line in checked(receipt['checks']).read_text().splitlines()]
require(len(rows) == len(expected_checks) == receipt['scalar_double_pole_checks'] == 200,
        'Different double-pole check count')
fractions, relative_errors = [], []
for row, original in zip(rows, expected_checks):
    born, observed = row['coefficients'][0], row['coefficients'][3]
    predicted = coefficient * born
    residual = observed - predicted
    error = abs(residual)
    tolerance = 1e-7 * max(abs(observed), abs(predicted)) + 1e-10 * scales[row['point']]
    fraction = error / tolerance if tolerance else 0.0
    relative = error / max(abs(observed), abs(predicted)) if max(abs(observed), abs(predicted)) else 0.0
    check = dict(point=row['point'], order=row['order'], status=row['status'], born=born,
                 observed_double_pole=observed, predicted_double_pole=predicted,
                 observed_ratio=observed / born, residual=residual, tolerance=tolerance,
                 fraction_of_tolerance=fraction, relative_double_pole_error=relative,
                 Born_scaled_residual=error / abs(born), point_scale=scales[row['point']],
                 passed=error <= tolerance)
    require(check == original and check['passed'], 'Retained double-pole row check differs')
    fractions.append(fraction)
    relative_errors.append(relative)
require(max(fractions) == receipt['maximum_fraction_of_tolerance'] and
        max(relative_errors) == receipt['maximum_relative_double_pole_error'] and receipt['passed'],
        'Historical diagnostic summary differs')
result = dict(passed=True, scope='CutTools qcd4 calibration double pole only', points=100,
              row_checks=200, coefficient=coefficient,
              maximum_fraction_of_tolerance=max(fractions),
              maximum_relative_double_pole_error=max(relative_errors))
require(result == root_review['recipe_result'], 'Local result differs from historical root reproduction')
print(json.dumps(result, indent=2))
