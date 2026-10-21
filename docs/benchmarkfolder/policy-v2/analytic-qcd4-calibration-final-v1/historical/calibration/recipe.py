#!/usr/bin/env python3
"""Recompute the retained 200-row pole diagnostic without a physics import."""
from pathlib import Path
import hashlib
import json
import math
import re

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
receipt = json.loads((HERE/'receipt.json').read_text())


def checked(record):
    path = ROOT/record['path']
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256'] or len(data) != record['bytes']:
        raise ValueError('Changed retained evidence: '+record['path'])
    return path


for record in (receipt['calibration'], receipt['build'], receipt['compiled_source_manifest'],
               receipt['parameter_card'], receipt['points'], receipt['model_correction'],
               receipt['archived_analytic_formula_source'], receipt['archived_Binoth_normalization_adapter'],
               receipt['independent_source_derivation_crosscheck'], receipt['checks']):
    checked(record)
for entry in receipt['actual_worker_source_excerpts'].values():
    checked(entry['file'])

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
couplings = checked(receipt['actual_worker_source_excerpts']['model_couplings']['file']).read_text()
literal = re.search(r'PARAMETER\s*\(PI=([0-9.]+D[+-]?\d+)\)', couplings, re.I).group(1)
pi = float(literal.upper().replace('D', 'E'))
g = 2*math.sqrt(alpha*pi)
coefficient = -(2*(4/3)+2*3)*g**2/(8*pi**2)
assert coefficient == receipt['formula']['source_binary64_ratio']
raw = json.loads(checked(receipt['calibration']).read_text())
rows = raw['rows']
assert raw['answer_dimensions'] == {'allocated': 1, 'loop_orders': 1, 'born_orders': 1}
assert [(r['point'], r['order']) for r in rows] == [(p, o) for p in range(1, 101) for o in (0, 1)]
scales = {}
for row in rows:
    assert len(row['coefficients']) == 4 and all(math.isfinite(x) for x in row['coefficients'])
    scales[row['point']] = max(scales.get(row['point'], 0.0), *map(abs, row['coefficients']))
failures, fractions, relative_errors = [], [], []
for row in rows:
    born, observed = row['coefficients'][0], row['coefficients'][3]
    predicted = coefficient*born
    error = abs(observed-predicted)
    tolerance = 1e-7*max(abs(observed), abs(predicted))+1e-10*scales[row['point']]
    fractions.append(error/tolerance if tolerance else 0.0)
    relative_errors.append(error/max(abs(observed), abs(predicted)) if max(abs(observed), abs(predicted)) else 0.0)
    if error > tolerance:
        failures.append((row['point'], row['order']))
assert len(rows) == receipt['scalar_double_pole_checks'] == 200
assert max(fractions) == receipt['maximum_fraction_of_tolerance']
assert max(relative_errors) == receipt['maximum_relative_double_pole_error']
assert not failures and receipt['passed']
print(json.dumps({'passed': True, 'scope': 'CutTools qcd4 calibration double pole only',
                  'points': 100, 'row_checks': 200, 'coefficient': coefficient,
                  'maximum_fraction_of_tolerance': max(fractions),
                  'maximum_relative_double_pole_error': max(relative_errors)}, indent=2))
