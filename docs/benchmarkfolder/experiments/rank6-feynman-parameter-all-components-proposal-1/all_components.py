#!/usr/bin/env python3
"""Source proposal: all Cartesian N6/R6 moments by independent quadrature.

The 296 unique parameter moments are integrated once and linearly combined
into the 210 MadLoop tensor components. NumPy extended precision is explicitly
not binary128/QP. Actual fixture integration is gated by --admit-physics.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
PROTO = HERE.parent/'rank6-feynman-parameter-proposal-1'
PROTO_INDEX_SHA256 = '03ffcd9f8b88855b59d78387cde255460a36d35cca251fce093741a9c0065db3'
QUAL = HERE.parent/'rank6-feynman-parameter-qualification-1'
QUAL_INDEX_SHA256 = 'c2fe806d2996e7ee758e2a7894318d04388b640126b9f636fd23be99650b90d7'
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
for _variable in THREAD_VARS:
    os.environ[_variable] = '1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_core():
    if sha(PROTO/'source-index.json') != PROTO_INDEX_SHA256:
        raise ValueError('independent scalar-prototype source index changed')
    index = json.loads((PROTO/'source-index.json').read_text())
    binding = next(x for x in index['files'] if x['path'] == 'quadrature.py')
    if sha(PROTO/'quadrature.py') != binding['sha256']:
        raise ValueError('certified Gaussian/core source changed')
    spec = importlib.util.spec_from_file_location('rank6_fp_immutable_core', PROTO/'quadrature.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_layout(core):
    index = json.loads((PROTO/'source-index.json').read_text())
    record = next(x for x in index['inputs'] if x['path'].endswith('/finite-audit-1/rows.json.gz'))
    rows = core.COMMON/'finite-audit-1/rows.json.gz'
    if sha(rows) != record['sha256']:
        raise ValueError('historical component/index join changed')
    snapshot = json.loads(gzip.decompress(rows.read_bytes()))
    monomials = [tuple(a) for a in snapshot['monomials']]
    if len(monomials) != 210 or len(set(monomials)) != 210:
        raise ValueError('expected all 210 unique Cartesian monomials')
    for rank, count in ((0, 1), (2, 15), (4, 70), (6, 210)):
        if sum(sum(a) <= rank for a in monomials) != count:
            raise ValueError('unexpected graded MadLoop coefficient layout')
        if any(sum(a) > rank for a in monomials[:count]):
            raise ValueError('prefix does not contain the stated ranks')
    return monomials


def plan(core, monomials):
    lookup = {a: i for i, a in enumerate(monomials)}
    predecessors = []
    for i, powers in enumerate(monomials):
        if not i:
            assert powers == (0, 0, 0, 0)
            predecessors.append(None)
            continue
        axis = next(axis for axis, power in enumerate(powers) if power)
        previous = list(powers); previous[axis] -= 1
        previous_index = lookup[tuple(previous)]
        if previous_index >= i:
            raise ValueError('monomial recursion must reference an earlier rank')
        predecessors.append((previous_index, axis))
    primitives = [(power, a) for power, count in ((4, 210), (3, 70), (2, 15), (1, 1))
                  for a in monomials[:count]]
    primitive_lookup = {item: i for i, item in enumerate(primitives)}
    transform = [[(primitive_lookup[(power, residual)], coefficient)
                  for coefficient, residual, power in core.tensor_terms(a)] for a in monomials]
    assert len(primitives) == 296
    return {'monomials': monomials, 'predecessors': predecessors,
            'primitives': primitives, 'transform': transform}


def numpy():
    # Environment is fixed before importing NumPy. This new diagnostic does
    # not change the environment of another running worker.
    import numpy as np
    if np.finfo(np.longdouble).nmant < 63:
        raise RuntimeError('proposal requires extended precision with at least a 64-bit significand')
    return np


def primitive_batch(xs, weights, selected, layout, work=None):
    """One bounded chunk, no model/reducer calls or complex-dtype assumptions."""
    np = numpy(); dtype = np.longdouble
    size = xs.shape[1]
    if xs.shape != (6, size) or weights.shape != (size,):
        raise ValueError('parameter chunk shape mismatch')
    if xs.dtype != np.dtype(dtype) or weights.dtype != np.dtype(dtype):
        raise ValueError('parameter coordinates and weights must retain extended precision')
    momenta = np.zeros((4, size), dtype=dtype)
    freal = np.zeros(size, dtype=dtype); fimag = np.zeros(size, dtype=dtype)
    for i in range(6):
        for axis in range(4):
            momenta[axis] += dtype(selected['shifts'][i][axis])*xs[i]
        freal += dtype(selected['masses_squared'][i][0])*xs[i]
        fimag += dtype(str(selected['masses_squared'][i][1]))*xs[i]
    for i in range(6):
        for j in range(i+1, 6):
            freal -= dtype(selected['invariants'][i][j])*xs[i]*xs[j]
    denominator = freal*freal+fimag*fimag
    inverse = {1: (freal/denominator, -fimag/denominator)}
    for power in (2, 3, 4):
        a, b = inverse[power-1], inverse[1]
        inverse[power] = (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])
    if work is None:
        work = np.empty((210, size), dtype=dtype)
    matrix = work[:, :size]
    matrix[0] = 1
    for i, predecessor in enumerate(layout['predecessors'][1:], 1):
        previous, axis = predecessor
        np.multiply(matrix[previous], momenta[axis], out=matrix[i])
    output = np.empty((296, 2), dtype=dtype)
    start = 0
    for power, count in ((4, 210), (3, 70), (2, 15), (1, 1)):
        for part in (0, 1):
            # Contiguous last-axis reduction uses NumPy's partial-pairwise
            # summation. A compensated vector sum combines these chunks.
            output[start:start+count, part] = np.sum(matrix[:count]*(weights*inverse[power][part]), axis=1, dtype=dtype)
        start += count
    if not np.all(np.isfinite(output)):
        raise FloatingPointError('nonfinite direct parameter moment; no repair or fallback')
    return output


def combine(primitives, layout):
    np = numpy(); dtype = np.longdouble
    result = np.zeros((210, 2), dtype=dtype)
    for i, terms in enumerate(layout['transform']):
        for primitive, coefficient in terms:
            # All factors are exact dyadic rationals; no factorial weights
            # are applied to the actual Cartesian component afterward.
            value = dtype(coefficient.numerator)/dtype(coefficient.denominator)
            result[i] += value*primitives[primitive]
    return result


def parameter_chunk(rules, first, start, stop):
    np = numpy(); dtype = np.longdouble
    n = len(rules[0][0]); size = stop-start
    indices = [None]*5
    remaining = np.arange(start, stop, dtype=np.int64)
    for axis in (4, 3, 2, 1):
        indices[axis] = remaining % n
        remaining //= n
    xs = np.empty((6, size), dtype=dtype)
    residual = np.ones(size, dtype=dtype)
    weights = np.full(size, rules[0][1][first], dtype=dtype)
    for axis in range(5):
        nodes, rule_weights = rules[axis]
        t = nodes[first] if axis == 0 else nodes[indices[axis]]
        xs[axis] = residual*t
        residual *= 1-t
        if axis:
            weights *= rule_weights[indices[axis]]
    xs[5] = residual
    return xs, weights


def integrate(n, selected, core, layout, deadline, chunk_size=4096):
    np = numpy(); dtype = np.longdouble
    if not (1 <= chunk_size <= 8192):
        raise ValueError('chunk size must be between 1 and8192')
    rules = []
    rule_records = []
    for alpha in (4, 3, 2, 1, 0):
        records = core.gaussian_rule(n, alpha, deadline)
        # Parse the preserved high-precision decimal strings, not the old
        # source's binary64 node/weight fields.
        rules.append((np.array([r['node_decimal'] for r in records], dtype=dtype),
                      np.array([r['weight_decimal'] for r in records], dtype=dtype)))
        rule_records.append({'alpha': alpha, 'records': records})
    total = np.zeros((296, 2), dtype=dtype); compensation = np.zeros_like(total)
    work = np.empty((210, chunk_size), dtype=dtype)
    chunks = 0
    with np.errstate(invalid='raise', divide='raise', over='raise'):
        for first in range(n):
            for start in range(0, n**4, chunk_size):
                core.check_deadline(deadline)
                xs, weights = parameter_chunk(rules, first, start, min(start+chunk_size, n**4))
                value = primitive_batch(xs, weights, selected, layout, work)
                updated = total+value
                # Neumaier compensation acts on every real/imaginary
                # primitive independently and retains longdouble precision.
                compensation += np.where(np.abs(total) >= np.abs(value),
                                         (total-updated)+value, (value-updated)+total)
                total = updated
                chunks += 1
    primitives = total+compensation
    tensors = combine(primitives, layout)
    if not np.all(np.isfinite(tensors)):
        raise FloatingPointError('nonfinite combined tensor component')
    return primitives, tensors, {'chunks': chunks, 'chunk_size': chunk_size,
                                 'monomial_work_bytes': work.nbytes, 'rules': rule_records}


def number(value):
    np = numpy()
    return np.format_float_scientific(value, unique=False, precision=np.finfo(np.longdouble).precision+3, trim='k')


def saved_selected(n):
    if sha(QUAL/'index.json') != QUAL_INDEX_SHA256:
        raise ValueError('selected-component qualification index changed')
    index = json.loads((QUAL/'index.json').read_text())
    name = 'data/float18-result.json.gz' if n == 18 else 'data/float-result.json.gz'
    record = next(x for x in index['files'] if x['path'] == name)
    raw = (QUAL/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('saved selected quadrature changed')
    result = json.loads(gzip.decompress(raw))
    row = next((row for row in result['orders'] if row['n'] == n), None)
    return row, result


def compare_saved(n, tensors, core):
    if n not in (8, 18):
        return {'status': 'no preexisting selected-component order; no result is substituted'}
    row, source = saved_selected(n)
    if row is None:
        raise ValueError('requested existing selected order missing')
    comparisons = {}
    with localcontext() as context:
        context.prec = 100
        for name, metadata in source['targets'].items():
            index = metadata['index']
            for part, offset in (('real', 0), ('imaginary', 1)):
                observed = Decimal(number(tensors[index, offset]))
                saved = Decimal.from_float(row['results'][name][part])
                residual = abs(observed-saved)/abs(observed+saved) if observed+saved else Decimal(0) if observed == saved else Decimal('Infinity')
                comparisons[name+'_'+part] = {'new_extended_value': str(observed), 'saved_binary64_value': str(saved),
                    'absolute_difference': str(abs(observed-saved)), 'symmetric_residual': str(residual),
                    'agrees_at_1e12': residual <= Decimal('1e-12')}
    return {'status': 'same-order independent scalar/vector implementation comparison', 'parts': comparisons}


def proposal():
    core = load_core(); monomials = load_layout(core); layout = plan(core, monomials)
    fixture = core.fixture()
    return {'schema': 1, 'status': 'source-only proposal; actual all-component fixture integration not executed',
            'fixture_sha256': core.FIXTURE_SHA256, 'source_core_index_sha256': PROTO_INDEX_SHA256,
            'selected_qualification_index_sha256': QUAL_INDEX_SHA256,
            'components': len(monomials), 'primitive_moments': len(layout['primitives']),
            'primitive_counts': {'4': 210, '3': 70, '2': 15, '1': 1},
            'monomials': monomials,
            'transform': [[{'primitive': i, 'coefficient': str(c)} for i, c in terms] for terms in layout['transform']],
            'proposed_orders': [8, 16, 18], 'aggregate_seconds': 180, 'chunk_size': 4096,
            'worker_contract': {'cpu': 10, 'cpu_reservation': False, 'external_address_space_limit_bytes': 4294967296,
                                'single_thread_environment': {name: '1' for name in THREAD_VARS}},
            'precision_contract': 'NumPy longdouble extended precision; at least64 significand bits; not Fortran113bitQP',
            'summation': 'contiguous pairwise chunk reductions followed by Neumaier primitive-vector accumulation',
            'total_floating_error_formally_certified': False, 'native_reducer_calls': 0,
            'physical_amplitudes_qualified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('proposal')
    run = sub.add_parser('run')
    run.add_argument('--orders', nargs='+', type=int, default=[8, 16, 18])
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--seconds', type=int, default=180)
    run.add_argument('--chunk-size', type=int, default=4096)
    run.add_argument('--admit-physics', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'proposal':
        print(json.dumps(proposal(), indent=2)); return 0
    if not args.admit_physics:
        parser.error('root review is required before --admit-physics fixture evaluation')
    if args.output.exists() or any(n not in (8, 16, 18) for n in args.orders) or len(set(args.orders)) != len(args.orders):
        parser.error('new output and distinct proposed orders8/16/18 are required')
    if not (1 <= args.seconds <= 180 and 1 <= args.chunk_size <= 8192):
        parser.error('bounds: 1..180seconds, chunks1..8192')
    started = time.monotonic(); deadline = started+args.seconds
    result = proposal(); result['status'] = 'running'; result['orders'] = []
    np = numpy()
    result['numpy_version'] = np.__version__
    result['extended_dtype'] = {'name': str(np.dtype(np.longdouble)), 'itemsize': int(np.dtype(np.longdouble).itemsize),
                                'significand_bits': int(np.finfo(np.longdouble).nmant)+1,
                                'max_exponent': int(np.finfo(np.longdouble).maxexp)}
    result['actual_cpu_affinity'] = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    core = load_core(); layout = plan(core, load_layout(core))
    try:
        for n in args.orders:
            result['attempted_order'] = n
            tick = time.monotonic()
            primitives, tensors, metadata = integrate(n, core.fixture(), core, layout, deadline, args.chunk_size)
            record = {'n': n, 'nodes': n**5, 'elapsed_seconds': time.monotonic()-tick, **metadata,
                      'primitive_values': [{'inverse_F_power': p, 'powers': a, 'real': number(v[0]), 'imaginary': number(v[1])}
                                           for (p, a), v in zip(layout['primitives'], primitives)],
                      'tensor_values': [{'index': i, 'powers': a, 'real': number(v[0]), 'imaginary': number(v[1]),
                                         'analytic_error_bound_excludes_roundoff': core.exact_bounds(n, a)}
                                        for i, (a, v) in enumerate(zip(layout['monomials'], tensors))],
                      'saved_selected_comparison': {'status': 'pending; completed raw order is retained first'}}
            result['orders'].append(record)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            record['saved_selected_comparison'] = compare_saved(n, tensors, core)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            if any(not part['agrees_at_1e12'] for part in record['saved_selected_comparison'].get('parts', {}).values()):
                raise ArithmeticError('same-order selected-component comparison failed; completed raw values retained')
        result['status'] = 'completed estimates; total floating error not formally certified'
    except Exception as error:
        result['status'] = 'failed or timed out; earlier completed orders retained'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
    result['elapsed_seconds'] = time.monotonic()-started
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['status'].startswith('completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
