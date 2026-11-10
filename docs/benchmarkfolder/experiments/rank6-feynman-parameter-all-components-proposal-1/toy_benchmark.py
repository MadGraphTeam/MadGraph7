#!/usr/bin/env python3
"""Reproduce a synthetic array-cost check; no Gaussian rule or integral input.

This script does not import all_components.py, read a fixture/model/card/result,
or call a reducer. Its arbitrary Cartesian values and complex weights test only
the cost of a 210-monomial table and 296 prefix reductions. These measurements
must not be reported as physical-integral or amplitude performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
for name in THREADS:
    os.environ[name] = '1'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk', type=int, default=4096)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--seconds', type=int, default=30)
    args = parser.parse_args(argv)
    if not (1 <= args.chunk <= 8192 and 1 <= args.repetitions <= 7
            and 1 <= args.seconds <= 30):
        parser.error('bounds: chunks1..8192, repetitions1..7, seconds1..30')
    import numpy as np
    dtype = np.longdouble
    if np.finfo(dtype).nmant < 63:
        raise RuntimeError('at least64 significant bits required; this is not QP')
    started = time.monotonic()
    deadline = started + args.seconds
    basis = [(a, b, c, rank-a-b-c) for rank in range(7)
             for a in range(rank, -1, -1)
             for b in range(rank-a, -1, -1)
             for c in range(rank-a-b, -1, -1)]
    positions = {a: i for i, a in enumerate(basis)}
    parents = []
    for powers in basis[1:]:
        axis = next(i for i, power in enumerate(powers) if power)
        residual = list(powers)
        residual[axis] -= 1
        parents.append((positions[tuple(residual)], axis))
    assert len(basis) == 210 and all(j < i for i, (j, _) in enumerate(parents, 1))
    x = np.linspace(dtype('-.5'), dtype('.5'), args.chunk, dtype=dtype)
    cartesian = np.stack([x+dtype(i)/10 for i in range(4)])
    real_base = dtype('.003')*(1+dtype('.1')*x)
    imag_base = dtype('.00001')*(1+x)
    real, imag = real_base.copy(), imag_base.copy()
    weights_real, weights_imag = [], []
    for power in range(1, 5):
        if power > 1:
            real, imag = real*real_base-imag*imag_base, real*imag_base+imag*real_base
        weights_real.append(real/dtype(args.chunk))
        weights_imag.append(imag/dtype(args.chunk))
    matrix = np.empty((210, args.chunk), dtype=dtype)

    def table():
        matrix[0] = 1
        for i, (previous, axis) in enumerate(parents, 1):
            np.multiply(matrix[previous], cartesian[axis], out=matrix[i])

    def dot():
        return [(matrix[:count]@weights_real[k], matrix[:count]@weights_imag[k])
                for k, count in enumerate((1, 15, 70, 210))]

    def pairwise():
        return [(np.sum(matrix[:count]*weights_real[k], axis=1, dtype=dtype),
                 np.sum(matrix[:count]*weights_imag[k], axis=1, dtype=dtype))
                for k, count in enumerate((1, 15, 70, 210))]

    def measure(function):
        values = []
        for _ in range(args.repetitions):
            if time.monotonic() >= deadline:
                raise TimeoutError('synthetic microbenchmark cooperative cap reached')
            tick = time.monotonic()
            function()
            values.append(time.monotonic()-tick)
        return {'median_seconds': statistics.median(values), 'all_seconds': values}

    table()
    timings = {name: measure(function) for name, function in
               (('reused_monomial_workspace', table), ('prefix_dot', dot),
                ('prefix_pairwise_sum', pairwise))}
    left, right = dot(), pairwise()
    difference = max(np.max(np.abs(a-b)) for u, v in zip(left, right)
                     for a, b in zip(u, v))
    direct = np.empty_like(matrix)
    for i, powers in enumerate(basis):
        direct[i] = np.prod(np.stack([cartesian[j]**power
                                     for j, power in enumerate(powers)]), axis=0)
    monomial_difference = np.max(np.abs(matrix-direct))
    result = {
        'schema': 1,
        'status': 'synthetic array-cost observation only; no physical timings',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy_version': np.__version__, 'chunk': args.chunk,
        'repetitions': args.repetitions,
        'dtype': {'name': str(np.dtype(dtype)), 'itemsize': np.dtype(dtype).itemsize,
                  'significand_bits': int(np.finfo(dtype).nmant)+1,
                  'is_fortran_qp': False},
        'monomial_count': len(basis), 'primitive_count': 296,
        'monomial_workspace_bytes': matrix.nbytes,
        'single_thread_environment': {name: os.environ[name] for name in THREADS},
        'affinity_reserved': False,
        'timings': timings,
        'dot_vs_pairwise_max_absolute_difference': str(difference),
        'recurrence_vs_direct_power_max_absolute_difference': str(monomial_difference),
        'elapsed_seconds': time.monotonic()-started,
        'fixture_reads': 0, 'gaussian_rule_calls': 0, 'native_reducer_calls': 0,
        'physics_integrals_evaluated': 0,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
