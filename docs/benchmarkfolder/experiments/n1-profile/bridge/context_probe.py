#!/usr/bin/env python3
"""Diagnose alternating lightlike variants with the existing profiling DSO."""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


class Complex(C.Structure):
    _fields_ = [("re", C.c_double), ("im", C.c_double)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10000)
    parser.add_argument("--cpu", type=int, default=1)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    lib = C.CDLL(str(args.library.resolve()))
    lib.ml_tirust_context_create.argtypes = [C.c_uint32, C.POINTER(C.c_void_p)]
    lib.ml_tirust_context_destroy.argtypes = [C.c_void_p]
    lib.ml_tirust_clear_cache.argtypes = [C.c_void_p]
    lib.ml_tirust_register_plan.argtypes = [C.c_void_p, C.c_int32, C.c_int32]
    lib.ml_tirust_profile_control.argtypes = [C.c_int32]
    lib.ml_tirust_profile_json.argtypes = [C.c_void_p, C.c_size_t]
    lib.ml_tirust_profile_json.restype = C.c_size_t
    lib.ml_tirust_last_error.argtypes = [C.c_void_p, C.c_size_t]
    lib.ml_tirust_last_error.restype = C.c_size_t
    lib.ml_tirust_evaluate_dp.argtypes = [C.c_void_p, C.c_int32, C.c_int32,
        C.POINTER(C.c_double), C.c_size_t, C.POINTER(C.c_double), C.c_size_t,
        C.POINTER(Complex), C.c_size_t, C.c_double, C.POINTER(Complex), C.c_size_t,
        C.c_int32, C.c_int32, C.c_int32, C.c_int32]
    mass = 91.188
    q = [[0., 0., 0., 0.], [mass / 2, 0., 0., mass / 2], [mass, 0., 0., 0.]]
    shifts = [(C.c_double * 12)(*(v for row in points for v in row))
              for points in (q, [q[0], q[2], q[1]])]
    inv = [(C.c_double * 9)(*values) for values in
           ([0., 0., mass * mass, 0., 0., 0., mass * mass, 0., 0.],
            [0., mass * mass, 0., mass * mass, 0., 0., 0., 0., 0.])]
    masses = (Complex * 3)()
    out = (Complex * 60)()
    reference = [None, None]
    results = []
    with Path('.tirust-build/numerical-tests.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for repetition in range(3):
            cases = [(level, split) for level in (0, 2) for split in (False, True)]
            if repetition % 2:
                cases.reverse()
            for level, split in cases:
                contexts = []
                for _ in range(2 if split else 1):
                    handle = C.c_void_p()
                    assert lib.ml_tirust_context_create(1, C.byref(handle)) == 0
                    assert lib.ml_tirust_register_plan(handle, 3, 2) == 0
                    contexts.append(handle)

                def evaluate(check):
                    handle = contexts[check if split else 0]
                    assert lib.ml_tirust_clear_cache(handle) == 0
                    status = lib.ml_tirust_evaluate_dp(handle, 3, 2, shifts[check], 12,
                        inv[check], 9, masses, 3, mass * mass, out, 60, 0, 1, check, 1)
                    if status != 0:
                        error = C.create_string_buffer(lib.ml_tirust_last_error(None, 0))
                        lib.ml_tirust_last_error(error, len(error))
                        raise RuntimeError(error.value.decode())
                    raw = bytes(out)
                    if reference[check] is None:
                        reference[check] = raw
                    assert raw == reference[check], 'output changed with context selection'

                assert lib.ml_tirust_profile_control(0) == 0
                for _ in range(100):
                    evaluate(0); evaluate(1)
                assert lib.ml_tirust_profile_control(level) == 0
                started = time.perf_counter()
                for _ in range(args.pairs):
                    evaluate(0); evaluate(1)
                elapsed = time.perf_counter() - started
                buffer = C.create_string_buffer(lib.ml_tirust_profile_json(None, 0))
                lib.ml_tirust_profile_json(buffer, len(buffer))
                profile = json.loads(buffer.value)
                for row in profile['rows']:
                    assert row['calls'] == args.pairs
                    assert row['direct_lanes'] == args.pairs
                    assert row['failed_calls'] == row['refused_lanes'] == 0
                    assert row['precision_tier_lanes'] == row['method_choice_lanes'] == 0
                assert len(profile['rows']) == (2 if level else 0)
                results.append({'repetition': repetition, 'split_contexts': split,
                    'profile_level': level,
                    'python_loop_wall_seconds': elapsed, 'profile': profile})
                for handle in contexts:
                    assert lib.ml_tirust_context_destroy(handle) == 0
    result = {'performance_eligible': False,
        'scope': 'Fixed massless triangle, two orderings; internal timers compare one vs two contexts, not complete amplitudes.',
        'all_evaluations_bit_identical_to_per_order_reference': True,
        'pairs_per_run': args.pairs, 'cpu': args.cpu,
        'library': str(args.library),
        'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shifts': [list(v) for v in shifts], 'invariants': [list(v) for v in inv],
        'mass_squared': 0., 'scale_squared': mass * mass,
        'reference_output_sha256': [hashlib.sha256(v).hexdigest() for v in reference],
        'runs': results}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for run in results:
        rows = run['profile']['rows']
        print(run['repetition'], run['profile_level'], run['split_contexts'],
              {key: sum(row[key] for row in rows) / args.pairs / 1000
               for key in ['ffi_total_ns', 'reduce_batch_ns', 'upstream_arguments_ns']})


if __name__ == '__main__':
    main()
