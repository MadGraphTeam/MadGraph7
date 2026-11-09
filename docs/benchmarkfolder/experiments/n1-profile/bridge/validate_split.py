#!/usr/bin/env python3
"""Validate isolated full-DP direction workspaces against the profiling control.

Run from the MadGraph checkout, after both diagnostic DSOs have finished
building. The coordinator holds the numerical flock and loads no native DSO;
each library runs in a separate, sequential subprocess. This is correctness
evidence, not a performance benchmark or full-amplitude qualification.
"""
import argparse
import contextlib
import ctypes as C
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[5]
LOCK = ROOT / '.tirust-build/numerical-tests.lock'
DEFAULT_BUDGET = 268_435_456
COUNTERS = ['subsets_reused', 'subsets_computed', 'masters_reused',
            'masters_computed', 'moments_reused', 'moments_computed',
            'entries', 'bytes', 'not_stored']
SENTINEL = (999.0, 999.0)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def relative(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


class Complex(C.Structure):
    _fields_ = [('re', C.c_double), ('im', C.c_double)]


@dataclass(frozen=True)
class Fixture:
    name: str
    n: int
    rank: int
    shifts: tuple
    invariants: tuple
    masses: tuple
    mu2: float

    @property
    def ncoef(self):
        return math.comb(self.rank + 4, 4)


def triangle(name='massless', energy=91.188 / 2, longitudinal=1.0,
             mass=(0.0, 0.0), scale_factor=1.0, reverse=False):
    q = [(0.0, 0.0, 0.0, 0.0),
         (energy, 0.0, 0.0, energy * longitudinal),
         (2 * energy, 0.0, 0.0, 0.0)]
    # Preserve the exact supplied zeros of the actual n1 massless legs.
    light = 0.0 if longitudinal == 1.0 else energy**2 * (1 - longitudinal**2)
    hard = (2 * energy)**2
    matrix = [[0.0, light, hard], [light, 0.0, light], [hard, light, 0.0]]
    permutation = [0, 2, 1] if reverse else [0, 1, 2]
    return Fixture(name + ('-reversed' if reverse else '-primary'), 3, 2,
                   tuple(x for i in permutation for x in q[i]),
                   tuple(matrix[i][j] for j in permutation for i in permutation),
                   (mass,) * 3, hard * scale_factor)


def tadpole(mass=4.0):
    return Fixture('tadpole-' + str(mass), 1, 0, (0.0,) * 4, (0.0,),
                   ((mass, 0.0),), 4.0)


def scaleless_bubble():
    return Fixture('scaleless-bubble-rank6', 2, 6,
                   (0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0),
                   (0.0,) * 4, ((0.0, 0.0),) * 2, 4.0)


def complex_box():
    return Fixture('complex-box-rank4', 4, 4,
                   (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                   (0.0, -1.0, -1.0, -1.0, -1.0, 0.0, -2.0, -2.0,
                    -1.0, -2.0, 0.0, -2.0, -1.0, -2.0, -2.0, 0.0),
                   ((4.0, -0.125),) * 4, 4.0)


class Harness:
    def __init__(self, library):
        self.lib = C.CDLL(str(library))
        self.events = []
        self.fixtures = {}
        self.payloads = {}
        self.handle = C.c_void_p()
        self.scenario = None
        i32, u64, ptr = C.c_int32, C.c_uint64, C.c_void_p
        signatures = {
            'abi_version': ([], C.c_uint32),
            'context_create': ([C.c_uint32, C.POINTER(ptr)], i32),
            'context_destroy': ([ptr], i32),
            'clear_cache': ([ptr], i32),
            'register_plan': ([ptr, i32, i32], i32),
            'set_cache_budget': ([ptr, C.c_int64], i32),
            'cache_statistics': ([ptr, C.POINTER(u64), C.c_size_t], i32),
            'supports': ([i32, i32, i32], i32),
            'last_error': ([ptr, C.c_size_t], C.c_size_t),
            'profile_control': ([i32], i32),
            'evaluate_dp': ([ptr, i32, i32, C.POINTER(C.c_double), C.c_size_t,
                             C.POINTER(C.c_double), C.c_size_t,
                             C.POINTER(Complex), C.c_size_t, C.c_double,
                             C.POINTER(Complex), C.c_size_t,
                             i32, i32, i32, i32], i32),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, 'ml_tirust_' + name)
            function.argtypes = args
            function.restype = result
        require(self.lib.ml_tirust_abi_version() == 1, 'unexpected ABI version')
        require(self.lib.ml_tirust_profile_control(0) == 0, 'cannot disable timers')

    def stats(self):
        counts = (C.c_uint64 * 9)()
        require(self.lib.ml_tirust_cache_statistics(self.handle, counts, 9) == 0,
                'cache statistics failed')
        return list(counts)

    def error(self):
        buffer = C.create_string_buffer(self.lib.ml_tirust_last_error(None, 0))
        self.lib.ml_tirust_last_error(buffer, len(buffer))
        return buffer.value.decode('utf-8', errors='replace')

    @contextlib.contextmanager
    def context(self, scenario):
        self.scenario = scenario
        status = self.lib.ml_tirust_context_create(1, C.byref(self.handle))
        require(status == 0 and self.handle.value, 'context creation failed')
        self.events.append({'scenario': scenario, 'label': 'create', 'operation': 'create',
                            'status': status, 'after': self.stats()})
        try:
            yield
        finally:
            status = self.lib.ml_tirust_context_destroy(self.handle)
            self.events.append({'scenario': scenario, 'label': 'destroy',
                                'operation': 'destroy', 'status': status})
            self.handle = C.c_void_p()
            require(status == 0, 'context destruction failed')

    def command(self, label, name, *args):
        before = self.stats()
        status = getattr(self.lib, 'ml_tirust_' + name)(self.handle, *args)
        error = self.error() if status else ''
        after = self.stats()
        self.events.append({'scenario': self.scenario, 'label': label, 'operation': name,
                            'arguments': list(args), 'status': status, 'error': error,
                            'before': before, 'after': after})
        require(status == 0, f'{label}: {name} failed: {error}')
        if name == 'clear_cache':
            require(after == [0] * 9, label + ': clear did not reset counts')
        return before, after

    def evaluate(self, label, fixture, check=0, cache=1, poles=0, finite=1,
                 expected='success', invalid=None):
        self.fixtures[fixture.name] = asdict(fixture)
        shifts = (C.c_double * len(fixture.shifts))(*fixture.shifts)
        invariants = (C.c_double * len(fixture.invariants))(*fixture.invariants)
        masses = (Complex * fixture.n)(*(Complex(*mass) for mass in fixture.masses))
        output = (Complex * (4 * fixture.ncoef))(*(Complex(*SENTINEL)
                                                  for _ in range(4 * fixture.ncoef)))
        sentinel = bytes(output)
        shifts_length = len(shifts)
        if invalid == 'nan_mass':
            masses[0].re = float('nan')
        elif invalid == 'positive_width':
            masses[0].im = 0.125
        elif invalid == 'short_shifts':
            shifts_length -= 1
        elif invalid is not None:
            raise ValueError(invalid)
        before = self.stats()
        status = self.lib.ml_tirust_evaluate_dp(
            self.handle, fixture.n, fixture.rank, shifts, shifts_length,
            invariants, len(invariants), masses, len(masses), fixture.mu2,
            output, len(output), poles, finite, check, cache)
        # Other guarded APIs reset the diagnostic, so capture it immediately.
        error = self.error() if status else ''
        raw = bytes(output)
        after = self.stats()
        output_sha = hashlib.sha256(raw).hexdigest()
        self.payloads.setdefault(output_sha, raw.hex())
        self.events.append({
            'scenario': self.scenario, 'label': label, 'operation': 'evaluate_dp',
            'fixture': fixture.name, 'options': {'check': check, 'cache': cache,
                                                'poles': poles, 'finite': finite},
            'input_override': invalid, 'expected': expected, 'status': status,
            'error': error, 'before': before, 'after': after,
            'output_sha256': output_sha, 'output_bytes': len(raw),
            'failure_sentinel_unchanged': raw == sentinel if status else None,
            'finite_slot_semantics': 'computed' if finite else 'unevaluated zero placeholder'})
        if expected == 'success':
            require(status == 0, f'{label}: expected success, got {status}: {error}')
        elif expected == 'failure':
            require(status != 0, label + ': expected a reduction failure')
        elif expected == 'error':
            require(status == 2, f'{label}: expected input ERROR, got {status}')
        else:
            require(expected == 'either', 'invalid expected status policy')
        if status:
            require(raw == sentinel, label + ': failed ABI call modified output')
            require(bool(error), label + ': failure omitted diagnostic')
        else:
            require(all(math.isfinite(x) for z in output for x in (z.re, z.im)),
                    label + ': successful output contains nonfinite values')
            if not finite:
                require(all(z.re == z.im == 0.0 for z in output[3 * fixture.ncoef:]),
                        label + ': finite placeholder is not zero')
        if (check or not cache) and status == 0:
            require(before == after, label + ': bypass touched retained-store counters')
        if invalid is not None or (not finite and poles == 3):
            require(before == after, label + ': invalid request changed numerical cache')
        return raw, status


def same(left, right, label):
    require(left == right, label + ': per-input result changed')


def verify_selector(raw, full, nc, selection, label):
    values = list(struct.iter_unpack('=dd', raw))
    original = list(struct.iter_unpack('=dd', full))
    keep = [selection in (0, 1), selection in (0, 2), selection in (0, 2), True]
    for slot in range(4):
        for index in range(slot * nc, (slot + 1) * nc):
            if keep[slot]:
                require(values[index] == original[index], label + ': selected slot changed')
            else:
                require(values[index] == (0.0, 0.0), label + ': excluded pole was retained')


def triangle_cases(h):
    with h.context('triangle-lifecycle'):
        h.command('register', 'register_plan', 3, 2)
        fixtures = [triangle(reverse=bool(check)) for check in (0, 1)]
        reference = [None, None]
        for iteration in range(4):
            for check in (0, 1):
                result = h.evaluate(f'alternate-{iteration}-{check}', fixtures[check], check=check)
                if reference[check] is None:
                    reference[check] = result
                else:
                    same(result, reference[check], f'alternate-{iteration}-{check}')
        counts = h.stats()
        require(sum(counts[i] for i in (0, 2, 4)) > 0, 'repeated primary produced no cache hits')
        before, after = h.command('same-budget', 'set_cache_budget', DEFAULT_BUDGET)
        same(after, before, 'same-budget counters')
        for check in (0, 1):
            same(h.evaluate(f'cache-off-{check}', fixtures[check], check=check, cache=0),
                 reference[check], 'cache-off')
        # Change zero patterns, scales, and masses; revisit each ordering after
        # every mutation. Directions are compared to their own references.
        variants = [dict(name='rescaled', energy=60.0),
                    dict(name='off-lightlike', longitudinal=0.75),
                    dict(name='different-scale', scale_factor=1.5),
                    dict(name='massive-complex', mass=(4.0, -0.125))]
        for variant in variants:
            for check in (0, 1):
                fixture = triangle(reverse=bool(check), **variant)
                # These mutations exercise workspace history but have no
                # independent DP-success qualification. A refusal is reported
                # as such; cross-library status/output parity still applies.
                mutated = h.evaluate(f'mutate-{fixture.name}', fixture, check=check,
                                     expected='either')
                if mutated[1] == 0:
                    require(mutated[0] != reference[check][0], fixture.name + ': mutation had no effect')
                same(h.evaluate(f'restore-{fixture.name}', fixtures[check], check=check),
                     reference[check], 'restore original geometry')
        # check_pass is a policy flag, not a promise about input ordering.
        for order, fixture in enumerate(fixtures):
            for check in (0, 1):
                same(h.evaluate(f'cross-order-{order}-check-{check}', fixture, check=check),
                     reference[order], 'cross-order workspace selection')
        for check in (0, 1):
            for selection in (1, 2, 3):
                selected, _ = h.evaluate(f'select-{check}-{selection}', fixtures[check],
                                         check=check, poles=selection)
                verify_selector(selected, reference[check][0], fixtures[check].ncoef,
                                selection, 'pole selection')
        for check in (0, 1):
            for selection in (0, 1, 2):
                partial = h.evaluate(f'partial-{check}-{selection}', fixtures[check],
                                     check=check, finite=0, poles=selection)
                same(h.evaluate(f'partial-repeat-{check}-{selection}', fixtures[check],
                                check=check, finite=0, poles=selection), partial,
                     'pole-only repeated result')
            h.evaluate(f'invalid-empty-poles-{check}', fixtures[check], check=check,
                       finite=0, poles=3, expected='error')
        h.command('zero-budget', 'set_cache_budget', 0)
        require(h.stats() == [0] * 9, 'changed budget did not clear numerical caches')
        for check in (0, 1, 0, 1):
            same(h.evaluate(f'zero-budget-call-{len(h.events)}', fixtures[check], check=check),
                 reference[check], 'zero-budget result')
            counts = h.stats()
            require(counts[6] == counts[7] == 0, 'zero budget retained entries or bytes')
        require(h.stats()[8] > 0, 'zero-budget path did not record refused insertion')
        before, after = h.command('same-zero-budget', 'set_cache_budget', 0)
        same(after, before, 'same-zero-budget counters')
        h.command('restore-budget', 'set_cache_budget', DEFAULT_BUDGET)
        require(h.stats() == [0] * 9, 'restored budget did not clear counters')
        for check in (0, 1):
            same(h.evaluate(f'after-budget-restore-{check}', fixtures[check], check=check),
                 reference[check], 'budget restoration')
        h.command('explicit-clear', 'clear_cache')
        for check in (0, 1):
            same(h.evaluate(f'after-clear-{check}', fixtures[check], check=check),
                 reference[check], 'explicit cache clearing')


def failure_cases(h):
    # Dedicated full-only contexts: aggregate entries must be zero after a
    # failed full reduction, without unrelated pole/QP entries in the count.
    for failed_direction in (0, 1):
        with h.context(f'failure-direction-{failed_direction}'):
            h.command('register', 'register_plan', 1, 0)
            original = [h.evaluate(f'warm-{check}', tadpole(), check=check) for check in (0, 1)]
            require(h.stats()[6] > 0, 'failure fixture did not populate a cache')
            h.evaluate('overflow', tadpole(1e308), check=failed_direction, expected='failure')
            require(h.stats()[6] == h.stats()[7] == 0, 'failed reduction retained cache entries')
            for check in (0, 1):
                same(h.evaluate(f'recover-{check}', tadpole(), check=check),
                     original[check], 'both-direction recovery')
            for invalid in ('nan_mass', 'positive_width', 'short_shifts'):
                h.evaluate('invalid-' + invalid, tadpole(), invalid=invalid, expected='error')
            for check in (0, 1):
                same(h.evaluate(f'after-invalid-{check}', tadpole(), check=check),
                     original[check], 'input-error recovery')


def supplemental_cases(h):
    fixture = scaleless_bubble()
    with h.context('scaleless-rank6'):
        h.command('register', 'register_plan', fixture.n, fixture.rank)
        reference = h.evaluate('primary', fixture)
        values = list(struct.iter_unpack('=dd', reference[0]))
        index = 0
        # Exact upstream IREGI ordering, not MadGraph's polynomial ordering:
        # crates/iregi-rust/src/basis.rs::tensor_basis, lines 16--25.
        for rank in range(7):
            for time in range(rank, -1, -1):
                for x in range(rank - time, -1, -1):
                    for y in range(rank - time - x, -1, -1):
                        expected = 2.0**rank / (rank + 1) if x + y == 0 else 0.0
                        require(abs(values[index][0] - expected) < 1e-12, 'rank6 UV moment')
                        require(abs(values[210 + index][0] + expected) < 1e-12, 'rank6 IR1 moment')
                        require(values[420 + index] == (0.0, 0.0), 'rank6 IR2 moment')
                        require(abs(values[630 + index][0]) < 1e-12, 'rank6 finite moment')
                        require(all(abs(values[slot * 210 + index][1]) < 1e-12
                                    for slot in range(4)), 'rank6 imaginary moment')
                        index += 1
        require(index == 210, 'rank6 basis size')
        for check, cache in ((0, 1), (1, 1), (0, 0), (1, 0)):
            same(h.evaluate(f'repeat-{check}-{cache}', fixture, check=check, cache=cache),
                 reference, 'rank6 repeat/check/cache parity')
        partial = h.evaluate('pole-only', fixture, finite=0)
        same(h.evaluate('pole-only-check', fixture, finite=0, check=1), partial,
             'rank6 pole-only check parity')
    with h.context('complex-mass-rank4'):
        fixture = complex_box()
        h.command('register', 'register_plan', fixture.n, fixture.rank)
        # The source Rust fixture establishes MP convergence, not a DP status
        # guarantee. Record and compare the actual DP status without claiming
        # successful qualification if either library refuses this fixture.
        reference = h.evaluate('primary', fixture, expected='either')
        for check, cache in ((0, 1), (1, 1), (0, 0), (1, 0)):
            same(h.evaluate(f'repeat-{check}-{cache}', fixture, check=check, cache=cache,
                            expected='either'), reference, 'complex box repeat/cache parity')


def identity(library):
    paths = [Path(__file__).resolve(), ROOT / 'vendor/tirust/tirust.h',
             ROOT / 'vendor/tirust/src/lib.rs']
    basis = ROOT / '.tirust-build/pole-development/crates/iregi-rust/src/basis.rs'
    if basis.is_file():
        paths.append(basis)
    paths.extend(path for path in (library.parent / 'src/lib.rs',
                                  library.parent / 'src/profile.rs',
                                  library.parent / 'preparation.json',
                                  library.parent / 'build-metadata.json') if path.is_file())
    return {'library': relative(library), 'library_sha256': digest(library),
            'source_and_metadata_sha256': {relative(path): digest(path) for path in paths}}


def worker(args):
    require(args.lock_fd is not None, 'worker requires inherited numerical-lock descriptor')
    require(os.fstat(args.lock_fd).st_ino == LOCK.stat().st_ino,
            'worker lock descriptor does not refer to numerical lock')
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    library = args.library.resolve()
    info = identity(library)
    require(info['library_sha256'] == args.expected_library_sha256, 'library changed before worker')
    h = None
    data = {'schema_version': 1, 'state': 'running', 'performance_eligible': False,
            'identity': info, 'cache_counter_names': COUNTERS,
            'output_encoding': 'hex of native C struct {double re,im}; coefficient-fast slots UV1,IR1,IR2,finite',
            'analytic_basis_order': 'Upstream tensor_basis: total rank ascending, then time/x/y exponents descending; z is the remainder. This is IREGI ordering, before Fortran MadGraph remapping.',
            'byte_order': sys.byteorder, 'checks': [],
            'scope': 'Synthetic raw-ABI correctness/lifecycle tests; no complete amplitudes or timing.'}
    status = 0
    try:
        h = Harness(library)
        for name, cases in [('triangle_lifecycle', triangle_cases),
                            ('failure_cleanup_and_recovery', failure_cases),
                            ('supplemental_fixtures', supplemental_cases)]:
            cases(h)
            data['checks'].append(name)
        require(digest(library) == info['library_sha256'], 'library changed during worker')
        data['state'] = 'complete'
    except Exception as exc:
        status = 1
        data.update(state='failed', failure=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc())
    finally:
        if h is not None:
            data.update(events=h.events, fixtures=h.fixtures, output_payloads=h.payloads)
            data['optional_fixture_outcomes'] = [
                {'scenario': event['scenario'], 'label': event['label'],
                 'fixture': event['fixture'], 'status': event['status'],
                 'outcome': 'numerically_evaluated' if event['status'] == 0 else 'refused_output_unchanged'}
                for event in h.events if event.get('expected') == 'either']
        write_json(args.worker_output, data)
    return status


def compare(original, split):
    differences, cache_differences = [], []
    if original['state'] != 'complete' or split['state'] != 'complete':
        differences.append({'kind': 'worker_failure', 'original': original['state'],
                            'split': split['state']})
    if original.get('fixtures') != split.get('fixtures'):
        differences.append({'kind': 'fixture_definitions_differ'})
    left, right = original.get('events', []), split.get('events', [])
    if len(left) != len(right):
        differences.append({'kind': 'event_count', 'original': len(left), 'split': len(right)})
    fields = ('scenario', 'label', 'operation', 'arguments', 'fixture', 'options',
              'input_override', 'expected', 'status', 'output_sha256',
              'failure_sentinel_unchanged')
    for index, (a, b) in enumerate(zip(left, right)):
        changed = {key: {'original': a.get(key), 'split': b.get(key)} for key in fields
                   if a.get(key) != b.get(key)}
        if changed:
            differences.append({'event': index, 'label': a['label'], 'changes': changed})
        for snapshot in ('before', 'after'):
            if a.get(snapshot) != b.get(snapshot):
                cache_differences.append({'event': index, 'label': a['label'],
                                          'snapshot': snapshot, 'original': a.get(snapshot),
                                          'split': b.get(snapshot)})
    return {'passed': not differences, 'differences': differences,
            'cache_statistics_identical': not cache_differences,
            'cache_statistics_differences': cache_differences,
            'cache_comparison_policy': 'Counters are reported separately; each worker strictly checks bypass, same-budget preservation, zero-budget exclusion, and failure clearing.',
            'exact_output_comparison': 'Per-call byte identity, including signed zeros, between original and split for the same input and options.',
            'failure_removal_limit': 'External recovery and empty-store assertions cannot inspect workspace object destruction; removal of both directions is established by the isolated source diff.'}


def coordinator(args):
    output = args.output.resolve()
    directory = output.parent / (output.stem + '-workers')
    require(not output.exists() and not directory.exists(),
            'use a fresh --output path; existing evidence is never reused or overwritten')
    directory.mkdir(parents=True)
    libraries = [('original', args.original_library.resolve()), ('split', args.split_library.resolve())]
    records = {}
    workers = {}
    # Only this process owns the lock lifetime. Workers inherit its locked
    # descriptor, run sequentially, and never load the other library.
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, library in libraries:
            worker_output = directory / (name + '.json')
            log_path = directory / (name + '.log')
            expected_sha = digest(library)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', name,
                       '--library', str(library), '--worker-output', str(worker_output),
                       '--expected-library-sha256', expected_sha, '--lock-fd', str(lock.fileno())]
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            with log_path.open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        pass_fds=(lock.fileno(),), stdout=log,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
            require(worker_output.is_file(), f'{name} worker produced no JSON; inspect {log_path}')
            records[name] = json.loads(worker_output.read_text())
            require(records[name]['identity']['library_sha256'] == expected_sha,
                    f'{name} worker identity does not match the requested library')
            workers[name] = {'returncode': result.returncode, 'command': command,
                             'result': relative(worker_output), 'result_sha256': digest(worker_output),
                             'log': relative(log_path), 'log_sha256': digest(log_path)}
    comparison = compare(records['original'], records['split'])
    for name, record in workers.items():
        if record['returncode']:
            comparison['passed'] = False
            comparison['differences'].append({'kind': 'worker_returncode',
                                              'worker': name, 'returncode': record['returncode']})
    data = {'schema_version': 1, 'state': 'complete' if comparison['passed'] else 'failed',
            'performance_eligible': False, 'source_sha256': digest(__file__),
            'workers': workers, 'comparison': comparison,
            'identities': {name: record['identity'] for name, record in records.items()},
            'call_counts': {name: sum(e['operation'] == 'evaluate_dp' for e in record.get('events', []))
                            for name, record in records.items()},
            'optional_fixture_refusals': {
                name: [event for event in record.get('optional_fixture_outcomes', []) if event['status']]
                for name, record in records.items()},
            'scope': 'Sequential, isolated raw-ABI DP tests. No timing or complete-amplitude claim; QP is unchanged by the experiment and not exercised.'}
    write_json(output, data)
    print(json.dumps({'state': data['state'], 'output': relative(output),
                      'call_counts': data['call_counts'],
                      'cache_statistics_identical': comparison['cache_statistics_identical']}))
    return 0 if comparison['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-library', type=Path,
                        default=ROOT / '.tirust-build/n1-profile/bridge/libmadloop_tirust.so')
    parser.add_argument('--split-library', type=Path,
                        default=ROOT / '.tirust-build/n1-profile/split-bridge/libmadloop_tirust.so')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=int, default=900, help='Maximum seconds per sequential worker')
    parser.add_argument('--worker', choices=('original', 'split'), help=argparse.SUPPRESS)
    parser.add_argument('--library', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--expected-library-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'run from the MadGraph checkout')
    if args.worker:
        return worker(args)
    require(args.output is not None, '--output is required')
    require(args.timeout > 0, '--timeout must be positive')
    return coordinator(args)


if __name__ == '__main__':
    raise SystemExit(main())
