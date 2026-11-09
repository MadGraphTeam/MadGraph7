#!/usr/bin/env python3
"""Execute extracted production rescue dispatch against the retained stress lane.

The numerical section holds the repository-wide exclusive numerical-tests lock.
This is a synthetic callback experiment, not the full physical matrix element.
"""
import argparse
import ctypes as c
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from madgraph.various.q_polynomial import FortranPolynomialRoutines


class Complex(c.Structure):
    _fields_ = [('re', c.c_double), ('im', c.c_double)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--native-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error(f'Run from {ROOT}')
    if args.output.exists():
        parser.error('Output exists; choose a new evidence file.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / '.tirust-build/numerical-tests.lock'
    with lock_path.open('a') as lock:
        print('Waiting for numerical-tests lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print('Acquired numerical-tests lock', flush=True)
        reference = json.loads(args.native_report.read_text())
        lane = reference['generated_adapter']['input']
        spec = importlib.util.spec_from_file_location('stability_transforms',
                                                     Path(__file__).with_name('check_stability.py'))
        transforms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transforms)
        fixtures = [lane, transforms.rotate(lane, 1), transforms.rotate(lane, 2)]
        positions = {tuple(a): i for i, a in
                     enumerate(FortranPolynomialRoutines(5).get_COLLIER_mapping())}
        pl, coefficients = [], []
        for fixture in fixtures:
            shifts = fixture['shifts']
            pl.extend(shifts[(i + 1) % 4][k] - shifts[i][k] for i in range(4) for k in range(4))
            numerator = [0j] * 126
            for alpha, value in fixture['coefficients']:
                numerator[positions[tuple(alpha)]] += complex(*value) * (-1) ** sum(alpha)
            coefficients.extend(Complex(z.real, z.imag) for z in numerator)
        pl_buffer = (c.c_double * 48)(*pl)
        mass_buffer = (Complex * 4)(*(Complex(*z) for z in lane['masses']))
        coefficient_buffer = (Complex * 378)(*coefficients)
        lib = c.CDLL(str(args.library.resolve()))
        call = lib.tirust_host_rescue
        call.argtypes = [c.c_double, c.c_int, c.c_int, c.c_void_p, c.c_void_p,
                         c.c_double, *([c.c_void_p] * 11)]
        call.restype = c.c_int
        report = {
            'label': reference['label'], 'state': 'running',
            'library_sha256': digest(args.library),
            'native_report_sha256': digest(args.native_report),
            'runner_sha256': digest(__file__),
            'build_metadata': json.loads((args.library.parent / 'build-metadata.json').read_text()),
            'thresholds': [1e-3, 1e-6], 'rows': [],
            'limitation': 'Executed production rescue block with synthetic real/imaginary integral callbacks; '
                          'full SLOOPMATRIX physics/filter initialization is not executed.',
        }
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        for threshold in report['thresholds']:
            for rotations in (0, 2):
                for projection, name in enumerate(('real', 'imaginary')):
                    result = (c.c_double * 3)()
                    accuracy, dp_acc, qp_acc = c.c_double(), c.c_double(), c.c_double()
                    retcode, count = c.c_int(), c.c_int()
                    modes, rotation_log, statuses = (c.c_int * 8)(), (c.c_int * 8)(), (c.c_int * 8)()
                    evaluations = (c.c_double * 24)()
                    status = call(threshold, rotations, projection, pl_buffer, mass_buffer,
                                  lane['mu2'], coefficient_buffer, result, c.byref(accuracy),
                                  c.byref(retcode), c.byref(count), modes, rotation_log, statuses,
                                  evaluations, c.byref(dp_acc), c.byref(qp_acc))
                    truth = [z[projection] for z in lane['truth']]
                    qp_truth = [z[projection] for z in reference['generated_adapter']['qp']['primary']['coefficients']]
                    scale = max(map(abs, truth))
                    qp_scale = max(map(abs, qp_truth))
                    row = {'threshold': threshold, 'rotations': rotations, 'projection': name,
                           'status': status, 'return_code': retcode.value, 'accuracy': accuracy.value,
                           'dp_accuracy': dp_acc.value, 'qp_accuracy': qp_acc.value,
                           'calls': count.value, 'modes': list(modes)[:count.value],
                           'rotation_indices': list(rotation_log)[:count.value],
                           'backend_statuses': list(statuses)[:count.value], 'result': list(result),
                           'evaluations': [list(evaluations)[3*i:3*i+3] for i in range(count.value)],
                           'fixture_truth': truth, 'same_input_native_qp_truth': qp_truth,
                           'relative_error_to_fixture_truth': max(abs(a-b) for a,b in zip(result,truth))/scale,
                           'relative_error_to_same_input_qp': max(abs(a-b) for a,b in zip(result,qp_truth))/qp_scale}
                    report['rows'].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + '\n')
                    print(json.dumps({k: row[k] for k in ('threshold','rotations','projection','return_code',
                          'accuracy','modes','relative_error_to_same_input_qp')}), flush=True)
                    if status != 0 or any(row['backend_statuses']):
                        raise RuntimeError('Host harness or numerical callback failed; partial evidence retained.')
        report['state'] = 'complete'
        report['all_default_kept_dp'] = all(r['return_code'] == 219 for r in report['rows'] if r['threshold'] == 1e-3)
        report['all_diagnostic_rescued_qp'] = all(r['return_code'] == 323 for r in report['rows'] if r['threshold'] == 1e-6)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        fcntl.flock(lock, fcntl.LOCK_UN)
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
