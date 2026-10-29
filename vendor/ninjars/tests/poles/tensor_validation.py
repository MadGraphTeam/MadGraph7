#!/usr/bin/env python3
"""Compare NinjaRS UV/IR tensor poles with independent COLLIER references.

Both precisions and every tensor component run by default. Native DP combined
and no-pole results are retained as diagnostics, while selected DP and every MP
mode are checked against COLLIER. --diagnostic-all makes ordinary DP numerical
discrepancies fatal too; see README.md for the known pre-existing controls.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
FIELDS = ('finite', 'uv', 'ir1', 'ir2')
FOCUSED = {(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (2, 0, 0, 0),
           (1, 1, 0, 0), (0, 2, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def monomials(rank):
    """Ninja/MadLoop tensor order, without multinomial symmetry factors."""
    terms = [(0, 0, 0, 0)]
    lower = list(terms)
    for r in range(1, rank + 1):
        block = []
        for index, length in enumerate([1, r, r*(r+1)//2, r*(r+1)*(r+2)//6]):
            for old in lower[:length]:
                term = list(old)
                term[index] += 1
                block.append(tuple(term))
        terms += block
        lower = block
    return terms


def write_jobs(cases, work, focused):
    selected, rows = [], []
    if not cases:
        raise ValueError('empty reference case list')
    for case in cases:
        powers = [tuple(component['powers']) for component in case['components']]
        if len(powers) != math.comb(case['rank']+4, 4) or set(powers) != set(monomials(case['rank'])):
            raise ValueError('incomplete or duplicate reference tensor: ' + case['name'])
        if case['collier_status'] != dict(error=0, accuracy=0):
            raise ValueError('COLLIER reference has a nonzero status: ' + case['name'])
        for component in case['components']:
            powers = tuple(component['powers'])
            if focused and powers not in FOCUSED:
                continue
            rank = sum(powers)
            # All components in this norm have the same mass dimension.
            scale = max(abs(complex(*other[key])) for other in case['components']
                        if sum(other['powers']) == rank for key in FIELDS)
            if not scale:
                # A complete odd tadpole tensor vanishes by symmetry. The
                # nonzero scalar tensor at this same point fixes its scale;
                # the characteristic mass restores the correct dimension.
                characteristic = max([case['mu2']] + [abs(complex(*m)) for m in case['mass2']])
                scalar = next(c for c in case['components'] if not any(c['powers']))
                scale = max(abs(complex(*scalar[key])) for key in FIELDS) * characteristic**(rank/2)
            selected.append(dict(case=case['name'], powers=powers,
                                 reference=component, common_scale=scale))
            n = case['n']
            rows.append(f'{n} {rank}')
            rows += ['1 0' if term == powers else '0 0' for term in monomials(rank)]
            rows += [' '.join(map(str, mass)) for mass in case['mass2']]
            rows += [' '.join(map(str, p)) for p in case['shifts']]
            for a in case['shifts']:
                invariants = []
                for b in case['shifts']:
                    d = [x-y for x, y in zip(a, b)]
                    invariants.append(d[0]**2-sum(x*x for x in d[1:]))
                rows.append(' '.join(map(str, invariants)))
            rows.append(str(case['mu2']))
    inputs = work / 'jobs.in'
    inputs.write_text(str(len(selected)) + '\n' + '\n'.join(rows) + '\n')
    (work / 'jobs.json').write_text(json.dumps(selected, indent=2) + '\n')
    return inputs, selected


def compare(work, selected, precision, diagnostic_all):
    failures, diagnostics, unstable, history = [], [], [], {}
    worst, count, selected_count, repeat_count = 0., 0, 0, 0
    sequence = [0, 1, 2, 3, 0, 2, 1, 3]
    expected_sequence = [(job, mode) for job in range(1, len(selected)+1) for mode in sequence]
    actual_sequence = []
    lines = (work / 'values.dat').read_text().splitlines()
    if len(lines) != len(expected_sequence):
        failures.append(dict(reason='incomplete or excess result rows', expected=len(expected_sequence), actual=len(lines)))
    for line in lines:
        fields = line.split()
        if len(fields) != 11:
            failures.append(dict(reason='malformed result row', row=line))
            continue
        job, mode, status = map(int, fields[:3])
        actual_sequence.append((job, mode))
        if not 1 <= job <= len(selected) or mode not in (0, 1, 2, 3):
            failures.append(dict(reason='unexpected job or mode', job=job, mode=mode))
            continue
        nums = list(map(float, fields[3:]))
        actual = [complex(*nums[k:k+2]) for k in range(0, 6, 2)]
        case = selected[job-1]
        ref = case['reference']
        expected = [complex(*ref['finite']),
                    complex(*ref['uv']) + complex(*ref['ir1']) if mode == 0 else
                    complex(*ref['uv' if mode == 1 else 'ir1']) if mode in (1, 2) else 0j,
                    complex(*ref['ir2']) if mode in (0, 2) else 0j]
        scale = case['common_scale']
        entry = dict(job=job, case=case['case'], powers=case['powers'], mode=mode, status=status)
        checked = precision == 'mp' or mode in (1, 2) or diagnostic_all
        count += 1
        selected_count += checked
        # Every mode is evaluated twice, with other modes interleaved. Compare
        # all printed binary128 digits, including the rational diagnostic.
        key = job, mode
        state = fields[2:]
        if key in history:
            repeat_count += 1
            if state != history[key]:
                failures.append(dict(entry, reason='reused context differs after mode cycling'))
        else:
            history[key] = state
        if status == 1:
            unstable.append(entry)
            if checked:
                failures.append(dict(entry, reason='reduction reported unstable'))
            continue
        ratios = []
        for x, y in zip(actual, expected):
            tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scale
            ratios.append(abs(x-y)/tolerance if tolerance else (0. if x == y else float('inf')))
        if checked:
            worst = max(worst, max(ratios))
        nonfinite = not all(math.isfinite(x.real) and math.isfinite(x.imag) for x in actual)
        if status != 0 or max(ratios) > 1 or nonfinite:
            error = dict(entry, actual=[[x.real, x.imag] for x in actual],
                         expected=[[x.real, x.imag] for x in expected], error_ratios=ratios)
            (failures if checked or status not in (0, 1) or nonfinite else diagnostics).append(error)
    if actual_sequence != expected_sequence:
        failures.append(dict(reason='unexpected/missing/duplicated evaluation sequence'))
    result = dict(passed=not failures, cases=len(selected), evaluations=count,
                  checked_against_collier=selected_count, repeated_state_checks=repeat_count,
                  precision=precision, diagnostic_all=diagnostic_all,
                  worst_checked_normalized_error=worst, failures=failures,
                  ordinary_dp_diagnostics=diagnostics, unstable=unstable)
    (work / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lib', type=Path, required=True, help='installed NinjaRS lib directory')
    parser.add_argument('--output', type=Path, required=True, help='build and evidence directory')
    parser.add_argument('--reference', type=Path, default=HERE / 'collier_reference.json')
    parser.add_argument('--precision', choices=['dp', 'mp', 'both'], default='both')
    parser.add_argument('--focused', action='store_true', help='61 representative components instead of all 455')
    parser.add_argument('--diagnostic-all', action='store_true')
    parser.add_argument('--fc', default=os.environ.get('FC', 'gfortran'))
    args = parser.parse_args()
    output, lib = args.output.resolve(), args.lib.resolve()
    if output == HERE or HERE in output.parents:
        parser.error('--output must be outside the retained test-source directory')
    output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.reference.read_text())
    summaries = []
    precisions = ['dp', 'mp'] if args.precision == 'both' else [args.precision]
    for precision in precisions:
        work = output / precision
        work.mkdir(exist_ok=True)
        inputs, selected = write_jobs(cases, work, args.focused)
        command = shlex.split(args.fc) + ['-O2', str(HERE / 'ninjars_poles.f90'), '-I'+str(lib)]
        command += shlex.split((lib / 'ninjars-link-before.txt').read_text())
        command += ['-L'+str(lib), '-lninjars']
        command += shlex.split((lib / 'ninjars-native-libs.txt').read_text())
        command += ['-o', str(work / 'ninjars_poles')]
        with (work / 'build.log').open('w') as stream:
            subprocess.run(command, cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=True)
        with (work / 'evaluation.log').open('w') as stream:
            subprocess.run([str(work / 'ninjars_poles'), str(inputs), str(work / 'values.dat'), precision],
                           cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = compare(work, selected, precision, args.diagnostic_all)
        summary = {key: value for key, value in result.items()
                   if key not in ('failures', 'ordinary_dp_diagnostics', 'unstable')}
        summary.update(failures=len(result['failures']),
                       ordinary_dp_diagnostics=len(result['ordinary_dp_diagnostics']),
                       unstable=len(result['unstable']))
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    # An ordinary DP unstable control is only qualified by a successful MP
    # evaluation of every corresponding mode at the same input in this run.
    rescued = None
    if args.precision == 'both':
        dp = json.loads((output / 'dp/comparison.json').read_text())
        mp = json.loads((output / 'mp/comparison.json').read_text())
        mp_failures = {(e['job'], e['mode']) for e in mp['failures'] if 'job' in e and 'mode' in e}
        rescued = all((e['job'], e['mode']) not in mp_failures for e in dp['unstable']) and mp['passed']
    provenance = dict(reference_sha256=sha(args.reference),
                      driver_sha256=sha(HERE / 'ninjars_poles.f90'),
                      library_sha256=sha(lib / 'libninjars.a'),
                      installation_sha256=sha(lib / 'ninjars-config.json'),
                      ordinary_dp_unstable_controls_pass_mp=rescued,
                      results=summaries)
    (output / 'summary.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if not all(s['passed'] for s in summaries) or rescued is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
