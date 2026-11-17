#!/usr/bin/env python3
"""Compare bare-loop pole selections with a common point-wise scale."""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ('born', 'finite', 'single', 'double')


def main():
    global HERE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    HERE = args.output.resolve()
    failures, comparisons, counts, worst = [], [], {}, 0.0
    expected_cases = ['qcd-z', 'qcd-tt', 'ew-mumu', 'li-h']
    configurations = switch_rows = 0
    def status_ok(status, backend, ct):
        if ct in (1, 2):
            return status == 110
        if ct in (4, 5):
            return status == 120
        if ct == -1:
            h, t, u = status//100, (status//10)%10, status%10
            return t in (1, 2) and ((h == 2 and u == backend) or (backend == 8 and h == 3 and u == 2))
        return False
    def validate_rows(data, path):
        dimensions = list(map(int, (path.parent / 'dimensions.dat').read_text().split()))
        nso = dimensions[1]
        expected = {(point, order) for point in range(1, 4) for order in range(nso+1)}
        rows = data['records']
        keys = [(r['point'], r['split_order']) for r in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            failures.append(dict(case=path.parent.parent.name, reason='missing/duplicate/unexpected point/order rows', path=str(path)))
        for row in rows:
            if not status_ok(row['status'], data['backend'], data['ctmode']):
                failures.append(dict(case=path.parent.parent.name, reason='unexpected evaluation status', path=str(path), status=row['status']))
            if not all(math.isfinite(row[key]) for key in KEYS):
                failures.append(dict(case=path.parent.parent.name, reason='nonfinite result', path=str(path)))
        return expected
    for name in expected_cases:
        work = HERE / 'processes' / name
        required = {(backend, mode, 1) for backend in (7, 8) for mode in range(4)}
        required |= {(8, mode, 4) for mode in range(4)}
        if name.startswith('qcd-'):
            required |= {(8, mode, ct) for mode in range(4) for ct in (-1, 2, 4, 5)}
        results = {}
        for path in sorted(work.glob('backend-*-mode-*-ct-*/result.json')):
            data = json.loads(path.read_text())
            key = data['backend'], data['mode'], data['ctmode']
            validate_rows(data, path)
            configurations += 1
            results[key] = {(r['point'], r['split_order']): r for r in data['records']}
            counts[work.name + '/' + path.parent.name] = {str(s): sum(r['status'] == s for r in data['records'])
                                        for s in sorted({r['status'] for r in data['records']})}
        missing = sorted(required - set(results))
        if missing:
            failures.append(dict(case=name, reason='missing required configurations', missing=missing))
        if not results:
            continue
        def compare(label, a, b, indices=KEYS):
            nonlocal worst
            assert a.keys() == b.keys(), label
            n = 0
            for point in a:
                scale = max(abs(r[point][field]) for r in results.values() for field in KEYS)
                for field in indices:
                    x, y = a[point][field], b[point][field]
                    tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scale
                    ratio = abs(x-y) / tolerance if tolerance else (0 if x == y else float('inf'))
                    worst = max(worst, ratio)
                    n += 1
                    if not math.isfinite(x) or not math.isfinite(y) or ratio > 1:
                        failures.append(dict(case=work.name, comparison=label, point=point,
                                             field=field, a=x, b=y, tolerance=tolerance, ratio=ratio))
            comparisons.append(dict(case=work.name, comparison=label, values=n))
        for (backend, mode, ct), rows in results.items():
            if backend == 8 and (7, mode, 1) in results:
                compare(f'NinjaRS CTMode={ct},mode={mode} vs COLLIER', rows, results[7, mode, 1])
            if mode and (backend, 0, ct) in results:
                compare(f'backend={backend},CTMode={ct},mode={mode} finite unchanged',
                        rows, results[backend, 0, ct], ('born', 'finite'))
        for backend, ct in sorted({(b, c) for b, _, c in results}):
            if all((backend, mode, ct) in results for mode in range(4)):
                total, uv, ir, none = (results[backend, mode, ct] for mode in range(4))
                sum_poles = {point: dict(uv[point], single=uv[point]['single']+ir[point]['single'],
                                         double=uv[point]['double']+ir[point]['double']) for point in total}
                # Loop-induced returned quantities are squared amplitudes and
                # need not be additive in the selected amplitude-level poles.
                if not name.startswith('li-'):
                    compare(f'backend={backend},CTMode={ct} UV+IR=both', total, sum_poles, ('single', 'double'))
                zero = {point: dict(none[point], single=0, double=0) for point in none}
                compare(f'backend={backend},CTMode={ct} no-pole selector', none, zero, ('single', 'double'))
                compare(f'backend={backend},CTMode={ct} UV has no double pole', uv, zero, ('double',))
        for path in sorted(work.glob('filtered-backend-*-mode-*-ct-*/result.json')):
            data = json.loads(path.read_text())
            validate_rows(data, path)
            configurations += 1
            key = data['backend'], max(0, data['mode']), data['ctmode']
            rows = {(r['point'], r['split_order']): r for r in data['records']}
            if key in results:
                compare(f'backend={key[0]},mode={key[1]},CTMode={key[2]} normal filters', rows, results[key])
        for path in sorted(work.glob('*backend-*-mode-*-ct-*/switch_values.dat')):
            data = json.loads((path.parent / 'result.json').read_text())
            switched = {}
            for line in path.read_text().splitlines():
                fields = line.split()
                iteration, mode, point, split_order, status = map(int, fields[:5])
                if not status_ok(status, data['backend'], data['ctmode']):
                    failures.append(dict(case=name, reason='unexpected runtime-switch status', path=str(path), status=status))
                rows = switched.setdefault((iteration, mode), {})
                if (point, split_order) in rows:
                    failures.append(dict(case=name, reason='duplicate runtime-switch row', path=str(path)))
                rows[point, split_order] = dict(zip(KEYS, map(float, fields[5:9])))
                switch_rows += 1
            expected_modes = list(enumerate([0, 1, 2, 3, 0, 2, 1, 0], 1))
            expected_points = validate_rows(data, path.parent / 'result.json')
            if set(switched) != set(expected_modes) or any(set(rows) != expected_points for rows in switched.values()):
                failures.append(dict(case=name, reason='incomplete runtime-switch sequence', path=str(path)))
            for (iteration, mode), rows in switched.items():
                key = data['backend'], mode, data['ctmode']
                if key in results:
                    filtered = path.parent.name.startswith('filtered-')
                    compare(f'backend={key[0]},CTMode={key[2]},filters={filtered} runtime switch {iteration} to mode {mode}', rows, results[key])
        if name == 'qcd-tt':
            for relative in ['backend-7-mode-0-ct-1/switch_values.dat',
                             'backend-8-mode-0-ct-1/switch_values.dat',
                             'filtered-backend-8-mode--1-ct-1/switch_values.dat',
                             'filtered-backend-8-mode-1-ct-1/result.json']:
                if not (work / relative).is_file():
                    failures.append(dict(case=name, reason='missing required cache/filter control', path=relative))
    report = dict(passed=not failures, comparisons=comparisons, failure_count=len(failures),
                  worst_normalized_error=worst, failures=failures, statuses=counts,
                  validated_configurations=configurations, runtime_switch_rows=switch_rows,
                  loop_induced_additivity_checked=False)
    (HERE / 'process_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=not failures, comparisons=len(comparisons), failures=len(failures),
                         worst_normalized_error=worst), indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
