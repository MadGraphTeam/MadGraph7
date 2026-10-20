#!/usr/bin/env python3
"""Read-only checks of sealed kernel inputs, raw samples and optional statistics."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute', action='store_true',
                        help='also recompute fixed-bootstrap statistics; no compiled program runs')
    args = parser.parse_args()
    seal = json.loads((HERE / 'seal.json').read_text())
    assert seal['schema'] == 'dp-factor-kernel-seal-v1' and seal['complete']
    assert seal['file_count'] == len(seal['files'])
    for name, record in seal['files'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path = BASE / relative
        assert path.is_file() and not path.is_symlink(), name
        contents = path.read_bytes()
        assert len(contents) == record['bytes'], name
        assert hashlib.sha256(contents).hexdigest() == record['sha256'], name
    assert sha(BASE / 'correctness-seal.json') == seal['correctness_seal_sha256']
    for rejected in seal['rejected_preflights']:
        directory = BASE / rejected
        assert not json.loads((directory / 'preflight.json').read_text())['passed']
        assert not (directory / 'measurements.json').exists()
    report = BASE / seal['completed_run']
    plan = json.loads((report / 'plan.json').read_text())
    data = json.loads((report / 'measurements.json').read_text())
    assert plan['complete'] and data['complete'] and data['all_samples_retained']
    assert json.loads((report / 'preflight.json').read_text())['passed']
    assert data['plan_sha256'] == sha(report / 'plan.json')
    assert data['preflight_sha256'] == sha(report / 'preflight.json')
    expected = {(c['key'], f['case']) for c in plan['configurations'] for f in c['fixtures']}
    assert len(expected) == 36 and len(data['results']) == 36
    assert {(r['key'], r['case']) for r in data['results']} == expected
    count = 0
    for record in data['results']:
        log = report / record['key'] / f'case-{record["case"]}.log'
        assert sha(log) == record['log_sha256']
        rows = [line.split() for line in log.read_text().splitlines()]
        assert len(rows) == 89
        assert [r[:2] for r in rows[:4]] == [['VALIDATED', str(i)] for i in range(4)]
        assert rows[-1] == ['PASS', '21', '5000']
        samples = []
        for row in rows[4:-1]:
            assert row[0] == 'SAMPLE' and len(row) == 7
            pair, position, arm = map(int, row[1:4])
            assert arm == plan['schedule'][(pair - 1) % 8][position - 1]
            cpu, wall, checksum = map(float, row[4:])
            assert cpu > 0 and wall > 0
            samples.append(dict(pair=pair, position=position, arm=arm,
                                cpu_seconds=cpu, wall_seconds=wall, checksum=checksum))
        assert [(r['pair'], r['position']) for r in samples] == [
            (p, o) for p in range(1, 22) for o in range(1, 5)]
        assert samples == record['samples']
        count += len(samples)
    assert count == 3024
    if args.recompute:
        from analyze import analyze
        actual = analyze(report)
        expected_analysis = json.loads((report / 'analysis.json').read_text())
        assert actual == expected_analysis
    print(json.dumps(dict(passed=True, bound_files=seal['file_count'],
                          bound_bytes=sum(r['bytes'] for r in seal['files'].values()),
                          completed_fixtures=36, timed_arm_samples=count,
                          statistics_recomputed=args.recompute)))


if __name__ == '__main__':
    main()
