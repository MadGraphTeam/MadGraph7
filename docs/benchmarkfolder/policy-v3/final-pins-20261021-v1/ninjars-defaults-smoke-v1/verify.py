#!/usr/bin/env python3
"""Bounded verifier for the installed-default software/export/native smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

if not __debug__:
    raise SystemExit('Optimized Python is unsupported for proof verification')

HERE = Path(__file__).resolve().parent

def require(value, message):
    if not value:
        raise ValueError(message)

def load(name):
    return json.loads((HERE / name).read_text())

def card_value(path, key):
    lines = path.read_text().splitlines()
    index = lines.index('#' + key)
    return next(line.strip() for line in lines[index + 1:]
                if line.strip() and not line.lstrip().startswith(('#', '!')))

def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--live', action='store_true')
    options = args.parse_args()
    manifest = load('manifest.json')
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'manifest.json'}, 'Unexpected/missing capsule files')
    for name, bound in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe bound path')
        path = HERE / relative
        require(not path.is_symlink(), 'Symlinked capsule file')
        data = path.read_bytes()
        require(len(data) == bound['bytes'] and hashlib.sha256(data).hexdigest() == bound['sha256'],
                'Capsule file changed: ' + name)
    if options.live:
        root = next(p for p in HERE.parents if (p / 'madgraph').is_dir() and (p / 'vendor').is_dir())
        for name, bound in load('source-bindings.json')['files'].items():
            path = root / name
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == bound['sha256'],
                    'Live source changed: ' + name)
            require(oct(path.stat().st_mode & 0o777) == bound['mode'], 'Live source mode changed: ' + name)
    for name, count, failed in [('tests.log', 18, False), ('installer-successor-tests.log', 19, False),
                                ('live-tests.log', 37, True)]:
        text = (HERE / 'software' / name).read_text()
        require(re.search(r'Ran %d tests in ' % count, text), 'Wrong software count: ' + name)
        require(('FAILED (failures=1)' if failed else '\nOK\n') in text, 'Wrong software outcome: ' + name)
    recording = (HERE / 'actual/unavailable-fortran-test.log').read_text()
    require('Ran 1 test in ' in recording and '\nOK\n' in recording, 'Recording Fortran control changed')
    for name, selection in [('ordinary', '8|6|7|1'), ('nlo', '8|6|7|1'),
                            ('loop-induced', '7|6|1'), ('absent', '6|7|1')]:
        for path in (HERE / 'actual/export-cards' / name).rglob('MadLoopParams*.dat'):
            require(card_value(path, 'MLReductionLib') == selection, 'Wrong actual export selection')
            expected = '.FALSE.' if name == 'loop-induced' else '.TRUE.'
            for key in ('COLLIERComputeUVpoles', 'COLLIERComputeIRpoles'):
                require(card_value(path, key).upper() == expected, 'Changed loop-induced/pole default')
    rows = {}
    for name, status in [('default', 218), ('explicit6', 216)]:
        rows[name] = [line.split() for line in (HERE / 'actual/raw' / name / 'values.dat').read_text().splitlines()]
        require(len(rows[name]) == 2 and {int(r[1]) for r in rows[name]} == {0, 1}, 'Wrong native split orders')
        require(all(int(r[0]) == 1 and int(r[2]) == status for r in rows[name]), 'Wrong actual native backend')
        timed = (HERE / 'actual/raw' / name / 'timed_status.dat').read_text().split()
        require(timed == [str(status), '1'], 'Wrong timed backend count')
    for a, b in zip(rows['default'], rows['explicit6']):
        left, right = [float(x) for x in a[3:7]], [float(x) for x in b[3:7]]
        scale = max(map(abs, left + right))
        require(all(abs(x-y) <= 1e-7*max(abs(x),abs(y)) + 1e-10*scale for x,y in zip(left,right)),
                'Native coefficients disagree')
    events = [json.loads(line) for line in (HERE / 'actual/process-ledger.jsonl').read_text().splitlines()]
    spawned = {(e['pid'], e['start_ticks']) for e in events if e['event'] == 'spawn'}
    reaped = {(e['pid'], e['start_ticks']) for e in events if e['event'] == 'reaped' and e['reaped'] and e['returncode'] == 0}
    require(len(spawned) == 2 and spawned == reaped, 'Native children not reaped')
    text = (HERE / 'actual/generation-absent.log').read_text()
    require('madgraph.InvalidCmd: NinjaRS is disabled.' in text, 'Missing explicit unavailable8 diagnostic')
    require(load('actual/absent-cli-checks.json')['numerical_check_directory_created'] is False,
            'Unexpected disabled-backend evaluation')
    require(load('actual/compile-result.json')['returncode'] == 0, 'Actual driver build failed')
    print('PASS: 18 default + 19 installer + 1 recording controls; actual SA/NLO/LI/absent cards, 218/216 parity, expected unavailable8 rejection, ownership' + ('; exact live sources' if options.live else ''))

if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        raise SystemExit(str(exc))
