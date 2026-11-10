#!/usr/bin/env python3
"""Portable integrity and bounded-outcome checks; no native execution."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def good(event):
    return event['returncode'] == 0 and not event['timed_out'] and not event.get('error')


def inventory(path):
    files = {}
    for p in sorted(path.rglob('*')):
        need(not p.is_symlink(), 'Symlink in evidence')
        if p.is_file():
            files[str(p.relative_to(path))] = sha(p)
    return files


def main():
    here = Path(__file__).resolve().parent
    index = json.loads((here / 'evidence-index.json').read_text())
    for name, expected in index['files'].items():
        need(sha(here / name) == expected, 'Changed evidence: ' + name)
    build = json.loads((here / 'build.json').read_text())
    smoke = json.loads((here / 'smoke.json').read_text())
    run = json.loads((here / 'run.json').read_text())
    analysis = json.loads((here / 'analysis.json').read_text())
    for result in (build, smoke, run):
        need(result['state'] == 'complete', 'Incomplete execution')
        need(result['review_hash'] == index['review_hash'], 'Review identity changed')
        need(result['cpu'] == 8 and result['jobs'] == 1, 'CPU/jobs changed')
        need(result['automatic_retry'] is False and result['performance_eligible'] is False and
             result['host_stability_qualified'] is False, 'Invalid scope')
    need(len(build['commands']) == 6 and all(good(e) for e in build['commands']), 'Compiler failures')
    need(smoke['build_json_sha256'] == run['build_json_sha256'] == sha(here / 'build.json'), 'Build binding')
    need(run['smoke_json_sha256'] == sha(here / 'smoke.json'), 'Smoke binding')
    need(all(good(smoke[k]) for k in ('loader', 'symbols', 'evaluation')), 'Smoke failures')
    need(inventory(here / 'raw/card-only') == smoke['files_after'], 'Smoke files changed')
    need((here / 'raw/card-only/card-smoke.hex').read_text().splitlines()[-1] == 'COMPLETE_CARD_ONLY', 'Smoke incomplete')
    expected = {(a, b) for a in ('original', 'candidate') for b in (8, 9)}
    need(len(run['cases']) == 4 and {(c['arm'], c['backend']) for c in run['cases']} == expected, 'Wrong cases')
    for case in run['cases']:
        name = case['arm'] + '-' + str(case['backend']);directory = here / 'raw' / name
        need(case['complete'] is True and not case.get('error'), 'Incomplete case')
        need(good(case['evaluation']) and good(case['loader_command']), 'Process failed')
        need(inventory(directory) == case['files_after'], 'Case files changed: ' + name)
        need(sha(directory / 'inputs.json') == case['inputs_sha256'], 'Input binding changed')
        inputs = json.loads((directory / 'inputs.json').read_text())
        for path, expected_hash in inputs['after'].items():
            need(case['files_after'][path] == expected_hash, 'Numerical input changed')
        need((directory / 'observer-run.log').read_text().splitlines()[-1] == 'COMPLETE_DIAGNOSTIC_ONLY', 'Driver incomplete')
        points = analysis['cases'][name]['points']
        need([p['point'] for p in points] == [1, 2] and all(p['groups'] == p['native_calls'] == 200 for p in points), 'Observed call count changed')
        need([p['native_refusals'] for p in points] == ([0, 37] if case['backend'] == 9 else [0, 0]), 'Refusal accounting changed')
    need(analysis['runner_envelope']['verified'] and analysis['transport_interpretation_eligible'], 'Input-control qualification failed')
    need(all(c['nonmass_input_bits_match'] for c in analysis['comparisons'].values()), 'Nonmass input mismatch')
    print(json.dumps({'verified_files': len(index['files']), 'cases': 4, 'observed_groups': 1600,
                      'threshold_tirust_aggregate_complete': False, 'checked_qp_qualified': False,
                      'scope': 'Portable integrity/outcome verification only'}, sort_keys=True))


if __name__ == '__main__':
    main()
