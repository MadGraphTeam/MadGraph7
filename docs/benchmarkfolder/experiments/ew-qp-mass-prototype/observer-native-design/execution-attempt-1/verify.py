#!/usr/bin/env python3
"""Portable hash/outcome verification; never compiles or evaluates anything."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    here = Path(__file__).resolve().parent
    index = json.loads((here / 'evidence-index.json').read_text())
    for name, expected in index['files'].items():
        path = here / name
        require(not path.is_symlink() and path.is_file(), name)
        require(sha(path) == expected, 'Changed file: ' + name)
    build = json.loads((here / 'build.json').read_text())
    run = json.loads((here / 'run.json').read_text())
    analysis = json.loads((here / 'offline-analysis.json').read_text())
    require(build['state'] == 'complete', 'Build did not complete')
    require(len(build['commands']) == 16 and len(build['outputs']) == 18,
            'Unexpected build artifact or command count')
    require(all(c['returncode'] == 0 and not c['timed_out'] for c in build['commands']),
            'Compiler failure')
    require(run['build_json_sha256'] == sha(here / 'build.json'), 'Build binding')
    require(run['state'] == 'incomplete' and len(run['cases']) == 4, 'Run status')
    require(not analysis['transport_interpretation_eligible'], 'Invalid eligibility')
    require(not analysis['comparisons'], 'Unexpected numerical comparison')
    require(not run['automatic_retry'] and not run['performance_eligible'], 'Invalid scope')
    seen = set()
    for case in run['cases']:
        name = case['arm'] + '-' + str(case['backend'])
        require(name not in seen, 'Duplicate case')
        seen.add(name)
        raw = here / 'raw' / name
        require(case['evaluation']['returncode'] == 0 and not case['complete'], 'Wrong failure')
        require(not case['evaluation']['timed_out'], 'Unexpected timeout')
        require(case['error']['type'] == 'IncompleteDiagnostic', 'Wrong classification')
        require('/ident_card.dat' in (raw / 'stdout.log').read_text(), 'Missing warning')
        require((raw / 'param.log').stat().st_size == 0, 'Unexpected model log')
        require(not (raw / 'mass-observer.hex').exists() and
                not (raw / 'observer-run.log').exists(), 'Unexpected observations')
        for path, expected in case['files_after'].items():
            require(sha(raw / path) == expected, 'Runtime hash mismatch: ' + name + '/' + path)
    require(seen == {'original-8', 'original-9', 'candidate-8', 'candidate-9'}, 'Cases')
    print(json.dumps({'verified_files': len(index['files']), 'state': run['state'],
                      'transport_interpretation_eligible': False}, sort_keys=True))


if __name__ == '__main__':
    main()
