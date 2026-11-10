#!/usr/bin/env python3
"""Portable integrity and offline numerical-record verifier; never runs reducers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
from analyze import analyze
from analyze_physical import analyze_physical
from run import blocks


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-root', type=Path,
                        help='Optionally hash ignored native artifacts beneath this repository root; never execute them.')
    args = parser.parse_args()
    index = json.loads((HERE/'delivery-index.json').read_text())
    for relative, expected in index['files'].items():
        path = BASE/relative
        assert path.resolve().is_relative_to(BASE.resolve()), relative
        assert digest(path) == expected, relative
    for directory in (BASE, HERE):
        frozen = json.loads((directory/'source-index.json').read_text())
        for relative, expected in frozen['files'].items():
            assert digest(directory/relative) == expected, relative
    results = HERE/'results'
    parsed = {name: blocks((results/(name+'.log')).read_text())
              for name in ('original', 'off', 'on')}
    assert parsed['original'] == parsed['off'] == parsed['on']
    assert (results/'off-mpfr-flags.dat').read_bytes() == (results/'on-mpfr-flags.dat').read_bytes()
    raw = json.loads((results/'on-shadow.json').read_text())
    disabled = json.loads((results/'off-shadow.json').read_text())
    assert raw.get('mpfr_reader_available') is True
    assert disabled.get('mpfr_reader_available') is True
    assert raw['key_flag_changes'] == raw['output_flag_changes'] == 0
    assert raw['uniqueness_complete'] and raw['dropped_events'] == 0
    status = {str(i+1): int(next(line.split()[1] for line in block if line.startswith('STATUS ')))
              for i, block in enumerate(parsed['on'])}
    assert sum(event['kind'] == 'parent_begin' for event in raw['events']) == 36
    raw['public_status_by_parent'] = status
    summary = analyze(raw)
    summary['original_off_on_exact_parity'] = True
    summary['off_on_mpfr_flag_parity'] = True
    assert summary == json.loads((results/'summary.json').read_text())
    source = {name: digest(results/name)
              for name in ('on-shadow.json', 'summary.json', 'execution.json')}
    physical = analyze_physical(raw, summary, source)
    assert physical == json.loads((results/'physical-analysis.json').read_text())
    native_checked = 0
    if args.native_root:
        for relative, expected in index['native_artifacts_sha256'].items():
            assert digest(args.native_root/relative) == expected, relative
            native_checked += 1
    print(json.dumps({'evidence_files_verified': len(index['files']),
                      'native_artifacts_hashed_only': native_checked,
                      'triplet_records': 108, 'physical_parents': 24,
                      'physical_scalar_requests': physical['physical_scopes']['physical']['requests'],
                      'all_recorded_parity_checks_passed': True,
                      'recomputed_public_status_bound_summary': True,
                      'recomputed_physical_analysis': True,
                      'reducer_executions': 0}, indent=2))


if __name__ == '__main__':
    main()
