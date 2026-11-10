#!/usr/bin/env python3
"""Validate existing classifier mock streams only; never launch a process."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CLASSIFIER = HERE.parent
EXPECTED = (
    ['CLASSIFIER_MOCK_COMPLETE', '8', '8'],
    ['CLASSIFIER_MOCK_COMPLETE', '16', '8'],
    ['COMPLETE_CLASSIFIER_MOCK_ONLY'],
)
SOURCE_HASHES = {
    'fixture/driver.f90': '7129998ce945189c14f09704daedc77001370be08f782bac17e93294440d51c5',
    'fixture/mock_hosts.f90': '46faba494413a88dce9f2aec16663e7a1883ae8c3c8ee8426e3e7b188017b126',
    'fixture/tables.f90': '63ca88e1f9dbb9636144b291bbd6a1f7ff2814fc334335491e88c6af0e52d03b',
    'cms_scalar_observer.f90': '1b61c18bbc86d1b5fa25456be73751aaf275789fd043800419614e280db800ac',
    'classification.json': '56cafacc1289137a6d4d32d919e91228e09dd1ff69bab4ed9225534181ac94eb',
    'compile-plan.json': '0f97e07f2a99246b74cf135d5b6346ffcdd7e61a7c83967b409299699b44d07e',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    for name, digest in SOURCE_HASHES.items():
        require(sha(CLASSIFIER / name) == digest, 'Frozen source changed: ' + name)


def validate_text(stdout, stderr=''):
    """A text-only gate; successful process, binary and loader binding are separate."""
    require(isinstance(stdout, str) and isinstance(stderr, str), 'Streams must be text')
    require(not stderr, 'Unexpected stderr, including runtime warnings/errors')
    require(stdout.isascii() and '\x00' not in stdout, 'Unexpected stdout encoding/control data')
    require(stdout.endswith('\n'), 'Truncated stdout: no final newline')
    require('SUBNORMAL_FIXTURE_UNSUPPORTED' not in stdout,
            'Incomplete qualification: a precision skipped the subnormal fixture')
    rows = [line.split() for line in stdout.splitlines()]
    require(rows == list(EXPECTED), 'Missing, extra, reordered or malformed completion records')
    return {
        'passed': True,
        'precision_kinds': [8, 16],
        'completed_overloads_per_precision': 8,
        'completed_overloads_total': 16,
        'max_event_assertions_per_overload': [1, 1, 3, 3, 9, 9, 13, 13],
        'count_scope': 'Frozen fixture assertions completed; not a per-event numerical journal',
        'subnormal_fixture_skipped': False,
        'physical_accuracy_qualified': False,
        'actual_analytic_execution_qualified': False,
        'full_madloop_rescue_qualified': False,
        'ieee_transparency_qualified': False,
        'performance_qualified': False,
    }


def validate_files(stdout, stderr, event, run_dir):
    """Optional strict envelope for captures held in an otherwise empty run directory."""
    verify_sources()
    stdout, stderr, run_dir = Path(stdout), Path(stderr), Path(run_dir)
    require(stdout.name != stderr.name, 'Streams must have distinct paths')
    require(stdout.parent.resolve() == run_dir.resolve() == stderr.parent.resolve(),
            'Both streams must be directly inside the private run directory')
    require(type(event.get('returncode')) is int and event['returncode'] == 0, 'Nonzero/missing exit')
    require(event.get('timed_out') is False, 'Missing timeout result or timeout')
    require(not event.get('error') and not event.get('interrupted'), 'Process error/interruption')
    expected_files = {stdout.name, stderr.name}
    files = list(run_dir.iterdir())
    require({p.name for p in files} == expected_files and all(p.is_file() and not p.is_symlink() for p in files),
            'Unexpected, missing, nonregular or symlink output')
    digests = {p.name: sha(p) for p in (stdout, stderr)}
    require(event.get('output_files') == digests, 'Output inventory/hash mismatch')
    result = validate_text(stdout.read_text(encoding='ascii'), stderr.read_text(encoding='ascii'))
    result.update(native_exit_checked=True, output_files=digests,
                  runtime_and_build_binding='Required separately in parent controller')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stdout', type=Path, required=True)
    parser.add_argument('--stderr', type=Path, required=True)
    parser.add_argument('--event', type=Path, required=True, help='One captured process event JSON object')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result = validate_files(args.stdout, args.stderr, json.loads(args.event.read_text()), args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
