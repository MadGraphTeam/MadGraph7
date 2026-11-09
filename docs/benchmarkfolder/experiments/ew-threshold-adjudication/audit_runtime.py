#!/usr/bin/env python3
"""Bind the diagnostic's external libraries to the frozen campaign manifest."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'docs/benchmarkfolder/experiments/physical-stress/diagnostics/ew_n3_epem_ttxa/candidate-9-qp-init-dp/candidate-inputs.json'
PRIVATE = ROOT / '.tirust-build/ew-threshold-adjudication/private-libraries/libmadloop_tirust.so'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    runtime = json.loads(OLD.read_text())['runtime']
    result = {'historical_input_manifest_sha256': digest(OLD), 'entries': {}, 'all_equal': True}
    for category in ('archives', 'libraries', 'metadata'):
        result['entries'][category] = {}
        for path, expected in runtime[category].items():
            p = Path(path)
            # The bridge is resolved from this immutable snapshot, independently
            # of publication of a later installation at the original path.
            resolved = PRIVATE if category == 'libraries' and p.name == PRIVATE.name else p
            actual = digest(resolved)
            equal = actual == expected
            result['entries'][category][path] = {'resolved': str(resolved), 'expected_sha256': expected,
                'actual_sha256': actual, 'equal': equal}
            result['all_equal'] &= equal
    out = HERE / 'runtime-audit.json'
    if out.exists():
        previous = json.loads(out.read_text())
        if previous != result:
            raise ValueError('Runtime no longer matches recorded audit')
    else:
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if not result['all_equal']:
        raise ValueError('Historical library mismatch')
    print('Historical runtime/static archives: all identities equal.')


if __name__ == '__main__':
    audit()
