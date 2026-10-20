#!/usr/bin/env python3
"""Recheck the retained diagnostic; never compile or evaluate amplitudes."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT/'tools'))
from ninjars_report_v2 import compare


def main():
    seal = json.loads((HERE/'SHA256SUMS.json').read_text())
    for name, expected in seal['files'].items():
        path = HERE/name
        assert path.stat().st_size == expected['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
    result = json.loads((HERE/'result.json').read_text())
    assert result['complete'] and not result['acceptance_eligible']
    records = {b: json.loads((HERE/f'data/candidate-qcd-3-{b}-0.json').read_text())
               for b in ('6', '8')}
    assert result['measurements'] == records
    assert compare({b: r['rows'] for b, r in records.items()}) == result['agreement']
    assert result['agreement']['passed']
    original = json.loads((HERE.parent.parent/'symmetric-factorization/'
        'qcd3-factor-trace-v1/result.json').read_text())['measurements']['original']
    assert records['6']['rows'] == original['rows']
    assert records['6']['status_counts'] == {'216': 98, '328': 2}
    assert records['8']['status_counts'] == {'218': 98, '322': 2}
    print(json.dumps(dict(verified_files=len(seal['files']),
        agreement_passed=True, ninja_rows_equal_sealed_original=True,
        acceptance_eligible=False), indent=2))


if __name__ == '__main__':
    main()
