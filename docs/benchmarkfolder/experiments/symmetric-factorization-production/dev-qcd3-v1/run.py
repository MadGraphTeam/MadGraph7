#!/usr/bin/env python3
"""Fresh CLI diagnostic for the in-development private Fortran export."""
import sys
sys.dont_write_bytecode = True

import hashlib
import json
import os
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'madgraph/__init__.py').is_file())
REPORT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import ninjars_benchmark as bench
from ninjars_report_v2 import compare


def dump(name, value):
    bench.dump(REPORT/name, value)


def main():
    os.sched_setaffinity(0, {5})
    bench.WORK = ROOT/'.benchmark-work/experiments/symmetric-factorization-production/dev-qcd3-v1'
    bench.REPORT = REPORT
    config = json.loads((ROOT/'docs/benchmarkfolder/policy-v2/cases.json').read_text())
    case = next(c for c in config['cases'] if c['id'] == 'qcd-3')
    points = ROOT/'docs/benchmarkfolder/points/qcd-3.dat'
    result = dict(complete=False, acceptance_eligible=False,
        scope='Fresh CLI export from retained development patch; full100-point '
              'Ninja/NinjaRS correctness control, not final collection or performance.',
        case=case, points_sha256=hashlib.sha256(points.read_bytes()).hexdigest())
    dump('result.json', result)
    try:
        bench.generate('candidate', case)
        output = bench.output_dir('candidate', case)
        old = ROOT/'.benchmark-work/policy-v2/generation-abi3-terminal-fixed/outputs/candidate/qcd-3'
        assert (output/'Cards/param_card.dat').read_bytes() == (old/'Cards/param_card.dat').read_bytes()
        result['masses'] = bench.build('candidate', case)
        directory = bench.subprocess_dir('candidate', case)
        calls = '\n'.join(p.read_text() for p in directory.glob('mp_coef_construction_*.f'))
        import re
        count = len(re.findall(r'CALL MP_\w*(?:UPDATE|TERMINAL)_WL_[2-4]_1_FACTOR_FFV1L[12]_[12]\(', calls))
        assert count == 136, count
        result['factored_call_sites'] = count
        dump('result.json', result)
        records = {}
        for backend in (6, 8):
            records[str(backend)] = bench.measure('candidate', case, backend,
                points, 1, 0, 5, config=config)
            result['measurements'] = records
            dump('result.json', result)
            print('backend', backend, 'complete', records[str(backend)]['status_counts'], flush=True)
        agreement = compare({key: value['rows'] for key, value in records.items()})
        result['agreement'] = agreement
        assert agreement['passed'], agreement['failures']
        prototype = ROOT/'docs/benchmarkfolder/experiments/symmetric-factorization/qcd3-factor-trace-v1/result.json'
        original = json.loads(prototype.read_text())['measurements']['original']['rows']
        result['ninja_rows_equal_sealed_original'] = records['6']['rows'] == original
        assert result['ninja_rows_equal_sealed_original']
        result['complete'] = True
        dump('result.json', result)
        print('All100-point comparisons passed', agreement['max_tolerance_fraction'], flush=True)
    except Exception as error:
        result['error'] = repr(error)
        dump('result.json', result)
        raise


if __name__ == '__main__':
    main()
