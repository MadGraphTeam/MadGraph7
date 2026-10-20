#!/usr/bin/env python3
"""Read-only verification of the relocated-copy diagnostic; never executes it."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def rows(text):
    result = []
    for match in re.finditer(r'(?m)^\s*RELOCATION_ROW\s+(\d+)\s+(.+)$', text):
        values = [float(v.replace('D', 'E')) for v in match[2].split()]
        assert len(values) == 4 and all(map(math.isfinite, values))
        result.append({'order': int(match[1]), 'coefficients': values})
    assert [r['order'] for r in result] == [0, 1]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Also rehash retained work output and installed inputs')
    args = parser.parse_args()
    seal = read('manifest.json')
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    assert actual == set(seal['files'])
    for name, record in seal['files'].items():
        p = HERE / name
        assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], name
    prepared = read('prepared.json')
    runtime = read('runtime-results.json')
    assert prepared['complete'] and runtime['complete']
    assert prepared['revision'] == 'f4bffc61ed67e5d083b80ff9901768d4d5779989'
    assert not prepared['acceptance_eligible'] and not runtime['acceptance_eligible']
    assert runtime['failures'] == [] and runtime['cpu'] == 2
    assert prepared['library_availability'] == ['.TRUE.', '.FALSE.', '.TRUE.', '.FALSE.', '.FALSE.', '.FALSE.', '.FALSE.', '.TRUE.']
    assert sum(b['calls'] for b in prepared['factored_batches'].values()) == 136
    assert len(prepared['copied_assets']) == 7
    assert all(v['regular_file'] for v in prepared['copied_assets'].values())
    for name, batch in prepared['factored_batches'].items():
        assert sha(HERE / 'inputs/generated' / name) == batch['sha256']
    for name, record in prepared['copied_assets'].items():
        if name != 'libninjars.a':
            assert sha(HERE / 'inputs/copied-assets' / name) == record['sha256']
    expected = {'dp': 110, 'qp': 120, 'stable': 218}
    assert set(runtime['measurements']) == set(expected)
    measurements = {}
    for label, code in expected.items():
        record = runtime['measurements'][label]
        path = HERE / (label + '.log')
        assert sha(path) == record['log_sha256']
        assert record['returncode'] == 0 and record['return_code'] == code
        text = path.read_text()
        codes = re.findall(r'MadLoop return code\s*=\s*(\d+)', text)
        assert codes and int(codes[-1]) == code
        measurements[label] = rows(text)
        assert measurements[label] == record['rows']
    reference = read('reference-input.json')
    assert sha(HERE / 'param_card.dat') == reference['parameter_card_sha256']
    measurements['retained_Ninja6'] = reference['rows']
    scale = max(abs(v) for table in measurements.values() for row in table for v in row['coefficients'])
    largest, comparisons = 0., 0
    for left, right in itertools.combinations(measurements, 2):
        for a, b in zip(measurements[left], measurements[right]):
            assert a['order'] == b['order']
            for x, y in zip(a['coefficients'], b['coefficients']):
                tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scale
                assert abs(x-y) <= tolerance
                largest = max(largest, abs(x-y) / tolerance)
                comparisons += 1
    assert comparisons == 48
    if args.live:
        spec = importlib.util.spec_from_file_location('relocation_validation', HERE / 'validate.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert not module.PRIVATE.exists() and not module.OUTPUT.exists()
        assert module.bench.content_digest(module.files(module.HIDDEN)) == prepared['hidden_installation_sha256']
        assert module.bench.generated_source_manifest(module.RELOCATED) == prepared['sources']
        assert module.sha(module.RELOCATED / 'SubProcesses/P0_uux_zgg/check') == prepared['executable_sha256']
        assert module.factor_sites() == prepared['factored_batches']
        module.verify_main()
        assert sha(ROOT / reference['source']) == reference['source_sha256']
    print(json.dumps({'passed': True, 'bound_files': len(seal['files']), 'live': args.live,
                      'coefficient_comparisons': comparisons, 'maximum_tolerance_fraction': largest}))


if __name__ == '__main__':
    main()
