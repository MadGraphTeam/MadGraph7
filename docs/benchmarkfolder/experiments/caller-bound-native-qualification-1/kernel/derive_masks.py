"""Derive structural classes from authenticated source routing, never events."""
import hashlib
import json
from pathlib import Path

ROUTING = '.tirust-build/kinematic-provenance-current-executor-1/results/derived/cms/routing.json'
ROUTING_SHA = 'ca9ee4363075a15a9f0d81e3dc744331064da6f724c449b9125d8d64f2a1d499'


def proportional(a, c):
    if len(a) != len(c) or not any(c):
        raise ValueError('conservation shape')
    k = next(i for i, x in enumerate(c) if x)
    if a[k] % c[k]:
        return False
    factor = a[k] // c[k]
    return all(x == factor*y for x, y in zip(a, c))


def pair_class(a, c, fixed_shell):
    if proportional(a, c):
        return 1  # protected structural zero
    for e in range(len(a)):
        for sign in (-1, 1):
            residual = list(a)
            residual[e] -= sign
            if proportional(residual, c):
                return 1 if fixed_shell[e] else 0
    return 2  # composite; still requires a fresh numerical certificate


def derive(raw):
    case = raw['cases']['cms_ew_n2_epem_mupmum']
    assert case['dp_qp_symbolic_match'] is True
    legs = sorted(case['external_provenance'], key=lambda x: x['leg'])
    assert [x['leg'] for x in legs] == list(range(1, len(legs)+1))
    fixed = [(x['generated_keep_offshell_mass'] is False and
              x['pmass_expression'] == 'ZERO' and
              x['external_call_mass_expression'] == 'ZERO') for x in legs]
    rows = []
    for route in case['reductions']:
        c = route['conservation_coefficients']
        n = len(route['modes']['1']['shifts'])
        modes = {}
        for mode in ('1', '2', '4', '5'):
            r = route['modes'][mode]
            corners = r['corners']
            shifts = r['shifts']
            assert len(corners) == len(shifts) == n
            assert shifts[0] == [sum(v[e] for v in corners) for e in range(len(c))]
            for j in range(1, n):
                assert shifts[j] == [sum(v[e] for v in corners[:j]) for e in range(len(c))]
            primary = route['modes']['1']['corners']
            expected = ([[-x for x in v] for v in reversed(primary)]
                        if mode in ('2','5') and n > 2 else primary)
            assert corners == expected
            pairs = []
            for j in range(1, n):
                for i in range(j):
                    a = [x-y for x, y in zip(shifts[i], shifts[j])]
                    pairs.append({'pair':[i+1,j+1], 'unreduced_coefficients':a,
                                  'class':pair_class(a,c,fixed)})
            modes[mode] = pairs
        rows.append({'route_id':route['reduction_identifier'], 'nprop':n, 'modes':modes})
    assert len(rows) == 26 and len({x['route_id'] for x in rows}) == 26
    return {'schema':'cms-source-structural-masks-v1', 'routing_sha256':ROUTING_SHA,
            'classes':{'0':'unknown','1':'protected shell/zero','2':'composite'},
            'routes':rows, 'event_or_fixture_conditions':False,
            'runtime_eligibility':'No eligibility until fresh source-derived bounds and runtime certificate pass',
            'operation_counts_and_caller_bound_integration_qualified':False}


if __name__ == '__main__':
    data = Path(ROUTING).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ROUTING_SHA
    print(json.dumps(derive(json.loads(data)), sort_keys=True, indent=2))
