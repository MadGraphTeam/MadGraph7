"""Prepare an integer-only compiled lookup check; never invoke a compiler."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path.cwd()
HERE = ROOT / '.tirust-build/general-kinematics-integer-check'
OLD = ROOT / '.tirust-build/general-kinematics-classification'
ORACLE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer/routing-modes.json'
OLD_INDEX = '97f5100a4f7a679e53a6508f669b54d4f71aa2a40868f90e435f4d932cced533'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(OLD / 'source-index.json') == OLD_INDEX
    old = json.loads((OLD / 'source-index.json').read_text())
    for name, item in old['files'].items():
        assert sha(OLD / name) == item['sha256'], name
    expected = json.loads((OLD / 'source-tests.json').read_text())['frozen_routing_sha256']
    assert sha(ORACLE) == expected
    data = json.loads(ORACLE.read_text())
    (HERE / 'qcd_lookup.f').write_bytes((OLD / 'generated/qcd_n3_identity.f').read_bytes())
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('integer_check_identity',
                OLD / 'src/madgraph/loop/kinematic_identity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The N=1 boundary and explicit unknown/hole branches have no pair payload.
    boundary = {'reductions': [
        {'reduction_id': 1, 'n': 1, 'known': True,
         'directions': {'primary': {'pairs': []}, 'reverse': {'pairs': []}}},
        {'reduction_id': 3, 'n': 2, 'known': False}]}
    (HERE / 'boundary_lookup.f').write_text(module.render(boundary, 'BOUNDARY_'))
    rows = []
    counts = {'known_qcd': 0, 'known_n2': 0, 'reversed_pairs': 0,
              'invalid_qcd': 0, 'n1_or_unknown_boundary': 0}

    def add(which, group, ct, n, i, j, result, bucket):
        rows.append([which, group, ct, n, i, j] + list(result))
        counts[bucket] += 1

    unknown = (0, 0, 0, 0, 0, 0)
    for mode, groups in data['groups_by_mode'].items():
        ct = int(mode)
        assert ct in (4, 5)
        for group in groups:
            g, n = group['group'], group['n']
            for pair in group['pairs']:
                kind = {'direct_single_leg': 1, 'complement_of_single_leg': 2,
                        'composite': 3}[pair['classification']]
                leg = pair['single_leg'] or 0
                sign = closure = 0
                if leg:
                    vector = pair['literal_coefficients_k']
                    other = {v for k, v in enumerate(vector, 1) if k != leg}
                    assert len(other) == 1
                    closure = other.pop()
                    sign = vector[leg - 1] - closure
                    assert sign in (-1, 1)
                    assert (closure == 0) == (kind == 1)
                i, j = pair['propagators']
                # Oracle classifications/leg witnesses are independent of the
                # classifier under test; only exact integer coefficients enter.
                value = (kind, leg, sign, closure, int(bool(leg)), 1)
                for actual_ct in (ct, ct - 3):
                    add(1, g, actual_ct, n, i, j, value, 'known_qcd')
                    add(1, g, actual_ct, n, j, i,
                        (kind, leg, -sign, -closure, int(bool(leg)), 1), 'reversed_pairs')
                    if n == 2:
                        counts['known_n2'] += 2
            for i, j in [(0, 1), (1, 0), (-1, 1), (1, -1),
                         (n + 1, 1), (1, n + 1), (1, 1), (n, n)]:
                add(1, g, ct, n, i, j, unknown, 'invalid_qcd')
            for bad_n in (-1, 0, 1, n + 1):
                add(1, g, ct, bad_n, 1, 2, unknown, 'invalid_qcd')
            for bad_ct in (-1, 0, 3, 6, 7, 8, 9):
                add(1, g, bad_ct, n, 1, 2, unknown, 'invalid_qcd')
    for g in (-2147483647, -1, 0, 117, 2147483647):
        add(1, g, 4, 2, 1, 2, unknown, 'invalid_qcd')
    for g in (-1, 0, 1, 2, 3, 4):
        for ct in (1, 2, 4, 5):
            for n, i, j in [(1, 1, 1), (1, 1, 2), (2, 1, 2), (0, 0, 0)]:
                add(2, g, ct, n, i, j, unknown, 'n1_or_unknown_boundary')
    (HERE / 'cases.dat').write_text(str(len(rows)) + '\n' +
        ''.join(' '.join(map(str, row)) + '\n' for row in rows))
    evidence = {'source_only': True, 'compiler_executed': False, 'integer_execution': False,
                'input_index_sha256': OLD_INDEX, 'oracle_sha256': sha(ORACLE),
                'case_count': len(rows), 'counts': counts,
                'note': 'known_n2 is a subset count; N1 is a serialization boundary, not physical process qualification.'}
    (HERE / 'preparation.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    paths = ['prepare.py', 'run.py', 'driver.f90', 'README.md', 'qcd_lookup.f',
             'boundary_lookup.f', 'cases.dat', 'preparation.json']
    index = {'files': {name: sha(HERE / name) for name in paths},
             'inputs': {str((OLD / 'source-index.json').relative_to(ROOT)): OLD_INDEX,
                        str(ORACLE.relative_to(ROOT)): sha(ORACLE)},
             'compiler': {'path': os.environ['FC'], 'sha256': sha(Path(os.environ['FC']))}}
    (HERE / 'source-index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'cases': len(rows), 'index_sha256': sha(HERE / 'source-index.json')}))


if __name__ == '__main__':
    main()
