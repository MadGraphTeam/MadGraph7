#!/usr/bin/env python3
"""Derive historical CT4/5 incidence using integer arithmetic and pinned sources.

Run from the MadGraph root. The default creates routing-modes.json once;
--verify recomputes it without writing; --output selects a new artifact path.
No compiler, native library, MadGraph import or floating arithmetic is used.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FROZEN = Path('docs/benchmarkfolder/experiments/qcd-loop-kinematics-trace')
ROUTING = FROZEN / 'design/topology/routing.json'
TARGETS = FROZEN / 'design-amendment-1/targets.json'
PROCESS = Path('.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg/SubProcesses/P0_uux_zgg')
SWITCH = PROCESS / 'TIR_interface.f'
ADAPTER = PROCESS / 'TIRUST_interface.f'
PINS = {
    ROUTING: '1198fbb0fc62de9a4bf23abafb704aa4a4b75f4e7df12bac605970d14e6a1fde',
    TARGETS: '6c11a10b5e89ea2346a211c492a24f07797050a3df3190b40394a529566c3ba5',
    SWITCH: '2a1dffb507f359a9e0c0fe5ff4f597aa752306284e027a2d8c6231acb2b15241',
    ADAPTER: '90aa42f27796c7b8762ccfc5c7240a3dc90e21a399078a51031355eda15846d0',
}
PHYSICAL_SIGNS = (1, 1, -1, -1, -1)
TOTAL = [1] * 5
TARGET = [1, 0, 0, 1, 0]  # k1+k4 = p1-p4; leg 4 is the amended soft leg.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def add(vectors):
    return [sum(vector[i] for vector in vectors) for i in range(5)]


def subtract(left, right):
    return [a - b for a, b in zip(left, right)]


def physical(vector):
    return [value * sign for value, sign in zip(vector, PHYSICAL_SIGNS)]


def equivalent(vector, target):
    """Return exact witnesses vector=sign*target+multiple*TOTAL, or None."""
    for sign in (1, -1):
        residual = [a - sign * b for a, b in zip(vector, target)]
        if len(set(residual)) == 1:
            return {'sign': sign, 'conservation_multiple': residual[0]}
    return None


def classify(vector):
    support = [i for i, value in enumerate(vector) if value]
    require(0 < len(support) < 5, 'unexpected zero or pure-conservation pair')
    signs = {vector[i] for i in support}
    require(len(signs) == 1 and next(iter(signs)) in (-1, 1), 'not a signed subset')
    sign = next(iter(signs))
    if len(support) == 1:
        category, leg = 'direct_single_leg', support[0] + 1
    elif len(support) == 4:
        category = 'complement_of_single_leg'
        leg = next(i + 1 for i in range(5) if i not in support)
    else:
        category, leg = 'composite', None
    canonical_support = support if len(support) <= 2 else [i for i in range(5) if i not in support]
    canonical = [int(i in canonical_support) for i in range(5)]
    witness = equivalent(vector, canonical)
    require(witness is not None, 'canonical conservation witness missing')
    return {
        'classification': category,
        'literal_coefficients_k': vector,
        'literal_coefficients_physical_p': physical(vector),
        'literal_support_legs': [i + 1 for i in support],
        'literal_support_size': len(support),
        'literal_sign': sign,
        'single_leg': leg,
        'single_leg_kind': ('Z' if leg == 3 else 'massless') if leg else None,
        'canonical_channel_legs': [i + 1 for i in canonical_support],
        'canonical_coefficients_k': canonical,
        'canonical_coefficients_physical_p': physical(canonical),
        'canonical_witness': witness,
        'target_p1_minus_p4': canonical == TARGET,
    }


def anchor(lines, path, line, expected):
    require(expected in lines[line - 1], f'{path}:{line} anchor changed')
    return {'path': str(path), 'line': line, 'text': lines[line - 1].strip()}


def derive():
    require(Path.cwd().resolve() == ROOT, 'run from the MadGraph repository root')
    inputs, bindings = {}, {}
    for path, expected in PINS.items():
        raw = (ROOT / path).read_bytes()
        require(sha(raw) == expected, f'pinned source/input mismatch: {path}')
        inputs[path] = raw
        bindings[str(path)] = {'bytes': len(raw), 'sha256': expected}
    routing = json.loads(inputs[ROUTING])
    amended = json.loads(inputs[TARGETS])
    require(amended['bindings'][str(ROUTING)] == PINS[ROUTING], 'amendment routing binding differs')
    require(amended['target_pair_count'] == 19, 'amended target count changed')
    require([g['group'] for g in routing['groups']] == list(range(1, 117)), 'group sequence changed')
    primary_targets = {row['group']: row for row in amended['target_pairs']}
    require(len(primary_targets) == 19, 'expected one amended target per selected group')
    switch_lines = inputs[SWITCH].decode().splitlines()
    adapter_lines = inputs[ADAPTER].decode().splitlines()
    anchors = [anchor(switch_lines, SWITCH, line, text) for line, text in (
        (327, 'SUBROUTINE ML5_0_MP_SWITCH_ORDER'),
        (341, 'IF (CTMODE.NE.2.AND.CTMODE.NE.5)'),
        (345, 'IF (NLOOPLINE.LE.2)'),
        (361, 'NEW_M2L(I)=M2L(NLOOPLINE-I+2)'),
        (370, 'NEW_PL(J,I) = -PL(J,NLOOPLINE+1-I)'),
    )]
    anchors += [anchor(adapter_lines, ADAPTER, line, text) for line, text in (
        (164, 'CALL ML5_0_MP_SWITCH_ORDER'), (167, 'P_TMP=0'),
        (171, 'P_TMP(I,0)=P_TMP(I,0)+PL(I,J)'),
        (183, 'PREFIX_SUM=REAL(0,KIND=KI_QTIR)'),
        (185, 'PREFIX_SUM=PREFIX_SUM+PL(I,J)'),
        (191, 'SHIFTS(I,J)=P_TMP(I,J-1)'),
    )]
    all_modes, target_modes, summaries = {}, {}, {}
    for mode in (4, 5):
        groups, selected = [], []
        categories, detailed, support_sizes = Counter(), Counter(), Counter()
        for frozen in routing['groups']:
            n, group_id = frozen['n'], frozen['group']
            original = [[int(leg in corner) for leg in range(1, 6)] for corner in frozen['corner_legs']]
            require(len(original) == n and add(original) == TOTAL, 'invalid corner partition')
            reversed_order = mode == 5 and n > 2
            corner_positions = list(range(n, 0, -1)) if reversed_order else list(range(1, n + 1))
            corner_sign = -1 if reversed_order else 1
            corners = [[corner_sign * value for value in original[position - 1]] for position in corner_positions]
            shifts = [add(corners)] + [add(corners[:k]) for k in range(1, n)]
            original_shifts = [TOTAL.copy()] + [add(original[:k]) for k in range(1, n)]
            mass_permutation = [1] + list(range(n, 1, -1)) if reversed_order else list(range(1, n + 1))
            offsets = []
            for shift, source in zip(shifts, mass_permutation):
                residual = subtract(shift, original_shifts[source - 1])
                require(len(set(residual)) == 1, 'shift permutation requires non-conservation change')
                offsets.append(residual[0])
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    row = {'propagators': [i + 1, j + 1], **classify(subtract(shifts[i], shifts[j]))}
                    primary_pair = sorted([mass_permutation[i], mass_permutation[j]])
                    primary_vector = subtract(original_shifts[primary_pair[0] - 1], original_shifts[primary_pair[1] - 1])
                    row['primary_order_pair'] = primary_pair
                    row['primary_order_witness'] = equivalent(row['literal_coefficients_k'], primary_vector)
                    require(row['primary_order_witness'] is not None, 'mode permutation changes canonical channel')
                    if mode == 4:
                        saved = next(p for p in frozen['pairs'] if p['propagators'] == row['propagators'])
                        require(saved['literal_difference_coefficients'] == row['literal_coefficients_k'],
                                'CT4 literal incidence differs from frozen proof')
                    categories[row['classification']] += 1
                    detailed[row['classification'] + (':' + row['single_leg_kind'] if row['single_leg'] else '')] += 1
                    support_sizes[row['literal_support_size']] += 1
                    if row['target_p1_minus_p4']:
                        require(group_id in primary_targets, 'derived target group absent from amendment')
                        expected = primary_targets[group_id]
                        require(primary_pair == expected['pair']['propagators'], 'target permutation differs from amendment')
                        require(n == expected['n'] and frozen['rank'] == expected['rank'], 'target metadata changed')
                        selected.append({'ctmode': mode, 'group': group_id, 'n': n, 'rank': frozen['rank'],
                                         'source_call': frozen['call'], 'pair': row,
                                         'target_witness': equivalent(row['literal_coefficients_k'], TARGET)})
                    pairs.append(row)
            groups.append({'group': group_id, 'ctmode': mode, 'n': n, 'rank': frozen['rank'],
                           'source_call': frozen['call'], 'reversed_and_negated': reversed_order,
                           'corner_source_positions': corner_positions, 'corner_sign': corner_sign,
                           'corner_coefficients_k': corners, 'corner_coefficients_physical_p': [physical(c) for c in corners],
                           'shift_coefficients_k': shifts, 'shift_coefficients_physical_p': [physical(q) for q in shifts],
                           'mass_source_denominators': mass_permutation,
                           'shift_source_denominators_modulo_conservation': mass_permutation,
                           'shift_conservation_offsets': offsets, 'pairs': pairs})
        require(len(selected) == 19 and {t['group'] for t in selected} == set(primary_targets), 'target coverage differs')
        require(sum(categories.values()) == 477 and categories['composite'] == 235, 'pair coverage changed')
        expected_counts = ({'direct_single_leg': 146, 'complement_of_single_leg': 96, 'composite': 235}
                           if mode == 4 else {'direct_single_leg': 216, 'complement_of_single_leg': 26, 'composite': 235})
        require(dict(categories) == expected_counts, 'literal classification count changed')
        all_modes[str(mode)] = groups
        target_modes[str(mode)] = selected
        summaries[str(mode)] = {'groups': len(groups), 'pairs': sum(categories.values()), 'targets': len(selected),
                                'classifications': dict(sorted(categories.items())),
                                'detailed_classifications': dict(sorted(detailed.items())),
                                'literal_support_sizes': {str(k): v for k, v in sorted(support_sizes.items())}}
    script = Path(__file__).read_bytes()
    return {'schema': 1, 'native_executed': False, 'compiler_executed': False, 'arithmetic': 'exact Python integers only',
            'input_bindings': bindings, 'script': {'bytes': len(script), 'sha256': sha(script)},
            'source_anchors': anchors,
            'basis': {'k': ['p1', 'p2', '-p3', '-p4', '-p5'], 'conservation_coefficients_k': TOTAL,
                      'conservation_coefficients_physical_p': list(PHYSICAL_SIGNS)},
            'witness_definition': 'literal = sign * reference + conservation_multiple * (1,1,1,1,1) in k basis',
            'target': {'name': 'p1-p4', 'soft_leg': 4, 'coefficients_k': TARGET,
                       'coefficients_physical_p': physical(TARGET), 'hard_comparator': 'p1-p5'},
            'limitations': [
                'Formal incidence only; no observed matrix clamp, backend status or evaluated invariant.',
                'The first shift is the computed sum of all corners, not an imposed zero.',
                'Conservation witnesses do not assert floating-point equality or bitwise invariance under reversal.',
                'The direct/complement classification uses literal coefficients, not canonical representatives.',
                'Original-promoted B and repaired C events share topology but can have different channel and Born values.',
            ],
            'summary': summaries, 'target_pairs': target_modes, 'groups_by_mode': all_modes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--output', type=Path)
    options.add_argument('--verify', nargs='?', const=HERE / 'routing-modes.json', type=Path)
    args = parser.parse_args()
    result = derive()
    if args.verify:
        require(json.loads(args.verify.read_text()) == result, 'saved mode routing differs from the recomputed proof')
    else:
        destination = args.output or HERE / 'routing-modes.json'
        with destination.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
