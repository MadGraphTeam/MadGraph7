#!/usr/bin/env python3
"""Audit the historical u u~ > Z g g routing using source text only.

No MadGraph imports, compilation, shared-library loading, or numerical calls.
The default prints a compact summary; --output saves deterministic evidence;
--verify recomputes that evidence and requires exact JSON equality.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[6]
DEFAULT_EXPORT = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
PROCESS = Path('SubProcesses/P0_uux_zgg')
ALL = 31


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compact(text):
    return ''.join(text.split()).upper()


def statements(text):
    """Retain fixed-format statement anchors and join column-six continuations."""
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line[0] in 'cC*!':
            continue
        require(len(line) >= 6, f'unexpected short source line {number}: {line}')
        continuation = line[5] not in ' 0'
        fragment = line[6:].strip()
        if continuation:
            require(bool(result), f'orphan continuation at {number}')
            result[-1]['text'] += fragment
            result[-1]['last_line'] = number
        else:
            result.append({'line': number, 'last_line': number, 'text': fragment})
    return result


def arguments(text):
    result, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            require(depth >= 0, 'unbalanced argument list')
        elif char == ',' and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
    require(depth == 0, 'unbalanced argument list')
    result.append(text[start:].strip())
    return result


def legs(mask):
    return [leg for leg in range(1, 6) if mask & (1 << (leg - 1))]


class Sources:
    def __init__(self, root):
        self.root, self.cache, self.manifest = root, {}, {}

    def read(self, relative):
        key = str(relative)
        if key not in self.cache:
            raw = (self.root / relative).read_bytes()
            self.manifest[key] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            self.cache[key] = statements(raw.decode('utf-8'))
        return self.cache[key]

    def routine(self, relative, name):
        source = self.read(relative)
        matches = [(i, re.fullmatch(r'SUBROUTINE' + re.escape(name) + r'\((.*)\)',
                                   compact(s['text']))) for i, s in enumerate(source)]
        matches = [(i, match) for i, match in matches if match]
        require(len(matches) == 1, f'{relative}: expected one routine {name}')
        start, match = matches[0]
        end = next((i for i in range(start + 1, len(source))
                    if compact(source[i]['text']).startswith('SUBROUTINE')), len(source))
        return arguments(match[1]), source[start:end]


def anchor(path, statement):
    return {'file': str(path), 'line': statement['line'],
            'last_line': statement['last_line'], 'statement': statement['text']}


def prove_producers(sources, mp):
    name = 'mp_helas_calls_ampb_1.f' if mp else 'helas_calls_ampb_1.f'
    path = PROCESS / name
    producers, masks = [], {}
    for statement in sources.read(path):
        match = re.fullmatch(r'CALL(\w+)\((.*)\)', compact(statement['text']))
        if not match:
            continue
        routine, args = match[1], arguments(match[2])
        output = re.fullmatch(r'W\((\d+)\)', args[-1])
        if not output:
            continue
        slot = int(output[1])
        require(slot not in masks, f'{path}: W({slot}) overwritten')
        base = routine.removeprefix('MP_')
        require(routine.startswith('MP_') == mp, 'unexpected producer precision')
        external = base in ('IXXXXX', 'OXXXXX', 'VXXXXX')
        definition = Path('Source/DHELAS') / ('aloha_functions.f' if external else routine + '.f')
        formals, body = sources.routine(definition, routine)
        require(len(formals) == len(args), f'{routine}: signature/call arity mismatch')
        evidence = []
        if external:
            momentum = re.fullmatch(r'P\(0,(\d+)\)', args[0])
            require(momentum is not None, 'unexpected external momentum argument')
            leg = int(momentum[1])
            require(1 <= leg <= 5 and slot == leg, 'unexpected external wavefunction ordering')
            sign_name = 'NSV' if base == 'VXXXXX' else 'NSF'
            sign = int(args[formals.index(sign_name)])
            require(sign == (-1 if base == 'OXXXXX' else 1), 'unexpected external direction')
            factor = '(-NSF)' if base == 'IXXXXX' else sign_name
            for component in range(4):
                expected = f'{formals[-1]}%P({component})=P({component})*{factor}'
                hits = [s for s in body if compact(s['text']) == expected]
                require(len(hits) == 1, f'{routine}: missing or repeated momentum rule {expected}')
                evidence.append(anchor(definition, hits[0]))
            mothers = []
            mask = 1 << (leg - 1)
            physical_sign = -sign if base == 'IXXXXX' else sign
            require(physical_sign == (-1 if leg <= 2 else 1), 'external W is not minus incoming k')
        else:
            mothers = [(i, int(m[1])) for i, arg in enumerate(args[:-1])
                       if (m := re.fullmatch(r'W\((\d+)\)', arg))]
            require(len(mothers) == 2, f'{routine}: historical producer must have two mothers')
            require(all(slot in masks for _, slot in mothers), f'{routine}: undefined mother')
            expected = formals[-1] + '%P(:)=' + ''.join('+' + formals[i] + '%P(:)' for i, _ in mothers)
            writes = [s for s in body if re.match(re.escape(formals[-1]) + r'%P\(', compact(s['text']))]
            require(len(writes) == 1 and compact(writes[0]['text']) == expected,
                    f'{routine}: unexpected offshell momentum rule')
            evidence.append(anchor(definition, writes[0]))
            mask = 0
            for _, mother in mothers:
                require(not mask & masks[mother], 'producer has overlapping external legs')
                mask |= masks[mother]
        masks[slot] = mask
        producers.append({'wavefunction': slot, 'routine': routine, 'mask': mask,
                          'legs': legs(mask), 'mothers': [slot for _, slot in mothers],
                          'call': anchor(path, statement), 'momentum_rules': evidence})
    require(sorted(masks) == list(range(1, 36)), 'historical W1..W35 producer set changed')
    return masks, producers


def prove_interface(sources, names):
    path = PROCESS / 'CT_interface.f'
    proofs = []
    for name, n, nw in sorted(names):
        formals, body = sources.routine(path, name)
        expected = [f'WE({i})=W{i}' for i in range(1, nw + 1)]
        expected += ([f'PAIRING({i})=P{i}' for i in range(1, n + 1)]
                     if nw != n else ['PAIRING(I)=1'])
        expected += ['TEMP=1', 'DOJ=1,NLOOPLINE', 'DOK=TEMP,(TEMP+PAIRING(J)-1)',
                     'PL(I,J)=PL(I,J)-W(WE(K))%P(I)',
                     'MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)',
                     'TEMP=TEMP+PAIRING(J)', 'ID=LOOPNUM']
        hits = []
        for item in expected:
            found = [s for s in body if compact(s['text']) == item]
            require(len(found) == 1, f'{name}: missing/repeated routing statement {item}')
            hits.append(anchor(path, found[0]))
        proofs.append({'routine': name, 'arguments': formals, 'anchors': hits})
    path = PROCESS / 'TIRUST_interface.f'
    source = sources.read(path)
    expected = ['P_TMP=0', 'P_TMP(I,0)=P_TMP(I,0)+PL(I,J)',
                'PREFIX_SUM=PREFIX_SUM+PL(I,J)', 'P_TMP(I,J)=PREFIX_SUM',
                'SHIFTS(I,J)=P_TMP(I,J-1)']
    shift_proof = []
    for item in expected:
        hits = [s for s in source if compact(s['text']) == item]
        require(len(hits) == 2, f'DP/MP cumulative-shift rule changed: {item}')
        shift_proof.extend(anchor(path, s) for s in hits)
    return proofs, shift_proof


def audit(export):
    sources = Sources(export)
    masks, dp = prove_producers(sources, False)
    mp_masks, mp = prove_producers(sources, True)
    require(masks == mp_masks, 'DP/MP external-leg incidence differs')
    require([(p['wavefunction'], p['mothers']) for p in dp] ==
            [(p['wavefunction'], p['mothers']) for p in mp], 'DP/MP producer graphs differ')
    groups, names = [], set()
    counts, arc_sizes, literal_sizes, singleton_legs = Counter(), Counter(), Counter(), Counter()
    by_n = defaultdict(Counter)
    path = PROCESS / 'loop_CT_calls_1.f'
    for statement in sources.read(path):
        match = re.fullmatch(r'CALL(ML5_0_LOOP_(\d+)(?:_(\d+))?)\((.*)\)', compact(statement['text']))
        if not match:
            require(not compact(statement['text']).startswith('CALLML5_0_LOOP_'), 'unparsed loop call')
            continue
        name, n = match[1], int(match[2])
        nw = int(match[3]) if match[3] else n
        args = arguments(match[4])
        require(len(args) == nw + n + 3 + (n if match[3] else 0), 'loop call argument count changed')
        pairing = list(map(int, args[:n])) if match[3] else [1] * n
        start = n if match[3] else 0
        wavefunctions = list(map(int, args[start:start + nw]))
        require(sum(pairing) == nw and all(x > 0 for x in pairing), 'invalid corner grouping')
        require(args[-2] == 'I_SO', 'unexpected squared-order argument')
        rank, group = int(args[-3]), int(args[-1])
        corners, offset, used = [], 0, 0
        for size in pairing:
            corner = 0
            for w in wavefunctions[offset:offset + size]:
                require(w in masks and not corner & masks[w], 'overlapping/missing corner mothers')
                corner |= masks[w]
            require(not used & corner, 'loop corners overlap')
            corners.append(corner)
            used |= corner
            offset += size
        require(used == ALL, f'group {group}: corners do not partition all five external legs')
        # Actual adapter q1 is the computed closure sum, not an assigned zero.
        # q2=C1, q3=C1+C2, ... . Compare this literal representation with the
        # forward arc, whose invariant is equivalent only modulo sum(k)=0.
        shifts, prefix = [ALL], 0
        for corner in corners[:-1]:
            prefix |= corner
            shifts.append(prefix)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                arc = 0
                for corner in corners[i:j]:
                    arc |= corner
                positive = shifts[i] & ~shifts[j]
                negative = shifts[j] & ~shifts[i]
                require(bool(positive) != bool(negative), 'unexpected mixed-sign literal difference')
                literal, sign = (positive, 1) if positive else (negative, -1)
                require(literal in (arc, ALL ^ arc), 'arc/literal mismatch modulo conservation')
                single = arc if len(legs(arc)) == 1 else ALL ^ arc
                leg = legs(single)[0] if len(legs(single)) == 1 else None
                category = 'single_Z' if leg == 3 else 'single_massless' if leg else 'composite'
                counts[category] += 1
                by_n[n][category] += 1
                arc_sizes[len(legs(arc))] += 1
                literal_sizes[len(legs(literal))] += 1
                if leg:
                    singleton_legs[leg] += 1
                pairs.append({'propagators': [i + 1, j + 1], 'forward_arc_mask': arc,
                              'forward_arc_legs': legs(arc), 'literal_difference_mask': literal,
                              'literal_difference_sign': sign, 'literal_difference_legs': legs(literal),
                              'literal_difference_coefficients': [sign if literal & (1 << k) else 0
                                                                  for k in range(5)],
                              'literal_is_arc_complement': literal == ALL ^ arc,
                              'classification_modulo_conservation': category, 'single_leg': leg})
        names.add((name, n, nw))
        groups.append({'group': group, 'n': n, 'rank': rank, 'call': anchor(path, statement),
                       'wavefunctions': wavefunctions, 'pairing': pairing, 'corner_masks': corners,
                       'corner_legs': [legs(mask) for mask in corners], 'literal_shift_masks': shifts,
                       'literal_shift_legs': [legs(mask) for mask in shifts], 'pairs': pairs})
    require([g['group'] for g in groups] == list(range(1, 117)), 'historical 116-group sequence changed')
    require(dict(counts) == {'composite': 235, 'single_massless': 178, 'single_Z': 64},
            'historical pair classification changed')
    require([pair['literal_difference_coefficients'] for pair in groups[42]['pairs']] ==
            [[0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 0, 0, -1]],
            'historical group43 literal closure-sum routing changed')
    interface, shift_proof = prove_interface(sources, names)
    parser_raw = Path(__file__).read_bytes()
    return {'schema_version': 1, 'scope': 'Static historical source proof; no floating-point evaluation.',
            'convention': 'k=(p1,p2,-p3,-p4,-p5); W%P=-sum(k over producer subset); PL=sum(k over corner).',
            'ordering': 'Original CTMODE 1/4 order. CTMODE 2/5 reversal is outside this supplement.',
            'closure_caveat': 'Literal q1 is sum of all corners. Singleton/composite classification uses exact momentum conservation, not an assertion that floating-point closure is zero.',
            'parser': {'sha256': hashlib.sha256(parser_raw).hexdigest(), 'bytes': len(parser_raw)},
            'sources': dict(sorted(sources.manifest.items())), 'producers': {'DP': dp, 'MP': mp},
            'interface_proof': interface, 'shift_construction_proof': shift_proof,
            'groups': groups,
            'summary': {'groups': len(groups), 'pairs': sum(counts.values()),
                        'classification_modulo_conservation': dict(sorted(counts.items())),
                        'by_n': {str(n): dict(sorted(c.items())) for n, c in sorted(by_n.items())},
                        'single_leg_counts': {str(k): v for k, v in sorted(singleton_legs.items())},
                        'forward_arc_subset_sizes': {str(k): v for k, v in sorted(arc_sizes.items())},
                        'literal_difference_subset_sizes': {str(k): v for k, v in sorted(literal_sizes.items())}},
            'group43': next(g for g in groups if g['group'] == 43)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-root', type=Path, default=DEFAULT_EXPORT)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--output', type=Path)
    operation.add_argument('--verify', type=Path)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'run from the MadGraph checkout')
    evidence = audit(args.export_root.resolve())
    if args.output:
        require(not args.output.exists(), 'output already exists; preserve prior evidence')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    if args.verify:
        require(json.loads(args.verify.read_text()) == evidence, 'saved evidence differs from current source proof')
    print(json.dumps(evidence['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
