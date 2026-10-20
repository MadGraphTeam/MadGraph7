#!/usr/bin/env python3
"""Source-only region census; does not compile, transform, or evaluate code."""
import sys
sys.dont_write_bytecode = True
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
SOURCE = ROOT / '.benchmark-work/policy-v2/generation-abi3-tensor-factor-v1/outputs/candidate'
TARGETS = {'FFV1L1_2', 'FFV1L2_1'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logical(text):
    statements = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line[0] in 'cC*!':
            continue
        assert len(line) >= 6
        body = re.sub(r'\s+', '', line[6:]).upper()
        if line[5] not in ' 0':
            assert statements
            statements[-1]['text'] += body
        else:
            statements.append({'line': number, 'label': line[:5].strip(), 'text': body})
    return statements


def analyze_case(case):
    output = SOURCE / case
    sub = next(output.glob('SubProcesses/P*_*'))
    files = {}
    def bound(path):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
        return path.read_text()
    polynomial = bound(sub / 'polynomial.f')
    public = {}
    for match in re.finditer(r'(?m)^      SUBROUTINE (\w+UPDATE_WL_\d_\d)\(', polynomial):
        end = polynomial.index('\n      END\n', match.start()) + len('\n      END\n')
        body = polynomial[match.start():end]
        if match.group(1).startswith('MP_'):
            continue
        statements = [row['text'] for row in logical(body)]
        assert not any(text.startswith('CALL') or 'IEEE_' in text or 'EXTERNAL' in text for text in statements)
        public[match.group(1)] = hashlib.sha256(body.encode()).hexdigest()
    # A source screen for the generated arithmetic call graph, not a general
    # proof of arbitrary Fortran effects or runtime/compiler implementation.
    aloha = {}
    active = set()
    def inspect_aloha(name):
        if name in aloha:
            return aloha[name]['screen_passed']
        path = output / 'Source/DHELAS' / (name + '.f')
        if not path.exists() or name in active:
            return False
        active.add(name)
        rows = logical(bound(path))
        joined = '\n'.join(row['text'] for row in rows)
        children = [m.group(1) for row in rows if (m := re.match(r'CALL(\w+)\(', row['text']))]
        forbidden = [word for word in ('IEEE_', 'EXTERNAL', 'BIND(C', 'OPEN(', 'READ(', 'WRITE(', 'PRINT', 'STOP', 'SYSTEM(')
                     if word in joined]
        passed = not forbidden and all(inspect_aloha(child) for child in children)
        aloha[name] = {'source': path.relative_to(ROOT).as_posix(), 'calls': children,
                      'forbidden_tokens': forbidden, 'screen_passed': passed}
        active.remove(name)
        return passed
    batches = []
    totals = Counter()
    for path in sorted(sub.glob('coef_construction_*.f')):
        rows = logical(bound(path))
        call_names = [m.group(1) for row in rows if (m := re.match(r'CALL(\w+)\(', row['text']))]
        for name in call_names:
            if name not in public and not name.endswith('CREATE_LOOP_COEFS'):
                inspect_aloha(name)
        selected = {}
        for index, row in enumerate(rows):
            updater = re.match(r'CALL(\w+UPDATE_WL_([234])_1)\(', row['text'])
            prior = re.match(r'CALL(FFV1L[12]_[12])\(', rows[index-1]['text']) if index else None
            if updater and prior and prior.group(1) in TARGETS and updater.group(1) in public:
                selected[index] = {'line': row['line'], 'vertex': prior.group(1), 'rank': int(updater.group(2))}
                totals[f'{prior.group(1)}:rank{updater.group(2)}'] += 1
        policies = {}
        for policy in ('two_vertices', 'aloha_graph', 'plus_create_loop_coefs_conditional'):
            regions, current = [], []
            unknown = Counter()
            def flush():
                if current:
                    regions.append(list(current))
                    current.clear()
            for index, row in enumerate(rows):
                match = re.match(r'CALL(\w+)\(', row['text'])
                if row['label']:
                    flush()
                if not match:
                    flush()
                    continue
                name = match.group(1)
                allowed = name in public or name in TARGETS
                if policy != 'two_vertices':
                    allowed |= name in aloha and aloha[name]['screen_passed']
                if policy == 'plus_create_loop_coefs_conditional':
                    allowed |= name.endswith('CREATE_LOOP_COEFS')
                if not allowed:
                    flush()
                    unknown[name] += 1
                elif index in selected:
                    current.append(selected[index])
            flush()
            policies[policy] = {'regions_with_targets': len(regions), 'targets': sum(map(len, regions)),
                'target_count_histogram': dict(sorted(Counter(map(len, regions)).items())),
                'regions': regions, 'barrier_calls': dict(sorted(unknown.items()))}
        batches.append({'file': path.relative_to(ROOT).as_posix(), 'target_calls': len(selected),
                        'all_calls': len(call_names), 'policies': policies})
    return {'source_output': output.relative_to(ROOT).as_posix(), 'source_files': files,
        'public_updater_bodies': public, 'aloha_source_screen': aloha, 'targets_by_shape': dict(sorted(totals.items())),
        'target_calls': sum(totals.values()), 'batches': batches,
        'region_totals': {policy: sum(b['policies'][policy]['regions_with_targets'] for b in batches)
                          for policy in ('two_vertices', 'aloha_graph', 'plus_create_loop_coefs_conditional')}}


if __name__ == '__main__':
    result = {'schema': 'dp-ieee-sharing-source-census-v1', 'scope': 'Source-only feasibility census; no runtime effects proof, code generation, compilation, or evaluation.',
        'script_sha256': sha(Path(__file__)), 'cases': {case: analyze_case(case) for case in ('qcd-3', 'qcd-4')}}
    path = HERE / 'analysis.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({case: {key: data[key] for key in ('target_calls', 'targets_by_shape', 'region_totals')} for case, data in result['cases'].items()}, indent=2))
