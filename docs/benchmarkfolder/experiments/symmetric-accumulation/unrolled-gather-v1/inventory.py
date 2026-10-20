#!/usr/bin/env python3
"""Source-only inventory of compact linear-vertex calls and ALOHA coefficients."""
from collections import Counter, defaultdict
import hashlib
import json
from math import comb
from pathlib import Path
import re

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
HERE = Path(__file__).resolve().parent


def logical(path):
    rows = []
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw[0] in 'cC*!':
            continue
        body = re.sub(r'\s+', '', raw[6:]).upper()
        if raw[5] not in ' 0':
            rows[-1] += body
        else:
            rows.append(body)
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(case):
    output = ROOT/'.benchmark-work/experiments/terminal-production'/case/'production'
    proc = next((output/'SubProcesses').glob('P*_*'))
    pattern = re.compile(r'CALLMP_\w+(UPDATE|TERMINAL)_WL_(\d+)_(\d+)'
        r'\(WL\(1,0,1,-?\d+\),(\d+),COEFS,(\d+),(\d+),WL\(')
    counts = Counter()
    calls = []
    sources = {}
    for path in sorted(proc.glob('mp_coef_construction_*.f')):
        sources[str(path.relative_to(ROOT))] = sha(path)
        vertex = None
        for row in logical(path):
            match = re.match(r'CALL(MP_\w+)\(.*?,COEFS\)$',row)
            if match:
                vertex = match[1]
            match = pattern.match(row)
            if match:
                mode,left,right,cut,ins,outs = match.groups()
                counts[f'{mode}:{left}/{right}'] += 1
                if int(left) >= 2 and int(right) == 1:
                    assert vertex, 'Missing preceding ALOHA vertex'
                    calls.append(dict(mode=mode,left=int(left),cut=int(cut),
                                      ins=int(ins),outs=int(outs),vertex=vertex))
    routines = {}
    for name in sorted({call['vertex'] for call in calls}):
        path = output/'Source/DHELAS'/(name+'.f')
        assignments = {}
        for row in logical(path):
            match = re.fullmatch(r'COEFF\((\d+),(\d+),(\d+)\)=(.*)',row)
            if match:
                j,m,k = map(int,match.groups()[:3])
                assert (j,m,k) not in assignments
                assignments[j,m,k] = match[4]
        sources[str(path.relative_to(ROOT))] = sha(path)
        uses = [c for c in calls if c['vertex']==name]
        shapes = {(c['outs'],c['ins']) for c in uses}
        shape_rows = []
        for outs,ins in sorted(shapes):
            active = {(j,m,k):assignments[j,m,k] for j in range(1,outs+1)
                      for m in range(5) for k in range(1,ins+1)}
            zeros = [key for key,rhs in active.items() if rhs in ('0Q0','0D0','0')]
            grouped = defaultdict(list)
            for key,rhs in active.items():
                if key not in zeros:
                    grouped[rhs].append(key)
            repeats = [values for values in grouped.values() if len(values)>1]
            same_k = []
            for values in repeats:
                for k in sorted({key[2] for key in values}):
                    group = [key for key in values if key[2] == k]
                    if len(group) > 1:
                        same_k.append(group)
            same_jk = []
            for values in same_k:
                for j in sorted({key[0] for key in values}):
                    group = [key for key in values if key[0] == j]
                    if len(group) > 1:
                        same_jk.append(group)
            shape_rows.append(dict(out_size=outs,in_size=ins,coefficients=len(active),
                literal_zero_entries=zeros,nonzero_literal_entries=len(active)-len(zeros),
                identical_expression_groups=repeats,
                identical_expression_extra_entries=sum(len(g)-1 for g in repeats),
                identical_expression_same_k_groups=same_k,
                identical_expression_same_k_extra_entries=sum(len(g)-1 for g in same_k),
                identical_expression_same_jk_groups=same_jk,
                identical_expression_same_jk_extra_entries=sum(len(g)-1 for g in same_jk)))
        routines[name] = dict(calls=len(uses),shapes=shape_rows,
            caveat='Nonliteral expressions contain arbitrary complex wavefunctions/couplings; '
            'no pure-real/pure-imaginary property is inferred. Same-K equal expressions can '
            'reuse one A(K,L,I)*B product across different destination components/monomials; '
            'cross-K equality alone cannot. Sharing needs a separate arithmetic/flags proof.')
    weighted_slots = weighted_zeros = weighted_duplicates = weighted_same_k = 0
    products_after_literal_zeros = possible_shared_products = 0
    for call in calls:
        shape = next(s for s in routines[call['vertex']]['shapes']
                     if s['out_size']==call['outs'] and s['in_size']==call['ins'])
        weighted_slots += shape['coefficients']
        weighted_zeros += len(shape['literal_zero_entries'])
        weighted_duplicates += shape['identical_expression_extra_entries']
        weighted_same_k += shape['identical_expression_same_k_extra_entries']
        factor = comb(call['left']+4,4) * (call['cut'] if call['mode']=='UPDATE' else 1)
        # All observed terminal shapes have equal cut and outgoing sizes.
        assert call['mode']=='UPDATE' or call['cut']==call['outs']
        products_after_literal_zeros += shape['nonzero_literal_entries']*factor
        possible_shared_products += factor*shape[
            'identical_expression_same_k_extra_entries' if call['mode']=='UPDATE'
            else 'identical_expression_same_jk_extra_entries']
    return dict(call_counts=dict(counts),compact_rank_one_calls=len(calls),
        calls=calls,routines=routines,sources=sources,
        call_weighted_B_slots=weighted_slots,
        call_weighted_literal_zero_B_slots=weighted_zeros,
        call_weighted_repeated_expression_extra_slots=weighted_duplicates,
        call_weighted_same_k_repeated_expression_extra_slots=weighted_same_k,
        complex_products_after_literal_zero_skip_before_A_zero_skip=products_after_literal_zeros,
        possible_shared_products_same_A_and_B=possible_shared_products,
        same_A_B_product_fraction=possible_shared_products/products_after_literal_zeros if products_after_literal_zeros else None,
        literal_zero_fraction=weighted_zeros/weighted_slots if weighted_slots else None,
        scope='Static generated calls per coefficient batch; not runtime frequency. Existing '
        'B==0 branches already skip literal-zero products. These counts establish potential '
        'mask/index overhead only, not additional arithmetic savings.')


def main():
    result = {case:inventory(case) for case in ('qcd-3','ew-2')}
    (HERE/'inventory.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for case,data in result.items():
        print(case,{key:data[key] for key in ('compact_rank_one_calls','call_weighted_B_slots',
            'call_weighted_literal_zero_B_slots','literal_zero_fraction',
            'call_weighted_repeated_expression_extra_slots',
            'call_weighted_same_k_repeated_expression_extra_slots')})


if __name__=='__main__':
    main()
