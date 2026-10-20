#!/usr/bin/env python3
"""Exact, offline census of retained inputs; never imports a reduction engine."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict, OrderedDict
from decimal import Decimal
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CAPTURE = ROOT / 'docs/benchmarkfolder/experiments/generic-tensor-plan/capture/actual-mp-requests.jsonl.gz'
TAGGED = ROOT / 'docs/benchmarkfolder/experiments/ninjars-qp-numerator-dp-probe-v1'
EXPECTED = {
    CAPTURE: '5e6fd2c0bad620460af7742e19d4f34b1f2ce281af22e3623001c7ba74411312',
    TAGGED / 'manifest.json': '03d63cee3b8c38ee1936f9662e816206f3b99f3a2ec55b8cc938c209ff8aefbd',
    TAGGED / 'attempt-2/raw/tensor-capture.hex': '0dc33873b706be11593a59af3d3256b9958d3f86779afe86eca9fba4a9522f08',
    TAGGED / 'independent-inventory/check.py': 'efc0ec2b4ce7545a05831579c873cf6cacea3071fbfdaf20989f1311930eb2d4',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()

def scalar(value, negative=False):
    return (int(value < 0 or (not value and negative)), abs(value.numerator), value.denominator)

def rounded(value, precision):
    """MPFR-style nearest-even finite rounding, with unbounded binary exponent."""
    if not value:
        return value
    sign, value = (-1 if value < 0 else 1), abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < (Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)):
        exponent -= 1
    power = exponent - precision + 1
    unit = Fraction(1 << power) if power >= 0 else Fraction(1, 1 << -power)
    q, r = divmod((value / unit).numerator, (value / unit).denominator)
    q += 2*r > (value / unit).denominator or (2*r == (value / unit).denominator and q % 2)
    return sign * q * unit

def decimal_scalar(text):
    value = Decimal(text)
    assert value.is_finite(), 'nonfinite captured input: retain failure rather than select records'
    return scalar(rounded(Fraction(value), 113), value.is_signed())

def strip_zero_sign(values):
    return tuple((0, 0, d) if n == 0 else (s, n, d) for s, n, d in values)

def geometry(record, historical=False, rank=True):
    # Stage policy and precision prevent cross-backend/mixed-precision cache claims.
    fields = tuple((tag, strip_zero_sign(record[tag]) if historical else record[tag])
                   for tag in ('S', 'P', 'M', 'U'))
    return (record['n'], record['rank'] if rank else None, record['input_bits'],
            record['working_bits'], record['policy'], fields)

def monomials(rank):
    result, lower = [(0, 0, 0, 0)], [(0, 0, 0, 0)]
    for degree in range(1, rank + 1):
        block = []
        for component, count in enumerate((1, degree, degree*(degree+1)//2,
                                            degree*(degree+1)*(degree+2)//6)):
            for parent in lower[:count]:
                item = list(parent)
                item[component] += 1
                block.append(tuple(item))
        result.extend(block)
        lower = block
    assert len(set(result)) == math.comb(rank+4, 4)
    return result

def support(record):
    coefficients = list(zip(record['C'][::2], record['C'][1::2]))
    nodes = monomials(record['rank'])
    assert len(coefficients) == len(nodes)
    position = {node: i for i, node in enumerate(nodes)}
    needed = set()
    degree = defaultdict(Counter)
    counts = Counter()
    for i, (real, imag) in enumerate(coefficients):
        rz, iz = real[1] == 0, imag[1] == 0
        label = 'zero' if rz and iz else 'real_only' if iz else 'imaginary_only' if rz else 'general_complex'
        counts[label] += 1
        degree[str(sum(nodes[i]))][label] += 1
        counts['zero_real_components'] += rz + iz
        counts['negative_zero_components'] += (rz and real[0]) + (iz and imag[0])
        if not (rz and iz):
            while i not in needed:
                needed.add(i)
                parent = list(nodes[i])
                if not any(parent):
                    break
                c = max(j for j in range(4) if parent[j])
                parent[c] -= 1
                i = position[tuple(parent)]
    counts.update(coefficients=len(coefficients), records=1,
                  all_zero_records=int(not needed), dense_nonconstant_nodes=len(nodes)-1,
                  needed_nonconstant_nodes=len(needed-{0}),
                  unused_nonconstant_nodes=len(nodes)-1-len(needed-{0}))
    return dict(counts), {k: dict(v) for k, v in degree.items()}

def lru_hits(keys, capacity):
    cache, hits = OrderedDict(), 0
    for key in keys:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits

def stats(records):
    keys = [geometry(r) for r in records]
    occurrences = Counter(keys)
    complete = Counter((key, r['C']) for key, r in zip(keys, records))
    previous, latest_hits = {}, 0
    for r, key in zip(records, keys):
        shape = (r['n'], r['rank'], r['policy'], r['input_bits'], r['working_bits'])
        latest_hits += previous.get(shape) == key
        previous[shape] = key
    return {
        'records': len(records), 'status_counts': dict(Counter(str(r['status']) for r in records)),
        'unique_geometry': len(occurrences), 'repeated_geometry_records': len(records)-len(occurrences),
        'geometry_multiplicity_histogram': dict(sorted(Counter(str(v) for v in occurrences.values()).items())),
        'unique_full_inputs': len(complete), 'repeated_full_inputs': len(records)-len(complete),
        'geometry_groups_with_different_coefficients': sum(len({r['C'] for r in records if geometry(r) == key}) > 1 for key in occurrences),
        'most_recent_geometry_per_shape_hits': latest_hits,
        'hypothetical_global_lru_hits': {str(n): lru_hits(keys, n) for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)},
        'unique_geometry_ignoring_rank_diagnostic_only': len({geometry(r, rank=False) for r in records}),
        'historical_zero_canonicalized_unique_geometry': len({geometry(r, historical=True) for r in records}),
        'geometry_negative_zero_components': sum(s and not n for r in records for tag in ('S','P','M','U') for s,n,d in r[tag]),
    }

def grouped(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return {name: stats(rows) for name, rows in sorted(groups.items())}

def sparsity(records):
    total, by_shape, degrees = Counter(), defaultdict(Counter), defaultdict(Counter)
    for r in records:
        local, per_degree = support(r)
        total.update(local)
        by_shape[f"{r['n']}/{r['rank']}/{r['policy']}"].update(local)
        for degree, row in per_degree.items():
            degrees[degree].update(row)
    return {'total': dict(total), 'by_shape': {k: dict(v) for k,v in sorted(by_shape.items())},
            'by_degree': {k: dict(v) for k,v in sorted(degrees.items())}}

def load():
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, path
    with gzip.open(CAPTURE, 'rb') as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == '1fafb19d4b534c3e222e5b265dfd8b139f262d3353f847a5c39955c468fa9f4d'
    original = []
    for sequence, line in enumerate(raw.splitlines(), 1):
        source = json.loads(line)
        n, rank = source['n'], source['rank']
        assert source['count'] == math.comb(rank+4, 4)
        r = dict(sequence=sequence, n=n, rank=rank, status=source['status'],
                 input_bits=113, working_bits=256, policy='generic256')
        for tag, field, count in [('C','coefficients',2*source['count']), ('S','s_matrix',n*n),
                                  ('P','momenta',4*n), ('M','masses',2*n), ('U','scale',1)]:
            assert len(source[field]) == count
            r[tag] = tuple(decimal_scalar(value) for value in source[field])
        if source['status'] == 0:
            assert len(source['output']) == 8 and all(Decimal(x).is_finite() for x in source['output'])
        original.append(r)
    assert len(original) == 1392
    path = TAGGED / 'independent-inventory/check.py'
    spec = importlib.util.spec_from_file_location('sealed_independent_parser', path)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    tagged = []
    for source in decoder.parse_independently(TAGGED / 'attempt-2/raw/tensor-capture.hex'):
        stage, n, rank = source['stage'], source['n'], source['rank']
        r = {k: source[k] for k in ('sequence','point','ctmode','loop_id','order','stage','n','rank','status')}
        r.update(direction=source['ctmode']-(3 if stage == 3 else 0), input_bits=113 if stage == 3 else 53,
                 working_bits=256 if stage == 3 else 53,
                 policy={1:'native53',2:'compensated53',3:'generic256'}[stage])
        width = source['bits']//64
        def read(keys):
            words = [word for key in keys for word in source['data'][key]]
            return tuple(scalar(*decoder.decode(words[i:i+width])) for i in range(0,len(words),width))
        r['C'] = read([f'C:{i}' for i in range(math.comb(rank+4,4))])
        r['S'] = read([f'S:{i},{j}' for j in range(1,n+1) for i in range(1,n+1)])
        r['P'] = read([f'P:{i},{j}' for j in range(1,n+1) for i in range(4)])
        r['M'] = read([f'M:{j}' for j in range(1,n+1)])
        r['U'] = read(['U:'])
        tagged.append(r)
    for stage in (1,2,3):
        for point in (44,80):
            for direction in (1,2):
                assert sorted(r['loop_id'] for r in tagged if (r['stage'],r['point'],r['direction']) == (stage,point,direction)) == list(range(1,117))
    return original, tagged

def compute():
    original, tagged = load()
    tags = defaultdict(list)
    for r in tagged:
        if r['stage'] == 3:
            tags[geometry(r)].append((r['point'],r['direction'],r['loop_id'],r['order']))
    links = [{'sequence': r['sequence'], 'geometry_sha256': digest(geometry(r)),
              'candidate_tags': [list(tag) for tag in sorted(tags.get(geometry(r), []))]} for r in original]
    per_record = []
    for name, rows in [('untagged_mp', original), ('tagged_rescue', tagged)]:
        for r in rows:
            per_record.append(dict(dataset=name, sequence=r['sequence'], n=r['n'], rank=r['rank'],
                policy=r['policy'], geometry_sha256=digest(geometry(r)), full_input_sha256=digest((geometry(r),r['C'])),
                sparsity=support(r)[0]))
    result = {
        'schema':'retained-input-reuse-census-v1', 'complete':True, 'acceptance_eligible':False,
        'source_data':[{'path':str(p.relative_to(ROOT)), 'sha256':h} for p,h in EXPECTED.items()],
        'boundary_precision':'Input 53/113 significant bits; working/tracked precision at MP boundary 256. Derived geometry/master precisions are not captured.',
        'untagged_mp': {'global':stats(original), 'by_shape':grouped(original,lambda r:f"{r['n']}/{r['rank']}"), 'sparsity':sparsity(original)},
        'tagged_rescue': {'global_policy_separated':stats(tagged),
            'by_stage':grouped(tagged,lambda r:str(r['stage'])),
            'by_point_direction_stage':grouped(tagged,lambda r:f"{r['point']}/{r['direction']}/{r['stage']}"),
            'by_point_stage':grouped(tagged,lambda r:f"{r['point']}/{r['stage']}"), 'sparsity':sparsity(tagged)},
        'cross_capture_input_links': {'semantics':'Exact boundary geometry only, across different library/source identities; all possible tags retained, no label inference.',
             'candidate_count_histogram':dict(Counter(str(len(row['candidate_tags'])) for row in links)), 'records':links},
        'record_inventory':per_record,
    }
    return result

def selftest():
    assert rounded(Fraction(1)+Fraction(1,2**113),113) == 1
    assert rounded(Fraction(1)+Fraction(3,2**113),113) == 1+Fraction(1,2**111)
    assert decimal_scalar('-0') == (1,0,1) and decimal_scalar('0.000') == (0,0,1)
    assert decimal_scalar('1.0') == decimal_scalar('1e0')
    assert monomials(1) == [(0,0,0,0),(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
    nodes = monomials(2)
    c = [(0,0,1)] * (2*len(nodes))
    c[2*nodes.index((0,0,0,2))] = (0,1,1)
    s,_ = support({'rank':2,'C':tuple(c)})
    assert s['needed_nonconstant_nodes'] == 2 and s['unused_nonconstant_nodes'] == 12
    assert lru_hits(['A','B','A','C','A','B'],2) == 2
    base = {'n':1,'rank':1,'input_bits':113,'working_bits':256,'policy':'generic256',
            'S':((0,1,1),),'P':((0,0,1),),'M':((0,0,1),),'U':((0,1,1),)}
    other = dict(base,P=((1,0,1),))
    assert geometry(base) != geometry(other) and geometry(base,True) == geometry(other,True)
    assert geometry(base) != geometry(dict(base,working_bits=384))
    assert geometry(base) != geometry(dict(base,rank=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    selftest()
    actual = compute()
    path = HERE/'result.json'
    if args.write:
        assert not path.exists(), 'never overwrite retained census'
        path.write_text(json.dumps(actual,indent=2,sort_keys=True)+'\n')
    else:
        assert actual == json.loads(path.read_text())
    print(json.dumps({'passed':True,'records':len(actual['record_inventory']),
        'untagged':actual['untagged_mp']['global'],
        'tagged_qp':actual['tagged_rescue']['by_stage']['3'],
        'sparsity':actual['untagged_mp']['sparsity']['total']},sort_keys=True))

if __name__ == '__main__':
    main()
