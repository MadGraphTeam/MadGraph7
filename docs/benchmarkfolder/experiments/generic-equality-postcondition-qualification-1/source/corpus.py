"""Pure frozen-word corpus and independent Fraction oracle; no native execution.

Expected refusal reasons describe the mathematical/source capacity domain.
The actual Fortran API returns a logical result only, never a refusal code.
"""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse
import copy
import hashlib
import importlib.util
import json
import random

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'generic-equality-postcondition-proposal-1'
PARENT_SHA='3d2e4b775334374082771f670996e67138205b66cf7d5a7ff084013235046f4c'
MAXIMUM=2**127-1


def require(value,message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_module():
    require(digest(PARENT/'source-index.json')==PARENT_SHA,'Frozen helper index changed')
    index=json.loads((PARENT/'source-index.json').read_text())
    require({str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==
            set(index['files'])|{'source-index.json'},'Frozen helper inventory changed')
    for name,expected in index['files'].items():
        path=PARENT/name
        require(not path.is_symlink() and digest(path)==expected['sha256'] and
                path.stat().st_size==expected['bytes'],'Frozen helper payload changed: '+name)
    spec=importlib.util.spec_from_file_location('frozen_equality_algorithm_crosscheck',PARENT/'reference.py')
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def value(bits,width):
    """Independent rational IEEE semantics; does not call the frozen spec decoder."""
    require(type(bits) is int and 0<=bits<2**width,'malformed-word')
    nf,ne,bias={64:(52,11,1023),128:(112,15,16383)}[width]
    sign=-1 if bits//2**(width-1) else 1
    stored=(bits//2**nf)%(2**ne)
    fraction=bits%2**nf
    require(stored!=2**ne-1,'nonfinite')
    return (sign*Fraction(2)**(stored-bias)*(1+Fraction(fraction,2**nf)) if stored else
            sign*Fraction(fraction,2**nf)*Fraction(2)**(1-bias))


def exact_word(number,width):
    number=Fraction(number)
    nf,ne,bias={64:(52,11,1023),128:(112,15,16383)}[width]
    if not number:
        return 0
    sign=int(number<0)
    number=abs(number)
    floor=number.numerator.bit_length()-number.denominator.bit_length()
    if number<Fraction(2)**floor:
        floor-=1
    stored=max(0,floor+bias)
    require(stored<2**ne-1,'Unrepresentable fixture exponent')
    scaled=number/Fraction(2)**max(floor-nf,1-bias-nf)
    require(scaled.denominator==1,'Unrepresentable fixture significand')
    fraction=int(scaled)-(2**nf if stored else 0)
    require(0<=fraction<2**nf,'Unrepresentable fixture fraction')
    bits=(sign<<(width-1))|(stored<<nf)|fraction
    require(value(bits,width)==(-number if sign else number),'Fixture encoding failed independent decode')
    return bits


def vectors(numbers,width):
    return [[[exact_word(v,width),0] for v in row] for row in numbers]


def odd_dyadic(number):
    """Positive odd significand/exponent obtained from an exact Fraction."""
    number=abs(number)
    if not number:
        return 0,0
    numerator=number.numerator
    zeros=(numerator&-numerator).bit_length()-1
    return numerator>>zeros,zeros-(number.denominator.bit_length()-1)


def independently_expected(case):
    """Fraction equality plus an independently computed exact capacity domain."""
    width=case['width'];external=case['external'];coefficients=case['coefficients'];q=case['q']
    result={'expected_accept':False,'mathematical_equality':None,'domain_reason':None,
            'component_absolute_sum_bits':None}
    if width not in (64,128):
        result['domain_reason']='unsupported-width';return result
    if case['reference_width']!=width:
        result['domain_reason']='reference-width';return result
    if not 1<=len(coefficients)<=64:
        result['domain_reason']='coefficient-count';return result
    if case['shape']:
        result['domain_reason']='malformed-array-shape-'+str(case['shape']);return result
    try:
        ev=[[value(e[mu][0],width) for mu in range(4)] for e in external]
        qv=[value(q[mu][0],width) for mu in range(4)]
        require(all(value(e[mu][1],width)==0 for e in external for mu in range(4)) and
                all(value(q[mu][1],width)==0 for mu in range(4)),'nonreal')
    except ValueError as error:
        result['domain_reason']=str(error);return result
    result['mathematical_equality']=all(
        sum((coefficient*e[mu] for coefficient,e in zip(coefficients,ev)),Fraction(0))==qv[mu]
        for mu in range(4))
    bits=[]
    for mu in range(4):
        terms=[]
        for coefficient,e in zip(coefficients,ev):
            if not coefficient or not e[mu]:
                continue
            significand,exponent=odd_dyadic(e[mu])
            odd_coefficient,extra=odd_dyadic(Fraction(coefficient))
            product=significand*odd_coefficient
            if product>MAXIMUM:
                result['domain_reason']='coefficient-product-capacity';return result
            terms.append((product,exponent+extra))
        if qv[mu]:
            terms.append(odd_dyadic(qv[mu]))
        if terms:
            common=min(exponent for _,exponent in terms)
            if any(exponent-common>126 or significand.bit_length()+exponent-common>127
                   for significand,exponent in terms):
                result['domain_reason']='alignment-capacity';return result
            absolute_sum=sum(significand*2**(exponent-common) for significand,exponent in terms)
            if absolute_sum>MAXIMUM:
                result['domain_reason']='absolute-sum-capacity';return result
            bits.append(absolute_sum.bit_length())
        else:
            bits.append(0)
    result['component_absolute_sum_bits']=bits
    result['expected_accept']=result['mathematical_equality']
    result['domain_reason']='exact-equality' if result['expected_accept'] else 'not-equal'
    return result


def reference_shape(case):
    external=copy.deepcopy(case['external']);q=copy.deepcopy(case['q']);shape=case['shape']
    if shape==1:
        external=[[part[:1] for part in vector] for vector in external]
    elif shape==2:
        external=[vector[:3] for vector in external]
    elif shape==3:
        external=external[:max(0,len(external)-1)]
    elif shape==4:
        q=[part[:1] for part in q]
    elif shape==5:
        q=q[:3]
    return external,q


def build():
    reference=frozen_module()
    cases=[]
    def add(tag,width,external,coefficients,q,reference_width=None,shape=0,provenance=None):
        case={'id':len(cases)+1,'tag':tag,'width':width,
              'reference_width':width if reference_width is None else reference_width,
              'shape':shape,'coefficients':list(coefficients),'external':copy.deepcopy(external),
              'q':copy.deepcopy(q),'provenance':provenance}
        require(0<=len(coefficients)<=65 and len(external)==len(coefficients),'Fixture storage count')
        require(all(type(c) is int and -(1<<31)<=c<(1<<31) for c in coefficients),'Fixture coefficient kind')
        require(all(0<=v<2**128 for e in external for p in e for v in p) and
                all(0<=v<2**128 for p in q for v in p),'Fixture raw storage word')
        case['expected']=independently_expected(case)
        ex,qq=reference_shape(case)
        cross=reference.check(width,case['reference_width'],ex,case['coefficients'],qq)
        require(cross['ok']==case['expected']['expected_accept'],'Independent oracle/spec disagreement: '+tag)
        if cross['ok']:
            require(cross['absolute_sum_bits']==case['expected']['component_absolute_sum_bits'],
                    'Independent capacity/spec disagreement: '+tag)
        cases.append(case)
    rng=random.Random(0xE0A1)
    for width in (64,128):
        for trial in range(250):
            n=rng.randrange(1,9)
            exponents=[rng.randrange(-18,19) for _ in range(4)]
            numbers=[[Fraction(rng.randrange(-512,513))*Fraction(2)**exponents[mu]
                      for mu in range(4)] for _ in range(n)]
            coefficients=[rng.randrange(-8,9) for _ in range(n)]
            qvalues=[sum((coefficients[e]*numbers[e][mu] for e in range(n)),Fraction(0))
                     for mu in range(4)]
            external=vectors(numbers,width);q=vectors([qvalues],width)[0]
            add(f'random-exact-{width}-{trial+1}',width,external,coefficients,q)
            changed=copy.deepcopy(q)
            changed[0][0]=exact_word(qvalues[0]+Fraction(2)**(exponents[0]-1),width)
            add(f'random-perturb-{width}-{trial+1}',width,external,coefficients,changed)
    rng=random.Random(0x12864)
    for width in (64,128):
        nf,ne={64:(52,11),128:(112,15)}[width]
        for trial in range(300):
            def finite():
                return (rng.randrange(2)<<(width-1))|(rng.randrange(2**ne-1)<<nf)|rng.getrandbits(nf)
            external=[[[finite(),0] for _ in range(4)] for _ in range(3)]
            q=[[finite(),0] for _ in range(4)]
            add(f'random-finite-{width}-{trial+1}',width,external,[rng.randrange(-4,5) for _ in range(3)],q)
    witnesses=json.loads((PARENT/'witnesses.json').read_text())
    for w in witnesses['cases']:
        provenance={k:w[k] for k in ('point','mode','route','pair','trace_raw_sha256','trace_gzip_sha256',
                                    'owned_snapshot_observed','q_component_ABI_observed')}
        provenance['q_words']='Conditional nearest-even reconstruction; zeros canonical positive.'
        add(f'conditional-CMS{w["point"]}-mode{w["mode"]}',w['width'],w['inferred_external_P_bits'],
            w['coefficients_relative_to_repaired_P'],w['reconstructed_q_bits'],provenance=provenance)
        add(f'double-counted-ALOHA-sign-CMS{w["point"]}-mode{w["mode"]}',w['width'],
            w['captured_external_W_bits'],w['coefficients_relative_to_repaired_P'],
            w['reconstructed_q_bits'],provenance=provenance)
    for width in (64,128):
        nf,ne={64:(52,11),128:(112,15)}[width]
        zero=vectors([[0]*4],width)[0]
        one=vectors([[1]*4],width)
        negative_zero=1<<(width-1)
        add(f'signed-zero-{width}',width,
            [[[negative_zero,0],[0,negative_zero],[negative_zero,negative_zero],[0,0]]],
            [-(1<<31)],[[0,negative_zero],[negative_zero,0],[0,0],[negative_zero,negative_zero]])
        maximum=((2**ne-2)<<nf)|(2**nf-1)
        for number,label in ((1,'minimum-subnormal'),(2**nf-1,'maximum-subnormal'),
                             (2**nf,'minimum-normal'),(maximum,'maximum-finite'),
                             (maximum|negative_zero,'negative-maximum-finite')):
            external=[[[number,0] for _ in range(4)]]
            add(f'{label}-{width}',width,external,[1],external[0])
        external=vectors([[1,-1,Fraction(1,2),0]],width)
        for coefficient in (-(1<<31),1<<30):
            q=vectors([[coefficient,-coefficient,Fraction(coefficient,2),0]],width)[0]
            add(f'coefficient-{coefficient}-{width}',width,external,[coefficient],q)
        for exponent in (125,126,127,200):
            high=Fraction(2)**exponent
            external=vectors([[high]*4,[high]*4,[1]*4,[1]*4],width)
            add(f'cancel-before-capacity-exponent-{exponent}-{width}',width,external,[1,-1,1,-1],zero)
        external=[[[maximum,0] for _ in range(4)],[[maximum,0] for _ in range(4)],[[1,0] for _ in range(4)]]
        add(f'cancel-full-format-exponent-span-{width}',width,external,[1,-1,1],[[1,0] for _ in range(4)])
        infinity=(2**ne-1)<<nf
        for number,label in ((infinity,'positive-inf'),(infinity|negative_zero,'negative-inf'),
                             (infinity|1,'signaling-nan'),(infinity|(1<<(nf-1)),'quiet-nan')):
            for where in ('external-real','external-imag','q-real','q-imag'):
                external=vectors([[0]*4],width);q=copy.deepcopy(zero)
                if where.startswith('external'):
                    external[0][0][int(where.endswith('imag'))]=number
                else:
                    q[0][int(where.endswith('imag'))]=number
                add(f'{label}-{where}-{width}',width,external,[0],q)
        for where in ('external','q'):
            external=vectors([[0]*4],width);q=copy.deepcopy(zero)
            (external[0] if where=='external' else q)[0][1]=1
            add(f'nonreal-minimum-subnormal-{where}-{width}',width,external,[0],q)
        add(f'wrong-reference-width-{width}',width,one,[1],one[0],reference_width=128 if width==64 else 64)
        for shape in range(1,6):
            add(f'malformed-shape-{shape}-{width}',width,one,[1],one[0],shape=shape)
        add(f'zero-external-count-{width}',width,[],[],zero)
        add(f'unsupported-external-count65-{width}',width,one*65,[0]*65,zero)
        add(f'maximum-external-count64-{width}',width,one*64,[0]*64,zero)
    add('unsupported-width-zero',0,vectors([[0]*4],64),[1],vectors([[0]*4],64)[0])
    add('unsupported-width-80',80,vectors([[0]*4],64),[1],vectors([[0]*4],64)[0])
    # These are valid128-bit transport patterns but invalid zero-extended DP words.
    for bits,label in ((1<<64,'DP-high-bit64'),(2**128-1,'DP-sign-extended-minus-one')):
        external=vectors([[0]*4],64);external[0][0][0]=bits
        add(label,64,external,[0],vectors([[0]*4],64)[0])
    width=128
    one_bits=exact_word(1,width)
    for offset in (1,2):
        external=[[[one_bits+offset,0] for _ in range(4)]]
        add(f'QP-113bit-transport-positive-LSB{offset}',width,external,[1],external[0])
        changed=copy.deepcopy(external[0]);changed[0][0]=one_bits
        add(f'QP-113bit-transport-negative-LSB{offset}',width,external,[1],changed)
    number=Fraction(2**113-1,2**112)
    add('coefficient-product-overflow-before-cancellation',128,vectors([[number]*4,[number]*4],128),
        [(1<<31)-1,-((1<<31)-1)],vectors([[0]*4],128)[0])
    high=Fraction(2**113-1)*2**15
    add('alignment-magnitude-overflow-before-cancellation',128,vectors([[high]*4,[high]*4,[1]*4,[1]*4],128),
        [1,-1,1,-1],vectors([[0]*4],128)[0])
    return {'schema':'private-exact-geometry-native-corpus-v1','parent_source_index_sha256':PARENT_SHA,
            'scope':'Future standalone integer helper only; no ownership, numerator, final-dot or physical qualification.',
            'cases':cases}


def encode_case(case):
    lines=[' '.join(map(str,(case['id'],case['width'],case['reference_width'],len(case['coefficients']),case['shape']))),
           ' '.join(map(str,case['coefficients']))]
    for vector in case['external']:
        lines.extend(f'{real:032X} {imag:032X}' for real,imag in vector)
    lines.extend(f'{real:032X} {imag:032X}' for real,imag in case['q'])
    return '\n'.join(lines)+'\n'


def encode(corpus):
    return str(len(corpus['cases']))+'\n'+''.join(encode_case(case) for case in corpus['cases'])


def expectations(corpus):
    rows={}
    for case in corpus['cases']:
        rows[str(case['id'])]={'id':case['id'],'tag':case['tag'],'width':case['width'],
            'reference_width':case['reference_width'],'n':len(case['coefficients']),'shape':case['shape'],
            'input_case_sha256':hashlib.sha256(encode_case(case).encode()).hexdigest(),
            **case['expected'],'provenance':case['provenance']}
    return {'schema':'private-exact-geometry-native-expectations-v1',
        'source_parent_sha256':PARENT_SHA,'case_count':len(rows),
        'counts':{'expected_accept':sum(row['expected_accept'] for row in rows.values()),
                  'expected_refuse':sum(not row['expected_accept'] for row in rows.values()),
                  'domain_reasons':dict(sorted(Counter(row['domain_reason'] for row in rows.values()).items()))},
        'input_echo_policy':'Reconstruct each canonical encode_case from full driver INPUT/COEFF/X/Q echo and compare its SHA256. '
                            'All input words are retained in fixtures.hex; expected refusal reasons are offline domain metadata, not native return codes.',
        'cases':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    arguments=parser.parse_args()
    corpus=build();text=encode(corpus);expected=expectations(corpus)
    if arguments.write:
        with (HERE/'fixtures.hex').open('x') as output:
            output.write(text)
        with (HERE/'expectations.json').open('x') as output:
            json.dump(expected,output,sort_keys=True,indent=2);output.write('\n')
    else:
        require((HERE/'fixtures.hex').read_text()==text,'Frozen fixture text changed')
        require(json.loads((HERE/'expectations.json').read_text())==expected,'Frozen oracle expectations changed')
    print(json.dumps({'case_count':len(corpus['cases']),'fixture_bytes':len(text.encode()),
                      'expectations_bytes':len(json.dumps(expected,sort_keys=True,indent=2).encode())+1,
                      'counts':expected['counts'],'Fortran_compilations':0,'native_executions':0},sort_keys=True))
