"""Recompute every final1392 replay/convergence control with exact rationals."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
from collections import Counter
import hashlib,json

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
SOURCE=ROOT/'.benchmark-work/policy-v2/final-rust-abi3-v2/original-replay.jsonl'
CAPTURE=REPORT/'replay-convergence-final.jsonl'
FIELD_NAMES=['finite.re','finite.im','single.re','single.im','double.re','double.im','rational.re','rational.im']

def bound(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def show(value):
    with localcontext() as ctx:
        ctx.prec=100
        return str(Decimal(value.numerator)/Decimal(value.denominator))

def recover113(text):
    value=F(text)
    if not value:return (value,text.lstrip().startswith('-'))
    negative=value<0;value=abs(value)
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    if value<F(2)**exponent:exponent-=1
    assert -16382<=exponent<=16383,'capture values are normal binary128 or zero'
    scaled=value*F(2)**(112-exponent)
    q,r=divmod(scaled.numerator,scaled.denominator)
    if 2*r>scaled.denominator or (2*r==scaled.denominator and q%2):q+=1
    rounded=F(q)*F(2)**(exponent-112)
    return (-rounded if negative else rounded,negative)

def values(observation):
    assert observation['status']==0 and observation['input_unchanged'] is True
    assert observation['output_bytes']==512 and len(observation['output'])==8
    raw=observation['raw'];assert len(raw)==10 and all(x['finite'] for x in raw)
    return [F(int(x['numerator']),int(x['denominator'])) for x in raw]

original=[json.loads(line) for line in SOURCE.read_text().splitlines()]
capture=[json.loads(line) for line in CAPTURE.read_text().splitlines()]
assert len(original)==1392 and len(capture)==2785
low,high,declared=capture[:1392],capture[1392:2784],capture[-1]
historical_changed=ascii_changed=convergence_failed=sign_changed=0
historical_components=Counter();changed_shapes=Counter();shapes=Counter()
max_laurent_convergence=max_rational_convergence=F(0)
max_historical_physical=max_historical_rational=F(0)
worst_historical_physical=worst_historical_rational=None
min_precision={256:100000,384:100000}
physical_change_bins=Counter();rational_change_bins=Counter()
with (REPORT/'replay-exact-checks.jsonl').open('x') as stream:
 for sequence,(previous,a,b) in enumerate(zip(original,low,high)):
    assert a['kind']=='replay' and b['kind']=='convergence'
    assert a['sequence']==b['sequence']==sequence and a['input']==b['input']==previous
    assert a['observed']['working_bits']==256 and b['observed_384']['working_bits']==384
    x,y=values(a['observed']),values(b['observed_384'])
    norm=max(map(abs,y[:6]));differences=[abs(v-w) for v,w in zip(x,y)]
    low_defects=[abs(x[i]-x[6+i]-x[8+i]) for i in [0,1]]
    high_defects=[abs(y[i]-y[6+i]-y[8+i]) for i in [0,1]]
    laurent=all(value<=norm/F(10**31) for value in differences[:6])
    rational=all(value<=norm/F(10**31) for value in differences[8:])
    split=all(value<=norm/F(10**60) for value in low_defects+high_defects)
    passed=laurent and rational and split
    assert b['comparison']['passed']==passed
    convergence_failed+=not passed
    if norm:
        max_laurent_convergence=max(max_laurent_convergence,max(differences[:6])/norm)
        max_rational_convergence=max(max_rational_convergence,max(differences[8:])/norm)
    signs=[i for i,(v,w) in enumerate(zip(a['observed']['raw'],b['observed_384']['raw'])) if v['negative']!=w['negative']]
    assert b['comparison']['sign_changed_indices']==signs
    sign_changed+=bool(signs)
    for bits,observation in [(256,a['observed']),(384,b['observed_384'])]:
        min_precision[bits]=min(min_precision[bits],*(v['precision'] for v in observation['raw']))
    old=[recover113(text) for text in previous['output']]
    new=[recover113(text) for text in a['observed']['output']]
    changed=[i for i,(v,w) in enumerate(zip(old,new)) if v!=w]
    ascii_different=a['observed']['output']!=previous['output']
    assert a['historical_binary128_values_and_zero_signs_equal']==(not changed)
    assert a['historical_ascii_equal']==(not ascii_different)
    historical_changed+=bool(changed);ascii_changed+=ascii_different
    shape=f"n{previous['n']}-r{previous['rank']}";shapes[shape]+=1
    if changed:changed_shapes[shape]+=1
    historical_components.update(FIELD_NAMES[i] for i in changed)
    deltas=[abs(v[0]-w[0]) for v,w in zip(old,new)]
    physical_norm=max(abs(v[0]) for v in old[:6]+new[:6])
    physical_scaled=max(deltas[:6])/physical_norm if physical_norm else F(0)
    rational_scaled=max(deltas[6:])/physical_norm if physical_norm else F(0)
    detail={'sequence':sequence,'shape':shape,'field':FIELD_NAMES[deltas.index(max(deltas[:6]))],
        'old':previous['output'],'new':a['observed']['output'],
        'scaled_difference':show(physical_scaled),'absolute_difference':show(max(deltas[:6]))}
    if physical_scaled>max_historical_physical:
        max_historical_physical=physical_scaled;worst_historical_physical=detail
    if rational_scaled>max_historical_rational:
        max_historical_rational=rational_scaled;worst_historical_rational={**detail,
            'field':FIELD_NAMES[6+deltas[6:].index(max(deltas[6:]))],
            'scaled_difference':show(rational_scaled),'absolute_difference':show(max(deltas[6:]))}
    for label,boundary in [('1e-31',F(1,10**31)),('1e-20',F(1,10**20)),('1e-10',F(1,10**10)),('1e-7',F(1,10**7))]:
        physical_change_bins[label]+=physical_scaled>boundary
        rational_change_bins[label]+=rational_scaled>boundary
    check={'sequence':sequence,'shape':shape,'input_history_exact':True,'status_256':0,'status_384':0,
        'laurent_converges_1e_31':laurent,'rational_converges_1e_31':rational,
        'finite_CC_R1_identity_1e_60':split,'sign_changed_indices':signs,
        'historical_binary128_changed_fields':[FIELD_NAMES[i] for i in changed],
        'historical_ascii_changed':ascii_different,
        'historical_maximum_complete_laurent_component_scaled_difference':show(physical_scaled),
        'historical_maximum_basis_rational_component_scaled_difference':show(rational_scaled)}
    stream.write(json.dumps(check,allow_nan=False)+'\n')
summary={'schema':'final-release1392-exact-rational-classification-v1','input':bound(SOURCE),'capture':bound(CAPTURE),
    'controller':bound(Path(__file__)),'exact_checks':bound(REPORT/'replay-exact-checks.jsonl'),
    'records_each_precision':1392,'fields_each_precision':11136,'status_failures':0,
    'failed_convergence_records':convergence_failed,'sign_changed_convergence_records':sign_changed,
    'maximum_laurent_scaled_256_384_difference':show(max_laurent_convergence),
    'maximum_same_basis_rational_scaled_256_384_difference':show(max_rational_convergence),
    'minimum_tracked_precision_observed':min_precision,
    'historical_binary128_changed_records':historical_changed,'historical_ascii_changed_records':ascii_changed,
    'historical_changed_component_counts':dict(historical_components),'historical_changed_topology_rank_counts':dict(changed_shapes),
    'all_topology_rank_counts':dict(shapes),'maximum_historical_complete_laurent_component_scaled_difference':show(max_historical_physical),
    'maximum_historical_basis_rational_component_scaled_difference':show(max_historical_rational),
    'worst_historical_complete_laurent_record':worst_historical_physical,
    'worst_historical_basis_rational_record':worst_historical_rational,
    'historical_complete_laurent_records_above_scaled_difference':dict(physical_change_bins),
    'historical_basis_rational_records_above_scaled_difference':dict(rational_change_bins),
    'historical_normalization':'largest absolute real/imag complete Laurent component across old and new113-bit outputs; basis-dependent rational excluded from the scale',
    'historical_values_are_oracle':False,'historical_bit_identity_passed':historical_changed==0,
    'final_source_convergence_passed':convergence_failed==0 and sign_changed==0,
    'all1392_inputs_in_original_order_without_filtering':True,'performance_claim':False}
assert historical_changed==declared['failed_binary128_replay_records']
assert ascii_changed==declared['historical_ascii_changed_records_retained']
assert convergence_failed==declared['failed_convergence_records']
assert sign_changed==declared['sign_changed_convergence_records_require_review']
(REPORT/'replay-exact-classification.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:summary[k] for k in ['failed_convergence_records','historical_binary128_changed_records',
 'maximum_laurent_scaled_256_384_difference','maximum_historical_complete_laurent_component_scaled_difference',
 'maximum_historical_basis_rational_component_scaled_difference','historical_complete_laurent_records_above_scaled_difference']},indent=2))
