#!/usr/bin/env python3
"""Draft from explicit closed receipts; final mode requires the public packet."""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIVATE=ROOT/'.tirust-build/kinematic-matched-native-input-executor-2/results'
PINS={
    'run-closure.json':'52faa470d8be22b81f8636edba21f5fba6c2d13be1f0241a8212a7b2055839a7',
    'matched-comparison.json':'af4b58a379ac35a78066a1bcad38ed3f4fe9e16e6b0d4583dbd4e6c940eba6d5',
    'root-run-review.json':'9ca6601ab14a58d25eee4d77159805731e6c3d9ecd7264c364e636ed9501bf66',
    'prior-candidate-word-parity.json':'5f9f4cc6c8d2ad35944158cd7ad9fe30c8c0a91729b69456cc9528cf7d07c8e9',
}
POLICY_PINS={'scientific_index':'357e116b51879cda1e1145c9d040194732c21ffd9fcd2557d9f68451901b262b',
             'policy_note':'414a65385fb3e69c45c91a63ed04eb73d5a9012ce041aca00f21fbdbe539f943'}


def require(value,message):
    if not value:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def encoded(value):return json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'


def safe(name):
    p=PurePosixPath(name)
    require(name and not p.is_absolute() and '..' not in p.parts and str(p)==name and '\\' not in name,'Unsafe input path')
    return name


def parsed(raw):
    def unique(pairs):
        out={}
        for key,value in pairs:
            require(key not in out,'Duplicate JSON key');out[key]=value
        return out
    def bad(value):raise ValueError('Nonfinite token: '+value)
    return json.loads(raw,object_pairs_hook=unique,parse_constant=bad)


class Public:
    def __init__(self,root,expected):
        self.root=Path(root)
        require(not any(p.is_symlink() for p in (self.root,*self.root.parents)),'Symlink public root')
        raw=(self.root/'evidence-index.json').read_bytes()
        require(sha(raw)==expected,'Public index changed')
        self.index,self.payloads=parsed(raw),{}
        paths=list(self.root.rglob('*'))
        require(not any(p.is_symlink() for p in paths),'Symlink public member')
        require({p.relative_to(self.root).as_posix() for p in paths if p.is_file()}==set(self.index['files'])|{'evidence-index.json'},'Public inventory differs')
        for name,identity in self.index['files'].items():
            raw=(self.root/safe(name)).read_bytes()
            require({'bytes':len(raw),'sha256':sha(raw)}==identity,'Public payload changed')
            self.payloads[name]=raw
        self.expected=expected

    def read(self,name):
        name=safe(name);item=self.index.get('compressed_text',{}).get(name)
        raw=gzip.decompress(self.payloads[item['path']]) if item else self.payloads[name]
        if item:require(len(raw)==item['raw_bytes'] and sha(raw)==item['raw_sha256'],'Expanded public payload differs')
        return raw


def exact64(word):
    require(isinstance(word,str) and len(word)==16,'Expected binary64 output word')
    bits=int(word,16);exponent=(bits>>52)&2047;fraction=bits&((1<<52)-1)
    require(exponent!=2047,'Ineligible nonfinite output word')
    significand=fraction if exponent==0 else (1<<52)+fraction
    power=-1074 if exponent==0 else exponent-1023-52
    return (-1 if bits>>63 else 1)*Fraction(significand)*(Fraction(2)**power)


def rational(value):return Fraction(int(value['numerator']),int(value['denominator']))


def derive(receipts,public_verified):
    c,m,r,p=(parsed(receipts[n]) for n in PINS)
    require(c['passed'] is True and c['failure'] is None and c['performance_measurement'] is False,'Run did not pass')
    require(r['passed'] and r['run_closure_sha256']==PINS['run-closure.json']
            and r['comparison_sha256']==PINS['matched-comparison.json'],'Root review has another run')
    require((r['native_calls'],r['native_input_pairs'],r['outer_calls'],r['runtime_maps'])==(104,52,4,4),'Coverage differs')
    require(r['all_native_success'] and r['exact_geometry_and_canonical_numerators']
            and r['input_finiteness_and_caller_environment_gate'] and r['both_bootstrap_controls_passed']
            and r['fresh_raw_comparator_replay_identical'],'Root gates incomplete')
    require(m['passed'] and m['execution_complete'] and m['native_input_gate_passed']
            and m['native_inputs_finite'] and m['outer_output_gate_passed']
            and m['bootstrap_controls_passed'] and m['measurement_interpretation_allowed'],'Whole comparison gates failed')
    native=m['input_match']
    require(native['exact_input_match'] and native['certified_geometry_match'] and native['caller_binding_verified']
            and native['qualified_input_match'] and len(native['rows'])==52,'Native pair gates failed')
    calls={item['backend']:item['calls'] for item in native['parsed']}
    require(set(calls)=={8,9} and all(len(v)==52 for v in calls.values()),'Missing native calls')
    require(all({(c['phase'],c['group']) for c in rows}=={(phase,group) for phase in (0,1) for group in range(1,27)}
                for rows in calls.values()),'Native phase/group coverage differs')
    for a,b in zip(calls[8],calls[9]):
        require(a['key']==b['key'] and a['meta']==b['meta'] and a['environment']==b['environment'],'Native metadata/environment differs')
        for item in (a,b):
            require((item['mode'],item['width'],item['digits'],item['complete'],item['status'],item['unchanged'])==(4,128,113,1,0,1),'Native call is ineligible')
            for key in ('SHIFT','INV','MASS','SCALE','CANON'):
                require(a['words'][key]==b['words'][key],'Actual native input/canonical numerator words differ')
                require(all(len(w)==32 and (int(w,16)>>112)&32767!=32767 for w in item['words'][key]),'Nonfinite native input')
        if a['phase']==1 and a['group']==3:
            require(a['n']==4 and a['meta']['USE'][7]==a['meta']['USE'][13]==1,'Required certified direct USE absent')
    for backend in ('8','9'):
        o=m['observations'][backend]
        require(o['eligible'] and o['protocol_complete'] and o['measurement_interpretation_allowed']
                and not o['failures'] and not o['incomplete_contribution_warnings'],'Whole worker is ineligible')
        after=[s for s in o['states'] if s['stage']=='A']
        require(len(after)==2 and all((s['actual_mode'],s['code'],s['fpe_dp'],s['fpe_qp'])==(4,120,0,0) for s in after),'Invalid post-call mode/software flags')
    require(len(m['bootstrap_comparisons'])==2 and all(x['eligible'] and x['passed'] and x['ordinary_target_passed']
            and not x['interpretation_blocked'] for x in m['bootstrap_comparisons']),'Bootstrap comparison gates failed')
    measured=m['measurement_comparison'];require(measured['eligible'] and measured['passed'] and measured['ordinary_target_passed'] and not measured['interpretation_blocked'],'Measurement cannot be interpreted')
    require(m['source_evidence']['scientific']['scientific_index_sha256']==POLICY_PINS['scientific_index'],
            'Private input-policy source differs')
    rows=[]
    require(len(measured['residuals'])==8,'Expected four coefficients in two answer orders')
    for name in ('Born','finite','single_pole','double_pole'):
        pair=[x for x in measured['residuals'] if x['coefficient']==name]
        require(len(pair)==2 and {x['order'] for x in pair}=={0,1},'Wrong answer-order population')
        zero,one=sorted(pair,key=lambda x:x['order'])
        require({k:v for k,v in zero.items() if k!='order'}=={k:v for k,v in one.items() if k!='order'},'Duplicate answer orders differ')
        for item in pair:
            for backend,field in (('8','reference_word'),('9','candidate_word')):
                raw=next(x for x in m['observations'][backend]['raw_rows'] if x['phase']==1 and x['point']==20 and x['order']==item['order'])
                require(raw['words'][item['column']]==item[field],'Comparison word differs from the eligible outer row')
        a,b=exact64(zero['reference_word']),exact64(zero['candidate_word'])
        require(a==rational(zero['reference']) and b==rational(zero['candidate']),'Words differ from retained rational coefficients')
        relative=abs(a-b)/max(abs(a),abs(b)) if a or b else Fraction(0)
        symmetric=abs(a-b)/abs(a+b) if a+b else Fraction(0)
        require(relative==rational(zero['relative_error']) and symmetric==rational(zero['symmetric_relative_error']),'Residual arithmetic differs')
        require(zero['passed'] and zero['ordinary_target_passed'],'Coefficient gate failed')
        rows.append({'coefficient':name,'backend8':zero['reference']['decimal'],'backend9':zero['candidate']['decimal'],
                     'backend8_word':zero['reference_word'],'backend9_word':zero['candidate_word'],
                     'binary64_word_ULP_gap':abs(int(zero['reference_word'],16)-int(zero['candidate_word'],16)),
                     'relative_error':zero['relative_error']['decimal'],'symmetric_relative_error':zero['symmetric_relative_error']['decimal']})
    require([x['binary64_word_ULP_gap'] for x in rows]==[0,1,1,0],'Expected word differences changed')
    require(p['passed'] and p['bootstrap_words_bit_identical'] and p['measurement_words_bit_identical']
            and p['ancillary_accuracy_words_bit_identical'] and len(p['word_rows'])==16
            and all(x['equal'] and x['current_word']==x['prior_word'] for x in p['word_rows']), 'Prior candidate word parity failed')
    require(p['failed_reference_coefficients_decoded']==0 and p['eligibility']['prior_stress_aggregate_passed'] is False,'Prior failure was relabelled')
    binding=p['source_binding_proof']
    require(all(binding[key] for key in ('all_non_executable_runtime_provider_pins_identical',
            'private_MP_invariant_helper_byte_identical',
            'matched9_executable_body_equals_prior_after_removing_exactly_one_begin_and_end_input_recorder_call',
            'postdiagnostic_failure_contraction_and_normalization_tail_byte_identical',
            'selected_build_source_bindings_checked'))
            and binding['same_TIRust_runtime_provider']['sha256']=='eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9',
            'Prior/current candidate source or runtime binding differs')
    require((p['bootstrap_coefficient_words_compared'],p['measurement_coefficient_words_compared'],p['ancillary_accuracy_words_compared'])==(8,8,4),
            'Prior word-parity coverage differs')
    columns={'Born':0,'finite':1,'single_pole':2,'double_pole':3}
    for item in p['word_rows']:
        raw=next(x for x in m['observations']['9']['raw_rows'] if (x['phase'],x['point'],x['order'])==(item['phase'],item['point'],item['order']))
        require(raw['words'][columns[item['coefficient']]]==item['current_word'],'Prior/current parity has another outer word')
    for item in p['ancillary_accuracy_words']:
        raw=next(x for x in m['observations']['9']['raw_rows'] if (x['phase'],x['point'],x['order'])==(item['phase'],item['point'],item['order']))
        require(item['equal'] and item['current_word']==item['prior_word']==raw['words'][4],'Ancillary output-word parity differs')
    require(all(not x for x in (m['accuracy_qualified'],m['normal_host_stability_qualified'],m['performance_measurement'],m['winner_selected'],r['independent_accuracy_qualified'],r['performance_qualified'])),'Unqualified scope changed')
    return {'schema':'madloop-later-matched-input-report-data-v1','public_packet_verified':public_verified,
            'ready_to_seal':public_verified,'closed_receipt_pins':PINS,'native_calls':104,'native_pairs':52,'outer_calls':4,
            'input_policy_source_pins':POLICY_PINS,
            'point':20,'bootstrap_point':1,'matched_native_inputs':True,'whole_worker_gates_passed':True,
            'bootstrap_controls_passed':True,'fixed_QP_return':120,'table':rows,
            'maximum_relative_error':measured['maximum_relative_error']['decimal'],
            'prior_candidate9_word_parity':True,'prior_stress_failure_preserved':True,
            'original_legacy8_unchanged':True,'private8_policy_change':True,
            'independent_accuracy_qualified':False,'normal_checked_execution_qualified':False,
            'generic_source_qualified':False,'performance_qualified':False,'algorithmic_winner_selected':False,
            'new_native_physics_evaluations':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--public-packet',type=Path)
    parser.add_argument('--public-index-sha256')
    parser.add_argument('--public-map',type=Path)
    parser.add_argument('--output',type=Path,default=HERE/'matched-data.json')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.preview:
        require(not args.public_packet,'Preview and public modes are separate')
        receipts={n:(PRIVATE/n).read_bytes() for n in PINS}
        scientific=ROOT/'.tirust-build/kinematic-matched-native-input-proposal-1'
        require(sha((scientific/'index.json').read_bytes())==POLICY_PINS['scientific_index']
                and sha((scientific/'PROPOSAL.md').read_bytes())==POLICY_PINS['policy_note'],'Closed input-policy source changed')
    else:
        require(args.public_packet and args.public_index_sha256 and args.public_map,'Final mode requires explicit public packet/index/map')
        public=Public(args.public_packet,args.public_index_sha256)
        mapping=parsed(args.public_map.read_bytes())
        require(set(mapping)==set(PINS)|set(POLICY_PINS),'Public receipt/source mapping differs')
        receipts={n:public.read(mapping[n]) for n in PINS}
        for name,expected in POLICY_PINS.items():require(sha(public.read(mapping[name]))==expected,'Public policy source differs')
    for n,raw in receipts.items():require(sha(raw)==PINS[n],'Closed receipt changed: '+n)
    # Private preview authenticates each explicit closed payload; final public
    # mode authenticates the entire immutable public packet instead.
    if args.preview:
        for n,item in parsed(receipts['run-closure.json'])['payloads'].items():
            raw=(PRIVATE/safe(n)).read_bytes();require({'bytes':len(raw),'sha256':sha(raw)}==item,'Closed run payload changed')
    else:
        prefix=PurePosixPath(mapping['run-closure.json']).parent.as_posix()
        for n,item in parsed(receipts['run-closure.json'])['payloads'].items():
            raw=public.read(prefix+'/'+safe(n));require({'bytes':len(raw),'sha256':sha(raw)}==item,'Public closed-run payload differs')
    value=derive(receipts,not args.preview)
    if not args.preview:
        value.update(public_packet_index_sha256=args.public_index_sha256,
                     public_packet_payloads_verified=len(public.payloads),public_receipt_map=mapping)
    data=encoded(value)
    if args.verify_only:require(args.output.read_text()==data,'Report data changed')
    else:
        with args.output.open('x') as stream:stream.write(data)
    print(encoded({'passed':True,'preview':args.preview,'ready_to_seal':not args.preview,
                   'data_sha256':sha(data.encode()),'new_native_physics_evaluations':0}),end='')


if __name__=='__main__':main()
