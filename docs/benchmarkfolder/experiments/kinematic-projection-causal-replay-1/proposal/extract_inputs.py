#!/usr/bin/env python3
"""Pure exact-word extraction for ten bounded CMS group3 replay inputs."""
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
VERIFY_ONLY=False


def require(ok,message):
    if not ok:raise ValueError(message)


def pin(path):
    raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def checked(path,expected):
    raw=Path(path).read_bytes()
    require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==
            {k:expected[k] for k in ('bytes','sha256')},'Bound bytes differ: '+str(path))
    return raw


def seal(path,expected):
    raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'Index differs: '+str(path))
    index=json.loads(raw)
    for name,expected in index['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe index path')
        checked(Path(path).parent/p,expected)
    return index


def replace_pair(words,raw_words,mode):
    """N=4, Fortran column-major S; replace only (2,4) and its transpose."""
    width={1:1,4:2}[mode]
    require(len(words)==41*width and len(raw_words)==width,'Native word shape differs')
    offsets=[width*(24+(4-1)*4+(2-1)),width*(24+(2-1)*4+(4-1))]
    out=list(words)
    for start in offsets:out[start:start+width]=raw_words
    allowed={i for start in offsets for i in range(start,start+width)}
    changed=[i for i,(a,b) in enumerate(zip(words,out)) if a!=b]
    require(set(changed)<=allowed,'Intervention changed another input')
    require(all(words[start:start+width]!=out[start:start+width] for start in offsets),
            'Both selected scalar entries must change')
    return out,{'matrix_pair':[2,4],'transpose_pair':[4,2],
                'allowed_word_positions_one_based':sorted(i+1 for i in allowed),
                'changed_word_positions_one_based':[i+1 for i in changed]}


def negative_bounds(value):
    require(value<0,'Captured RAW must be strictly negative')
    magnitude=-value
    exponent=magnitude.numerator.bit_length()-magnitude.denominator.bit_length()
    power=lambda e:Fraction(2**e,1) if e>=0 else Fraction(1,2**(-e))
    if magnitude<power(exponent):exponent-=1
    lower,upper=-power(exponent+1),-power(exponent)
    require(lower<value<=upper<0,'Exact negative enclosure failed')
    return {'strict_lower':lower,'inclusive_upper':upper,'base2_exponent':exponent}


def write_input(name,point,mode,words):
    p=HERE/'inputs'/name
    if not VERIFY_ONLY:p.parent.mkdir(exist_ok=True)
    raw='KPROJ_WORDS_V1\n'+f'{point} {mode} 4 2 {len(words)}\n'+'\n'.join(words)+'\n'
    if VERIFY_ONLY:
        require(p.read_text()==raw,'Saved replay input differs from frozen trace: '+name)
    else:
        with p.open('x') as stream:stream.write(raw)
    return {'path':str(p.relative_to(ROOT)),**pin(p)}


def main():
    global VERIFY_ONLY
    require(sys.argv[1:] in ([],['--verify-only']),'Only --verify-only is supported')
    VERIFY_ONLY=sys.argv[1:]==['--verify-only']
    require(Path.cwd()==ROOT,'Run from repository root')
    bindings=json.loads((HERE/'bindings.json').read_bytes())
    for name,expected in bindings['indices'].items():seal(ROOT/name,expected)
    for name,expected in bindings['files'].items():checked(ROOT/name,expected)
    for name,expected in bindings['toolchain'].items():checked(name,expected)
    for name,expected in bindings['fixed_supplemental_providers'].items():checked(name,expected)
    closures={k:json.loads(checked(ROOT/v['path'],v)) for k,v in bindings['closures'].items()}
    decoder_path=ROOT/'.tirust-build/kinematic-provenance-analysis-1/decode_trace.py'
    spec=importlib.util.spec_from_file_location('projection_bound_decoder',decoder_path)
    decoder=importlib.util.module_from_spec(spec);sys.modules[spec.name]=decoder;spec.loader.exec_module(decoder)
    derivation,route,sourcepins=decoder.source_binding('cms')
    identity=next(x for x in derivation['target_identities'] if x['reduction_identifier']==3)
    require(identity['rank']==2 and identity['identity_kind']=='group','Wrong source identity')
    certificates=[];requests=[]
    for label,bound in bindings['traces'].items():
        require(closures[bound['closure']]['workers'][label]['trace_pin']==
                {k:bound[k] for k in ('bytes','sha256')},'Trace not bound to closed worker')
        raw=checked(ROOT/bound['path'],bound)
        _,records=decoder.parse(raw,'low-first')
        rows=[r for r in records if r['id']==3]
        keys={(r['section'],r['point'],r['serial'],r['id'],r['mode'],r['rank'],r['n'],r['split_order']) for r in rows}
        require(len(keys)==1 and rows and all(r['point']==bound['point'] and r['mode']==bound['mode']
                and r['rank']==2 and r['n']==4 for r in rows),'Group3 call is not unique/bound')
        suffix='DP' if bound['mode']==1 else 'QP'
        native=decoder.one(rows,'KT_TIRUST_'+suffix)
        rawpair=decoder.one(rows,'KT_RAW_'+suffix,2,4)
        matrix=decoder.one(rows,'KT_MATRIX_'+suffix)
        status=decoder.one(rows,'KT_STATUS_'+suffix)
        require(all(x is not None for x in (native,rawpair,matrix,status)),'Missing input/status boundary')
        data=native['data'];values=decoder.values
        require(all(x['value'] is not None for group in data.values() for x in group),'Nonfinite native input')
        real,imag=rawpair['data']['diffsq'];require(imag['value']==0,'Complex RAW is outside single-real replacement')
        bounds=negative_bounds(real['value'])
        native_shifts=decoder.vectors(data['shifts_corner_major'],4)
        matrix_shifts=decoder.vectors(matrix['data']['shifts_component_major'],4,True)
        require(native_shifts==matrix_shifts,'Native/matrix shifts differ')
        for corner in range(4):
            for component in range(4):
                require(data['shifts_corner_major'][4*corner+component]['words']==
                        matrix['data']['shifts_component_major'][4*component+corner]['words'],
                        'Shift transpose changes captured bits')
        exact_square=decoder.squared(decoder.subtract(native_shifts[1],native_shifts[3]))
        require(exact_square<0,'Captured shift-pair exact square is not spacelike')
        s=data['invariants_column_major'];require(s[13]['words']==s[7]['words'],'Native critical pair is asymmetric')
        require(identity['modes'][str(bound['mode'])]['shifts'][1]==[1,0,0,0] and
                identity['modes'][str(bound['mode'])]['shifts'][3]==[1,1,0,-1],
                'Frozen signed source mapping differs')
        words=[word for field in ('shifts_corner_major','squared_masses','invariants_column_major','scale')
               for value in data[field] for word in value['words']]
        require(len(words)==(41 if bound['mode']==1 else 82),'Captured native shape differs')
        expected_status=0 if bound['point']==1 else 1
        require(status['data']['native_status']==expected_status,'Original status differs from closed analysis')
        stock=label.replace('-enabled','-group3-stock')+'.words'
        requests.append({'id':Path(stock).stem,'variant':'stock','source_trace':label,
                         'point':bound['point'],'mode':bound['mode'],'input':write_input(stock,bound['point'],bound['mode'],words)})
        certificate={'trace':bound,'identity':dict(zip(('section','point','serial','id','mode','rank','n','split_order'),next(iter(keys)))),
            'source_lines':{'native':native['line'],'raw_2_4':rawpair['line'],'matrix':matrix['line'],'status':status['line']},
            'original_native_status':expected_status,'raw_real':real,'raw_imaginary':imag,'negative_raw_bounds':bounds,
            'critical_native_words':s[13]['words'],'exact_captured_shift_pair_square':exact_square,
            'negative_shift_square_bounds':negative_bounds(exact_square),'raw_minus_exact_shift_square':real['value']-exact_square,
            'symbolic_pair':'p1-p3-r','first_shift_preserved_words':[x['words'] for x in data['shifts_corner_major'][:4]],
            'source_conservation_coefficients':identity['conservation_coefficients'],
            'stock_word_sha256':hashlib.sha256(' '.join(words).encode()).hexdigest(),'intervention':None}
        if bound['point']!=1:
            require(s[13]['value']==0 and s[7]['value']==0,'Stress original native critical entries are not zero')
            changed,proof=replace_pair(words,real['words'],bound['mode'])
            name=label.replace('-enabled','-group3-raw24')+'.words'
            requests.append({'id':Path(name).stem,'variant':'raw24','source_trace':label,
                             'point':bound['point'],'mode':bound['mode'],'input':write_input(name,bound['point'],bound['mode'],changed)})
            certificate['intervention']=proof
        certificates.append(certificate)
    require(len(requests)==10,'Expected exactly ten independent replay requests')
    result={'schema':'private-exact-kinematic-projection-replay-certificate-v1','source_only':True,
            'native_executions':0,'failed_tensor_outputs_read':False,'input_trace_only':True,
            'source_pins':sourcepins,'source_identity':identity,'requests':requests,'certificates':certificates,
            'intervention_is_diagnostic_only':True,'production_fixture_or_policy_change':False,
            'success_status_is_not_accuracy_proof':True}
    serialized=json.dumps(decoder.wire(result),sort_keys=True,indent=2)+'\n'
    if VERIFY_ONLY:
        require((HERE/'input-certificate.json').read_text()==serialized,'Saved certificate differs')
    else:
        with (HERE/'input-certificate.json').open('x') as out:out.write(serialized)
    print(json.dumps({'requests':len(requests),'traces':len(certificates),'native_executions':0,'verification_only':VERIFY_ONLY,'certificate':pin(HERE/'input-certificate.json')},indent=2))


if __name__=='__main__':main()
