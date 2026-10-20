#!/usr/bin/env python3
"""Offline exact-dyadic inventory; no reducer calls or numerical executables."""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import parse_capture_v2 as capture

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE/name).read_text())


def scalar(words):
    width=64*len(words)
    assert width in (64,128)
    value=sum(int(word,16)<<(64*i) for i,word in enumerate(words))
    fraction_bits,bias,exponent_bits=(52,1023,11) if width==64 else (112,16383,15)
    negative=bool(value>>(width-1))
    exponent=(value>>fraction_bits)&((1<<exponent_bits)-1)
    assert exponent!=(1<<exponent_bits)-1
    significand=value&((1<<fraction_bits)-1)
    power=(exponent-bias-fraction_bits) if exponent else (1-bias-fraction_bits)
    if exponent:significand+=1<<fraction_bits
    number=Fraction(significand<<power,1) if power>=0 else Fraction(significand,1<<(-power))
    return (-number if negative else number),negative


def rounded_word(number,negative_zero):
    value=float(number) if number else (-0.0 if negative_zero else 0.0)
    return struct.pack('>d',value).hex().upper()


def new_counter():
    return {'real_components':0,'different_exact_values':0,'different_promoted_bits':0,
            'different_after_rounding_qp_to_dp':0,'sign_only_zero_changes':0,
            'maximum_absolute_difference':Fraction(0),'maximum_scaled_difference':Fraction(0)}


def count(counter,dp_words,qp_words,context):
    assert len(qp_words)==2*len(dp_words)
    changes=0
    for index,dp_word in enumerate(dp_words):
        a,az=scalar([dp_word]);b,bz=scalar(qp_words[2*index:2*index+2])
        delta=abs(a-b);scale=max(abs(a),abs(b));relative=delta/scale if scale else Fraction(0)
        changed=rounded_word(b,bz)!=dp_word
        counter['real_components']+=1
        counter['different_exact_values']+=a!=b
        counter['different_promoted_bits']+=(a!=b or (a==0 and az!=bz))
        counter['different_after_rounding_qp_to_dp']+=changed
        counter['sign_only_zero_changes']+=(a==b==0 and az!=bz)
        if delta>counter['maximum_absolute_difference']:
            counter['maximum_absolute_difference']=delta
            counter['worst_absolute_location']={**context,'component':index,'dp':dp_word,'qp':qp_words[2*index:2*index+2]}
        if relative>counter['maximum_scaled_difference']:
            counter['maximum_scaled_difference']=relative
            counter['worst_scaled_location']={**context,'component':index,'dp':dp_word,'qp':qp_words[2*index:2*index+2]}
        changes+=changed
    return changes


def finish(counter):
    result=dict(counter)
    for key in ('maximum_absolute_difference','maximum_scaled_difference'):
        value=result.pop(key)
        result[key]={'exact_numerator':str(value.numerator),'exact_denominator':str(value.denominator),'float_diagnostic':float(value)}
    return result


def compute():
    binding=read('bindings.json');prepared=read('prepared.json');result=read('attempt-2/capture-result.json')
    assert result['complete'] and result['all_100_rows_statuses_accuracies_exact']
    assert result['prepared_sha256']==sha(HERE/'prepared.json')
    assert result['bindings_sha256']==sha(HERE/'bindings.json')
    assert result['runner_sha256']==sha(HERE/'run_capture_v2.py')
    assert result['parser_sha256']==sha(HERE/'parse_capture_v2.py')
    for name,record in result['raw'].items():
        assert sha(HERE/'attempt-2/raw'/name)==record['sha256']
    observed=[]
    for line in (HERE/'attempt-2/raw/values.dat').read_text().splitlines():
        v=line.split();assert len(v)==8
        observed.append({'point':int(v[0]),'order':int(v[1]),'status':int(v[2]),
                         'coefficients':list(map(float,v[3:7])),'accuracy':float(v[7])})
    assert observed==binding['reference_rows']
    assert [(r['point'],r['order']) for r in observed]==[(p,o) for p in range(1,101) for o in (0,1)]
    raw=HERE/'attempt-2/raw/tensor-capture.hex'
    records=capture.parse(raw);matching=capture.match(records)
    assert matching==read('attempt-2/matching.json')
    indexed={r['sequence']:r for r in records}
    totals={tag:new_counter() for tag in ('C','S','P','M','U')}
    per_key=[];dp_output_changes=Counter()
    for pair in matching['records']:
        dp=indexed[pair['native_sequence']];qp=indexed[pair['qp_sequence']];comp=indexed[pair['compensated_sequence']]
        changed={tag:0 for tag in totals}
        for field,words in dp['data'].items():
            tag=field.split(':')[0]
            if tag in totals:
                changed[tag]+=count(totals[tag],words,qp['data'][field],{'key':pair['key'],'field':field})
            elif tag in ('R','R1'):
                dp_output_changes[tag]+=sum(a!=b for a,b in zip(words,comp['data'][field]))
        per_key.append({'key':pair['key'],'changed_rounded_components':changed})
    # The loader paths are real files from the same frozen reference library set.
    loaded={}
    for path,digest in result['dynamic_libraries'].items():
        if Path(path).name.startswith(('libninja','libcollier')):
            names=[name for name,value in binding['runtime']['libraries'].items() if value==digest]
            assert names,('unbound loaded reducer',path)
            loaded[path]={'sha256':digest,'bound_original_names':names}
    assert any('libninja' in Path(path).name for path in loaded)
    assert any('libcollier' in Path(path).name for path in loaded)
    return {'schema':'captured-dp-qp-input-inventory-v1','complete':True,'engine_calls':0,
            'raw_sha256':sha(raw),'bindings_sha256':sha(HERE/'bindings.json'),
            'capture_result_sha256':sha(HERE/'attempt-2/capture-result.json'),
            'parser_sha256':sha(HERE/'parse_capture_v2.py'),
            'inventory_source_sha256':sha(Path(__file__)),
            'all_100_original_rows_statuses_accuracies_exact':True,
            'status_counts':dict(Counter(str(r['status']) for r in observed if r['order']==0)),
            'record_count':len(records),'matched_records':len(per_key),
            'stage_status_counts':matching['status_counts'],
            'native_compensated_inputs_bit_identical':True,
            'native_compensated_output_word_changes':dict(dp_output_changes),
            'loaded_reducers_match_original':loaded,
            'by_field':{tag:finish(counter) for tag,counter in totals.items()},
            'per_key':per_key,
            'limitations':['QP coefficients and geometry come from the normal improved MP point; differences do not isolate only numerator rounding.',
                           'All reduction status0 values are native success flags; the DP point-level direction gate still failed at44/80.',
                           'No replay, physical mixed-result acceptance or runtime improvement is established.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    value=compute();target=HERE/'input-inventory.json'
    if args.write:
        assert not target.exists();target.write_text(json.dumps(value,indent=2)+'\n')
    else:
        assert json.loads(target.read_text())==value
    print(json.dumps({k:v for k,v in value.items() if k not in ('per_key','loaded_reducers_match_original')},indent=2))
