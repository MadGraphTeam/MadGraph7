#!/usr/bin/env python3
"""Offline rounded-QP input-bundle changes measured in binary64 steps."""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import inventory
import parse_capture_v2 as capture

HERE=Path(__file__).resolve().parent
MASK=(1<<63)-1


def quantiles(values):
    values=sorted(values)
    if not values:return {'count':0}
    result={'count':len(values),'minimum':values[0],'maximum':values[-1]}
    for label,numerator,denominator in [('p50',1,2),('p90',9,10),('p95',19,20),('p99',99,100)]:
        result[label]=values[max(0,(numerator*len(values)+denominator-1)//denominator-1)]
    result['buckets']=dict(Counter('0' if v==0 else '1' if v==1 else '2-4' if v<=4 else
                           '5-16' if v<=16 else '17-256' if v<=256 else '257-4096' if v<=4096 else '4097+' for v in values))
    return result


def compute():
    assert inventory.compute()==inventory.read('input-inventory.json')
    records=capture.parse(HERE/'attempt-2/raw/tensor-capture.hex')
    matching=capture.match(records);indexed={r['sequence']:r for r in records}
    classes=('same_sign_nonzero','zero_to_nonzero','nonzero_to_zero','opposite_sign_nonzero','both_zero_same_sign','both_zero_changed_sign')
    data={tag:{'classes':{name:[] for name in classes},'max_absolute':Fraction(0),'max_relative':Fraction(0)} for tag in ('C','S','P','M','U')}
    for pair in matching['records']:
        dp=indexed[pair['native_sequence']];qp=indexed[pair['qp_sequence']]
        for field,words in dp['data'].items():
            tag=field.split(':')[0]
            if tag not in data:continue
            for component,word in enumerate(words):
                original,negative_original=inventory.scalar([word])
                high,negative_high=inventory.scalar(qp['data'][field][2*component:2*component+2])
                rounded_word=inventory.rounded_word(high,negative_high)
                rounded,negative_rounded=inventory.scalar([rounded_word])
                a=int(word,16);b=int(rounded_word,16)
                ordinal_a=-(a&MASK) if a>>63 else a&MASK
                ordinal_b=-(b&MASK) if b>>63 else b&MASK
                distance=abs(ordinal_b-ordinal_a)
                if not original and not rounded:
                    category='both_zero_changed_sign' if negative_original!=negative_rounded else 'both_zero_same_sign'
                elif not original:category='zero_to_nonzero'
                elif not rounded:category='nonzero_to_zero'
                elif negative_original!=negative_rounded:category='opposite_sign_nonzero'
                else:category='same_sign_nonzero'
                data[tag]['classes'][category].append(distance)
                absolute=abs(rounded-original);scale=max(abs(original),abs(rounded))
                relative=absolute/scale if scale else Fraction(0)
                for name,value in [('max_absolute',absolute),('max_relative',relative)]:
                    if value>data[tag][name]:
                        data[tag][name]=value
                        data[tag][name+'_location']={'key':pair['key'],'field':field,'component':component,
                            'dp_word':word,'rounded_qp_word':rounded_word,'distance':distance,'class':category}
    out={}
    for tag,item in data.items():
        result={'real_components':sum(map(len,item['classes'].values())),
                'classes':{name:quantiles(values) for name,values in item['classes'].items()},
                'same_sign_nonzero_changed':quantiles([x for x in item['classes']['same_sign_nonzero'] if x])}
        for name in ('max_absolute','max_relative'):
            value=item[name]
            result[name]={'numerator':str(value.numerator),'denominator':str(value.denominator),'float_diagnostic':float(value),
                          'location':item.get(name+'_location')}
        assert result['real_components']==inventory.read('input-inventory.json')['by_field'][tag]['real_components']
        out[tag]=result
    return {'schema':'rounded-qp-input-bundle-ulp-inventory-v1','complete':True,'engine_calls':0,
            'source_sha256':inventory.sha(Path(__file__)),'input_inventory_sha256':inventory.sha(HERE/'input-inventory.json'),
            'definition':'ULP distance is the count of adjacent binary64 representable-value steps, with signed zeros sharing ordinal0. Zero and opposite-sign transitions are reported separately; quantiles use nearest-rank on integer distances.',
            'comparison':'Round captured normal-QP input components once to binary64, then compare their exact values and words with originalDP. This is a changed QP input bundle, not isolated numerator arithmetic.',
            'magnitude_scope':'Absolute component values use their stored units (coefficient units vary with rank); neither component differences nor relative maxima are amplitude error bounds.',
            'by_field':out}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    value=compute();path=HERE/'ulp-inventory.json'
    if args.write:
        assert not path.exists();path.write_text(json.dumps(value,indent=2)+'\n')
    else:assert json.loads(path.read_text())==value
    print(json.dumps(value,indent=2))
