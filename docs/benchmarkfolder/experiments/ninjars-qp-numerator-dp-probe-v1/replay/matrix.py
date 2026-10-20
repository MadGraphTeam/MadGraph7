"""Integer-only replay construction and output validation; no engine calls."""
import importlib.util
import math
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent
CAPTURE=HERE.parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

parser=module('sealed_capture_parser',CAPTURE/'parse_capture_v2.py')
integer=module('sealed_independent_integer_decoder',CAPTURE/'independent-inventory/check.py')

def cases():
    records=parser.parse(CAPTURE/'attempt-2/raw/tensor-capture.hex')
    matched=parser.match(records)
    by_sequence={r['sequence']:r for r in records}
    return [(entry['key'],*[by_sequence[entry[k]] for k in ('native_sequence','compensated_sequence','qp_sequence')])
            for entry in matched['records']]

def input_fields(n,rank):
    keys=[f'C:{i}' for i in range(math.comb(rank+4,4))]
    for j in range(1,n+1):
        keys.extend(f'S:{i},{j}' for i in range(1,n+1))
        keys.extend(f'P:{i},{j}' for i in range(4))
        keys.append(f'M:{j}')
    return keys+['U:']

def promote(word):
    raw=int(word,16)
    sign=raw>>63
    exponent=(raw>>52)&2047
    fraction=raw&((1<<52)-1)
    assert exponent!=2047
    if exponent:
        number=(sign<<127)|((exponent-1023+16383)<<112)|(fraction<<60)
    elif fraction:
        bits=fraction.bit_length()
        unbiased=bits-1-1074
        significand=fraction<<(113-bits)
        number=(sign<<127)|((unbiased+16383)<<112)|(significand-(1<<112))
    else:number=sign<<127
    return [f'{number&((1<<64)-1):016X}',f'{number>>64:016X}']

def rounded(words):
    assert len(words)%2==0
    return [integer.round64(*integer.decode(words[i:i+2])) for i in range(0,len(words),2)]

def expected_cell(native,qp,cell):
    assert 1<=cell<=8
    bits=64 if cell in (1,2,4,5) else 128
    fields={}
    for name in input_fields(native['n'],native['rank']):
        dp=native['data'][name]
        if cell==8 or (cell==7 and name.startswith('C:')):
            fields[name]=qp['data'][name]
            continue
        if cell in (4,5,6) and name.startswith('C:'):
            dp=rounded(qp['data'][name])
        fields[name]=dp if bits==64 else [piece for word in dp for piece in promote(word)]
    return bits,fields

def input_text():
    output=[]
    for index,(key,native,comp,qp) in enumerate(cases(),1):
        output.append('CASE '+str(index)+' '+' '.join(map(str,key)))
        for tag,record in [('DP',native),('QP',qp)]:
            output.append(tag)
            output.extend(' '.join(record['data'][name]) for name in input_fields(record['n'],record['rank']))
        output.append('END')
    return '\n'.join(output)+'\n'

def check(path):
    lines=iter(path.read_text().splitlines())
    result=[]
    controls=[]
    for index,(key,native,comp,qp) in enumerate(cases(),1):
        n,rank=key[-2:]
        for cell in range(1,9):
            header=next(lines).split()
            assert header==['BEGIN',str(index),*map(str,key),str(cell)],header
            bits,expected=expected_cell(native,qp,cell)
            assert next(lines).split()==['INPUT',str(bits)]
            observed={}
            for name in input_fields(n,rank):
                tag,indices=name.split(':')
                fields=next(lines).split()
                expected_index=indices.split(',') if indices else []
                assert fields[:1+len(expected_index)]==[tag,*expected_index]
                words=fields[1+len(expected_index):]
                assert words==expected[name],('actual replay input changed',key,cell,name)
                observed[name]=words
            status_row=next(lines).split()
            assert len(status_row)==3 and status_row[:2]==['RESULT',str(bits)]
            status=int(status_row[2]);assert status in (0,1,2)
            outputs={}
            for k in range(4):
                fields=next(lines).split()
                lead=['R',str(k)] if k<3 else ['R1']
                assert fields[:len(lead)]==lead
                words=fields[len(lead):]
                assert len(words)==bits//32 and all(re.fullmatch('[0-9A-F]{16}',word) for word in words)
                outputs[f'R:{k}' if k<3 else 'R1:']=words
            assert next(lines).split()==['END',str(index),str(cell)]
            expected_record={1:native,2:comp,8:qp}.get(cell)
            if expected_record:
                passed=status==expected_record['status'] and all(outputs[name]==expected_record['data'][name] for name in outputs)
                controls.append({'key':key,'cell':cell,'passed':passed})
            nonfinite=[]
            for name,words in outputs.items():
                for component in range(2):
                    high=int(words[(component+1)*(bits//64)-1],16)
                    exponent=(high>>52)&2047 if bits==64 else (high>>48)&32767
                    if exponent==(2047 if bits==64 else 32767):nonfinite.append([name,component])
            result.append({'key':key,'cell':cell,'bits':bits,'status':status,'output':outputs,'nonfinite_components':nonfinite})
    assert next(lines).split()==['COMPLETE','464','3712','0']
    assert next(lines,None) is None
    return {'complete':True,'records':len(result),'all_actual_input_words_match':True,
            'reproduction_controls':controls,'all_reproduction_controls_exact':all(r['passed'] for r in controls),
            'interpretation_permitted':all(r['passed'] for r in controls),
            'cells':result,'engine_calls_by_checker':0,
            'scope':'Stored QP input-bundle and reduction-domain diagnostic; no physical mixed-amplitude acceptance.'}
