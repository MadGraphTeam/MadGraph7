"""Strict, read-only parsing/matching of the isolated integer capture stream."""
from collections import Counter
import math
import re


def parse(path):
    lines=iter(path.read_text().splitlines())
    assert next(lines).split()==['LAYOUT','3FF0000000000000','0000000000000000','3FFF000000000000']
    assert next(lines).split()==['COMPLEX64','3FF0000000000000','4000000000000000']
    assert next(lines).split()==['COMPLEX128','0000000000000000','3FFF000000000000','0000000000000000','4000000000000000']
    records=[]
    while True:
        begin=next(lines,None)
        if begin is None:break
        fields=begin.split()
        assert len(fields)==9 and fields[0]=='BEGIN',begin
        sequence,point,mode,group,order,n,rank,stage=map(int,fields[1:])
        assert sequence==len(records)+1 and sequence<=2500
        assert point in (44,80) and order==1 and 1<=group<=116 and 1<=n<=5 and 0<=rank<=4
        assert (stage in (1,2) and mode in (1,2)) or (stage==3 and mode in (4,5))
        payload=next(lines).split()
        assert len(payload)==5 and payload[:2]==['PAYLOAD',str(sequence)],payload
        bits,status,nc=map(int,payload[2:])
        assert bits==(128 if stage==3 else 64) and status in (0,1,2)
        assert nc==math.comb(rank+4,4)
        words_per_real=bits//64
        expected=[('C',(i,),2*words_per_real) for i in range(nc)]
        for j in range(1,n+1):
            expected.extend(('S',(i,j),words_per_real) for i in range(1,n+1))
            expected.extend(('P',(i,j),words_per_real) for i in range(4))
            expected.append(('M',(j,),2*words_per_real))
        expected.append(('U',(),words_per_real))
        expected.extend(('R',(i,),2*words_per_real) for i in range(3))
        expected.append(('R1',(),2*words_per_real))
        data={}
        for tag,index,count in expected:
            row=next(lines).split()
            assert row[0]==tag and tuple(map(int,row[1:1+len(index)]))==index,row
            words=row[1+len(index):]
            assert len(words)==count and all(re.fullmatch('[0-9A-F]{16}',v) for v in words),row
            for k in range(0,count,words_per_real):
                high=int(words[k+words_per_real-1],16)
                exponent=(high>>52)&2047 if bits==64 else (high>>48)&32767
                assert exponent!=(2047 if bits==64 else 32767),('nonfinite',sequence,tag,index)
            data[tag+':' + ','.join(map(str,index))]=words
        assert next(lines).split()==['END',str(sequence)]
        records.append({'sequence':sequence,'point':point,'ctmode':mode,'loop_id':group,
                        'order':order,'n':n,'rank':rank,'stage':stage,'bits':bits,
                        'status':status,'data':data})
    assert len(records)==1392, len(records)
    return records


def match(records):
    by_stage={stage:[] for stage in (1,2,3)}
    tables={stage:{} for stage in (1,2,3)}
    for record in records:
        direction=record['ctmode']-(3 if record['stage']==3 else 0)
        key=(record['point'],direction,record['loop_id'],record['order'],record['n'],record['rank'])
        stage=record['stage']
        assert key not in tables[stage],('duplicate',stage,key)
        tables[stage][key]=record;by_stage[stage].append(key)
    assert by_stage[1]==by_stage[2]==by_stage[3]
    assert len(by_stage[1])==464
    assert Counter((k[0],k[1]) for k in by_stage[1])==Counter({(p,d):116 for p in (44,80) for d in (1,2)})
    shapes={}
    for stage in (1,2,3):
        for point in (44,80):
            for direction in (1,2):
                cohort=[k for k in by_stage[stage] if k[:2]==(point,direction)]
                assert sorted(k[2] for k in cohort)==list(range(1,117))
                for key in cohort:
                    group=key[2];shape=key[4:]
                    assert shapes.setdefault(group,shape)==shape,('shape changed',stage,key)
    pairs=[]
    for key in by_stage[1]:
        native,compensated,qp=(tables[stage][key] for stage in (1,2,3))
        inputs=lambda r:{k:v for k,v in r['data'].items() if not k.startswith(('R:', 'R1:'))}
        assert inputs(native)==inputs(compensated),('native/compensated input mismatch',key)
        pairs.append({'key':list(key),'native_sequence':native['sequence'],
                      'compensated_sequence':compensated['sequence'],'qp_sequence':qp['sequence']})
    return {'complete':True,'matched_records':len(pairs),'records':pairs,
            'status_counts':{str(s):dict(Counter(str(r['status']) for r in records if r['stage']==s)) for s in (1,2,3)},
            'native_compensated_inputs_bit_identical':True,
            'geometry_scope':'DP and improved-QP geometry retained separately; no assertion of equal external kinematics'}
