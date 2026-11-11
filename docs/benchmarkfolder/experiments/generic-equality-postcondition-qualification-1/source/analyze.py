"""Strict integer-only replay of native input echoes and environment observations."""
from collections import Counter
import hashlib
import re


def require(value,message):
    if not value:raise ValueError(message)


def request_for(mode,constants):
    if mode==0:return [-1]*5
    rounds=constants[:4];flags=constants[4:9];all_flags=constants[9]
    rounding=rounds[mode-1] if 1<=mode<=4 else rounds[0]
    sticky=flags[mode-5] if 5<=mode<=9 else all_flags if mode==10 else 0
    halts=flags[mode-11] if 11<=mode<=15 else all_flags if mode==19 else 0
    ftz,daz={16:(1,0),17:(0,1),18:(1,1)}.get(mode,(-1,-1))
    return [rounding,sticky,halts,ftz,daz]


def analyze(text,expected):
    lines=[line.split() for line in text.splitlines() if line.strip()]
    cursor=0
    def take(tag,length=None):
        nonlocal cursor
        require(cursor<len(lines),'Truncated native observation')
        row=lines[cursor];cursor+=1
        require(row[0]==tag and (length is None or len(row)==length),'Unexpected native record: '+tag)
        return row[1:]
    require(take('SCHEMA',2)==['exact-geometry-fenv-v1'],'Wrong native observation schema')
    constants=list(map(int,take('CONSTANTS',11)))
    # This qualification explicitly targets the pinned GNU libc/x86_64 ABI.
    require(constants[:4]==[0,1024,2048,3072] and constants[4:9]==[1,4,8,16,32] and
            constants[9]==61,'Unsupported GNU/x86_64 fenv constants')
    count=int(take('NCASES',2)[0]);modes=int(take('MODES',2)[0])
    require(count==expected['case_count'] and modes==20,'Wrong native case/mode coverage')
    require(take('SNAPSHOT_FIELDS')==['round','sticky','halts','mxcsr_control','x87cw','mxcsr_raw','x87sw'],
            'Wrong environment snapshot layout')
    require(take('CASE_FIELDS')==['id','mode','status','accepted','restore_ok','request_round','request_sticky',
            'request_halts','request_ftz','request_daz','baseline(7)','before(7)','after(7)','restored(7)'],
            'Wrong native case layout')
    rows=[];mode_status={};counts=Counter();word_count=0
    for identifier in range(1,count+1):
        row=expected['cases'][str(identifier)]
        header=list(map(int,take('INPUT',6)))
        require(header==[identifier,row['width'],row['reference_width'],row['n'],row['shape']],
                'Native input identity/layout mismatch')
        coefficients=list(map(int,take('COEFF')))
        require(coefficients[0]==identifier and len(coefficients)==row['n']+1,'Native coefficient echo shape mismatch')
        require(all(-(1<<31)<=c<(1<<31) for c in coefficients[1:]),'Native default-integer coefficient overflow')
        canonical=[' '.join(map(str,header)),' '.join(map(str,coefficients[1:]))]
        for leg in range(1,row['n']+1):
            for mu in range(4):
                fields=take('X',6)
                require(list(map(int,fields[:3]))==[identifier,leg,mu],'Native external-word ordering mismatch')
                require(all(re.fullmatch('[0-9A-Fa-f]{32}',v) for v in fields[3:]),'Malformed native INTEGER128 echo')
                canonical.append(' '.join(v.upper() for v in fields[3:]));word_count+=2
        for mu in range(4):
            fields=take('Q',5)
            require(list(map(int,fields[:2]))==[identifier,mu],'Native difference-word ordering mismatch')
            require(all(re.fullmatch('[0-9A-Fa-f]{32}',v) for v in fields[2:]),'Malformed native q INTEGER128 echo')
            canonical.append(' '.join(v.upper() for v in fields[2:]));word_count+=2
        require(hashlib.sha256(('\n'.join(canonical)+'\n').encode()).hexdigest()==row['input_case_sha256'],
                'Complete native INTEGER128/default-integer input echo differs from frozen fixture')
        executed=[];skipped=[]
        for mode in range(20):
            values=list(map(int,take('CASE',39)))
            require(values[:2]==[identifier,mode],'Native mode identity/ordering mismatch')
            status,accepted,restored_ok=values[2:5]
            request=values[5:10];baseline=values[10:17];before=values[17:24]
            after=values[24:31];restored=values[31:38]
            require(status in (0,1),'Native environment setup failed')
            require(restored_ok==1 and restored==baseline,'Observed environment restoration failed')
            require(before==after,'Integer helper changed observed environment or sticky flags')
            require(request==request_for(mode,constants),'Native requested environment cohort differs')
            for snapshot in (baseline,before,after,restored):
                rounding,sticky,halts,control,cw,csr,sw=snapshot
                require(rounding in constants[:4] and 0<=sticky<=61 and sticky&~61==0 and
                        0<=halts<=61 and halts&~61==0,'Malformed native environment observation')
                require(0<=csr<2**32 and 0<=cw<2**16 and 0<=sw<2**16 and control==(csr&~63),
                        'Inconsistent native hardware/control word observation')
                require(sticky==((csr|sw)&constants[9]),'IEEE flag API disagrees with retained hardware status')
            require(mode_status.get(mode,status)==status,'Environment support differs between cases')
            mode_status[mode]=status
            if status==1:
                require(mode in (16,17,18),'Mandatory GNU/x86_64 cohort was skipped')
                require(accepted==-1 and before==baseline,'Unsupported cohort did work or changed state')
                skipped.append(mode);counts['skipped_mode_observations']+=1
                continue
            require(accepted==int(row['expected_accept']),'Native helper differs from independent Fraction/capacity oracle')
            if mode==0:
                require(before==baseline,'Untouched baseline was changed')
            else:
                require(before[:3]==request[:3],'Actual environment differs from requested cohort')
                require(before[4]&0x0c00==request[0] and (before[5]>>3)&0x0c00==request[0],
                        'Actual x87/SSE rounding differs from requested cohort')
                require((~before[4])&61==request[2] and (~(before[5]>>7))&61==request[2],
                        'Actual x87/SSE exception masks differ from requested cohort')
                if request[3]>=0:
                    require((before[5]>>15)&1==request[3] and (before[5]>>6)&1==request[4],
                            'Actual FTZ/DAZ differs from requested cohort')
            executed.append(mode);counts['executed_mode_observations']+=1
            counts['accepts' if accepted else 'refusals']+=1
        require(0 in executed,'No unchanged-environment execution')
        rows.append({'id':identifier,'tag':row['tag'],'accepted':row['expected_accept'],
            'executed_modes':executed,'explicitly_unsupported_modes':skipped,
            'input_case_sha256':row['input_case_sha256']})
    require(take('DONE',2)==[str(count)] and cursor==len(lines),'Incomplete/extra native observations')
    return {'status':'passed','case_count':count,'counts':dict(counts),'case_results':rows,
        'mode_status':mode_status,'native_INTEGER128_input_words_joined':word_count,
        'native_API':'Logical acceptance only; refusal reasons/error arrays are not native API outputs.',
        'zero_error_bound_scope':'Exact equality authorizes zero construction error; no caller error-array transport observed.',
        'new_scientific_reducer_calls':0,'snapshot_ownership_qualified':False,
        'physical_or_performance_qualification':False,
        'fenv_scope':'Only observed round/sticky/halting/MXCSR/x87 fields in executed explicit driver cohorts; unsupported modes excluded.'}
