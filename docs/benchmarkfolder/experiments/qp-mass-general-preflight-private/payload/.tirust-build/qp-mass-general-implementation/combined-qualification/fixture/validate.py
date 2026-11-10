#!/usr/bin/env python3
"""Strict offline gate for the fully enumerated caller-boundary fixture."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parents[2]


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power2(n):
    return Fraction(2**n,1) if n>=0 else Fraction(1,2**(-n))


def exponent(value):
    value=abs(Fraction(value));require(value>0,'Nonzero exponent required')
    e=value.numerator.bit_length()-value.denominator.bit_length()
    if value<power2(e):e-=1
    assert power2(e)<=value<power2(e+1)
    return e


def nearest(value,p):
    value=Fraction(value)
    if not value:return value
    unit=power2(exponent(value)-p+1)
    scaled=abs(value)/unit
    q,r=divmod(scaled.numerator,scaled.denominator)
    if 2*r>scaled.denominator or (2*r==scaled.denominator and q%2):q+=1
    return (-1 if value<0 else 1)*q*unit


def real_words(value,p):
    value=Fraction(value)
    if not value:return ['0000000000000000']*(1 if p==53 else 2)
    value=nearest(value,p);e=exponent(value)
    fraction=int(abs(value)/power2(e-p+1))-(1<<(p-1))
    bias=1023 if p==53 else 16383
    bits=((1 if value<0 else 0)<<(63 if p==53 else 127))|((e+bias)<<(p-1))|fraction
    return [f'{(bits>>(64*i))&((1<<64)-1):016X}' for i in range(1 if p==53 else 2)]


def complex_words(real,imag=0,p=53):
    return real_words(real,p)+real_words(imag,p)


def derived(p):
    return nearest(nearest(Fraction(3)+nearest(Fraction(1,10),p),p)/3,p)


def model_words(payload):
    dp={'ZERO':complex_words(0),'MDL_REAL':complex_words(3),
        'MDL_DERIVED':complex_words(derived(53)),
        'CMASS_MDL_CMS':complex_words(5,Fraction(-1,8))}
    zero='0000000000000000'
    mp={'MP__ZERO':[zero]*4}
    if payload=='valid':
        mp.update(MP__MDL_REAL=complex_words(3,p=113),
                  MP__MDL_DERIVED=complex_words(derived(113),p=113),
                  MP__CMASS_MDL_CMS=complex_words(5,Fraction(-1,8),p=113))
    elif payload=='finite_poison':
        mp.update(MP__MDL_REAL=complex_words(Fraction(35,2),p=113),
                  MP__MDL_DERIVED=complex_words(Fraction(-77,4),p=113),
                  MP__CMASS_MDL_CMS=[zero,'4003580000000000',zero,'8000000000000000'])
    elif payload in ('qnan','snan'):
        high='7FFF800000000000' if payload=='qnan' else '7FFF000000000000'
        negative='FFFF'+high[4:]
        mp.update(MP__MDL_REAL=['0000000000000011',high,zero,zero],
                  MP__MDL_DERIVED=['0000000000000022',negative,zero,zero],
                  MP__CMASS_MDL_CMS=['0000000000000033',high,'0000000000000044',negative])
    else:raise ValueError('Unknown payload '+payload)
    return dp,mp


def expected_rows(case,matrix,setup='FFFFF'):
    """Independent binary arithmetic expectations; no floating-point parser."""
    arm,mode,payload,trap,route=(case[k] for k in ('arm','mode','payload','trap','route'))
    dp,mp=model_words(payload)
    rows=[['FORMAT','QPMASS_CALLER_FIXTURE_V1'],
          ['CASE',arm,str(mode),payload,str(trap),route],
          ['FLAG_ORDER','invalid','divide_by_zero','overflow','underflow','inexact'],
          ['BINARY128_ONE','0000000000000000','3FFF000000000000'],
          ['FLAGS','SETUP',setup]]
    rows.extend(['MODEL_DP',k,*v] for k,v in dp.items())
    rows.extend(['MODEL_MP',k,*v] for k,v in mp.items())
    rows.extend([['FLAGS','BEFORE','FFFFF'],['PRECALL']])
    if arm=='trap_control':
        if not trap:rows.extend([['FLAGS','AFTER','TFFFF'],['COUNTS','0','0','0'],['COMPLETE','0','0']])
        return rows
    paired=arm=='guarded' and mode>=4
    calls=matrix[arm+'_'+route]['expected']
    for seq,item in enumerate(calls,1):
        s=str(seq)
        rows.append(['CALL',s,'QPMASS' if paired else 'LEGACY',
                     *[str(item[k]) for k in ('n','npair','nmother','rank','sqso','group')]])
        rows.append(['PAIRING',s,*map(str,item['pairing'])])
        rows.append(['MOTHERS',s,*map(str,item['mothers'])])
        for index,symbol in enumerate(item['masses'],1):
            rows.append(['ARG_DP',s,str(index),*dp[symbol]])
            if paired:rows.append(['ARG_MP',s,str(index),*mp['MP__'+symbol]])
        outputs=[dp[item['masses'][0]],dp[item['masses'][-1]],complex_words(item['rank'],item['group'])]
        rows.extend(['RESULT',s,str(i),*v] for i,v in enumerate(outputs,1))
        rows.append(['STATUS',s,'0'])
    n=len(calls)
    rows.extend([['FLAGS','AFTER','FFFFF'],['COUNTS',str(n),str(0 if paired else n),str(n if paired else 0)],
                 ['COMPLETE',str(n),'0']])
    return rows


def parse_case(text,job,matrix):
    rows=[line.split() for line in text.splitlines()]
    require(all(rows),'Blank journal row')
    require(len(rows)>4,'Incomplete journal header')
    require(rows[4][:2]==['FLAGS','SETUP'] and len(rows[4])==3,'Missing SETUP flags')
    setup=rows[4][2]
    require(len(setup)==5 and set(setup)<={'T','F'},'Invalid flag encoding')
    require(setup[:4]=='FFFF','Unexpected model setup invalid/divzero/overflow/underflow')
    if job['case']['payload']!='valid':require(setup=='FFFFF','Poison storage/observer raised IEEE flags')
    expected=expected_rows(job['case'],matrix,setup=setup)
    require(len(rows)==len(expected),f"Wrong journal row count {len(rows)} != {len(expected)}")
    for index,(actual,wanted) in enumerate(zip(rows,expected,strict=True),1):
        require(actual==wanted,f'Journal row {index} mismatch: {actual!r} != {wanted!r}')
    return rows


def validate(events_path,commands_path=HERE/'commands.json'):
    package=json.loads(commands_path.read_text())
    require(len(package['runs'])==218,'Incomplete run manifest')
    require(len({x['name'] for x in package['runs']})==218,'Duplicate run manifest')
    for name,digest in package['inputs'].items():require(sha(ROOT/name)==digest,'Source changed: '+name)
    matrix=json.loads((HERE/'argument-matrix.json').read_text())
    report=json.loads(events_path.read_text())
    jobs=report['jobs'];byname={}
    names={x['name'] for x in package['runs']}
    for event in jobs:
        if event['name'] in names:
            require(event['name'] not in byname,'Duplicate native fixture event')
            byname[event['name']]=event
    require(set(byname)==names,'Missing or extra fixture execution events')
    observations={};hashes={};duration=0
    for job in package['runs']:
        event=byname[job['name']]
        require(event.get('returncode')==job['expected_exit'],'Unexpected exit: '+job['name'])
        require(event.get('timed_out') is False and not event.get('error'),'Timeout/launch error: '+job['name'])
        require(event.get('cwd')==job['cwd'],'Changed run cwd: '+job['name'])
        command=event.get('command',[])
        require(len(command)==len(job['argv'])+3 and command[1:3]==['-c','8'] and command[3:]==job['argv'],
                'Changed native argv or affinity: '+job['name'])
        require(not event.get('changed_inputs'),'Run changed existing input: '+job['name'])
        require(event.get('expected_output_closure_matched') is True,'Unverified output closure: '+job['name'])
        require(0<=event['elapsed_seconds']<=3,'Case runtime cap exceeded: '+job['name'])
        duration+=event['elapsed_seconds']
        path=ROOT/job['cwd']/'caller.hex'
        require(path.is_file(),'Missing journal: '+job['name'])
        digest=sha(path)
        require(event.get('output_files',{}).get('caller.hex')==digest,'Journal hash mismatch: '+job['name'])
        observations[job['name']]=parse_case(path.read_text(),job,matrix)
        hashes[str(path.relative_to(ROOT))]=digest
    require(duration<=package['execution_seconds_max'],'Aggregate fixture runtime cap exceeded')
    dp_pairs=0;qp_pairs=0
    for job in package['runs']:
        case=job['case']
        if case['arm']!='legacy':continue
        counterpart=job['name'].replace('fixture-legacy-','fixture-guarded-',1)
        left=observations[job['name']];right=observations[counterpart]
        if case['mode']<4:
            require(left[:1]+left[2:]==right[:1]+right[2:],'DP legacy/guarded exact journal mismatch')
            dp_pairs+=1
        else:
            # QP receives additional MP args; compare the DP arguments, outputs,
            # synthetic statuses and model/setup/final flags without dropping
            # the separately checked MP records from parse_case.
            keys={'MODEL_DP','MODEL_MP','ARG_DP','RESULT','STATUS','FLAGS','PAIRING','MOTHERS'}
            require([r for r in left if r[0] in keys]==[r for r in right if r[0] in keys],
                    'QP paired caller DP output/model/flags mismatch')
            qp_pairs+=1
    require((dp_pairs,qp_pairs)==(90,18),'Incomplete matched pair coverage')
    return dict(schema='qp-mass-caller-fixture-validation-v1',passed=True,normal_cases=216,
                positive_invalid_trap_controls=2,dp_exact_pairs=dp_pairs,qp_transport_pairs=qp_pairs,
                commands_sha256=sha(commands_path),events_sha256=sha(events_path),journal_hashes=hashes,
                interpretation='Recording-stub caller argument/output/status/IEEE qualification only; no physical amplitude or reducer qualification')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--events',type=Path,required=True)
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    result=validate(args.events)
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.output:
        require(args.output.resolve().is_relative_to(HERE),'Output outside owned fixture')
        with args.output.open('x') as stream:stream.write(text)
    print(text,end='')


if __name__=='__main__':main()
