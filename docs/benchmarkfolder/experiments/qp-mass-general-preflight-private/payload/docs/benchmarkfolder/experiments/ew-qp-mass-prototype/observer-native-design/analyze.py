#!/usr/bin/env python3
"""Compare exact observer records offline; never infer checked-QP acceptance."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import struct
import hashlib
import sys

if sys.flags.optimize:
    raise RuntimeError('Evidence validation requires Python assertions; do not use -O/PYTHONOPTIMIZE')

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
POINT_LABELS={'INPUT_P','RETURN_P','AMPLITUDE','ACCURACY','PUBLIC_STATUS',
              'FPE_DP_BEGIN','FPE_QP_BEGIN','FPE_DP_END','FPE_QP_END',
              'MODEL_UPDATE','GENUINE_QP_INTEGRAND','CONSUMED_MP_PS'}
COUNTS={'R8':1,'R16':2,'C16':2,'C32':4}


def parse(path,arm='original',backend=9):
    assert arm in ('original','candidate') and backend in (8,9)
    points={};point=None;group=None;footer=None;observed_records=0
    lines=path.read_text().splitlines()
    if not lines or lines[0]!='FORMAT EW_QP_MASS_HEX_V1 storage-order-int64-words':
        raise ValueError('Missing exact observer format marker')
    for line in lines[1:]:
        row=line.split();tag=row[0]
        assert footer is None, 'Records after footer'
        if tag not in ('BEGIN','END','FOOTER'):observed_records+=1
        if tag=='BEGIN':
            k=int(row[1]);assert k not in points and point is None
            point={'number':k,'begin_flags':row[2],'data':{},'groups':[],'ended':False}
            points[k]=point;group=None
        elif tag=='GROUP':
            assert point and int(row[1])==point['number']
            key=list(map(int,row[2:8]));assert key[0]==len(point['groups'])+1
            assert 1<=key[1]<=200 and key[2]==1 and key[3]==4
            assert 1<=key[4]<=5 and 0<=key[5]<=3
            group={'key':key,'flags':row[8],'data':{},'native':[],'errors':[]}
            point['groups'].append(group)
        elif tag in COUNTS:
            assert point and len(row)==3+COUNTS[tag]
            label=row[1];i=int(row[2]);assert i>0
            assert all(re.fullmatch('[0-9A-F]{16}',w) for w in row[3:])
            target=point if label in POINT_LABELS or label.startswith('MODEL_') else group
            assert target is not None
            target['data'].setdefault(label,[]).append([tag,i,*row[3:]])
        elif tag in ('INTEGER','LOGICAL'):
            assert point and len(row)==4
            label=row[1];target=point if label in POINT_LABELS or label.startswith('MODEL_') else group
            assert target is not None
            if tag=='LOGICAL':assert row[2] in ('T','F')
            else:int(row[2])
            target['data'].setdefault(label,[]).append([tag,row[2],row[3]])
        elif tag=='NATIVE':
            assert group and len(row)==4
            group['native'].append({'backend':int(row[1]),'status':int(row[2]),'flags':row[3]})
        elif tag=='ERROR_BYTES':
            assert group and all(re.fullmatch('[0-9A-F]{2}',x) for x in row[1:])
            group['errors'].append(bytes(int(x,16) for x in row[1:]).decode('utf-8','replace'))
        elif tag=='END':
            assert point and int(row[1])==point['number']
            assert int(row[2])==len(point['groups'])
            assert int(row[3])==(point['groups'][-1]['key'][1] if point['groups'] else 0)
            point['end_flags']=row[4];point['ended']=True;point=None;group=None
        elif tag=='FOOTER':
            assert footer is None and point is None
            footer=dict(zip(('records','dropped','flag_changes','limit'),map(int,row[1:])))
        elif tag=='OBSERVER_FLAG_CHANGE':
            raise ValueError('Observer changed an IEEE flag; do not qualify this capture')
        else:raise ValueError('Unknown observer record '+tag)
    if set(points)!={1,2} or not all(p['ended'] for p in points.values()):
        raise ValueError('Incomplete point stream')
    if footer is None or footer['dropped'] or footer['flag_changes'] or footer['records']>footer['limit']:
        raise ValueError('Incomplete/capped/flag-changing capture')
    assert footer['records']==observed_records and footer['limit']==1000000
    def array(target,label,size,kind):
        rows=target['data'].get(label,[])
        assert len(rows)==size and [r[1] for r in rows]==list(range(1,size+1)),label
        assert all(r[0]==kind for r in rows),label
    def scalar(target,label,kind,value=None):
        rows=target['data'].get(label,[])
        assert len(rows)==1 and rows[0][0]==kind,label
        if value is not None:assert rows[0][1]==str(value),label
    for point in points.values():
        for label,n,kind in [('INPUT_P',20,'R8'),('RETURN_P',20,'R8'),('AMPLITUDE',8,'R8'),
                             ('ACCURACY',2,'R8'),('CONSUMED_MP_PS',20,'R16'),('MODEL_MU_R',1,'R8'),
                             ('MODEL_MP__MU_R',1,'C32'),('MODEL_GAL',2,'C16'),('MODEL_MP__GAL',2,'C32')]:
            array(point,label,n,kind)
        for mass in ('MT','MZ','MW','MH'):
            array(point,'MODEL_MDL_'+mass,1,'R8');array(point,'MODEL_MP__MDL_'+mass,1,'R16')
        scalar(point,'PUBLIC_STATUS','INTEGER');scalar(point,'MODEL_UPDATE','INTEGER',1)
        scalar(point,'GENUINE_QP_INTEGRAND','LOGICAL','T')
        for label in ('FPE_DP_BEGIN','FPE_QP_BEGIN','FPE_DP_END','FPE_QP_END'):scalar(point,label,'LOGICAL')
        assert len(point['groups'])==200
        assert sorted(g['key'][1] for g in point['groups'])==list(range(1,201))
        for g in point['groups']:
            n,rank=g['key'][4:6];nc=math.comb(rank+4,4)
            for label,size,kind in [('DP_M2L',n,'C16'),('HELPER_DP_PL',4*n,'R8'),('HELPER_QP_PL',4*n,'R16'),
                               ('REPAIRED_MP_PS',20,'R16'),('CURRENT_MP_P',20,'R16'),('NUMERATOR_COEFS',nc,'C32'),('ENTRY_PL',4*n,'R16')]:
                array(g,label,size,kind)
            for i in range(1,n+1):array(g,'ARG_M'+str(i),1,'C16')
            if arm=='candidate':
                assert 'ENTRY_M2_DP' not in g['data']
                array(g,'ENTRY_M2_QP',n,'C32');array(g,'QP_M2L',n,'C32')
                for i in range(1,n+1):array(g,'MP_ARG_M'+str(i),1,'C32')
            else:
                array(g,'ENTRY_M2_DP',n,'C16')
                assert not any(x=='ENTRY_M2_QP' or x=='QP_M2L' or x.startswith('MP_ARG_') for x in g['data'])
            scalar(g,'GROUP_STABLE','LOGICAL');array(g,'GROUP_RESULT',3,'C16')
            scalar(g,'ACTUAL_BACKEND','INTEGER',backend)
            if len(g['native'])>1:raise ValueError('Repeated native call in one observed helper')
            assert all(x['backend']==backend for x in g['native'])
            raw_fields={label for label in g['data'] if label.startswith(('RAW_','NRS_RESULT','NRS_R1'))}
            if raw_fields:
                assert len(g['native'])==1 and g['native'][0]['status']==0
                assert not any(label.startswith('NRS_') if backend==9 else label.startswith('RAW_') for label in raw_fields)
            if g['native']:
                for label,size,kind in [('ORDERED_PL',4*n,'R16'),('ORDERED_MASSES',n,'C32'),('NATIVE_SHIFTS',4*n,'R16'),
                                   ('KINEMATIC_S',n*n,'C32'),('NATIVE_INVARIANTS',n*n,'R16'),('NATIVE_MASSES',n,'C32'),('NATIVE_SCALE',1,'R16')]:
                    array(g,label,size,kind)
                if backend==8:array(g,'NRS_NUMERATOR',nc,'C32')
                else:
                    scalar(g,'CHECK_PASS','LOGICAL','F');scalar(g,'CACHE_ENABLED','LOGICAL','T')
                if g['native'][0]['status']==0:
                    if g['native'][0]['backend']==9:
                        for label in ('RAW_UV1','RAW_IR1','RAW_IR2','RAW_FINITE'):array(g,label,nc,'C32')
                    else:array(g,'NRS_RESULT',3,'C32');array(g,'NRS_R1',1,'C32')
    return {'points':points,'footer':footer}


def finite_value(hexwords,bits):
    """Decode exact IEEE finite value from little-endian storage-order words."""
    assert len(hexwords)==bits//64
    word=sum(int(h,16)<<(64*i) for i,h in enumerate(hexwords))
    mantissa_bits,exponent_bits,bias=(52,11,1023) if bits==64 else (112,15,16383)
    sign=-1 if word>>(bits-1) else 1
    exponent=(word>>mantissa_bits)&((1<<exponent_bits)-1)
    mantissa=word&((1<<mantissa_bits)-1)
    if exponent==(1<<exponent_bits)-1:return None
    power=(exponent-bias if exponent else 1-bias)-mantissa_bits
    if exponent:mantissa+=1<<mantissa_bits
    return sign*Fraction(mantissa)*(Fraction(2)**power)


def decimal(value):
    if value is None:return 'nonfinite'
    with localcontext() as ctx:
        ctx.prec=45
        return str(Decimal(value.numerator)/Decimal(value.denominator))


def row_values(row):
    tag=row[0]
    if tag=='R8':return [finite_value(row[2:],64)]
    if tag=='R16':return [finite_value(row[2:],128)]
    if tag=='C16':return [finite_value(row[2:3],64),finite_value(row[3:4],64)]
    if tag=='C32':return [finite_value(row[2:4],128),finite_value(row[4:6],128)]
    raise ValueError(tag)


def compare_data(a,b,labels):
    return [label for label in sorted(labels) if a.get(label)!=b.get(label)]


def compare_pair(a,b,kind):
    """Bitwise nonmass checks; never require outputs to remain unchanged."""
    failures=[];changes=[];outputs=[];masses=[]
    for k in (1,2):
        left,right=a['points'][k],b['points'][k]
        pointlabels={'INPUT_P','CONSUMED_MP_PS'}|{x for x in left['data'] if x.startswith('MODEL_') and x!='MODEL_UPDATE'}
        # Metadata flags can legitimately differ after changed native masses.
        for label in pointlabels:
            if left['data'].get(label)!=right['data'].get(label):failures.append({'point':k,'scope':'point','label':label})
        if [g['key'] for g in left['groups']]!=[g['key'] for g in right['groups']]:
            failures.append({'point':k,'scope':'group_sequence'});continue
        for l,r in zip(left['groups'],right['groups']):
            labels={'DP_M2L','HELPER_DP_PL','HELPER_QP_PL','REPAIRED_MP_PS','CURRENT_MP_P',
                    'NUMERATOR_COEFS','ENTRY_PL','ORDERED_PL','NATIVE_SHIFTS'}
            labels|={x for x in l['data'] if x.startswith('ARG_')}
            if kind=='same_backend':labels|={'NATIVE_SCALE','NRS_NUMERATOR'}
            else:labels|={'ORDERED_MASSES','NATIVE_MASSES','KINEMATIC_S','NATIVE_INVARIANTS'}
            for label in compare_data(l['data'],r['data'],labels):
                failures.append({'point':k,'ordinal':l['key'][0],'group':l['key'][1],'label':label})
            for label in ('NATIVE_MASSES','KINEMATIC_S','NATIVE_INVARIANTS','NATIVE_SCALE','GROUP_RESULT','GROUP_STABLE'):
                if l['data'].get(label)!=r['data'].get(label):
                    changes.append({'point':k,'group':l['key'][1],'label':label,
                                    'classification':'pre-existing scale convention' if kind=='cross_backend' and label=='NATIVE_SCALE' else 'observed difference'})
            if kind=='same_backend' and 'QP_M2L' in r['data']:
                for i,(old,new) in enumerate(zip(l['data']['DP_M2L'],r['data']['QP_M2L']),1):
                    x,y=row_values(old),row_values(new)
                    if x!=y:
                        masses.append({'point':k,'group':l['key'][1],'helper_mass_index':i,
                                       'old_dp_square_bits':old,'new_qp_square_bits':new,
                                       'new_minus_old':[decimal(b-a) if a is not None and b is not None else 'nonfinite' for a,b in zip(x,y)]})
        for j,(l,r) in enumerate(zip(left['data']['AMPLITUDE'],right['data']['AMPLITUDE'])):
            x,y=row_values(l)[0],row_values(r)[0]
            denominator=max(abs(x),abs(y)) if x is not None and y is not None else None
            relative=(abs(x-y)/denominator if denominator else Fraction(0)) if denominator is not None else None
            outputs.append({'point':k,'flat_index':j+1,'left':decimal(x),'right':decimal(y),
                            'relative_symmetric':decimal(relative),'symmetric_relative_le_3e_6':relative is not None and relative<=Fraction(3,10**6)})
    return {'nonmass_input_bits_match':not failures,'failures':failures,
            'differences':changes,'mass_square_changes':masses,'amplitude_comparison':outputs,
            'qualification':'Fixed-QP diagnostics only. Agreement is not host stability or physical accuracy acceptance.'}


def summarize(capture,backend):
    result=[]
    for k,p in capture['points'].items():
        native=[n for g in p['groups'] for n in g['native']]
        assert all(n['backend']==backend for n in native)
        assert all(g['data']['ACTUAL_BACKEND'][0][1]==str(backend) for g in p['groups'])
        result.append({'point':k,'groups':len(p['groups']),'native_calls':len(native),
                       'native_refusals':sum(n['status']!=0 for n in native),
                       'early_adapter_returns':len(p['groups'])-len(native),
                       'public_status':p['data']['PUBLIC_STATUS'],
                       'accuracy_bits':p['data']['ACCURACY'],
                       'fpe_end':{key:p['data'].get(key) for key in ('FPE_DP_END','FPE_QP_END')},
                       'refused_groups':[{'group':g['key'][1],'native':g['native'],'errors':g['errors']}
                                         for g in p['groups'] if not g['native'] or any(n['status']!=0 for n in g['native'])]})
    return result


def historical_control(capture,expected):
    mismatches=[]
    for key,e in expected['points'].items():
        p=capture['points'][int(key)]['data']
        for label,field in [('AMPLITUDE','amplitude_bits'),('ACCURACY','accuracy_bits')]:
            if [row[2] for row in p[label]]!=e[field]:mismatches.append({'point':int(key),'field':label})
        if p['PUBLIC_STATUS'][0][1]!=str(e['status']):mismatches.append({'point':int(key),'field':'PUBLIC_STATUS'})
        for label,field in [('FPE_DP_END','fpe_dp'),('FPE_QP_END','fpe_qp')]:
            if field in e and (p[label][0][1]=='T')!=e[field]:mismatches.append({'point':int(key),'field':label})
    return {'exact_retained_binary64_status_and_available_host_flag_parity':not mismatches,
            'mismatches':mismatches,'host_flags_available':'fpe_dp' in expected['points']['1'],
            'scope':'Historical fixed-QP control only, no checked-QP acceptance.'}


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def runner_envelope(run_root,run_path,build_path):
    """Require the successful runner envelope, not just a valid early journal."""
    run=json.loads(run_path.read_text());built=json.loads(build_path.read_text())
    prepared=json.loads((HERE/'prepared.json').read_text())
    parts={'prepared_sha256':file_hash(HERE/'prepared.json'),'build_py_sha256':file_hash(HERE/'build.py'),
           'run_py_sha256':file_hash(HERE/'run.py')}
    review=hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert run['state']=='complete' and built['state']=='complete'
    assert run['review_hash']==built['review_hash']==review
    assert run['build_json_sha256']==file_hash(build_path)
    assert run['cpu']==8 and run['process_seconds']==120 and run['total_seconds']==480
    assert run['automatic_retry'] is False and run['host_stability_qualified'] is False
    assert all(run['environment'][k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS'))
    expected={arm+'-'+str(b) for arm in ('original','candidate') for b in (8,9)}
    assert len(run['cases'])==4
    assert {c['arm']+'-'+str(c['backend']) for c in run['cases']}==expected
    loaded=None
    for case in run['cases']:
        name=case['arm']+'-'+str(case['backend']);directory=run_root/name
        assert case['complete'] is True and not case.get('error') and not case.get('inventory_error')
        for key in ('evaluation','loader_command'):
            assert case[key]['returncode']==0 and case[key]['timed_out'] is False and not case[key].get('error')
        exe=prepared['arms'][case['arm']]['executable']
        assert case['executable_sha256']==built['outputs'][exe]
        assert {'mass-observer.hex','observer-run.log','stdout.log','ldd.txt','inputs.json'}<=set(case['files_after'])
        for relative,h in case['files_after'].items():
            path=directory/relative
            assert path.resolve().is_relative_to(directory.resolve())
            assert file_hash(path)==h,(name,relative)
        assert file_hash(directory/'inputs.json')==case['inputs_sha256']
        inputs=json.loads((directory/'inputs.json').read_text())
        for relative,h in inputs['after'].items():assert case['files_after'][relative]==h
        lines=[s.strip() for s in (directory/'observer-run.log').read_text().splitlines() if s.strip()]
        assert lines[0]=='FIXED_QP_TRANSPORT_ONLY_UNCHECKED' and lines[-1]=='COMPLETE_DIAGNOSTIC_ONLY'
        observed=[s.split() for s in lines if s.startswith('OBSERVED ')]
        assert len(observed)==2 and [int(s[1]) for s in observed]==[1,2]
        if loaded is None:loaded=case['loaded_libraries']
        else:assert loaded==case['loaded_libraries']
        for soname,identity in prepared['expected_runtime'].items():
            actual=case['loaded_libraries'][soname]
            assert actual['sha256']==identity['sha256']
            assert Path(actual['path'])==ROOT/identity['path']
    return {'verified':True,'run_json_sha256':file_hash(run_path),'build_json_sha256':file_hash(build_path),'review_hash':review}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,default=ROOT/'.tirust-build/ew-qp-mass-prototype/observer-native-design/runs')
    parser.add_argument('--run-json',type=Path,default=HERE/'run.json')
    parser.add_argument('--build-json',type=Path,default=HERE/'build.json')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    captures={};summary={'scope':'Fixed CTModeRun=4 transport diagnostics, no checked-QP or performance claim','cases':{},'comparisons':{}}
    controls=json.loads((HERE/'historical-controls.json').read_text())['cases']
    try:summary['runner_envelope']=runner_envelope(args.run_root,args.run_json,args.build_json)
    except (OSError,ValueError,AssertionError,IndexError,KeyError) as error:
        summary['runner_envelope']={'verified':False,'error':str(error)}
    for arm in ('original','candidate'):
        for backend in (8,9):
            name=arm+'-'+str(backend)
            try:
                capture=parse(args.run_root/name/'mass-observer.hex',arm,backend)
                summary['cases'][name]={'complete':True,'points':summarize(capture,backend)}
                if name in controls:summary['cases'][name]['historical_control']=historical_control(capture,controls[name])
                captures[name]=capture
            except (OSError,ValueError,AssertionError,IndexError,KeyError) as error:
                summary['cases'][name]={'complete':False,'error':str(error)}
    pairs=[('original-'+str(b),'candidate-'+str(b),'same_backend') for b in (8,9)]
    pairs += [(arm+'-8',arm+'-9','cross_backend') for arm in ('original','candidate')]
    for left,right,kind in pairs:
        if left in captures and right in captures:summary['comparisons'][left+'_vs_'+right]=compare_pair(captures[left],captures[right],kind)
    summary['transport_interpretation_eligible']=(summary['runner_envelope']['verified'] and len(captures)==4 and
        all(c['complete'] for c in summary['cases'].values()) and len(summary['comparisons'])==4 and
        all(summary['cases'][name]['historical_control']['exact_retained_binary64_status_and_available_host_flag_parity'] for name in controls) and
        all(c['nonmass_input_bits_match'] for c in summary['comparisons'].values()))
    summary['limitations']=['NinjaRS historical host IEEE/FPE flags were not retained; do not claim their parity.',
        'No automatic host stability acceptance; status120 from positive CT mode remains unchecked.',
        'The symmetric relative metric is an offline diagnostic, not a replacement for MadLoop acceptance.',
        'No timing, performance, threshold physical accuracy or all-five-backend claim.']
    with args.output.open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')


if __name__=='__main__':main()
