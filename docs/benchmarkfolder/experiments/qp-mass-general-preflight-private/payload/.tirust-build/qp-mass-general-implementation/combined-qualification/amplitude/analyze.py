#!/usr/bin/env python3
"""Offline exact comparisons; parent controller supplies the execution envelope."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

if sys.flags.optimize:
    raise RuntimeError('Evidence checks require assertions')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
KINDS={'R8':1,'R16':2,'C16':2,'C32':4}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def flags(word):
    require(re.fullmatch('[TF]{5}',word) is not None,'Malformed IEEE flag word')
    return word


def parse_dp(path):
    """No native/QP tensor records are available in the fixed-DP observer."""
    lines=path.read_text().splitlines()
    require(lines and lines[0]=='FORMAT EW_QP_MASS_HEX_V1 storage-order-int64-words','Missing format')
    result={'data':{},'begin_flags':None,'end_flags':None};count=0;state='start';footer=None
    for line in lines[1:]:
        row=line.split();require(bool(row),'Empty journal row');tag=row[0]
        require(footer is None,'Records after footer')
        if tag=='BEGIN':
            require(state=='start' and len(row)==3 and row[1]=='1','Wrong point start')
            result['begin_flags']=flags(row[2]);state='active'
        elif tag=='END':
            require(state=='active' and len(row)==5 and row[1:4]==['1','0','0'],'QP group or incomplete DP point')
            result['end_flags']=flags(row[4]);state='ended'
        elif tag=='FOOTER':
            require(state=='ended' and len(row)==5,'Missing DP end/footer')
            footer=list(map(int,row[1:]));require(footer==[count,0,0,1000000],'Capped/dropped/flag-changing DP stream')
        else:
            require(state=='active','Record outside point');count+=1
            require(tag in KINDS or tag in ('INTEGER','LOGICAL'),'Unexpected DP transport record: '+tag)
            if tag in KINDS:
                require(len(row)==3+KINDS[tag] and int(row[2])>0,'Wrong array shape')
                require(all(re.fullmatch('[0-9A-F]{16}',x) for x in row[3:]),'Wrong storage words')
                item=[tag,int(row[2]),*row[3:]]
            else:
                require(len(row)==4,'Wrong scalar shape');flags(row[3])
                require(row[2] in ('T','F') if tag=='LOGICAL' else re.fullmatch('-?[0-9]+',row[2]),'Wrong scalar value')
                item=[tag,row[2],row[3]]
            result['data'].setdefault(row[1],[]).append(item)
    require(footer is not None,'Missing DP footer')
    expected={'INPUT_P':('R8',20),'RETURN_P':('R8',20),'AMPLITUDE':('R8',8),'ACCURACY':('R8',2),
              'MODEL_MU_R':('R8',1),'MODEL_MP__MU_R':('C32',1),'MODEL_GAL':('C16',2),'MODEL_MP__GAL':('C32',2)}
    for mass in ('MT','MZ','MW','MH'):
        expected['MODEL_MDL_'+mass]=('R8',1);expected['MODEL_MP__MDL_'+mass]=('R16',1)
    scalar={'PUBLIC_STATUS':'INTEGER',**{label:'LOGICAL' for label in ('FPE_DP_BEGIN','FPE_QP_BEGIN','FPE_DP_END','FPE_QP_END')}}
    require(set(result['data'])==set(expected)|set(scalar),'Missing/unexpected DP labels')
    for label,(kind,size) in expected.items():
        rows=result['data'][label]
        require(len(rows)==size and all(r[0]==kind for r in rows) and [r[1] for r in rows]==list(range(1,size+1)),label)
    for label,kind in scalar.items():
        rows=result['data'][label];require(len(rows)==1 and rows[0][0]==kind,label)
    result['footer']=footer
    return result


def canonical_data(data,final=False):
    """Only scalar event-time flags are excluded; bit arrays are untouched."""
    return {key:[r if final or r[0] not in ('INTEGER','LOGICAL') else r[:2] for r in value]
            for key,value in data.items()}


def qp_compare(old,new):
    failures=[];refusals=[]
    for number in (1,2):
        a,b=old['points'][number],new['points'][number]
        if canonical_data(a['data'])!=canonical_data(b['data']):failures.append([number,'point_data'])
        for label in ('PUBLIC_STATUS','FPE_DP_END','FPE_QP_END'):
            if a['data'][label]!=b['data'][label]:failures.append([number,label+'_and_final_flags'])
        if a['end_flags']!=b['end_flags']:failures.append([number,'final_host_IEEE'])
        if [g['key'] for g in a['groups']]!=[g['key'] for g in b['groups']]:
            failures.append([number,'group_order']);continue
        refused=[]
        for left,right in zip(a['groups'],b['groups'],strict=True):
            group=left['key'][1]
            if canonical_data(left['data'])!=canonical_data(right['data']):failures.append([number,group,'all_mass_nonmass_output_bits'])
            la=[(n['backend'],n['status']) for n in left['native']]
            ra=[(n['backend'],n['status']) for n in right['native']]
            if la!=ra:failures.append([number,group,'native_backend_status'])
            if left['errors']!=right['errors']:failures.append([number,group,'native_failure_message'])
            if any(n['status']!=0 for n in right['native']):refused.append(group)
        refusals.append(refused)
    return {'exact_mass_nonmass_output_status_final_flags_match':not failures,'failures':failures,
            'refused_groups_by_point':refusals,
            'intermediate_sticky_IEEE_event_timing_compared':False}


def check_run(run):
    directory=Path(run['cwd'])
    for path,checksum in run['input_sha256'].items():require(sha(path)==checksum,'Changed staged input: '+path)
    for output in run['expected_outputs']:require((directory/output).is_file(),'Missing '+run['name']+'/'+output)
    require((directory/'param.log').stat().st_size>0,'Empty param.log')
    log=(directory/'observer-run.log').read_text().splitlines()
    banner='FIXED_QP_TRANSPORT_ONLY_UNCHECKED' if run['mode']==4 else 'FIXED_DP_OUTPUT_ONLY_UNCHECKED'
    require(log and log[0]==banner and log[-1]=='COMPLETE_DIAGNOSTIC_ONLY','Incomplete/wrong-mode observer log')
    rows=[line.split() for line in log[1:-1]]
    require(rows and len(rows[0])==2 and rows[0][0]=='BOOTSTRAP','Missing bootstrap')
    int(rows[0][1])
    ready=[i for i,row in enumerate(rows) if row==['READY_DIAGNOSTIC_ONLY']]
    require(len(ready)==1,'Missing/duplicate ready marker');i=ready[0]
    warm=rows[1:i];require(2<=len(warm)<=1002,'Unbounded/missing warmup')
    for index,row in enumerate(warm,1):
        point=1+(index-1)%run['point_count']
        require(len(row)==4 and row[:3]==['WARMUP',str(index),str(point)],'Wrong warmup sequence')
        int(row[3])
    require(all((int(row[3])%100)//10 in (1,2) for row in warm[-2:]),'Warmup never reached two successes')
    observed=rows[i+1:]
    require(len(observed)==run['point_count'],'Wrong observed point count')
    for number,row in enumerate(observed,1):
        require(len(row)==3 and row[:2]==['OBSERVED',str(number)],'Malformed observed status');int(row[2])
    return [int(row[2]) for row in observed]


def main():
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--stage',choices=('dp','qp','all'),default='all')
    options=arg.parse_args()
    prepared=json.loads((HERE/'prepared.json').read_text())
    spec=prepared['frozen_qp_parser'];require(sha(spec['path'])==spec['sha256'],'Frozen parser changed')
    loader=importlib.util.spec_from_file_location('frozen_mass_comparator',spec['path'])
    parser=importlib.util.module_from_spec(loader);loader.loader.exec_module(parser)
    runs={r['name']:r for r in prepared['runs']};out=[]
    if options.stage in ('dp','all'):
        for backend in (8,9):
            captures=[]
            for arm in ('legacy-dp','guarded-dp'):
                run=runs[f'{arm}-{backend}'];codes=check_run(run)
                capture=parse_dp(Path(run['cwd'])/'mass-observer.hex')
                require(codes==[int(capture['data']['PUBLIC_STATUS'][0][1])],'DP log/journal code disagreement')
                captures.append(capture)
            out.append({'mode':'dp','backend':backend,'exact_all_observed_bits_status_flags_match':captures[0]==captures[1],
                        'QP_transport_observed':False,'observed_points':1})
    if options.stage in ('qp','all'):
        for backend in (8,9):
            run=runs[f'guarded-qp-{backend}'];codes=check_run(run)
            oracle=prepared['oracles'][str(backend)];require(sha(oracle['path'])==oracle['sha256'],'Frozen oracle changed')
            left=parser.parse(Path(oracle['path']),'candidate',backend)
            right=parser.parse(Path(run['cwd'])/'mass-observer.hex','candidate',backend)
            require(codes==[int(right['points'][n]['data']['PUBLIC_STATUS'][0][1]) for n in (1,2)],'QP log/journal code disagreement')
            compared=qp_compare(left,right)
            require([len(x) for x in compared['refused_groups_by_point']]==([0,37] if backend==9 else [0,0]),'Changed fixed-QP refusal pattern')
            out.append({'mode':'qp','backend':backend,**compared})
    passed=all(r.get('exact_all_observed_bits_status_flags_match',r.get('exact_mass_nonmass_output_status_final_flags_match',False)) for r in out)
    result={'stage':options.stage,'comparison_passed':passed,'comparisons':out,
            'source_manifest_sha256':sha(HERE/'prepared.json'),
            'controller_envelope_required_separately':True,'native_execution_qualified_by_this_script':False,
            'physical_accuracy_or_stability_acceptance':False,'performance_claim':False}
    print(json.dumps(result,indent=2,sort_keys=True))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
