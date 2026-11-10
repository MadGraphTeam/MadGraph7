"""Strict offline parsing of the replacement main's named IEEE observations."""
import math
from pathlib import Path
import re
import signal
import struct

FLAGS=('invalid','divide_by_zero','overflow','underflow','inexact')
ROUNDS={'nearest','to_zero','up','down','other'}


def require(value,message):
    if not value:raise ValueError(message)


def accuracy(word):
    require(re.fullmatch('[0-9A-F]{16}',word) is not None,'Malformed accuracy word')
    return struct.unpack('<d',int(word,16).to_bytes(8,'little'))[0]


def parse(path,*,allow_partial=False):
    lines=Path(path).read_text().splitlines()
    require(lines and lines[0]=='FORMAT ADAPTIVE_CHECKED_CONTROL_V1','Wrong control format')
    result={'mode':None,'policy':None,'states':[],'calls':[],'observed':{},'ready':False,
            'complete':False,'fault_enter':None,'fault_survived':None,'observation_changes':[]}
    state=None;active=None;warmup=0;mapping_ready=False
    for line in lines[1:]:
        row=line.split();require(bool(row),'Empty control record');tag=row[0]
        require(not result['complete'],'Record after completion')
        if tag=='ROUND':
            require(state is None or len(state['flags'])==5,'Incomplete prior IEEE state')
            require(len(row)==3 and row[2] in ROUNDS,'Malformed rounding')
            state={'phase':row[1],'rounding':row[2],'flags':{}};result['states'].append(state)
        elif tag=='STATE':
            require(state is not None and len(row)==6 and row[1]==state['phase'],'IEEE state without matching rounding')
            require(row[2] in FLAGS and row[2] not in state['flags'],'Duplicate/unknown named flag')
            require(all(x in ('T','F') for x in row[3:]),'Malformed IEEE state booleans')
            state['flags'][row[2]]=dict(zip(('set','halting','supported'),(x=='T' for x in row[3:])))
        elif tag=='READY_FOR_MAPPING':
            require(not mapping_ready and len(row)==1,'Duplicate mapping barrier');mapping_ready=True
        elif tag=='MODE':
            require(result['mode'] is None and len(row)==3,'Duplicate/malformed mode')
            require((row[1]=='physical' and row[2] in ('masked','trapped')) or
                    (row[1]=='fault' and row[2] in FLAGS[:3]),'Unknown control mode')
            result['mode'],result['policy']=row[1:]
        elif tag=='CALL_ENTER':
            require(result['mode']=='physical' and active is None and len(row)==4,'Malformed/repeated call entry')
            stage,index,point=row[1],int(row[2]),int(row[3]);require(point in (1,2),'Wrong point')
            require(bool(result['calls']) or stage=='BOOTSTRAP','First physical call must be bootstrap')
            if stage=='BOOTSTRAP':require(not result['calls'] and index==0 and point==1,'Wrong bootstrap')
            elif stage=='WARMUP':
                warmup+=1;require(1<=warmup<=1002 and index==warmup and point==1+(warmup-1)%2 and not result['ready'],'Wrong warmup')
            else:require(stage=='OBSERVED' and result['ready'] and index==point and point not in result['observed'],'Wrong observed entry')
            active={'stage':stage,'index':index,'point':point,'returned':False};result['calls'].append(active)
        elif tag=='CALL_RETURN':
            require(active is not None and len(row)==7 and row[1:4]==[active['stage'],str(active['index']),str(active['point'])],'Wrong call return')
            code=int(row[4]);require(code>=0,'Negative public code')
            values=[accuracy(word) for word in row[5:]]
            # Preserve NaN/Inf words without introducing nonstandard JSON values.
            active.update(returned=True,code=code,H=code//100,T=(code//10)%10,U=code%10,
                          accuracy_words=row[5:],accuracy=[x if math.isfinite(x) else None for x in values])
            if active['stage']=='OBSERVED':result['observed'][active['point']]=dict(active)
            active=None
        elif tag=='READY_INITIALIZATION_FINISHED_NOT_ACCEPTANCE':
            require(len(row)==1 and not result['ready'] and active is None and warmup>=2,'Wrong initialization marker')
            last=result['calls'][-2:];require(all(x['stage']=='WARMUP' and x['returned'] and x['T'] in (1,2) for x in last),'Initialization did not settle')
            result['ready']=True
        elif tag in ('FAULT_ENTER','FAULT_SURVIVED'):
            require(result['mode']=='fault' and len(row)==2 and row[1]==result['policy'],'Wrong fault marker')
            key=tag.lower();require(result[key] is None,'Repeated fault marker');result[key]=row[1]
        elif tag=='OBSERVATION_FLAG_CHANGE':
            require(len(row)==2,'Malformed observation failure');result['observation_changes'].append(row[1])
        elif tag=='COMPLETE_CHECKED_DIAGNOSTIC':
            require(result['mode']=='physical' and len(row)==2 and int(row[1])==len(result['observation_changes']),'Wrong completion')
            require(result['ready'] and active is None and set(result['observed'])=={1,2},'Incomplete physical calls')
            result['complete']=True
        else:raise ValueError('Unknown control record '+tag)
    require(mapping_ready,'Missing actual-map barrier')
    if state is not None and len(state['flags'])!=5:
        require(allow_partial,'Truncated final IEEE state')
    result['partial']=not result['complete']
    if result['complete']:
        expected=['PROGRAM_START','AFTER_POLICY','SETPARA_BEFORE','SETPARA_AFTER']
        for call in result['calls']:
            expected.extend(call['stage']+suffix for suffix in
                            ('_PREPROCESS_BEFORE','_PREPROCESS_AFTER','_BEFORE','_AFTER'))
        expected.extend(['EXIT_BEFORE','EXIT_AFTER'])
        require([s['phase'] for s in result['states']]==expected,'Missing/reordered per-call IEEE observations')
    if not allow_partial:require(result['complete'],'Incomplete checked-control stream')
    return result


def assess(result,*,exit_code):
    """Completion and IEEE-policy checks are separate from numerical acceptance."""
    failures=[];states=result['states'];policy=result['policy']
    configured=[s for s in states if s['phase']=='AFTER_POLICY']
    if len(configured)!=1:failures.append('missing_policy_state')
    if result['observation_changes']:failures.append('observer_changed_flags')
    rounding={s['rounding'] for s in states if len(s['flags'])==5}
    if len(rounding)!=1 or 'other' in rounding:failures.append('rounding_changed_or_unrecognized')
    for state in states:
        if state['phase']=='PROGRAM_START':continue
        if len(state['flags'])!=5:failures.append('partial_state');continue
        for index,name in enumerate(FLAGS):
            flag=state['flags'][name];expected=(policy!='masked' and index<3)
            if not flag['supported']:failures.append('unsupported_'+name)
            if flag['halting']!=expected:failures.append('halting_policy_changed_'+name)
    if result['mode']=='fault':
        if [s['phase'] for s in states]!=['PROGRAM_START','AFTER_POLICY']:
            failures.append('missing_or_reordered_fault_initial_states')
        valid=exit_code==-signal.SIGFPE and result['fault_enter']==policy and result['fault_survived'] is None and not failures
        return {'fault_control_demonstrated':valid,'policy_failures':failures,'physical_qualification':False}
    valid=exit_code==0 and result['complete'] and not failures
    return {'physical_worker_completed_with_requested_IEEE_policy':valid,'policy_failures':failures,
            'physical_qualification':False,'actual_per_call_precision_available':False}
