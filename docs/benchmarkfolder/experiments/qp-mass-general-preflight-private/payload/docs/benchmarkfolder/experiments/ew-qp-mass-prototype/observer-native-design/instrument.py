#!/usr/bin/env python3
"""Add matched integer-only observer calls; never evaluate a numerical kernel."""
import importlib.util
from pathlib import Path
import re
import sys

if sys.flags.optimize:
    raise RuntimeError('Source validation requires Python assertions; do not use -O/PYTHONOPTIMIZE')

SPEC = importlib.util.spec_from_file_location('frozen_mass_transform', Path(__file__).resolve().parent.parent/'transform.py')
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
fixed = BASE.fixed
routines = BASE.routines


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old,new)


def around(text, predicate, before='', after=''):
    lines=text.splitlines(keepends=True)
    hits=[]
    for first,last,statement in BASE.logical_blocks(text):
        if statement is not None and predicate(statement):
            hits.append((first,last))
    assert len(hits)==1, hits
    first,last=hits[0]
    return ''.join(lines[:first])+before+''.join(lines[first:last])+after+''.join(lines[last:])


def use_observer(text):
    return around(text,lambda s:s.upper().startswith('SUBROUTINE '),after=fixed('USE EW_MASS_OBSERVER'))


def call(kind,label,n,value):
    return fixed("CALL OBS_"+kind+"('"+label+"',"+str(n)+','+value+')')


def helper(text, candidate):
    info=next(iter(routines(text).values()))
    masses=[arg for arg in info['args'] if re.fullmatch(r'M\d+',arg)]
    begin=fixed('IF(DOING_QP) THEN')+fixed('CALL OBS_GROUP(LOOPNUM,SQUAREDSOINDEX,CTMODE,NLOOPLINE,RANK)')
    for mass in masses:
        begin+=call('C16','ARG_'+mass,1,'(/'+mass+'/)')
        if candidate:begin+=call('C32','MP_ARG_'+mass,1,'(/MP_'+mass+'/)')
    begin+=fixed('ENDIF')
    text=around(text,lambda s:s.upper().startswith('IF (CHECKPHASE.'),after=begin)
    squares=fixed('IF(DOING_QP) THEN')+call('C16','DP_M2L','NLOOPLINE','M2L(1)')
    if candidate:squares+=call('C32','QP_M2L','NLOOPLINE','MP_M2L(1)')
    squares+=fixed('ENDIF')
    text=around(text,lambda s:s.replace(' ','').upper()=='SQSOINDEX=SQUAREDSOINDEX',after=squares)
    geometry=fixed('IF(DOING_QP) THEN')+call('R8','HELPER_DP_PL','4*NLOOPLINE','PL(0,1)')
    geometry+=call('R16','HELPER_QP_PL','4*NLOOPLINE','MP_PL(0,1)')
    geometry+=fixed('CALL EW_MASS_GROUP_SNAPSHOT(LOOPNUM,SQUAREDSOINDEX,RANK)')+fixed('ENDIF')
    text=around(text,lambda s:s.replace(' ','').upper()=='COMPLEX_MASS=.FALSE.',before=geometry)
    selection=fixed('IF(DOING_QP) CALL OBS_STATUS(\'ACTUAL_BACKEND\',MLREDUCTIONLIB(I_LIB))')
    text=around(text,lambda s:s.upper().startswith('CALL ML5_0_CHOOSE_LOOPLIB('),after=selection)
    result=fixed('IF(DOING_QP) THEN')+call('C16','GROUP_RESULT',3,'LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)')
    result+=fixed("CALL OBS_BOOLEAN('GROUP_STABLE',S(SQUAREDSOINDEX,LOOPNUM))")+fixed('ENDIF')
    text=once(text,'      ELSE\n        LOOPRES(1',result+'      ELSE\n        LOOPRES(1')
    return use_observer(text)


def adapter(text, candidate, backend):
    text=use_observer(text)
    entry=call('R16','ENTRY_PL','4*NLOOPLINE','PL(0,1)')
    entry+=call('C32' if candidate else 'C16','ENTRY_M2_QP' if candidate else 'ENTRY_M2_DP','NLOOPLINE','M2L(1)')
    text=around(text,lambda s:s.replace(' ','').upper()=='STABLE=.FALSE.',before=entry)
    ordered=call('R16','ORDERED_PL','4*NLOOPLINE','PL(0,1)')+call('C32','ORDERED_MASSES','NLOOPLINE','MASSES(1)')
    text=around(text,lambda s:s.upper().startswith('CALL ML5_0_MP_SWITCH_ORDER('),after=ordered)
    shifts='SHIFTS' if backend==9 else 'P_NRS'
    matrix='INVARIANTS' if backend==9 else 'REAL_S_MAT'
    scale='SCALE' if backend==9 else 'NRS_SCALE'
    prep=call('R16','NATIVE_SHIFTS','4*NLOOPLINE',shifts+'(0,1)')
    prep+=call('C32','KINEMATIC_S','NLOOPLINE*NLOOPLINE','S_MAT(1,1)')
    prep+=call('R16','NATIVE_INVARIANTS','NLOOPLINE*NLOOPLINE',matrix+'(1,1)')
    prep+=call('C32','NATIVE_MASSES','NLOOPLINE','MASSES(1)')
    prep+=call('R16','NATIVE_SCALE',1,'(/'+scale+'/)')
    if backend==8:prep+=call('C32','NRS_NUMERATOR','CURR_MAXCOEF','TENSORCOEFS(0)')
    else:
        prep+=fixed("CALL OBS_BOOLEAN('CHECK_PASS',CHECK_PASS)")
        prep+=fixed("CALL OBS_BOOLEAN('CACHE_ENABLED',CACHE_ENABLED)")
    status='STATUS' if backend==9 else 'NRS_STATUS'
    after=fixed('CALL OBS_NATIVE('+str(backend)+','+status+')')
    after+=fixed('IF('+status+'.EQ.0) THEN')
    if backend==9:
        for j,name in enumerate(('RAW_UV1','RAW_IR1','RAW_IR2','RAW_FINITE'),1):
            after+=call('C32',name,'NC','RAW(1,'+str(j)+')')
    else:
        after+=call('C32','NRS_RESULT',3,'NRS_RES(0)')+call('C32','NRS_R1',1,'(/R1/)')
    after+=fixed('ENDIF')
    callname='TIRUST_TENSOR_EVALUATE' if backend==9 else 'NINJARS_TENSOR_EVALUATE'
    text=around(text,lambda s:s.upper().startswith('CALL '+callname+'('),before=prep,after=after)
    return text


def instrument(sources,candidate):
    result=dict(sources)
    ct=result['CT_interface.f']
    allr=routines(ct)
    selected=[name for name in allr if re.fullmatch(r'ML5_0_LOOP_\d+(?:_\d+)?'+('_QPMASS' if candidate else ''),name)]
    assert len(selected)==7
    for name in selected:
        old=allr[name]['text'];ct=once(ct,old,helper(old,candidate))
    name='ML5_0_MP_NINJARS_LOOP'+('_QPMASS' if candidate else '')
    old=allr[name]['text'];ct=once(ct,old,adapter(old,candidate,8))
    result['CT_interface.f']=ct
    tir=result['TIRUST_interface.f']
    name='ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC'+('_QPMASS' if candidate else '')
    old=routines(tir)[name]['text'];result['TIRUST_interface.f']=once(tir,old,adapter(old,candidate,9))
    mp=result['mp_compute_loop_coefs.f']
    result['mp_compute_loop_coefs.f']=once(mp,'      CALL MP_UPDATE_AS_PARAM()\n',
        '      CALL MP_UPDATE_AS_PARAM()\n'+fixed('CALL EW_MASS_UPDATE_SNAPSHOT(PS,COMPUTE_INTEGRAND_IN_QP)'))
    return result
