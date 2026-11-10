#!/usr/bin/env python3
"""Independent sealed numerical replay, including a confined /tmp relocation.

Only the selected sealed package is read from the original workspace. No
original generated work tree, live process, compiler, loader or reducer is used.
"""
import __future__
import argparse
import ast
import builtins
import ctypes
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent/'numerical-retention-1/sealed-1'
EXPECTED='8f5f7110cc101dc79e8fe75d5dd1ef342e1e72797986a5a5026680e28d4496be'
WORKSPACE=HERE.parents[4]
def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def write(name,value): (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

STATE={'active':False,'negative':False,'allowed':(), 'operations':{},'blocked':[], 'phase':None,
       'pending_open':None,'fd_paths':{}}
def inside(path,roots): return any(path==p or path.startswith(p+os.sep) for p in roots)
def pathcheck(path,operation,metadata=False):
    if not STATE['active'] or isinstance(path,int) or path is None: return
    path=os.path.abspath(os.fsdecode(path))
    roots=STATE['allowed']
    allowed=inside(path,roots) or (metadata and any(p.startswith(path.rstrip(os.sep)+os.sep) for p in roots))
    if not allowed:
        STATE['blocked'].append({'phase':STATE['phase'],'operation':operation,'path':path,'intentional_negative':STATE['negative']})
        raise PermissionError('Replay access denied: '+operation+' '+path)
    STATE['operations'][operation]=STATE['operations'].get(operation,0)+1

def audit_event(event,args):
    if not STATE['active']: return
    if event=='open': pathcheck(STATE['pending_open'] or args[0],'open')
    elif event in ('os.listdir','os.scandir'): pathcheck(args[0],event)
    elif event in ('subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec','ctypes.dlopen'):
        STATE['blocked'].append({'phase':STATE['phase'],'operation':event,'intentional_negative':STATE['negative']})
        raise PermissionError('Replay operation denied: '+event)

sys.addaudithook(audit_event)

@contextmanager
def confined(package,scratch,phase):
    saved={name:getattr(os,name) for name in ('stat','lstat','readlink','access','open','close')}
    old_temp=tempfile.tempdir
    STATE.update(active=True,negative=False,allowed=(str(package),str(scratch)),phase=phase,operations={},blocked=[],fd_paths={},pending_open=None)
    def absolute(path,kwargs):
        fd=kwargs.get('dir_fd')
        if fd is None or os.path.isabs(path): return path
        if fd not in STATE['fd_paths']: raise PermissionError('Unknown directory file descriptor')
        return os.path.join(STATE['fd_paths'][fd],path)
    def wrapper(name):
        original=saved[name]
        def invoke(path,*args,**kwargs):
            pathcheck(absolute(path,kwargs),name,metadata=True)
            return original(path,*args,**kwargs)
        return invoke
    def opening(path,*args,**kwargs):
        full=absolute(path,kwargs);pathcheck(full,'os.open')
        STATE['pending_open']=full
        try:
            fd=saved['open'](path,*args,**kwargs)
            STATE['fd_paths'][fd]=os.path.abspath(os.fsdecode(full));return fd
        finally: STATE['pending_open']=None
    def closing(fd):
        try: return saved['close'](fd)
        finally: STATE['fd_paths'].pop(fd,None)
    for name in ('stat','lstat','readlink','access'): setattr(os,name,wrapper(name))
    os.open=opening;os.close=closing
    tempfile.tempdir=str(scratch)
    try: yield
    finally:
        STATE['active']=False;tempfile.tempdir=old_temp
        for name,original in saved.items(): setattr(os,name,original)

def module(package,name):
    spec=importlib.util.spec_from_file_location(name,package/'retain.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def inventory(package):
    indexraw=(package/'index.json').read_bytes();assert sha(indexraw)==EXPECTED
    index=json.loads(indexraw)
    actual={}
    for p in package.rglob('*'):
        assert not p.is_symlink()
        if p.is_file() and p.name!='index.json':
            raw=p.read_bytes();actual[p.relative_to(package).as_posix()]={'sha256':sha(raw),'bytes':len(raw)}
    assert actual==index['files']
    assert index['closed'] is True and index['full_matrix_closed'] is True
    assert index['performance_acceptance_eligible'] is False and index['overall_goal_qualified'] is False
    return index,actual

def negatives():
    checks={
      'workspace_file':lambda:Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust/tools/tirust_benchmark.py').read_bytes(),
      'process_file':lambda:Path('/proc/self/maps').read_bytes(),
      'subprocess':lambda:subprocess.Popen(['/bin/true']),
      'dynamic_library':lambda:ctypes.CDLL(None),
      'workspace_metadata':lambda:os.stat('/shared/external/madloop_tirust/madgraph7_madloop_tirust/.benchmark-work/forbidden'),
    }
    STATE['negative']=True
    try:
        for name,call in checks.items():
            try: call()
            except PermissionError: pass
            else: raise AssertionError('Guard negative control failed: '+name)
    finally: STATE['negative']=False
    return list(checks)

def pass_once(package,scratch,phase):
    started=time.monotonic()
    with confined(package,scratch,phase):
        controls=negatives()
        index,outer=inventory(package)
        ret=module(package,'independent_retention_'+phase)
        loaded,store=ret.load(package,EXPECTED)
        assert loaded==index and len(store.members)==1221
        members={name:{'sha256':sha(raw),'bytes':len(raw)} for name,raw in store.members.items()}
        assert members==index['retained_files']
        assert all(not name.endswith(('.o','.a','.so','.mod','.pyc')) and not raw.startswith(b'\x7fELF') and b'\0' not in raw for name,raw in store.members.items())
        consoles={name:raw for name,raw in store.members.items() if '/experiments/numerical/' in name and name.endswith('-output.txt')}
        assert len(consoles)==66
        warnings={name:[{'line':i,'text':line} for i,line in enumerate(raw.decode().splitlines(),1) if re.search(r'##W03\b',line,re.I)] for name,raw in consoles.items()}
        scan={'file_count':66,'pattern':r'##W03\b','case_insensitive':True,
          'selection':"'/experiments/numerical/' in name and name.endswith('-output.txt')",
          'total_warning_lines':sum(len(rows) for rows in warnings.values()),
          'files':{name:{'sha256':sha(raw),'bytes':len(raw),'matches':warnings[name]} for name,raw in sorted(consoles.items())}}
        assert scan['total_warning_lines']==0
        replay=ret.replay_completed(package,EXPECTED)
        summary=json.loads((package/'summary.json').read_bytes())
        assert replay==summary['supervisor_replay']
        assert len(replay)==11 and set(replay)==set(index['case_ids'])
        assert summary['full_matrix_regression_passed'] is True
        assert all(r['status']=='passed' and r['returncode']==0 and r['raw_replay_passed'] is True and len(r['lanes'])==6 for r in replay.values())
        assert all(l['status']=='complete' and l['numerical_agreement_passed'] is True and l['matched_saved_points']==100 for r in replay.values() for l in r['lanes'])
        sentinel={'/portable-checkpoint','/portable-checkpoint/docs','/portable-checkpoint/docs/benchmarkfolder'}
        virtual=[x for x in STATE['blocked'] if not x['intentional_negative'] and x['operation']=='lstat' and x.get('path') in sentinel]
        unexpected=[x for x in STATE['blocked'] if not x['intentional_negative'] and x not in virtual]
        if unexpected: print(json.dumps({'unexpected_denials':unexpected},sort_keys=True),flush=True)
        assert not unexpected
        operations=dict(STATE['operations']);blocked=list(STATE['blocked'])
    return {'phase':phase,'passed':True,'index_sha256':EXPECTED,'package_outer_files_including_index':len(outer)+1,
      'retained_members':len(members),'retained_bytes':sum(x['bytes'] for x in members.values()),
      'retained_inventory_sha256':sha(canonical(members)),'replay_sha256':sha(canonical(replay)),
      'summary_sha256':sha((package/'summary.json').read_bytes()),'case_count':len(replay),'lane_count':sum(len(r['lanes']) for r in replay.values()),
      'saved_point_lane_observations':sum(l['matched_saved_points'] for r in replay.values() for l in r['lanes']),
      'negative_controls':controls,'blocked_operations':blocked,'allowed_operation_counts':operations,
      'blocked_virtual_sentinel_metadata_probes':len(virtual),'unexpected_denied_operations':0,
      'runtime_console_scan_sha256':sha(canonical(scan)),'elapsed_seconds':time.monotonic()-started},summary,replay,scan

def case_summary(summary,replay):
    numerical={r['id']:r for r in summary['numerical']['cases']};rows=[]
    for case_id in summary['case_ids']:
        row=numerical[case_id];assert row['regression_pass'] is True and row['all_six_complete'] is True
        lanes=[]
        for lane in row['lanes']:
            actual=next(x for x in replay[case_id]['lanes'] if (x['source'],x['backend'])==(lane['source'],lane['backend']))
            assert lane['points']==actual['matched_saved_points']==100
            lanes.append({key:lane[key] for key in ('source','backend','points','regression_pass','ordinary_target_pass','accepted_dp_count','accepted_qp_count','status_counts')}
                         |{'ordinary_target_miss_rows':len(lane['ordinary_target_misses'])})
        rows.append({'case':case_id,'role':row['role'],'process':row['process'],'model':row['model'],
          'regression_pass':row['regression_pass'],'ordinary_target_pass':row['ordinary_target_pass'],
          'results_sha256':replay[case_id]['results_sha256'],'lanes':lanes})
    return {'registered_regression_tolerance':3e-6,'ordinary_target':1e-8,
      'metric':'abs(a-b)/abs(a+b), with the frozen negligible-pole criterion; no alternate metric substituted',
      'case_count':11,'lane_count':66,'points_per_lane':100,'rows':rows,
      'tighter_target_miss_cases':[r['case'] for r in rows if not r['ordinary_target_pass']],
      'limitations':['Raw-point replay validates retained evidence; no new scientific evaluation was performed.',
        'FPE flags and internal retry counts are not observed in this campaign.',
        'Accepted precision labels do not reveal internal rescue attempts.',
        'No timing, stress, or overall-goal acceptance follows from this ordinary-matrix replay.']}

def main():
    assert not (HERE/'proof.json').exists(),'Proof already exists; do not overwrite'
    sys.dont_write_bytecode=True
    original_index,original_files=inventory(PACKAGE)
    with tempfile.TemporaryDirectory(prefix='tirust-independent-retention-') as temp:
        temp=Path(temp);scratch=temp/'scratch';scratch.mkdir();relocated=temp/'relocated'
        shutil.copytree(PACKAGE,relocated)
        local,summary,replay,scan=pass_once(PACKAGE,scratch,'selected_package_only')
        shifted,other,replayed,other_scan=pass_once(relocated,scratch,'relocated_no_workspace')
        assert summary==other and replay==replayed and scan==other_scan
        assert local['replay_sha256']==shifted['replay_sha256']
        assert inventory(PACKAGE)==(original_index,original_files)
        proof={'passed':True,'expected_package_index_sha256':EXPECTED,
          'package_path':'docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-retention-1/sealed-1',
          'source_hashes':{n:original_files[n]['sha256'] for n in ('retain.py','campaign_report_numerical.py','numerical_supervisor.py','source-index.json')},
          'passes':[local,shifted],'relocated_replay_exactly_equal':True,'sealed_package_unchanged':True,
          'native_or_process_execution':False,'original_generated_work_reads':False,
          'performance_acceptance_eligible':False,'overall_goal_qualified':False}
        write('proof.json',proof);write('cases.json',case_summary(summary,replay));write('runtime-console-scan.json',scan)
    print(json.dumps({'passed':True,'index_sha256':EXPECTED,'cases':11,'lanes':66,'members':1221,
      'relocated_exact_replay':True,'original_work_reads':False,'native_execution':False},sort_keys=True))

if __name__=='__main__': main()
