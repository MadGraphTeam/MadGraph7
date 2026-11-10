"""Offline reanalysis of the one retained checked/trap experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


def canonical(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def sha(data):return hashlib.sha256(data).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def first_named_flags(text):
    current=None;first={}
    for number,line in enumerate(text.splitlines(),1):
        row=line.split()
        if row[0]=='CALL_ENTER':current={'stage':row[1],'index':int(row[2]),'point':int(row[3])}
        elif row[0]=='STATE' and row[3]=='T' and row[2] not in first:
            first[row[2]]={'line':number,'phase':row[1],'call':current,'flag_set':True,
                           'halting':row[4]=='T','supported':row[5]=='T'}
    return first


def stream(text):
    """Additional raw-record proof, independent of derived receipt JSON."""
    groups={};point=None;key=None;inputs=[];outputs={'RAW_UV1','RAW_IR1','RAW_IR2','RAW_FINITE','GROUP_RESULT'}
    typed={'R8','R16','C16','C32'}
    for line in text.splitlines():
        row=line.split();tag=row[0]
        if tag=='BEGIN':point=int(row[1]);inputs.append(' '.join(row[:-1]))
        elif tag=='GROUP':
            key=(point,int(row[2]));groups[key]={'key':list(map(int,row[2:8])),'native':None,'outputs':[],'errors':[],'stable':None,'check':None,'cache':None}
            inputs.append(' '.join(row[:-1]))
        elif tag=='END':inputs.append(' '.join(row[:-1]))
        elif tag in typed:
            if row[1] in outputs:groups[key]['outputs'].append(row)
            if row[1] not in outputs|{'ACCURACY','AMPLITUDE'} or row[1]=='AMPLITUDE' and int(row[2]) in (1,5):
                inputs.append(' '.join(row))
        elif tag=='NATIVE':groups[key]['native']={'backend':int(row[1]),'status':int(row[2]),'flags':row[3]}
        elif tag=='ERROR_BYTES':groups[key]['errors'].append(bytes(int(x,16) for x in row[1:]).decode())
        elif tag in ('INTEGER','LOGICAL'):
            if row[1] in {'MODEL_UPDATE','GENUINE_QP_INTEGRAND','ACTUAL_BACKEND','CHECK_PASS','CACHE_ENABLED'}:inputs.append(' '.join(row[:-1]))
            if row[1]=='GROUP_STABLE':groups[key]['stable']=row[2]
            if row[1]=='CHECK_PASS':groups[key]['check']=row[2]
            if row[1]=='CACHE_ENABLED':groups[key]['cache']=row[2]
    return {'groups':groups,'input_Born_record_count':len(inputs),'input_Born_sha256':sha(('\n'.join(inputs)+'\n').encode())}


def served_proof(old,new):
    retained=[];new_retained=[];failures=[];transitions=[];passes=[]
    for key,group in sorted(old['groups'].items()):
        right=new['groups'][key]
        if group['native']['status']==0:
            retained.append({'point':key[0],'ordinal':key[1],'rows':group['outputs']})
            new_retained.append({'point':key[0],'ordinal':key[1],'rows':right['outputs']})
            if group['outputs']!=right['outputs'] or group['stable']!=right['stable'] or right['native']['status']!=0:failures.append(list(key))
        else:
            transitions.append({'point':key[0],'ordinal':key[1],'loop_group':group['key'][1],
                                'mode':group['key'][3],'old_status':group['native']['status'],'new_status':right['native']['status'],
                                'old_errors':group['errors'],'new_errors':right['errors']})
    for start in (1,201,401):
        pair=[]
        for data in (old,new):
            rows=[data['groups'][(2,index)] for index in range(start,start+200)]
            pair.append({'mode':rows[0]['key'][3],'calls':len(rows),
                         'served':sum(g['native']['status']==0 for g in rows),
                         'refused':sum(g['native']['status']!=0 for g in rows),
                         'refused_loop_groups':[g['key'][1] for g in rows if g['native']['status']!=0],
                         'check_pass_true':sum(g['check']=='T' for g in rows),'cache_enabled_true':sum(g['cache']=='T' for g in rows)})
        passes.append({'original':pair[0],'adaptive':pair[1]})
    encode=lambda rows:json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    return {'formerly_served_calls':len(retained),'formerly_served_outputs_preserved':not failures,
            'original_served_records_sha256':sha(encode(retained)),'adaptive_served_records_sha256':sha(encode(new_retained)),
            'failures':failures,'refusal_transitions':transitions,'passes':passes,
            'selected_input_Born_records':old['input_Born_record_count'],
            'selected_input_Born_bits_match':old['input_Born_sha256']==new['input_Born_sha256'],
            'original_input_Born_sha256':old['input_Born_sha256'],'adaptive_input_Born_sha256':new['input_Born_sha256']}


def reanalyze(blobs):
    record=json.loads(blobs['run1/execution.json']);workers={};captures={}
    with tempfile.TemporaryDirectory(prefix='checked-trap-offline-') as directory:
        base=Path(directory)/'portable/repository/docs/benchmarkfolder/experiments/packet'
        for name,data in blobs.items():
            path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        control=load('checked_trap_retained_control',base/'source/control_log.py')
        parser=load('checked_trap_retained_parser',base/'source/parser.py')
        controller=load('checked_trap_retained_controller',base/'source/controller.py')
        for name,row in record['workers'].items():
            prefix='run1/'+name+'/';raw=blobs[prefix+'checked-control.log'].decode()
            parsed=control.parse(base/(prefix+'checked-control.log'),allow_partial=row['exit_code']!=0)
            assessment=control.assess(parsed,exit_code=row['exit_code'])
            assert assessment==row['control_assessment'],name
            assert canonical(parsed)==json.loads(blobs[prefix+'named-control.json']),name
            lines=blobs[prefix+'stdout.log'].decode().splitlines();omissions=[i for i,line in enumerate(lines,1) if '##W03' in line]
            result={'exit_code':row['exit_code'],'kind':row['kind'],'control_assessment':assessment,
                    'first_named_set_flags':first_named_flags(raw),'W03_count':len(omissions),'W03_lines':omissions,
                    'calls':parsed['calls'],'mode':parsed['mode'],'policy':parsed['policy'],
                    'named_control_complete':parsed['complete'],'original_offline_failure':row.get('offline_assessment_failure')}
            if row['kind']=='physical':
                if prefix+'mass-observer.hex' in blobs:
                    capture=parser.parse(base/(prefix+'mass-observer.hex'),allow_partial=row['exit_code']!=0);captures[name]=capture
                    summary=parser.summarize(capture)
                    assert canonical(summary)==json.loads(blobs[prefix+'journal-summary.json'])
                    checked=parser.qualify_checked(capture,parsed['observed'],omissions=len(omissions))
                    checked['IEEE_policy_worker_completed']=assessment['physical_worker_completed_with_requested_IEEE_policy']
                    assert checked==row['checked_assessment'];result.update(journal_summary=summary,checked_assessment=checked)
                else:
                    result['journal_disposition']='Absent: worker trapped during warmup before observed point capture; preserved failure, no checked-physics qualification'
            workers[name]=result
        comparisons={}
        for left,right in [('original-masked','adaptive-masked'),('original-trapped','adaptive-trapped'),('adaptive-masked','adaptive-trapped')]:
            if left in captures and right in captures:comparisons[left+'__'+right]=parser.compare_inputs_and_born(captures[left],captures[right])
        parity={'outputs_exact':False,'reason':'Missing candidate policy capture'}
        if 'adaptive-masked' in captures and 'adaptive-trapped' in captures:
            parity=controller.compare_policy_outputs(captures['adaptive-masked'],captures['adaptive-trapped'])
        comparisons['adaptive_masked_trapped_output_parity']=parity
        assert canonical(comparisons)==json.loads(blobs['run1/comparisons.json'])
    proof=served_proof(stream(blobs['run1/original-masked/mass-observer.hex'].decode()),stream(blobs['run1/adaptive-masked/mass-observer.hex'].decode()))
    assert proof['formerly_served_calls']==489 and proof['formerly_served_outputs_preserved'] and proof['selected_input_Born_bits_match']
    assert proof['original_served_records_sha256']=='20a33da98d358622564c86190f9df075ec3102aefc5601040817393eb1ef375e'
    assert proof['selected_input_Born_records']==101086 and proof['original_input_Born_sha256']=='f436936f4c3479e685a0d7dd44e79d9a3ca12cecd1ce8b8eeb5ddbec13fdd209'
    return {'schema':'checked-trap-one-experiment-offline-outcome-v1','workers':workers,'raw_record_proof':proof,
            'masked_adaptive_checked_outputs_qualified':workers['adaptive-masked']['checked_assessment']['checked_outputs_qualified'],
            'masked_adaptive_QP_three_pass_coverage':workers['adaptive-masked']['checked_assessment']['qp_coverage_observed'],
            'combined_checked_trap_coverage_qualified':False,
            'reason_combined_gate_failed':'Both trapped physical workers SIGFPE during threshold warmup2, before observed journals; candidate masked/trapped parity unavailable',
            'exact_hardware_exception_at_trap_identified':False,'per_call_working_precision_available':False,
            'independent_physical_accuracy_qualified':False,'performance_qualified':False,
            'native_or_loader_executed_by_reanalysis':False,'original_records_modified':False}
