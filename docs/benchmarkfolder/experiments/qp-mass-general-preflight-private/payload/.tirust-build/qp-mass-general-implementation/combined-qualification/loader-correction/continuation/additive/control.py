#!/usr/bin/env python3
"""Additive original-QP control: one link, then two separately authorized cases."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode=True
RESULTS=Path(__file__).resolve().parent
PARENT=RESULTS.parents[2]
HERE=PARENT/'current-original-qp'
ROOT=PARENT.parents[2]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def prior_observations(base,spec):
    """Bind already successful current guarded workers; never run them here."""
    path=PARENT/'loader-correction/continuation/run.json'
    report=json.loads(path.read_text())
    correction=json.loads((RESULTS/'prepared.json').read_text())
    base.require(report['correction_manifest_sha256']==correction['mass_correction_prepared_sha256'],
                 'Wrong corrected parent binding')
    base.require(report['scientific_scope_unchanged'] is True, 'Changed parent scientific scope')
    base.require(report['state']=='complete' and report['input_hashes_after']=='unchanged',
                 'Complete frozen guarded qualification is required first')
    base.require(report['manifest_sha256']==spec['parent_manifest_sha256'],'Wrong parent run')
    jobs={j['name']:j for j in report['jobs']}
    base.require(len(jobs)==len(report['jobs']),'Duplicate parent job')
    expected=json.loads((PARENT/'prepared.json').read_text())
    pins={str(path.relative_to(ROOT)):sha(path)}
    for stage in expected['stages']:
        validation=jobs[stage['validation']['name']]
        base.require(validation['returncode']==0 and not validation['timed_out'] and not validation.get('error'),
                     'Parent stage parser did not pass')
    for job in expected['stages'][-1]['jobs']:
        event=jobs[job['name']]
        base.require(event['returncode']==0 and not event['timed_out'] and not event.get('error')
                     and not event['changed_inputs'] and event['expected_output_closure_matched'],
                     'Parent QP worker incomplete')
        base.require(event['command'][1:3]==['-c','8'] and event['command'][3:]==job['argv'],
                     'Parent QP command differs')
        for name,digest in event['output_files'].items():
            observed=ROOT/job['cwd']/name
            base.require(sha(observed)==digest,'Changed guarded observation')
            pins[str(observed.relative_to(ROOT))]=digest
    return pins

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('action',choices=('verify','build','run'))
    args=cli.parse_args()
    assert args.action in ('verify','run'), 'No build is authorized or needed for loader correction'
    assert Path.cwd().resolve()==ROOT
    correction=json.loads((RESULTS/'prepared.json').read_text())
    for name,digest in correction['inputs'].items():
        assert sha(ROOT/name)==digest, name
    spec=json.loads((HERE/'prepared.json').read_text())
    assert sha(PARENT/'control.py')==spec['shared_controller_sha256']
    base=load('original_qp_shared_controller',PARENT/'control.py')
    base.HERE=HERE
    def save_corrected(name,value):
        target=RESULTS/name; temporary=RESULTS/(name+'.tmp')
        with temporary.open('x') as stream:
            json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
        temporary.replace(target)
    base.save=save_corrected
    verify_original=base.verify_inputs
    def verify_corrected(value):
        verify_original(value)
        for name,digest in correction['inputs'].items():
            base.require(sha(ROOT/name)==digest,'Changed additive correction/failure input: '+name)
    base.verify_inputs=verify_corrected
    assert sha(base.OLD)==base.OLD_SHA
    cleanup=load('original_qp_owned_cleanup',base.OLD)
    base.require(spec['cpu']==8 and spec['jobs']==1 and spec['build_seconds_max']==30,
                 'Wrong additive resource scope')
    base.require(len(spec['build_jobs'])==1 and len(spec['stages'])==1
                 and len(spec['stages'][0]['jobs'])==2,'Wrong additive commands')
    base.require(8 in os.sched_getaffinity(0),'CPU8 unavailable')
    base.verify_inputs(spec)
    if args.action=='verify':
        print(json.dumps({'verified_inputs':len(spec['inputs']),'native_launched':False}));return
    base.require(not (HERE/(args.action+'.json')).exists(),'Original stage already attempted; no retry')
    base.require(not (RESULTS/(args.action+'.json')).exists(),'Corrected stage already attempted; no retry')
    prior=prior_observations(base,spec) if args.action=='run' else {}
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    report={'state':'running','action':args.action,'manifest_sha256':sha(HERE/'prepared.json'),
        'controller_sha256':sha(__file__),'shared_controller_sha256':spec['shared_controller_sha256'],
        'correction_manifest_sha256':sha(RESULTS/'prepared.json'),
        'prior_observation_sha256':prior,'jobs':[],'started_unix':time.time(),
        'inherited_affinity':sorted(os.sched_getaffinity(0)),
        'loadavg_before':Path('/proc/loadavg').read_text().strip(),'automatic_retry':False}
    base.save(args.action+'.json',report)
    try:
        with cleanup.signal_cleanup():
            (base.build if args.action=='build' else base.run)(spec,report,cleanup)
        report['state']='complete'
    except BaseException as error:
        report['state']='failed';report['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        try:
            base.verify_inputs(spec)
            for name,digest in prior.items():base.require(sha(ROOT/name)==digest,'Guarded observation changed')
            report['input_hashes_after']='unchanged'
        except BaseException as error:
            report['state']='failed';report['input_hashes_after']='FAILED: '+str(error)
        report['finished_unix']=time.time();report['loadavg_after']=Path('/proc/loadavg').read_text().strip()
        base.save(args.action+'.json',report)
    print(json.dumps({'state':report['state'],'jobs':len(report['jobs']),'error':report.get('error')}))
    sys.exit(0 if report['state']=='complete' else 1)

if __name__=='__main__':main()
