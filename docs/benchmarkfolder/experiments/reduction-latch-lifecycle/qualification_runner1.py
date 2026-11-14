#!/usr/bin/env python3
"""One admitted private matrix relink + four plain SM parity workers; no retries."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source-proposal1'
OLD=HERE.parent/'qcd-n3-genuine-qp-diagnostic/source-proposal5'
SOURCE_SHA='c9a61e13b12c8566579d760268b739c7ec83c17198bf686f1afd9b0ef37b34c5'
OLD_SHA='efc8c5b5c7a775fd3b4215171bd0d989bb5d6791710ceeb65d58816ba76b03e5'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)
def dump(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def payloads(root,expected):
    index=root/'source-index.json';need(digest(index)==expected,'Source seal changed: '+str(root))
    value=json.loads(index.read_text())
    for name,entry in value['files'].items():
        path=root/name;need(path.is_file() and not path.is_symlink(),'Source payload kind changed')
        need(path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],'Source payload changed: '+name)

def verify():
    seal=json.loads((HERE/'qualification-runner1.index.json').read_text())
    need(digest(__file__)==seal['runner_sha256'],'Runner bytes changed')
    payloads(SOURCE,SOURCE_SHA);payloads(OLD,OLD_SHA)
    binding=json.loads((SOURCE/'bindings.json').read_text())
    for name,expected in binding['dependencies'].items():need(digest(name)==expected,'Bound input changed: '+name)
    plan=json.loads((SOURCE/'sm-replay-plan.json').read_text())
    need(len(plan['compile_commands'])==2 and len(plan['workers'])==4,'Admission shape differs')
    need([(x['arm'],x['backend'],x['mode'],x['observed']) for x in plan['workers']]==[('original',9,4,0),('candidate',9,4,0),('original',9,5,0),('candidate',9,5,0)],'Worker order differs')
    return binding,plan

# Verify indexed helper bytes before importing their existing implementation.
def helpers():
    payloads(OLD,OLD_SHA)
    sys.path.insert(0,str(OLD))
    import run as legacy
    return legacy

def run():
    legacy=helpers()
    with legacy.numerical_lock():
        started=time.monotonic();deadline=started+600
        binding,plan=verify();work=Path(plan['working_directory']);work.mkdir(parents=True,exist_ok=False)
        result={'passed':False,'source_index_sha256':SOURCE_SHA,'runner_sha256':digest(__file__),
                'original15_runtime_providers':True,'production_mutation':False,
                'stage_cap_seconds':600,'public_call_cap':48,'native_workers_admitted':4,
                'workers_completed':0,'scope':'Three-point byte parity; status120/accuracy-1 unchecked, no accuracy or performance acceptance.'}
        workers=[]
        try:
            # This is the exact inherited build, with a fresh private matrix object only.
            _,proof,original=legacy.inherited_build(binding)
            manifest=json.loads((SOURCE/'donor-closure.json').read_text())
            dump(work/'runtime-proof.json',legacy.donor_closure.create(work/'closed-runtime',manifest,binding['runtime']))
            for number,command in enumerate(plan['compile_commands'],1):
                verify()
                legacy.run_command(command['argv'],work,'compile'+str(number),binding=binding,deadline=deadline,cap=120)
                verify()
            candidate=Path(plan['candidate_executable']);candidate_sha=digest(candidate)
            need(legacy.exec_target.interpreter(candidate.read_bytes())==binding['target_interpreter'],'Candidate interpreter changed')
            build={'original_executable':str(original),'original_sha256':proof['executable_sha256'],
                   'candidate_executable':str(candidate),'candidate_sha256':candidate_sha,
                   'matrix_object_sha256':digest(work/'loop_matrix.o'),'link_map_sha256':digest(work/'link.map'),
                   'compile_commands':2,'original_objects':proof['original_objects'],'other_objects':proof['objects']}
            dump(work/'build.json',build)
            for spec in plan['workers']:
                folder=work/spec['id'];folder.mkdir()
                inputs=legacy.fixture_inputs(folder,9);dump(folder/'inputs.json',inputs)
                need((folder/'nominal/param_card.dat').is_file() and not (folder/'nominal/param_card.dat').is_symlink(),'Private SETPARA card missing')
                exe,checksum=(original,proof['executable_sha256']) if spec['arm']=='original' else (candidate,candidate_sha)
                try:
                    verify();need(digest(exe)==checksum,'Executable changed before trace')
                    dump(folder/'runtime-pretrace.json',legacy.donor_closure.verify(work/'closed-runtime',manifest,binding['runtime']))
                    legacy.run_command(legacy.target_command(binding,exe,checksum,[],trace=True),folder,'loader',binding=binding,deadline=deadline,cap=30)
                    legacy.validate_loader((folder/'loader.log').read_text(),binding)
                    verify();need(digest(exe)==checksum,'Executable changed before worker')
                    dump(folder/'runtime-preworker.json',legacy.donor_closure.verify(work/'closed-runtime',manifest,binding['runtime']))
                    legacy.run_command(legacy.target_command(binding,exe,checksum,[9,spec['mode'],0]),folder,'worker',binding=binding,deadline=deadline,cap=90,native=True,exe=exe)
                    parsed=legacy.analyze.parse_worker((folder/'trace.txt').read_text(),(folder/'worker.log').read_text(),9,spec['mode'],0,legacy.point_bytes())
                    need(parsed['public_calls']<=12,'Per-worker public call bound exceeded')
                    dump(folder/'parsed.json',parsed);workers.append({'arm':spec['arm'],**parsed})
                    result['workers_completed']=len(workers)
                finally:
                    # Preserve failures and all private output receipts; inputs/providers must remain exact.
                    legacy.verify_files(inputs);verify();need(digest(exe)==checksum,'Executable changed after worker')
                    legacy.donor_closure.verify(work/'closed-runtime',manifest,binding['runtime'])
                    dump(folder/'outputs.json',{str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file() and p.name!='outputs.json'})
            need(len(workers)==4,'Missing planned workers')
            total=sum(w['public_calls'] for w in workers);need(total<=48,'Total public call bound exceeded')
            checks=[]
            for mode in (4,5):
                original_worker=next(w for w in workers if w['arm']=='original' and w['mode']==mode)
                candidate_worker=next(w for w in workers if w['arm']=='candidate' and w['mode']==mode)
                equal=original_worker['public_raw']==candidate_worker['public_raw']
                checks.append({'mode':mode,'public_input_output_status_IEEE_byte_equal':equal,'points':[1,93,39]})
                need(equal,'Original/candidate public byte parity failed: '+str(mode))
            result.update(passed=True,all_four_workers=True,public_calls=total,diagnostic_calls=12,answer_order_columns=24,comparisons=checks,candidate_executable_sha256=candidate_sha)
        except BaseException as error:
            result['failure']={'type':type(error).__name__,'message':str(error)}
            raise
        finally:
            result['elapsed_seconds_including_final_guards']=time.monotonic()-started
            result['stage_within_cap']=result['elapsed_seconds_including_final_guards']<=600
            if not result['stage_within_cap']:result['passed']=False
            dump(work/'terminal.json',result)
        need(result['stage_within_cap'],'Inclusive stage deadline exceeded')
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('verify','run'));args=parser.parse_args()
    if args.stage=='verify':
        binding,plan=verify();print(json.dumps({'passed':True,'dependencies':len(binding['dependencies']),'workers':len(plan['workers']),'native_executed':False},sort_keys=True));return
    with helpers().interruptions():print(json.dumps(run(),sort_keys=True,indent=2))
if __name__=='__main__':main()
