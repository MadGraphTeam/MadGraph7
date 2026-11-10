#!/usr/bin/env python3
"""Portable raw-log replay for real-Z controls; no compiler/native execution."""
import argparse,gzip,hashlib,importlib.util,json,math,re,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK='.tirust-build/born-effective-runtime/native89-controller2'
def need(ok,why):
    if not ok:raise ValueError(why)
def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text())

def verify(unfrozen=False):
    outer={}
    for path in sorted(HERE.rglob('*')):
        need(not path.is_symlink(),'Symlink object')
        if not path.is_file() or path.name=='evidence-index.json':continue
        need('__pycache__' not in path.parts and path.suffix not in ('.pyc','.o','.a','.so','.mod','.pkl'),'Native/cache payload')
        data=path.read_bytes();need(not data.startswith((b'\x7fELF',b'!<arch>\n')),'Native payload')
        if path.suffix=='.gz':json.loads(gzip.decompress(data).decode('utf-8'))
        else:data.decode('utf-8');need(b'\0' not in data,'NUL in text')
        outer[str(path.relative_to(HERE))]={'sha256':digest(data),'bytes':len(data)}
    if (HERE/'evidence-index.json').exists():need(read(HERE/'evidence-index.json')['files']==outer,'Outer inventory differs')
    else:need(unfrozen,'Missing outer index')
    compressed=(HERE/'payload.json.gz').read_bytes();raw=gzip.decompress(compressed);payload=json.loads(raw)
    files=payload['files'];blobs=payload['blobs'];inventory=read(HERE/'payload-inventory.json')
    need(inventory['files']==files and inventory['gzip_sha256']==digest(compressed),'Payload inventory differs')
    need(inventory['unique_text_contents']==len(blobs) and inventory['uncompressed_bytes']==len(raw),'Payload dimensions differ')
    for name,rec in files.items():
        data=blobs[rec['sha256']].encode('utf-8')
        need(digest(data)==rec['sha256'] and len(data)==rec['bytes'],'Text changed: '+name)
        need(b'\0' not in data and not data.startswith((b'\x7fELF',b'!<arch>\n')),'Native blob')
    def key(name):return name.split('/madgraph7_madloop_tirust/',1)[1] if '/madgraph7_madloop_tirust/' in name else name
    def text(name):return blobs[files[key(name)]['sha256']]
    def doc(name):return json.loads(text(name))
    other=read(HERE/'other-bound-identities.json')['files']
    def identity(name):return (files.get(key(name)) or other.get(key(name),{})).get('sha256')
    for name,rec in read(HERE/'direct-copies.json').items():need(digest((HERE/name).read_bytes())==rec['sha256']==identity(rec['original']),'Direct copy differs')
    source_index=doc(WORK+'/source-index.json')
    need(identity(WORK+'/source-index.json')=='55af12e109954110be3326e1a0d52dd28cb51bdf419585e2c12ffbd2fd1c03b4','Wrong reviewed source index')
    for name,rec in source_index['files'].items():need(files[WORK+'/'+name]==rec,'Reviewed source changed: '+name)
    prepared=doc(WORK+'/prepared.json')
    need(identity(WORK+'/prepared.json')=='de50e4a9c9599080bd9c6bf9f70f36b8f38e2e90390dda609e6cecd0728e4ff8','Wrong prepared workload')
    need(identity(WORK+'/control.py')=='23a5351646770306fb031f4459297a11005c752cbdf47cb565765e8143039867','Wrong controller')
    old=doc(WORK+'/history/prepared3dc89.json')
    need(old['commands']==prepared['commands'] and old['input_bindings']==prepared['input_bindings'],'Successor changes physical/native workload')
    for name,want in (prepared['source_bindings']|prepared['input_bindings']).items():need(identity(name)==want,'Prepared binding missing/different: '+name)
    commands={c['id']:c for c in prepared['commands']};need(len(commands)==24,'Wrong prepared case count')
    need(all(c['case']=='real_uux_z' and c['backend'] in (8,9) and c['arm'] in ('original','mass','graph','full') for c in commands.values()),'Scope expanded')
    observed={};counts={};maxres={};full_exact=0
    with tempfile.TemporaryDirectory(prefix='born89-portable-') as temp:
        work=Path(temp)/'source/a/b';work.mkdir(parents=True)
        path=work/'control.py';path.write_text(text(WORK+'/control.py'))
        spec=importlib.util.spec_from_file_location('retained_native89_parser',path);controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
        for stage,expected_runs,expected_comparisons in (('ordinary',8,6),('fixed-qp',16,12)):
            prefix=WORK+'/execution-'+stage;summary=doc(prefix+'/summary.json')
            need(summary['state']=='complete' and not summary['changed_bound_files'],'Stage incomplete/changed')
            need(summary['prepared_sha256']==identity(WORK+'/prepared.json'),'Stage admission differs')
            need(len(summary['runs'])==expected_runs and len({x['id'] for x in summary['runs']})==expected_runs,'Run coverage differs')
            need(summary['cpu']==14 and summary['elapsed_seconds']<=summary['per_stage_seconds']==300,'Stage bounds differ')
            for run in summary['runs']:
                command=commands[run['id']];base=prefix+'/runs/'+run['id']
                need(command['stage']==stage and run['passed'] and run['maps_passed'],'Failed/wrong-stage run')
                event=doc(base+'/execution.json');mapping=doc(base+'/mapping.json');runtime=doc(base+'/runtime.json')
                need(event==run['command'] and event['returncode']==0 and not event['timed_out'],'Native event differs/fails')
                need(event['command'][1:3]==['-c','14'] and event['command'][-1]==command['argv'][0],'Native command differs')
                need(mapping['cpu_allowed_list']=='14' and mapping['executable']==command['argv'][0],'Actual native mapping differs')
                need(digest(mapping['maps_text'].encode())==mapping['maps_sha256']==run['maps_sha256']==identity(base+'/process-maps.txt'),'Mapping bytes differ')
                for name,rec in runtime.items():
                    if name!=mapping['executable']:need(mapping['mapped_files_sha256_after_exit'][name]==rec['sha256'],'Mapped file changed')
                need(set(mapping['mapped_elf_paths'])==set(runtime),'Actual ELF closure differs')
                need(mapping['expected_runtime_sha256']=={n:r['sha256'] for n,r in runtime.items() if n!=mapping['executable']},'Declared/actual runtime closure differs')
                need(runtime[command['argv'][0]]['sha256']==command['executable_sha256'],'Executable changed')
                need(any(Path(n).name=='libmadloop_tirust.so' and r['sha256']=='eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9' for n,r in runtime.items()),'Wrong current TIRust mapping')
                need(identity(base+'/parsed.json')==run['parsed_sha256'] and identity(base+'/runtime.json')==run['runtime_sha256'],'Parsed/runtime binding differs')
                for name,want in command['inputs'].items():need(identity(base+'/'+name)==want,'Actual run input changed')
                rawlog=text(base+'/output.log');need(not re.search(r'##W03\b',rawlog,re.IGNORECASE),'Unstable contribution warning')
                log=work/'output.log';log.write_text(rawlog);parsed=controller.parse(log,command)
                need(parsed==doc(base+'/parsed.json') and parsed['passed'],'Offline parser replay differs/fails')
                need(not parsed['unstable_contribution_warnings'],'W03 diagnostic accepted')
                for row in parsed['rows']:
                    counts[row['ret']]=counts.get(row['ret'],0)+1
                    need(all(math.isfinite(v) for vals in row['coefficients'].values() for v in vals),'Nonfinite coefficient')
                    need(row['requested_qp_integrand_policy']==('MP(C)' if command['policy'] else 'DP(REAL(C,8))'),'Requested policy label differs')
                    if stage=='ordinary':need(row['ret']==210+command['backend'] and row['actual_tier']=='DP' and row['actual_integrand_input_view']=='host DP point P','Ordinary DP result mislabeled')
                    else:
                        need(row['ret']==120 and not row['stability_checked'] and row['actual_tier']=='QP','Fixed QP misqualified')
                        need(row['actual_integrand_precision']==('QP' if command['policy'] else 'DP'),'Actual integrand precision differs')
                        need(row['actual_integrand_input_view']==('C (MP_P)' if command['policy'] else 'REAL(C,8)'),'Actual input view differs')
                        if command['arm']=='full':full_exact+=1
                observed[(command['case'],command['backend'],command['mode'],command['policy'],command['arm'])]=parsed
            need(len(summary['comparisons'])==expected_comparisons,'Comparison coverage differs')
            stage_max=0.
            for comparison in summary['comparisons']:
                k=(comparison['case'],comparison['backend'],comparison['mode'],comparison['policy'])
                oldrows=observed[k+('original',)]['rows'];newrows=observed[k+(comparison['arm'],)]['rows'];diffs=[];states=[]
                for a,b in zip(oldrows,newrows,strict=True):
                    for split in a['coefficients']:
                        for slot,(x,y) in enumerate(zip(a['coefficients'][split],b['coefficients'][split],strict=True)):
                            rel=abs(x-y)/(abs(x)+abs(y)) if x or y else 0.;diffs.append({'point':a['point'],'split':split,'slot':slot,'relative':rel});stage_max=max(stage_max,rel)
                    states.append({'point':a['point'],'ret_equal':a['ret']==b['ret'],'flags_equal':a['flags']==b['flags'],'rounding_equal':a['rounding_unchanged']==b['rounding_unchanged'],'mp_state_equal':(a['mp_done'],a['mp_done_once'])==(b['mp_done'],b['mp_done_once'])})
                need(diffs==comparison['coefficients'] and states==comparison['states'] and comparison['passed'],'Comparison replay differs')
                need(comparison['criterion']==3e-6 and all(x['relative']<=3e-6 for x in diffs),'Regression gate failed')
                if stage=='ordinary':need(all(all(v for n,v in x.items() if n!='point') for x in states),'Ordinary state differs')
                else:need(comparison['unchecked'] is True,'Fixed comparison qualified as checked')
            maxres[stage]=stage_max
    need(counts=={218:12,219:12,120:48} and full_exact==12,'Actual precision/consistency coverage differs')
    return {'state':'verified','portable':True,'real_uux_z_only':True,'ordinary_runs':8,'fixed_qp_runs':16,
            'raw_streams_reparsed':24,'status_counts':counts,'max_relative':maxres,'full_fixedQP_exact_Born_and_C8_rows':12,
            'W03_warnings':0,'fixed_QP_stability_acceptance':False,'stress_or_performance_qualification':False,
            'native_or_compiler_executed':False,'text_paths':len(files)}

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--unfrozen',action='store_true')
    print(json.dumps(verify(cli.parse_args().unfrozen),indent=2,sort_keys=True))
