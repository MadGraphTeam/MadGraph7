#!/usr/bin/env python3
"""Verify portable records and replay only frozen pure-Python journal analysis."""
import argparse,gzip,hashlib,importlib.util,json,re,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE='.tirust-build/qp-mass-general-implementation/combined-qualification'
CONT=BASE+'/loader-correction/continuation'
FIX=CONT+'/additive-offline-analysis'
def need(ok,why):
    if not ok:raise ValueError(why)
def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text())

def verify(unfrozen=False):
    outer={}
    for p in sorted(HERE.rglob('*')):
        need(not p.is_symlink(),'Symlink in portable package')
        if not p.is_file() or p.name=='evidence-index.json':continue
        need('__pycache__' not in p.parts and p.suffix not in ('.o','.a','.so','.mod','.pyc','.pkl'),'Native/cache payload')
        data=p.read_bytes();need(not data.startswith((b'\x7fELF',b'!<arch>\n')),'Binary payload')
        if p.suffix=='.gz':json.loads(gzip.decompress(data).decode('utf-8'))
        else:data.decode('utf-8');need(b'\0' not in data,'NUL in text')
        outer[str(p.relative_to(HERE))]={'sha256':digest(data),'bytes':len(data)}
    if (HERE/'evidence-index.json').exists():need(read(HERE/'evidence-index.json')['files']==outer,'Outer inventory differs')
    else:need(unfrozen,'Missing outer evidence index')
    compressed=(HERE/'payload.json.gz').read_bytes();raw=gzip.decompress(compressed);payload=json.loads(raw)
    files=payload['files'];blobs=payload['blobs'];inventory=read(HERE/'payload-inventory.json')
    need(inventory['files']==files and inventory['gzip_sha256']==digest(compressed),'Payload inventory differs')
    need(inventory['unique_text_contents']==len(blobs) and inventory['uncompressed_bytes']==len(raw),'Payload dimensions differ')
    for name,record in files.items():
        data=blobs[record['sha256']].encode('utf-8')
        need(len(data)==record['bytes'] and digest(data)==record['sha256'],'Text blob differs: '+name)
        need(b'\0' not in data and not data.startswith((b'\x7fELF',b'!<arch>\n')),'Native blob')
    def key(name):
        marker='/madgraph7_madloop_tirust/'
        return name.split(marker,1)[1] if marker in name else name
    def text(name):return blobs[files[key(name)]['sha256']]
    def doc(name):return json.loads(text(name))
    for name,record in read(HERE/'direct-copies.json').items():
        need(digest((HERE/name).read_bytes())==record['sha256']==files[record['original']]['sha256'],'Direct copy differs')
    continuation=doc(CONT+'/run.json');additive=doc(CONT+'/additive/run.json')
    need(continuation['state']=='complete' and continuation['input_hashes_after']=='unchanged','Continuation not closed')
    need(len(continuation['jobs'])==234 and continuation['reused_fixture_events']==218,'Continuation event scope differs')
    need(continuation['scientific_scope_unchanged'] is True,'Changed continuation science')
    need(additive['state']=='failed' and additive['input_hashes_after']=='unchanged','Original additive failure lost')
    for report in (continuation,additive):
        need(len({e['name'] for e in report['jobs']})==len(report['jobs']),'Duplicate event')
        for event in report['jobs']:
            need(files[key(event['log'])]['sha256']==event['log_sha256'],'Native/validation log differs')
            need(not event['timed_out'] and not event.get('changed_inputs'),'Timed out/changed event')
            for name,want in event.get('output_files',{}).items():
                need(files[key(event['cwd']+'/'+name)]['sha256']==want,'Captured output differs')
    original_workers=[e for e in additive['jobs'] if e['name'] in ('current-original-qp-8','current-original-qp-9')]
    need(len(original_workers)==2 and all(e['returncode']==0 and e['expected_output_closure_matched'] for e in original_workers),'Original QP workers incomplete')
    need(additive['jobs'][-1]['name']=='validate-current-original-qp' and additive['jobs'][-1]['returncode']==1,'Stale validator failure lost')
    execution=doc(FIX+'/analysis-execution.json');lineage=doc(FIX+'/lineage.json');result=doc(FIX+'/analysis.stdout.json')
    need(execution['returncode']==0 and not execution['native_rerun'] and not execution['compiler_executed'] and not execution['changed_inputs'],'Offline repair incomplete')
    original=text(BASE+'/current-original-qp/analyze.py');corrected=text(FIX+'/analyze.py')
    addition="    spec['required_prior_run']['path']=str(HERE.parent/'run.json')\n"
    need(corrected.count(addition)==1 and corrected.replace(addition,'',1)==original,'Repair changes more than parent path')
    need(result['comparison_passed'] and result['prior_completed_run_sha256']==files[CONT+'/run.json']['sha256'],'Wrong corrected completed parent')
    need(not result['physical_accuracy_or_host_stability_acceptance'] and not result['broader_graph_or_born_qualification'] and not result['performance_claim'],'Improper qualification')
    need(files[FIX+'/source-index.json']['sha256']=='4b2e0cd422cc094c47e6440b10e0cd7a586efaf91edbcdf1e5eb12432ef6823f','Offline source index changed')
    native=read(HERE/'native-identities.json')['files']
    for name,want in lineage['bindings'].items():
        need((files.get(key(name)) or native.get(key(name),{})).get('sha256')==want,'Repair input binding differs')
    warnings=doc(FIX+'/warning-audit.json')['records'];counts={}
    for record in warnings:
        lines=text(record['path']).splitlines()
        hits=[{'line':i,'text':line} for i,line in enumerate(lines,1) if re.search(r'##W03\b',line,re.IGNORECASE)]
        need(hits==record['warnings'] and len(hits)==record['warning_count'],'W03 audit differs')
        need(not record['amplitude_coefficients_qualified'],'Incomplete/unchecked amplitude qualified')
        counts[(record['arm'],record['backend'])]=len(hits)
    need(counts=={('current_original',8):0,('current_original',9):74,('current_guarded',8):0,('current_guarded',9):74},'Warning scope differs')
    child=doc(BASE+'/current-original-qp/child-prepared.json')
    # Load only the retained Python parser/comparator definitions. Their main
    # functions are not invoked; no native library or executable is loaded.
    with tempfile.TemporaryDirectory(prefix='mass-portable-') as temp:
        work=Path(temp)/'source/a/b/c/d';work.mkdir(parents=True)
        def module(name,source):
            path=work/(name+'.py');path.write_text(source)
            spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
        parser=module('retained_mass_parser',text(child['frozen_parser']['path']))
        comparator=module('retained_mass_comparator',corrected)
        for compared in result['comparisons']:
            backend=compared['backend'];capture={}
            for label,entry in result['journal_bindings'][str(backend)].items():
                need(files[key(entry['path'])]['sha256']==entry['sha256'],'Journal binding differs')
                path=work/(str(backend)+'-'+label+'.hex');path.write_text(text(entry['path']))
                capture[label]=parser.parse(path,'original' if label.endswith('original') else 'candidate',backend)
            for field,left,right,axis in [('same_current_runtime_mass_axis','current_original','current_guarded',True),('secondary_original_historical49af_vs_currenteb3d','historical_original','current_original',False),('secondary_guarded_historical49af_vs_currenteb3d','historical_candidate','current_guarded',False)]:
                got=comparator.compare(capture[left],capture[right],axis)
                need(got==compared[field],'Frozen comparator replay differs: '+field)
            primary=compared['same_current_runtime_mass_axis']
            for point in primary['points']:
                expected=37 if backend==9 and point['point']==2 else 0
                need(len(point['left_native_refusals'])==len(point['right_native_refusals'])==expected,'Native refusal proof differs')
    return {'state':'verified','portable':True,'text_paths':len(files),'native_identity_records':len(native),
            'continuation_events':234,'reused_fixture_events':218,'new_continuation_amplitude_workers':6,
            'additive_native_workers':2,'offline_comparisons_replayed':6,
            'original9_W03':74,'guarded9_W03':74,'point2_refusals_each9_arm':37,
            'coefficient_qualification':False,'native_or_compiler_executed_by_verifier':False}

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--unfrozen',action='store_true')
    print(json.dumps(verify(cli.parse_args().unfrozen),indent=2,sort_keys=True))
