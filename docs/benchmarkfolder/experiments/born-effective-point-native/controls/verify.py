#!/usr/bin/env python3
"""Verify this portable text bundle without ignored builds or native execution."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
INDEX='artifact-index.json'
NAMES=('scalar_n3','scalar_n5','split_n3','split_n5','repair')
PREFIX='.tirust-build/born-effective-runtime/control-stub/'


def need(condition,message):
    if not condition:raise ValueError(message)


def digest(data):return hashlib.sha256(data).hexdigest()


def path(name):
    value=Path(name)
    need(not value.is_absolute() and '..' not in value.parts,'Unsafe bundle path: '+name)
    result=HERE/value
    need(result.resolve().is_relative_to(HERE),'Path escapes bundle: '+name)
    return result


def sha(name):return digest(path(name).read_bytes())
def read(name):return json.loads(path(name).read_text())


def inventory():
    result={}
    forbidden={'.exe','.o','.a','.so','.mod','.smod','.pyc','.pkl','.pickle'}
    for item in sorted(HERE.rglob('*')):
        need(not item.is_symlink(),'Symlink in bundle')
        if not item.is_file():continue
        name=str(item.relative_to(HERE))
        if name==INDEX:continue
        need(item.suffix.lower() not in forbidden and '__pycache__' not in item.parts,'Binary/cache artifact in bundle: '+name)
        data=item.read_bytes();data.decode('utf-8')
        need(b'\x00' not in data,'Nontext bytes: '+name)
        result[name]={'sha256':digest(data),'bytes':len(data)}
    return result


def output_validation(text,wanted):
    rows=[line.strip() for line in text.splitlines() if line.strip()]
    names=[line[5:] for line in rows if line.startswith('PASS ')]
    count=Counter(names)
    all_pass=[i for i,line in enumerate(rows) if line=='ALL PASS']
    missing=[name for name in wanted if count[name]==0]
    duplicate={name:n for name,n in count.items() if n!=1}
    unexpected=[name for name in count if name not in wanted]
    failed=[line for line in rows if line.startswith('FAIL')]
    late=bool(all_pass) and any(line.startswith('PASS ') for line in rows[all_pass[0]+1:])
    return {'passed':not(missing or duplicate or unexpected or failed or late) and len(all_pass)==1,
        'expected_count':len(wanted),'observed_pass_names':names,'missing':missing,
        'duplicate_counts':duplicate,'unexpected':unexpected,'failure_lines':failed,
        'all_pass_count':len(all_pass),'pass_after_terminal':late,
        'last_nonempty_line':rows[-1] if rows else None}


def semantic(live_root=None):
    mapping=read('source-map.json')['copied_text']
    for original,record in mapping.items():
        need(sha(record['copy'])==record['sha256'],'Copied original differs: '+original)
        need(path(record['copy']).stat().st_size==record['bytes'],'Copied size differs')
        if live_root:
            live=live_root/original
            need(live.is_file() and digest(live.read_bytes())==record['sha256'],'Live original differs: '+original)
    def original_hash(name):
        need(name in mapping,'Required copied original missing: '+name)
        return sha(mapping[name]['copy'])
    manifest=read('original/manifest.json')
    jobs=read('original/compiler-jobs.json')['jobs']
    expected=read('original/expected.json')['cases']
    native=read('original/native.json')
    proof=read('build/compile-proof.json')
    need(tuple(j['name'] for j in jobs)==NAMES,'Fixture sequence changed')
    need(set(expected)==set(NAMES) and sum(map(len,expected.values()))==51,'Wrong case inventory')
    need(sha('original/prepare.py')==manifest['generator_sha256'],'Generator identity differs')
    need(sha('original/compiler-jobs.json')==manifest['compiler_jobs_sha256'],'Compiler recipe identity differs')
    need(sha('original/expected.json')==manifest['expected_sha256'],'Expected identity differs')
    for name,want in manifest['inputs'].items():
        need(original_hash(name)==want,'Source dependency differs: '+name)
    for name,want in manifest['files'].items():
        need(sha('original/'+name)==want,'Prepared source differs: '+name)
    for row in manifest['proof']:
        if row['variant']=='repair':continue
        raw=path('original/sources/'+row['variant']+'/aggregate.raw.f').read_text()
        stripped=re.sub(r'(?m)^C RUNTIME-OBSERVER-BEGIN ([A-Z_]+)\n.*?^C RUNTIME-OBSERVER-END \1\n','',raw,flags=re.S)
        for prefix,kind in [('TX_','checked'),('LEG_','legacy')]:
            match=re.search(r'(?m)^      SUBROUTINE '+prefix+r'SMATRIX(?:_SPLITORDERS)?\(P, FLAV_IDX, ANS\)\n',stripped)
            need(match is not None,'Missing exact aggregate')
            body=stripped[match.start():].encode()[:row[kind+'_before_observer_bytes']]
            need(digest(body)==row[kind+'_before_observer_sha256'],'Observer does not invert exactly')
    need(native['state']=='complete' and native['native_executed'] and native['passed_cases']==51,'Native control run not complete')
    need(native['cpu']==14 and native['serial'] and native['inclusive_budget_seconds']==60.0,'Native ownership differs')
    need(0<=native['elapsed_seconds']<=60.0,'Inclusive runtime bound exceeded')
    need(native['controller_sha256']==sha('original/run.py'),'Runtime controller differs')
    need(native['helper_sha256']==sha('support/bounded_owner.py'),'Bounded owner differs')
    before=native['bindings_before'];after=native['bindings_after']
    need(before==after,'Sources/artifacts changed across native run')
    need(before['source_manifest_sha256']==sha('original/manifest.json'),'Native manifest binding differs')
    need(before['declared_jobs_sha256']==sha('original/compiler-jobs.json'),'Native job binding differs')
    need(before['expected_sha256']==sha('original/expected.json'),'Native expected binding differs')
    need(proof['state']=='complete' and proof['cpu']==14 and proof['max_compiler_jobs']==1,'Compiler proof not successful serial CPU14')
    need(proof['binary_artifacts_copied'] is False,'Unexpected copied binaries')
    need(proof['parent_build_sha256']==before['build_report_sha256'],'Parent build identity differs')
    need(proof['parent_build_inputs_sha256']==before['build_inputs_sha256'],'Compiler source inventory identity differs')
    need(proof['parent_build_log_sha256']==before['build_log_sha256'],'Compiler log identity differs')
    need(proof['parent_build_controller_sha256']==sha('build/original_build.py')==before['build_controller_sha256'],'Build controller differs')
    need(proof['compiled_source_bindings']==before['compiled_source_bindings'],'Compile/native source subset differs')
    need(proof['artifact_hashes_only']==before['artifact_bindings'],'Compile/native artifact subset differs')
    need(proof['compiled_source_bindings']=={PREFIX+name:value for name,value in manifest['files'].items()},'Compiled sources differ from exact package')
    need(proof['helper_sha256']==native['helper_sha256'],'Compiler/runtime process owner differs')
    compiled={x['name']:x for x in proof['commands']}
    need(len(proof['commands'])==5 and compiled==before['successful_compiler_events'],'Five exact compiler events not retained')
    suffix=proof['compile_log_exact_suffix']
    need(suffix['sha256']==sha(suffix['copy']) and suffix['bytes']==path(suffix['copy']).stat().st_size,'Compile log suffix changed')
    headers=[]
    for row in path(suffix['copy']).read_text().splitlines():
        if row.startswith('{'):
            try:value=json.loads(row)
            except json.JSONDecodeError:continue
            if str(value.get('name','')).startswith('control-stub/'):headers.append(value)
    need([h['name'] for h in headers]==['control-stub/'+name for name in NAMES],'Compile log headers incomplete')
    need(len(native['commands'])==5,'Expected five native commands')
    observed_count=0
    for job,event,header in zip(jobs,native['commands'],headers):
        name=job['name'];built=compiled['control-stub/'+name]
        need(event['name']==name,'Runtime sequence differs')
        for record in [built,event]:
            need(record['returncode']==0 and not record['timed_out'] and not record.get('error'),'Failed compiler/runtime event')
            need(record['process_group_cleanup']=='SIGKILL sent if group existed; direct child reaped','Missing process cleanup')
            need(record['command'][1:3]==['-c','14'],'CPU wrapper differs')
        declared=job['compile_argv'];pos=declared.index('gfortran')
        need(built['command'][3:]==[proof['environment']['FC'],*declared[pos+1:]],'Compiler argv differs from declared source')
        need(header['argv']==built['command'][3:],'Compile log argv differs')
        need(event['declared_argv']==job['run_argv']==native['declared_commands'][name],'Declared runtime argv differs')
        need(event['effective_argv']==[job['run_argv'][-1]]==event['command'][3:],'Nested/wrong runtime argv')
        need(0<event['timeout_seconds']<=60.0,'Incorrect shared deadline')
        log_copy=mapping[event['log']]['copy']
        need(sha(log_copy)==event['log_sha256'] and path(log_copy).stat().st_size==event['log_bytes'],'Native log differs')
        calculated=output_validation(path(log_copy).read_text(),expected[name])
        need(calculated==event['output_validation'] and calculated['passed'],'PASS record validation differs')
        executable=PREFIX+'work/'+name+'/control.exe'
        need(event['executable_sha256']==proof['artifact_hashes_only'][executable],'Executable identity differs')
        observed_count+=len(calculated['observed_pass_names'])
    need(observed_count==51,'Missing native cases')
    if live_root:
        for name,want in proof['artifact_hashes_only'].items():
            need(digest((live_root/name).read_bytes())==want,'Live compiled artifact differs: '+name)
    return {'state':'verified','native_cases':observed_count,'native_jobs':5,
            'copied_original_text_files':len(mapping),'portable':live_root is None,
            'scope':'controlled Fortran state/branch fixtures; no physical-amplitude or performance qualification'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-index',action='store_true',help='Freeze this subtree after semantic verification; no native execution')
    parser.add_argument('--live-root',type=Path,help='Optional extra check of original working paths and binary hashes')
    args=parser.parse_args()
    files=inventory()
    result=semantic(args.live_root.resolve() if args.live_root else None)
    value={'schema':1,'scope':'Portable exact control source/build/native evidence; text only',
           'file_count':len(files),'files':files}
    if args.write_index:
        path(INDEX).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    else:
        need(read(INDEX)==value,'Portable subtree inventory changed')
    result.update(index_sha256=sha(INDEX),indexed_files=len(files))
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
