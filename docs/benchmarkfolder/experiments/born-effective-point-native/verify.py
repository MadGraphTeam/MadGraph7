#!/usr/bin/env python3
"""Portable verification of recorded text evidence; never build or run physics."""
import argparse
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
INDEX='evidence-index.json'


def need(condition,message):
    if not condition:raise ValueError(message)


def digest(data):return hashlib.sha256(data).hexdigest()
def read(name):return json.loads((HERE/name).read_text())
def sha(name):return digest((HERE/name).read_bytes())


def safe(name):
    p=Path(name)
    need(not p.is_absolute() and '..' not in p.parts,'Unsafe relative name: '+name)


def indexed_files(unfrozen):
    files={}
    for p in sorted(HERE.rglob('*')):
        need(not p.is_symlink(),'Symlink object in portable package')
        if not p.is_file():continue
        name=str(p.relative_to(HERE))
        if name==INDEX:continue
        need('__pycache__' not in p.parts and p.suffix.lower() not in ('.pyc','.pkl','.pickle','.o','.a','.mod','.smod','.exe','.so'),'Native/cache file in bundle: '+name)
        data=p.read_bytes()
        if p.suffix=='.gz':
            text=gzip.decompress(data).decode('utf-8');json.loads(text)
        else:
            data.decode('utf-8');need(b'\x00' not in data,'NUL bytes in text: '+name)
        files[name]={'sha256':digest(data),'bytes':len(data)}
    if (HERE/INDEX).exists():
        index=read(INDEX)['files']
        need(set(index)==set(files),'Outer index does not cover exactly every package file')
        for name,want in index.items():
            safe(name)
            need(want['sha256']==files[name]['sha256'] and want['bytes']==files[name]['bytes'],'Outer indexed file changed: '+name)
    else:
        need(unfrozen,'Missing evidence-index.json; only explicit --unfrozen permits pre-freeze verification')
    return files


def bundle_check(bundle):
    files=bundle['files'];blobs=bundle['blobs']
    need({r['sha256'] for r in files.values()}==set(blobs),'Unused or missing text blobs')
    for key,text in blobs.items():
        need(isinstance(text,str),'Nontext blob')
        data=text.encode('utf-8')
        need(b'\x00' not in data and digest(data)==key,'UTF8 blob SHA mismatch')
    for name,record in files.items():
        safe(name);data=blobs[record['sha256']].encode('utf-8')
        need(len(data)==record['bytes'],'Text byte count differs: '+name)
    return files,blobs


def successful(event):
    need(event['returncode']==0 and not event['timed_out'] and not event.get('error'),'Failed recorded process event')


def rounded128(words):
    """Exact binary128 rational rounded by Python to nearest-even binary64."""
    need(len(words)==2 and all(re.fullmatch('[0-9A-F]{16}',w) for w in words),'Malformed binary128 words')
    low,high=(int(w,16) for w in words);raw=low|(high<<64)
    exponent=(raw>>112)&32767;mantissa=raw&((1<<112)-1);negative=bool(raw>>127)
    need(exponent!=32767,'Nonfinite binary128 observation')
    if not exponent and not mantissa:value=-0.0 if negative else 0.0
    else:
        if exponent:mantissa|=1<<112
        value=float(Fraction((-1 if negative else 1)*mantissa)*Fraction(2)**((exponent or 1)-16383-112))
    return struct.pack('>d',value).hex().upper()


def exact_record(run_id,command,row):
    bits=row['bits']
    published={x[1]:x[2] for x in bits['COEFFICIENT_BITS'] if x[0]=='0'}
    user={x[0]:x[1] for x in bits['BORN_USER_BITS']}
    dp={x[0]:x[1] for x in bits['BORN_C8_BITS']}
    mp={x[0]:rounded128(x[1:]) for x in bits['BORN_C_BITS']}
    coordinates={(x[0],x[1]):rounded128(x[2:]) for x in bits['C_BITS']}
    view={(x[0],x[1]):x[2] for x in bits['DP_VIEW_BITS']}
    need(len(coordinates)==len(bits['C_BITS'])==len(view)==len(bits['DP_VIEW_BITS']),'Duplicate/missing coordinate record')
    need(set(published)==set(user)==set(dp)==set(mp)=={'0','1'},'Born split-order coverage differs')
    return dict(id=run_id,arm=command['arm'],policy=command['policy'],
        **{k:row[k] for k in ('call','point','ret','mp_done','mp_done_once')},
        published=published,user_DP_Born=user,C8_DP_Born=dp,C_MP_Born_rounded_binary64=mp,
        DP_view_matches_round_C=coordinates==view,
        published_matches_requested_view=published==(mp if command['policy'] else dp))


def raw_rows(text):
    rows=[];current=None
    for line in text.splitlines():
        words=line.strip().split()
        if not words:continue
        tag=words[0]
        if tag=='RESULT':
            need(len(words)==8,'Malformed RESULT row')
            current={'call':int(words[1]),'point':int(words[2]),'ret':int(words[3]),
                'mp_done':words[4]=='T','mp_done_once':words[5]=='T',
                'finite':words[6]=='T','rounding_unchanged':words[7]=='T',
                'bits':{},'coefficients':{},'born':{}}
            rows.append(current)
        elif current is not None:
            if tag=='FLAGS':current['flags']=words[1:]
            elif tag=='POST_OBSERVATION_FLAGS':current['observation_flags']=words[1:]
            elif tag=='ACCURACY':current['accuracy']=[float(w.replace('D','E')) for w in words[1:]]
            elif tag=='COEFFICIENTS':
                need(words[1] not in current['coefficients'],'Duplicate coefficient row')
                current['coefficients'][words[1]]=[float(w.replace('D','E')) for w in words[2:]]
            elif tag.endswith('_BITS'):current['bits'].setdefault(tag,[]).append(words[1:])
            elif tag in ('BORN_USER','BORN_C8','BORN_C'):
                need(tag not in current['born'],'Duplicate Born reference row')
                current['born'][tag]={'ok':words[1]=='T','values':[float(w.replace('D','E')) for w in words[2:]]}
    return rows


def verify(unfrozen=False):
    inventory=indexed_files(unfrozen)
    encoded=gzip.decompress((HERE/'sources.json.gz').read_bytes())
    bundle=json.loads(encoded)
    files,blobs=bundle_check(bundle)
    source_index=read('source-inventory.json')
    need(source_index['files']==files and source_index['blob_count']==len(blobs),'Source inventory differs from bundle')
    need(source_index['bundle_sha256']==sha('sources.json.gz') and source_index['uncompressed_bytes']==len(encoded),'Compressed source binding differs')
    native=read('native-identities.json')['files']
    def identity(name):
        need(name in files or name in native,'Unbound original identity: '+name)
        return files.get(name,native.get(name))['sha256']
    def source_json(name):
        need(name in files,'Missing original text: '+name)
        return json.loads(blobs[files[name]['sha256']])
    for name,record in native.items():
        safe(name);need(re.fullmatch('[0-9a-f]{64}',record['sha256']) and record['bytes']>=0,'Invalid native hash-only identity')
    for name,record in read('link-identities.json')['links'].items():
        need(identity(name)==record['sha256'],'Original link identity differs')
    direct=read('direct-copy-identities.json')
    for copied,record in direct.items():
        safe(copied)
        need(sha(copied)==record['sha256']==identity(record['original']),'Direct copy not exact: '+copied)
        need((HERE/copied).read_bytes()==blobs[files[record['original']]['sha256']].encode(),'Direct copy differs from packed text')
    prior=json.loads(gzip.decompress((HERE/'prior-build-history.json.gz').read_bytes()))
    prior_files,prior_blobs=bundle_check(prior)
    prior_index=read('prior-build-index.json')['files']
    omitted=prior['native_or_alias_hash_only']
    need(not(set(prior_files)&set(omitted)) and set(prior_files)|set(omitted)==set(prior_index),'Prior-build partition incomplete')
    need({**prior_files,**omitted}==prior_index,'Prior-build entries differ from original index')
    need(sha('prior-build-index.json')=='c28a90390f08baeeb8d375d365c3de1c1098e2db5c8d603beefb1d8a4c014f9f','Prior-build index pin differs')
    build=read('build/complete.json');material=read('build/materialized.json')
    need(build['state']=='complete' and not build['changed_inputs'],'16-arm build incomplete/changed')
    need(build['cpu']==14 and build['max_compiler_jobs']==1 and not build['native_executed'],'Compiler scope differs')
    need(build['elapsed_seconds']<=build['total_budget_seconds']==600,'Original compile budget exceeded')
    need(sha('build/initial-inputs.json')==build['build_inputs_sha256'],'Build input inventory differs')
    for name,want in read('build/initial-inputs.json').items():need(identity(name)==want,'Build input identity differs: '+name)
    for name,want in build['artifacts'].items():need(identity(name)==want,'Build artifact identity differs: '+name)
    for name,key in [('build.py','controller_sha256'),('build.log','build_log_sha256')]:
        need(identity('.tirust-build/born-effective-runtime/attempt1/'+name)==build[key],'Build controller/log differs')
    need(material['status']=='complete' and len(material['cases'])==16,'Expected 16 materialized arms')
    expected={(case,arm) for case in ('real_uux_z','cms_epem_ttx','ew_epem_mupmum','qcd_uux_zgg') for arm in ('original','mass','graph','full')}
    need({(r['case'],r['arm']) for r in material['cases']}==expected,'Materialized arm coverage differs')
    names=[]
    for event in build['commands']:
        successful(event);names.append(event['name'])
    need(len(names)==len(set(names))==37,'Expected 32 arm and five control compile jobs')
    for case,arm in expected:
        for suffix in ('model-aloha','runtime-link'):need(case+'/'+arm+'/'+suffix in names,'Missing arm build stage')
    for case in material['cases']:
        for name,want in case['files'].items():need(identity(case['export']+'/'+name)==want,'Materialized source differs: '+name)
    exact_build=read('build/exact-driver.json')
    need(exact_build['state']=='complete' and not exact_build['changed_original_inputs'],'Exact driver build failed/changed')
    need(exact_build['original_build_sha256']==sha('build/complete.json'),'Exact driver parent build binding differs')
    for event in exact_build['commands']:successful(event)
    for name,want in exact_build['outputs'].items():need(identity(name)==want,'Exact driver output binding differs')
    for prefix,field in [('driver-hex-addendum-attempt3/build.py','controller_sha256'),('driver-hex-addendum-attempt3/source.json','source_sha256'),('driver-hex-addendum-attempt3/build.log','log_sha256'),('driver-hex-addendum-attempt2/build.json','prior_partial_build_sha256')]:
        need(identity('.tirust-build/born-effective-runtime/'+prefix)==exact_build[field],'Exact driver source/controller differs')
    prepared=read('legacy-prepared.json')
    commands={c['id']:c for c in prepared['commands']}
    need(len(commands)==len(prepared['commands'])==28,'Legacy prepared coverage differs')
    original_root=read('controls/source-map.json')['original_repository_root'].rstrip('/')+'/'
    exact_records=[]
    for stage,count in [('ordinary',12),('fixed-qp',16)]:
        summary=read(stage+'/summary.json')
        need(summary['state']=='complete' and not summary['changed_bound_files'],'Legacy stage failed/changed')
        need(summary['cpu']==14 and summary['prepared_sha256']==sha('legacy-prepared.json'),'Legacy admission identity differs')
        need(summary['elapsed_seconds']<=summary['per_stage_seconds']==300,'Legacy stage bound exceeded')
        need(len(summary['runs'])==count and len({r['id'] for r in summary['runs']})==count,'Legacy run coverage differs')
        for run in summary['runs']:
            command=commands[run['id']]
            need(run['passed'] and run['maps_passed'],'Legacy run/mapping failed')
            base=stage+'/runs/'+run['id']+'/'
            event=read(base+'execution.json');successful(event)
            need(event==run['command'],'Summary/event command differs')
            need(event['command'][1:3]==['-c','14'],'Runtime CPU differs')
            parsed=read(base+'parsed.json');mapping=read(base+'mapping.json');runtime=read(base+'runtime.json')
            need(sha(base+'parsed.json')==run['parsed_sha256'] and sha(base+'runtime.json')==run['runtime_sha256'],'Parsed/runtime report binding differs')
            need(digest(mapping['maps_text'].encode())==mapping['maps_sha256']==run['maps_sha256'],'Mapping snapshot bytes differ')
            need(mapping['cpu_allowed_list']=='14','Recorded affinity differs')
            expected_runtime={p:r['sha256'] for p,r in runtime.items()}
            libraries={p:v for p,v in expected_runtime.items() if p!=mapping['executable']}
            need(mapping['expected_runtime_sha256']==libraries,'Mapping expected runtime differs')
            need(mapping['mapped_files_sha256_after_exit']==libraries,'Mapped runtime changed')
            need(mapping['executable']==event['command'][-1],'Mapped executable differs')
            need(mapping['executable'].startswith(original_root),'Executable outside recorded original root')
            need(identity(mapping['executable'][len(original_root):])==expected_runtime[mapping['executable']]==command['executable_sha256'],'Executable build/runtime identity differs')
            need(Path(event['command'][0]).name=='taskset' and event['command'][1:]==['-c','14']+command['argv'],'Executed/prepared argv differ')
            execution_cwd=event['cwd']
            if Path(execution_cwd).is_absolute():
                need(execution_cwd.startswith(original_root),'Run cwd outside recorded original root')
                execution_cwd=execution_cwd[len(original_root):]
            safe(execution_cwd)
            for name,want in command['inputs'].items():
                # Controller4 adds these two standard-layout aliases only to
                # its execution copy; the original attempt1 cwd stays frozen.
                input_cwd=execution_cwd if name in ('Cards/ident_card.dat','Cards/param_card.dat') else command['cwd']
                need(identity(input_cwd+'/'+name)==want,'Runtime input identity differs')
                need(identity(execution_cwd+'/'+name)==want,'Executed copy input identity differs')
            need(parsed['passed'] and not parsed['parse_errors'] and all(x['pass'] for x in parsed['checks']),'Parsed legacy checks failed')
            need(len(parsed['warmup'])==4 and len(parsed['rows'])==3,'Warmup/observation counts differ')
            need(raw_rows((HERE/base/'output.log').read_text())==parsed['rows'],'Raw native records differ from parsed observations')
            need([r['call'] for r in parsed['rows']]==[1,2,3] and [r['point'] for r in parsed['rows']]==[1,2,1],'Point ordering differs')
            for row in parsed['rows']:
                need(row['finite'] and row['rounding_unchanged'],'Nonfinite/changed-rounding observation')
                need(all(math.isfinite(v) for a in row['coefficients'].values() for v in a),'Nonfinite recorded coefficient')
                need(row['flags']==row['observation_flags'],'Born observation changed sticky flags')
                if stage=='fixed-qp':exact_records.append(exact_record(run['id'],command,row))
        if stage=='ordinary':
            need(len(summary['comparisons'])==6,'Expected six ordinary comparisons')
            for comparison in summary['comparisons']:
                need(comparison['passed'] and comparison['criterion']==3e-6,'Ordinary regression comparison failed')
                need(all(r['relative']<=comparison['criterion'] for r in comparison['coefficients']),'Coefficient exceeds criterion')
                need(all(all(v for k,v in r.items() if k.endswith('_equal')) for r in comparison['states']),'Ordinary control state differs')
        else:
            need(not summary['comparisons'],'Fixed-QP stage must not imply independent amplitude accuracy')
    exact=read('fixed-qp/exact-summary.json')
    need(rounded128(['0000000000000000','8000000000000000'])=='8000000000000000','Signed-zero decoder failure')
    need(len(exact_records)==48 and exact_records==exact['records'],'Independent exact Born/coordinate recomputation differs')
    for name,want in exact['source_bindings'].items():need(identity(name)==want,'Exact-view analysis source differs')
    full=[r for r in exact['records'] if r['arm']=='full']
    need(len(full)==exact['full_rows']==24,'Expected 24 full-policy coordinate/Born observations')
    need(sum(r['DP_view_matches_round_C'] for r in full)==exact['full_C8_coordinate_exact_matches']==24,'C8 coordinate check failed')
    need(sum(r['published_matches_requested_view'] for r in full)==exact['full_requested_view_exact_matches']==24,'Published precision-policy view failed')
    need(sum(r['published_matches_requested_view'] for r in exact_records if r['arm']=='original')==exact['original_requested_view_exact_matches']==12,'Original policy-control outcomes differ')
    for failure in ('failure-permission','failure-layout'):
        need(read(failure+'/summary.json')['state']=='failed','Historical failure was lost')
    spec=importlib.util.spec_from_file_location('portable_checked_controls',HERE/'controls/verify.py')
    controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)
    control_inventory=controls.inventory();control_index=controls.read(controls.INDEX)
    need(control_index['files']==control_inventory and control_index['file_count']==len(control_inventory),'Nested controls inventory differs')
    control_result=controls.semantic()
    need(control_result['native_cases']==51,'Expected 51 native control cases')
    return {'state':'verified','portable':True,'outer_index_present':(HERE/INDEX).exists(),
        'outer_files':len(inventory),'source_text_paths':len(files),'unique_text_blobs':len(blobs),
        'prior_build_text_paths':len(prior_files),'prior_build_hash_only':len(omitted),
        'materialized_compiled_arms':16,'legacy_native_runs':28,'controlled_native_cases':51,
        'scope':'ordinary regression and fixed-QP coordinate/Born policy checks; fixed-QP is not independent amplitude accuracy or performance qualification'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unfrozen',action='store_true',help='Explicitly allow absent outer index during packaging; all semantic gates still apply')
    args=parser.parse_args()
    print(json.dumps(verify(args.unfrozen),indent=2,sort_keys=True))


if __name__=='__main__':main()
