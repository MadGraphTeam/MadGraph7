#!/usr/bin/env python3
"""Exclusive UTF-8 retention from closed stages; never executes native code."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PARENT=ROOT/'docs/benchmarkfolder/experiments/fortran-qp-mass-current-qualification-1'
PARENT_SHA='893c51f75dedbb6bbb1a70f78bed84b00d97b958ac678c194dc28cbb3aaa6b0b'
GEN_SOURCE=ROOT/'.tirust-build/qp-mass-cli-generation-proposal-3'
GEN=ROOT/'.tirust-build/qp-mass-cli-generation-execution-3'
QUAL=ROOT/'.tirust-build/qp-mass-fresh-cli-qualification-proposal-1'
EXECUTION=ROOT/'.tirust-build/qp-mass-fresh-cli-qualification-execution-1'
logical={};copies=[]


def require(value,message):
    if not value:raise ValueError(message)


def sha(value):return hashlib.sha256(value).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def text_bytes(path):
    path=Path(path)
    require(path.is_file() and path.suffix.lower() not in ('.o','.mod','.a','.so'),
        'Native/object/module/archive payload forbidden: '+str(path))
    data=path.read_bytes();require(not data.startswith(b'\x7fELF'),'ELF payload forbidden')
    data.decode('utf-8')
    return data


def add(name,path,expected=None):
    require(name not in logical and not Path(name).is_absolute() and '..' not in Path(name).parts,'Bad/duplicate logical path')
    data=text_bytes(path)
    if expected:require(sha(data)==expected,'Closed source changed: '+str(path))
    compressed=len(data)>=2048
    target=('compressed/'+name+'.gz') if compressed else ('payload/'+name)
    dest=HERE/target;dest.parent.mkdir(parents=True,exist_ok=True)
    if compressed:
        result=io.BytesIO()
        with gzip.GzipFile(filename='',mode='wb',fileobj=result,mtime=0,compresslevel=9) as stream:stream.write(data)
        stored=result.getvalue()
    else:stored=data
    with dest.open('xb') as stream:stream.write(stored)
    logical[name]={'path':target,'compression':'gzip' if compressed else 'none',
        'raw_sha256':sha(data),'raw_bytes':len(data)}
    copies.append({'logical':name,'origin':str(Path(path).resolve()),'raw_sha256':sha(data),
        'raw_bytes':len(data),'stored':target,'stored_sha256':sha(stored),'stored_bytes':len(stored)})


def tree(prefix,directory):
    directory=Path(directory)
    for path in sorted(directory.rglob('*')):
        if path.is_file():add(prefix+'/'+str(path.relative_to(directory)),path)


def source_capsule(prefix,directory,seal):
    index=read(directory/'source-index.json');require(sha((directory/'source-index.json').read_bytes())==seal,'Source seal changed')
    names={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    require(names==set(index['files'])|{'source-index.json'},'Source capsule payload set changed')
    for name,row in index['files'].items():add(prefix+'/'+name,directory/name,row['sha256'])
    add(prefix+'/source-index.json',directory/'source-index.json',seal)


def predecessor_subset():
    raw=(PARENT/'index.json').read_bytes();require(sha(raw)==PARENT_SHA,'Published parent index changed')
    index=json.loads(raw)
    with (HERE/'predecessor-index.json').open('xb') as stream:stream.write(raw)
    prefixes=('original-cli/','implementation/','source-2/','support/','auxiliary-cli/','stages/cache0/')
    points={'inputs/docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/points/qcd_n3_uux_zgg.dat',
        'inputs/docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/points/qcd_n3_uux_zgg.json'}
    selected=sorted(name for name in index['logical_files'] if name.startswith(prefixes) or name in points)
    require(len(selected)==394,'Unexpected predecessor subset')
    seen=set()
    for name in selected:
        row=index['logical_files'][name];physical=row['path'];pin=index['files'][physical]
        require(not Path(physical).is_absolute() and '..' not in Path(physical).parts,'Bad predecessor path')
        stored=(PARENT/physical).read_bytes()
        require(sha(stored)==pin['sha256'] and len(stored)==pin['bytes'],'Parent stored payload changed')
        data=gzip.decompress(stored) if row['compression']=='gzip' else stored
        require(sha(data)==row['raw_sha256'] and len(data)==row['raw_bytes'],'Parent decoded payload changed')
        data.decode('utf-8');target='predecessor-storage/'+physical
        if target not in seen:
            dest=HERE/target;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as stream:stream.write(stored)
            seen.add(target)
        logical['predecessor/'+name]={**row,'path':target}
        copies.append({'logical':'predecessor/'+name,'origin':str(PARENT/physical),
            'raw_sha256':sha(data),'raw_bytes':len(data),'stored':target,
            'stored_sha256':sha(stored),'stored_bytes':len(stored),'parent_index':PARENT_SHA})
    physical_files={}
    for name in ('replay.py','math_replay.py'):
        data=(PARENT/name).read_bytes();require(sha(data)==index['files'][name]['sha256'],'Parent mathematical helper changed')
        target='predecessor-'+name
        with (HERE/target).open('xb') as stream:stream.write(data)
        physical_files[name]=target
    return {'mode':'authenticated-subset','index_path':'predecessor-index.json','index_sha256':PARENT_SHA,
        'logical_prefix':'predecessor/','selected_logical_files':selected,'physical_files':physical_files,
        'whole_parent_numerical_replay_claimed':False,
        'published_parent':'docs/benchmarkfolder/experiments/fortran-qp-mass-current-qualification-1'}


def path_aliases(build,work):
    pre=read(build/'prelink.json');result=read(build/'result.json');cwd=Path(pre['cwd'])
    rows={}
    def bind(spelling,kind):
        path=Path(spelling);path=path if path.is_absolute() else cwd/path
        absolute=posixpath.normpath(str(path));resolved=str(path.resolve(strict=True));data=path.read_bytes()
        wanted=pre['bound'].get(resolved) or result['actual_link_loads'].get(resolved) or result['runtime']['files'].get(resolved)
        require(wanted==sha(data),'Path alias not joined to saved prelink/LOAD/runtime identity: '+spelling)
        record={'absolute':absolute,'resolved':resolved,'sha256':sha(data),'bytes':len(data)}
        if spelling in rows:require(rows[spelling]['resolved']==resolved,'Ambiguous historical path alias')
        else:rows[spelling]={**record,'kinds':[]}
        if kind not in rows[spelling]['kinds']:rows[spelling]['kinds'].append(kind)
    for line in (work/'private-link.map').read_text().splitlines():
        if line.startswith('LOAD '):bind(line[5:].strip(),'raw linker LOAD spelling')
    for name in re.findall(r'(?<!\S)(/[^\s()]+)',result['runtime']['listing']):bind(name,'saved loader listing spelling')
    return {'schema':'fresh-cli-build-path-aliases-v1','historical_cwd':str(cwd),'aliases':rows,
        'scope':'Read-only filesystem alias corroboration after closed BUILD; no loader/compiler execution',
        'native_queries':0}


def main():
    require(Path.cwd()==ROOT,'Run from the MadGraph working repository')
    require(not (HERE/'logical-manifest.json').exists(),'Exclusive capture; no overwrite/retry')
    parent=predecessor_subset()
    source_capsule('generation-source',GEN_SOURCE,'044c90afa64f3c65a6f96a46d67c781b8d643aec39e1e9506611299b3e59021b')
    source_capsule('qualification-source',QUAL,'c2acf316159ab0c12ab775b4ca92ff0d0eaa122a29db602deb629bae93da4450')
    bootstrap=read(GEN_SOURCE/'bootstrap-static-report.json')['visited_files']
    for name,row in bootstrap.items():add('bootstrap-sources/'+name,ROOT/row['source'],row['sha256'])
    add('bootstrap-sources/PLUGIN/__init__.py',ROOT/'PLUGIN/__init__.py')
    tree('generation',GEN)
    gen_result=read(GEN/'result.json');fresh=ROOT/'.tirust-build/qp-mass-cli-generation-work-3/output/qcd_n3_uux_zgg'
    require(len(gen_result['source_inventory'])==195,'Actual fresh source family changed')
    for name,row in gen_result['source_inventory'].items():add('fresh-cli/'+name,fresh/name,row['sha256'])
    for name in ('Cards/param_card.dat','Cards/MadLoopParams.dat'):
        add('fresh-cli/'+name,fresh/name)
    tree('all-source-audit',ROOT/'.tirust-build/qp-mass-cli-generation-all-source-audit-1')
    tree('source-review',ROOT/'.tirust-build/qp-mass-cli-generation-closed-independent-audit-3')
    build=EXECUTION/'build';run=EXECUTION/'run'
    tree('build',build);tree('run',run)
    built=read(build/'result.json');directory=Path(built['output'])/'SubProcesses/P0_uux_zgg'
    add('build/private-link.map',directory/'private-link.map',built['link_map_sha256'])
    for name in ('check_sa.f','makefile'):
        relative='SubProcesses/P0_uux_zgg/'+name
        add('build-source/'+relative,directory/name,built['sources'][relative])
    aliases=path_aliases(build,directory);dump(HERE/'build-path-aliases.json',aliases)
    reviews={
        'generation-source-3':'.tirust-build/qp-mass-cli-generation-independent-audit-3',
        'qualification-source-1':'.tirust-build/qp-mass-fresh-cli-qualification-independent-audit-1',
        'build-primary-1':'.tirust-build/qp-mass-fresh-cli-build-independent-audit-1',
        'build-independent-1':'.tirust-build/qp-mass-fresh-cli-qualification-build-independent-audit-1',
        'run-primary-1':'.tirust-build/qp-mass-fresh-cli-run-independent-audit-1',
        'run-independent-1':'.tirust-build/qp-mass-fresh-cli-qualification-run-independent-audit-1'}
    for label,path in reviews.items():tree('reviews/'+label,ROOT/path)
    for label,path in {
        'generation-preflight-3.json':'.tirust-build/qp-mass-cli-generation-preflight-3.json',
        'qualification-preflight-1.json':'.tirust-build/qp-mass-fresh-cli-qualification-preflight-1.json',
        'qualification-tests-1.log':'.tirust-build/qp-mass-fresh-cli-qualification-tests-1.log',
        'qualification-verified-build-1.json':'.tirust-build/qp-mass-fresh-cli-qualification-verified-build-1.json',
    }.items():add('reviews/'+label,ROOT/path)
    excluded=[]
    for name,checksum in built['objects'].items():excluded.append({'role':'actual BUILD numerical object/module/archive','path':str(Path(built['output'])/name),'sha256':checksum})
    excluded.append({'role':'actual BUILD/RUN executable','path':str(directory/'check'),'sha256':built['executable_sha256']})
    for name,checksum in built['runtime']['files'].items():excluded.append({'role':'observed runtime provider','path':name,'sha256':checksum})
    dump(HERE/'excluded-native-identities.json',{'schema':'hash-only-native-identities-v1','files':excluded,
        'native_payload_bytes_packaged':0,'source':'Closed BUILD result; identities only, no live binary replay'})
    review_pins={name:row['raw_sha256'] for name,row in logical.items()
        if ((name.startswith('reviews/') or name.startswith(('source-review/','all-source-audit/')))
            and (name.endswith('receipt.json') or name.endswith('chain-bindings.json')))}
    dump(HERE/'logical-manifest.json',{'schema':'qp-mass-fresh-cli-capture-v1','logical_files':logical,
        'predecessor':parent,'review_pins':review_pins,'excluded_binaries':excluded,
        'native_binaries_packaged':0,'new_native_executions':0,'public_writes':False})
    dump(HERE/'copy-map.json',{'schema':'qp-mass-fresh-cli-retention-copy-map-v1','files':copies,
        'target':'docs/benchmarkfolder/experiments/fortran-qp-mass-fresh-cli-qualification-1',
        'native_queries':0,'Git_mutations':0,'public_writes':False})
    print(json.dumps({'captured_logical_files':len(logical),'predecessor_subset_files':len(parent['selected_logical_files']),
        'copied_decoded_bytes':sum(row['raw_bytes'] for row in logical.values()),'new_native_executions':0}))


if __name__=='__main__':main()
