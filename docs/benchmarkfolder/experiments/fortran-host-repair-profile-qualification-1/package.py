"""Read-only text capture of closed records; no native or publication action."""
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT=Path.cwd()
HERE=Path(__file__).resolve().parent
TARGET=HERE/'package'
SOURCE=ROOT/'.tirust-build/fortran-host-repair-profile-proposal-1'
EXECUTION=ROOT/'.tirust-build/fortran-host-repair-profile-execution-1'
WORK=ROOT/'.tirust-build/fortran-host-repair-profile-work-1'
PUBLIC='docs/benchmarkfolder/experiments/fortran-host-repair-profile-qualification-1'
LANES=('baseline','observed-0','observed-1','observed-2')


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def collect():
    payloads={};origins={}
    def add(name,raw,origin):
        if name.startswith('/') or '..' in Path(name).parts or '\\' in name:raise ValueError('Unsafe payload name')
        raw.decode('utf8')
        if b'\0' in raw or raw.startswith((b'\x7fELF',b'!<arch>')):raise ValueError('Native/nontext payload:'+name)
        if name in payloads and payloads[name]!=raw:raise ValueError('Payload collision:'+name)
        payloads[name]=raw;origins[name]={'recorded_origin':origin,'sha256':sha(raw),'bytes':len(raw)}
    def file(name,path):
        path=Path(path);add(name,path.read_bytes(),str(path))
    def tree(prefix,path):
        path=Path(path)
        for p in sorted(path.rglob('*')):
            if p.is_file():file(prefix+'/'+str(p.relative_to(path)),p)
    tree('source',SOURCE)
    source=json.loads(payloads['source/source-index.json'])
    for name in source['support']:file('support/'+name,ROOT/name)
    predecessor=json.loads(payloads['source/predecessor.json'])
    tree('scaffold',ROOT/predecessor['donor_proposal'])
    for phase in ('build','parity','profile'):tree('results/'+phase,EXECUTION/phase)
    pin=json.loads(payloads['source/input-pins.json']);donor=Path(pin['output'])
    for name in pin['generated_sources']:file('donor/'+name,donor/name)
    for name in ('generation.json','compilation.json','Cards/param_card.dat',*pin['numerical_cards']):file('donor/'+name,donor/name)
    for name,path in [('closed-observation.json',Path(pin['closed_observation'])),('closed-warmup.dat',Path(pin['closed_observation']).with_name('warmup.dat')),
                      ('closed-input.json',Path(pin['closed_input'])),('points.dat',Path(pin['points'])),('points.json',Path(pin['points']).with_suffix('.json'))]:file('inputs/'+name,path)
    for phase in ('parity','profile'):
        for lane in LANES:
            process=WORK/phase/lane/'SubProcesses/P0_uux_z'
            file('learning/'+phase+'/'+lane+'.dat',process/'warmup.dat')
            file('learning/'+phase+'/'+lane+'-dimensions.dat',process/'dimensions.dat')
    tools=json.loads(payloads['source/toolchain-pins.json'])
    for name in tools['donor_lineage']:file('environment-donor/'+name,ROOT/name)
    audit_dirs={'source':'fortran-host-repair-profile-independent-review-1',
                'primary':'fortran-host-repair-profile-independent-audit-1',
                'build':'fortran-host-repair-profile-independent-build-review-1',
                'parity':'fortran-host-repair-profile-independent-parity-review-1',
                'profile':'fortran-host-repair-profile-independent-profile-review-1',
                'owner-build':'fortran-host-repair-profile-owner-build-review-1',
                'owner-parity':'fortran-host-repair-profile-owner-parity-review-1',
                'owner-profile':'fortran-host-repair-profile-owner-profile-review-1'}
    for label,name in audit_dirs.items():tree('audits/'+label,ROOT/'.tirust-build'/name)
    file('audits/source/guarded-preflight.json',ROOT/'.tirust-build/fortran-host-repair-profile-preflight-1.json')
    resolution={};build=json.loads(payloads['results/build/result.json']);excluded={}
    for variant,row in build['variants'].items():
        process=Path(row['output'])/'SubProcesses/P0_uux_z'
        for name in ('makefile','private-link.map','check_sa.f','improve_ps.f','repair_profile.f90'):file('compiled/'+variant+'/'+name,process/name)
        aliases=[]
        for spelling in re.findall(r'^LOAD\s+(.+)$',(process/'private-link.map').read_text(),re.M):
            resolved=str((process/spelling).resolve())
            aliases.append({'spelling':spelling,'resolved':resolved,'sha256':row['actual_link_loads'][resolved]})
        runtime=[]
        for raw in row['runtime']['listing'].splitlines():
            line=raw.strip()
            if not line or line.startswith('linux-vdso.so.1 '):continue
            match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line)
            if match is None:raise ValueError('Unexpected runtime listing')
            spelling=match[1];resolved=str(Path(spelling).resolve())
            runtime.append({'spelling':spelling,'resolved':resolved,'sha256':row['runtime']['files'][resolved]})
        resolution[variant]={'map_aliases':aliases,'runtime_aliases':runtime}
        excluded[variant]={'executable':{'path':str(process/'check'),'sha256':row['executable_sha256']},
                           'objects_modules_archives':row['objects'],'runtime_providers':row['runtime']['files']}
    add('link-runtime-resolution.json',encoded(resolution),'Read-only historical spelling resolution, never followed during portable replay')
    add('excluded-native-identities.json',encoded(excluded),'Recorded native identities only; no binary bytes copied')
    if (HERE/'capture-history').exists():tree('capture-history',HERE/'capture-history')
    for name in ('replay_warmup.py','replay_measurements.py','verify_evidence.py','test_verify.py','test_replay_measurements.py','run_tests.py','README.md','package.py'):
        if (HERE/name).exists():file(name,HERE/name)
    retention={'schema':'host-repair-profile-retention-v1','intended_public_target':PUBLIC,
               'production_numerical_source':'0fcd2cd873d61d134837e1d9e87afb13a47824fd',
               'actual_export_revision':json.loads(payloads['donor/generation.json'])['source_identity']['revision'],
               'source_index_sha256':sha(payloads['source/source-index.json']),
               'binary_bytes_retained':False,'native_executions_during_capture':0,
               'performance_acceptance':False,'speedup_claimed':False,'production_promoted':False,
               'all_five_backends_qualified':False,'general_stress_qualified':False,
               'origins':origins,'recorded_BUILD_commands':133,
               'make_internal_children_individually_recorded':False,
               'scope':'One SM n1 backend9 descriptive host-repair attribution; DP includes nestedMP, no subtraction.',
               'administrative_history':'Retains source-authoring failed expectation receipt, both source-review fixes, and secondary PARITY audit concurrency assertion history. All native stages passed.'}
    add('retention.json',encoded(retention),'Portable retention metadata')
    return payloads,origins


def main():
    if TARGET.exists():raise ValueError('Fresh package target required; preserve previous draft')
    payloads,origins=collect();TARGET.mkdir()
    files={};logical={}
    plain={'verify_evidence.py','replay_measurements.py','replay_warmup.py','test_verify.py','test_replay_measurements.py','run_tests.py','package.py','README.md'}
    for name,raw in sorted(payloads.items()):
        compressed=len(raw)>4096 and name not in plain
        physical='compressed/'+name+'.gz' if compressed else name
        stored=gzip.compress(raw,compresslevel=9,mtime=0) if compressed else raw
        path=TARGET/physical;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(stored)
        files[physical]={'sha256':sha(stored),'bytes':len(stored)}
        logical[name]={'path':physical,'sha256':sha(raw),'bytes':len(raw),'encoding':'gzip+utf8' if compressed else 'utf8'}
    index={'schema':'host-repair-profile-portable-v1','files':files,'payloads':logical,
           'native_bytes_retained':False,'performance_acceptance':False,'speedup_claimed':False,
           'production_promoted':False,'all_five_backends_qualified':False,'general_stress_qualified':False}
    (TARGET/'evidence-index.json').write_bytes(encoded(index))
    publication={PUBLIC+'/'+str(p.relative_to(TARGET)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
                 for p in sorted(TARGET.rglob('*')) if p.is_file()}
    (HERE/'publication-map.json').write_bytes(encoded({'target':PUBLIC,'files':publication}))
    print(json.dumps({'index_sha256':sha((TARGET/'evidence-index.json').read_bytes()),'files':len(files)+1,
                      'stored_bytes':sum(row['bytes'] for row in files.values())+(TARGET/'evidence-index.json').stat().st_size,
                      'decoded_bytes':sum(len(v) for v in payloads.values()),'publication_map_sha256':sha((HERE/'publication-map.json').read_bytes())}))


if __name__=='__main__':main()
