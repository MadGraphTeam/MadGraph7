"""Finite text/data mirror for the selected eight cases; never run physics.

Do not execute during a quiet timing window. The resulting capsule preserves
original JSON bytes/paths and is not a portable full-runtime qualification.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
HERE=Path(__file__).resolve().parent
AUTH=HERE/'lightweight-final-eight-qualified-v2'
JOBS=HERE/'generic-random-parity-held-v2/jobs'
CASES=('qcd-1','ew-1','qcd-2','ew-2','qcd-3','ew-3','supplement-gg-h','supplement-gg-hh')
EXPECTED={'manifest.json':'de92bfae90b1068ba0fcb22034163d0b97ec9e4ab416ad17146bfa605e87f1a2',
    'summary.json':'c46368e08f039d39d4acb6369f067407785315cf5a591099db97f1e9adb6b12e',
    'cross-backend-discrepancy.json':'392f3ff2360df961aa1cb2170b08ca2aea446e1db5dc55c27b710bd76b13d94d'}
AUDITS={'light-eight-arithmetic-audit-v1.json':'99e0f508d32c2c668fe0a54a9c2e4cf3361e8915197ad54782137ef4c027baa7',
    'light8-final-raw-custody-ownership-audit-v1.json':'7a99193f3970403891d1e2dd5285843779c8063274dac4966a206ec4489811d4',
    'light-eight-final-independent-composite-audit-v1.json':'64ac6c50cb1712272e11dbca41e797e7389d893df88ca67c1189e8f1bdf14e33'}
ALLOWED={'.json','.raw','.dat','.f','.py','.c','.inc','.log','.stdout','.stderr','.txt'}
def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def data_path(record):
    path=Path(record['path']);path=(path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Artifact outside working repository')
    data=path.read_bytes()
    if sha(data)!=record['sha256'] or ('bytes' in record and len(data)!=record['bytes']):
        raise ValueError('Changed retained artifact: '+str(path))
    return path
def binding(path):
    path=Path(path).resolve();data=path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(data),bytes=len(data))

def curate(destination,audits):
    if destination.exists():raise ValueError('Preserve previous capsule')
    for name,expected in EXPECTED.items():
        if sha((AUTH/name).read_bytes())!=expected:raise ValueError('Qualified authority changed')
    summary=read(AUTH/'summary.json')
    if (set(summary['cases'])!=set(CASES) or summary['requested_tools']!=[6,8] or
            summary['physical_qualified'] is not True or summary['complete'] is not True or
            any(case['N']!=100 or set(case['tools'])!={'6','8'} for case in summary['cases'].values())):
        raise ValueError('Final selected100-point coverage differs')
    if {p.name for p in audits}!=set(AUDITS) or len(audits)!=3:
        raise ValueError('Exact three independent final audits required')
    for path in audits:
        if sha(path.read_bytes())!=AUDITS[path.name]:raise ValueError('Independent audit changed')
    files={};aliases={}
    def take(path,label):
        path=Path(path).resolve()
        if not path.is_relative_to(ROOT) or path.suffix.lower() not in ALLOWED:
            raise ValueError('Not an allowed finite text artifact: '+str(path))
        key=str(path.relative_to(ROOT));data=path.read_bytes();data.decode('utf8')
        if data.startswith((b'\x7fELF',b'!<arch>')):raise ValueError('Binary build product forbidden')
        if key in files:
            if files[key]['original']['sha256']!=sha(data):raise ValueError('Source changed during curation')
            files[key]['labels'].append(label);return
        compress='guard' in label and len(data)>200000
        name=f'{len(files):04d}-{path.name}'+('.gz' if compress else '')
        target=destination/'data'/name;payload=gzip.compress(data,compresslevel=9,mtime=0) if compress else data
        restored=gzip.decompress(payload) if compress else payload
        if restored!=data:raise ValueError('Mirror differs from original bytes')
        copied=dict(path=str(target.relative_to(ROOT)),sha256=sha(payload),bytes=len(payload))
        files[key]=dict(original=binding(path),copy=copied,encoding='gzip' if compress else 'identity',labels=[label])
    def keep(record,label):
        path=data_path(record);aliases[record['path']]=str(path.relative_to(ROOT));take(path,label);return path
    for name in EXPECTED:take(AUTH/name,'qualified '+name)
    raw=read(AUTH/'manifest.json')
    keep(raw['cohort_record'],'cohort')
    for record in raw['bindings']:keep(record,'assembly binding')
    for cid in CASES:
        for tool in (6,8):
            job=JOBS/f'{cid}-{tool}';proof_path=job/'science-proof.json';take(proof_path,f'{cid}/{tool} science proof')
            proof=read(proof_path);start=read(keep(proof['start'],f'{cid}/{tool} start'))
            prepared=read(keep(start['prepared'],f'{cid}/{tool} prepared'))
            keep(start['source_freeze'],f'{cid}/{tool} source freeze')
            keep(start['guards'][str(tool)]['source'],f'{cid}/{tool} guard source')
            keep(start['helper']['source'],f'{cid}/{tool} checkpoint source')
            manifest=read(keep(start['random_manifest'],f'{cid}/{tool} random authority'))
            if isinstance(manifest.get('configuration'),dict):keep(manifest['configuration'],'random configuration')
            point=manifest['cases'][cid]
            for key in ('points','metadata'):keep(point[key],f'{cid} {key}')
            keep(prepared['physical_card'],f'{cid} retained physical card')
            keep(prepared['parity_points'],f'{cid} exact first3 points')
            keep(proof['parity_receipt'],f'{cid}/{tool} parity receipt')
            for role in ('plain','observed','science'):
                result=read(keep(proof[role+'_result'],f'{cid}/{tool}/{role} result'))
                for key in ('raw','parser_view','guard'):keep(result[key],f'{cid}/{tool}/{role} {key}')
                for key in ('stdout','stderr'):
                    keep(result['execution'][key],f'{cid}/{tool}/{role} execute {key}')
                    keep(result['build'][key],f'{cid}/{tool}/{role} build {key}')
                custody=result['custody'];keep(custody['checkpoint_log'],f'{cid}/{tool}/{role} checkpoint log')
                for name,records in custody['static'].items():
                    for phase,record in records.items():keep(record,f'{cid}/{tool}/{role} {phase} {name}')
                for phase,states in custody['filters'].items():
                    for name,state in states.items():
                        if state['available']:keep(state['snapshot'],f'{cid}/{tool}/{role} {phase} {name}')
                output=ROOT/result['output'];directory=ROOT/result['execution']['cwd']
                for name in ('param_card.dat','MadLoopParams.dat','MadLoopParams_default.dat'):
                    take(output/'Cards'/name,f'{cid}/{tool}/{role} {name}')
                for name in ('check_sa.f','CT_interface.f'):take(directory/name,f'{cid}/{tool}/{role} {name}')
                origin=ROOT/prepared['origin']
                take(origin/'SubProcesses'/directory.name/'CT_interface.f',f'{cid} original interface')
    for path in audits:take(path,'independent audit receipt')
    for path in (HERE/'approved-cheap-science-execution-v1/terminal.json',
                 HERE/'generic-random-parity-held-v2/science-handoff.json',
                 HERE/'random-runner-qualification-v2/assemble.py',
                 HERE/'random-runner-qualification-v2/test_assemble.py',
                 HERE/'random-runner-qualification-v2/adapter-controls.json',
                 ROOT/'.benchmark-work/final-release-audit-20261021-v1/qcd1-random-science-audit-v1.json',
                 ROOT/'.benchmark-work/final-release-audit-20261021-v1/audit-light-eight-arithmetic-v1.py'):
        take(path,'ownership or software control')
    dependency=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/paper-comparison/dp-origin-card-policy-software-v1/manifest.json'
    take(dependency,'tracked DP71 software capsule')
    take(Path(__file__),'finite curation source')
    destination.mkdir();(destination/'data').mkdir()
    for entry in files.values():
        source=data_path(entry['original']);data=source.read_bytes()
        payload=gzip.compress(data,compresslevel=9,mtime=0) if entry['encoding']=='gzip' else data
        if sha(payload)!=entry['copy']['sha256']:raise ValueError('Source changed before mirror write')
        with (ROOT/entry['copy']['path']).open('xb') as stream:stream.write(payload)
    (destination/'relocation-map.json').write_text(json.dumps(aliases,indent=2,sort_keys=True)+'\n')
    note='''# Final lightweight native-DP observations\n\nEight selected processes, Ninja6/NinjaRS8, exactly 100 shared points each.\nBorn cases use full sums; loop-induced cases use the preregistered all-minus\nhelicity. The empirical fraction resolution is 1%, and a zero count has a\none-sided 95% upper bound of approximately 2.95%. No heavy full100 extension or\nruntime acceptance claim is part of this capsule.\n\nOriginal JSON/raw bytes and their private paths are preserved. The finite\ninventory maps each original file to an exact copy. Large repetitive guard\nlogs are gzip mirrors with their original uncompressed hashes; standard gzip\ndecompression recovers the original bytes. No executable, object, module,\nlibrary or generated build tree is included.\n\nThis is an observations/audit archive. Full source/runtime qualification used\nthe retained private compiled targets; a fresh checkout must rebuild and rerun\nthe documented process commands to repeat that physical qualification. The\ntracked DP71 software capsule and final report contain the method and recipe.\nThe numerical observations and independent arithmetic audit can be inspected\nwithout claiming a fresh native-runtime validation.\n'''
    note+=('\nHardware IEEE flags remain separate diagnostics. NRS8 has IEEE-invalid '
        'set on all 300 observations of qcd2/ew2/qcd3/ew3/ggHH, with finite '
        'coefficients/native success and zero common or algorithmic FPE. Matching '
        'plain-prefix flags establish that the recorder did not introduce them. '
        'This counts observations with a sticky flag, not invalid operations.\n')
    (destination/'README.md').write_text(note)
    inventory=dict(schema='lightweight-final-science-capsule-v1',requested_cases=list(CASES),requested_tools=[6,8],
        points_per_case=100,unique_physical_points=800,science_observations=4800,
        source_runtime_qualified=True,independent_audits=[binding(p) for p in audits],
        performance_claimed=False,portable_full_runtime_qualification=False,files=files,
        readme=binding(destination/'README.md'),relocation_map=binding(destination/'relocation-map.json'))
    (destination/'inventory.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
    print(json.dumps(binding(destination/'inventory.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--audit',type=Path,action='append',required=True)
    args=parser.parse_args();curate(args.destination.resolve(),[p.resolve() for p in args.audit])
