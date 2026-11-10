#!/usr/bin/env python3
"""Pure replay of the sealed static-audit packet; no workspace/native reads."""
from pathlib import Path
import gzip
import hashlib
import json

def digest(data): return hashlib.sha256(data).hexdigest()

def verify(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    index=json.loads((root/'evidence-index.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual==set(index['files'])|{'evidence-index.json'}
    payload={}
    for name,item in index['files'].items():
        path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts and str(path)==name
        current=root
        for part in path.parts:
            current=current/part
            assert not current.is_symlink()
        data=current.read_bytes()
        assert len(data)==item['bytes'] and digest(data)==item['sha256'],name
        payload[name]=data
    j=lambda name:json.loads(payload[name])
    build=j('build.json');summary=j('summary.json');plan=j('plan.json')
    assert build['passed'] is True and summary['passed'] is True
    assert build['elapsed_seconds']<=120 and len(plan)==4
    assert build['source_index_sha256']==digest(payload['source-index.json'])
    assert summary['source_index_sha256']==build['source_index_sha256']
    assert summary['executable_sha256']==build['executable_sha256']
    previous=None
    for i,argv in enumerate(plan,1):
        r=j(f'command-{i}.json')
        assert r['argv'][-len(argv):]==argv and r['argv'][1:3]==['-c','12']
        assert r['argv'][4:6]==['--as=4294967296','--']
        assert r['returncode']==0 and r['process_reaped'] is True and r['native'] is False
        assert not r.get('failure') and not r.get('timed_out')
        assert r['target_loader_environment_at_execve']=={} and r['seconds_cap']<=120
        if previous is not None: assert r['started_unix']>=previous
        previous=r['finished_unix']
    compression=j('map-compression.json');raw=gzip.decompress(payload['link.map.gz'])
    assert digest(raw)==compression['raw_sha256']==build['link_map_sha256']==summary['link_map_sha256']
    assert digest(payload['link.map.gz'])==compression['gzip_sha256']
    assert len(raw)==compression['raw_bytes'] and len(payload['link.map.gz'])==compression['gzip_bytes']
    map_loads=[str(Path(line[5:])) for line in raw.decode().splitlines() if line.startswith('LOAD ')]
    loads=j('link-loads.json');assert map_loads==[x['path'] for x in loads]
    assert len(loads)==summary['load_records']==81
    assert len({x['resolved_path'] for x in loads})==summary['unique_resolved_loads']==55
    assert {p:h for p,h in build['original_objects'].items()}=={
        x['path']:x['sha256'] for x in loads if x['path'] in build['original_objects']}
    wrappers=j('wrapper-audit.json');expected=set(wrappers['expected_wrappers']);assert len(expected)==14
    relocations=wrappers['object_relocations'];edges=wrappers['linked_edges']
    assert {x['target'] for x in relocations if x['target_undefined_in_object']}==expected
    assert len(relocations)==summary['matching_object_relocations']==18
    assert sum(x['target_undefined_in_object'] for x in relocations)==summary['external_relocations']==17
    assert {'__wrap_'+x for x in expected}<={x['target'] for x in edges if not x['caller'].startswith('__wrap_')}
    assert all(any(x['caller']=='__wrap_'+name and x['target']==name for x in edges) for name in expected)
    assert len(edges)==summary['linked_edges']==32
    for name in expected:
        lines=[line for line in payload['executable-symbols.txt'].decode().splitlines() if line.split()[-1]=='__wrap_'+name]
        assert len(lines)==1 and lines[0].split()[-2]=='T'
    assert build['target_interpreter'] in payload['executable-elf.txt'].decode()
    return {'passed':True,'scope':summary['scope'],'payload_files':len(payload),'wrappers':len(expected),
            'external_relocations':17,'same_object_relocations':1,'runtime_executed':False}

if __name__=='__main__': print(json.dumps(verify(),indent=2))
