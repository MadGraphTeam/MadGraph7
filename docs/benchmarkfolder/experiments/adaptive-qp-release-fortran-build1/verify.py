#!/usr/bin/env python3
"""Read-only build provenance verification; never import a numerical library."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    index = json.loads((HERE/'evidence-index.json').read_text())['files']
    paths = list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()} == set(index)|{'evidence-index.json'}
    for name, entry in index.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=HERE/name
        assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
        assert not p.read_bytes().startswith(b'\x7fELF') and p.suffix not in ('.o','.so','.a','.mod','.pyc')
    b=json.loads((HERE/'bindings.json').read_text());r=json.loads((HERE/'build.json').read_text())
    assert r['complete'] and r['source_index_sha256']==b['source_index_sha256']
    assert r['source_index_sha256']=='f85e28784762a70806398ab5b878f93f01c0a1ca51475d960c50036a9ad7995f'
    assert r['cpu']==12 and not r['cpu_reservation'] and r['jobs']==1
    assert r['shared_cap_seconds']==120 and r['rlimit_as']==[4294967296]*2
    assert r['elapsed_seconds']<120 and len(r['commands'])==1
    c=r['commands'][0];p=c['progress']
    assert c['exit_code']==p['returncode']==0 and p['process_spawned'] and p['process_reaped']
    assert p['process_wall_seconds']<120
    assert r['protected_inputs_before']==r['protected_inputs_after']
    assert r['legacy_executable_mode_before']==r['legacy_executable_mode_after']==0o700
    assert not any(b[k] for k in ('loader_executed','native_executed','physics_qualified','production_promoted'))
    texts={}
    for name,entry in b['copies'].items():
        data=(HERE/name).read_bytes();assert hashlib.sha256(data).hexdigest()==entry['sha256']
        raw=gzip.decompress(data) if entry['gzip'] else data
        assert hashlib.sha256(raw).hexdigest()==entry['raw_sha256'] and len(raw)==entry['raw_bytes']
        texts[Path(entry['path']).name]=raw
    work=ROOT/b['work_directory'];dso=ROOT/b['selected_release_dso']
    assert c['argv'].count(str(dso))==1
    loads=set(re.findall(r'^LOAD\s+(.+)$',texts['adaptive.map'].decode(),re.M))
    assert str(dso) in loads and r['link_inputs']['adaptive'][str(dso)]==b['selected_release_sha256']
    objects=[str(work/(n+'.o')) for n in ('mtirust','qp_tier_records','driver','optional_getter','observer')]
    assert [x for x in c['argv'] if x.endswith('.o')]==objects
    for obj in objects:
        name=Path(obj).name
        assert obj in loads and r['artifacts'][name]==r['reused_artifacts_sha256'][name]==r['link_inputs']['adaptive'][obj]
    assert c['argv'].count('-Wl,--wrap=ml_tirust_evaluate_mp')==1
    assert c['argv'].count('-Wl,--wrap=ml_tirust_private_qp_precision_snapshot')==1
    assert not any('--wrap=ml_tirust_qp_precision_statistics' in x for x in c['argv'])
    if '--live' in sys.argv:
        assert sha(ROOT/b['source_index_path'])==b['source_index_sha256']
        for name,digest in r['protected_inputs_after'].items():assert sha(Path(name))==digest,name
        for mapping in r['link_inputs'].values():
            for name,digest in mapping.items():assert sha(Path(name))==digest,name
        for name,digest in r['artifacts'].items():assert sha(work/name)==digest,name
        assert (work/'legacy').stat().st_mode & 0o7777==0o700 and os.access(work/'legacy',os.X_OK)
        assert sha(dso)==b['selected_release_sha256']
    print(json.dumps({'verification':'pass','portable_files':len(index),'live_pins':len(r['protected_inputs_after']) if '--live' in sys.argv else None,'native_executed':False}))

if __name__=='__main__':main()
