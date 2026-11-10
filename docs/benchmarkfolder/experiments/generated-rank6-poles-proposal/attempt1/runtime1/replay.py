"""Verify and reparse the retained attempt entirely from portable text files."""
import gzip
import hashlib
import json
from pathlib import Path
import runpy

HERE=Path(__file__).resolve().parent

def sha(raw):return hashlib.sha256(raw).hexdigest()

def replay():
    index=json.loads((HERE/'evidence-index.json').read_text())
    for name,digest in index['files'].items():
        path=HERE/name
        assert not path.is_symlink() and sha(path.read_bytes())==digest,name
    for name,digest in index['source_files'].items():assert sha((HERE.parent/name).read_bytes())==digest,name
    copies=json.loads((HERE/'copies.json').read_text());raw={}
    for name,item in copies.items():
        data=(HERE/item['path']).read_bytes()
        assert sha(data)==item['sha256']
        if item['compression']=='gzip':data=gzip.decompress(data)
        assert len(data)==item['raw_bytes'] and sha(data)==item['raw_sha256']
        data.decode('utf8');assert b'\0' not in data
        raw[name]=data
    execution=json.loads(raw['execution.json'])
    assert execution['runner_sha256']=='8fd4ee125e8169f1e8cd232c16cca2ab58ac0844fdb9083c75553448eeffd8d2'
    assert execution['failure']=="ValueError('Complex mass mutation differs')" and not execution['complete']
    for item in execution['workers'].values():
        assert item['exit_code']==item['loader']['exit_code']==0
        assert item['progress']['process_reaped'] and item['loader']['progress']['process_reaped']
        assert item['progress']['process_wall_seconds']<=900 and item['loader']['progress']['process_wall_seconds']<=30
        assert item['mapping']['cpu_allowed_list']=='12'
        assert item['mapping']['expected_runtime_sha256']==item['mapping']['mapped_files_sha256_after_exit']==item['runtime_sha256']
    class TextPath:
        def __init__(self,name):self.name=name
        def __truediv__(self,name):return TextPath(self.name+'/'+name)
        def read_text(self):return raw[self.name].decode('utf8')
        def __str__(self):return self.name
    result=runpy.run_path(str(HERE.parent/'post-analysis1/analyze.py'))['analyze']([TextPath('ew'),TextPath('qcd')])
    assert result==json.loads((HERE.parent/'post-analysis1/results.json').read_text())
    return {'passed':True,'scope':'Portable raw-text replay; native qualification retains open items.',
            'files':len(copies),'result_status':result['status'],'open_items':len(result['open_items'])}

if __name__=='__main__':print(json.dumps(replay(),sort_keys=True))
