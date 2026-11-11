"""Read-only packet hashes, authenticated metadata replay and exact emission."""
import hashlib
import json
from pathlib import Path
from derive_dag import derive
from emit_data import emit


def binding(path):
    data=path.read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def verify():
    here=Path(__file__).resolve().parent;repo=here.parents[1]
    index=json.loads((here/'index.json').read_text())
    assert {p.name for p in here.iterdir()}==set(index['files'])|{'index.json'}
    for name,expected in index['files'].items():
        p=here/name;assert p.is_file() and not p.is_symlink()
        assert binding(p)==expected,name
    for name,expected in index['input_pins'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert binding(repo/name)==expected,name
    d=derive();assert d==json.loads((here/'dag-metadata.json').read_text())
    assert emit(d)==(here/'cms_bounds_data.inc').read_text()
    return {'passed':True,'source_files_verified':d['source_files_verified'],
        'routes':len(d['routes']),'supported_modes':sum(m['supported'] for r in d['routes'] for m in r['modes'].values()),
        'compiled':False,'native_executions':0,'runtime_environment_qualified':False}


if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
