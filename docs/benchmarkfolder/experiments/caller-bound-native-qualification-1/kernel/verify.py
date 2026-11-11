"""Read-only source/pure-data replay. No compiler, loader, or reducer calls."""
import hashlib
import json
from pathlib import Path
from derive_masks import derive, ROUTING


def binding(path):
    data = path.read_bytes()
    return {'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()}


def verify():
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    index = json.loads((here/'index.json').read_text())
    assert {x.name for x in here.iterdir()} == set(index['files'])|{'index.json'}
    for name, expected in index['files'].items():
        path=here/name
        assert path.is_file() and not path.is_symlink()
        assert binding(path)==expected, name
    pins=json.loads((here/'pins.json').read_text())
    for name,expected in pins['files'].items():
        assert not name.startswith('/') and '..' not in Path(name).parts
        assert binding(repo/name)==expected, name
    derived=repo/pins['derived_root']
    generated=json.loads((derived/'cms-source-bindings.json').read_text())
    for name,expected in generated['files'].items():
        assert not name.startswith('/') and '..' not in Path(name).parts
        assert binding(repo/name)==expected, name
    masks=derive(json.loads((repo/ROUTING).read_text()))
    assert masks==json.loads((here/'cms-structural-masks.json').read_text())
    return {'passed':True, 'payloads':len(index['files']),
            'pinned_source_evidence_files':len(pins['files']),
            'original_generated_source_files':len(generated['files']),
            'source_routes_replayed':len(masks['routes']),
            'Fortran_compiled':False,'native_reducer_executions':0,
            'accuracy_qualified':False,'runtime_caller_integration_complete':False}


if __name__=='__main__':
    print(json.dumps(verify(),sort_keys=True))
