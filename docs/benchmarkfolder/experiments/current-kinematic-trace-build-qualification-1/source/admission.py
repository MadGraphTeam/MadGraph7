"""Source-only binding primitives. Importing this module runs no child process."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT/'.tirust-build/kinematic-provenance-current-proposal-1/proposal'
HISTORICAL = ROOT/'.tirust-build/kinematic-provenance-prototype-1'
WORK = HERE.parent/'work'
RESULTS = HERE.parent/'results'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': sha(raw)}


def verify_record(path, expected):
    raw = Path(path).read_bytes()
    require(len(raw) == expected['bytes'] and sha(raw) == expected['sha256'],
            'Bound bytes changed: '+str(path))
    return raw


def read_json(path):
    return json.loads(Path(path).read_bytes())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        json.dump(data, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')


def sealed(index_path, expected=None):
    raw = Path(index_path).read_bytes()
    if expected is not None:
        require(sha(raw) == expected, 'Reviewed source index differs')
    data = json.loads(raw)
    for name, pin in data['files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts,
                'Unsafe source index path')
        verify_record(Path(index_path).parent/path, pin)
    return data


def helpers():
    """Hash every reused helper before executing any of its Python source."""
    data = read_json(HERE/'helper-bindings.json')
    for name, pin in data['files'].items():
        verify_record(ROOT/name, pin)
    return data


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def historical_modules():
    helpers()
    load('routing', HISTORICAL/'routing.py')
    saved = load('saved_source', HISTORICAL/'saved_source.py')
    trace = load('trace_insertions_original', HISTORICAL/'propose_trace.py')
    return saved, trace


def initial_admission():
    amendment = read_json(HERE/'amendment.json')
    for key in ('parent_index', 'historical_index'):
        pin = amendment[key]
        verify_record(ROOT/pin['path'], pin)
        sealed(ROOT/pin['path'], pin['sha256'])
    helpers()
    require(amendment['initial_native_modes'] == [1, 4], 'Wrong initial modes')
    require(amendment['maximum_workers'] == 30, 'Wrong worker ceiling')
    require(amendment['maximum_outer_amplitude_entries']==60 and
            amendment['bootstrap_outer_entries_per_worker']==1 and
            amendment['measurement_outer_entries_per_worker']==1 and
            amendment['LoopPoleMode']==0 and amendment['force_stability_check'] is False,
            'Superseded diagnostic policy cannot be executed')
    return amendment
