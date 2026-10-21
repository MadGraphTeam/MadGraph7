"""Shared paths and write-once evidence helpers; no compilation in this study."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
WORK = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-process-v2'
OLD = HERE.parent / 'ninjars-mp-tensor-allocation-process-v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def binding(path):
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size}

def sources(directory):
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}

def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True)
        stream.write('\n')
