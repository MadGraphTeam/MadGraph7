"""Read-only parent closure for this private source proposal."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT1 = ROOT / '.tirust-build/generic-invariant-provenance-proposal-1'
PARENT2 = ROOT / '.tirust-build/generic-invariant-provenance-proposal-2'


def bound(path, pin):
    data = path.read_bytes()
    expected = pin if isinstance(pin, str) else pin['sha256']
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('changed parent source: ' + str(path))
    if isinstance(pin, dict) and len(data) != pin['bytes']:
        raise ValueError('changed parent source size: ' + str(path))
    return data


def verify_parents(live=False):
    for name, pin in json.loads((HERE / 'parent-pins.json').read_bytes()).items():
        bound(ROOT / name, pin)
    total = 0
    for parent in (PARENT1, PARENT2):
        index = json.loads((parent / 'source-index.json').read_bytes())
        for name, pin in index['files'].items():
            bound(parent / name, pin)
            total += 1
    if live:
        current = json.loads((PARENT1 / 'source-pins.json').read_bytes())['current_source_files']
        for name, pin in current.items():
            bound(ROOT / name, pin)
        model = json.loads((PARENT2 / 'model-source-pins.json').read_bytes())
        for name, pin in model.items():
            bound(ROOT / name, pin)
    return {'parent_payloads': total, 'live_source_check': live}
