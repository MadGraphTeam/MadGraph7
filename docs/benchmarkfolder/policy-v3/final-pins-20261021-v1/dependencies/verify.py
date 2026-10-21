"""Verify this source-provenance capsule offline; runs no numerical engines."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid manifest path: {relative}')
        data = (root / relative).read_bytes()
        if len(data) != entry['bytes']:
            raise ValueError(f'Length mismatch: {relative}')
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Hash mismatch: {relative}')
    pins = json.loads((root / 'pins.json').read_text())
    if pins['numerical_validation_claimed']:
        raise ValueError('This capsule does not certify numerical validation')
    print(f'PASS: {len(manifest["files"])} source-provenance files; no engines executed')


if __name__ == '__main__':
    main()
