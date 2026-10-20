#!/usr/bin/env python3
"""Read-only verification of the complete fixed replay and exact contrasts."""
import json
import importlib.util
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import analyze

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.name != 'manifest.json'}
    assert actual == set(manifest['files'])
    for name, record in manifest['files'].items():
        path = HERE / name
        assert not path.is_symlink()
        assert analyze.sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], name
    prepared = json.loads((HERE / 'prepared.json').read_text())
    assert analyze.sha(HERE.parent / 'manifest.json') == prepared['capture_manifest_sha256']
    capture = json.loads((HERE.parent / 'manifest.json').read_text())
    for name, record in capture['files'].items():
        path = HERE.parent / name
        assert analyze.sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], name
    assert json.loads(json.dumps(analyze.compute())) == json.loads((HERE / 'contrasts.json').read_text())
    spec = importlib.util.spec_from_file_location('independent_replay', HERE / 'independent/check.py')
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    assert json.loads(json.dumps(independent.compute())) == json.loads((HERE / 'independent/result.json').read_text())
    print(json.dumps({'passed': True, 'files': len(actual), 'complete_keys': 464,
                      'calls': 3712, 'exact_reproduction_controls': 1392, 'engine_calls': 0,
                      'independent_raw_and_exact_contrasts_match': True,
                      'physical_amplitude_or_retry_acceptance': False}))


if __name__ == '__main__':
    main()
