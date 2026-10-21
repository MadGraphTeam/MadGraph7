#!/usr/bin/env python3
"""Seal/verify complete retained native evidence without executing numerics."""
import sys
sys.dont_write_bytecode = True
import argparse
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE/'manifest.json'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def payload(plain_paths):
    files = {}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file(): continue
        name = str(path.relative_to(HERE))
        if name == 'manifest.json' or name in plain_paths: continue
        assert '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'), name
        assert not path.is_symlink(), name
        with path.open('rb') as stream: magic = stream.read(8)
        assert not magic.startswith(b'\x7fELF') and magic != b'!<arch>\n', name
        files[name] = {'bytes':path.stat().st_size,'sha256':sha(path)}
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('seal','verify'))
    args = parser.parse_args()
    compressed_path = HERE/'compressed-artifacts.json'
    compressed = json.loads(compressed_path.read_text())
    assert compressed['complete'] and len(compressed['files']) == 6
    plain_paths = set(compressed['files'])
    # Bind materialized working files if they exist, but never include the
    # duplicate plaintext payload in Git. The deterministic archives preserve
    # each original byte sequence and its independent decompressed identity.
    for name,record in compressed['files'].items():
        archive = HERE/record['compressed']['path']
        assert sha(archive) == record['compressed']['sha256']
        assert archive.stat().st_size == record['compressed']['bytes']
        plain = HERE/name
        if plain.exists():
            assert sha(plain) == record['uncompressed']['sha256']
            assert plain.stat().st_size == record['uncompressed']['bytes']
    if args.action == 'seal':
        assert not MANIFEST.exists()
        assert (HERE/'screening-independent-audit.json').is_file()
        screening = json.loads((HERE/'supplement/run-screening/result.json').read_text())
        selection = json.loads((HERE/'selection-analysis.json').read_text())
        assert screening['complete'] and screening['returncode'] == 0
        assert selection['complete'] and selection['provisional_request_screen_choice'] == 'Interpreted'
        manifest = {'schema':'native-float256-complete-evidence-v1','complete':True,
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'outcome':'Retain interpreted production; neither native policy demonstrates request-scope gain',
            'production_native_patch_applied':False,'additional_numerical_samples_authorized':False,
            'materialized_plaintext_exclusions':sorted(plain_paths),
            'compressed_index_sha256':sha(compressed_path),
            'final_generic_test_patch_sha256':sha(HERE/'production-proposals-v1/final-tests.patch'),
            'files':payload(plain_paths)}
        manifest['identity'] = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    else: manifest = json.loads(MANIFEST.read_text())
    identity = manifest.pop('identity')
    assert hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest() == identity
    assert set(manifest['materialized_plaintext_exclusions']) == plain_paths
    assert manifest['compressed_index_sha256'] == sha(compressed_path)
    assert payload(plain_paths) == manifest['files']
    print(json.dumps({'complete':True,'action':args.action,'identity':identity,
        'manifest_sha256':sha(MANIFEST),'files':len(manifest['files']),
        'retained_bytes':sum(row['bytes'] for row in manifest['files'].values()),
        'engine_calls':0,'production_changed':False}))


if __name__ == '__main__': main()
