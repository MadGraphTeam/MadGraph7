#!/usr/bin/env python3
"""Retain a terminal guarded replay as text; never execute or load native code."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROPOSAL = 'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-physical-proposal-2'
WORK = '.tirust-build/cayley-guarded-adaptive-physical-proposal-2'
RUN = WORK + '/run1'
SOURCE_SHA = '7d9a40657b9df702fe1bdd1b3f302363813b1ba4a9514a2bcc88467f371dcd1c'
BINDING_SHA = 'ee2933ed02f2f2c6ba1b004c0feb951553275c2967f829b5bb0efb894de7c61a'
BUILD_QUAL = 'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-qualification-1'
BUILD_QUAL_SHA = '94f7a23d4c10373353b04e876da32585bc0274852fc573e8d94b740c9fa0ae3e'
NATIVE_SUFFIXES = {'.so', '.a', '.o', '.mod', '.rlib', '.pyc'}

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--terminal-confirmed', required=True,
                     help='Root-owned terminal session identity, recorded verbatim.')
    args = cli.parse_args()
    assert Path.cwd().resolve() == ROOT
    assert not (HERE / 'manifest.json').exists(), 'One terminal capture only; no overwrite/retry.'
    assert args.terminal_confirmed.strip()
    blobs, native = {}, {}

    def key_for(path):
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    def add(path, expected=None):
        path = Path(path)
        key = key_for(path)
        # Native and external dependencies are identities only, from frozen pins
        # or the completed runtime closure. No native bytes are read here.
        if not path.is_relative_to(ROOT) or path.suffix in NATIVE_SUFFIXES or '.so.' in path.name or path.name in {'checked','guarded-qp'}:
            assert expected, ('Missing native identity', key)
            native[key] = {'sha256': expected, 'bytes': path.stat().st_size,
                           'payload_retained': False, 'hash_origin': 'frozen-input-or-terminal-runtime-closure'}
            return
        assert not path.is_symlink(), key
        raw = path.read_bytes()
        assert not raw.startswith((b'\x7fELF', b'!<arch>')), key
        raw.decode('utf8')
        actual = digest(raw)
        assert expected is None or actual == expected, key
        if key in blobs:
            assert blobs[key] == raw
        blobs[key] = raw

    add(ROOT / PROPOSAL / 'source-index.json', SOURCE_SHA)
    source = json.loads(blobs[PROPOSAL + '/source-index.json'])
    for name, sha in source['sources'].items():
        add(ROOT / PROPOSAL / name, sha)
    for name, sha in source['dependencies'].items():
        add(Path(name) if Path(name).is_absolute() else ROOT / name, sha)
    add(ROOT / WORK / 'binding1.json', BINDING_SHA)
    add(ROOT / WORK / 'preparation-failure-1/failure.json')
    add(ROOT / BUILD_QUAL / 'evidence-index.json', BUILD_QUAL_SHA)
    add(ROOT / BUILD_QUAL / 'README.md')
    binding = json.loads(blobs[WORK + '/binding1.json'])
    add(Path(binding['build_receipt_path']), binding['build_receipt_sha256'])
    execution = ROOT / RUN / 'execution.json'
    raw_execution = execution.read_bytes()
    record = json.loads(raw_execution)
    assert 'elapsed_seconds' in record, 'No closed final publication yet.'
    assert record['source_index_sha256'] == SOURCE_SHA
    assert record['artifact_binding_sha256'] == BINDING_SHA
    assert record['guarded_library_sha256'] == binding['library']['sha256']
    # Every spawned/reaped outcome is retained even when SIGFPE or timeout failed
    # a numerical gate. Root supplies terminal admission, not a success assertion.
    for row in [record.get('loader', {})] + list(record['workers'].values()):
        progress = row.get('progress', {})
        assert not progress.get('process_spawned') or progress.get('process_reaped'), row
    for name, sha in record.get('runtime_sha256_after', {}).items():
        add(Path(name), sha)
    for path in sorted((ROOT / RUN).rglob('*')):
        if path.is_file() and 'runtime-guarded' not in path.parts:
            add(path)
    for name, row in record['workers'].items():
        for key, sha in row.get('complete_case_files_sha256', {}).items():
            assert digest(blobs[key]) == sha, key
    oldrun = ROOT / '.tirust-build/adaptive-qp-checked-trap-proposal-1/run1'
    for case in ('original-masked','original-trapped','adaptive-masked','adaptive-trapped'):
        for name in ('checked-control.log','stdout.log','native-maps.txt','named-control.json'):
            path = oldrun / case / name
            if path.is_file():
                add(path)
    assert execution.read_bytes() == raw_execution, 'Terminal receipt drifted.'

    payload_dir = HERE / 'payloads'
    payload_dir.mkdir(exist_ok=False)
    text_payloads = {}
    for key, raw in sorted(blobs.items()):
        text = raw.decode('utf8')
        pieces = [text[offset:offset+131072].encode('utf8') for offset in range(0,len(text),131072)] or [b'']
        chunks = []
        for piece in pieces:
            packed = gzip.compress(piece, compresslevel=9, mtime=0)
            assert len(packed) <= 1048576
            name = 'payloads/' + digest(piece) + '.gz'
            path = HERE / name
            if path.exists():
                assert path.read_bytes() == packed
            else:
                path.write_bytes(packed)
            chunks.append(name)
        text_payloads[key] = {'sha256': digest(raw), 'bytes': len(raw), 'chunks': chunks}
    manifest = {'schema':'guarded-physical-portable-text-v1',
                'terminal_confirmation': args.terminal_confirmed,
                'source_index_sha256':SOURCE_SHA, 'artifact_binding_sha256':BINDING_SHA,
                'execution_sha256':digest(raw_execution),
                'text_payloads':text_payloads, 'native_identities':native,
                'native_payloads_retained':False, 'historical_native_paths_required':False,
                'native_loader_or_process_executed_by_capture':False,
                'runtime_failure_retained':record.get('failure'),
                'execution_complete_as_recorded':record['execution_complete']}
    save(HERE / 'manifest.json', manifest)
    print(json.dumps({'text_inputs':len(blobs),'native_identity_only':len(native),
                      'execution_sha256':digest(raw_execution),
                      'execution_complete_as_recorded':record['execution_complete']},indent=2))

if __name__ == '__main__':
    main()
