#!/usr/bin/env python3
"""Append portable retention to the closed proof; never execute its binaries."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
CLOSED = ROOT / '.tirust-build/subprocess-cache-successor-results-2'
REPORT = ROOT / '.tirust-build/subprocess-cache-successor-retention-1/capsule'
SOURCE = ROOT / '.tirust-build/subprocess-cache-successor-proposal-2'
SOURCE1 = ROOT / '.tirust-build/subprocess-cache-successor-proposal-1'
OUTPUT = ROOT / '.tirust-build/subprocess-cache-successor-work-2/export'
KEEP_PLAIN = {'materialized-source.json', 'source-freeze.json', 'qualification-summary.json', 'verify_evidence.py', 'README.md'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    if not args.execute:
        print(json.dumps({'source_only_default': True, 'report': str(REPORT), 'requires_closed_stages': 7,
                          'native_execution': False}, indent=2))
        return
    assert not REPORT.exists(), 'Fresh private capture required'
    shutil.copytree(CLOSED, REPORT)
    stages = {name: json.loads((REPORT / ('stage-' + name + '.json')).read_text())
              for name in ('freeze', 'materialize', 'generate', 'prepare-build', 'build', 'smoke', 'run')}
    assert all(item['status'] == 'passed' for item in stages.values())
    assert sha((SOURCE / 'source-index.json').read_bytes()) == '3d1f787bae4d5af45c2a34119f46b3235e64087224c698fca3eb0eb66816ada7'
    assert json.loads((REPORT / 'results.json').read_text())['passed']
    # Save the closed initial file set before appending any retention files.
    original = {str(p.relative_to(REPORT)): file_sha(p) for p in REPORT.rglob('*') if p.is_file()}
    assert not any(p.is_symlink() for p in REPORT.rglob('*'))
    generated = json.loads((REPORT / 'generated-source-freeze.json').read_text())
    build = json.loads((REPORT / 'build.json').read_text())
    def append(name, raw):
        raw.decode('utf-8'); assert b'\0' not in raw
        path = REPORT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    def save(name, value):
        append(name, (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())
    def copy_sealed_source(source, target):
        index = json.loads((source/'source-index.json').read_text())
        for name in [*index['files'], 'source-index.json']:
            raw = (source/name).read_bytes()
            if name != 'source-index.json':
                assert len(raw) == index['files'][name]['bytes']
                assert sha(raw) == index['files'][name]['sha256']
            append(target + '/' + name, raw)
    copy_sealed_source(SOURCE, 'source-proposal')
    copy_sealed_source(SOURCE1, 'source1-unexecuted')
    for rel in ('subprocess-cache-successor-source2-tests-1.log',
                'subprocess-cache-successor-first5-execution-1.json',
                'subprocess-cache-successor-last2-execution-1.json',
                'subprocess-cache-successor-root-build-review-1.json'):
        append('admission/' + rel, (ROOT/'.tirust-build'/rel).read_bytes())
    for name in generated['rendered']:
        raw = (OUTPUT / 'SubProcesses' / name).read_bytes()
        assert sha(raw) == generated['rendered'][name]
        append('retained-rendered/' + name, raw)
    retained_generated = {}
    for name, item in generated['source_inventory'].items():
        path = OUTPUT / name
        if item['kind'] == 'link':
            assert path.is_symlink() and str(path.readlink()) == item['target']
        if 'sha256' in item:
            raw = path.read_bytes()
            assert sha(raw) == item['sha256'] and len(raw) == item['size']
            try:
                raw.decode('utf-8'); assert b'\0' not in raw
                append('retained-generated/' + name, raw)
                retained_generated[name] = {'retained': True, 'sha256': sha(raw)}
            except (UnicodeDecodeError, AssertionError):
                assert path.suffix not in ('.f', '.f90', '.inc', '.dat', '.py', '.h', '.c')
                retained_generated[name] = {'retained': False, 'reason': 'Binary represented by source inventory hash only', 'sha256': sha(raw)}
    save('retained-generated-index.json', retained_generated)
    map_resolution = {'scope': 'Post-run resolution of closed actual linker LOAD spellings against unchanged prebound inputs.', 'executables': {}}
    for variant, binary in build['executables'].items():
        target = 'lifecycle_' + ('card_smoke' if variant == 'card-smoke' else variant)
        map_path = REPORT / 'execution' / (target + '.map')
        loads = []
        for line in map_path.read_text().splitlines():
            if not line.startswith('LOAD '):
                continue
            spelling = line[len('LOAD '):]
            path = Path(spelling)
            if not path.is_absolute(): path = OUTPUT / 'SubProcesses' / path
            resolved = str(path.resolve()); checksum = file_sha(path)
            assert build['all_link_inputs_before_after'][resolved] == binary['link_loads'][resolved] == checksum
            loads.append({'spelling': spelling, 'resolved': resolved, 'sha256': checksum})
        map_resolution['executables'][variant] = {'map_sha256': file_sha(map_path), 'loads': loads}
    save('link-map-resolution.json', map_resolution)
    classification = {'scope': 'Post-run classification of read-only immutable Nix locale data; no additional ELF mapping admitted.', 'files': {}}
    for path in (REPORT / 'runs').glob('*/runtime-maps-*.txt'):
        variant = 'card-smoke' if path.parent.name == 'card-smoke' else path.parent.name.split('-')[0]
        binary = build['executables'][variant]
        expected = set(binary['runtime']['files']) | {binary['path']}
        for line in path.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith('/') or fields[5] in expected: continue
            spelling = fields[5]; mapped = Path(spelling); stat = mapped.stat()
            assert spelling.startswith('/nix/store/') and spelling.endswith('/lib/locale/locale-archive')
            assert fields[1] == 'r--p' and int(fields[4]) == stat.st_ino
            assert tuple(int(x, 16) for x in fields[3].split(':')) == (os.major(stat.st_dev), os.minor(stat.st_dev))
            with mapped.open('rb') as stream: header = stream.read(64)
            assert header[:4] != b'\x7fELF'
            item = {'sha256': file_sha(mapped), 'size': stat.st_size, 'inode': stat.st_ino,
                    'map_device': fields[3], 'header64_hex': header.hex()}
            assert spelling not in classification['files'] or classification['files'][spelling] == item
            classification['files'][spelling] = item
    save('non-elf-map-classification.json', classification)
    for name in ('verify_evidence.py', 'independent-review.json', 'independent-review.py', 'README.md'):
        append(name, (HERE / name).read_bytes())
    for name in ('lineage.json', 'portable-successor.patch', 'test_verify.py',
                 'verifier-tests.log', 'verifier-test-receipt.json',
                 'test_verify.attempt1.py', 'verifier-tests-attempt1.log'):
        if (HERE / name).is_file():
            append('retention-validation/' + name, (HERE / name).read_bytes())
    append('retention-capture.py', Path(__file__).read_bytes())
    save('retention-binding.json', {'schema': 1, 'closed_original_files': original,
         'capture_sha256': file_sha(Path(__file__)), 'no_native_execution': True,
         'preceding_lifecycle_index_sha256': '212b6046a48030f26e12facfbd85b4179df0036633f14bbbc38c2788df376964',
          'source1_unexecuted_index_sha256': '62b2af837f5192555a2258035a352a2e65e4d37dce4fe13b8935ed451fd39779'})
    for name, checksum in original.items():
        assert file_sha(REPORT / name) == checksum, 'Closed evidence changed during retention'
    files, compressed = {}, {}
    for path in sorted(REPORT.rglob('*')):
        if not path.is_file(): continue
        name = str(path.relative_to(REPORT)); raw = path.read_bytes()
        raw.decode('utf-8'); assert b'\0' not in raw
        if len(raw) > 65536 and name not in KEEP_PLAIN and path.suffix not in ('.py', '.f', '.f90', '.c', '.h'):
            packed = gzip.compress(raw, mtime=0)
            packed_name = 'compressed/' + name + '.gz'
            destination = REPORT / packed_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream: stream.write(packed)
            compressed[name] = {'path': packed_name, 'raw_sha256': sha(raw), 'raw_bytes': len(raw)}
            path.unlink()  # Only this fresh private capture; closed originals stay untouched.
            name, raw = packed_name, packed
        files[name] = {'kind': 'file', 'sha256': sha(raw), 'size': len(raw)}
    index = {'schema_version': 1, 'files': files, 'compressed_text': compressed,
             'scope': 'Closed current-source lifecycle proof; native binaries retained by hash only.'}
    spec = importlib.util.spec_from_file_location('new_lifecycle_portable_verifier', REPORT / 'verify_evidence.py')
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    def reader(name):
        if name == 'evidence-index.json': return (json.dumps(index, sort_keys=True, indent=2) + '\n').encode()
        return (REPORT / name).read_bytes()
    proof = verifier.verify(read_bytes=reader, derive_summary=True)
    save('qualification-summary.json', proof['summary'])
    raw = (REPORT / 'qualification-summary.json').read_bytes()
    index['files']['qualification-summary.json'] = {'kind': 'file', 'sha256': sha(raw), 'size': len(raw)}
    verifier.verify(read_bytes=reader)
    physical_files = {str(path.relative_to(REPORT)) for path in REPORT.rglob('*') if path.is_file()}
    assert physical_files == set(index['files']), 'Unindexed payload before seal'
    for name, checksum in original.items():
        assert file_sha(CLOSED/name) == checksum, 'Closed source changed during capture'
    save('evidence-index.json', index)
    proof = verifier.verify(REPORT)
    print(json.dumps({name: value for name, value in proof.items() if name != 'summary'}, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
