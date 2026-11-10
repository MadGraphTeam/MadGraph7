#!/usr/bin/env python3
"""Retain one closed build; inspect ELF bytes, never load or execute them."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
sys.dont_write_bytecode = True
ROOT = Path.cwd().resolve()
WORK = ROOT / '.tirust-build/cayley-guarded-adaptive-build-qualification-1'
OUT = ROOT / 'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-qualification-1'
PROPOSAL = ROOT / 'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-3'
ATTEMPT = ROOT / '.tirust-build/cayley-guarded-adaptive-build-proposal-3/attempt1'
DSO = '69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b'
SOURCE_INDEX = '39d6ba2bd3c7aef5c8f5008360661c988a93d86ab1265cb3bc815707233b9e72'
RECEIPT = '59167eec84a767f0712b6905285a99564458ff0667427de87ce1921534b46632'
LOG = '7ea452558263c98b91781e979f843c8cf236d951b89450cbbc436275945e8a3d'

def digest(raw): return hashlib.sha256(raw).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
def main():
    assert ROOT.name == 'madgraph7_madloop_tirust'
    receipt_raw = (ATTEMPT / 'execution.json').read_bytes()
    assert digest(receipt_raw) == RECEIPT
    record = json.loads(receipt_raw)
    assert record['complete'] and not record['loader_or_native_test_executed']
    assert len(record['protected_inputs_before']) == 1193
    assert record['protected_inputs_before'] == record['protected_inputs_after']
    assert digest((ATTEMPT / 'release-build.log').read_bytes()) == LOG
    assert digest((PROPOSAL / 'source-index.json').read_bytes()) == SOURCE_INDEX
    OUT.mkdir(exist_ok=False)
    copies = {}
    def retain(source, name, compressed=False):
        raw = source.read_bytes()
        if compressed:
            raw.decode('utf8')
            payload = gzip.compress(raw, mtime=0)
        else:
            payload = raw
            if source.suffix == '.gz': gzip.decompress(raw).decode('utf8')
            else: raw.decode('utf8')
        destination = OUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        copies[name] = {'origin': str(source.relative_to(ROOT)), 'origin_sha256': digest(raw),
                        'origin_bytes': len(raw), 'encoding': 'gzip-utf8' if compressed else 'exact'}
    source_index = json.loads((PROPOSAL / 'source-index.json').read_text())
    actual = {str(p.relative_to(PROPOSAL)) for p in PROPOSAL.rglob('*') if p.is_file()}
    assert actual == set(source_index['files']) | {'source-index.json'}
    for name, entry in source_index['files'].items():
        raw = (PROPOSAL / name).read_bytes()
        assert digest(raw) == entry['sha256'] and len(raw) == entry['bytes']
        retain(PROPOSAL / name, 'proposal3/' + name)
    retain(PROPOSAL / 'source-index.json', 'proposal3/source-index.json')
    for name in ('execution.json', 'release-build.log', 'copy-cache.log'):
        retain(ATTEMPT / name, 'attempt1/' + name + '.gz', compressed=True)
    reader_source = ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-rescue-private-release-build/elf_metadata.py'
    assert digest(reader_source.read_bytes()) == '3327e253c4ea1b581b252fc9547f9b7c3f71f610c131e8c23b0dc1039b26f73a'
    retain(reader_source, 'elf_metadata.py')
    reader_index = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-design/evidence-index.json'
    assert digest(reader_index.read_bytes()) == '51f45d9e6f2c13227dc504fa284cba967f947db0f073dfb18bb9bf924c28c298'
    retain(reader_index, 'reader-origin-index.json')
    spec = importlib.util.spec_from_file_location('retained_pure_elf_reader', OUT / 'elf_metadata.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    # Exactly one read of the current DSO, by the existing pure byte reader.
    metadata = reader.inspect(Path(record['private_library_not_loaded']['path']))
    assert metadata['sha256'] == DSO and metadata['bytes'] == 69162336
    dump(OUT / 'elf-metadata.json', metadata)
    source_map = json.loads((PROPOSAL / 'source-copies.json').read_text())
    source_work = ROOT / source_map['work_root']
    def keep(name):
        return (name.startswith('sources/adaptive-qp-rescue-prototype3/src/')
            or name in ('sources/adaptive-qp-rescue-prototype3/Cargo.toml', 'sources/adaptive-qp-rescue-prototype3/Cargo.lock', 'sources/adaptive-qp-rescue-prototype3/tirust.h')
            or name in ('sources/pole-development/Cargo.toml', 'sources/pole-development/Cargo.lock', 'sources/pole-development/rust-toolchain.toml', 'sources/pole-development/LICENSE')
            or any(name.startswith('sources/pole-development/' + crate + '/src/') or name == 'sources/pole-development/' + crate + '/Cargo.toml'
                   for crate in ('crates/iregi-rust', 'crates/iregi-masters', 'madloop-pole-kernels')))
    texts = {}
    for name, entry in source_map['copied'].items():
        if keep(name):
            raw = (source_work / name).read_bytes()
            assert len(raw) == entry['bytes'] and digest(raw) == entry['sha256']
            texts[name] = raw.decode('utf8')
    assert len(texts) == 133
    archive = json.dumps(texts, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n'
    chunks = []
    for number, offset in enumerate(range(0, len(archive), 2000000), 1):
        raw = archive[offset:offset + 2000000].encode('utf8')
        payload = gzip.compress(raw, mtime=0)
        assert len(payload) <= 1048576
        name = f'source-texts/part-{number:03d}.txt.gz'
        (OUT / name).parent.mkdir(exist_ok=True)
        (OUT / name).write_bytes(payload)
        chunks.append({'path': name, 'utf8_bytes': len(raw), 'utf8_sha256': digest(raw)})
    archive_raw = archive.encode('utf8')
    dump(OUT / 'source-texts.json', {'schema': 'focused-code-text-retention-v1', 'selected_count': len(texts),
         'selected_names': sorted(texts), 'archive_utf8_bytes': len(archive_raw), 'archive_utf8_sha256': digest(archive_raw), 'chunks': chunks,
         'excluded_hash_only': {name: entry for name, entry in source_map['copied'].items() if name not in texts},
         'selection': 'Bridge src/ABI header, direct core/master/kernel crate src and manifests, bridge/workspace locks and manifests, rust-toolchain and TIRust LICENSE. Unrelated examples, docs, render assets, fixtures and workspace crates are hash-only in the original 499-file identity table.'})
    guard = source_map['only_changed_source']
    original = ROOT / source_map['origin'][guard]['path']
    retain(original, 'original-gram.rs.gz', compressed=True)
    artifacts = []
    diagnostics = []
    finished = []
    for line in (ATTEMPT / 'release-build.log').read_text().splitlines():
        try: item = json.loads(line)
        except json.JSONDecodeError: continue
        if item.get('reason') == 'compiler-artifact': artifacts.append(item)
        elif item.get('reason') == 'compiler-message': diagnostics.append(item)
        elif item.get('reason') == 'build-finished': finished.append(item)
    assert artifacts == record['cargo_artifacts'] and finished == [{'reason': 'build-finished', 'success': True}]
    qualification = {'schema': 'guarded-adaptive-build-qualification-v1', 'scope': 'Private release compilation only',
        'source_index_sha256': SOURCE_INDEX, 'execution_sha256': RECEIPT, 'build_log_sha256': LOG,
        'library': record['private_library_not_loaded'], 'library_qualified_only_as': 'A newly compiled ELF shared object; not loaded or numerically exercised',
        'cargo_artifacts': len(artifacts), 'newly_compiled': sum(not a['fresh'] for a in artifacts), 'reused': sum(a['fresh'] for a in artifacts),
        'warning_diagnostics': sum(a['message']['level'] == 'warning' for a in diagnostics),
        'protected_binding_count': len(record['protected_inputs_before']), 'copied_source_identity_count': len(source_map['copied']),
        'retained_code_text_count': len(texts), 'sole_changed_source': guard,
        'compile_seconds_administrative_only': record['commands'][1]['progress']['process_wall_seconds'],
        'total_seconds_administrative_only': record['elapsed_seconds'],
        'native_library_loaded': False, 'native_tests_executed': False, 'installed_or_promoted': False,
        'precision_qualified': False, 'physical_accuracy_qualified': False, 'stability_qualified': False, 'runtime_performance_qualified': False}
    dump(OUT / 'qualification.json', qualification)
    dump(OUT / 'bindings.json', {'schema': 'portable-guarded-build-bindings-v1', 'copies': copies,
        'library': record['private_library_not_loaded'], 'historical_paths_required_by_default_verifier': False,
        'native_payloads_retained': False, 'current_DSO_read_count': 1, 'inspection': 'Existing pure ELF byte reader; no loader, ldd, ctypes, subprocess, debugger or new compiler/test process.'})
    shutil.copyfile(WORK / 'verify.py', OUT / 'verify.py')
    shutil.copyfile(WORK / 'package.py', OUT / 'retention-script.py')
    shutil.copyfile(WORK / 'README.md', OUT / 'README.md')
    files = {}
    for path in sorted(OUT.rglob('*')):
        assert not path.is_symlink()
        if path.is_file():
            raw = path.read_bytes()
            assert not raw.startswith(b'\x7fELF')
            assert path.suffix not in ('.so', '.rlib', '.o', '.a', '.mod', '.pyc')
            if path.suffix == '.gz': gzip.decompress(raw).decode('utf8')
            else: raw.decode('utf8')
            assert len(raw) <= 1048576, str(path)
            files[str(path.relative_to(OUT))] = {'bytes': len(raw), 'sha256': digest(raw)}
    dump(OUT / 'evidence-index.json', {'schema': 'portable-guarded-adaptive-build-evidence-v1',
         'scope': 'Closed private release build, source provenance and pure ELF metadata; no numerical, precision, stability or performance acceptance', 'files': files})
    names = sorted(str((OUT / name).relative_to(ROOT)) for name in files)
    names.append(str((OUT / 'evidence-index.json').relative_to(ROOT)))
    (WORK / 'stage-paths.txt').write_text('\n'.join(sorted(names)) + '\n')
    print(json.dumps({'payload_files': len(files), 'index_sha256': digest((OUT / 'evidence-index.json').read_bytes()),
                      'retained_source_texts': len(texts), 'source_chunks': len(chunks), 'private_library_sha256': DSO, 'native_loaded': False}, sort_keys=True))
if __name__ == '__main__': main()
