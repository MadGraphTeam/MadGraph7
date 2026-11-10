#!/usr/bin/env python3
"""Pure inventory/build/ELF verification; never load a shared library."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    entries = json.loads((HERE / 'evidence-index.json').read_text())['files']
    paths = list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()} == set(entries) | {'evidence-index.json'}
    for name, entry in entries.items():
        path = HERE / name
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
        assert not path.read_bytes().startswith(b'\x7fELF') and path.suffix not in ('.so', '.o', '.a', '.mod', '.pyc')
    binding = json.loads((HERE / 'bindings.json').read_text())
    assert not binding['native_loaded'] and not binding['native_tests_executed'] and not binding['production_promoted']
    for name, entry in binding['copies'].items():
        assert sha(HERE / name) == entry['sha256'] and (HERE / name).stat().st_size == entry['bytes']
        if '--live' in sys.argv:
            assert sha(ROOT / entry['path']) == entry['sha256']
    assert sha(ROOT / binding['source_build_package']['path']) == binding['source_build_package']['sha256']
    reader = binding['elf_reader_origin']
    assert sha(ROOT / reader['index_path']) == reader['index_sha256']
    assert sha(HERE / 'elf_metadata.py') == reader['reader_sha256']
    controller = json.loads((HERE / 'controller/source-index.json').read_text())['files']
    assert sha(HERE / 'controller/source-index.json') == '22dab40ded40e662aa7fddcee018fa5837a260e2e6a205a1258930e11125def6'
    for name, entry in controller.items():
        assert sha(HERE / 'controller' / name) == entry['sha256']
    record = json.loads((HERE / 'attempt1/execution.json').read_text())
    assert record['complete'] and not record['native_tests_executed']
    assert record['source_index_sha256'] == 'f5550b8719e76e177ad6b73d456cc655815770eb7092a8fabdc776c415c03213'
    assert record['cpu_affinity_override'] is None and record['jobs'] == 1 and record['compile_seconds'] == 900
    assert record['process']['process_reaped'] and record['process']['returncode'] == 0
    assert record['process']['process_wall_seconds'] <= record['compile_seconds_this_attempt'] <= 900
    assert record['protected_inputs_before'] == record['protected_inputs_after']
    assert record['command'][1:3] == ['build', '--release'] and '--locked' in record['command'] and '--offline' in record['command']
    assert not record['private_library_not_loaded']['cargo_record']['profile']['test']
    assert record['private_library_not_loaded']['cargo_record']['profile']['opt_level'] == '3'
    assert record['automatic_retry_count'] == 0
    for item in record['tool_versions'].values():
        assert item['process_reaped'] and item['returncode'] == 0
    assert record['environment']['CARGO_BUILD_JOBS'] == '1'
    assert not any(k in record['environment'] for k in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER'))
    assert record['private_library_not_loaded'] == binding['library']
    metadata = json.loads((HERE / 'elf-metadata.json').read_text())
    assert metadata['sha256'] == binding['library']['sha256']
    assert metadata['bytes'] == binding['library']['bytes']
    exports = metadata['metadata']['exported_ml_tirust_symbols']
    assert len(exports) == 13 and 'ml_tirust_qp_precision_statistics' in exports
    assert metadata['metadata']['elf_type'] == 'ET_DYN'
    for name, digest in record['artifact_sha256'].items():
        if name.startswith('runtime/'):
            assert digest == binding['library']['sha256']
        else:
            assert sha(HERE / 'attempt1' / name) == digest
    if '--live' in sys.argv:
        for name, digest in record['protected_inputs_after'].items():
            assert sha(Path(name)) == digest, name
        for key in ('path', 'isolated_copy'):
            assert sha(Path(binding['library'][key])) == binding['library']['sha256']
        spec = importlib.util.spec_from_file_location('authenticated_private_elf_reader', HERE / 'elf_metadata.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.inspect(Path(binding['library']['isolated_copy'])) == metadata
    cache = json.loads((HERE / 'cache-use.json').read_text())
    artifacts = []
    for line in (HERE / 'attempt1/build.log').read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get('reason') == 'compiler-artifact':
            artifacts.append(item)
    assert cache['artifacts'] == artifacts
    assert cache['fresh_count'] == sum(item['fresh'] for item in artifacts)
    assert cache['rebuilt_count'] == sum(not item['fresh'] for item in artifacts)
    print(json.dumps({'verification': 'pass', 'portable_files': len(entries),
                      'native_loaded': False, 'native_tests_executed': False,
                      'private_library_sha256': metadata['sha256']}, sort_keys=True))


if __name__ == '__main__':
    main()
