#!/usr/bin/env python3
"""Portable text-only build verifier. Never read or load any native artifact."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE_INDEX = '39d6ba2bd3c7aef5c8f5008360661c988a93d86ab1265cb3bc815707233b9e72'
RECEIPT = '59167eec84a767f0712b6905285a99564458ff0667427de87ce1921534b46632'
LOG = '7ea452558263c98b91781e979f843c8cf236d951b89450cbbc436275945e8a3d'
LIBRARY = '69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b'
PREFIX = '/shared/external/madloop_tirust/madgraph7_madloop_tirust/'
WORK = PREFIX + '.tirust-build/cayley-guarded-adaptive-build-proposal-3/'

def digest(raw): return hashlib.sha256(raw).hexdigest()
def data(name):
    path = HERE / name
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == '.gz' else raw

def document(name): return json.loads(data(name))
def verify():
    index = document('evidence-index.json')
    all_paths = list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in all_paths)
    assert {str(p.relative_to(HERE)) for p in all_paths if p.is_file()} == set(index['files']) | {'evidence-index.json'}
    for name, entry in index['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path = HERE / name
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'] <= 1048576 and digest(raw) == entry['sha256'], name
        assert not raw.startswith(b'\x7fELF') and path.suffix not in ('.so', '.rlib', '.o', '.a', '.mod', '.pyc')
        data(name).decode('utf8')
    binding = document('bindings.json')
    assert not binding['native_payloads_retained'] and not binding['historical_paths_required_by_default_verifier']
    assert binding['current_DSO_read_count'] == 1
    for name, entry in binding['copies'].items():
        raw = data(name) if entry['encoding'] == 'gzip-utf8' else (HERE / name).read_bytes()
        assert digest(raw) == entry['origin_sha256'] and len(raw) == entry['origin_bytes'], name
    assert digest(data('proposal3/source-index.json')) == SOURCE_INDEX
    source_index = document('proposal3/source-index.json')
    for name, entry in source_index['files'].items():
        raw = (HERE / 'proposal3' / name).read_bytes()
        assert digest(raw) == entry['sha256'] and len(raw) == entry['bytes'], name
    plan = document('proposal3/plan.json')
    source_map = document('proposal3/source-copies.json')
    assert len(source_map['copied']) == len(source_map['origin']) == 499
    assert source_map['work_root'] == '.tirust-build/cayley-guarded-adaptive-build-proposal-3'
    receipt = document('attempt1/execution.json.gz')
    assert digest(data('attempt1/execution.json.gz')) == RECEIPT
    assert digest(data('attempt1/release-build.log.gz')) == LOG
    assert receipt['complete'] and receipt['root_admission_received'] and not receipt['loader_or_native_test_executed']
    assert receipt['overall_seconds'] == plan['overall_seconds'] == 1800
    assert receipt['cpu'] == plan['cpu'] == 11 and not receipt['cpu_exclusive'] and not plan['cpu_exclusive']
    assert receipt['jobs'] == plan['jobs'] == 1 and receipt['automatic_retries'] == plan['automatic_retry_count'] == 0
    assert receipt['address_space_limit_bytes_per_build_process'] == plan['address_space_limit_bytes_per_build_process'] == 8589934592
    assert receipt['address_space_limit_bytes_per_cache_copy_process'] == plan['address_space_limit_bytes_per_cache_copy_process'] == 4294967296
    assert receipt['command'] == plan['build_command']
    assert receipt['environment'] == plan['environment']
    assert receipt['environment']['CARGO_BUILD_JOBS'] == '1' and receipt['environment']['CARGO_NET_OFFLINE'] == 'true'
    assert not any(k in receipt['environment'] for k in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER'))
    assert plan['build_command'][1] == '--as=8589934592' and plan['cache_copy_command'][1] == '--as=4294967296'
    assert '--locked' in receipt['command'] and '--offline' in receipt['command']
    assert receipt['command'][receipt['command'].index('build'):] == ['build','--release','--manifest-path',
        '.tirust-build/cayley-guarded-adaptive-build-proposal-3/sources/adaptive-qp-rescue-prototype3/Cargo.toml',
        '--locked','--offline','--features','poles-only','--lib','--message-format=json']
    assert receipt['protected_inputs_before'] == receipt['protected_inputs_after']
    protected = receipt['protected_inputs_before']
    assert len(protected) == 1193
    assert protected[PREFIX + 'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-3/source-index.json'] == SOURCE_INDEX
    for name, digest_value in plan['tool_files'].items(): assert protected[name] == digest_value
    for name, digest_value in plan['protected_extra_files'].items(): assert protected[PREFIX + name] == digest_value
    for name, entry in source_map['copied'].items():
        assert protected[WORK + name] == entry['sha256']
        origin = source_map['origin'][name]
        assert protected[PREFIX + origin['path']] == origin['sha256']
    old_release = document('proposal3/old-release-execution.json')
    assert old_release['complete'] and old_release['private_library_not_loaded']['sha256'] == 'c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7'
    assert old_release['protected_inputs_before'] == old_release['protected_inputs_after']
    for name, value in old_release['protected_inputs_before'].items(): assert protected[name] == value
    expected_environment = old_release['environment'].copy()
    expected_environment.update({k: PREFIX + v for k,v in plan['isolated_environment_paths'].items()})
    assert receipt['environment'] == expected_environment
    assert data('proposal3/cargo-version.log') == b'cargo 1.97.0 (c980f4866 2026-06-30)\n'
    assert data('proposal3/rustc-version.log') == b'rustc 1.97.1 (8bab26f4f 2026-07-14) (built from a source tarball)\n'
    assert not plan['cache_seed']['guarded_failed_attempt_cache_reused'] and plan['cache_seed']['source'] == '.tirust-build/cargo-build/release'
    assert [c['label'] for c in receipt['commands']] == ['copy-cache','release-build']
    for item, name in zip(receipt['commands'], ('cache_copy_command','build_command')):
        assert item['command'] == plan[name]
        progress = item['progress']
        assert progress['process_spawned'] and progress['process_reaped'] and progress['returncode'] == 0
        assert progress['spawn_unix'] <= progress['exit_observed_unix'] <= receipt['finished_unix']
    assert 0 < receipt['elapsed_seconds'] < receipt['overall_seconds']
    assert receipt['commands'][1]['progress']['process_wall_seconds'] < receipt['elapsed_seconds']
    artifacts = []
    diagnostics = []
    finished = []
    for line in data('attempt1/release-build.log.gz').decode('utf8').splitlines():
        try: item = json.loads(line)
        except json.JSONDecodeError: continue
        if item.get('reason') == 'compiler-artifact': artifacts.append(item)
        elif item.get('reason') == 'compiler-message': diagnostics.append(item)
        elif item.get('reason') == 'build-finished': finished.append(item)
    assert artifacts == receipt['cargo_artifacts'] and len(artifacts) == 144
    assert sum(a['fresh'] for a in artifacts) == 140
    assert finished == [{'reason':'build-finished','success':True}]
    newly_compiled = [a for a in artifacts if not a['fresh']]
    assert {a['target']['name'] for a in newly_compiled} == {'iregi_rust','iregi_masters','madloop_pole_kernels','madloop_tirust_adaptive_private'}
    expected_manifests = {'iregi_rust': 'pole-development/crates/iregi-rust', 'iregi_masters': 'pole-development/crates/iregi-masters',
        'madloop_pole_kernels': 'pole-development/madloop-pole-kernels', 'madloop_tirust_adaptive_private': 'adaptive-qp-rescue-prototype3'}
    for artifact in newly_compiled:
        assert artifact['manifest_path'] == WORK + 'sources/' + expected_manifests[artifact['target']['name']] + '/Cargo.toml'
        profile = artifact['profile']
        assert profile['opt_level'] == '3' and profile['debuginfo'] == 0 and not profile['test'] and not profile['debug_assertions']
    assert sum(a['message']['level']=='warning' for a in diagnostics) == 41
    for artifact in artifacts:
        if artifact['package_id'].startswith('path+'):
            assert artifact['manifest_path'].startswith(WORK + 'sources/')
    bridge = next(a for a in newly_compiled if a['target']['name']=='madloop_tirust_adaptive_private')
    assert bridge['features'] == ['default','poles-only']
    assert WORK + 'target/release/libmadloop_tirust_adaptive_private.so' in bridge['filenames']
    assert bridge['executable'] is None and set(bridge['target']['crate_types']) == {'cdylib','rlib'}
    library = receipt['private_library_not_loaded']
    assert library == binding['library'] and library['sha256'] == LIBRARY and library['bytes'] == 69162336
    assert library['path'] == WORK + 'attempt1/runtime/libmadloop_tirust_adaptive_private.so'
    metadata = document('elf-metadata.json')
    assert metadata['sha256'] == LIBRARY and metadata['bytes'] == library['bytes']
    assert metadata['metadata']['elf_type'] == 'ET_DYN'
    assert metadata['metadata']['machine'] == 'x86_64' and metadata['metadata']['elf_class'] == 64
    exports = metadata['metadata']['exported_ml_tirust_symbols']
    assert len(exports) == 13 and 'ml_tirust_qp_precision_statistics' in exports
    reader_index = document('reader-origin-index.json')
    assert digest(data('reader-origin-index.json')) == '51f45d9e6f2c13227dc504fa284cba967f947db0f073dfb18bb9bf924c28c298'
    assert digest(data('elf_metadata.py')) == reader_index['files']['elf_metadata.py']['sha256'] == '3327e253c4ea1b581b252fc9547f9b7c3f71f610c131e8c23b0dc1039b26f73a'
    text_manifest = document('source-texts.json')
    archive = b''
    for chunk in text_manifest['chunks']:
        raw = data(chunk['path'])
        assert len(raw) == chunk['utf8_bytes'] and digest(raw) == chunk['utf8_sha256']
        archive += raw
    assert len(archive) == text_manifest['archive_utf8_bytes'] and digest(archive) == text_manifest['archive_utf8_sha256']
    texts = json.loads(archive)
    assert sorted(texts) == text_manifest['selected_names'] and len(texts) == text_manifest['selected_count'] == 133
    assert set(texts).isdisjoint(text_manifest['excluded_hash_only'])
    assert set(texts) | set(text_manifest['excluded_hash_only']) == set(source_map['copied'])
    for name, entry in text_manifest['excluded_hash_only'].items(): assert entry == source_map['copied'][name]
    changed = []
    for name, text in texts.items():
        raw = text.encode('utf8')
        entry = source_map['copied'][name]
        assert len(raw) == entry['bytes'] and digest(raw) == entry['sha256']
        if entry['sha256'] != source_map['origin'][name]['sha256']: changed.append(name)
    guard = 'sources/pole-development/crates/iregi-rust/src/direct/exec/gram.rs'
    assert changed == [guard] and source_map['only_changed_source'] == guard
    for name, entry in source_map['copied'].items():
        if name != guard: assert entry['sha256'] == source_map['origin'][name]['sha256']
    original = data('original-gram.rs.gz')
    assert digest(original) == source_map['origin'][guard]['sha256']
    old = '    let ratio = terms / size(&det_x);\n    ratio.is_finite().then_some(ratio)\n'
    new = '    let denominator = size(&det_x);\n    if denominator.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {\n        return None;\n    }\n    let ratio = terms / denominator;\n    ratio.is_finite().then_some(ratio)\n'
    assert original.decode('utf8').count(old) == 1
    assert original.decode('utf8').replace(old,new) == texts[guard]
    for name, wanted in [('sources/adaptive-qp-rescue-prototype3/Cargo.lock',plan['bridge_Cargo_lock_sha256']),
        ('sources/pole-development/Cargo.lock',plan['TIRust_workspace_Cargo_lock_sha256'])]:
        assert digest(texts[name].encode('utf8')) == wanted
    assert digest(data('proposal3/guard.patch')) == plan['guard_patch_sha256']
    for number in (1,2):
        prefix = f'proposal3/failed-attempt{number}/'
        failed_raw = data(prefix+'execution.json.gz')
        failed = json.loads(failed_raw)
        closure = document(prefix+'closure.json')
        assert not failed['complete'] and not failed['protected_input_closure_complete']
        assert digest(failed_raw) == closure['failed_execution_sha256']
        assert failed['protected_inputs_before'] == closure['protected_inputs_after']
        assert closure['read_hash_only'] and not closure['protected_input_drift'] and not closure['new_shared_library_emitted']
        assert not failed['loader_or_native_test_executed']
        assert all(c['progress']['process_reaped'] for c in failed['commands'])
        if number==1:
            assert len(closure['protected_inputs_after']) == 1163
            assert failed['commands'][-1]['progress']['returncode'] == 101
            assert 'rustc-LLVM ERROR: out of memory' in data(prefix+'release-build.log.gz').decode('utf8')
        else:
            assert len(closure['protected_inputs_after']) == 1186 and failed['budget_exhausted']
            assert failed['error']=='TimeoutError: The single 600-second build deadline expired'
            assert failed['commands'][-1]['progress']['returncode'] == -9
    qualification = document('qualification.json')
    assert qualification['library'] == library and qualification['source_index_sha256'] == SOURCE_INDEX
    assert qualification['execution_sha256'] == RECEIPT and qualification['build_log_sha256'] == LOG
    assert qualification['cargo_artifacts'] == 144 and qualification['newly_compiled'] == 4 and qualification['reused'] == 140
    assert qualification['warning_diagnostics'] == 41 and qualification['protected_binding_count'] == 1193
    assert qualification['copied_source_identity_count'] == 499 and qualification['retained_code_text_count'] == 133
    assert qualification['sole_changed_source'] == guard
    assert qualification['compile_seconds_administrative_only'] == receipt['commands'][1]['progress']['process_wall_seconds']
    assert qualification['total_seconds_administrative_only'] == receipt['elapsed_seconds']
    for key in ('native_library_loaded','native_tests_executed','installed_or_promoted','precision_qualified',
                'physical_accuracy_qualified','stability_qualified','runtime_performance_qualified'):
        assert qualification[key] is False
    return {'verification':'pass','payload_files':len(index['files']), 'protected_bindings':1193,
            'retained_code_texts':133,'source_identities':499,'newly_compiled':4,'reused':140,
            'private_library_sha256':LIBRARY,'native_artifact_read_or_loaded':False,
            'historical_native_paths_required':False,'physical_or_performance_acceptance':False}

if __name__=='__main__': print(json.dumps(verify(),sort_keys=True))
