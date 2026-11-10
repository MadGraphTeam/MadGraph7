#!/usr/bin/env python3
"""Read/hash-only verification of an isolated source copy and build plan."""
import hashlib
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    index = json.loads((HERE / 'source-index.json').read_text())
    for name, entry in index['files'].items():
        path = HERE / name
        assert path.resolve().is_relative_to(HERE)
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], str(path)
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
              and p.name != 'source-index.json' and '__pycache__' not in p.parts}
    assert actual == set(index['files'])
    # Retain the original failed controller result separately from the later
    # additive read/hash-only closure; neither is rewritten as a success.
    failure_dir = HERE / 'failed-attempt1'
    failed_bytes = gzip.decompress((failure_dir / 'execution.json.gz').read_bytes())
    failed = json.loads(failed_bytes)
    closure = json.loads((failure_dir / 'closure.json').read_text())
    assert not failed['complete'] and not failed['protected_input_closure_complete']
    assert hashlib.sha256(failed_bytes).hexdigest() == closure['failed_execution_sha256']
    assert failed['protected_inputs_before'] == closure['protected_inputs_after']
    assert closure['read_hash_only'] and not closure['protected_input_drift']
    assert all(c['progress']['process_reaped'] for c in failed['commands'])
    assert failed['commands'][-1]['progress']['returncode'] == 101
    failure_log = gzip.decompress((failure_dir / 'release-build.log.gz').read_bytes()).decode('utf8')
    assert 'rustc-LLVM ERROR: out of memory' in failure_log and 'SIGABRT' in failure_log
    assert not closure['new_shared_library_emitted']
    timeout_dir = HERE / 'failed-attempt2'
    timeout_bytes = gzip.decompress((timeout_dir / 'execution.json.gz').read_bytes())
    timeout = json.loads(timeout_bytes)
    timeout_closure = json.loads((timeout_dir / 'closure.json').read_text())
    assert not timeout['complete'] and timeout['budget_exhausted']
    assert not timeout['protected_input_closure_complete']
    assert timeout['error'] == 'TimeoutError: The single 600-second build deadline expired'
    assert hashlib.sha256(timeout_bytes).hexdigest() == timeout_closure['failed_execution_sha256']
    assert timeout['protected_inputs_before'] == timeout_closure['protected_inputs_after']
    assert timeout_closure['read_hash_only'] and not timeout_closure['protected_input_drift']
    assert all(c['progress']['process_reaped'] for c in timeout['commands'])
    assert timeout['commands'][-1]['progress']['returncode'] == -9
    assert not timeout_closure['new_shared_library_emitted']
    source = json.loads((HERE / 'source-copies.json').read_text())
    work = ROOT / source['work_root']
    assert work.resolve().is_relative_to(ROOT / '.tirust-build/cayley-guarded-adaptive-build-proposal-3')
    copied_paths = {str(p.relative_to(work)) for p in (work / 'sources').rglob('*') if p.is_file()}
    assert copied_paths == set(source['copied'])
    assert not any(p.is_symlink() for p in (work / 'sources').rglob('*'))
    old = '    let ratio = terms / size(&det_x);\n    ratio.is_finite().then_some(ratio)\n'
    new = ('    let denominator = size(&det_x);\n'
           '    if denominator.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {\n'
           '        return None;\n    }\n'
           '    let ratio = terms / denominator;\n'
           '    ratio.is_finite().then_some(ratio)\n')
    changed = []
    protected = {}
    for name, entry in source['copied'].items():
        path = work / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], str(path)
        origin = source['origin'][name]
        original = ROOT / origin['path']
        assert original.resolve() == Path(origin['resolved_path'])
        assert original.stat().st_size == origin['bytes'] and sha(original) == origin['sha256']
        protected[str(original)] = origin['sha256']
        protected[str(path)] = entry['sha256']
        if path.read_bytes() != original.read_bytes():
            changed.append(name)
            assert original.read_text().count(old) == 1
            assert path.read_text() == original.read_text().replace(old, new)
    assert changed == [source['only_changed_source']]
    for link in source['materialized_symlinks']:
        path = ROOT / link['source']
        assert path.is_symlink() and str(path.readlink()) == link['target']
        assert str(path.resolve()) == link['resolved']
    previous = json.loads((HERE / 'old-release-execution.json').read_text())
    assert previous['complete'] and previous['private_library_not_loaded']['sha256'] == 'c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7'
    for name, digest in previous['protected_inputs_before'].items():
        assert sha(Path(name)) == digest, name
        protected[name] = digest
    plan = json.loads((HERE / 'plan.json').read_text())
    for name, digest in plan['protected_extra_files'].items():
        assert sha(ROOT / name) == digest, name
        protected[str(ROOT / name)] = digest
    for name, digest in plan['tool_files'].items():
        assert sha(Path(name)) == digest, name
        protected[name] = digest
    original_environment = previous['environment']
    expected = original_environment.copy()
    expected.update({key: str(ROOT / value) for key, value in plan['isolated_environment_paths'].items()})
    assert plan['environment'] == expected
    assert plan['environment']['CARGO_BUILD_JOBS'] == '1'
    assert not any(k in plan['environment'] for k in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER'))
    prior_plan = json.loads((ROOT / 'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-2/plan.json').read_text())
    for key in ('base_release_sha256', 'bridge_Cargo_lock_sha256',
                'TIRust_workspace_Cargo_lock_sha256', 'guard_patch_sha256',
                'tool_files', 'effective_tool_paths', 'supervisor',
                'cpu', 'cpu_exclusive', 'jobs', 'release_profile',
                'address_space_limit_bytes_per_build_process',
                'address_space_limit_bytes_per_cache_copy_process', 'outer_alarm_ownership'):
        assert plan[key] == prior_plan[key], key
    normalize = lambda argv: [v.replace('cayley-guarded-adaptive-build-proposal-3', 'cayley-guarded-adaptive-build-proposal-2') for v in argv]
    assert normalize(plan['cache_copy_command']) == prior_plan['cache_copy_command']
    assert normalize(plan['build_command']) == prior_plan['build_command']
    assert plan['overall_seconds'] == 1800 and prior_plan['overall_seconds'] == 600
    assert plan['cache_seed']['source'] == '.tirust-build/cargo-build/release'
    assert not plan['cache_seed']['guarded_failed_attempt_cache_reused']
    protected[str(HERE / 'source-index.json')] = sha(HERE / 'source-index.json')
    protected.update({str(HERE / name): record['sha256'] for name, record in index['files'].items()})
    return protected

if __name__ == '__main__':
    pins = verify()
    print(json.dumps({'passed': True, 'protected_bindings': len(pins),
                      'compile_or_native_execution': False, 'non_guard_source_changes': 0}, indent=2))
