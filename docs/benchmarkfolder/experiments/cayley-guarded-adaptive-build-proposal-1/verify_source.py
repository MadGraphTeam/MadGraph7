#!/usr/bin/env python3
"""Read/hash-only verification of an isolated source copy and build plan."""
import hashlib
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
    source = json.loads((HERE / 'source-copies.json').read_text())
    work = ROOT / source['work_root']
    assert work.resolve().is_relative_to(ROOT / '.tirust-build/cayley-guarded-adaptive-build-proposal-1')
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
    protected[str(HERE / 'source-index.json')] = sha(HERE / 'source-index.json')
    protected.update({str(HERE / name): record['sha256'] for name, record in index['files'].items()})
    return protected

if __name__ == '__main__':
    pins = verify()
    print(json.dumps({'passed': True, 'protected_bindings': len(pins),
                      'compile_or_native_execution': False, 'non_guard_source_changes': 0}, indent=2))
