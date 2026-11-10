#!/usr/bin/env python3
"""Read-only ELF/source preparation; writes only this new correction's metadata."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    assert Path.cwd().resolve() == ROOT
    assert not (HERE/'prepared.json').exists()
    assert not (HERE/'source-index.json').exists()
    assert sha(PARENT/'prepared.json') == '906cc984cc59068b89823ef7108f3f80f28e373e2ee8a15dfe9df7abb86c2a1e'
    assert sha(PARENT/'build.json') == '0f1b01f60d2044271519e97566fb6772c1a66a4f4310edc06b281070b92e497e'
    assert sha(PARENT/'run.json') == 'a5d7541c314367999c11dd4335a15f86b9bcf7d44294849b10d6b25558c33b12'
    failed = json.loads((PARENT/'run.json').read_text())
    assert failed['state'] == 'failed' and len(failed['jobs']) == 1
    assert failed['jobs'][0]['name'] == 'loader-fixture' and failed['jobs'][0]['returncode'] == 0
    assert failed['input_hashes_after'] == 'unchanged'
    base = load('mass_loader_prepare_base', PARENT/'control.py')
    control = load('mass_loader_prepare_control', HERE/'control.py')
    elf_path = ROOT/'.tirust-build/cms-scalar-refusal-prototype/mock-qualification/elf.py'
    elf = load('mass_loader_prepare_elf', elf_path)
    original = json.loads((PARENT/'prepared.json').read_text())
    base.verify_inputs(original)
    base.executable_outputs(original)
    closures = {c['job']['name']: control.closure(c, elf, base) for c in original['runtime_checks']}
    for check in original['runtime_checks'][1:]:
        expected = {Path(name).name: info for name, info in check['expected'].items()}
        assert closures[check['job']['name']]['closure'] == expected
    inputs = {}
    for path in (PARENT/'control.py', PARENT/'prepared.json', PARENT/'evidence-index.json',
                 PARENT/'build.json', PARENT/'run.json', PARENT/'fixture/loader-list.txt', elf_path,
                 HERE/'control.py', HERE/'prepare.py', HERE/'README.md'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    value = {'schema': 'qp-mass-loader-correction-v1', 'inputs': inputs,
             'elf_reader': str(elf_path.relative_to(ROOT)), 'elf_closures': closures,
             'fixture_expected': closures['loader-fixture']['closure'],
             'fixture_loader_log': str((HERE/'fixture-loader.log').relative_to(ROOT)),
             'scientific_scope_unchanged': True, 'cpu': 8, 'affinity_exclusive': False,
             'prior_failed_run_sha256': sha(PARENT/'run.json'),
             'process_launched': False}
    (HERE/'prepared.json').write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    files = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}
    (HERE/'source-index.json').write_text(json.dumps({'files': files, 'process_launched': False},
                                                   indent=2, sort_keys=True)+'\n')
    print(json.dumps({'prepared_sha256': sha(HERE/'prepared.json'),
                      'source_index_sha256': sha(HERE/'source-index.json'),
                      'elf_closures': len(closures), 'process_launched': False}))


if __name__ == '__main__':
    main()
