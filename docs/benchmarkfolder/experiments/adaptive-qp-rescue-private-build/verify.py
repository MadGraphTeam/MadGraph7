#!/usr/bin/env python3
"""Verify frozen source/build evidence without executing any native artifact."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inventory(folder, name, exact=False):
    entries = json.loads((folder / name).read_text())['files']
    for relative, entry in entries.items():
        path = folder / relative
        assert not path.is_symlink() and not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], str(path)
    if exact:
        paths = list(folder.rglob('*'))
        assert not any(p.is_symlink() for p in paths)
        assert {str(p.relative_to(folder)) for p in paths if p.is_file()} == set(entries) | {name}
    return entries


def main():
    live = '--live' in sys.argv
    entries = check_inventory(HERE, 'evidence-index.json', True)
    for name in entries:
        path = HERE / name
        assert path.suffix not in ('.o', '.a', '.so', '.mod', '.pyc')
        assert not path.read_bytes().startswith(b'\x7fELF')
    copies = json.loads((HERE / 'copy-bindings.json').read_text())['copies']
    for name, entry in copies.items():
        path = HERE / name
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
        if live:
            assert sha(ROOT / entry['original_path']) == entry['sha256']
    sources = [HERE / f'revision-{i}/source' for i in (1, 2, 3)]
    for i, source in enumerate(sources, 1):
        check_inventory(source, 'source-index.json', True)
        controller = HERE / f'revision-{i}/controller'
        check_inventory(controller, 'source-index.json', True)
        record = json.loads((HERE / f'attempt-{i}/execution.json').read_text())
        assert record['source_index_sha256'] == sha(source / 'source-index.json')
        assert record['cpu'] == 11 and record['jobs'] == 1 and not record['native_tests_executed']
        assert record['protected_inputs_before'] == record['protected_inputs_after']
        assert record['process']['process_reaped']
        assert '--no-run' in record['command'] and '--locked' in record['command'] and '--offline' in record['command']
        assert record['complete'] == (i == 3)
        assert record['returncode'] == (0 if i == 3 else 101)
        if i >= 2:
            budget = json.loads((HERE / 'further-build-budget.json').read_text())
            assert record['shared_further_build_budget'] == budget
            assert 0 < record['compile_seconds_this_attempt'] <= 900
            assert record['process']['process_wall_seconds'] <= record['compile_seconds_this_attempt']
        for name, digest in record['artifact_sha256'].items():
            assert sha(HERE / f'attempt-{i}' / name) == digest
        if live:
            for name, digest in record['protected_inputs_before'].items():
                assert sha(Path(name)) == digest, name
    before = (sources[0] / 'src/lib.rs').read_text()
    after = (sources[1] / 'src/lib.rs').read_text()
    needle = '        unsafe { write(out, &state.qp_precision.statistics()); }\n        Ok(SUCCESS)\n    })\n'
    assert before.count(needle) == 1 and after == before.replace(needle, needle + '}\n')
    assert (sources[2] / 'src/lib.rs').read_bytes() == (sources[1] / 'src/lib.rs').read_bytes()
    before_test = (sources[1] / 'src/adaptive_tests.rs').read_text()
    after_test = (sources[2] / 'src/adaptive_tests.rs').read_text()
    assert after_test == before_test.replace('use super::*;', 'use super::*;\nuse symbolica::domains::SelfRing;')
    result = json.loads((HERE / 'attempt-3/execution.json').read_text())
    executable = result['test_executable_not_run']
    assert executable['sha256'] == 'd22953ed3b955adf7dcce3831259060cec9e974816d432a4a86fd000b1afe8b7'
    assert executable['cargo_record']['profile']['test']
    if live:
        assert sha(Path(executable['cargo_record']['executable'])) == executable['sha256']
        for name, entry in json.loads((HERE / 'dependency-artifacts-after.json').read_text())['artifacts'].items():
            assert sha(Path(name)) == entry['sha256'], name
    print(json.dumps({'verification': 'pass', 'portable_files': len(entries), 'build_attempts': 3,
                      'successful_build': 3, 'native_tests_executed': False,
                      'source_index': sha(sources[2] / 'source-index.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
