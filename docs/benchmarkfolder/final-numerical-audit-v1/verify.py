#!/usr/bin/env python3
"""Read-only verification of this bounded source/evidence audit."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    files = {str(p.relative_to(HERE)): p for p in HERE.rglob('*')
             if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    assert set(files) == set(manifest['files'])
    for name, path in files.items():
        assert not path.is_symlink()
        assert path.stat().st_size == manifest['files'][name]['bytes']
        assert sha(path) == manifest['files'][name]['sha256'], name
    result = json.loads((HERE / 'verification.json').read_text())
    assert result['passed'] and result['builds_started'] == result['physics_evaluations_started'] == 0
    assert len(result['runs']) == 5 and all(x['returncode'] == 0 for x in result['runs'])
    assert not result['final_correctness_gate_claimed'] and not result['final_performance_gate_claimed']
    assert result['production_ordinary_rust_tests_bound_by_transport_evidence'] == 18
    assert (result['production_ignored_replay_requests'], result['production_ignored_replay_output_fields']) == (1392, 11136)
    old = ROOT / 'docs/benchmarkfolder/experiments/ninjars-mp-transport-audit-v1/proposed/src/lib.rs'
    proposed = HERE / 'proposed/lib.rs'
    assert sha(old) == result['source_sha256']['vendor/ninjars/src/lib.rs']
    assert proposed.read_text() == old.read_text() + '\n#[cfg(test)]\nmod panic_contract_tests;\n'
    patch = ''.join(difflib.unified_diff(old.read_text().splitlines(True), proposed.read_text().splitlines(True),
        fromfile='a/vendor/ninjars/src/lib.rs', tofile='b/vendor/ninjars/src/lib.rs'))
    patch += ''.join(difflib.unified_diff([], (HERE / 'proposed/panic_contract_tests.rs').read_text().splitlines(True),
        fromfile='/dev/null', tofile='b/vendor/ninjars/src/panic_contract_tests.rs'))
    assert patch == (HERE / 'panic-contract-tests.patch').read_text()
    if args.sources:
        for name, expected in result['source_sha256'].items():
            # PLAN is an observed requirement snapshot hash, not numerical
            # implementation. Root continues appending its execution ledger.
            if name == 'references/PLAN.md':
                continue
            assert sha(ROOT / name) == expected, name
    print(json.dumps({'passed': True, 'files': len(files), 'sources': args.sources,
                      'builds_started': 0, 'physics_evaluations_started': 0}))


if __name__ == '__main__':
    main()
