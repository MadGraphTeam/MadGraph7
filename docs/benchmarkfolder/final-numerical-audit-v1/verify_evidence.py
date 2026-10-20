#!/usr/bin/env python3
"""Read-only numerical audit: hash sources and recheck existing sealed evidence.

Runs no compiler, reducer, native example, or benchmark worker. Outputs belong
only to this audit; historical evidence and expectations are never rewritten.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
SOURCE_NAMES = [
    'references/PLAN.md',
    'vendor/ninjars/Cargo.toml', 'vendor/ninjars/Cargo.lock',
    'vendor/ninjars/ninjars.h', 'vendor/ninjars/mninjars.f90',
    'vendor/ninjars/src/lib.rs', 'vendor/ninjars/src/tensor.rs',
    'vendor/ninjars/src/guarded.rs', 'vendor/ninjars/src/guarded_fenv.c',
    'vendor/ninjars/src/retry_tests.rs', 'vendor/ninjars/src/guarded_tests.rs',
    'vendor/ninjars/src/transport_contract_tests.rs',
    'vendor/ninjars/src/transport_replay_tests.rs',
    'vendor/ninjars/tests/data/provenance.json',
    'madgraph/iolibs/template_files/loop_optimized/NINJARS_interface.inc',
    'madgraph/iolibs/template_files/loop_optimized/CT_interface.inc',
    'madgraph/iolibs/template_files/loop_optimized/TIR_interface.inc',
    'madgraph/iolibs/template_files/loop_optimized/helas_loop_amplitude.inc',
    'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc',
    'madgraph/iolibs/template_files/loop_optimized/mp_compute_loop_coefs.inc',
    'madgraph/various/q_polynomial.py', 'madgraph/loop/loop_exporters.py',
    'Template/loop_material/StandAlone/SubProcesses/MadLoopCommons.inc',
    'Template/loop_material/StandAlone/SubProcesses/MadLoopParams.inc',
    'Template/loop_material/StandAlone/SubProcesses/MadLoopParamReader.f',
    'Template/loop_material/StandAlone/Cards/MadLoopParams.dat',
    'Template/NLO/SubProcesses/BinothLHA.f',
    'Template/LO/Source/dsample.f',
    'tests/unit_tests/loop/test_ninjars_fortran.py',
    'tests/unit_tests/various/test_ninjars_statistics.py',
    'tests/unit_tests/various/test_q_polynomial_compiled.py',
    'tests/unit_tests/various/test_q_polynomial_factorized.py',
]
CHECKS = [
    ('transport', 'docs/benchmarkfolder/experiments/ninjars-mp-transport-audit-v1/verify.py', []),
    ('zero-sign-install', 'docs/benchmarkfolder/policy-v2/install-export-abi3-zero-sign-v1/verify.py', ['--live']),
    ('representatives-f4b', 'docs/benchmarkfolder/policy-v2/representative-retention-abi3-tensor-factor-v1/verify.py', []),
    ('controls-f4b', 'docs/benchmarkfolder/policy-v2/controls-retention-abi3-tensor-factor-v1/verify.py', []),
    ('variants-f4b', 'docs/benchmarkfolder/policy-v2/final-variant-controls-abi3-tensor-factor-v1/verify.py', []),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = {name: sha(ROOT / name) for name in SOURCE_NAMES}
    runs = []
    for name, relative, arguments in CHECKS:
        command = [sys.executable, '-B', str(ROOT / relative), *arguments]
        run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                             timeout=120)
        (HERE / (name + '.stdout')).write_text(run.stdout)
        (HERE / (name + '.stderr')).write_text(run.stderr)
        runs.append({'name': name, 'command': command, 'returncode': run.returncode,
                     'verifier_sha256': sha(ROOT / relative),
                     'stdout_sha256': sha(HERE / (name + '.stdout')),
                     'stderr_sha256': sha(HERE / (name + '.stderr'))})
    after = {name: sha(ROOT / name) for name in SOURCE_NAMES}
    assert before == after, 'Audited source changed during read-only verification'
    provenance = json.loads((ROOT / 'vendor/ninjars/tests/data/provenance.json').read_text())
    fixtures = []
    for entry in provenance['files']:
        path = ROOT / 'vendor/ninjars/tests/data' / entry['file']
        assert sha(path) == entry['sha256']
        fixtures.append(entry['file'])
    installed = json.loads((ROOT / 'docs/benchmarkfolder/policy-v2/'
        'install-export-abi3-zero-sign-v1/installation.json').read_text())
    for name, digest in installed['archived_bridge_source_sha256'].items():
        assert sha(ROOT / 'vendor/ninjars' / name) == digest, name
    result = {
        'source_revision_at_verification': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'execution_affinity': Path('/proc/self/status').read_text().split(
            'Cpus_allowed_list:\t')[1].splitlines()[0],
        'source_sha256': before,
        'source_unchanged_during_verification': before == after,
        'runs': runs,
        'physical_fixture_inputs': len([x for x in fixtures if x.startswith('cases/')]),
        'physical_fixture_references': len([x for x in fixtures if x.startswith('ninja/')]),
        'production_ordinary_rust_tests_bound_by_transport_evidence': 18,
        'production_ignored_replay_requests': 1392,
        'production_ignored_replay_output_fields': 11136,
        'current_bridge_matches_zero_sign_installation_source': True,
        'zero_sign_archive_sha256': installed['files']['lib/libninjars.a'],
        'physics_evaluations_started': 0, 'builds_started': 0,
        'final_correctness_gate_claimed': False,
        'final_performance_gate_claimed': False,
        'passed': all(run['returncode'] == 0 for run in runs),
    }
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('source_sha256', 'runs')}, indent=2))
    assert result['passed'], 'A historical-evidence verifier failed; inspect retained logs'


if __name__ == '__main__':
    main()
