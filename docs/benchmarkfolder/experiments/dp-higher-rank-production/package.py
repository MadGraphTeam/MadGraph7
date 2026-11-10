#!/usr/bin/env python3
"""Freeze portable qualification text after the numerical gates pass."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT/'.tirust-build/dp-higher-rank-production'
CASE = 'qcd_n3_uux_zgg'
SOURCES = (
    'madgraph/loop/dp_terminal_proof.py',
    'madgraph/loop/loop_exporters.py',
    'madgraph/various/q_polynomial.py',
    'tests/unit_tests/loop/test_dp_terminal_wavefunctions.py',
    'tests/unit_tests/loop/test_dp_terminal_export.py',
    'tests/unit_tests/various/test_dp_terminal_compiled.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


if (HERE/'evidence-index.json').exists():
    raise SystemExit('Evidence is already frozen; do not overwrite it.')
comparison = json.loads((HERE/'cuttools/comparison.json').read_text())
assert comparison['bitwise_equal'] and comparison['status_equal']
assert comparison['retained_committed_control_bit_and_status_equal']
assert comparison['compiler_policy_equal'] and comparison['external_native_inputs_equal']
assert comparison['comparison']['passed']
source_audit = json.loads((HERE/'compiled-source-audit.json').read_text())
assert source_audit['passed'] and source_audit['source_snapshots_stable']
assert source_audit['classification_counts'] == {
    'immediate_query_and_private_call_substitutions_only': 1,
    'two_private_helpers_added_only': 1}
regressions = json.loads((HERE/'regressions.json').read_text())
assert regressions['returncode'] == 0 and regressions['sources_unchanged']
assert regressions['source_sha256_before'] == regressions['source_sha256_after']
source_hashes = {name: sha(ROOT/name) for name in SOURCES}
assert source_hashes == regressions['source_sha256_after']
for name in SOURCES:
    destination = HERE/'source-snapshot'/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/name, destination)
for name in ('logs', 'build-attempts'):
    shutil.copytree(WORK/name, HERE/name)
for side in ('control', 'candidate'):
    output = WORK/'outputs'/side/CASE
    paths = source_audit[side+'_inventory']
    for name, record in paths.items():
        assert sha(output/name) == record['sha256']
    for name in (*paths, 'Cards/param_card.dat', 'Cards/MadLoopParams.dat',
                 'Cards/ident_card.dat', 'compilation.json', 'generation.json'):
        destination = HERE/'compiled-source'/side/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((output/name).read_bytes())
for suffix in ('.dat', '.json'):
    destination = HERE/'points'/(CASE+suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/'docs/benchmarkfolder/points'/(CASE+suffix), destination)
invocations = [json.loads(path.read_text()) for path in sorted((HERE/'protocol').glob('*/invocation.json'))]
observation_dirs = [path.parent for path in (HERE/'protocol').glob('*/invocation.json')
                    if json.loads(path.read_text())['argv'][1] == 'observe']
assert len(observation_dirs) == 1
summary = {
    'schema_version': 1, 'status': 'passed', 'performance_acceptance_eligible': False,
    'source_and_test_sha256': source_hashes,
    'scope': 'Production rank pairs 1x1/2x1/3x1; completed-source proof, compiled IEEE contracts, deterministic CLI and ordinary CutTools numerical qualification.',
    'source_audit': 'compiled-source-audit.json', 'identical_native_source_files': source_audit['unchanged_selected_files'],
    'source_changes': source_audit['classification_counts'],
    'control_selected_updates': 26, 'candidate_selected_updates': 147,
    'new_updates': {'2x1': 85, '3x1': 36}, 'declined_selected_updates': 0,
    'compiled_tests': {'tests': 7, 'ieee_output_comparisons': 4608000, 'paired_trap_probes': 104,
                       'summary': 'compiled/qualification-summary.json'},
    'regressions': {'tests': 114, 'subtests': 220, 'record': 'regressions.json'},
    'numerical': {'points': 100, 'rows': 200, 'coefficients': 800, 'backend': 1,
                  'precision': 'DP', 'point_status': 211, 'strict_host_threshold': 1e-6,
                  'bitwise_equal': True, 'status_equal': True,
                  'retained_committed_control_bit_and_status_equal': True,
                  'comparison': 'cuttools/comparison.json'},
    'observation_protocol': str((observation_dirs[0]/'tirust_benchmark_worker.py').relative_to(HERE)),
    'protocol_invocations': invocations,
    'source_proof_hardening_attempt': 'initial-reference-census/status.json',
    'source_order_attempt': 'source-order-attempt/status.json',
    'limitations': ['No performance acceptance measurement.',
                    'Ordinary real-mass QCD n3 points only; no new backend8/9 or stress-point replay.',
                    'Runtime admission frequency is not measured by this qualification.',
                    'Control disables only higher-rank terminal selection in a private actual-CLI wrapper; both final exports use seed0.']}
dump(HERE/'validation-summary.json', summary)
files = {}
for path in sorted(HERE.rglob('*')):
    if path.is_symlink():
        raise ValueError('Symlink in portable evidence: '+str(path))
    if not path.is_file():
        continue
    data = path.read_bytes()
    if b'\0' in data or data.startswith(b'\x7fELF') or path.suffix == '.pyc':
        raise ValueError('Nontext artifact in portable evidence: '+str(path))
    files[path.relative_to(HERE).as_posix()] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
dump(HERE/'evidence-index.json', {'schema_version': 1, 'files': files,
                               'scope': 'Portable text evidence; native objects and executables represented by hashes only.'})
print(json.dumps({'portable_files': len(files), 'index_sha256': sha(HERE/'evidence-index.json')}))
