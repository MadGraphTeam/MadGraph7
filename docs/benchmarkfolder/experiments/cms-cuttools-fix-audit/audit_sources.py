#!/usr/bin/env python3
"""Read source/ELF metadata; write only this new audit package. No numerical jobs."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CMS = ROOT / 'docs/benchmarkfolder/experiments/complex-mass'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(command, pattern):
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return {'command': command, 'returncode': result.returncode,
            'matching_lines': [line for line in result.stdout.splitlines() if re.search(pattern, line)]}


def audit():
    index_hash = digest(CMS / 'artifact-index.json')
    index = json.loads((CMS / 'artifact-index.json').read_text())
    for entry in index['files']:
        path = ROOT / entry['path']
        assert digest(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], entry['path']
    source_paths = [
        'madgraph/iolibs/template_files/loop_optimized/CT_interface.inc',
        'madgraph/iolibs/template_files/loop/CT_interface.inc',
        'madgraph/iolibs/template_files/loop_optimized/helas_loop_amplitude.inc',
        'madgraph/iolibs/template_files/loop/helas_loop_amplitude.inc',
        'madgraph/iolibs/template_files/loop_optimized/mp_compute_loop_coefs.inc',
        'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc',
        'madgraph/loop/loop_exporters.py',
        'Template/loop_material/StandAlone/SubProcesses/MadLoopParamReader.f',
        'Template/loop_material/StandAlone/SubProcesses/makefile',
        'Template/NLO/SubProcesses/fks_singular.f',
        'vendor/CutTools/src/cts/cts_cuttools.f90',
        'vendor/CutTools/src/cts/cts_cutroutines.f90',
        'vendor/CutTools/src/avh/avh_olo.f90',
        'tools/tirust_complex_stress.py']
    hashes = {name: digest(ROOT / name) for name in source_paths}
    archive = 'vendor/CutTools/includects/libcts.a'
    ninja = 'HEPTools/ninja/lib/libninja.so'
    collier = 'HEPTools/collier/libcollier.so'
    executable = '.tirust-build/complex-stress-cuttools-qp-transport/SubProcesses/P0_epem_mupmum/check'
    evidence = {
        'CutTools_archive': inspect(['nm', '-A', '--defined-only', archive],
            r'(MOD_olo_onshell|MOD_onshellthrs)'),
        'Ninja_scalar_state': inspect(['readelf', '-Ws', ninja],
            r'avh_olo.*(onshellthrs|nonzerothrs|MOD_olo_onshell)'),
        'Collier_foreign_scalar_symbols': inspect(['nm', '-D', collier],
            r'avh_olo|ctsinit|ctsxcut'),
        'private_CMS_entrypoints': inspect(['nm', '--defined-only', executable],
            r'(forcuttools.*MOD_onshellthrs|ctsxcut|ctsinit)')}
    assert evidence['CutTools_archive']['matching_lines']
    assert evidence['Ninja_scalar_state']['matching_lines']
    assert not evidence['Collier_foreign_scalar_symbols']['matching_lines']
    symbols = '\n'.join(evidence['private_CMS_entrypoints']['matching_lines'])
    assert 'ctsxcut_qpinput_' in symbols and 'ctsinit_' in symbols
    assert not re.search(r'\bctsxcut_\s*$', symbols, re.M)
    lines = []
    for label, record in evidence.items():
        lines.append(label + ': ' + ' '.join(record['command']))
        lines.extend(record['matching_lines'] or ['(no matching symbols)'])
        lines.append('')
    (HERE / 'namespace-evidence.txt').write_text('\n'.join(lines))
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    assert digest(CMS / 'artifact-index.json') == index_hash
    for entry in index['files']:
        path = ROOT / entry['path']
        assert digest(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], entry['path']
    return {'scope': 'Read-only production-qualification outline and source/ELF metadata audit.',
        'production_source_changed_by_audit': False, 'compiler_or_numerical_jobs_run': False,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'frozen_CMS_index_sha256_before': index_hash, 'frozen_CMS_index_sha256_after': index_hash,
        'frozen_CMS_indexed_files_verified': len(index['files']),
        'frozen_CMS_indexed_files_verified_before_and_after': True,
        'frozen_CMS_primary_extreme_points_qualified': index['primary_extreme_points_qualified'],
        'frozen_CMS_production_cuttools_changed': index['production_cuttools_changed'],
        'source_sha256': hashes,
        'binary_metadata_inputs_sha256': {name: digest(ROOT / name)
            for name in (archive, ninja, collier, executable)},
        'binary_metadata_resolved_paths': {name: str((ROOT / name).resolve())
            for name in (archive, ninja, collier, executable)},
        'namespace_evidence_sha256': digest(HERE / 'namespace-evidence.txt'),
        'outline_sha256': digest(HERE / 'README.md'), 'audit_script_sha256': digest(Path(__file__)),
        'interpretation': 'Pinned symbols isolate CutTools scalar state from Ninja/Collier, but not from other CutTools callers. No new numerical qualification is established.'}


if __name__ == '__main__':
    result = audit()
    (HERE / 'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('PASS: frozen CMS index and all indexed files unchanged; source/ELF metadata recorded.')
