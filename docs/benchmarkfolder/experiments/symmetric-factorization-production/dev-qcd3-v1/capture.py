#!/usr/bin/env python3
"""Retain a completed fresh-export diagnostic without rerunning amplitudes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT/'tools'))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'SHA256SUMS.json').exists(), 'Preserve the completed capture'
    result = json.loads((HERE/'result.json').read_text())
    assert result['complete'] and result['agreement']['passed']
    assert not result['acceptance_eligible']
    assert result['factored_call_sites'] == 136
    assert result['ninja_rows_equal_sealed_original']
    work = ROOT/'.benchmark-work/experiments/symmetric-factorization-production/dev-qcd3-v1'
    output = work/'outputs/candidate/qcd-3'
    directory, = output.glob('SubProcesses/P*_*')
    compiled = json.loads((HERE/'inputs/candidate/qcd-3/compiled-sources.json').read_text())
    assert bench.generated_source_manifest(output) == compiled
    compilation = json.loads((output/'compilation.json').read_text())
    assert sha(directory/'check') == compilation['executable_sha256']
    for source in sorted((work/'logs').glob('*.log')):
        target = HERE/'logs'/source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
    for pattern in ('polynomial.f', 'mp_coef_construction_*.f',
                    'mp_compute_loop_coefs.f', 'coef_specs.inc',
                    'loop_max_coefs.inc', 'check_sa.f', 'makefile'):
        for source in sorted(directory.glob(pattern)):
            target = HERE/'generated'/source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
    for name in ('values.dat', 'timing.dat', 'timed_status.dat', 'dimensions.dat'):
        target = HERE/'last-backend8-raw'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(directory/name, target)
    for name in ('madgraph/loop/loop_exporters.py',
                 'tests/unit_tests/loop/test_factorized_wl_export.py'):
        target = HERE/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    unit_log = ROOT/'.benchmark-work/logs/factor-production-export-tests-20261020-v2.log'
    shutil.copy2(unit_log, HERE/'export-tests.log')
    runtime = dict(libraries=policy.library_hashes(output),
        metadata={name: policy.bound_file(path)
                  for name, path in policy.runtime_metadata_files(output).items()},
        executable_sha256=sha(directory/'check'),
        compiled_sources_sha256=compiled['files_sha256'])
    bench.dump(HERE/'runtime.json', runtime)
    files = {}
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            files[str(path.relative_to(HERE))] = dict(sha256=sha(path), bytes=path.stat().st_size)
    bench.dump(HERE/'SHA256SUMS.json', dict(
        schema='fresh-qp-factor-cli-diagnostic-v1', files=files,
        scope='Fresh dirty-patch CLI QCD3 export/build and 100-point Ninja6/NinjaRS8 '
              'correctness. Backend6 rows equal the sealed original; both use '
              '98 DP and two QP evaluations. Timing fields are diagnostic only; '
              'this is not a final clean-source acceptance collection.'))
    print('Captured', len(files), 'files; final acceptance remains pending.')


if __name__ == '__main__':
    main()
