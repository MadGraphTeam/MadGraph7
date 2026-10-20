#!/usr/bin/env python3
"""Retain and seal completed diagnostic evidence without amplitude evaluation."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import shutil

import run as control
import verify
import ninjars_benchmark_policy_v2 as policy

HERE, BASE, ROOT = control.HERE, control.BASE, control.ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'SHA256SUMS.json').exists()
    groups,results={},{}
    for backend in (6,8):
        result=json.loads((HERE/f'backend-{backend}/result.json').read_text())
        assert result['complete'] and result['comparison']['passed']
        results[backend]=result
        for variant,record in result['measurements'].items():
            actual=verify.raw(HERE/f'backend-{backend}'/variant/'values.dat')
            assert actual==record['rows']
            groups[f'{backend}/{variant}']=actual
    maximum,count=verify.comparisons(groups)
    control.dump(HERE/'all-six-stream-comparison.json',dict(passed=True,comparisons=count,
        maximum_tolerance_fraction=maximum,stream_count=6,points=100,orders=2))
    parent=HERE/'parent-source'
    parent.mkdir(exist_ok=False)
    source_sub=next(control.SOURCE.glob('SubProcesses/P*_*'))
    for backend in (6,8):
        shutil.copy2(control.REFERENCE/f'data/candidate-qcd-3-{backend}-0.json',parent/f'reference-{backend}.json')
    for name in ('generation.json','compilation.json','compiled-sources.json','Cards/param_card.dat','Source/make_opts'):
        target=parent/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(control.SOURCE/name,target)
    for source in [*source_sub.glob('mp_*.f'),*(control.SOURCE/'Source/DHELAS').glob('FFV1L[12]_[12].f')]:
        target=parent/'generated'/source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source,target)
    plan=HERE/'source-selection'
    shutil.copytree(BASE/'process-plan-v1',plan,ignore=shutil.ignore_patterns('__pycache__'))
    runtime=json.loads((HERE/'runtime.json').read_text())
    preparation=json.loads((HERE/'preparation.json').read_text())
    for variant in control.VARIANTS:
        output=control.WORK/variant
        sub=control.directory(variant)
        bound=preparation['builds'][variant]
        assert sha(sub/'check')==bound['executable_sha256']
        assert control.bench.generated_source_manifest(output)['files_sha256']==bound['generated_sources_sha256']
        assert policy.library_hashes(output)==runtime['variants'][variant]['libraries']
        assert {n:policy.bound_file(p) for n,p in policy.runtime_metadata_files(output).items()}==runtime['variants'][variant]['metadata']
    description='''# Actual-shape DP factorization: completed QCD3 controls

The copied fresh-development QCD3 source uses70 coefficients and136 selected adjacent DP FFV full-update calls (ranks2/3 only). Its preexisting private QP implementation is identical in all three clones. Production DP remains unchanged.

Original, capture-only and factor-trace were evaluated sequentially for original Ninja6 and NinjaRS8 using all100 fixed points, normal stability, MLStabThres1e-9 and COLLIERRequiredAccuracy1e-12. Every shared setting was asserted equal to its bound fresh-development reference. Capture-only adds the same new batch IEEE import and immediately-before-call INEXACT query while retaining the original updater; it exactly reproduces all original rows, accuracy estimates and statuses in both backends. This isolates the new IEEE boundary for these physical inputs; it is not a general generated-batch trap-semantics proof.

The factor changes association and may change zero signs. Exact bit identity is therefore not its acceptance contract. Every Born/Laurent component and split-order row passed the common policy-v2 mixed bound. The independent all-six-stream comparison is retained separately. Instrumentation counts only the final full100-point pass; warm-up and numerical-output passes occur before the reset. All originals, generated sources, counters, return-code histograms and raw output rows remain available.

'''
    for backend,result in results.items():
        factor=result['measurements']['factor-trace']
        original=result['measurements']['original']
        traces=factor['guard_or_capture_counts']
        total,fast,fallback=(sum(row[i] for row in traces) for i in (1,2,3))
        old_rescue=[r['point'] for r in original['rows'] if r['order']==0 and r['status']//100==3]
        new_rescue=[r['point'] for r in factor['rows'] if r['order']==0 and r['status']//100==3]
        description+=(f'Backend{backend}: original statuses{original["status_counts"]}; factor statuses{factor["status_counts"]}. '
            f'Factor changes{result["printed_coefficient_bit_changes"]} printed coefficient bits and'
            f'{len(result["accuracy_or_status_changes"])} accuracy/status rows. Maximum within-backend tolerance fraction'
            f'{result["comparison"]["max_tolerance_fraction"]:.9g}. Guard totals: {total} calls, {fast} admitted, {fallback} fallback. '
            f'QP points original{old_rescue}, factor{new_rescue}.\n\n')
    description+=f'''All-six-stream tolerance fraction: {maximum:.9g} across{count} component comparisons. The Ninja6 rescue-reduction hypothesis is negative: both original common rescue points remain. Any backend8 change is separately listed above; no stability threshold or rescue policy was weakened.

Timing fields are diagnostic only: these runs include trace counters, initialization and concurrent task work. No process speed claim follows. A separately prepared uninstrumented fixed21-pair plan remains outside this seal.

Source-selection setup rejections are retained: strict full-file identity first rejected the fresh ALOHA declaration reordering; the executable statement sequence, declaration multiset and exact coefficient/row maps were then proved identical. A second inventory attempt incorrectly assumed both orientations began with P2 assignment; it was corrected to admit the explicitly checked P1/P2 orientation. No amplitudes ran in either failed inventory attempt.

Read-only verification from the repository root:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/process-qcd3-v1/verify.py
```

`run.py prepare`, then `run.py run --backend 6` and `--backend 8` reproduce the three-cohort method in a newly renamed report/work namespace. Preparation refuses existing outputs. Backend8 requires sole Symbolica runtime-slot ownership and the shared supported lock-directory environment. Keep local TMPDIR/build/cache paths. The retained source-selection manifest, approved plan, generated Fortran, physical card, point file and library/module metadata bind the exact tested experiment. Executables and objects are excluded build products with their hashes retained.
'''
    (HERE/'README.md').write_text(description)
    files={}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.o','.mod','.pyc'):
            continue
        assert not path.is_symlink()
        data=path.read_bytes()
        assert not data.startswith(b'\x7fELF')
        files[str(path.relative_to(HERE))]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    control.dump(HERE/'SHA256SUMS.json',dict(schema='dp-factor-actual-qcd3-diagnostic-v1',
        complete=True,acceptance_eligible=False,file_count=len(files),files=files,
        scope='Actualshape70 isolated DP full-factor correctness/IEEE-boundary/guard evidence for Ninja6/NinjaRS8; no performance claim'))
    print('Sealed',len(files),'files; no amplitudes run by capture')


if __name__=='__main__':
    main()
