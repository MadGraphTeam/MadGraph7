# Actual-shape DP factorization: completed QCD3 controls

The copied fresh-development QCD3 source uses70 coefficients and136 selected adjacent DP FFV full-update calls (ranks2/3 only). Its preexisting private QP implementation is identical in all three clones. Production DP remains unchanged.

Original, capture-only and factor-trace were evaluated sequentially for original Ninja6 and NinjaRS8 using all100 fixed points, normal stability, MLStabThres1e-9 and COLLIERRequiredAccuracy1e-12. Every shared setting was asserted equal to its bound fresh-development reference. Capture-only adds the same new batch IEEE import and immediately-before-call INEXACT query while retaining the original updater; it exactly reproduces all original rows, accuracy estimates and statuses in both backends. This isolates the new IEEE boundary for these physical inputs; it is not a general generated-batch trap-semantics proof.

The factor changes association and may change zero signs. Exact bit identity is therefore not its acceptance contract. Every Born/Laurent component and split-order row passed the common policy-v2 mixed bound. The independent all-six-stream comparison is retained separately. Instrumentation counts only the final full100-point pass; warm-up and numerical-output passes occur before the reset. All originals, generated sources, counters, return-code histograms and raw output rows remain available.

Backend6: original statuses{'216': 98, '328': 2}; factor statuses{'216': 98, '328': 2}. Factor changes444 printed coefficient bits and196 accuracy/status rows. Maximum within-backend tolerance fraction0.187690071. Guard totals: 326400 calls, 326400 admitted, 0 fallback. QP points original[44, 80], factor[44, 80].

Backend8: original statuses{'218': 98, '322': 2}; factor statuses{'218': 98, '322': 2}. Factor changes472 printed coefficient bits and196 accuracy/status rows. Maximum within-backend tolerance fraction0.0519808031. Guard totals: 326400 calls, 326400 admitted, 0 fallback. QP points original[44, 80], factor[44, 80].

All-six-stream tolerance fraction: 0.819445014 across12000 component comparisons. The Ninja6 rescue-reduction hypothesis is negative: both original common rescue points remain. Any backend8 change is separately listed above; no stability threshold or rescue policy was weakened.

Timing fields are diagnostic only: these runs include trace counters, initialization and concurrent task work. No process speed claim follows. A separately prepared uninstrumented fixed21-pair plan remains outside this seal.

Source-selection setup rejections are retained: strict full-file identity first rejected the fresh ALOHA declaration reordering; the executable statement sequence, declaration multiset and exact coefficient/row maps were then proved identical. A second inventory attempt incorrectly assumed both orientations began with P2 assignment; it was corrected to admit the explicitly checked P1/P2 orientation. No amplitudes ran in either failed inventory attempt.

Read-only verification from the repository root:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/process-qcd3-v1/verify.py
```

`run.py prepare`, then `run.py run --backend 6` and `--backend 8` reproduce the three-cohort method in a newly renamed report/work namespace. Preparation refuses existing outputs. Backend8 requires sole Symbolica runtime-slot ownership and the shared supported lock-directory environment. Keep local TMPDIR/build/cache paths. The retained source-selection manifest, approved plan, generated Fortran, physical card, point file and library/module metadata bind the exact tested experiment. Executables and objects are excluded build products with their hashes retained.
