# Direct fixed-QP initialization diagnostic

The preserved `check_sa.f` is the frozen baseline's standalone template, rendered
with its Fortran writer for the same CMS process, `READPS=.TRUE.` and four calls
to the same physical point. It does not request `FORCE_STABILITY_CHECK`. The
numerical card selects NinjaRS alone with `CTModeInit=CTModeRun=4`, `CheckCycle=1`,
and the historical primary `OSThres=1e-8`. The template sets `MU_R=sqrt(s)`.

`backend-8-qp.log` and `backend-8-qp-result.dat` contain a successful four-call
run on ordinary point 1. The final return code 120 and accuracy -1 identify
fixed-mode QP without an independent stability certification. Its finite and
pole coefficients agree with the automatic NinjaRS result at this point to
approximately `1e-14`. The elapsed time is diagnostic context only.

`backend-8-point1-osthres-1e-24` repeats this exact direct driver and point with
only the on-shell threshold changed. It also completes, with identical printed
coefficients. Consequently the lower threshold alone does not explain the
earlier persistent-worker initialization timeouts. The generated
`SLOOPMATRIX_THRES` maps fixed-QP status 120 to initialization status 140 when
forced stability checks are enabled. The persistent worker requires a final
T=1 or T=2 status before issuing READY, so this combination cannot complete its
warmup. The standalone driver does not request those forced checks.

At point 20, TIRust QP gives identical coefficients with `OSThres=1e-14` and
`1e-24`; it agrees with NinjaRS QP within `2.02e-16` and Ninja QP within
`2.72e-14`. At point 21 with `OSThres=1e-24`, TIRust and NinjaRS agree within
`1.35e-16`, while Ninja differs by `1.21e-7`: this passes `3e-6` and misses the
ordinary-point `1e-8` target. A forced CutTools QP diagnostic at point 20
disagrees substantially with the other three. The failed comparisons are
retained in `summary.json`. These unchecked diagnostics do not replace the
primary failures or establish complete five-backend stress agreement.

`cuttools-adjudication/` causally diagnoses that CutTools discrepancy in isolated
generated copies. Its independent absolute scalar cutoff `1e-6` exceeds the
physical transfer and selects inappropriate zero-invariant scalar branches.
Reducing that private cutoff fixes the large discrepancy. The remaining
`1.2e-5` finite difference disappears when a private copied CutTools entrypoint
receives the existing repaired QP momenta directly: finite and pole coefficients
then agree with TIRust within `2.11e-11`. The independent MadGraph FKS double-pole
formula agrees with TIRust within `1.67e-16`. Original baseline/candidate CutTools
outputs are byte-identical. These controlled tests attribute the disagreement
to legacy scalar classification and DP denominator transport. Production
CutTools and the original unqualified extreme automatic-policy runs are
unchanged; no passing diagnostic replaces a failed primary observation.

`cms_mp_shell_probe.f` calls the actual generated `SET_MP_PS` and
`MP_BUILD_KINEMATIC_MATRIX`. Its raw output and 80-digit geometric audit show
that the promoted DP shell residuals near `1e-11 GeV²` become approximately
`2.5e-29 GeV²` after QP repair. At `OSThres=1e-24`, adjacent massless invariants
are zero while the physical nonzero forward invariants are retained.
The four-shift probe covers all 21 saved points.

The additional `actual-matrix-trace/` experiment records every matrix call in
four physical TIRust QP evaluations of point 20. Its 104 matrices contain 960
entries, including 296 diagonal entries, 416 entries matching individual
external momenta, and 40 entries carrying the small physical invariant.
The recovered external virtualities are zero; the physical invariant differs
from the 80-digit expectation by at most `1.31e-17` relative. The final amplitude
file is byte-identical to the untraced run. The trace only adds output statements
inside an ignored generated copy; production kinematic guards are unchanged.

The earlier `invalid-candidate-card-*` and `invalid-local-resource-path-*`
attempts stopped before reduction. Their setup failures are retained. The
fixed standalone baseline accepts only its own card parameter names, and
`MadLoopParams.dat` must be found in `MadLoop5_resources` together with the
matching color/helicity files.

After generating and building the baseline CMS export using the parent runner,
reproduce from a fresh ignored copy:

```sh
source tools/tirust_env.sh
cp -a .benchmark-work/tirust/outputs/baseline/cms_ew_n2_epem_mupmum .tirust-build/complex-stress-direct-qp
cp docs/benchmarkfolder/experiments/complex-mass/direct-qp/check_sa.f .tirust-build/complex-stress-direct-qp/SubProcesses/P0_epem_mupmum/check_sa.f
env PWD="$PWD/.tirust-build/complex-stress-direct-qp/SubProcesses/P0_epem_mupmum" make -C .tirust-build/complex-stress-direct-qp/SubProcesses/P0_epem_mupmum SHELL="$(command -v bash)" -j1 check
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/run_case.py reproduced-default --point 1
```

For backend 9, copy the candidate export into
`.tirust-build/complex-stress-direct-qp-candidate`, replace its `check_sa.f`
with the same preserved driver, and relink in that copy. The runner selects
the preserved `candidate-MadLoopParams.dat` automatically with `--backend 9`.
Recompute the retained comparisons and input audits with:

```sh
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/summarize.py
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/check_shell_audit.py
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/actual-matrix-trace/analyze.py
```

The per-case runner refuses to overwrite an existing evidence label, verifies
the frozen point stream and preserved driver, and uses the common numerical
lock. `--onshell-threshold` selects a separately recorded alternate numerical
card. It never changes the primary export or upgrades a diagnostic into a
qualified full-stream observation.
