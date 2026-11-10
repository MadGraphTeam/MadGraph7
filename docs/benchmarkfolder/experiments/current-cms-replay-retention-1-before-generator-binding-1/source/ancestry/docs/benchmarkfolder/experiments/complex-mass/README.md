# Independent complex-mass qualification

This campaign runs `e+ e- > mu+ mu- [virt=QED]` with the pinned
`loop_qcd_qed_sm_Gmu` model and the real MG CLI option
`set complex_mass_scheme True`. Nonzero parameter-card widths are retained.
The generated model is compiled and a separate Fortran probe executes `SETPARA`
to verify `CMSParam=1` and complex squared W/Z masses equal to `M²-iMΓ`.
Probe source, raw values, model/card hashes, and compiler identity are retained.

The 21 deterministic physical points comprise 12 ordinary 1 TeV samples,
three Z-resonance samples (`MZ-ΓZ`, `MZ`, `MZ+ΓZ`), three internal W-pair
threshold samples (`2MW*(1-1e-6)`, `2MW`, `2MW*(1+1e-6)`), and three forward
samples at angles `1e-3`, `1e-6`, and `1e-8`. The forward normalized Gram
determinants are approximately `5e-7`, `5e-13`, and `5e-17`. Their values are
computed with 80-digit decimal arithmetic from the serialized physical inputs.
The three independent vectors are incoming legs 1/2 and outgoing leg 3;
a four-vector Gram determinant would vanish identically for 2→2 scattering.

Run from the MadGraph repository after installing all reducers:

```sh
source tools/tirust_env.sh
python tools/tirust_complex_stress.py generate --source baseline
python tools/tirust_complex_stress.py build --source baseline
python tools/tirust_complex_stress.py points --source baseline
python tools/tirust_complex_stress.py run --source baseline --backend 1
python tools/tirust_complex_stress.py run --source baseline --backend 6
python tools/tirust_complex_stress.py run --source baseline --backend 7
python tools/tirust_complex_stress.py run --source baseline --backend 8
python tools/tirust_complex_stress.py generate --source candidate
python tools/tirust_complex_stress.py build --source candidate
python tools/tirust_complex_stress.py run --source candidate --backend 9
python tools/tirust_complex_stress.py summarize
```

Candidate exports must use the committed integration source and installed final
TIRust build. Automatic runs preserve MadLoop's own stability checks and report
actual precision usage, including genuine-QP rescues. Fixed-QP diagnostics use
the separate `direct-qp/` standalone driver. The persistent worker forces
stability checks, which remap unchecked fixed-QP statuses to initialization
statuses and prevent its warmup from ending. The earlier three fixed-QP worker
timeouts are retained with protocol errata; they do not establish reducer
failures or QP performance. The runner now rejects that incompatible request.
Numerical
execution for backends 8 and 9 acquires `.tirust-build/numerical-tests.lock`,
shared with the native bridge audits. `--point-group ordinary` selects the
12 ordinary samples, and `--point-group moderate` selects the first 19 samples;
both write separately pinned streams and leave all omitted stress failures
visible. `--timeout` bounds initialization and protocol waits, and `--tag`
retains a separate retry directory.

This is a numerical stress campaign, not a main-matrix or runtime acceptance
campaign. All raw coefficients, status codes, logs and failures are retained.
The comparison uses the existing MadLoop `3e-6` regression criterion and reports
the `1e-8` ordinary-point target separately. Failed or missing runs are not
qualified, and rerunning requires preserving the original run directory first.

The retained primary stress policy uses `MLStabThres=1e-3`; it is independent
of the stricter main benchmark policy. All five reducers complete the first
19 points and satisfy the `3e-6` comparison. On the 12 ordinary points, native
DP TIRust differs from the four references by at most `5.96e-13`. At the
`1e-3` forward angle the largest relative difference is `2.13e-6`, so this
stress point does not meet the ordinary-point `1e-8` target. The two more
extreme forward points remain unqualified: TIRust, Ninja and NinjaRS return
failure statuses on both, and COLLIER on the most extreme point. The complete
21-point CutTools attempt was interrupted before initialization finished;
that interruption is not a numerical disagreement. Its separate 19-point
run succeeds. No failed whole-stream run becomes qualified through a passing
subset comparison.

The baseline full source rebuild produces exactly the executable bytes used
by the earlier observations. `inputs/baseline/source-rebuild-equivalence.json`
binds those observations to the rebuild and verifies the recorded dependency
hashes. The original run metadata and failures are preserved.

The baseline helper probe in
`inputs/baseline/cms_ew_n2_epem_mupmum/cms_onshell_probe.f` isolates the existing
DP and QP `BUILD_KINEMATIC_MATRIX` routines. At the primary `OSThres=1e-8`,
both replace the nonzero spacelike invariants of the two most forward points
with zero. This is a shared historical guard behavior; the production guard is
unchanged. The separate `direct-qp/run_case.py` adjudication runner accepts
`--onshell-threshold 1e-24`, saves the alternate numerical card, and cannot
satisfy the primary-policy qualification gate. Lowering the guard in DP alone
would risk interpreting floating-point on-shell residuals as physical masses.

The separate `direct-qp/cuttools-adjudication/` experiment resolves the large
CutTools-versus-TIRust point-20 discrepancy causally. Two isolated interventions
identify CutTools' independent scalar cutoff and its DP denominator transport.
With the private cutoff corrected and repaired QP momenta passed directly, the
existing CutTools reduction agrees with TIRust within `2.11e-11`; MadGraph's
independent QED double-pole formula agrees within `1.67e-16`. Baseline and
candidate CutTools produce byte-identical original outputs. These diagnostic
changes are confined to ignored copies. The original two extreme points remain
unqualified under the primary automatic policy, and their failures are retained.

`artifact-index.json` binds every retained CMS source, card, raw log, input,
failure, summary, and helper/test source by SHA-256. Verify the frozen delivery
from the repository root with:

```sh
python docs/benchmarkfolder/experiments/complex-mass/freeze_artifacts.py
```

Raw `.log` files are ignored by the repository's general rules and must be
force-added using the indexed paths when publishing this evidence.
