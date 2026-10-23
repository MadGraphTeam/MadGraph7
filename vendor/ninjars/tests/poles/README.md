# Independent UV/IR pole checks

These fixtures compare the **bare tensor loop integral** with COLLIER, without
UFO counterterms or R2 vertices. The eight kinematic cases contain 455 individual
tensor components: massive tadpole, scaleless lightlike bubble, massive bubble,
massless off-shell bubble, massive triangle, infrared triangle, massive box and
complex-mass bubble. The largest tested rank is four. The scaleless bubble has
separate nonzero UV and IR residues even though their sum vanishes. Tensor
triangles and boxes also include nonzero UV residues.

The current qualification uses upstream NinjaRS
[`41a4c00e`](https://github.com/alphal00p/ninjars/commit/41a4c00e7635e3ab188fc26bacf0aae6941e8082)
and oneloopmaster
[`5b710437`](https://github.com/alphal00p/oneloopmaster/commit/5b710437ada83cbc6e8c6d51e56f7d6ee6a0edb5).
Pole selection is provided by those upstream libraries.
`upstream-migration-validation.json` binds the installed archive and dependency
pins to the fresh numerical campaign, production installation smoke, 31 passing
bridge Rust tests (7 ignored), and upstream provider/core test commands and counts.

Run from the MadGraph repository root after `install ninjars --update`:

```sh
python3 vendor/ninjars/tests/poles/tensor_validation.py \
  --lib HEPTools/ninjars/lib \
  --output .benchmark-work/uv-ir-independent
```

The driver uses the installed Fortran module and native linking metadata. The
default runs all 455 components in both double precision and the binary128-input,
256-bit rescue. `--focused` selects 61 representative components. The precision
can also be selected explicitly with `--precision dp` or `--precision mp`.
Every context cycles through `both, UV, IR, none, both, IR, UV, none`; repeated
evaluations of the same mode must agree exactly, including all printed rescue
digits. Binary128 inputs and selected poles traverse the production Fortran ABI.
Run the two precisions serially: the restricted Symbolica license permits one
active numerical worker per user.

Each output directory records the input stream, raw numerical values, statuses,
comparison results, compiler logs and hashes of the reference and installed
library. No generated binaries belong in this source directory. The default
check requires:

- All UV-only and IR-only DP evaluations to succeed and agree with COLLIER.
- All four modes in the higher-precision evaluation to succeed and agree.
- Exact agreement of repeated evaluations after changing modes in a reused
  context.
- Successful higher-precision evaluation of every ordinary-DP unstable control
  when both precisions are run together.

Ordinary DP uses the unchanged fast engine. The original ABI-3 library already
returns `UNSTABLE` for the wholly spacelike massive-box case. Fourteen rank-four
massive-triangle components also miss the strict COLLIER tolerance in the native
combined result, by at most 4.09 times its bound. The historical ABI-3 to first
ABI-4 comparison found identical combined DP outputs for all 455 components,
including statuses. The current selected DP path passes all components, and the
rescue passes all modes. These
ordinary-DP diagnostics remain visible in the JSON results. To make them fatal
as well, use `--diagnostic-all`; this intentionally fails on the recorded native
DP controls. They are not counted as newly validated native-DP accuracy results.
`tensor-validation.json` records the current eight-mode-cycle qualification.
`native-dp-baseline.json` preserves the historical ABI-3 to first ABI-4
combined-output comparison. The subsequent migration to upstream pole APIs
reproduces **all 7,280 prior ABI-4 DP/MP output records exactly**, across all four
modes and their repeats, including Laurent coefficients, rational terms and
statuses. This includes 910 combined-mode records in each precision. The
baseline and current library, input and output hashes are bound in
`upstream-migration-validation.json`.

The numerical comparison uses
`abs(a-b) <= 1e-7*max(abs(a),abs(b)) + 1e-10*S` for each complex coefficient.
Here `S` is the largest COLLIER finite/UV/IR component at the same kinematic point
and homogeneous tensor rank, so all components have the same mass dimension.
If an entire odd tadpole tensor vanishes by symmetry, the nonzero scalar
reference at that point is multiplied by
`max(mu_R^2, abs(m_i^2))^(rank/2)` to provide the corresponding dimension. There
is no arbitrary absolute tolerance floor. The original zero components are
retained and checked.

## COLLIER reference provenance and reproduction

The committed reference was generated with **COLLIER 1.2.9**, the COLI branch
(`SetMode_cll(1)`), its default required accuracy, GNU Fortran 15.3.0 on Linux
x86_64, and the installed library used by MadLoop. Every COLLIER error and
accuracy flag was zero. `provenance.json` records the input, driver, library and
reference checksums.
The reference JSON SHA-256 is
`18f6694dd5af873d847e39bd299957454c698aeb31f3bf4b2389593bc44d3970`.

The COLLIER driver evaluates each tensor four times with independent regulator
settings. It takes differences from the finite result to extract UV, IR single
and IR double poles, matching MadLoop's COLLIER wrapper. The base uses
`DeltaUV=DeltaIR1=0`, `DeltaIR2=pi^2/6`, and equal UV/IR scales; the latter shift
matches the Ninja/OneLOop normalization. Numerators use contravariant components
of `q`, with denominators `(q+p_i)^2-m_i^2`. There are no extra symmetry factors.
Complex squared masses are passed without changing their signs.

Regenerate the fixture with an independently installed COLLIER:

```sh
python3 vendor/ninjars/tests/poles/generate_collier_reference.py \
  --collier HEPTools/collier \
  --work-dir .benchmark-work/uv-ir-collier-reference \
  --output .benchmark-work/uv-ir-collier-reference/reference.json
```

Then pass `--reference` with the regenerated JSON to `tensor_validation.py`.
The generator checks the COLLIER status flags and tensor component counts.

`collier_poles.f90` adapts the tensor reference driver from
[NinjaRS at revision aa6a6307](https://github.com/alphal00p/ninjars/blob/aa6a6307dd081dc3714f9accebc0df9e5561ae73/validation/rank-n2/harness/collier_tensor_driver.f90),
separating the UV and IR regulator variations. This directory uses the same
GPL-3.0-or-later license as the bridge and NinjaRS.

## Generated MadLoop comparisons

`madloop-validation.json` records a separate, actual CLI-generated campaign:

| Process | Model | COLLIER CT modes | NinjaRS CT modes |
|---|---|---|---|
| `u u~ > z [virt=QCD]` | `loop_sm` | 1 | −1, 1, 2, 4, 5 |
| `g g > t t~ [virt=QCD]` | `loop_sm` | 1 | −1, 1, 2, 4, 5 |
| `e+ e- > mu+ mu- [virt=QED]` | `loop_qcd_qed_sm_Gmu` | 1 | 1, 4 |
| `g g > h [sqrvirt=QCD]` | `loop_sm` | 1 | 1, 4 |

Every listed configuration uses all four explicit pole selectors and three
retained phase-space points, with all helicities and color sums. Two additional
legacy-filter controls bring the total to **74 configurations and 444 rows**
(each point has a total and a split-order row). Three runtime mode-switch series
add 144 rows. All 187 numerical, cache, status and completeness comparisons pass;
the largest numerical difference is `8.37505e-5` times the allowed tolerance.
Fixed DP modes return 110, forced QP returns 120, and all automatic stability
checks in this campaign return 218. Fixed-mode return codes do not assert a
separate stability acceptance test.

The same disposable generated sources are used for both tools. Immediately
after counterterm construction, the harness zeroes `AMPL(:,1:NCTAMPS)` and its
corresponding MP/DP copies, before interference or squaring. The original
generated sources and their hashes are retained in the working output. This
removes UFO UV and R2 terms without modifying the loop numerator, reduction
wrapper or production source. The reported quantities therefore test the bare
loop contribution. For loop-induced output, each selected squared amplitude is
compared directly with COLLIER; UV+IR additivity is **not** asserted for squared
amplitudes.

Reproduce the full matrix from the repository root:

```sh
python3 vendor/ninjars/tests/poles/madloop_validation.py --campaign \
  --output .benchmark-work/uv-ir-madloop \
  --ninjars HEPTools/ninjars/lib --collier HEPTools/collier \
  --ew-model /path/to/loop_qcd_qed_sm_Gmu
```

The output contains the command cards submitted to `bin/madgraph`, generated
sources, build logs, parameter cards, numerical outputs and the comparison JSON.
`--points` can select another directory with the four point-file names retained
under `points/`; the qualification above uses the committed files and zero-width
cards under `cards/`. The exact model source checksums and the immutable archive
URL for the EW model are recorded in `madloop-validation.json`. The EW source
is the converted UFO used in the existing benchmark, rather than an unverified
new download. COLLIER and NinjaRS must already be installed.

For individual checks, use `--case qcd-z --prepare` to generate and compile,
then `--case qcd-z --backend 8 --ctmode 4` to evaluate its four pole modes.
The complete campaign and comparator reject missing configurations, missing or
duplicate point/order rows, nonfinite values, unexpected return codes, and
incomplete runtime-switch sequences.

The current campaign regenerated and compiled **all four processes** through
the portable harness against the upstream installation. Every one of its 74
configuration result records agrees exactly with the earlier ABI-4 campaign,
including statuses. The earlier 12-configuration `qcd-z` portability smoke is
retained as historical provenance in `madloop-validation.json`; the current
qualification uses the full fresh campaign. No new timing claim is made by
this numerical migration check.
