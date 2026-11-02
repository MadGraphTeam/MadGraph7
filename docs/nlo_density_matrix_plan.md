# Spin-density matrices in the FKS standalone output: plan

Goal: from `output standalone_fortran --fks`, return for one phase-space point
the spin-density matrix rho(h1,h2) of **the massive particle(s) the user
selects** (e.g. the W+ in `p p > w+ j`, the W+ and Z in `p p > w+ z`, the t and
t~ in `p p > t t~`). Every other leg is summed over helicity, as today. This is
returned for **every NLO building block**:

- the Born, B;
- the colour- and charge-linked Borns, B_kl;
- the spin-correlated Born, B~;
- the extra-counterterm Borns (`born_cnt_N.f`);
- the real emission, R;
- the tree x loop virtual, V (finite part, 1/eps and 1/eps^2).

Then check the **soft and collinear limits of every density-matrix element**,
not only of its trace.

Status (2026-10-31): **M0 done; M3 items 1-4 (the MadLoop virtual) done;
M1 Born family done for the Born, colour links and spin-correlated Born.**
The extra-counterterm Born (`born_cnt`) has been refactored and is in
validation. M2 and M3 item 5 (poles per element in the FKS output) are still
to do. See section 6 for what was implemented
and how it was validated. The work builds on three existing pieces:

| piece | where | what it gives us |
|---|---|---|
| polarised NLO frame | `Template/NLO/SubProcesses/boost_to_frame.f`, `docs/nlo_polarisation_boost_plan.md` | an IR-safe helicity frame (`me_frame`), `*_frame` wrappers for every Born/real/virtual call |
| LO density | `matrix_standalone_v4.inc` (`GET_AMP` -> `GET_JAMP` -> `GET_MATRIX` / `GET_INTER` -> `GET_ALL_INTER` / `GET_DENSITY`) | the target code shape |
| loop-induced density | `loop_optimized/compute_color_flows.inc` (`STORE_JAMP`, `JAMPL_ALL`, `GET_ALL_INTER`, `GET_DENSITY`) | the MadLoop half of the virtual |

## 1. Where things stand today

### 1.1 FKS standalone output

The output command is `output standalone_fortran --fks [--limits]`. It is
written by `ProcessExporterFortranFKS_SA`
(`madgraph/iolibs/export_fks.py:5692`).

- **Default (Born-only) mode.** `trim_born_dir` keeps only `check_sa_fks.f`,
  `born.f`, `sborn_sf.f` and the split-order helpers (`export_fks.py:5762`).
  `check_sa_fks.f` prints B, B~ and every B_kl at one RAMBO point. It does no
  frame boost: neither `boost_to_frame.f` nor `frame_info.inc` is kept.
- **`--limits` mode.** Keeps the reals and runs `test_soft_col_limits`
  (`launch_ext_program.py:668`).
- **The virtual is never shipped.** `V*/` and `MadLoop5_resources` are dropped
  in both modes (`drop_virtuals`, `export_fks.py:5747`).

### 1.2 Born templates

`born.f` comes from `bornmatrix_colorlinks_fks.inc` (with
`born_color_links_in_born = True`, the default).

- It is already partly factorised: `GET_AMP_BORN` (781) -> `GET_JAMP_BORN`
  (825) -> `BORN_COLOR_CONTRACT` / `BORN_COLOR_SQUARE`.
- It also has a helicity-summed colour density `/C_BORN_RHO/`
  (`BORN_RHO_ADD`/`TRACE`), from which every link is computed as
  `Tr(S^l R)` (see the fks-colour-link-density work).
- That `R` is a **colour** density summed over all helicities, not a spin
  density. The spin-correlated Born pairs `J(h_g)` with `J(-h_g)` of the FKS
  gluon, through `BORN_COLOR_CONTRACT(0,JAMPH(2),JAMPH(1))` (775).

### 1.3 Real and extra-counterterm templates

- **Real** (`realmatrix_splitorders_fks.inc`): `MATRIX_%(p)s` (201) has HELAS
  calls, JAMPs and the colour sum all inline. There is no
  `GET_AMP`/`GET_JAMP` split.
- **Extra counterterm** (`born_cnt_splitorders_fks.inc`): `BORN_CNT%d` (224)
  is inline in the same way.

### 1.4 MadLoop virtual

- **`--density` reaches only part of the code.** It sets `UseDensity`
  (`loop_exporters.py:861`), but the density routines live in
  `compute_color_flows.f`. That file is written only when `ComputeColorFlows`
  is on, which today means loop-induced or the `compute_color_flows` option.
  So `[virt=QCD] ... --density` silently produces nothing.
- **Tree x loop is explicitly unsupported.** `JAMPB` is a single-helicity
  common block (there is no `JAMPB_ALL`), and the interference keeps only the
  real part:

  ```fortran
  TEMP(K) = 2.0d0*HEL_MULT*DBLE(COLOR_COEF*JAMPL1(K,I,M)*DCONJG(JAMPB(J,N)))
  ```

  The comment at `compute_color_flows.inc:757-763` says so.
- **Bug:** `GET_HEL_INDEX` hard-codes `COMMON/ML5_0_HELCONFIGS/` instead of
  `%(proc_prefix)s`.

### 1.5 Frame

- `boost_to_frame.f` provides these wrappers: `sborn_frame`,
  `sborn_onehel_frame`, `sborn_sf_frame`, `extra_cnt_frame`,
  `smatrix_real_frame` and `binothlha_frame`.
- They use the masks from `get_frame_mask_born`/`_real`. The real mask is
  derived from the Born one through `i_fks`.
- `fks_singular.f` and `test_soft_col_limits.f` already call those wrappers
  everywhere.
- `RunCardNLO.check_validity` (`banner.py:6145`) refuses a frame built from a
  single beam, and one that mixes beam and final-state legs. Those are exactly
  the non-IR-safe cases.

## 2. Design decisions

### D1 — One kernel set per matrix element, LO style

Every tree-level piece (Born, extra-cnt Born, real) is rewritten as these
routines:

```
GET_AMP_x(P, NHEL, AMP)                    HELAS calls only
GET_JAMP_x(AMP, JAMP)                      colour decomposition, JAMP(NCOLOR,NAMPSO)
GET_MATRIX_x(JAMP, ANS)                    colour sum, per squared split order (real)
GET_INTER_x(JAMP1, JAMP2, ILINK, INTER)    complex sesquilinear form with colour matrix ILINK
                                           (ILINK=0 plain colour matrix, 1..NLINKS links)
GET_ALL_INTER_x(P, NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, ..., INTER)
GET_DENSITY_x(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, ..., INTER)
```

- **Signatures.** They match `matrix_standalone_v4.inc` (`GET_INTER` at 466,
  `GET_ALL_INTER` at 609, `GET_DENSITY` at 501).
- **Output layout.** The packed upper triangle has length
  `N_COMB*(N_COMB+1)/2`, and the changing-leg index runs fastest in
  `ALLOW_HEL`, as at LO.
- **Existing entry points stay.** `SBORN`, `SB_SF_LINK`, `SMATRIX_x`,
  `SBORN_CNTn` and `BORN(...,IMODE,...)` keep their signatures and are
  reimplemented on top of the kernels. `fks_singular.f` is not touched.
- **Python side.** `get_JAMP_lines`, `get_JAMP_lines_split_order`,
  `HelasCallWriter` and `get_born_color_link_dict` already produce the routine
  bodies. Only the templates change.
- **MadLoop counterpart.** The "get_jamp" step is `COMPUTE_COLOR_FLOWS` plus
  `STORE_JAMP`, and the "get_inter" step is `DO_COMPUTE_INTER_JAMP`. These are
  renamed and argument-aligned, so the virtual reads like the tree pieces.

### D2 — The FKS machinery sees densities through a hermitian projector W

Every counterterm formula in `fks_singular.f` (`sbornsoft`, `sborncol_*`,
`sreal_deg`, `getpoles`, `bornsoftvirtual`) is **linear** in B, B_kl and B~,
with real coefficients that do not depend on the helicity of the density legs.

Every scalar matrix-element routine therefore gets a global projector
`/DENSITY_PROJ/ W(N_COMB,N_COMB)`, and returns `Tr(W rho)` instead of the
helicity sum over the changing legs:

- **W = identity:** today's unpolarised result. This is the default, so the
  feature is inert unless requested.
- **W = E_ii:** the polarised result for helicity combination i. This agrees
  with `t{+}`-style polarisation, which gives a free cross-check.
- **W = E_ij + E_ji:** gives `2 Re rho_ij`.
- **W = i(E_ij - E_ji):** gives `-2 Im rho_ij`.

For hermitian W, every block returns a **real** scalar, including the
azimuthal term of B~ (the proof is in 3.3). So `amp_split`, the `wgt` arrays,
`fks_singular.f`, `check_poles.f` and `test_soft_col_limits.f` all work
unchanged.

"Check every element" becomes "run the existing limit test once per element
of a hermitian basis" (`N_COMB^2` runs). `GET_DENSITY_x` stays as the
full-matrix API for printing and for f2py.

Rejected alternative: vectorise `fks_singular.f` over density elements. It is
invasive, it duplicates validated code, and every new counterterm would need
two implementations.

### D3 — Frame: reuse `me_frame`, defaulting to the density legs

- **Default frame.** The rest frame of the summed momenta of the selected
  legs, which are all massive and final state (D4).
  - With one leg (`p p > w+ j`), this is the W rest frame. It relies on
    `impose_frame_rest`, as validated for `p p > z{0} j [QCD]` with
    `me_frame=[3]`.
  - With two legs (`p p > w+ z`, `t t~`), it is the pair rest frame, as
    validated with `me_frame=[3,4]`.
- **Configuration.** It can be set by `--me_frame=...` on the output line
  (Born-only mode has no run_card) or by the run_card in `--limits` mode. The
  same `check_validity` rules apply: no single-beam frame and no mixed frame.
- **Implementation.** It is written into `frame_info.inc` as a hard-coded
  `frame_id` default. `boost_to_frame.f`, `impose_frame_rest.f` and
  `frame_info.inc` are added to the trimmed Born-only directory. Every call in
  `check_sa_fks.f` goes through the `*_frame` wrappers.
- **Why it matters more than for polarisation.** Off-diagonal elements depend
  on the little-group phases of the density legs, which a polarised (diagonal)
  run never sees. The frame must therefore be IR safe for the **phases**, not
  only for the helicity assignment. The negative control in M2 checks exactly
  this.

### D4 — Which legs carry density indices

The user lists the **massive final-state particle(s)** whose density matrix is
wanted, e.g. `--density=3` for the W+ of `p p > w+ j`, or `--density=3,4` for
`p p > w+ z` or `p p > t t~`. **Every other leg is summed over helicity**,
exactly as in the unpolarised result. So `Tr rho` is always the usual
unpolarised quantity.

The output refuses a selected leg that is massless or in the initial state.
These are the same reasons as for polarised NLO (`check_process_format`,
`madgraph_interface.py:1707-1810`; `fks_common.carry_polarization`):

| selected leg | |
|---|---|
| massive colour singlet, final state (W, Z) | allowed. This is the first target. |
| massive coloured, final state (t, t~) | allowed. `j_fks` may be the selected leg; a massive leg has a soft singularity but no collinear one. |
| massless (gluon, light quark, photon, lepton) | refused. For a coloured one, the helicity transfer across `g -> q q~` is undefined. |
| initial state | refused. A coloured one becomes an internal line of the real through backward splitting. |

The `carry_polarization` map (Born leg -> real leg) translates `POS` from the
Born numbering to the real numbering in each nFKSprocess.

### D5 — Virtual: force colour flows, store both JAMP sets per helicity

- In `get_context`:
  `ComputeColorFlows = self.compute_color_flows or LoopInduced or UseDensity`.
  When `UseDensity` is set and there is a Born, use
  `ME_COMPUTATION_FROM_JAMP=.TRUE.` (the "force the use of jamp"), and keep
  `DIRECT_ME_COMPUTATION=.TRUE.` behind a validation switch so the existing
  JAMP double-check (`JAMP_DOUBLECHECK_THRES=1e-9`) runs in M3.
- Add `JAMPB_ALL(NBORNFLOWS,NBORNAMPSO,NCOMB)` next to `JAMPL_ALL`, filled in
  `STORE_JAMP`.
- `DO_COMPUTE_INTER_JAMP` takes `(JAMPL1, JAMPB1, JAMPL2, JAMPB2)` and returns
  the **complex** value

  ```
  V_h1h2 = sum_ij C_ij [ L_i(h1) B_j(h2)* + B_i(h1) L_j(h2)* ]
  ```

  for each of K = finite, 1/eps, 1/eps^2. The tree x tree slot (`ANS(0)`)
  becomes `B_i(h1) C B_j(h2)*`. The `DBLE` and the factor 2 are removed; they
  are recovered for h1=h2.
- **Helicity recycling must be off.** `HelicityFilterLevel=2` maps helicities
  equal up to a sign onto one evaluation (`HEL_MULT>1`). That is valid for
  `|M|^2` but not for off-diagonal elements, and the mapped helicities never
  get their JAMPs stored. Density outputs write `HelicityFilterLevel=1` into
  `MadLoopParams.dat` and refuse 2.
- **Stability rotations must not reach the stored JAMPs.** MadLoop's stability
  test re-evaluates rotated points (`ROTATE_PS`, `improve_ps.inc:384`). A
  rotation multiplies the helicity amplitudes by `exp(i h phi)`: the trace is
  invariant, the off-diagonal elements are not. `STORE_JAMP` must run only on
  the evaluation at the user's point: the DP reference evaluation, or its QP
  rescue at the same point.
- Fix the `GET_HEL_INDEX` prefix bug.

### D6 — Implementation restrictions

These do not limit the physics (D4 covers that). They are configurations the
density code does not support, so they raise an error at output time instead
of giving a wrong answer.

- **Old colour-link mode.** Density requires
  `born_color_links_in_born = True` (the default). The old one-file-per-link
  path (`b_sf_NNN.f`, `bornmatrix_splitorders_fks.inc`) is refused rather than
  ported. *Agreed.*
- **Monte-Carlo helicity sampling.** The non-selected legs must be summed
  exactly (`isum_hel=0`), not sampled one helicity per call. The standalone
  output already sums exactly. This only matters if the kernels are reused in
  aMC integration later.
- **What the plan does not deliver.** It delivers standalone checks (density
  per phase-space point plus limit tests). No event generation (madevent, aMC
  events or MadSpin) reads these densities yet, although the kernels are
  written so that it could later.

## 3. Physics of the checks

### 3.1 Born-level identities (exact, per element)

- `Tr rho_B = SBORN`; `rho_B` is hermitian and positive semi-definite.
- `Tr rho_B^{kl} = SB_SF_LINK(k,l)`.
- Colour conservation **per element**: `sum_{k != l} rho_B^{kl} = -C_l rho_B`
  for every (h1,h2). This is new: today it is only checked on the trace.
- `rho_B(i,i)` equals the polarised Born with the corresponding `{+}`/`{-}`
  restriction.

### 3.2 Soft limit

At leading power the eikonal factor is spin-independent, so per element

```
R_h1h2  ->  sum_kl  eik_kl(xi -> 0)  rho_B^{kl}_h1h2
```

including the massive self-eikonal `k = l = t`. This is IR safe only if the
helicity frames of the real and the Born coincide as `xi -> 0`; D3 guarantees
it.

### 3.3 Collinear limit

The density leg is never the splitting parton (D4), so

```
R_h1h2 -> P_ab(z) rho_B_h1h2 + Q_ab(z) [ a T_h1h2 + conj(a) T'_h1h2 ]
```

where:
- `a = azifact` (in the me_frame, `azifact_me_frame`);
- `T = sum J(+,h1) C J(-,h2)*` and `T' = sum J(-,h1) C J(+,h2)*`, with the
  sums over the other helicities and with `+`/`-` the helicity of the FKS
  mother.

Pitfall: for the density, **T' is not the complex conjugate of T**. It is the
conjugate of the *transposed* element. Both orientations must be stored.
Contracted with a hermitian W, though, `Tr(W T') = conj(Tr(W T))`, so the
existing `2 Re(a * BORNTILDE)` code is correct with `BORNTILDE := Tr(W T)`.
This is why D2 works for B~.

### 3.4 Virtual poles

Per element:

```
V^(-2)_h1h2 = -(as/2pi) sum_i C_i rho_B_h1h2
V^(-1)_h1h2 = (as/2pi) [ -sum_i gamma_i rho_B + sum_{k<l} log-terms rho_B^{kl} ]_h1h2
```

These are the `getpoles` formulas with `Tr(W .)`. `check_poles.f` (via
`binothlha_frame`) is then reused unchanged per hermitian basis element (D2).

There is no limit check of the finite part. It is covered by the trace
identity and by the diagonal-vs-polarised identity.

## 4. Milestones

| | content | depends on |
|---|---|---|
| **M0** | plumbing, all inert | — |
| **M1** | Born family: refactor + density | M0 |
| **M2** | Real: refactor + density + limit tests per element | M1 |
| **M3** | Virtual: tree x loop density in MadLoop + poles per element | M0 (parallel to M1/M2: different files) |
| **M4** | tests, docs, f2py | M1-M3 |

### M0 — plumbing

1. `amcatnlo_interface.check_output`: accept `--density=i,j,...` (legs in the
   user's order) and `--me_frame=...` with `--fks`. Apply the D4 rules and the
   `check_validity` frame rules there, at output time.
2. In density mode keep `V*/` and `MadLoop5_resources` (today always dropped),
   and pass `density` into the loop exporter's `cmd_options`.
   `ProcessOptimizedExporterFortranFKS` already inherits from
   `LoopProcessOptimizedExporterFortranSA`.
3. Leg maps:
   - user order -> Born P-dir order: reuse `user_leg_order` /
     `frame_map_born` (`export_fks.py:1805-1831`);
   - Born -> real: through `i_fks`, as in `get_frame_mask_real`;
   - Born -> virtual: verify that the `V*` dir uses the Born order. It is
     expected to, but this is not checked yet.
4. Write a `density_info.inc` per P dir (`N_CHANGING`, `N_COMB`, `POS`,
   `ALLOW_HEL` for the Born, and per nFKSprocess for the real), generated with
   the same code as `export_v4.py:6498` (`get_helicity_per_particle`). Add
   `/DENSITY_PROJ/ W` with a BLOCK DATA identity.
5. Ship `boost_to_frame.f` + `impose_frame_rest.f` + `frame_info.inc` in
   Born-only mode, and route `check_sa_fks.f` through the `*_frame` wrappers.

Gate: without `--density`, the output is identical apart from the new files,
and `test_cmd_fks_standalone` and `test_export_fks` IOTests pass.

### M1 — Born family

1. `bornmatrix_colorlinks_fks.inc`:
   - generalise `BORN_COLOR_CONTRACT` into `GET_INTER_BORN(J1,J2,ILINK,INTER)`,
     complex, with the packed `CF(:,ILINK)`;
   - add `GET_ALL_INTER_BORN` / `GET_DENSITY_BORN` returning
     `INTER(NPACK, 0:NLINKS)` from one amplitude pass;
   - add `GET_DENSITY_BORNTILDE` returning both T and T' (3.3) for the FKS
     mother;
   - make `BORN`/`SBORN` honour W.
2. `born_cnt_splitorders_fks.inc`: split `BORN_CNT%d` into
   `GET_AMP_BORN_CNT%d` / `GET_JAMP_BORN_CNT%d` / `GET_INTER_BORN_CNT%d` /
   `GET_DENSITY_BORN_CNT%d`, and add W.
3. Charge links: `sborn_sf` charge branch = `Tr(W rho_B) Q_m Q_n`. No new
   kernel.
4. `check_sa_fks.f`: with density on, print `rho_B`, every `rho_B^{kl}` and T,
   and write `Density_matrix.dat` as the LO `check_sa.f` does.
5. Gates:
   - the 3.1 identities to <= 1e-13 per element;
   - `W = I` A/B against the current `born.f` to <= 1e-14 on the 17
     subprocesses used for the colour-density work;
   - diagonal vs polarised output (`w+{0}`, `w+{T}`, then `w+{0} z{T}` ..., then `t{+} t~{-}`) to <= 1e-13; run in the M2 process order (`w+ j`, `w+ z`, then t t~).

### M2 — Real and the limit tests

1. `realmatrix_splitorders_fks.inc`: split `MATRIX_%(p)s` into
   `GET_AMP` / `GET_JAMP` / `GET_MATRIX` / `GET_INTER` / `GET_ALL_INTER` /
   `GET_DENSITY` (the LO layout, split-order variant as in
   `matrix_standalone_splitOrders_v4.inc:507-743`). `SMATRIX` honours W with
   the real-side `POS`.
2. `test_soft_col_limits.f` / `FKSSALauncher._run_limits`: add a loop over
   the `N_COMB^2` hermitian basis elements. Set W, run the existing soft and
   collinear tests, and report PASSED/FAILED per element. Expect `Re`/`Im`
   entries whose value is exactly zero; the test must handle the 0/0 ratio
   (use an absolute floor relative to `Tr rho`).
3. Positive set, **in this order**. Every element must PASS before moving to
   the next process.
   1. `p p > w+ j [QCD]`, `--density=3` (frame [3], N_COMB=3, 9 elements).
      The W is a colour singlet. Soft and collinear ISR/FSR, including
      `g -> g g` with the B~ azimuth, all act on legs that are not selected.
      This is the cleanest first test of D2/D3.
   2. `p p > w+ z [QCD]`, `--density=3,4` (frame [3,4], N_COMB=9,
      81 elements). This adds two selected legs and a pair frame. It is ISR
      only.
   3. Only then t t~: `u u~ > t t~ [QCD]` and `g g > t t~ [QCD]`,
      `--density=3,4`. A selected leg is now coloured, so the massive
      self-eikonal, soft emission from a selected leg and `j_fks` = selected
      leg all appear.
   4. Optional extras afterwards: `u u~ > w+ w- [QCD]`, `t > b w+ [QCD]`
      (density on the w+).
4. **Negative controls**, required to show that the per-element test is
   sensitive:
   - frame from an initial-state leg (bypassing the guard);
   - lab frame, `frame_id=0`.

   ISR should make off-diagonal elements FAIL while the trace still PASSES. If
   they do not fail, the test is too weak and M2 is not done.
5. A/B with `W = I` against the current real code: <= 1e-14, and an aMC
   `p p > t t~ j [QCD]` fixed-order run unchanged within the rounding control
   of `amcatnlo-ab-rounding-control`.

### M3 — Virtual

1. `loop_exporters.get_context`: force `ComputeColorFlows` /
   `AmplitudeReduction` when `UseDensity` (D5). Write `HelicityFilterLevel=1`
   into `MadLoopParams.dat` for density outputs, and refuse a user value of 2.
2. `compute_color_flows.inc`: add `JAMPB_ALL`, extend `STORE_JAMP` and
   `REINITIALIZE_JAMP_ALL`, make `DO_COMPUTE_INTER_JAMP` complex with four
   JAMP arguments (D5), and drop the `NOTE: ... not supported` block.
   `GET_ALL_INTER`/`GET_DENSITY` already exist; un-gate them from
   `LoopInduced` where needed. Fix `GET_HEL_INDEX`.
3. `loop_matrix_standalone.inc`: guard `STORE_JAMP` against the stability
   rotations (D5), and contract with W for `ANS` when W != I.
4. Validate first in plain MadLoop standalone, which has no FKS, in the same
   order as M2: `u d~ > w+ g [virt=QCD]` with `--density=3`, then
   `u d~ > w+ z [virt=QCD]` with `--density=3,4`, then
   `u u~ > t t~ [virt=QCD]` with `--density=3,4`. Today `[virt=...]
   --density` gives nothing: make it work, or make it fail loudly first.
   Gates:
   - trace = `SLOOPMATRIX` `ANS(1..3)`, using the JAMP double-check at 1e-9;
   - diagonal = polarised `[virt=QCD]` (`w+{0}`, `w+{T}`, ..., `t{L}`, ...);
   - hermiticity.
5. In the FKS output: `check_poles` per hermitian basis element via
   `binothlha_frame`. Use the `nlo-polarised-checkpoles-frame` lesson: group
   the diagnostics by phase-space point, and use the ratio identity
   `V^(-1)_h1h2 / I^(-1)[rho]_h1h2` rather than printouts.

### M4 — tests and API

- **Acceptance:** extend `tests/acceptance_tests/test_cmd_fks_standalone.py`
  with density cases, added in this order:
  1. `p p > w+ j [QCD] --density=3 --limits`
  2. `p p > w+ z [QCD] --density=3,4 --limits`
  3. then `p p > t t~ [QCD] --density=3,4 --limits`

  Each one checks the Born identities, every limit element and the virtual
  poles.
- **Unit tests:** guard rules (D4, D6); leg maps (user -> Born -> real ->
  virtual).
- **IOTests:** `born.f`, `matrix_N.f` and `born_cnt_N.f` goldens change with
  the refactor. **Ask before regenerating** them.
- **Optional f2py:** a `PY_GET_DENSITY` per block, reusing the LO
  `f2py_splitter` pattern.
- **Docs:** add `help_output` text, and a section in this file with the
  validation numbers.

## 5. Risks and open questions

1. **Refactor cost on the production path.** The real and Born kernels are
   also used by aMC event generation. The LO split cost nothing measurable,
   but the A/B gates in M1/M2 are mandatory, not optional.
2. **Off-diagonal phase conventions across codes.** The Born/real (HELAS
   templates) and MadLoop must use the same external-wavefunction phases for
   T, B_kl and V to be comparable element by element. Both use the ALOHA
   `ixxxxx`/`oxxxxx`/`vxxxxx`, but MadLoop applies its own `ImprovePS` to
   the point. Check this explicitly in M3 with the Born slot: `ANS(0)` density
   from MadLoop vs `rho_B` from `born.f`, per element.
3. **Zero-valued elements.** Many `Im rho` entries vanish identically, e.g.
   from CP. The limit-test pass criterion needs an absolute tolerance relative
   to `Tr rho`, not a ratio.
4. **"Reduced Born"** (agreed) means the extra-counterterm Borns
   (`born_cnt_N.f`) plus the Born at mapped (n-body) kinematics.
5. **Cost of the per-element limit test.** It runs `N_COMB^2` times.
   `w+ j` gives 3 x 3 = 9 runs, `t t~` 4 x 4 = 16, and `w+ z` 9 x 9 = 81. That is
   acceptable standalone, but a single run that applies all W at once
   (vector W) is a later optimisation.

## 6. Implementation log

### M3 items 1-4: the tree x loop virtual (MadLoop)

`generate u d~ > w+ g [virt=QCD]; output standalone_fortran DIR --density=3`
now writes the density matrix. The changes:

- **`loop_exporters.get_context`.** `--density` forces `ComputeColorFlows`.
  Before, the density routines were silently missing for a process with a
  Born.
- **`compute_color_flows.inc`:**
  - `JAMPB_ALL`, the Born colour flows of every helicity, is stored next to
    `JAMPL_ALL`.
  - `DO_COMPUTE_INTER_JAMP` takes both helicities' Born flows and returns the
    complex `L(1)B(2)* + B(1)L(2)*` and `B(1)B(2)*`.
  - New `GET_DENSITY_ALL` returns `INTER(NPACK, 0:3)`: Born, finite part,
    1/eps and 1/eps^2. `GET_DENSITY` keeps its signature: the finite part with
    a Born, the old sum for loop-induced.
  - `GET_ALL_INTER` no longer re-reads `HelFilter.dat`. A filtered helicity
    has zero colour flows, because they are reset before each reference
    evaluation.
  - `GET_HEL_INDEX` hard-coded `ML5_0_`; fixed.
  - The loop-induced `SMATRIXHEL` is now written for loop-induced processes
    only. Before, it clashed with the Born's in `born_matrix.f`, so even
    `set loop_color_flows True` failed to link for a process with a Born.
- **`loop_matrix_standalone.inc` and `mp_compute_loop_coefs.inc`.** The
  colour flows are stored only during the reference evaluation
  (`STORE_DENSITY_JAMPS = .NOT.EVAL_DONE(2)`), in DP or in the QP rescue. They
  are never stored in the rotated or other-CTMode evaluations of the stability
  test. With density, the matrix element is computed from the colour flows
  (`ME_COMPUTATION_FROM_JAMP`).
- **`HelicityFilterLevel`.** It is set to 1 in `MadLoopParams.dat`, and
  `GET_DENSITY_ALL` stops on 2.
- **`born_matrix.f`.** In density mode the Born's tree-level density routines
  are renamed `<prefix>BORN_GET_DENSITY` / `BORN_GET_ALL_INTER` (the new
  `density_prefix` of `write_matrix_element_v4`).
- **`check_sa.f`.** Prints the trace of each slot next to `ANS(0:3)` and
  writes `density.dat`.
- **Non-optimized loop output.** `--density` is refused there, since it has no
  colour-flow code.

Validation (all `[virt=QCD]`, RAMBO point of `check_sa`):

| check | result |
|---|---|
| trace vs `ANS(0:3)`, `u d~ > w+ g`, `u d~ > w+ z`, `u u~ > t t~`, `g g > t t~` | <= 3e-16 |
| trace vs the same process without `--density` (direct computation) | <= 2e-14 (except `u u~ > t t~`, below) |
| diagonal vs polarised outputs: `w+{-,0,+} g`, `w+{0} z{+}`, `u u~ > t{+} t~{-}`, `g g > t{-} t~{-}` | <= 4e-14 (2e-12 for `w+ z`) |
| 1/eps^2 and 1/eps densities of `u d~ > w+ g` | proportional to the Born density element by element (3 coloured partons) |
| `NRotations_DP=2` vs no stability test | bit-identical |
| guard removed (store at every evaluation), negative control | diagonal unchanged, off-diagonal wrong by O(1): the rotation phase |
| `CTModeRun=4` (QP) vs DP | ~1e-10 (DP reduction accuracy) |

**Pre-existing issue found, not caused by this work.** For
`u u~ > t t~ [virt=QCD]`, MadLoop's amplitude-level (colour-flow) reduction
and its default squared-amplitude path disagree. The finite part is 0.538
vs 0.678. The single pole differs by exactly 21/4 x Born x as/2pi at every
point; the Born and the double pole agree. The facts:

- The colour-flow value equals the sum of the four polarised default outputs
  (0.5383267901237969). So the odd one out is the unpolarised default
  (grouped-loop) output.
- It reproduces on the untouched commit with `set loop_color_flows True`.
- `u u~ > d d~` and `g g > t t~` agree between the two paths.
- It is unchanged with `HelicityFilterLevel=1` or the loop filter off.

It needs its own investigation before the t t~ step of M2/M4.

### M0: plumbing

- **Options.** `output standalone_fortran --fks --density=i,j [--me_frame=k,l]`.
  Both are parsed in `amcatnlo_interface.check_output` (syntax only) and
  checked per P directory in `ProcessExporterFortranFKS_SA`:
  - each density leg must be a massive final-state particle;
  - the frame must not use part of the initial state.
- **Leg numbering.** User legs map to Born positions through
  `user_leg_order`. The virtual is checked to order its legs like the Born.
  It is handed `cmd_options['density']` in Born positions, so the MadLoop
  virtual of each P directory computes its density matrix (M3 above).
- **Generated files.**
  - `density_info.inc` (density outputs only): `POS_B`, `ALLOW_HEL_B`,
    `N_CHANGING_B`, `N_COMB_B`.
  - `me_frame_sa.inc` (every FKS standalone output): `frame_id` for the Born
    driver, 0 without `--density`/`--me_frame`. In `--limits` mode `me_frame`
    is also written into the run_card (as a user-set value, else it is not
    written).
- **Frame code split.** `boost_to_frame.f` was split:
  `get_frame_mask_born/real`, `get_me_frame_boost` and `boost_to_me_frame`
  moved to the new `me_frame_boost.f`. They call no matrix element, so the
  trimmed `check_fks` links them. `check_sa_fks.f` now boosts to the frame
  before `SBORN`/`SBORN_SF`, as `sborn_frame` does.
- **Kept virtuals.** Density outputs keep `V*/` and `MadLoop5_resources`, and
  set `HelicityFilterLevel=1`.

Validation:

- **No density, `p p > w+ j [QCD]`.** `check_fks` is bit-identical to the
  output before the change in every P directory.
- **`--density=3` (W rest frame).** B and B_ij are unchanged to 2e-14
  (Lorentz invariance); BORNTILDE moves, as its azimuthal reference is frame
  dependent.
- **`--limits --density=3`.** `test_soft_col_limits` passes in all 6 P
  directories with `FRAME_ID=8`, at trace level; per element is M2.
- **Virtual build.** The kept virtual compiles with its density routines.
- **Refusals.** Massless, initial-state, nonexistent, repeated and mixed-frame
  legs are each refused with a message.
- **Test suites.** The unit tests `tests/unit_tests/loop` (75) and iolibs
  `test_export_fks`/`test_export_v4`/`test_density_library` (94) pass. The
  IOTests `gg_hh` (loop-induced), `gg_wmtbx`, `loop_induced_standalone_output`,
  `TIR_output`, `Loop_sqso_uux_ddx`, `test_ppw_fksall` and the v4 standalone
  are unchanged, so no golden needs regenerating.
- **Not done in M0.** The projector W (D2) is left for M1, where it gets its
  first user.

### M1: Born family

All of the code below is generated only with `--density`. That is done
through a `density` context in `write_split_me_fks`. Without `--density`,
`born.f` is byte-identical to before.

- **Kernels (`bornmatrix_colorlinks_fks.inc`):**
  - `BORN_DENS_TABLES` groups the helicity configurations by the helicities
    of the non-density legs (the "rest").
  - `BORN_DENS_JAMPS` computes the JAMPs of every helicity. It uses
    `GET_AMP_BORN`/`GET_JAMP_BORN`, and shares `saveamp` when called from
    `SBORN`.
  - The existing `BORN_COLOR_CONTRACT` is already the sesquilinear form
    `sum S_ij A_i conj(B_j)`, so it serves as GET_INTER.
- **Output:**
  - `GET_DENSITY_BORN(P, RHOD(NPACK,0:NLINKS))`: the Born and every
    colour-linked Born.
  - `GET_DENSITY_BORNTILDE(P, T, TP)`: the two transition densities of the
    spin-correlated Born.
- **Projector W:** `/C_DENSITY_PROJ/ DENS_W, DENS_W_ON` lives in
  `density_info.inc`, and is off by default (BLOCK DATA). When it is on,
  `SBORN` calls `BORN_RHO_W`. That fills the colour density
  `RHO(ij) = Re sum W(C2,C1) J_i(C1) J_j(C2)*` and the projected BORNTILDE.
  So `SBORN`, `SB_SF_LINK` and `ANS_CNT(2)` return Tr(W rho) with no change
  to `sborn_sf.f` or `fks_singular.f`. The Born cache is bypassed while W
  is on.
- **Check driver.** `check_sa_fks_density.f` is called from `check_sa_fks.f`
  through `density_sa_call.inc`, which is empty without `--density`. It
  prints and writes `density_born.dat`, and checks the traces and the
  projector.

Validation on `p p > w+ j [QCD] --density=3` (W rest frame), in P
directories `udx_wpg` and `gu_wpd`:

| check | result |
|---|---|
| Tr rho_B vs SBORN; Tr rho^l vs SB_SF_LINK; Tr T vs BORNTILDE | <= 2e-15 |
| projector W on every hermitian basis element: SBORN, SB_SF_LINK, BORNTILDE vs Tr(W rho) | <= 2.4e-16 |
| rho^l / rho_B per element (3 coloured partons) | constant to 6e-16: exactly 1/6 (q q~ link) and -3/2 (q g link) |
| diagonal of rho_B vs polarised `w+{-}`, `w+{0}`, `w+{+}` outputs | identical to the last printed digit |
| `born.f` without `--density` | byte-identical in all 6 P directories |

**Extra-counterterm Born (`born_cnt_splitorders_fks.inc`), production
template.**

- **Refactor.** `BORN_CNT` now calls `GET_AMP_BORN_CNT`,
  `GET_JAMP_BORN_CNT`, `GET_MATRIX_BORN_CNT` (colour sum of one helicity)
  and `GET_INTER_BORN_CNT` (spin-correlated contraction), with the same
  loop order as before.
- **Bug fixed along the way.** The old code divided all of `BORNS(:,:)` by
  DENOM inside the loop over the two helicities of the FKS mother, so the
  first helicity was divided by DENOM twice. This is harmless when DENOM=1,
  as for the `a g > t t~ [real=QCD QED]` extra counterterm (whose limit test
  passes, 72 checks). It is wrong otherwise.
- **Projector W.** It is not implemented for these extra counterterms, which
  only exist in mixed QCD+QED processes; `SBORN_CNT` stops if W is on.

### M2: real emission and per-element limit tests

- **Real template refactor (`realmatrix_splitorders_fks.inc`), production.**
  `MATRIX_N` is now `GET_AMP_N` -> diagram weights -> `GET_JAMP_N` ->
  `GET_MATRIX_N`, with the same loop order. The unprojected limit tests of
  `p p > w+ j` give a `test_ME.log` byte-identical to the one from the
  templates before M2.
- **Real density code (with `--density` only):**
  - `GET_INTER_N` is the colour contraction of two helicities.
  - `SMATRIX_N_W` returns Tr(W rho) of the real emission. The density legs
    are at `POS_B`, shifted by one when at or after `i_fks` of the current
    FKS configuration (the mapping of `get_frame_mask_real`).
  - The projected branch runs before `NTRY` is incremented, so the helicity
    filter is still set up by the first unprojected call. The Born's
    projected branch now does the same.
- **Per-element limit test.** `DENSITY_W_INIT` (in `born.f`, density only)
  reads the projector once from `density_w.dat`, if that file exists. With
  `--density`, `FKSSALauncher._run_limits` first runs `test_soft_col_limits`
  unprojected, as before. It then runs it once per element of the hermitian
  basis:
  - `rho_II` for E_II;
  - `Re_rho_IJ` for E_IJ + E_JI;
  - `Im_rho_IJ` for i(E_IJ - E_JI).

  Each run writes `test_ME_<element>.log`.

Results, `p p > w+ j [QCD] --density=3`, frame = W rest frame
(`me_frame=[3]`): **all 9 elements pass in all 6 P directories** (32-40
checks each). The projected logs are confirmed to have read the projector,
and their numbers differ from the unprojected ones.

**Negative control.** The same output with `me_frame=[1,2]` gives
`FRAME_ID=6`, i.e. no boost: each configuration stays in its own partonic
c.m.

- The unprojected trace, the three diagonal elements and the three
  `Re_rho_IJ` still pass.
- **All three `Im_rho_IJ` fail in every P directory** (12-26 failed checks
  out of 32-40).

So the per-element test does detect a helicity frame that is not IR safe,
which the trace (and here even the diagonal) does not.

**Elements that vanish in the Born.** For `p p > w+ z` the Born is planar
(2 -> 2), so in the WZ rest frame every `Im_rho_IJ` of the Born is zero; the
check_fks point gives ~1e-21. The counterterm of such an element is then
rounding noise, and `checkres2`, which follows |real/counterterm - 1| along
the approach to the limit, cannot converge: all 36 `Im` elements failed.

The launcher now handles these elements differently:
- **When it applies:** the element of the Born density read from
  `density_born.dat` (the check_fks point) satisfies
  |Tr(W rho_B)| < 1e-12 Tr rho_B.
- **What is run instead:** W + 1, reported as `<element>+trace`. This checks
  that the element of the real emission vanishes in the limit, at the scale
  of the trace.
- **Every other element** keeps the strict test.

Results with `--limits`, per-element (frame = the density legs):

| process | P dirs | elements | runs | failed | `+trace` |
|---|---|---|---|---|---|
| `p p > w+ j [QCD]`, `--density=3` | 6 | 9 | 60 | 0 | 0 |
| `p p > w+ z [QCD]`, `--density=3,4` | 2 | 81 | 164 | 0 | 72 (all Im) |
| `p p > t t~ [QCD]`, `--density=3,4` | 3 | 16 | 51 | 0 | 3 |

The t t~ case covers soft emission from the selected (coloured, massive)
legs; its Born identities hold to <= 7e-16.

**Negative controls (`me_frame=[1,2]`, no boost):**
- **w+ j:** every `Im_rho_IJ` fails in all P directories, while the trace,
  the diagonal and the `Re` elements still pass.
- **w+ z:** everything passes. This is expected: there is initial-state
  radiation only, and the FKS initial-state mapping keeps the W and Z
  momenta of the Born in the collinear limit, so any frame is IR safe there.
  The frame sensitivity of the test is therefore shown by w+ j.

**IOTest goldens.**
- The real-emission refactor changes `matrix_N.f` in the FKS IOTests, 67
  files in total:

  | IOTest | files |
  |---|---|
  | `test_pptt_fksrealew` | 43 |
  | `test_pptt_fksreal` | 11 |
  | `test_ppw_fksall` | 6 |
  | `test_wprod_fksew` | 6 |
  | `test_tdecay_fksreal` | 1 |

  It is the same code reorganised; the numbers are bit-identical, as checked
  above. **The goldens are not regenerated yet (pending approval).**
- Beware: `test_manager -R` prints "successfully tested" even when files
  differ. Only `grep 'differs from the reference'` is reliable.
- The loop IOTests are unchanged, after the removal of two unconditional
  comment lines from `loop_matrix_standalone.inc`.
- `born.f`, `born_cnt` (no goldens) and the EW-Sudakov IOTests are unchanged.

### M3 item 5: virtual poles per element (FKS output)

- **MadLoop (`compute_color_flows.inc`, density only).** The new
  `DENSITY_PROJECT_ANS` runs at the end of `SLOOPMATRIX`, after the final
  squared-order sum. If `density_w.dat` exists in the working directory
  (read once), it replaces `ANS(0:3,0:NSQUAREDSO)` by Re Tr(W rho):
  - rho comes from the colour flows stored at the reference evaluation
    (`COMPUTE_INTER_JAMP` for each pair of combinations);
  - the projection covers the Born slot and the finite, 1/eps and 1/eps^2
    slots.

  The density legs and their combinations reach the colour-flow file
  through the new `get_density_replace_dict`, which `check_sa` now uses
  too. A single-helicity call with W on stops with an error.
- **Effect on the FKS chain.** `check_poles`, `BinothLHA` and
  `fks_singular.f` are unchanged:
  - MadFKS's predicted poles use the projected Born and colour-linked Borns
    (born.f reads the same file);
  - MadLoop returns the projected virtual.
- **`FKSSALauncher`.** For a `--limits --density` output (which keeps the
  virtual) it builds `check_poles`, with `madloop=true` on the make line, as
  the aMC@NLO launch puts it in make_opts. It then runs 20 points for the
  helicity sum, then once per element (the same `+trace` rule). A run fails
  when more than 10% of its points fail, as for aMC@NLO.

`p p > w+ j [QCD] --density=3`: **poles cancel for the sum and for all 9
elements in all 6 P directories, 20/20 points each.** In `P0_udx_wpg` the
single-pole coefficient (MadFKS vs MadLoop) is:

| projector | MadFKS | MadLoop |
|---|---|---|
| sum | -6.96626499239e-2 | -6.96626499287e-2 |
| rho_11 | -4.01206545365e-2 | -4.01206545365e-2 |
| Im rho_12 | 4.51173808685e-2 | 4.51173808685e-2 |

All three target processes, `--limits --density`, regenerated with the final
templates (frame = the density legs):

| process | P dirs | soft/collinear per element | poles per element |
|---|---|---|---|
| `p p > w+ j [QCD]`, `--density=3` | 6 | 60/60 | 60/60 |
| `p p > w+ z [QCD]`, `--density=3,4` | 2 | 164/164 (72 `+trace`) | 164/164 |
| `p p > t t~ [QCD]`, `--density=3,4` | 3 | 51/51 (3 `+trace`) | 51/51 |

### MC counterterms (`test_MC`) per element

1. **A `test_MC` that can fail.** Commit `dd82ab7e9` (branch
   `claude/test-mc-real-check`, held upstream because a real `test_MC`
   aborts some NLO EW runs) is applied here. Without it, every MC check
   compared the MC counterterm with itself and passed by construction.
2. **The colour-flow weights of the projected Born.** `get_mbar` splits the
   Born over leading colour flows with `JAMP2`:
   `bornbars(i) = jamp2(i)/sum * Born`. The projected `SBORN` branch did
   not fill `JAMP2`, so the MC counterterm would have been zero.
   `BORN_DENS_JAMPS` now sums `AMP2`/`JAMP2` over the helicities exactly as
   `BORN` does. The colour-flow fractions stay those of the helicity sum, so
   `bornbars(i) = fraction_i * Tr(W rho_B)` is linear in W and sums to the
   projected Born.
3. **The Born rotation in `get_mbar`** (`montecarlocounter.f`, production
   code). For `j_fks=2` initial-state radiation, `get_mbar` rotates the whole
   Born by R_y(pi). Before this fix it skipped the rotation only when the
   frame boost was non-trivial (`me_boosted`). A requested frame that
   coincides with the Born c.m. (WZ for `w+ z`, t t~ for `t t~`) is not
   boosted, so the Born was still rotated. That rotates the helicity states
   of the density legs and changes the off-diagonal elements:
   - `w+ z`: 40 off-diagonal MC runs failed;
   - `t t~`: 20 failed;
   - all failures were collinear tests of the `j_fks=2` FKS configurations;
     the sum and the diagonal passed.

   The rotation is now skipped whenever a frame is requested
   (`frame_id.ne.0`). The ME counterterm (`sborncol_isr`) never rotates the
   Born. Runs without a frame (`frame_id=0`) are unchanged. The same change
   is made in `montecarlocounter_alt.f` (not compiled).
4. **Launcher.** `test_MC` runs with the aMC@NLO input for the shower of the
   run_card: once for the helicity sum, then per element.

Final results, all `--limits --density`, regenerated:

| process | ME limits per element | MC limits (sum + elements) | poles (sum + elements) |
|---|---|---|---|
| `p p > w+ j [QCD]` | all pass | 60/60 | 60/60 |
| `p p > w+ z [QCD]` | all pass | 164/164 | 164/164 |
| `p p > t t~ [QCD]` | all pass | 51/51 | 51/51 |

The failing run before fix 3 doubles as the sensitivity check of the
per-element MC test. It caught a phase error in the off-diagonal elements
that the sum and the diagonal elements do not see.

