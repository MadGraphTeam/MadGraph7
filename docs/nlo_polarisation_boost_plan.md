# Polarised cross-sections at NLO: frame-boost implementation plan

Status: **not started.** This document records the assessment of the existing
(dead) polarised-NLO code and the plan to make `p p > z{0} z{0} j [QCD]` work.

## 1. Assessment of the existing implementation

### The guard

`madgraph/interface/madgraph_interface.py:1240`

```python
if '[' in process and '{' in process:
    if 'noborn' in process or 'sqrvirt' in process: valid = True
    else: raise InvalidCmd('Polarization restriction can not be used for NLO processes')
    # below are the check when [QCD] will be valid for computation
    # if order.strip().lower() != 'qcd':
    #     raise InvalidCmd('...generic NLO computations')
```

The commented-out block at `:1251` is the only authored trace of an intended
`[QCD]` path. The surviving `check(p)` at `:1254` rejects colour-charged **and
massive** particles, so even `p p > z{0} z{0} [noborn=QCD]` is refused today.
The live NLO polarisation path is massless-colourless loop-induced only, which
is exported through the **LO** madevent template and therefore inherits
`me_frame` for free. That is why an NLO boost was never needed.

### Generation is alive and correct

`FKSLeg` inherits `Leg`, so `polarization` survives; `fks_common.py:747`
(`to_leg`) copies it explicitly. With the guard bypassed in-memory,
`p p > z{0} z{0} j [QCD]` generates cleanly — 8 born processes, with `{0}` on
the Z legs of every born **and** every real:

```
born:  d~ g > z{0} z{0} d~        legs: [(-1,[]), (21,[]), (23,[0]), (23,[0]), (-1,[])]
real:  d~ g > z{0} z{0} d~ g      legs: [(-1,[]), (21,[]), (23,[0]), (23,[0]), (-1,[]), (21,[])]
```

The front end is not the blocker. Everything downstream is.

### The frame machinery is LO-only

| piece | location |
|---|---|
| `boost_to_frame` (pure `boostx`, no rotation) | `Template/LO/SubProcesses/genps.f:1761` |
| `common /to_frame_me/frame_id` | `Template/LO/Source/run.inc:27` |
| call sites | `madgraph/iolibs/template_files/auto_dsig_v4.inc:183`, `:452`, `:488` |
| `mapid` (`ids(i)=btest(id,i)`) | `Template/LO/SubProcesses/cluster.f:128` |
| near-duplicate | `MadSpin/src/driver.f:1924` |

`Template/NLO/` has **zero** occurrences of `frame_id`, `me_frame` or
`boost_to_frame`.

### The six checks

| | status |
|---|---|
| Born boost | **No.** `call sborn(p_born,…)` x20 in `fks_singular.f`, always raw partonic-CM momenta |
| Virtual boost | **No.** `Call BinothLHA(p_born,…)` — `fks_singular.f:7087` |
| Real boost | **No.** `call smatrix_real(pp,wgt)` — `fks_singular.f:4701` |
| Counterterm boost | **No**, for any of them — soft `sbornsoft`, collinear `sborncol_isr`/`sborncol_fsr`, soft-collinear, degenerate `sreal_deg`, `bornsoftvirtual`, `extra_cnt`. Each has its own reduced kinematics (`p1_cnt(0,1,0/1/2)`, `p_born`, `p_born_used`) and so needs its own frame |
| Boost before eikonal/splitting kernel | **No — and this is the hard part** (see M2 step 4) |
| run_card option | **No.** `me_frame`/`frame_id` are added in `RunCardLO` (`banner.py:4438-4440`), resolved at `:4869`, and the `frame` display block is appended only from the LO process-dependent setup (`:5191`). `RunCardNLO` (`:5758`) has none of it |

### On rotations

A rotation is **not** needed for the squared MEs. HELAS builds polarisation
vectors in each particle's own momentum-direction helicity basis, so a rotation
`R` gives `eps_lambda(Rp) = R eps_lambda(p) * exp(-i*lambda*dphi)` — a pure
per-particle phase that cancels in `|M|^2` when every external helicity is
fixed. Born, real and virtual are all rotation-invariant.

Where orientation *does* matter is the collinear counterterm azimuthal phases
(`getaziangles`, and the ISR `cphi_mother=1` assumption). That is where the
"the two Z should be on the z axis" concern actually lands.

## 2. Design decisions

### D1 — Where to apply the boost

Options: inside the generated ME wrappers (`SBORN`, `SMATRIX_REAL`, …), or
caller-side in `fks_singular.f`.

**Recommend caller-side.** `SBORN` caches amplitudes against
`savemom(j,1)=p1(0,j), savemom(j,2)=p1(3,j)`
(`madgraph/iolibs/template_files/born_fks.inc:85-97`) and that cache is shared
with `sborn_sf`, `born_hel` and `extra_cnt` via `calculatedBorn`/`saveamp`. If
one entry point boosts and another does not, the cache silently returns
amplitudes from the wrong frame — a bug with no symptom. Boosting once per
kinematic configuration in the caller makes coherence structural, and gives the
azimuthal code access to the same boost vector.

Concretely: one helper `boost_to_me_frame(p, npart, pboost_out, p_out)` in the
NLO template, plus a common block holding the boost vector for the current
configuration.

### D2 — How `me_frame` indexes legs

At LO, run_card positions == fortran positions, so `mapid` is enough. At NLO
this breaks twice:

- `FKSLegList.sort()` (`fks_common.py:793`) reorders born legs relative to the
  user's process, so user leg *n* != born position *n*.
- Born has `nexternal-1` legs, each real has `nexternal`, with the map given by
  `set_pdg` (`Template/NLO/SubProcesses/chooser_functions.f:454-540`): real *k*
  -> born *k* for `k < i_fks`, born *k-1* for `k > i_fks`, and `i_fks` has no
  born counterpart.

**Decided:** keep `me_frame` in **user process numbering**.

Note the premise above was worse than stated: `sort_proc` (`fks_common.py:731`)
*renumbers* the legs (`leg['number'] = n + 1`) after permuting them, so the
user's numbering is not merely reordered, it is destroyed. Verified: the user
writes `p p > j z{0}` and the born comes out `g d > z{0} d` with the Z carrying
`number=3`, not 4. It therefore cannot be recovered downstream and has to be
captured before `sort_proc` runs.

**Implemented in M0** (see the M0 section for status):

- `fks_common.get_user_leg_order()` computes the FKS ordering on a copy and
  returns the pre-sort numbers; `FKSProcess.__init__` calls it *before*
  `sort_proc` and stores `self.user_leg_order`, which `FKSHelasProcess` carries
  to the exporter.
- `frame_info.inc` per P dir holds only `frame_map_born(nexternal-1)`.
- **No `frame_map_real` table is needed.** The FKS convention already fixes the
  real -> underlying-Born correspondence in terms of `i_fks` alone (`set_pdg`,
  `chooser_functions.f:454-540`): real *r* -> Born *r* for `r < i_fks`, Born
  *r-1* for `r > i_fks`, and `i_fks` itself has no Born counterpart. So
  `get_frame_mask_real` derives the real mask at runtime from the Born one.
  This drops the per-FKS-config table the plan originally called for.

### D3 — Smoothness and equivariance

The frame must be a continuous function of the momenta converging in the
singular limits, or the subtraction stops cancelling. "Rest frame of the sum of
the selected legs, recomputed from *each* configuration's own momenta"
satisfies this: as `xi -> 0` / `y -> 1` the Born's ZZ sum -> the real's ZZ sum.

**Do not** compute the frame once from the real and reuse it for the Born — it
looks simpler and is wrong away from the limit.

**The subtler requirement (equivariance).** A pure boost to the rest frame of a
system is *not* equivariant under longitudinal boosts: `B(L_z P) . L_z != B(P)`
whenever `P` has transverse momentum, the difference being a finite rotation.
So if the reduced Born lived in its own partonic CM rather than in the same
"tilde" frame as the real event, the polarisation axes of the two would differ
by an `O(xi)` rotation, the `1/(1-y)` pole would not cancel, and no azimuthal
fix could repair it.

FKS bookkeeping gives us this for free in the ISR case, and it has been
checked: at `y=1` the ISR mapping applies to the final-state spectators only a
transverse boost whose rapidity vanishes identically
(`genps_fks.f:3055-3068`), and `xp` was initialised from the Born momenta
(`genps_fks.f:1206-1222`). In the tilde frame the real event's spectators are
therefore bit-identical to the Born's -> same sum of selected momenta -> same
`me_frame` boost for real and counter-event -> no residual Wigner rotation of
the Z polarisation axes.

**Action:** assert this at runtime (or at minimum comment it) — check that the
boost 4-vector derived from `p_born_used` and from the real `p` agree at the
counter-event. It is cheap and it protects a property the whole subtraction
rests on.

**Not yet confirmed for FSR:** `generate_momenta_massless_final` applies a
non-transverse `boostwdir2(chybst,…)` to spectators
(`genps_fks.f:2324-2339`). Confirm `shybst -> 0` in the collinear limit before
assuming FSR inherits the same property.

## 3. Milestones

### M0 — Plumbing, no physics — **DONE**

Delivered:

| what | where |
|---|---|
| `me_frame` + `frame_id` in the NLO run card | `banner.py` `RunCardNLO.default_setup`, `frame_id` resolved in `update_system_parameter_for_include` |
| `frame_block` exposed only for polarised massive legs | `RunCardNLO.blocks` + `create_default_for_process` |
| `$frame` placeholder | `Template/NLO/Cards/run_card.dat`, after the beam energies (mirrors the LO card) |
| `common/to_frame_me/frame_id` | `Template/NLO/Source/run.inc` |
| `mapid_frame`, `get_frame_mask_born`, `get_frame_mask_real`, `get_me_frame_boost`, `boost_to_me_frame` | `Template/NLO/SubProcesses/boost_to_frame.f` (new) |
| `boost_to_frame.o` in every executable | `makefile_fks_dir` `FILES` |
| `frame_info.inc` per P dir | `export_fks.write_frame_info_file` |
| user leg order captured pre-sort | `fks_common.get_user_leg_order`, `FKSProcess`, `FKSHelasProcess` |

Gate met. The change is **runtime-inert by construction**, which is a stronger
statement than a bit-identical rerun: `frame_id` is assigned in `run_card.inc`
but read nowhere, and nothing calls the boost routines yet. Verified by grep
over a generated process dir. `p p > z j [QCD]` builds, links and runs clean.

`get_me_frame_boost` returns `trivial=.true.` both when nothing is selected and
when the selected system already has exactly zero 3-momentum, so the default
`me_frame=[1,2]` costs nothing and cannot perturb existing results even once
call sites are added in M1.

Unit-tested numerically (rest frame reached, invariant mass preserved,
4-momentum conservation preserved, in-place aliasing safe).

Also fixed on the way: adding a block to `RunCardNLO` without a `$frame`
placeholder in the card template made `banner.py:3178` append the block at the
end of the file, which broke the `test_default` round-trip in
`tests/unit_tests/various/test_banner.py`. The placeholder is the fix.

Original task list, for reference:

1. `me_frame` + `frame_id` into `RunCardNLO` (`banner.py:5758`), mirroring
   `banner.py:4438-4440`; replicate the `frame_id = sum(2**n)` resolution from
   `:4869` and the `frame` display-block trigger from `:5191` into the NLO
   process-dependent setup.
2. `integer frame_id / common/to_frame_me/frame_id` into
   `Template/NLO/Source/run.inc`.
3. `boost_to_me_frame` + `mapid` into the NLO SubProcesses (port of
   `genps.f:1761`, pure `boostx`, keep the trivial-boost short circuit).
4. Exporter writes `frame_info.inc` per P dir (D2). Add alongside the other
   per-dir includes at `export_fks.py:502`.

**Gate:** `frame_id=6` (default) reproduces current NLO results bit-for-bit on
an unpolarised `p p > z j [QCD]`. Nothing else changes yet.

### M1 — `[LOonly=QCD]`: Born boost only

`[LOonly=QCD]` generates a fake FKS config with no reals and no virtuals
(`export_fks.py:3676`), so this milestone touches exactly one ME call and zero
counterterms. Cheapest possible validation of D1 + D2 + the run_card.

- Boost `p_born` before `call sborn` at the LOonly-reachable sites.
- Relax the guard at `madgraph_interface.py:1240` for `LOonly` only, and drop
  the massive-particle rejection at `:1257` on that branch (today it blocks
  `z{0}` even in the allowed `noborn` mode).

**Gate — acceptance test 1 (`p p > z{0} j`):** `[LOonly=QCD]`
`calculate_xsect LO` vs. LO madevent `generate p p > z{0} j`, same `me_frame`,
same cuts/PDF/scales, `group_subprocesses False`. Must agree inside MC error.
This validates the NLO Born boost against an independently-validated
implementation. Pin `nhel` identically on both sides.

`p p > z{0} z{0}` is a deliberate **null** test here: at Born level the ZZ
system *is* the partonic CM, so the boost must be a no-op and the polarised
LOonly result must be `me_frame`-independent.

### M2 — `[real=QCD]`: reals + counterterms (the hard milestone)

`[real=QCD]` gives the full FKS subtraction without virtuals — everything hard,
nothing MadLoop.

1. Boost the real momenta before `call smatrix_real(pp,wgt)`
   (`fks_singular.f:4701`).
2. Boost each reduced configuration with its *own* frame: `p1_cnt(0,1,0/1/2)`,
   `p_born`, `p_born_used`, and the `extra_cnt` argument. Sites:
   `fks_singular.f:494` (`bornsoftvirtual`), `:705`/`:793`/`:904` (the
   `sreal`/`sreal_deg` counterterms), plus the ~20 `call sborn` sites.
3. Soft is easy: the eikonal `p_m.p_n/(p_m.k)(p_n.k)` is invariant and
   `sborn_sf` is colour-linked only (no spin correlation), so `sbornsoft` needs
   nothing beyond a boosted `p_born`.
4. **The azimuthal phases — the real work.** `SBORN` returns
   `ANS(2) = BORNTILDE`, the +/- gluon-helicity interference built from
   `JAMPH(1,.)`/`JAMPH(2,.)` (`born_fks.inc:277`). The correction multiplying
   it collapses to `-exp(-2*i*psi)`, with `psi` the emission azimuth about the
   mother in the HELAS `(e1,e2)` basis.

   **No extra Wigner phase is needed on `BORNTILDE`.** The mother's
   little-group phase `exp(-+2*i*theta_W)` cancels against the `exp(+2*i*theta_W)`
   picked up by a *boosted* `<ij>/[ij]` — same `theta_W`, since i, j and the
   mother are collinear. The controlling identity, exact in any single frame
   and for any mother direction, is `arg(azifact)/2 = phi_m + psi`. There is no
   `theta_m` dependence (`R_y(theta)` has a real SU(2) matrix), which is
   exactly why `getaziangles` returning only `phi` suffices. The generator's own
   analytic limits confirm it independently: `genps_fks.f:2344` gives
   `-exp(2i(phi_mother+phi_i))` (FSR), `:3099` gives `-exp(2*idir*i*phi_i)` (ISR).

   The massive-leg (Z) Wigner transformation is *not* a phase but a genuine
   helicity mixing — that is the polarised observable itself, not an error to
   correct. It is why `SBORN` must be handed boosted momenta.

   **ISR — BLOCKER B1, and the branch that actually runs.**
   `fks_singular.f:5034-5035` hardcodes `cphi_mother=1, sphi_mother=0`; that is
   numerically identical to `getaziangles` on a beam-axis momentum
   (its `sth.ne.0` guard at `:4111` returns exactly `(1,0)` for `cth=+-1`) and
   it dies under a boost. But repairing it is not the main problem:

   The ISR counter-events are called with `y_ij_fks = one` **exactly**
   (`fks_singular.f:793`, `:904`; `y_ij_fks_matrix(1)=y_ij_fks_matrix(2)=1.d0`
   at `genps_fks.f:2742`), so `1d0-y_ij_fks .lt. vtiny` always fires and
   `azifact = xij_aor` is taken every time (`:5006-5007`). The
   `IXXXXX`/`OXXXXX` spinor branch below it is live only in the sliver
   `vtiny=1e-8 <= 1-y < tiny`, where `tiny=1d-6` in production and `1d-12`
   under `colltest` (`:4658`) — i.e. it is **never exercised by
   `test_soft_col_limits`**. Both branches still have to be made consistent or
   that sliver becomes a discontinuity nothing catches.

   `xij_aor = -exp(2*idir*i*phi_i_fks)` is a precomputed partonic-CM constant
   (`genps_fks.f:3096-3101`) and a `0/0` limit of `<ij>/[ij]`. A boost maps
   exactly-parallel null vectors onto exactly-parallel null vectors, so **the
   degeneracy survives in every frame — it can never be recomputed after
   boosting.** The azimuthal direction must be carried through as a stored
   4-vector.

   Proposed fix: store `k_perp_dir = (0, cos(phi_i_fks), sin(phi_i_fks), 0)`
   alongside `xij_aor`. One vector covers both beams (for `j_fks=2` the basis
   is `(x, -y)`, so it yields `psi = -phi_i` — precisely the `idir` sign
   already in `xij_aor`, and the `idir` special case disappears). Then boost
   it, project out the mother component, take
   `psi' = atan2(k'_perp . e2, k'_perp . e1)`, apply `-exp(-2*i*psi')`. At
   `Lambda=1` this reproduces current numbers bit-for-bit, so it lands as a
   safe refactor **before** any boost is wired in.

   `azifact` is invariant under independent positive rescalings, so
   `p_i_fks_ev` being the rescaled null vector `p_i/xi`
   (`genps_fks.f:3085-3094`) is harmless: `Lambda(lambda*p) = lambda*(Lambda p)`.

   **ISR is a package of three changes** — do any subset and you apply part of
   the correction, which is worse than the current code:
   (a) carry `k_perp_dir` through the boost as above;
   (b) `getaziangles` on the boosted mother instead of the hardcode;
   (c) **delete** the `j_fks.eq.2` `R_y(pi)` flip (`:5013-5020`) — after a boost
   it no longer maps the mother onto +z. Note `montecarlocounter.f:3000-3006`
   double-encodes the same thing (both the rotation and `cphi_mother=-1.d0`);
   numerically identical today since only the square enters, but do not carry
   both forward.

   **FSR:** `getaziangles(p_born(0,imother_fks), …)` (`:4844`) and `azifact`
   built by `IXXXXX/OXXXXX` on `p_i_fks_ev`, `p(.,j_fks)` (`:4830`) must both be
   evaluated on momenta carrying the *same* boost as the Born. The FSR
   equivalent of B1 has not been checked — establish whether the FSR
   counter-events also collapse onto the `vtiny` branch.

   **Scope narrowing:** `Qterms_reduced_spacelike` returns a real, z-only
   number (`:5538-5584`), non-zero only for a **vector mother** (`col1=8`); the
   `abs(m_type).eq.3 .or. ch_m.ne.0d0` branch that zeroes `Q` and `wgt1(2)` is
   the same statement one level up. So **only gluon-mother ISR configurations
   are affected at all** — a process whose ISR mothers are all quarks passes
   the collinear test even with a completely wrong azimuthal phase. Choose the
   validation channel deliberately.

   This is the frame/orientation consistency issue: it lands on the
   counterterms, not on `|M|^2`.

5. **Same construct elsewhere**, needing the same treatment if polarised
   aMC@NLO (not just fixed-order) is in scope: `montecarlocounter.f:2969-3012`
   and `montecarlocounter_alt.f:1445-1533`. `sreal_deg` (`fks_singular.f:5844`)
   has no azimuthal term — it needs the boost for the ordinary Born only.

**Recommended order within M2:**

0. Refactor the ISR azimuthal factor to the `psi`-form at `Lambda=1` (the
   `k_perp_dir` construction above, boost = identity). Demand **bit-identical**
   `test_ME`. This de-risks the hardest change before any frame logic exists.
1. Wire in a **longitudinal-only** `me_frame` boost. Must also be unchanged —
   the mother stays on the beam axis and the z-rotation Wigner phase of a
   z-directed massless particle is trivial, so the existing hardcode remains
   exactly correct. A useful null.
2. Then the general boost.

**Gate — `test_soft_col_limits`.** Already runs automatically as `test_ME` on
every `calculate_xsect` (`amcatnlo_run_interface.py:5494`,
`Template/NLO/SubProcesses/test_soft_col_limits.f`; the collinear scan at
`:617-663` drives `1-y = 1e-2 … 1e-10`). The `Q(z)` azimuthal term is the
*same order* in the collinear limit as the `AP(z)` term, so a wrong phase does
not cancel — the ratio plateaus off 1. Run it before any cross-section
comparison.

**Gate — acceptance test 2 (`p p > z{0} z{0}`):** the Born boost is trivial but
the real has `z z j`, so in the *finite* region the ZZ frame != partonic CM.
This exercises the real boost with the Born held fixed.

**But it does not discriminate the azimuthal phase.** In the singular region —
which is precisely where `test_soft_col_limits` probes — the reduced Born is
`p p > z z` with zero pT(ZZ), so the `me_frame` boost degenerates to
longitudinal and the phase issue does not bite. `p p > z{0} z{0}` will pass
either way. The discriminating processes are those whose `me_frame` system
recoils already at Born level: **`p p > z{0} j`** and **`p p > z{0} z{0} j`**,
restricted to gluon-initiated channels (see the scope narrowing above).

### M3 — `[QCD]`: virtual

Boost `p_born` before `Call BinothLHA(p_born,…)` (`fks_singular.f:7087`).
MadLoop already handles polarised MEs (`loop_exporters.py:1552` restores the
helicity-averaging factor). Note MadLoop has its own stability rescue that
re-evaluates in rotated/boosted frames — check it does not fight the imposed
frame.

**Gate:** `check_poles` must still pass (the poles are proportional to the
Born, so a Born/virtual frame mismatch shows up as a pole mismatch). Then
`p p > z{0} z{0} j [QCD]` end to end.

### M4 — Unblock and document

Remove the `[QCD]` rejection at `madgraph_interface.py:1245`, delete the stale
commented guard at `:1251`, narrow `check()` at `:1254` so massive-colourless
is allowed where supported. Update `help_polarization` (`:803`), which
currently documents `me_frame` as LO-only.

## 4. Acceptance test

One new test in `tests/acceptance_tests/test_cmd_amcatnlo.py`, modelled on
`MECmdShell.generate` + `self.do('calculate_xsect LO -f')` (`:722`), with the
reference LO numbers produced the way `tests/acceptance_tests/test_cmd_madevent.py:2870`
does it.

| stage | process | mode | assertion |
|---|---|---|---|
| M0 | `p p > z j [QCD]` unpolarised | NLO | xsec unchanged vs. today (regression) |
| M1 | `p p > z{0} j` | `[LOonly=QCD]` vs LO madevent | agree within MC error |
| M1 | `p p > z{0} z{0}` | `[LOonly=QCD]` | `me_frame`-independent (null test) |
| M2 | `p p > z{0} z{0}` | `[real=QCD]` | real boost: `test_ME` passes; xsec != partonic-CM value. **Weak** — see below |
| M2 | `p p > z{0} j` | `[real=QCD]` | **discriminating**: `test_ME` on gluon-initiated channels |
| M3 | `p p > z{0} z{0} j` | `[QCD]` | `check_poles` passes; xsec stable |

**Which process tests what.** The two are complementary, but not in the obvious
way:

- `p p > z{0} j` — the Z recoils against the jet already at Born level, so the
  `me_frame` boost is non-trivial and non-longitudinal in *both* the Born and
  the counter-event, including in the singular region. This is the process that
  discriminates the azimuthal phase, and it doubles as the M1 cross-check
  against LO madevent.
- `p p > z{0} z{0}` — the Born boost is the identity (the ZZ system *is* the
  partonic CM), so it is a clean null test at M1 and exercises the real boost in
  the finite region at M2. In the singular region the boost degenerates to
  longitudinal, so it will **not** catch a wrong azimuthal phase.

Keep M1/M2 in the always-run set (fast, no MadLoop) and gate M3 behind the slow
marker. Run via `./tests/test_manager.py`, not pytest.

## 5. Open blockers

- **B1 — `xij_aor` cannot be recomputed after a boost.** See M2 step 4. Drives
  the `k_perp_dir` design. *Confirmed.*
- **B2 — `me_frame` leg remapping** between Born and real multiplicities (D2,
  plus the `skip` logic at `genps_fks.f:1206-1222`, `:1542-1553`).
  Sub-claim resolved: the `2**n` vs `2**(n-1)` convention is **not** a bug —
  `mapid` is `ids(i)=btest(id,i)` (`cluster.f:141`), which matches
  `sum(2**n)` at `banner.py:4869`. The comment at `genps.f:1758` saying
  `sum 2**(N-1)` is stale; fix the comment.
- **B3 — the `R_y(pi)` flip must be deleted, not boosted**, and
  `montecarlocounter.f:3000-3006` must not keep double-encoding it.
- **B4 — FSR equivariance unconfirmed.** D3 is established for ISR only;
  confirm `shybst -> 0` in the FSR collinear limit (`genps_fks.f:2324-2339`).
  Also establish whether FSR counter-events collapse onto the `vtiny` branch
  the way ISR ones do.
- **B5 — `SBORN` momentum-cache coherence.** The cache hard-`stop`s on an
  `(E, p_z)` mismatch (`born_fks.inc:85-98`), so every `SBORN` caller within an
  event must agree on boosted vs. unboosted. A cheap crash, and a cheap
  self-check — this is the mechanical argument for D1 (caller-side boost).
- **B6 — no guard against a lightlike/null `me_frame` sum** in `boost_to_frame`
  (`Template/LO/SubProcesses/genps.f:1761`). Pre-existing at LO; do not
  replicate it into the NLO port.

## 6. Risk

M2 step 4 is the only genuinely hard item; everything else is mechanical. D2 is
the decision most likely to need revisiting, and M1 proves it out cheaply.
