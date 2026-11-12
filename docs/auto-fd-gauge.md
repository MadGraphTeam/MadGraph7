# Automatic FD gauge, per process -- design note

Branch `claude/auto-fd-gauge-ew` (PR #422). The FD gauge (Feynman-diagram
gauge, arXiv:2203.10440 and 2405.01256) is chosen by itself for the processes
whose diagrams need it, and an output can hold subprocesses in both gauges.
This note records the criterion, how the decision is taken at process time,
how one output holds two gauges, and what that costs.

## Why

In the unitary gauge the `q^mu q^nu / m^2` term of the massive vector
propagator grows with the energy when the propagator ends on a current that
is not conserved. The individual diagrams then exceed their sum by a factor
growing with the energy, and digits are lost when they are summed. In the FD
gauge the goldstone travels as a fifth component of the vector wavefunction,
the cancellation happens diagram by diagram, and the precision no longer
depends on the energy. In single precision the mean number of correct digits
of |M|^2 for `u u > h j j j j` at 14 TeV goes from 2.8 (unitary) to 5.3 (FD),
and the MadMatrix mixed-precision mode `v` is only accurate in the FD gauge.
Most users do not know about `set gauge FD`.

## The criterion (`fd_gauge_criterion` in `madgraph_interface.py`)

The rule proposed in the review of PR #422, read off the generated diagrams:

1. Look at every internal massive vector line (a W or Z propagator) and at
   the vertex at each of its two ends.
2. An end is **conserved** when the vertex has two massless fermions (`e ve`,
   `u d~`, ...). A massless quark line that also radiates gluons stays
   conserved, since current conservation holds once the gluon insertions
   are summed. The SM charged leptons count as massless (their masses are
   negligible against the W/Z mass), so a `tau` current is conserved. An end
   is **not conserved** at a vertex with a massive quark (`t b`, `t t`, a
   massive `b`) or a bosonic vertex (`HVV`, `VVV`, `VVVV`).
3. The FD gauge is wanted when at least one massive vector propagator has at
   least one non-conserved end.
4. External massive vectors are not looked at. In a decay chain a decayed
   massive vector is a propagator, ending on its production and its decay
   vertex.

| process | massive propagator | ends | gauge |
|---|---|---|---|
| `u d~ > e+ ve g` | W | leptons, quark line + gluon | unitary |
| `g b > t e- ve~` | W | leptons, t-b | FD |
| `u d~ > w+ g`, `g b > t w-` | none (external W) | | unitary |
| `g g > t t~ g` | none | | unitary |
| `u u > h j j j j` | W/Z | quark line, HVV | FD |
| `t t~` with leptonic W decays | W | leptons, t-b | FD |
| `p p > w+ w-` | Z/a | quark line, WWV | FD |
| `p p > e+ e-`, `e+ e- > mu+ mu-` | Z | massless currents | unitary |
| `e+ e- > ta+ ta-` | Z | SM leptons | unitary |

Never FD: loop processes and processes with perturbation couplings, a
particle of spin above one anywhere in the diagrams (`p p > w+ y` with the
spin-2 `y` of `loop_smgrav` is wrong by a factor 300 in the FD gauge), bound
states, `[amplicol]` processes (pyAmplicol has no FD gauge yet, see below),
polarised legs (`{0}`, `{T}`, ...: the split of a cross section into
polarisations depends on the gauge and is defined in the unitary gauge),
propagator-only polarisations (`{G}`, `{H}`, `{Q}`, `{W}`, `{S}` are pieces
of the unitary-gauge propagator), a model without the Feynman gauge
(`1 not in model['gauge']`), and a model whose goldstone content `import_ufo`
could not check against the FD convention (tagged `fd_gauge_unverified`).

## The decision at process time, per process line

The gauge of a model is fixed at import (the goldstones are kept, dropped or
merged into the vectors), so a process is generated in the session gauge
first, then judged on its diagrams (`apply_automatic_gauge`, at the end of
every `generate` and `add process`):

- every line wants the FD gauge: the whole output goes to the FD gauge, as
  `set gauge FD` would do, and the proc_card says `set gauge FD`;
- no line wants it: everything stays in the unitary gauge;
- some lines want it: the session stays in the unitary gauge and those lines
  are generated again with `--gauge=FD`, as if the user had typed it. The
  proc_card carries the option on those lines, behind the comment
  `# --gauge=FD chosen automatically ...`.

The lines and the amplitudes each one produced are kept (`_process_lines`,
`_line_ranges`), so a line that changes gauge is generated again with its
process number, and a new last line that wants the FD gauge is generated
again alone. A line given `--gauge=X` by the user keeps it, and `set gauge X`
or `set auto_gauge False` turn the automatic choice off.

`generate ... --gauge=FD` is also an option for the user. Such a line is
generated with the session model imported in the FD gauge
(`model_for_gauge`, imported once per session model under
`aloha.gauge_scope`), and its process definitions, decay chains included,
carry `Process['gauge']`. The gauge is part of the matrix-element identity
and of the subprocess grouping keys, so processes of two gauges are never
merged. Refused with `--gauge=FD`: `[...]` lines (loop, NLO, `[amplicol]`,
interference), spin above one, bound states, the propagator
polarisations the FD gauge has no counterpart for (`{G}`, `{H}`, `{Q}`,
`{W}`: a polarised propagator is validated in the gauge of its line, so
`{0}`, `{T}`, `{A}`, `{S}`, `{0S}` are accepted), models without a checked
FD gauge, and models that define a coupling,
parameter or Lorentz name differently in the two gauges (none of sm, heft,
loop_sm does, measured).

## One output, two gauges

Supported by `madevent`, `standalone_fortran`, `standalone` and `mg7`.
Outputs in one gauge are byte-identical to before.

- **Fortran.** The ALOHA routines of the other gauge go to `Source/DHELAS_FD`
  (`libdhelas_fd`, with `aloha_functions_fd.f` and its `ALOHA` type of five
  components), and the P directories of that gauge link it (the mechanism of
  the P-wave onia libraries). Each P directory is written with a helas writer
  for the model of its gauge, and with that model as the exporter's model,
  under `aloha.gauge_scope`. `proc_characteristics` lists them
  (`gauge_of_dirs = ['P1_qq_wpwm=FD']`). The Fortran helicity recycler reads
  it per directory, and the madevent `sde_strategy` default avoids strategy 2
  when any directory is FD.
- **C++.** `src/HelAmps_<model>_fd.h` holds the FD routines in an
  `inline namespace gauge_fd` inside `madmatrix`: the same routine names and
  an `ALOHAOBJ` of five components, distinct symbols. The P directories of
  that gauge include it. One helas writer serves both gauges, because it
  numbers the couplings of the shared `Parameters`; the model and the call
  caches are swapped per gauge (`HelasCallWriter.gauge_cache`).
- **Other formats** write every matrix element into one library. With an
  automatic choice they get one gauge: FD when every process can be in it
  (correct for all of them), unitary otherwise. A `--gauge=` line of the
  user is refused there.
- **MadSpin and the reweighting** ask the banner for the proc_card without
  `--gauge=` options (`Banner.proc_card_in_one_gauge`; the shared reader
  `charge_card` keeps the card as written). MadSpin keeps its own gauge. The
  reweighting, which builds one library, generates the processes again with
  the automatic choice on, so its output gets one gauge the same way as any
  single-library output: FD when every process can be in it, else unitary
  (a polarised or spin-2 process keeps the unitary gauge).
- A model changed after its import (`add model`, `customize_model`) is never
  reloaded into another gauge: no automatic choice, and `--gauge=` refused.
- The FD gauge is cited when any process uses it.

Checked: the mixed `standalone_fortran` and `standalone` outputs of
`u u~ > w+ w- --gauge=FD` plus `g g > t t~` give the `check_sa` values of the
all-FD and all-unitary outputs to the last digit; madevent grouped and
ungrouped agree (458.2 +- 1.5 pb and 459.1 +- 0.9 pb), and the FD
`P1_qq_wpwm` gives 78.21 +- 0.63 pb against 78.33 +- 0.18 pb all-FD; mg7
reproduces the all-FD cross section with the same seed, 465.0596344083037 pb.

Not covered: `--me_exporter=` (a second exporter), `--postpone_model`,
P-wave onia, `check ... --gauge=`.

## What the FD gauge costs

The review asked whether FD is worth its price. Same madevent run in both
gauges (5000 events, one seed, sm):

| process | us per point, unitary -> FD | unweighting efficiency, unitary -> FD |
|---|---|---|
| `p p > w+ w-` | 38.9 -> 41.1 | 5.3% -> 5.7% |
| `p p > h j j QCD=0` | 113 -> 143 | 1.9% -> 2.7% |
| `t t~` dileptonic decay chain | 49.0 -> 52.0 | 6.2% -> 6.2% |

The matrix element costs 6% to 27% more per point in FD; the sampling
efficiency is the same or better (for VBF higgs the integration needed 266k
points instead of 462k, and less CPU in total). Cross sections agree within
errors.

## `[amplicol]` and the FD gauge

The review asks to impose the FD gauge on `[amplicol]` processes, since the
GPU version of pyAmplicol will run in mixed precision. pyAmplicol, vendored
under `pyamplicol/`, has unitary and Feynman vector propagators and its own
goldstone-partner compiler, but no FD gauge: porting it is a project of its
own on the pyAmplicol side. Until then an `[amplicol]` line stays in the
unitary gauge and no longer prevents the other lines from being FD.

## Outputs waiting for their model (`--postpone_model`)

`output X --postpone_model` writes the processes but defers the model (the
ALOHA routines and the parameters) to a later `output` into the same
directory, which is how several process definitions end up in one output.
The processes already written are in the gauge of the session at that time,
and the model written at the end has to match them, so while such an output
is pending (`postponed_export_pending`, i.e. `finalize` left
`previous_lorentz` on the interface) the gauge is fixed:

- `apply_automatic_gauge` keeps the current gauge, whatever the processes of
  the next `generate` would have preferred, with one info line per
  `generate`;
- the cases where an automatic FD gauge has to step aside (a loop process,
  `--axial_gauge=True`, `set auto_gauge False`) are refused with an
  `InvalidCmd` that says to start again with `set gauge unitary`
  (`refuse_automatic_gauge_change`);
- an explicit `set gauge` that changes the gauge drops the pending output with
  a warning, since its model can no longer be written: its processes have to
  be generated and output again.

Without this, `generate w+ w- > w+ w-` (FD), `output matchbox X
--postpone_model`, `generate u u~ > u u~ [virt=QCD]` (back to unitary),
`output matchbox X -f` asked the unitary `loop_sm` for the FD-only Lorentz
routines of the first process and crashed in the model conversion.

## Spin above one

`p p > w+ y` in `tests/input_files/loop_smgrav` (a spin-2 `y`) picked the
FD gauge (the model has goldstone bosons, with a convention that could be
checked) and its matrix element came out at 5148 instead of 16.95 at the
`check_sa` point: the merged vector/goldstone vertices are only right for
the spin content of the SM. `fd_gauge_criterion` therefore says no as soon
as a particle of spin above one (spin 3/2 or 2) is an external leg or takes
part in a vertex of a diagram.

## The multiparticles of the user across the reload

`set gauge X` reloads the model, and `add_default_multiparticles` then drops
the `define V z|a` definitions (their entries are lists, which it cannot look
up) and edits the lists of `p` and `j` in place for the flavour scheme of the
model, so `define p = p b b~` was lost too. With the automatic choice this
happened behind the user's back (`test_generate` of the acceptance tests
lost its `V`). `set2_gauge` now keeps a deep copy of the dictionary across
the reload and puts the user's definitions back, for the explicit `set gauge`
as well.

## What changes in the numbers: the widths

At the `check_sa` phase-space point (1 TeV), `output standalone_fortran` of
the `sm` in the unitary and in the FD gauge give:

| process                   | unitary                 | FD                      | relative |
|---------------------------|-------------------------|-------------------------|----------|
| `e+ e- > z h`             | 2.0988604744576469e-3   | 2.0988604744576469e-3   | 0        |
| `e+ e- > z z`             | 3.6882423618613985e-3   | 3.6882423618613976e-3   | 2e-16    |
| `u u > h u u QCD=0`       | 9.2275820633168406e-10  | 9.2275820633168571e-10  | 2e-16    |
| `u u~ > w+ w-`            | 5.1099582274945965e-3   | 5.1093405664217962e-3   | 1.2e-4   |
| `u d~ > w+ z`             | 3.3989086546951687e-3   | 3.3982047233175557e-3   | 2.1e-4   |
| `e+ e- > w+ w- z`         | 6.0729806614484482e-7   | 6.0674525348460480e-7   | 9.1e-4   |
| `u d > z u d QCD=0`       | 1.4704291881825141e-6   | 1.4711011009947315e-6   | 4.6e-4   |

With every width set to zero in the param_card the two gauges agree to
machine precision (`u u~ > w+ w-`: 5.1093404049749446e-3 against
5.1093404049750504e-3; `u d > z u d QCD=0`: 1.4718113670817815e-6 in
both), and the Feynman gauge reproduces the unitary one bit for bit with the
widths in. The differences are the known gauge dependence of the fixed-width
treatment, which the unitary gauge amplifies with the energy (its
`q^mu q^nu / m^2` terms): switching the default gauge moves the numbers of a
process with W bosons by this much. Two things follow from it:

- the acceptance tests that compare against stored unitary-gauge values
  (`test_standalone_merged_flavor_uq_zuq`, `test_decay_chain_symmetry_factor`,
  which also has the off-resonance gauge dependence of a decay-chain diagram
  subset: 0.9% for `e+ e- > z z h, z > u u~, z > e+ e-, h > b b~`) pin the
  unitary gauge;
- in the complex-mass scheme the two gauges do not agree either
  (`u u~ > w+ w-`: 5.1085553001036475e-3 unitary against
  5.1080495446772884e-3 FD, `u d > z u d QCD=0`: 1.4697541998128762e-6
  against 1.4704511965777327e-6), and they cannot as long as an external
  W or Z has a width: see the next section.

### The complex-mass scheme

The difference had two causes: an FD bug, now fixed, and a limit of the
complex-mass scheme that every gauge shares.

**The FD propagator contracted the goldstone component with the conjugate
mass.** The FD propagator projects the off-shell wavefunction with the
5-momentum `q^A = (q^mu, -i m)`, and its `js2` coefficient contracts the
fifth component with `+i m`. The routines wrote that as the complex
conjugate `DCONJG(FDQ(4))`. That is the same number for a real mass. For the
complex mass `m^2 = M^2 - i M Gamma` it gives `+i conj(m)`, which is not
analytic in `m` and breaks the gauge cancellations of the scheme. It
matters for an internal W, Z, h whose goldstone component is fed at both ends
(a longitudinal W, a massive fermion, a Higgs). With the width of the
external particles set to zero, so that only this effect is left (internal
widths kept):

| process (complex-mass scheme)       | widths set to zero | unitary                | FD before              | FD after               |
|-------------------------------------|--------------------|------------------------|------------------------|------------------------|
| `w+ w- > w+ w-`                     | W                  | 0.43405089229016358    | 0.43405107789616149    | 0.43405089229016591    |
| `t t~ > w+ w-`                      | t, W               | 7.5254435207372905e-2  | 7.5254488939839950e-2  | 7.5254435207374043e-2  |
| `b b > h b b QCD=0`                 | h                  | 1.8856866728922558e-9  | 1.8857291829819852e-9  | 1.8856866728922546e-9  |

The routines now use the analytic `-FDQ(4)`: the Fortran and madmatrix
writers in the complex-mass scheme (with a real mass the generated code is
unchanged, byte for byte), the Python `wavefunctions.multiply_propagator_factor`
and the Fortran and C++ `multiply_propagator_factor` templates. The C++
one now takes a complex mass, and madmatrix builds `FDQ[4]` from both parts of
the complex `M` (it used to write `-M` into the imaginary part). The madmatrix
`multiply_propagator_factor` of `helas_fd.h` keeps its conjugate: it takes a
real mass, no generated routine calls it, and it is part of an IOTest
reference. Neither
`u u~ > w+ w-` nor `u d > z u d QCD=0` was affected: their internal Z/W
propagators end on massless quarks, which have no goldstone coupling.

`goldstone_mass_mismatches` (the import-time check of the goldstone couplings)
read the couplings, built with the complex masses, against the real masses,
and warned about every massive particle of the `sm` in the complex-mass scheme
(t at 172.975 GeV, h at 124.978 GeV, ...). It now uses `CMASS_<mass>` when the
model has it.

Tests: `test_fd_complex_mass_scheme_matches_unitary` (test_process_checks,
Python evaluation, both gauges at 1e-10), `FDPropagatorComplexMassTest`
(test_helas_call_writers: the Fortran and madmatrix routines, and the
analyticity of the Python template), and the complex-mass case of
`test_goldstone_mass_mismatches`.

**An external particle with a width has no gauge-independent amplitude.** The
complex-mass scheme is gauge invariant for stable external states only. The
Ward identity that relates the longitudinal polarisation to the goldstone,
`k_mu M^mu = i mu M_G`, is exact only when the other external legs sit on
their complex mass shell `p^2 = mu^2`. An external W generated at
`p^2 = M_W^2` is `i M_W Gamma_W` away from it, and the identity is broken by
a term proportional to that. The FD gauge writes the longitudinal W as
`k^mu/M + eps~` and uses the identity to trade `k^mu/M` for the goldstone, so
it drops that term. The unitary gauge keeps it, multiplied by the `E^2/M^2`
of its longitudinal polarisations. So `u u~ > w+ w-` and
`u d > z u d QCD=0` cannot agree at machine precision with the width of
their external W/Z in, and no other choice in the FD routines makes them
agree: putting `mu/M` instead of 1 in the goldstone component of the external
W moves the FD number by 1e-4, but not onto the unitary one.

This is not specific to FD. In `w+ z > w+ z` the Feynman and unitary
propagators differ in a way that matters, and the three gauges give three
numbers (complex-mass scheme, `check_sa` point at 1 TeV):

| widths                    | unitary            | Feynman            | FD                 |
|---------------------------|--------------------|--------------------|--------------------|
| all in                    | 2.0314519609875217 | 2.0243218774357561 | 1.9415286927692104 |
| W and Z set to zero       | 1.9421716420326598 | 1.9421716420326913 | 1.9421716420327024 |

With the widths of the external particles set to zero, every process here
agrees between the gauges to machine precision: `u u~ > w+ w-` with the W
width at zero, 5.0417665595190201e-3 unitary against 5.0417665595190401e-3
FD; `u d > z u d QCD=0` with the Z width at zero, 1.4544995228660480e-6
against 1.4544995228660486e-6.

Of the gauge-dependent answers, the FD one is the better behaved. It stays
within O(Gamma/M) of the zero-width value, while the unitary gauge
amplifies the gauge-violating piece with the energy. At 1 TeV
`w+ w- > w+ w-` gives 0.47460 unitary, 0.43991 FD, and 0.43405 with the W
width at zero. `w+ z > w+ z` gives 2.0315 unitary, 1.9415 FD, and 1.9422
with the W and Z widths at zero. For a gauge-independent number, set the width
of the particles that appear as external states to zero (the param_card has
one width per particle, so this also removes it from their internal
propagators). `help set gauge` says so.

Two more things stood in the way of checking all this, and are fixed too:

- **`check gauge` stopped at its axial-gauge step in the complex-mass
  scheme.** Every ALOHA writer refused a custom propagator denominator with
  a complex mass, including the axial gauge `(P.PBar) P^2` of a massless
  vector, which has no mass or width at all. A custom denominator is now
  written as it is unless it uses the width (the polarised propagators keep
  a refusal, with a message that says why). With the external W width at
  zero, `check gauge w+ w- > w+ w-` gives the same number in the unitary,
  Feynman, axial and FD gauges (1e-13); `b b > h b b QCD=0` with the h width
  at zero, 3e-15. Test: `test_check_gauge_complex_mass_scheme`.
- **madmatrix (`output standalone`, and so mg7) did not compile in the
  complex-mass scheme.** Three things were missing:
  - `Parameters.cc` takes the `sqrt` of a `cxsmpl`, the complex masses. A
    host-only `sqrt(cxsmpl)` now sits next to `conj` in the four
    `mgOnGpuCxtypes.h`.
  - The real `cIPD` table held the complex `CMASS_*` of the propagators. A
    complex parameter now takes two slots (real, imaginary), and the call
    passes `cxmake( cIPD[k], cIPD[k+1] )`.
  - The unitary propagator wrote `1/M^2` into a real `OM`. In the
    complex-mass scheme `OM` is a `cxtype_amp`.

  Outside the complex-mass scheme the generated code is unchanged. Built in
  double precision (`FPTYPE=d`), at a pinned phase-space point, madmatrix
  equals the Fortran standalone to 1e-15 in the complex-mass scheme, in the
  unitary and in the FD gauge, with and without widths (`w+ w- > w+ w-`,
  `t t~ > w+ w-`). The default mixed-precision build is 1e-7 away. Test:
  `test_madmatrix_complex_mass_scheme_vs_fortran`. `check language` also
  built its Fortran, C++ and madmatrix outputs without the complex-mass
  option (the Fortran one did not compile: no `CMASS_*` in `coupl.inc`); it
  now passes it on.

Two further madmatrix gaps in the complex-mass scheme showed up once MadSpin
used it (any output with a running coupling, or with powers of a mass):
- `G` and the parameters computed from it (`mdl_G__exp__2`) are typed
  complex there. `computeDependentCouplings_fromG` left the complex `G` line
  in (`aS` undeclared, `G` redefined) and wrote `mdl_G__exp__2` as a
  `constexpr`. Now `G` is commented out like the real one, and a complex
  dependent parameter is a `cxtype_sv` (`cxtype` in the float-vector branch).
- `Parameters.cc` takes `pow` of a complex mass (heft): a host `pow(cxsmpl)`
  now sits next to `sqrt`.

**MadSpin now uses madmatrix in the complex-mass scheme.** `density_backend`
had forced the Fortran output there. `g g > t t~` with
`decay t > w+ b` / `decay t~ > w- b~` (`test_complex_mass_scheme`) gives
byte-identical decayed events with the madmatrix and the Fortran density
backends (same seed). `check language` agrees for `g g > t t~ g` (the `G^2`
coupling) and `heft g g > h h` (the `pow`).

**`check language` compared different phase-space points.** `t t~ > w+ w-`
(0.4%) and `w+ w- > w+ w-` in the FD gauge (5%) failed, with or without the
complex-mass scheme. There were three causes, none in a matrix element:
- madmatrix's `check_sa` matrix mode printed the classic RAMBO point, then
  evaluated another one. Its UMAMI buffer was overwritten with the momenta of
  a `RamboSamplingKernelHost` at the unchanged energy.
- The Fortran/C++ `check` drivers raise an energy at or below twice the sum
  of the external masses to that value (`t t~ > w+ w-` at 1 TeV runs at
  1013.676 GeV). madmatrix and the Python column did not.
- The C++ FD `vxxxxx` (`aloha_aux_functions.fd_cc`) wrote the full
  longitudinal polarisation and no goldstone component, which equals the FD
  one only at zero width (the external-width limit above).

All three are fixed. `check language` now also decides the energy once per
process and passes that value to every backend, so the drivers' own
threshold rules no longer matter. Every column of both checks agrees
(`test_check_language_massive_initial_state_and_fd`).

Review follow-ups:
- the width check of the complex-mass scheme looks at the numerator of a
  propagator as well as at a custom denominator, and runs for every
  off-shell routine (`check_complex_mass_propagator`);
- `$` exclusions (`p p > e+ e- $ z`) were broken in the complex-mass
  scheme. The window `|sqrt(s) - M| < BWCUTOFF * W` sits in the FCT
  definitions of the P1D routine, and the routine lost both `W` and
  `BWCUTOFF` from its arguments although every call site passes `BWCUTOFF`.
  In Fortran the arguments were shifted by one and `W` was uninitialised, so
  the result was silently wrong; madmatrix and C++ did not compile. `BWCUTOFF`
  is an argument again, and the window uses the real pole mass and the width
  of the complex mass (`M**2 = MR**2 - i MR W`). The window edges are then the
  fixed-width ones, and Fortran and madmatrix agree
  (`test_dollar_exclusion_complex_mass_scheme`). The Fortran lines use the
  generic `REAL`/`AIMAG`, so that the quadruple-precision copy of the routine
  (MadLoop, aMC@NLO) keeps a `complex*32` mass: `DIMAG` does not compile
  there, and `DBLE` would truncate the pole mass;
- `$` on a massless particle (`$ a`, `$ g`) is refused when the process is
  defined: there is no on-shell window to remove, and in the complex-mass
  scheme the width of the window, `-Im(M**2)/MR`, is 0/0, which made the
  theta function zero and silently dropped the diagram. Use `$$` or `/`
  instead. The `syntax` tutorial now compares `$ z` with `$$ z`;
- the `MADARITH_DOUBLEEXPANSION` (`FPTYPE=e`) variant of the complex-mass
  propagator in madmatrix cast the complex mass to a real and dropped the
  width. It now subtracts the complex `M*M`: real part in double-word
  arithmetic, imaginary part as the width. That build does not compile on
  this platform even without the complex-mass scheme (C++20 concepts under
  `-std=c++17`, then `Double<double>` conversions), so the variant is checked
  on the generated text only, against the real-mass one
  (`test_madmatrix_doubleexpansion_propagator`).

Not addressed here: the madmatrix `HRDCOD=1` build is still broken with
complex parameters (pre-existing). A MadLoop standalone output with `$`
(`u u~ > e+ e- $ z [virt=QCD]`) compiles its HELAS library but does not link,
in either scheme: the loop HELAS functions have no `theta_functionr`
(pre-existing).
