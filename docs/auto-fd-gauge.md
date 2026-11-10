# Automatic FD gauge for electroweak processes -- design note

Branch `claude/auto-fd-gauge-ew`. The FD gauge (Feynman-diagram gauge,
arXiv:2203.10440 and 2405.01256) is chosen by itself for a tree-level process
whose diagrams have no QCD vertex and involve a massive vector boson. This
note records the criterion, how the decision is taken at process time, and
what a per-process gauge inside one output would need.

## Why

In the unitary gauge the `q^mu q^nu / m^2` term of the massive vector
propagator and the longitudinal polarisation vector grow with the energy. For
vector-boson fusion and scattering the individual diagrams exceed their sum by
a factor growing with the energy and digits are lost when they are summed. In
the FD gauge the goldstone travels as a fifth component of the vector
wavefunction, the cancellation happens diagram by diagram and the precision
no longer depends on the energy: in single precision the mean number of
correct digits of |M|^2 for `u u > h j j j j` at 14 TeV goes from 2.8
(unitary) to 5.3 (FD), and the MadMatrix mixed-precision mode `v` is only
accurate in the FD gauge. Most users do not know about `set gauge FD`.

## The criterion (`fd_gauge_criterion` in `madgraph_interface.py`)

Read off the generated diagrams, never off the process line:

1. `electroweak_only`: no vertex of any diagram carries a QCD coupling order
   (every `model.get_interaction(vertex['id'])['orders']` has no `QCD`).
2. `massive_vector`: a massive spin-one particle is an external leg or an
   internal propagator of at least one diagram.
3. Tree level only: a loop amplitude or a process with perturbation
   couplings never qualifies.

The FD gauge is picked when 1 and 2 hold for *every* process of the session
(`generate` and the `add process` lines after it). `p p > w+ w- j j QCD=0`,
`p p > w+ w-`, `p p > h j j` (no QCD vertex at LO in the SM), Drell-Yan and
`e+ e- > mu+ mu-` switch; `p p > t t~` and `p p > w+ j` do not; `a a > mu+ mu-`
has nothing to gain and stays in the unitary gauge.

Why diagrams and not orders: `p p > w+ w-` or `p p > h j j` carry no order
constraint and are purely electroweak at LO; `QCD=0` on the line is a
sufficient but not necessary sign. Why not the sharper "massive vector
propagator" criterion alone: it would move every process with a W or Z,
`p p > w+ j`, `p p > z j j j`, `t t~` with decays, into the FD gauge, where the
madevent integration strategy differs (`sde_strategy = 2`, the single-colour
multi-channel strategy, is only used outside the FD gauge, see
`banner.py`) and where the default behaviour of the bulk of LHC samples
would change. The paper's showcase `u u > h j j j j` is mixed and benefits
too: `set gauge FD` is one line away, and the help text says so. Condition 2
keeps the switch to the processes with something to gain.

Never switched: loop processes, bound states (onia use their own ALOHA
functions), colour-ordered `[amplicol]` amplitudes, propagator-only
polarisations (`{G}`, `{H}`, `{Q}`, `{W}`, `{S}` are pieces of the
unitary-gauge propagator), a model without the Feynman gauge
(`1 not in model['gauge']`) and a model whose goldstone content `import_ufo`
could not check against the FD convention (no goldstone at all, or a
goldstone phase that no fermion vertex fixes): `UFOMG5Converter` then tags
the model with `fd_gauge_unverified` in its `limitations`, the interface
goes back to the unitary gauge and does not try again for that model.
`output ... --axial_gauge=True` (polarisation of the massless vectors in the
axial gauge of another leg, which the FD gauge has no room for) drops an
automatic FD gauge and generates the processes again in the unitary gauge.

## Point 2: the gauge decision at process time

Implemented. The gauge is still a property of the loaded model (the
goldstones are kept, dropped or merged into the vectors at import time), so
`set gauge` reloads the model and drops the amplitudes, and nothing cheaper
exists: the interaction ids and particle content of the FD model differ
from the unitary one, the amplitudes cannot be converted in place.

The decision therefore runs at the end of `do_add` (`apply_automatic_gauge`),
once the diagrams are known. When it wants another gauge,
`switch_gauge_and_regenerate` reloads the model through
`do_set('gauge X', log=False)` -- the loop `check gauge` already uses -- and
runs the `generate` / `add process` lines of the session again
(`_process_lines`, recorded with the process number each line was given, so
that the `P<n>_` directories keep their names). A process generated in the
unitary gauge first and found purely electroweak is generated twice; with
`QCD=0` the second pass is the only cost that cannot be avoided without
generating in the FD gauge first, which would penalise every QCD process.
Model reloads use the `_import_once` / pickle cache
(`py3_model_FDG.pkl`), a second or two for the SM.

What survives the reload: the multiparticle definitions (`define` lines are
in the history and `add_default_multiparticles` only drops a label whose
particles do not exist in the new model, which never happens between the
unitary and FD models since the goldstones are merged), the `import model`
line and the `_generate_info` of the first process. What is reset and
restored: `_export_format` / `_export_dir` around the `--axial_gauge` revert.

Who chose the gauge: `set2_gauge` records `_explicit_gauge` when called with
`log=True`, i.e. typed by the user; the internal callers (`check gauge`, the
automatic choice itself, the loop-model switches at import) pass
`log=False`. `set gauge default` hands the choice back. The startup dump of
the options into the history no longer writes `set gauge unitary` when it is
the default, since a proc_card run again would otherwise turn the default
into the user's choice; a gauge picked for a process is written as
`set gauge FD` ahead of the process lines (`record_automatic_gauge`), so that
the card reproduces the same code explicitly and the reweighting, which
replays the `set` lines of the card, runs in the gauge of the sample.
MadSpin skips `set gauge` lines on purpose and keeps its own gauge; its MG5
instance, the width calculator's and the reweighting's run with
`auto_gauge = False`.

`set auto_gauge False` turns the whole thing off (a MadGraph option,
recorded in the proc_card); turning it off while an automatic FD gauge is in
use goes back to the unitary gauge, as `set gauge` would.

## Point 3: a process-specific gauge inside one output

Not implemented; the gauge stays per output. The pieces that would have to
change, in generation order:

- **Model instances.** The particle content differs (goldstone bosons are
  merged in FD, absent in unitary) and so do the interaction ids, couplings
  (the FD merge rewrites the vector couplings) and the `_curr_gauge` flag
  the exporters read. Each process definition would carry its own model
  (`process['model']`), which the code supports in principle, but
  `_curr_model`, `_curr_helas_model` and `self._curr_exporter.model` are
  single objects, and `import_ufo._import_once` caches by
  `(model_path, aloha.unitary_gauge, prefix, decay)` with
  `aloha.unitary_gauge` a module-level global read at import and at ALOHA
  time. Two models would mean two values of that global alive at once.
- **ALOHA routines.** The FD gauge has five-component vector wavefunctions,
  its own `vxxxxx`, the inlined propagator factor (`get_fd_propagator_txt`),
  `DEFINE_GAUGE_DIR` and `aloha_functions_fd.f` in place of
  `aloha_functions.f`; `aloha_lib.KERNEL` is cleared on every gauge change.
  One output would need two sets of routines with distinct names (an
  `_fd` suffix, say) and two Fortran helper files, and the `wavefunctionsize`
  (6 vs 7, `export_v4.py`) would become per matrix element: the madevent
  `Source` makefile, `coupl.inc` and the common blocks are shared.
- **Per-process `proc_characteristic['gauge']`.** `banner.py` reads one
  value to pick `sde_strategy`, `gen_ximprove.py` hands it to the helicity
  recycler (`hel_recycle.py` peels 5 components in FD), `madmatrix` reads
  `aloha.unitary_gauge == 3` once per output (`model_handling.fd_gauge`).
  All three would need a per-subprocess value, the run card a per-process
  default, and the banner a list.
- **Downstream tools.** MadSpin regenerates the production process from the
  proc_card in its own gauge, the reweighting replays the `set gauge` line
  of the card: with several gauges in one card neither has a single line to
  replay. The citations are per output and would be unchanged.

The physics case for it is thin: the FD gauge is correct for QCD processes
too (the gauge only touches the non-QCD part), so an output mixing an
electroweak and a QCD process can simply be generated in the FD gauge
(`set gauge FD`) at the cost of the madevent single-colour strategy for the
QCD part. The current rule, "FD when every process of the output
qualifies", is the per-output projection of the same criterion.

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
- in the complex-mass scheme the gauges should agree exactly, and they do
  not here (`u u~ > w+ w-`: 5.1085553001036475e-3 unitary against
  5.1080495446772884e-3 FD, `u d > z u d QCD=0`: 1.4697541998128762e-6
  against 1.4704511965777327e-6). The FD gauge with `complex_mass_scheme`
  is not consistent at the 1e-4 level, a pre-existing issue worth its own
  investigation (real masses in the external polarisation vectors or in the
  goldstone sector are the usual suspects).
