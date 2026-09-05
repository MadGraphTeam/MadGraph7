# MLM merging in madspace — status, findings and open items

Working notes for the `feat-mlm` branch after merging `main` into it.

## What the feature is

`madspace/src/phasespace/mlm_clustering.cpp` compiles the diagram topologies of a
subprocess into a state machine of possible clustering histories, encoded as a
flat integer array. `madspace/src/kernels/mlm.hpp` walks that state machine for
one phase-space point, picks a clustering history by minimising a clustering
measure at each step, and reads off:

* the renormalisation scale,
* the factorisation scales,
* a clustering scale per outgoing jet (written to the LHE as `pt_clust_i`),
* the index of the diagram the history ends on (for colour flows and resonances
  in the LHE output).

`djb_clus`, `dj_clus`, `compute_scale` and `update_momenta` are ports of the
routines of the same name in `Template/NLO/SubProcesses/cluster.f`.

## Merge with main

The branch was 699 commits behind. Most of the conflict surface was not real:

* The generated instruction-set headers (`compgraphs/*.inc`, the cpu/gpu runtime
  mixins, `python/instruction_set.hpp`) are no longer checked in — main
  generates them from `instruction_set.yaml` via `generate_headers.py`. Only the
  two new `mlm_clustering_{hadronic,leptonic}` YAML entries were kept.
* `madevent.py` was split into `launch.py` / `run_interface.py`; the MLM hunks
  were re-applied to `launch.py`. The matrix element is now built in a loop over
  backends, so `me_inputs` is computed once before that loop.
* `run_card.toml` is now a template filled from `banner.py`, so `"mlm"`,
  `beam.jet_radius` and `beam.max_jet_flavor` are declared there.
* `LHECompleter` was refactored on main; main's version was taken. The branch's
  `TODO` about the permutation direction is gone because main fixed exactly
  that, by inverting the permutation before using it.

## Bugs found and fixed

1. **`update_momenta` arguments were swapped.** The kernel called it as
   `(..., p1_win, p2_win, ...)` against the signature
   `(..., i_remove, i_keep, ...)`, so it kept `particle2` and removed
   `particle1`. The state machine names the merged pseudo-particle by the lowest
   bit of the combined mask, i.e. by `particle1`, so every later state referred
   to a slot whose momentum had not been updated. It also made the
   `i_keep < 2` initial-state branch unreachable, and inconsistent with the
   caller's `is_initial = particle1 < 2`.

   The Fortran settles it: in `cluster_one_step`, `i` (removed) runs from 3 and
   `j` (kept) runs `1..i-1`, so the kept index is the lower one and `jwin <= 2`
   means "the kept particle is a beam".

2. **The winner selection was not reset between clustering steps.** Only
   `win_scale` was reset; `win_resonant` and `win_next_state` kept their values.
   After a resonant step, no non-resonant candidate could ever win the next
   step, and the walk would then jump to the previous step's target state.

3. **Nothing guaranteed a winner at all.** `win_next_state` starts at `-1` and is
   only assigned when a candidate beats the current best. A NaN measure never
   compares less than anything, so a step in which every candidate was NaN left
   `state = -1` and the kernel read outside the state machine — an actual crash,
   reproducible at a few events per thousand for strongly asymmetric beam
   energies. The boost and rotation `update_momenta` applies after an
   initial-state clustering can leave a pair exactly collinear, which is where
   the NaNs come from.

   Two fixes: `dj_clus` now zeroes a NaN result the way the Fortran does
   ("prevent numerical inaccuracies"), and the selection takes the first
   candidate unconditionally so a clustering is always chosen.

4. **State machine states were expanded more than once.** In `find_clusterings`
   the terminal branch guarded re-expansion with `if (current_states.size() == 0)`
   but the recursive branch did not, so a state reached along several paths had
   its whole transition list appended once per path. Harmless for the result,
   but the table grows exponentially with multiplicity.

5. **`bw_cutoff` was dead.** It was threaded from the run card all the way into
   the kernel signature and never read: the resonance window was `mass ± width`
   rather than `mass ± bw_cutoff * width`.

6. **External masses were indexed by topology slot, not by external leg.** The
   kernel indexes `external_masses` by leg, but the array was built as
   `incoming_masses ++ outgoing_masses`, which is in slot order. Invisible for
   the test processes, whose permutations only exchange identical gluons, but
   wrong for any permutation that moves a mass between legs. The permutation is
   now applied, and a mismatch between diagrams throws instead of being silently
   ignored.

7. **`alive` was hardcoded to `0xFFFFFF`** regardless of the particle count, and
   an empty (dead-end) state would have been read out of bounds. Both now
   handled.

## Vertex metadata, previously not populated

`is_qcd`, `is_jet1`, `is_jet2`, `mass_index` and the `massive_*` flags were all
hardcoded, so massless QCD was assumed everywhere and the resonance branch was
unreachable. They are now filled from the topology:

* masses and widths come from `Topology::Decay`, and from
  `t_propagator_masses()` for the t-channel chain,
* `Topology` gained `t_propagator_pdg_ids()`, filled alongside the existing
  t-channel masses and widths, because the flavour of a t-channel line changes
  along the chain and is needed to tell a QCD splitting from a QED one,
* the **external** pdg ids are not in `Topology` at all, so `MLMClustering`
  takes them as a new constructor argument (`external_pdg_ids`, plus
  `max_jet_flavor`). `launch.py` passes the `all_pids` it already computes. When
  the argument is omitted the old behaviour — everything is a QCD jet — is kept,
  so existing callers still work.

`is_jet` mirrors `isjet()` in `Template/LO/SubProcesses/reweight.f`
(`|pdg| <= max_jet_flavor`, gluons, and the merged-flavour placeholder 81). A
vertex counts as QCD when all three of its lines carry colour, which is the same
rule `launch.py` already uses in `build_multi_channel_data`. A propagator pdg of
0 means "not supplied" and falls back to deciding from the two children.

## Known differences from madevent, by design

These are the open physics questions, not defects. They are what makes a direct
comparison against the validated madevent MLM disagree.

* **Resonances.** MG5 reads the resonance structure off the SDE integration
  channel, which madspace/madnis does not have. This implementation instead
  prefers a resonant clustering over any non-resonant one, using the
  Breit-Wigner window. That is a hard switch, so a phase-space distribution can
  be discontinuous across the window boundary.

* **The factorisation scale.** The kernel uses a single scale for both beams:
  the smallest QCD clustering scale in the history, capped at mu_R. madevent
  tracks the two QCD lines connected to each beam separately and sets
  `mu_F(beam) = sqrt(pt2[jlast(beam)] * pt2[jcentral(beam)])`, so its two
  factorisation scales differ from each other and neither is the softest
  clustering. There is no equivalent of `jfirst`/`jlast`/`jcentral` here: the
  kernel does not trace parton lines through the event the way `ipartupdate`
  does.

* **The renormalisation scale.** The kernel takes the geometric mean of all
  clustering scales, replacing non-QCD ones by the largest scale. madevent takes
  `(pt2[jlast1] * pt2[jcentral1] * pt2[jlast2] * pt2[jcentral2])^(1/8)`, i.e.
  the geometric mean of four specific scales.

* **No Sudakov reweighting**, which is the other half of what the NLO code does.

## Open items

* **The trigger for MLM is `dynamical_scale_choice = "mlm"`, which is probably
  the wrong switch and needs revisiting.** In madevent, merging is turned on by
  `ickkw` (with `xqcut`), while `dynamical_scale_choice = -1` selects the CKKW
  clustering scale. Conflating the two means MLM merging cannot be combined with
  another scale choice, and the clustering scale cannot be used without merging.
  A separate `[matching]`-style switch is the likely shape.

* Pass the selected diagram index through to the matrix element. The plumbing is
  in place (`MatrixElement::diagram_in` replaces `random_diagram_in` when MLM is
  active) but the madmatrix side needs per-event diagram input support. Affects
  the LHE output only, not the scales.

* The `dj_clus` massless/massive test is `mass > 0` here, where the Fortran uses
  thresholds tied to `maxjetflavor` (`m >= 3` if `maxjetflavor > 4`, else
  `m >= 1`). Equivalent for realistic masses; only differs for a tracked mass
  between 0 and 1 GeV.

* Different diagrams can put different propagators on the same partition of
  external legs (a photon and a Z, say). The compiler keeps whichever has a
  width, since that is the one the resonance test is about, but this is an
  arbitrary choice where the diagrams genuinely disagree.
