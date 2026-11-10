# Symbolic kinematic provenance: next stress-case investigation

This read-only agent research concerns the unchanged production source at
`0fcd2cd873d61d134837e1d9e87afb13a47824fd`. It proposes no qualified numerical
change. The [closed invariant audit](../../experiments/invariant-classification-audit-1/README.md)
already distinguishes genuine small spacelike CMS transfers from the separate
EW mass-square inconsistency. No native calculation, guard change or production
edit was made for this research.

## Existing information and where it is lost

`LoopHelasAmplitude.get_denominators()` in
`madgraph/loop/loop_helas_objects.py:401–419` retains ordered corner external-leg
supports and masses. These are corner supports, not cumulative chord vectors:
`get_struct_external_leg_ids()` in `madgraph/core/helas_objects.py:2403–2415`
recursively excludes loop mothers. Full denominator structures determine loop
groups (`loop_helas_objects.py:649–677`).

The call emitter (`madgraph/iolibs/helas_call_writers.py:2001`) carries ordered
mothers, pairings and group identifiers. The optimized
`helas_loop_amplitude.inc:102` constructs corner momenta; `TIRUST_interface.inc:52`
applies ordering reversal and cumulative prefixes. Symbolic coefficient vectors
would need to follow the same ordering, including DP/QP reversal and independent
check modes 2/5 in `TIR_interface.inc:359`.

`BUILD_KINEMATIC_MATRIX` in `CT_interface.inc:163–299` receives numerical shifts
and masses without this provenance. Its energy-normalized zeroing branches
cannot identify whether a small invariant is a physical external photon shell
or a resolved composite spacelike transfer.

## Smallest next experiment

Use a private trace-only exporter through `madgraph/loop/loop_exporters.py:2668`.
Retain every existing arithmetic operation, threshold and output, while recording
group/pair identifiers, signed integer external-leg coefficients, raw `DIFFSQ`
and snap decisions. Accumulate corner supports into chords using the actual
HELAS momentum signs and conservation-complement cases.

Validate routing against the saved CMS points 20/21 and EW group 5, grouped and
ungrouped output, loop-induced output, reversal/rotation tests and explicit
off-shell legs. A support singleton alone is insufficient: external mass and
shell provenance matter, and the wavefunction `onshell` flag has decay-channel
semantics. Respect `offshell`/`KEEP_OFFSHELL_MASS` handling in
`helas_call_writers.py:1560` and `loop_exporters.py:3627`.

Only after exact routing validation consider a separately bounded experiment
that preserves a provably resolved composite spacelike transfer. Composite
provenance does not prove a nonzero invariant: physical soft/collinear limits
remain possible. True external-shell and ambiguous cases need their safeguards.
Independent finite/single-pole comparisons, stress/CMS qualification and runtime
checks would still be required; the frozen ordinary campaign must remain intact.

## Separate mass contract

The optimized exporter deliberately omits MP masses from these calls
(`loop_exporters.py:1528`) and forms `M2L=M**2` in DP at lines 1545–1547.
`TIRUST_interface.inc:156,178` promotes the resulting squared value to QP.
External phase-space repair receives MP model masses
(`loop_exporters.py:1390–1404`) and squares them in QP (`improve_ps.inc:620`).
Symbolic routing cannot reconcile those two mass contracts. Any separate mass
policy experiment requires all-backend and complex-mass validation. Neither
changing `OSThres` nor relaxing TIRust conditioning guards is qualified here.
