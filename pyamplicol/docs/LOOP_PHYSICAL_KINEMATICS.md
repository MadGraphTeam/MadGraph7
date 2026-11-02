# Physical invariant bindings

Physical loop programs use the process's declared external mass shells when
constructing the invariant matrix for reduction. The graph/numerator oracle
retains its raw off-shell kinematics. This distinction is fixed in the generated
plan and applies identically in double, DoubleFloat and arbitrary precision.

For each family, native graph routing supplies the exact external-momentum
signature of every propagator shift. If a shift difference is exactly an external
momentum or its negative, its squared invariant is bound to that external
particle's model mass squared. This includes the dependent external momentum
eliminated by momentum conservation. No component tolerance, numeric snapping,
default-parameter comparison or process-name selection is used.

The existing family mass evaluator keeps denominator masses as its first outputs.
An external mass expression reuses an output only when the model-normalized
symbolic expressions agree; otherwise it gains a separate output. Thus equal
current numerical values never tie independent parameters together. The runtime
copies the selected mass outputs into both symmetric invariant entries using
native batch operations. Point-dependent source scale rows and original lane IDs
are retained. Numerator currents, wavefunctions and supplied momenta are unchanged.

The same PACBIN family descriptor records the physical context, combined mass
output count and invariant-to-mass assignments. The checked loader rejects
invalid, duplicate, asymmetric and out-of-range assignments or incompatible
evaluator shapes. Existing descriptors without this field retain their original
raw behavior; regenerating a physical artifact activates the new bindings.

Focused tests cover both routing orientations, the dependent external leg,
massive parameter updates, equal-valued independent masses, all three carriers,
and malformed saved bindings. The original rounded six-leg RAMBO controls remain
in the physical regression, alongside exactly on-shell precision controls.

The v2 engine (the only one-loop engine) binds the same invariants without a stored
assignment: in every loop family an invariant whose leg set is one external leg,
or whose complement is, takes that leg's squared mass at the point's parameters,
the exact value its propagators use, in double, DoubleFloat and arbitrary
precision alike (`loops/v2/families.onshell_leg`; DESIGN §11.1 of the v2 design).

This is a physical-input contract, not an input-repair algorithm. A caller must
still supply momenta consistent with the declared physical process. General
input-consistency diagnostics and structured numerical failure/rescue delivery
remain separate work; this change does not certify arbitrary off-shell inputs as
physical scattering events.
