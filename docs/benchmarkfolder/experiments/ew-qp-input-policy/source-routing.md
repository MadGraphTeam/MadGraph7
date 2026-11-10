The source correction needs a coordinated QP mass path. Changing only the
generated expression, or only a callee declaration, would leave an inconsistent
Fortran calling convention. No production source or generated export was
modified for this review, and no compiler or reducer was run.

`source-routing.json` preserves the exact working-file SHA-256 values, numbered
excerpts, and the separately recorded repository HEAD and Git blob identities.
A working file may differ from the blob at that HEAD. The recorded HEAD is not
a claim that every audited byte belongs to that commit. Ignored historical
exports and installed libraries are independently bound by content hashes.

The observed mass has a specific representation contract. In the historical
`ew_n3_epem_ttxa/candidate-9-qp-init-dp` export, `Cards/param_card.dat:13` supplies
the top mass. `Source/MODEL/param_read.inc:25–27` reads both `MDL_MT` and
`MP__MDL_MT`; `lha_read.f:374–375,403–404` reads a found MP card entry through a
`REAL*8` buffer and assigns that to the `REAL*16` output. Thus, for this present
entry, `MP__MDL_MT` equals the promoted DP mass. It is not decimal 173.3 read
directly into binary128. The missing-parameter default branch at line 415 is
different: it assigns the QP default directly.

This equality must not be generalized to all model masses. The model can have
QP-derived real masses and complex masses. The compatible general rule is to
use the existing MP model symbol corresponding to the propagator's model mass,
not to reconstruct it by promoting every DP model value. The source selects
external real particle masses in `core/helas_objects.py:5659–5673`; the exporter
uses their MP symbols in `loop_exporters.py:1390–1404`. For a stable real particle
that is both an external leg and an internal propagator, the repair and proposed
denominator square must use that SAME MP symbol.

For this fixture the link is explicit: generated `SubProcesses/P0_epem_ttxa/
improve_ps.f:795–799` sets the two external top masses to `MP__MDL_MT`, and
lines 891–895 square those masses when rebuilding the energies. The loop group 5
call in `loop_CT_calls_1.f:113–114` names `MDL_MT`. Its QP counterpart must
therefore use `MP__MDL_MT**2`. The current shared `LOOP_4` instead declares the
mass arguments and `M2L` as `COMPLEX*16` at `CT_interface.f:1475,1484`, and
squares them at lines 1532–1535.

Model updates also matter. The historical `rw_para.f:32–40` loads the card and
calls `COUP`; `couplings.f:25–29` initializes the MP model expressions. Before
QP numerator evaluation, `loop_optimized/mp_compute_loop_coefs.inc:208–224`
calls `MP_UPDATE_AS_PARAM` and copies the repaired point. Generated
`couplings.f:134–155` includes `actualize_mp_ext_params.inc` before evaluating
the dependent MP expressions. In this particular model the entire actualize
file updates only `MU_R`, `AS`, and `G`; it is not a general resynchronization of
all externally edited DP masses. A source correction should preserve the
existing card/model-update contract and refresh runtime mass squares from the
current model values. It should not introduce a static mass `DATA` table.

The decisive optimized routing sites are:

| Layer | Current behavior | Coordinated change to qualify |
|---|---|---|
| `helas_call_writers.py:2001–2023` | Emits one `DCMPLX(LoopMass)` argument for each line. | Emit paired DP and MP model masses, using the existing model-selected real or `CMASS_...` name. |
| `loop_exporters.py:1528–1548` | Explicitly omits MP arguments for optimized output; generates `M2L=M**2`. | Supply paired formal names and a separate QP mass-square assignment with the same cyclic propagator order. |
| `loop_optimized/helas_loop_amplitude.inc:26,38,103` | Shared `LOOP_n` has DP mass arguments and DP `M2L`. | Keep those DP values; add `COMPLEX*32` MP mass arguments and `MP_M2L`, squared in QP. |
| Same template, lines 155/168/176 and 193/207/215 | Both interference and amplitude-reduction branches pass DP `M2L` to QP Ninja, NinjaRS, and TIRust. | Pass `MP_M2L` on all these QP routes, retaining the existing DP routes. |
| `loop_optimized/CT_interface.inc:849,908–914` | `MP_NINJA_LOOP` accepts DP squares and promotes them. | Change its generated mass formal to QP; retain a private QP copy for reordering. |
| `loop_optimized/NINJARS_interface.inc:166,200–202` | `MP_NINJARS_LOOP` has the same narrowing boundary. | Accept QP squares and preserve QP through ordering, invariants, and transport. |
| `loop_optimized/TIRUST_interface.inc:110,132–133,215,235` | Both QP tensor diagnostic and QP contraction entry accept DP squares. | Update both generated formals and their call sites together. |

The shared-call detail prevents a misleading one-line MP fix.
`loop_exporters.py:3579–3616` generates and stores `loop_CT_calls` once.
`helas_calls_split.inc:53–54` includes both DP and MP model symbols, so that
shared helper can supply paired masses. `mp_compute_loop_coefs.inc:522` reuses
it for amplitude reduction; for interference reduction the main
`loop_matrix_standalone.inc:1397` calls it after MP coefficient construction.
`turn_to_mp_calls` is used for coefficient and counterterm generation at
`loop_exporters.py:3075–3087`, not to create a separate MP reduction-call family.
The historical `mp_compute_loop_coefs.f` contains no `M2L` or `DATA M2L` object:
the relevant squares are runtime assignments in the shared `LOOP_n` routines.

There is a useful nonoptimized precedent, but it is not a complete solution:
`helas_call_writers.py:1475–1478` already emits paired DP/QP mass arguments, and
`loop/helas_loop_amplitude.inc:20–21,45–47` stores MP masses for numerator use.
That older path still constructs DP `M2L` and calls the legacy CutTools entry.
Changing optimized generation must not accidentally change that older ABI.

Complex masses require their existing model convention. `loop_helas_objects.py:
421–435` selects `CMASS_<mass>` where the complex-mass model's mass and width
symbols are not `ZERO`; this is not a test of their runtime numerical values.
`core/base_objects.py:2533–2555` distinguishes a card-supplied
mass, for which it constructs `sqrt(m**2-i*m*width)`, from a derived mass with
its own complex expression. The preserved CMS export's
`Source/MODEL/mp_intparam_definition.inc:63–79` computes the pole masses directly
in QP from MP masses and widths. Its `mp_coupl.inc` declares real MP masses and
widths separately from complex MP pole masses. The proposed reduction path
should square that generated MP complex mass. Promoting a DP `CMASS` would
retain its DP square-root/expression error; inferring a mass by square-rooting
an already rounded DP mass square is also incompatible with the MP numerator.

External momenta remain real and use the existing real particle-mass repair.
They must not be forced onto a complex pole shell. `SET_MP_PS` at
`loop_matrix_standalone.inc:2367–2389` and the off-shell `*` handling should remain
unchanged. Agreement of the clamp target with the external repair mass is the
stable-real, on-shell case; it is not a requirement that every internal complex
pole mass equal an external real mass.

The existing kinematic helper already has the required QP types:
`CT_interface.inc:245–248`. Its mass and massless substitutions at lines
277–297 remain intentional. `MP_SWITCH_ORDER` likewise already has QP masses
(`TIR_interface.inc:413–450`). The adapter boundary, not either helper, forces
the current DP square. The QP external APIs for Ninja (`mninja.F90:349–365`),
NinjaRS (`mninjars.f90:162–200`), and TIRust (`mtirust.f90:296–314`) already accept
QP complex squared masses. This proposal needs no precision or ABI change to
those external APIs. Direct callers of the generated MadLoop QP wrappers,
including tests and diagnostics, must be updated because their Fortran formal
types would change. The shared complex-mass availability classification should
also be checked against the mass representation actually dispatched.

CutTools is a separate limitation. `CT_interface.inc:25–29,114` and
`vendor/CutTools/src/cts/cts_cutroutines.f90:69–76` expose DP shifts and masses,
then the vendor promotes them for QP at lines 298–306. Passing a QP array to
that existing symbol is invalid. A distinct, separately qualified vendor QP
entry is required for complete QP denominator transport; the private CMS
prototype is not a production qualification. COLLIER's current interface is
also DP (`COLLIER_interface.inc:31–33,61–65,264`) and must retain that contract.
No broader change to legacy DP tensor reducers is implied.

The answer to the original causal question is therefore qualified. A literal
mass-call-expression-only change cannot safely repair the path. Carrying the
existing MP mass and its QP square consistently to the three QP-capable modern
adapters removes the identified DP-square intermediate for this top fixture.
Even squaring a promoted *unsquared* DP top mass would remove that specific
intermediate, but it is not the general solution for derived or CMS masses.
The parent saved-data calculation reduces the threshold matrix/shift mismatch
from about `8.8448e-13` to `7.68e-30 GeV²` by symbolic target replacement. That is
not an actual replay of the matrix helper or a prediction of the changed scalar
integrals. Remaining binary128 shell, prefix-sum, invariant, and conservation
roundoff can still be amplified by the extremely ill-conditioned geometry.
Neither higher-rank agreement nor automatic full-amplitude acceptance follows.

One separate inconsistency remains in the scale expressions: the QP Ninja and
NinjaRS templates square `MU_R` before promotion (`CT_interface.inc:1033`,
`NINJARS_interface.inc:253`), while TIRust squares after promotion
(`TIRUST_interface.inc:169`). That should be recorded and qualified separately;
it does not explain the identified squared-top-mass gap. No threshold, native
guard, stability policy, numerator option, or physical input is changed here.
