# Empty-prefix NLO batch regression

The archived `c5496ca9b148b16a931214e1d541ead100ed1aad` integration
build produced `CALL MP_TERMINAL_WL_1_1(...)` in an NLO coefficient batch
without importing `IEEE_EXCEPTIONS` or declaring `WL_ENTRY_INEXACT`. The
standalone path used a nonempty process prefix and compiled successfully.
The complete failed integration export and log remain separately retained in
`policy-v2/install-export-abi3-terminal`.

The per-batch detection expression in `split_HELASCALLS` used
`MP_\w+TERMINAL_WL_...`. FKS uses an empty process prefix, so the detection
discarded the required declarations even though the terminal call and
`IEEE_GET_FLAG` had been emitted correctly. The production correction changes
only that repetition to `\w*`.

`reproduce.py` binds the original method snapshot to the archived revision,
then executes that method and the isolated proposed correction. Both generate
three batches: an unaffected batch, a terminal batch and another unaffected
batch. Actual compilation uses `-fno-automatic`. The original prefixed case
passes, the original empty-prefix case reproduces the missing-name errors,
and both corrected cases pass. The unaffected batches retain their original
procedure boundaries. Sources and compiler diagnostics are retained here.

The formal unit regression covers empty and nonempty prefixes, batch-local
imports/declarations and the same compiler flags. The focused suite contains
13 passing tests. Only the four FKS `mp_coef_construction_1.f` golden files
were updated, adding the required import and logical declaration. The two FKS
IO checks are recorded in `io-check.log`. No polynomial arithmetic, private
guard or reduction behavior changes in this correction.

The fresh archived standalone/NLO integration build after the correction is
a separate validation. This namespace does not claim that an earlier failed
integration build passed, or that the synthetic batch compile replaces it.
