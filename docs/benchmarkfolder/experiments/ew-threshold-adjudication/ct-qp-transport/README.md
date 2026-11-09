# Private EW CutTools QP denominator transport replay

This diagnostic changes only the QP denominator-momentum transport for the
frozen two-point `e+ e- > t t~ a [virt=QED]` stream. The original-policy failures
remain recorded. It is not an accuracy or performance acceptance run.

The replay completed and the provenance validator passed. QP denominator
transport alone **does not resolve** this EW threshold discrepancy at the
original scalar cutoff. The ordinary point agrees with original CutTools to
`1.12e-14` relative or better. At the threshold point:

| Raw amplitude coefficient | Original CutTools | Private QP denominator transport |
|---|---:|---:|
| Born | 8.41000139800201356e6 | 8.41000139800201356e6 |
| Finite | 6.00226600087822632e11 | 6.02020348035903320e11 |
| Single pole | 1.45908285529699115e4 | -1.42959026708562553e8 |
| Double pole | -2.02026081349496999e4 | -2.02026081403520875e4 |

The finite coefficient changes by `0.00297955` using
`abs(a-b)/max(abs(a),abs(b))` (zero for two zero values), and the single pole
becomes highly discrepant. This diagnostic metric differs from the campaign's
`abs(a-b)/abs(a+b)` metric. The double pole agrees
with the historical Ninja/NinjaRS values within `3.62e-16` relative. This
isolates transport sensitivity without establishing a correct finite result.
The worker emitted 14 existing unstable-contribution warnings across its full
initialization/warmup/measured execution. Both measured points still return
unchecked status 120 and accuracy -1; those codes certify no numerical
stability. Every observed value and warning is retained.

The original `ML5_0_CTLOOP` DP entry is retained byte for byte. Seven generated
dispatch sites route `CTMode>=4` through a private QP copy receiving `MP_PL`;
DP initialization continues through the original entry. The private
`CTSXCUT_QPINPUT` symbol takes QP shifts and loads its QP denominators directly
from them in normal/reversed order. Masses, scale, numerator coefficients,
OneLOop scalar code and its original absolute `1e-6` cutoff are unchanged.

Input policy: exclusive backend 1, `CTModeInit=1`, `CTModeRun=4`,
`UseQPIntegrandForCutTools=.TRUE.`, `OSThres=1e-8`, original QP repair, and the
frozen worker's disabled forced stability checks. The reduction card retains
its historical resource-directory location; no cwd reduction card shadows
the other MadLoop resource files. The physical parameter card remains in cwd.
The exact two-point DAT stream is preserved.

The ignored clone is `.tirust-build/ew-threshold-ct-qp-transport`. Existing
numerical objects and archives are preserved except `CT_interface.o`; a private
`cts_cutroutines_qpinput.o` is linked before the unchanged CutTools archive.
The worker driver is unchanged. All external static archives are checked
against the frozen compilation manifest. An inert historical TIRust DSO
snapshot is linked for symbol resolution; the exclusive CutTools path calls no
Symbolica or TIRust API.

Reproduction from the repository root in a fresh matching environment:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/ew-threshold-adjudication/ct-qp-transport/run.py prepare
python3 docs/benchmarkfolder/experiments/ew-threshold-adjudication/ct-qp-transport/run.py build
python3 docs/benchmarkfolder/experiments/ew-threshold-adjudication/ct-qp-transport/run.py run --cpu 3
python3 docs/benchmarkfolder/experiments/ew-threshold-adjudication/ct-qp-transport/analyze.py
```

Native work requires coordination with the shared benchmark lanes. This
CutTools-only replay uses CPU 3, avoiding the queued CPU 2 diagnostics and the
CPU 9 EW bootstrap. The build timeout and worker protocol-response timeout are
180 seconds. The helper refuses to overwrite existing artifacts; failures must
be preserved before a separately recorded retry. An independent read-only
audit checked the unchanged DP routine, seven guarded dispatch sites and
private CutTools delta. The validator rechecks inputs, source patches,
numerical objects, archives, cards, runtime policy and preserved raw outputs.
