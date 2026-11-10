# One executed checked/trap diagnostic: masked QP passes, traps remain open

The adaptive bridge passes MadLoop's independent automatic checks with
exceptions masked. It does **not** qualify for the combined trapped-evaluation
gate: both original and adaptive libraries terminate by SIGFPE during the
threshold point's second warmup call, before observed-point capture.

| Worker | Recorded outcome |
|---|---|
| Original, masked | Ordinary point219, accuracy `5.432914922319511e-13`; threshold point420, accuracy1;111 QP conditioning refusals and2 W03 diagnostics |
| Adaptive, masked | Ordinary point219, same accuracy; threshold point323, accuracy0; all600 QP reductions served, zero W03 |
| Original, trapped | SIGFPE during warmup2 on threshold point2; no observed-point journal |
| Adaptive, trapped | SIGFPE at the same stage; no observed-point journal |
| Invalid fault control | Requested halting modes observed; deliberate `sqrt(-1)` terminates by SIGFPE |
| Divide-by-zero fault control | Requested modes observed; deliberate `1/0` terminates by SIGFPE |
| Overflow fault control | Requested modes observed; deliberate `HUGE*2` terminates by SIGFPE |

The ordinary point is accepted in DP; it supplies no QP coverage. The threshold
point has three complete200-group QP traversals with observed mode sequence
`[4,5,4]`. The original serves163 groups and refuses37 in each pass; the adaptive
bridge serves every group. Check-pass arguments are false/true/false across
those traversals, and caching is enabled for all600 calls. Two genuine-QP
model snapshots are retained. The journal does not explicitly label the
rotation; it does not report per-call working precision.

All489 previously served raw UV1/IR1/IR2/finite tensors and contracted group
results remain bitwise identical. Their independent raw-record digest is
`20a33da98d358622564c86190f9df075ec3102aefc5601040817393eb1ef375e`.
All101,086 selected physical-input/Born records agree, digest
`f436936f4c3479e685a0d7dd44e79d9a3ca12cecd1ce8b8eeb5ddbec13fdd209`.
Software failure latches and hardware event flags remain separate observations.
The111 former refusals receive explicit per-pass before/after records.

## Shared DP trap localized without rerunning the calculation

Captured maps, return PCs, exact ELF identities and offline symbolization
place the first fault inside **both bridges' existing DP path**:

```text
iregi_rust::direct::exec::gram::cayley_cancellation::<f64>
State::dp_checked
ml_tirust_evaluate_dp
__mtirust_MOD_tensor_dp
ml5_0_tirustloop_
```

The fault instruction is `DIVSD %xmm1,%xmm0`, bytes `f2 0f 5e c1`, at ELF
virtual address `0x12029f5` in original eb3d and `0x1203815` in adaptive c5de.
The instruction window is identical. The pinned reference source computes
`terms / size(&det_x)` and only afterward checks whether the result is finite.
The complete source witness, exact mapped-file hashes, PC-to-file translation,
tool arguments, symbol outputs and disassembly are retained. `addr2line` and
`objdump` inspected files only; no artifact was executed or debugger attached.

Named IEEE observations immediately before each trapped call show invalid,
divide-by-zero and overflow flags clear, with those halting modes enabled.
Masked execution first sets divide-by-zero after warmup2 at point2 in both
libraries; the adaptive execution also sets overflow during that call.
Inexact first appears after SETPARA and underflow after the ordinary bootstrap;
both remain nonhalting. These boundary observations do not capture faulting
register values or `siginfo`. The divide instruction is established; naming
the precise hardware exception at that instruction remains an inference.
The trap is reached before the observed adaptive QP evaluations and is not
evidence of an adaptive-QP-specific fault.

## What this qualifies

`masked_adaptive_checked_outputs_qualified` and masked QP three-pass coverage
are true. The original combined checked/trap field remains **false** because
the candidate trapped worker cannot complete, and masked/trapped output
parity is unavailable. All three positive fault controls pass. All seven
owned processes are reaped, actual15-file runtime maps match, inputs and
dependencies remain unchanged, and both admitted stages complete inside
their recorded caps. `execution_complete` describes that terminal transport;
it does not turn a SIGFPE into physical acceptance.

Independent physical truth, explicit rotation identity, per-call256/512-bit
provenance and runtime performance remain unqualified. Administrative wall
durations only establish consistency of the recorded180/600-second caps;
they are not benchmark measurements. This is the old private genuine-MP-mass
candidate, separate from `invariant-symmetry-1`. No adaptive prefix is promoted.
COLLIER and its reported cache issue are unchanged.

## Portable evidence and reproduction

This additive packet retains the original source freeze, two-command build,
seven-worker execution, all raw logs, partial trapped control streams, maps,
snapshots, acknowledgments, derived reports and original failures unchanged.
UTF-8 text is stored in content-addressed deterministic gzip payloads.
Native/executable/archive/module/dependency files are represented by identities
and sizes, not executable payloads. Recorded native hashes are provenance;
portable verification does not reopen historical absolute native paths or
claim to remeasure past wall time.

`analyze.py` reparses raw logs with the frozen Python parsers and computes
with fresh integer-key dictionaries. JSON normalization is used only when
comparing derived artifacts. It preserves missing trapped journals as recorded
failure outcomes. Journal flag words remain opaque; named driver observations
supply flag names. `verify.py` checks seals, source/build ancestry, exact25-object
isolation, native/link/runtime identities, output closures, original failure
receipts, raw-record preservation, checked accuracy and the false trap gate.
It performs no native evaluation, library loading or external-tool execution:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/adaptive-qp-checked-trap-proposal-1/qualification-1/verify.py
```

The first pure capture attempt rejected a symbolic native dependency path.
Its failure, source-reconstruction lineage and partial content-addressed
payloads are retained under `capture-draft1/`. The corrected capture records
the resolved identity for hash-only dependencies and creates no symlink
payload. This bookkeeping correction changed no source-freeze, build, run,
numerical result or native execution. There was no retry of the experiment.
