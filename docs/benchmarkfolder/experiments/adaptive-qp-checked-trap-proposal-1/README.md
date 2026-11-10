# Private adaptive QP checked/trap proposal — source only

This packet is prepared for root review. No compiler, linker, loader, native
worker, numerical lock, installation, or benchmark was run during preparation.
Only file reads/copies, source/specification writes, pure ELF metadata decoding,
and synthetic Python tests were performed. `build` and `run` each require
separate root admission; neither is authorized by source preparation alone.

The question is whether the old private genuine-MP-mass implementation with
the adaptive TIRust bridge passes **MadLoop's independent automatic stability
checks**, and whether the same complete physical evaluation survives enabled
invalid/divide-by-zero/overflow traps. It uses the same two saved ordinary and
threshold points as the completed fixed-QP replay. This is **not** the current
`invariant-symmetry-1` source campaign. Its numerical objects, kinematic policy,
and old mass implementation stay fixed. No adaptive installation prefix is
promoted, and no production, canonical campaign, report or reference file is
changed.

## Exact build isolation

One free-form replacement main is compiled and one executable is relinked.
The relink is mechanically derived from the frozen `link-guarded-qp` command:
replace its unique `check_sa.o` with the new main, change the output path, and
append a private linker-map flag. All **25 non-main numerical/observer objects**,
all archives, all libraries and the remaining link flags stay in their
original order and retain exact hashes. The driver declares only the scale
COMMON that its original main uses; those `DOUBLE PRECISION MU_R, ALL_MU_R` and
`COMMON/RSCALE/ MU_R, ALL_MU_R` declarations are checked against frozen
`coupl.inc`. No fixed-form model include is interpreted as free-form source.

The new main uses `-O0 -fno-fast-math -frounding-math -fsignaling-nans` and
runtime IEEE control, so every worker uses the same executable. Existing
numerical objects retain their historical compiler flags. Consequently this
main-only build cannot establish whole-program IEEE conformance or prove
every MPFR/libquadmath internal operation raises hardware exceptions.

The original bridge remains exact `eb3d67fd…`; the adaptive bridge remains
exact release `c5de24e6…`. Both complete hashes, literal ELF metadata, source
ancestry and native dependencies are in `prepared.json`. Two copied private
runtime directories contain the same 14 DSOs except for the bridge; the
original interpreter makes **15 runtime ELF objects**. The adaptive bytes are
copied under the executable's required `libmadloop_tirust.so` name. A filename
does not establish byte identity: direct loader listings and actual live
mapping snapshots must match the complete allowlist and hashes.

## Physical and IEEE protocol

The eight original resource/input files are frozen under `fixtures/`. The
MadLoop card changes exactly two values: `CTModeRun: 4 → -1` and
`NRotations_QP: 0 → 1`. Its inverse patch must reproduce the original card
byte-for-byte. Backend9 alone, threshold `1e-6`, DP rotations0, CTModeInit1,
finite/pole settings, cache controls and genuine-QP-integrand controls remain
unchanged.

`CTModeRun=-1` runs normal direction checks after initialization, with a genuine
QP rescue when the DP check fails. QP modes4/5 and the additional QP traversal
are observed separately. `FORCE_STABILITY_CHECK(.FALSE.)` retains normal
checks; it only disables an extra wrapper rerun of unknown initialization.
The old bootstrap and alternating warmup order are preserved, bounded by1002
calls. Two consecutive T=1/2 returns mean initialization finished. They do not
mean numerical acceptance; an H4 result remains a recorded failure.

The driver writes a separate strict control stream with **named** invalid,
divide-by-zero, overflow, underflow and inexact flags, their support and
halting states, and rounding. It records startup, policy configuration,
SETPARA, preprocessing and amplitude boundaries, and exit. Sticky flags are
not cleared. Observation-local IEEE status may be restored only around the
observation helper's own I/O. The existing software reduction-failure latches
remain separate from hardware flags and are never reset by this driver.

Four fresh physical workers run original/adaptive bridges with masked/trapped
policies. Trapped workers enable invalid, divide-by-zero and overflow before
SETPARA; underflow and inexact remain nonhalting. Three additional independent
fault controls perform actual volatile `sqrt(-1)`, `1/0`, and `HUGE*2`
arithmetic. Each must reach its flushed fault marker with the requested modes
and terminate by SIGFPE. A survivor, unsupported mode or different signal
fails that control. These controls prove the main's hardware policy, not the
precision or exception semantics of every library operation.

All seven workers wait at a deterministic mapping acknowledgment before
numerical work. The controller acknowledges only a complete actual ELF map.
The driver's inner wait is at most20 seconds; a shorter outer fault-worker
cap may terminate it and records a failure without retry. No timing result
uses this diagnostic barrier.

## Parsing and acceptance boundaries

The additive parser keeps repeated model snapshots, mode4/5 traversals,
native failures and raw slot shapes. A trapped final prefix is retained but
never qualified. The default physical process has200 groups per traversal;
small synthetic fixtures use an explicit separate group-count override.
Unknown records, incomplete complete-pass claims, incorrect cache/check
stores, wrong dimensions, missing raw slots, dropped/capped records and
observer-induced flag changes fail structural validation.

A checked candidate output requires public status219 or323, the independently
recorded H/T/U values, finite nonnegative accuracy below `1e-6`, complete
finite/pole output and no W03 omissions. A DP-only success is a valid checked
DP observation but leaves adaptive QP coverage open. A QP success additionally
requires observed full mode sequence `[4,5,4]`. The old journal does not label
rotation identity; the packet does not invent such a label or a per-call
working precision. No success is called a512-bit call.

The candidate's masked/trapped workers must preserve all returned amplitude,
accuracy and public-code bits and all corresponding raw tensor/contraction
results. Physical/model/native input bits and Born slots1/5 are compared
separately. Hardware/event flags and software latch values remain explicit
observations, never mislabeled as masses or momenta. Different original versus
adaptive pass counts remain coverage differences, not changed-physical-input
claims. Original eb3d refusals or exceptional results are valid retained
controls and do not count as candidate numerical acceptance.

The checked/trap coverage field is set only after strict process completion,
all seven actual mappings, exact immutable runtime/input closures, expected
output allowlists, all fault controls, both candidate checked-QP observations,
IEEE policy preservation and candidate output parity have passed. A timeout,
mapping failure or modified dependency prevents acceptance. SIGFPE itself is a
retained scientific outcome with complete process ownership, not a successful
physical result. Independent physical truth, precision convergence and
performance qualification remain separate and false in this diagnostic.

## Bounds, retention and commands

The single compile/relink attempt has a shared180-second cap. The single
runtime attempt has a shared600-second cap including validation, setup,
loaders, workers, scientific posthashing and terminal publication. Each of the
four physical workers is capped at120 seconds; each of the three fault
controls and two loader listings is capped at10 seconds. Execution uses
nonreserved CPU12, one owned process group at a time and4GiB address space.
The common numerical lock is nonblocking; exit75 does not consume a runtime
attempt. There is no retry, CPU reservation, automatic threshold tightening,
or fallback to a different implementation.

The reviewed owned-process supervisor defers interruption through creation,
kill/reap and ownership release; its CPU binding is explicitly adapted to12
and its deferral includes the shared deadline signal. Interrupted checkpoint
temporary files are retained under unique names. A missed deadline rescinds
all completion/coverage acceptance; only an administrative failed receipt is
published afterward. Every partial output, exception, return code, signal,
mapping snapshot and source/hash failure remains in the ignored attempt tree.

From the repository root:

```sh
source tools/tirust_env.sh
python3 -B -m unittest discover -s docs/benchmarkfolder/experiments/adaptive-qp-checked-trap-proposal-1 -p 'test_*.py'
python3 -B docs/benchmarkfolder/experiments/adaptive-qp-checked-trap-proposal-1/controller.py verify
# Only after separate root review and admission of the indicated stage:
python3 -B docs/benchmarkfolder/experiments/adaptive-qp-checked-trap-proposal-1/controller.py build
python3 -B docs/benchmarkfolder/experiments/adaptive-qp-checked-trap-proposal-1/controller.py run
```

`prepare` is the one-shot pure source/specification freeze already used for
this packet. No build1 or run1 directory forms part of source qualification.
