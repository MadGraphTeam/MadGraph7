# Causal replay of the CMS invariant projection

Ten isolated calls through the original installed MTIRUST Fortran adapter and
TIRust shared library establish that the captured invariant projection causes
four reproduced refusals. For CMS points 20 and 21, the stock rank-two,
four-propagator request returns status 1 in both DP (mode 1) and QP (mode 4),
with `singular Gram matrix for [1, 2, 3]`. Replacing only the symmetric real
entries S(2,4) and S(4,2) with their captured `RAW_DIFFSQ` words gives status 0
and an empty error in all four cases. The two stock point-1 controls also give
status 0. Shift vectors, masses, scale, every other invariant word, library,
registered plans and evaluation options remain identical within each pair.

This is a controlled input intervention, not a generalized Fortran fix. No
tensor coefficient buffer was inspected, and no amplitude was evaluated by
this replay. Successful statuses do not establish independent integral or
amplitude accuracy. The original projection policy and its consequences for
all topologies, physical on-shell identities, precision rescues and complete
amplitudes still need a qualified production implementation. Performance and
stability qualification are also pending.

## Exact matrix distinction

Two exact rational diagnostics are retained and recomputed from the binary
input words. The generic three-by-three Gram uses zero-based reference 0 and
indices 1,2,3 with `G(i,j)=(s(i,0)+s(j,0)-s(i,j))/2`; its determinant has units
GeV^6. It is not the Gram matrix named in the error.

For the reported subset `[1,2,3]`, TIRust uses origin 1, basis 2,3 and the
unscaled two-by-two matrix `H(i,j)=s(1,i)+s(1,j)-s(i,j)` (no factor one-half).
In these inputs `H=[[0,t],[t,2*t]]`, where `t=s(1,3)` in zero-based notation
(the Fortran S(2,4) intervention). Thus `det(H)=-t^2`, in GeV^4. The stock
stress matrices are identically zero; the raw24 interventions have nonzero
determinants. A zero first diagonal entry alone is not evidence of a singular
matrix: the ordinary and intervened matrices also have that zero diagonal entry.
The reference formula is attributed to TIRust revision
`854b79dcb7fffcb35e92b4a8a3dd443369b798c4`, `crates/iregi-rust/src/direct/exec.rs`,
whose recorded SHA256 is
`4053f5d3a24002829328f75bac5aae3d26f04764f841a8edbaaa0ef938836c81`.
The retained analysis records the relevant source regions; this packet does
not reread the reference checkout.

## Source and execution identity

The source proposal is sealed by
`ba130c540e466550ad2e4fa560b0b8df75a93912c4d1b271ef4d60c5fd4571a9`.
Its historical `source_only` and admission flags are preserved verbatim; the
later, separately reviewed build and run receipts record execution. The
closed build is `7b829cd965971060f21bf0938c8f1aeb42516d85ac3e36f851df9907bf47283f`;
the closed run is
`3a8e13486d4cd26425efefaceb7d6f2099be0a867a80e22fdb00fc3ff1e678d8`.
The installed adapter archive is
`0c86349a4fd0dadd918476731e899529ce1c7b17f5892a4af2c22dd762250c1a`;
the original shared library is
`eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9`.
No alternative library was built or loaded.

The packet retains 26 proposal payloads and its index, three closed build
commands, twenty closed run commands, all ten raw worker transcripts, twenty
READY/DONE maps, ten input files, per-worker loader admissions, exact input
certificates, protected-state receipts, root reviews and exact Gram analyses.
The original adapter archive, shared library, executable and other native
bytes are represented by recorded identities only, not included here.
The standalone executable's six-provider runtime set is deliberately distinct
from the full-amplitude fifteen-provider set in the earlier build packet.

Every call used a fresh context, the captured plan-registration sequence,
full finite evaluation, all poles, no cache and no stability-check pass.
The fixed stage and worker caps, nonexclusive CPU 12, 4 GiB memory cap,
thread settings, commands and environment are preserved in the proposal and
execution receipts. All workers exited with OS code 0; four returned the
expected numerical refusal status 1. Protocol completion, unchanged input
words, full error frames, loader maps and cleanup were checked separately
from numerical statuses. Recorded elapsed times are controller accountability,
not a benchmark.

## Portable verification

Keep these sibling packets beside this directory:

- `../current-kinematic-trace-build-qualification-1`, seal
  `89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033`.
- `../current-kinematic-trace-runtime-qualification-1`, seal
  `730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4`.

Run from the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/kinematic-projection-causal-replay-1/verify.py
```

Only Python's standard library is needed for portable verification; the
`source` command selects the pinned development Python on this host. The
verifier reads only this packet and its declared siblings. It verifies stored
and decompressed identities, source pins and closed receipt joins; replays
only the sealed in-memory protocol parser; independently reconstructs the
word inputs and certificates from captured traces; and recomputes both exact
Gram definitions. It does not execute retained controllers, invoke a compiler
or loader, inspect `/proc`, read external binary paths, evaluate an integral,
or consult live installations. The recorded original file checks are
historical evidence, not fresh verification of current external files.

`capture.py` documents the one-time explicit closed-origin selection and
deterministic gzip retention. It is not part of portable verification and
must not be used to discover or capture later executor runs. Native replay
requires separate admission and is not performed by these commands. The
sibling ordinary/stress evidence remains unchanged, including its six failed
stress contribution gates; this isolated replay does not replace those
failures with amplitude success.
