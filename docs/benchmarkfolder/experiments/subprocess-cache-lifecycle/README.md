# Generated subprocess cache lifecycle qualification

**PASS.** Attempt 5 completes the actual generated `u u~ > z [virt=QCD] @101`
and `d d~ > z [virt=QCD] @202` lifecycle protocol using the qualified TIRust
`eb3d67fd…` shared library and `0c86349a…` Fortran archive. Production code is
the private export of commit `059bb9c979194bfda783cad83443223def8df280`.

Four fresh workers compare cache disabled/enabled and plain/observed links.
Each has two warmup calls, 19 measured physical A/B/A calls, and 24 explicit
DP/QP tensor-key calls. The frozen analyzer and an independent raw-token
audit find exact coefficient, status, accuracy, input and IEEE-flag parity.
Every physical result has status 219, using native DP with a separate
stability recheck; the largest reported accuracy is `1.2269907815852879e-15`.
QP is exercised by the explicit generated tensor diagnostics, not by a
physical amplitude rescue in this experiment.

Both observed workers record two context generations and both subprocesses
register their N=3, rank=2 plan in each generation. Every measured physical
call crosses the real automatic native entry and exit clear boundaries.
The explicit seed has one entry/144 bytes before the next entry clear and
zero afterward. Final clears encounter populated caches in all 19 enabled
calls and empty caches in all 19 disabled calls. Recreation creates the new
context before its automatic entry clear. Separate mass, scale and shift
key tests observe five cold requests and seven repeats per precision;
disabled QP requests record 12 bypasses. All 67 native evaluations in each
observed worker succeed.

The immutable result is [attempt5/results.json](attempt5/results.json), with
the compact report binding in
[qualification-summary.json](attempt5/qualification-summary.json) and the
independent source/build/live-artifact/raw-data review in
[independent-review.json](attempt5/independent-review.json).
[DESIGN.md](DESIGN.md) describes the frozen protocol and observer.

This qualifies the generated automatic cache boundary and cross-prefix
lifecycle for these two QCD Born subprocesses. It is not a timing acceptance
result, an EW or loop-induced qualification, a poles-only test, or a rank-six
test. The generated tensors here reach N=3/rank=2. The QP key inputs are DP
values promoted to QP; a nonzero binary128-only input perturbation is outside
this experiment. The Born host tensor-cache clear is dummy; the observed
TIRust native clears are real.

## Retained failures and amendments

All failed attempts remain identifiable; no failed result was replaced by
a successful retry under the same identity.

| Attempt | Outcome and next narrowly reviewed amendment |
|---|---|
| 1 | Make query could not find `/bin/bash`; bind the installed Bash explicitly. |
| 2 | Plain link succeeded, then the provenance gate rejected an unbound compiler-selected linker script; bind its exact retained-map closure. |
| 3 | First worker stopped in the model reader before `LC_OPEN` or an amplitude because a bare card path sought `/ident_card.dat`; use explicit card directories and exact sibling ident cards. |
| 4 | Card-only nominal read passed; the alternate fixture's long number exceeded the reader's `CHARACTER*20` value width. No physical worker ran. |
| 5 | Shorten that single numeric token without changing 92 GeV; validate every numeric token, pass the three-read smoke, then complete all four workers. |

Attempt 4 compiled only two private harness drivers and relinked against the
same 33 numerical objects. Attempt 5 reused those exact binaries with no
compilation. The original CLI's model/DHELAS bootstrap occurred before the
second generated-source freeze; its exact artifacts are separately bound by
[cli-bootstrap-provenance.json](cli-bootstrap-provenance.json).

## Portable replay

From the MadGraph repository root:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle/verify_evidence.py
```

This verifies the final regular-file index, decompresses and hashes the
retained text linker maps, replays the unchanged analyzer and smoke checker,
and validates the five invocation envelopes and their source/build/runtime
bindings. It uses no compiler, reducer, live installation or absolute input
paths. `verify(read_bytes=...)` accepts an immutable report reader, allowing
report regeneration to bind the same frozen bytes.

Actual native commands, environments, caps and object/link inventories are
retained under each attempt's `execution/` directory. Native runners are
exclusive-write forensic protocols, not commands to overwrite this completed
evidence in place. The final physical execution used CPU 10, the common
numerical lock, a 30-second card-only smoke and four separate 180-second caps.
All five final commands returned zero without timeout or interruption.

`evidence-index.json` lists only portable regular files. Earlier `../…`
symlink spellings are inert metadata bound to their included regular target
bytes. Two large attempt-3 raw maps remain optional local originals; their
lossless gzip copies and raw/gzip hashes are included. Compiled binaries stay
in ignored working directories and are not part of this documentation.
