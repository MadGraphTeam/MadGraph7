# Private CMS invariant-certificate amplitude qualification

This packet preserves the first controller failure and the successor's closed
build, ordinary and stress observations for the CMS fixture
`e+ e- > mu+ mu- [virt=QED]`. It compares original and candidate Fortran with
explicit CutTools (1) and TIRust (9), DP mode 1 and QP mode 4. The original
TIRust shared library `eb3d67fd…` is unchanged. The candidate is an isolated
Fortran prototype; this packet does not promote it into production or qualify
a general MadLoop fix.

The ordinary point-1 run has eight eligible workers, sixteen outer amplitude
entries and sixteen READY/DONE map captures. Original/candidate CutTools
observations are exact matches. Original/candidate TIRust raw coefficient and
state rows also match exactly. Against CutTools the maximum recorded relative
errors are `3.01378215278465383e-13` in DP and
`8.73560044285410722e-16` in QP, satisfying the ordinary `1e-8` target.

The full stress stage is incomplete and exits 1. Its eight point-20 workers
completed, then the ninth worker (original CutTools, DP at point 21) exceeded
the 300-second measurement protocol timeout. That process was terminated with
signal 15 and reaped; the remaining seven workers were never launched. There
is no full `stress-comparison.json`. The separate root point-20 analysis uses
only the eight completed workers and cannot substitute for the full stage.

The completed point-20 subset does not qualify the candidate. Original TIRust
DP and QP and candidate TIRust DP retain nonzero software failure flags and
W03 warnings; their coefficient buffers are excluded from arithmetic.
Candidate TIRust QP clears those flags but disagrees with both CutTools QP
controls, with maximum relative error `1.23772904322100718`. The candidate
comparison gate is therefore false. Original/candidate CutTools control
observations agree exactly. An independent reducer must adjudicate this
disagreement before any numerical fix can be accepted.

Every retained failure remains a failure. `candidate_gate_passed` is a separate
comparison result and must not replace the stage exit status or make an
ineligible original worker numerically valid. Failed TIRust words
are retained as opaque evidence and excluded from coefficient arithmetic.
No retry, alternative TIRust library, timing claim or broad process-matrix
acceptance follows from these runs.

## Scientific and source scope

The private change feeds a source-derived conservative spacelike certificate
through six generated Fortran caller/interface files and five support sources.
It uses structural routing metadata, integer error bounds and environment
checks; no point-specific input substitutions are introduced. The same cards,
physical inputs, helicity/color sums, precision modes and original library
pins remain bound in the source and worker specifications.

All ordinary-run software stability flags are zero; stress failures remain
explicit above. Agreement here is unchecked fixed-mode evidence. It is not qualification of the normal
host stability/rescue path, Monte Carlo integration, production cache behavior,
all topologies/ranks or all requested processes. Controller and worker elapsed
times are accountability records, not benchmark measurements. Precise runtime
acceptance remains a separate local task for Hua-Sheng.

The 269-source generation/routing inventory and source-DAG derivation are
historical, hash-bound evidence from the declared sibling packets. This
portable packet does not duplicate or freshly parse all 269 original source
files. It authenticates the captured inventories, the factory metadata and
copied support source bytes. Any replay of the six candidate transformations
is limited to the retained, hash-bound donor reconstruction; it does not
establish a new general exporter derivation.

## Preserved failure and successor identity

The predecessor source is
`286ecfe5f5fab199dee34e4401cb5154ba1212870eaf2b921d3f44536c251b0e`.
Its closed attempt
`99cbb544826b741c030708ce4ab5bc01db83eac0ec49c4dce02a6be6858d76aa`
failed with `KeyError('size')` before any native command. Both protected-state
records and the exit-1 terminal review are retained unchanged. It provides no
candidate compilation or numerical conclusion.

The successor source is
`ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74`.
It adapts `bytes`/hash records to the inherited `size`/hash prerequisite API at
that boundary, preserves the original `bytes`/hash snapshot contract, and uses
a new executor directory. The 26 scientific/build payloads named in its change
proof remain byte-identical to the predecessor. Historical source-only flags
are preserved; later independent stage closures record execution.

The successor build is
`8cab371e9affd5297c0f48c30184361959d20f3d3a48c6dbd6109b86bf49bacb`.
It closes 129 owned commands and two loader queries, with no tensor or amplitude
call. The original/candidate builds have 147/151 prebound dependencies and
52/56 actual recorded link loads. Both use fifteen runtime providers. Records
preserve 206/201 reused objects, seven existing modules and four archives;
protected before/after snapshots match. Original binary bytes are not retained.

The ordinary closure is
`b364bd8535d0f5850cb5a21390c5678cbb1c1dd8fddfae6f0174cf50d261f7a7`.
The incomplete stress closure is
`c696421eb4948b4d2de5993cc84455b1ac04d0f659b245341729714f91ae8482`.
Its actual outcome is bound by retained root/terminal reviews. Each stage had separate admission. Closed
ownership records establish cleanup and lock release at those historical
boundaries; portable verification does not inspect live processes today.

## Portable verification

Keep these declared sibling packets beside this directory:

- `../current-kinematic-trace-build-qualification-1`, seal
  `89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033`.
- `../current-kinematic-trace-runtime-qualification-1`, seal
  `730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4`.
- `../caller-bound-native-qualification-1`, seal
  `51e1ba911d1bbe19c83141777f76fc967b0c876630fa3cef60f818adda205174`.

From the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/kinematic-invariant-amplitude-qualification-1/verify.py
```

Only Python's standard library is needed. Verification authenticates every
stored and decompressed byte, source index, declared dependency and closed
receipt join. It reparses raw driver-state/driver-bits/output files and
warnings, checks worker cards/options, runtime-map identities, backend/precision
usage, and reproduces the complete comparisons using exact rational values.
The FIFO READY/DONE/BYE protocol is retained as records in `result.json`, not
as a raw FIFO transcript; those records are checked against observed counts
and output files. The verifier does not claim raw FIFO transcript replay.

The source comparator is reused through a restricted in-memory path adapter
that reads only authenticated retained payloads. Its controller, external
source verification and native entry points are never executed. No compiler,
loader, scientific library, live process, original binary pathname or network
is used. Reported artifact hashes represent historical checks; this is not a
fresh verification of external binaries.

`capture.py` documents deterministic text retention from explicit sealed source
indices and CLOSED stage payload maps. It performs no active-run discovery.
The first failure, refused workers, root reviews and source limitations remain
available verbatim. Production promotion, checked-host stability and runtime
acceptance remain open.
