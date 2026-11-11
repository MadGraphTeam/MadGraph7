# Exact geometry postcondition: standalone qualification

Both compiled variants agree with the independent Fraction/capacity oracle on
**1,698 cases across all20 environment cohorts:67,920 observed helper calls**.
There are no skipped cohorts. Per variant,10,520 accepts and23,440 conservative
refusals match expectations, and66,320 transported INTEGER128 input words join
the frozen fixtures. The checked and optimized raw transcripts are identical.

The unchanged mathematical helper checks exact dyadic equality using guarded
signed128 arithmetic. It has no floating-point/environment dependencies; its
optimized object references only the integer division helper. The checked build
also uses integer overflow traps and Fortran bounds checks. All57 BUILD commands
pass. Actual linker LOAD records, bound PT_INTERP listings and initialized
worker maps retain the five-provider standalone runtime. Both phase owners
close without surviving processes.

Explicit driver-only controls exercise rounding, individual/all sticky flags,
individual/all halting masks with no pending flags, and FTZ/DAZ. The observed
rounding/status/mask/MXCSR/x87 fields are unchanged across each helper call and
restore exactly. All formatting occurs after restoration; failed restoration
would use a raw integer-only C write/exit. These20 cohorts are not an exhaustive
test of every possible floating-point environment or complete FPU register state.

The four retained CMS20/21 examples use conditional nearest-even reconstructions
of old q-component words. They do not observe an owned current-point snapshot
or actual q words at physical use. This packet qualifies the standalone integer
helper and recorded cohorts; **owned snapshot lifetime, physical geometry,
contracted amplitudes, final physical dot arithmetic and performance remain
unqualified**. It changes no production source or library.

All frozen scientific source, fixture/expectation generators, BUILD/RUN text,
maps, commands, raw observations and independent/root reviews are retained.
Nine excluded ELF/object/compiler-module identities are recorded; no binary is
packaged. Larger text uses deterministic gzip. The root review receipt preserves
two initial pure-script mistakes and their corrected replay; neither was a
native or scientific-source failure.

Portable replay authenticates every owned payload and nested source seal,
regenerates all1,698 independent Fraction/capacity expectations and replays every
input echo, answer and environment row. Numerical replay uses preloaded capsule
data with external filesystem, process, native-library and mutation APIs blocked.
A documented I/O-only substitution supplies the already authenticated frozen
reference module to the corpus generator; no arithmetic is replaced. Runtime
identity checks replay recorded LOAD/PT_INTERP/maps receipts and never open live
libraries, query a loader or execute a reducer. Historical absolute paths are
inert provenance keys.

```sh
python3 -B replay.py
# Optional binding to the independently recorded capsule index:
python3 -B replay.py --index-sha256 EXPECTED_INDEX_SHA256
python3 -B test_replay.py
```

Run from any directory while retaining the complete capsule. Replay writes
nothing. The tests additionally create disposable relocated/corrupted copies
and verify their rejection. Historical controllers and review scripts are
provenance only; execute `replay.py` for portable verification.

Precise performance acceptance remains Hua-Sheng's local TODO. The separately
frozen benchmark PDF and existing NinjaRS report are unchanged.
