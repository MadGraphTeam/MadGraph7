# Descriptor-free local scratch experiment

This is a separate isolated alternative to `../fortran-scratch/`. The public
diagnostic becomes a small recursive wrapper owning fixed local real/complex
buffers. It passes scalar buffer elements to a private worker's original
explicit-shape array arguments through Fortran sequence association. The worker
retains the original arithmetic, exact runtime leading dimensions, on-shell
handling, tensor mapping, failure checks, pole controls and cache policy. No
pointer descriptors or saved scratch are introduced.

Both wrapper and worker are recursive. The wrapper checks 1–6 propagators before
passing any storage, matching bridge v1. The seven argument sequences occupy
disjoint reserved intervals: real 1–24, 25–48, 49–72, 73–92, 93–128; complex 1–6
and 7–42. For one propagator, the zero-size denominator dummy receives the valid
element `REAL_WORK(73)` and accesses no elements. The wrapper never accesses its
scratch while the worker is active. All backing arrays are local explicit-shape
arrays with the same kind as the worker arguments.

`prepare_sequence.py` verifies the frozen original compiler recipe, injected
environment and static link inputs. The unchanged relink reconstructs the
original executable byte-for-byte before only `TIRUST_interface.o` is replaced
with the transformed source, using the original flags. The retained build
metadata, source diff and commands are in `adapter-sequence-build/`.

Static disassembly finds zero allocation, deallocation or packing calls in the
two wrappers and workers. At the unchanged unoptimized compiler settings, the
DP wrapper plus worker occupy 4401 bytes, compared with the original 4565; QP
occupies 4888 versus 5077. These are code-size observations, not runtime claims.
No production source, acceptance executable or common Fortran routine is edited.

The native layout fixture covers 1–6 propagators, ranks 0–2, primary/reversed
order and DP/QP, with bounds checking. It retains exact coefficient bits, mutated
input bits and stability flags. Supplemental verification checks all 72 case
headers, every expected hex record, success totals and actual loaded-library
identities before/after execution. Successfully served cases and refusals must
be distinguished. This array-layout fixture does not claim rank-six amplitude
qualification.

Reproduction from the MadGraph root (new output directories required):

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/n1-profile/fortran-sequence/prepare_sequence.py \
  --name adapter-sequence-build --build
python3 docs/benchmarkfolder/experiments/n1-profile/fortran-sequence/check_layout.py \
  --name layout-parity-1 \
  --library .tirust-build/n1-profile/split-bridge/libmadloop_tirust.so --run
```

The full-amplitude comparison must use the same frozen diagnostic split DSO
`1c3688011fb0b423142ab4ac4c832f5b71d8e4409ef323087dfac5ceb2075cd7` for both the
original and sequence-worker adapter. Its provenance is in `../bridge/`.
All numerical execution holds the shared exclusive lock. These experiments
remain performance-ineligible until a separately frozen acceptance campaign.

The native `layout-parity-1` result passed all 72 cases for each variant (144
records), with exact coefficient/input-mutation bits, all stability flags true,
no bounds-check failure and unchanged actual loaded-library identities.

The seven-pair full-amplitude diagnostic gave a variant/original runtime ratio
of 1.000726, with 95% interval [0.990139, 1.011427].
All amplitude coefficient bits and expected backend statuses agreed. The interval
does not establish a runtime benefit, so this variant remains unpromoted.
See `timing-summary.json` and the hashed sibling timing references in
`evidence-index.json`; these results do not supersede the failed qualified n1 gate.
