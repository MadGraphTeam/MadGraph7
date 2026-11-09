# Local scratch allocation experiment

This experiment changes only the generated TIRust adapter in an isolated copy of
the frozen `c91cb85f3aa942c90299b3518bb1f195f55334a5` QCD 2→1 process. It does not
change the production template, the frozen acceptance executable, common Fortran
routines, compiler flags, numerical expressions, caches, or the Rust library.

`transform.py` replaces seven automatic arrays in each DP/QP diagnostic routine
with two fixed local buffers. Contiguous pointer remapping retains each array's
actual runtime leading dimension. The adapter explicitly requires 1–6
propagators before associating the pointers, matching the existing bridge v1
qualification. Recursive routine declarations keep scratch storage local to each
invocation. The empty denominator array for a tadpole has shape `(4,0)` and an
empty target section. No saved scratch or cross-call state is introduced.

The real-buffer intervals are 1–24, 25–48, 49–72, 73–92 and 93–128; the complex
intervals are 1–6 and 7–42. These intervals hold, respectively, ordered prefixes,
shifts, the exact-N kinematic input matrix, the unused denominator scratch,
invariants, masses and the exact-N complex kinematic matrix. All previous
initialization and calculation statements remain. The raw tensor buffer and its
existing lower-rank section behavior are unchanged.

`prepare_scratch.py` verifies the original compiler recipe, injected compiler
environment and static link inputs against the frozen qualified n1 attempt. It
first reconstructs the original executable byte-for-byte with the unchanged
objects, then recompiles only `TIRUST_interface.o` with the same flags and relinks.
The generated source, patch, exact commands, object hashes and executable hash
are retained in `adapter-scratch-build/`. Its `static-audit.json` and disassembly
show zero allocation, deallocation or packing calls in both transformed
diagnostic routines, compared with seven allocation/deallocation pairs in each
original diagnostic. This is static evidence, not a runtime benefit claim.

`check_layout.py` builds separate original/scratch native drivers with Fortran
bounds checking. It covers 1–6 propagators, ranks 0–2, both primary and reversed
check directions, and DP/QP: 72 calls per variant. The dyadic massive fixture
comes from the pinned engine's existing pole tests. It records exact coefficient
bits, mutated input bits and stability flags. This test covers array layout;
it is separate from the whole-amplitude performance comparison and does not
extend the numerical qualification to ranks absent from the generated process.

Reproduction from the MadGraph root:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/n1-profile/fortran-scratch/prepare_scratch.py \
  --name adapter-scratch-build --build
python3 docs/benchmarkfolder/experiments/n1-profile/fortran-scratch/check_layout.py \
  --name layout-parity-1 \
  --library .tirust-build/n1-profile/split-bridge/libmadloop_tirust.so --run
```

Output directories must be new. The frozen diagnostic split-workspace library
has SHA-256 `1c3688011fb0b423142ab4ac4c832f5b71d8e4409ef323087dfac5ceb2075cd7`;
its separate source/build and native validation evidence is retained in the
sibling `bridge/` folder. Numerical execution takes the shared exclusive lock.
The later triads must use that same library for both adapter variants; comparisons
remain diagnostic until a separately frozen acceptance campaign is completed.

The native `layout-parity-1` result passed all 72 cases for each variant (144
records), with exact coefficient/input-mutation bits, all stability flags true,
no bounds-check failure and unchanged actual loaded-library identities.

The seven-pair full-amplitude diagnostic gave a variant/original runtime ratio
of 0.997481, with 95% interval [0.988260, 1.006787].
All amplitude coefficient bits and expected backend statuses agreed. The interval
does not establish a runtime benefit, so this variant remains unpromoted.
See `timing-summary.json` and the hashed sibling timing references in
`evidence-index.json`; these results do not supersede the failed qualified n1 gate.
