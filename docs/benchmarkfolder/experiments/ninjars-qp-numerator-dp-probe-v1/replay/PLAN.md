# Stored-input replay preparation — execution held

This standalone diagnostic consumes the sealed capture manifest
`03d63cee3b8c38ee1936f9662e816206f3b99f3a2ec55b8cc938c209ff8aefbd`.
It does not modify MadLoop, its retry controller, its thresholds, or any return
code. Preparation/build is allowed only outside a coordinated quiet window;
engine execution requires a separate explicit MP-slot handoff from root.

Each of the 464 complete semantic keys is replayed in captured order, without
selecting loops or dropping failures. The eight cells are fixed:

| Cell | Coefficients | Geometry/masses/scale | Reducer |
|---:|---|---|---|
| 1 | Original DP words | Original DP words | Native DP |
| 2 | Original DP words | Original DP words | Compensated DP |
| 3 | Exactly promoted original DP | Exactly promoted original DP | Generic 256-bit |
| 4 | Captured QP rounded once to DP | Original DP words | Native DP |
| 5 | Captured QP rounded once to DP | Original DP words | Compensated DP |
| 6 | Exactly promoted rounded-QP DP | Exactly promoted original DP | Generic 256-bit |
| 7 | Full captured QP | Exactly promoted original DP | Generic 256-bit |
| 8 | Full captured QP | Full captured QP | Generic 256-bit |

Input records use integer hexadecimal words. The driver checks binary64 and
binary128 layout, uses the unchanged installed ABI3 Fortran module/archive,
asserts round-to-nearest, and echoes the actual arguments immediately before
every call. This permits an independent integer check of every cell's original
geometry, exact promotion and single rounding. It checks all input words again
after each call; input mutation is a hard diagnostic failure. Coefficients are
already in the wrapper's NinjaRS convention: no further parity or routing change
is permitted. The supplied S matrices are used verbatim.

The complete raw status and four complex output fields are retained for every
cell, including nonzero statuses and nonfinite words. A failed result is never
treated as an oracle. Interpretation requires cells 1, 2 and 8 to reproduce their
corresponding successful captured native, compensated and QP output words/statuses
exactly for all 464 keys. This is a stricter diagnostic reproduction check than
a physics tolerance; it establishes that the standalone calling sequence and
state reuse reproduce the retained controls. Failure stops interpretation,
without deleting the record or selecting a passing subset.

One module context is retained across the entire fixed stream and finalized at
the end. No warmup or hidden adaptive retry is added. The maximum is 3,712 engine
calls and 900 seconds for the child process, on CPU 2. Preserve partial output,
stderr, exit code and elapsed time on timeout/failure. Timing is diagnostic only.

Generic calls retain the installed bridge's actual 256-bit arithmetic, binary128
input packing with 40 significant decimal digits, and on-shell threshold
`1e15 * 2^-112`. Native/compensated DP retains its native provider's `1e-9`
threshold. Both conventions are unchanged, and these cross-domain differences
are a confound even when the supplied geometry matches exactly. The rational
split is recorded but is not compared as a cross-engine physical observable;
all three full Laurent coefficients are compared.

The frozen runtime also has the separately identified signed-zero transport
defect: it parses at 113 bits, then widens through a rational, losing negative
zero signs inside the MP bridge. This replay retains that historical runtime
exactly, including the defect. Its input echo proves arguments at the Fortran
module boundary; it does not claim that the current MP bridge preserves their
zero signs internally. A corrected widening library is a separate runtime and
must not silently replace the captured archive or these reproduction controls.

The captured QP coefficients were formed at the improved MP external point,
with MP parameters and Born interference. Cells 4–7 therefore combine a changed
QP coefficient bundle with the original DP geometry. They do not isolate
numerator arithmetic at fixed external kinematics and cannot establish a valid
mixed-precision physical amplitude. No amplitude reassembly, direction averaging,
new stability policy or production rescue is part of this diagnostic. Full QP
and forced CTMode 4 remain genuine in production.

Preparation must rehash the complete capture seal, all consumed native assets,
Fortran module/header/source and link-order metadata against the captured runtime.
The replay build uses copied assets in its own work directory. Bind the compiler,
exact command, executable and resolved dynamic libraries before any launch.
The parent capture seal intentionally excludes this `replay/` namespace.
