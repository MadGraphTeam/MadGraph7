# Parent instruction intervals remain unqualified

The source2 `momentum_window` proof traverses ordered producer-batch bodies and
their reachable CALL/FUNCTION graph. `source_inputs` records the complete
loop_matrix bytes, but recording bytes is not effect analysis: the batch roots
exclude parent instructions between the entry capture and producer batches.
The finalizer's ENV_WRITE regular expression is not itself an applied proof.

A parent/control helper could change rounding temporarily between batch calls
and restore it before the wrapper's token check. The start/end token would
match while a W.P producer had run in a different mode. Therefore neither
endpoint equality nor the current batch-only source windows can establish the
full construction interval. This is a proof gap to close before scientific
admission, not a claim that such a setter exists in the retained generated SM
callers.

Likewise, parent instructions after the last batch could overwrite a momentum
component or whole W object. Those instructions are outside the current
batch-suffix alias proof. A full admission needs both entry-to-producer effect
closure and producer-to-consumer no-overwrite closure, including parent paths.

Phase A remains safe within its stated scope. Its exact emitted wrapper receives
synthetic FRESH/token histories from a recording driver. It does not execute
SLOOPMATRIX lifetime paths, nor change any invariant or call a reducer. Its
240-case ABI result cannot qualify the missing parent interval.

Before phase B can admit a full geometry window, derive and bind the actual
ordered parent instruction intervals from entry capture through all necessary
W.P producer calls. Analyze every inline operation, CALL and expression function
on those intervals, including control helpers, aliases and paths across batch
boundaries. Unsupported effects and temporarily changed environments must
reject admission conservatively. Add source counterexamples with parent-only
setters/restoration and parent post-batch momentum writes, then observe actual entry→producer→use execution and all
negative lifetime paths under the real compiler/ABI. Keep whole-producer purity
false unless independently established.

The included pure diagnostic uses the unchanged source2 parser on synthetic
Fortran text: batch-only analysis succeeds, while parent-root closure rejects
the setter. No Fortran or environment setter executes. It demonstrates the
scope distinction and does not qualify any physical caller.
