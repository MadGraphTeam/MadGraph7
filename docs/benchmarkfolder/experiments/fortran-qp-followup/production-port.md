# Conditional production port of the private terminal updater

This is a design proposal. The prototype remains isolated pending the quiet
full-process measurements and independent review.

The optimized HELAS writer can establish the producer/consumer graph directly
from all loop diagrams before emitting calls. Each loop wavefunction supplies
its output number, loop mother and rank pair; each loop amplitude supplies its
final wavefunction root. Eligible outputs are roots with no wavefunction
consumer anywhere in the complete graph. Reject nonunique producers, aliases,
missing predecessors, unsupported rank/layout combinations and unknown
consumers. If the proof is unavailable, emit the existing full calls.
Recheck `number == me_id` for producers, mothers and amplitude roots rather
than relying only on optimized relabeling. Reject a non-loop wavefunction with
a loop mother. Bind every emitted updater to its proved producer and exact
operands, so a custom call writer cannot qualify solely from the object graph.

Pass the proven terminal set explicitly to the multiple-precision coefficient
call writer. Do not alter the cached generic wavefunction call functions:
double precision and other callers must continue to receive the complete
updater. Replace only the updater call attached to an eligible wavefunction,
then leave the existing split-order skip statements and fixed-form labels
untouched. Check the resulting generated graph against the intended terminal
set and the diagonal-only CREATE/MERGE contract.

Add a private-updater mode to the existing expanded/compact polynomial writers.
Reuse their arithmetic generation and sparse handling. Change only the measured
diagonal loop bound and the conservative admission guard. Emit the existing
public full routines unchanged and call them on every rejected private path.
Retain exact binary128 recognition, component exponent bounds, nearest rounding,
already-set INEXACT, disabled INEXACT halting, dimension limits and full clearing.
The bounded classifier must use association-only scratch for `-fno-automatic`.

Only affected MP coefficient batches need the IEEE exception import and local
entry-flag variable. `split_HELASCALLS` emits a separate batch even when
`writer=None` concatenates it into the returned source; the parent
`MP_COMPUTE_LOOP_COEFS` need not gain a new IEEE procedure boundary. Born and
counterterm batches sharing `mp_helas_calls_split.inc` must remain unchanged.

Validation must compare the generated helper arithmetic with the isolated
prototype, retain the compiled bit/flag boundary fixtures, and test malformed
or shared graphs falling back to full calls. Actual exports must exercise
ordinary QCD, mixed orders, loop-induced amplitudes and the FKS
`-fno-automatic` build. Final process checks compare all Laurent/order rows,
accuracy and rescue status against the unchanged original and private
prototype. A QCD benefit must accompany an all-DP EW control without material
regression; no partial or failed stream can be promoted as acceptance evidence.
