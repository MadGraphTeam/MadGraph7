# Fixed paired pure-tensor comparison — held

Measure only the original and allocation-only generic tensor callbacks. This
experiment cannot establish a full NinjaRS runtime gain or satisfy a MadLoop
performance gate. Production code and the installed library stay unchanged.

The prerequisite correctness package is frozen at identity
`51646828d605e97cd74d84d8b3d840619deeabf342a7f4c91a7b3e5c04df99dd`.
Its 1,392 successful captured binary128 inputs cover nine topology/rank shapes;
all 86,436 callback complex outputs and 11,136 final ASCII fields matched exactly.
The allocation patch is unchanged from that package. The NaN-source assignment
counterexample remains visible; retained-slot resets use finite zero/one only.

## Untimed trace preparation

The existing capture contains bridge inputs, not the actual expansion vectors
and numerator sample points. One additional untimed reduction replay is needed
to retain those callback requests. It requires the exclusive MP slot; no such
replay is currently launched or authorized to overlap another Symbolica worker.

Use the exact frozen input stream, all 1,392 records in order, without selecting
representatives or omitting failed records. A tracing wrapper around the unchanged
original tensor records each evaluation/expansion invocation, topology/rank,
coefficient binding, lane/sample/partition identifiers, expansion powers, actual
vectors/scalars and original expected outputs. The allocation-only tensor must
match each output's exact stored rational value, tracked precision, sign and
finite state during preparation. All eight final ASCII fields must still match
the original retained record. Bind the original archive/source, engine/provider
versions, the input capture, and every trace file by hash.

Retain each Float's precision and sign, plus enough scientific decimal digits
(`ceil(precision*log10(2))+4`) to reconstruct it exactly. Reparse at that stored
precision and require exact rational value, precision and sign equality before
accepting the trace. Actual captured inputs must be finite; a nonfinite input or
any roundtrip mismatch stops preparation and remains a failed artifact. This
finite-input assertion does not replace the separate strict exceptional-value
tests in the correctness package. Put a 2 GiB limit on the serialized trace and
fail rather than silently truncate if exceeded.

## Pure replay and timer boundary

Compile a separate pure replay binary using the retained original tensor source
and the exact allocation patch. It has no amplitude, cut, geometry or master
provider construction/calls. It loads and validates the complete trace before
any timing. Numerator instances are retained by topology/rank for each arm, so
scratch is warmed and reused as in the bridge. Reset numerical inputs and bind
the recorded coefficients for each bridge record outside the timer. Construct
all request views and preallocate output buffers outside the timer as well.

For each of the 1,392 bridge records, time its complete ordered callback sequence
with one start/end clock pair. Inside that interval execute only the shared
dispatch loop and the selected tensor's evaluation/expansion methods, writing
the complete output buffers. Exclude coefficient binding, parsing, cloning input
fixtures, correctness comparisons, formatting and output serialization. Preserve
the original order of callbacks, lanes, monomials, arithmetic and output columns.
Sum the 1,392 intervals for the block total; retain every individual interval.
Validate every output after that record's timer ends, including precision and
sign. No result is accepted from a block with any changed/missing output.

Run one complete untimed warm traversal for each arm. Then collect exactly 21
adjacent pairs: original/allocation for even-indexed pairs and allocation/original
for odd-indexed pairs. Both arms traverse the identical complete stream in the
same order. The pair count, ordering, inputs, compiler flags and validation
criteria are fixed before execution. Do not discard slow pairs or repeat until
a preferred result appears. The state is reused across blocks; this measures
warmed callback variability, not fresh-process initialization variability.

Retain a separate identical-dispatch/control-only clock-overhead measurement;
never subtract it. If its median exceeds 1% of the original measured callback
total, label the experiment inconclusive and prepare a separately versioned
measurement design. Do not silently change the timing interval or repetition
count. Source review must confirm the timer cannot reach any geometry/master
engine. An operation/function counter in an untimed validation build may provide
additional evidence, but is absent from the timed binary.

## Execution and interpretation

Proposed CPU: 212, whose current thread-sibling list is `212`. Recheck affinity,
availability and host activity immediately before the run, and coordinate a
quiet window with root. Keep the existing corrected-original reference collector
on CPU 207 unchanged. Record all allowed background work and host activity; a
confounded run remains retained and cannot demonstrate a gain. Pure replay uses
no Symbolica context, but still needs timing coordination to avoid contention.

Report original/allocation medians, all 21 pair ratios, dispersion and a fixed
seed 100,000-resample paired bootstrap interval for the ratio of medians. The
predeclared kernel-gain criterion is an upper 95% interval endpoint below 1,
with all exactness and provenance checks passing. A failure or an interval
crossing 1 is an unresolved or negative kernel result. Report timings by shape
as descriptive diagnostics using every record; do not select shapes for the
primary comparison. Keep any inferred full-runtime bound explicitly separate
from measured callback times. A successful kernel result would justify a later
isolated full-process test, not production promotion.

All new source, trace and logs go under this new retained namespace; disposable
sources/builds/executables go under the matching `.benchmark-work/experiments/`
namespace. Expected resources: one CPU, a cached thin-LTO build of roughly
6–9 minutes, one untimed MP trace preparation of roughly 1–2 minutes, then
roughly 2–8 minutes for the fixed pure replay series. These are planning estimates.
No preparation replay or timing starts until root coordinates it.
