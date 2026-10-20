# Allocation-only callback experiment

This package implements the fixed plan in `PLAN.md` / `plan.json`. Compilation,
untimed trace preparation and timing are separate coordinated actions. Production
sources and the installed archive are never modified. Both binaries have compiled;
the untimed trace passed and all children exited. No timing result exists.

`capture.rs` wraps the unchanged original generic tensor. The original writes the
outputs returned to the reduction engine; the allocation patch writes a separate
buffer. Exact stored rational values, precision, sign and finite state must agree
before serializing each callback's requests and original outputs. All 11,136 final
ASCII fields must also match the previously retained uninstrumented capture. The
initial uncompiled wrapper draft returned the equal allocation outputs instead;
that superseded source and its preparation hash are retained in
`preparation-drafts/` and have never been used for measurements.

The input parser deliberately mirrors the frozen production/replay behavior:
parse each 40-digit binary128 record at 113 bits, then widen its exact rational
value to 256 bits. That rational conversion canonicalizes negative zero. This is
an existing transport property, separate from this allocation-only patch. Neither
the parser nor the fixed numerical patch is changed here. Callback trace values
are captured **after** that existing input transport; their decimal roundtrips
must preserve their actual value, precision and sign exactly, including any
negative zero produced inside the engine. A separate transport repair, if adopted,
requires its own provenance and refreshed correctness evidence.

`kernel.rs` imports only the tensor implementations, numeric types and request
types. The timed dispatch routine can call only the two `Numerator` callbacks.
Coefficient binding, output allocation, trace parsing, exact comparisons and
serialization occur outside the timer. The original and allocation tensors reuse
contexts by topology/rank across the full ordered stream and all blocks. One
untimed warm traversal per arm precedes exactly 21 alternating adjacent pairs.
Twenty-one separate control blocks measure the same dispatch and clock pairs
with no tensor work; their median is never subtracted.

Each build attempt retains its exact source snapshot, Cargo output, toolchain,
flags and binary hashes. `prepare.py` and `build.py` rehash every consumed frozen
source against the original correctness manifest. Binary products remain under
`.benchmark-work`; `kernel-symbols.txt` retains the actual symbol inventory for
review of license/engine/master references after compilation. The crate shares
the pinned dependency graph for reproducibility. The separate pure binary has no
amplitude, geometry, cut-provider, `oneloopmaster` or `SymbolicaMasters` symbols,
and its source does not construct those objects. It **does retain Symbolica
licensing symbols and state** through the pinned graph; it is not a license-free
binary. Symbol presence alone does not establish a runtime call. The timed source
dispatch was independently reviewed to call only the tensor callbacks. Do not
infer that the entire dependency graph is absent from the executable.

Build attempt 1 produced the pure kernel, then failed on a missing `SingleFloat`
trait import in the new capture wrapper. Attempt 2 added only that import and
compiled successfully; all source snapshots and raw compiler logs are retained.
The retry recopied unchanged source timestamps and consequently relinked the
kernel unnecessarily; this was redundant compilation, with no numerical change.

The original-driving trace passed all 1,392 bridge records, 86,436 callback complex
outputs and 11,136 final ASCII fields. Its 1,017,144 serialized Float values each
passed the bound Rust rational/precision/sign roundtrip assertion. Independent
offline review also verified every record, final ASCII field, metadata count and
hash; it does not claim to reconstruct hidden pre-serialization Float bits in
Python. The raw trace is 119,315,946 bytes, SHA256
`7a8cea5904b2bb665be87f5079cbf094713607e4ad25ca7c6fecef6e3c0977a5`.
Only its 27,040,351-byte gzip is retained here; regeneration uses the capture
command below and decompression reproduces the raw byte hash. The raw file and
executables stay in the disposable work directory.

Commands, from the repository root (run only after explicit coordination):

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/prepare.py
python docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/build.py --authorized-build
python docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/run.py capture \
  --build docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/builds/attempt-N/result.json \
  --authorized-run 'Explicit exclusive MP-slot handoff and context'
python docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/run.py kernel \
  --build docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/builds/attempt-N/result.json \
  --trace-result docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/traces/attempt-N/result.json \
  --authorized-run 'Explicit quiet-window handoff and allowed background'
python docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/analyze.py \
  docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/timings/attempt-N/result.json \
  --output docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-performance-v1/timings/attempt-N/summary.json
```

Preparation is one-time and refuses an existing disposable work directory. Build,
trace and timing attempts receive new numbered directories; failures remain
retained. All commands inherit or set CPU 212. The 100,000-resample paired bootstrap
uses Python `random.Random(20261020)`, resamples complete pair indices and uses
linear-interpolated 2.5/97.5 percentiles. This estimator is fixed before any data.
All 21 primary pairs and every output are mandatory; no slow sample is discarded.
Kernel timing cannot demonstrate a full MadLoop speedup or pass its acceptance gate.

The prospective execution-affinity addendum selects CPU 234 after the original
CPU 212 and the first alternative survey failed the host-idleness requirement.
`survey-all-1.json` retains all allowed singleton CPUs except 207 and all three
three-second intervals; CPU 234 had busy fractions 0.997%, 0.997%, and 0.333%.
Selection uses the smallest maximum busy fraction, then sum, then CPU number.
No numerical timing samples preceded this choice. The original plan and executed
trace remain unchanged. `execution-affinity.json` binds the new kernel-only
`run_cpu234.py`, its exact diff, the host survey, the frozen build/trace, and
`analyze_affinity.py`. Use these two new scripts for timing and analysis after
the explicit quiet handoff; their interface matches the commands above.
`affinity_addendum.py` independently recomputes eligibility and checks that the
controller changes only action restriction, affinity, and provenance validation.

The fixed series completed on CPU 234 in 310.066 seconds. All 42 timed blocks
retained all 1,392 records and 86,436 complex outputs, exactly matching stored
values, tracked precision and signs. Original median callback time was
7.240937685 seconds per complete block; allocation-only median was 6.571593567.
The ratio of medians is 0.9075611271 with paired bootstrap 95% interval
[0.8982879334,0.9127498167]. The unsubtracted timer-control median is
0.0007009% of original callback time, below the declared 1% gate. No pair was
excluded. This demonstrates an isolated callback gain on the old-bound parser;
it neither changes production nor establishes a full MadLoop speedup.

`build-inputs.json` is the retained pre-compilation preparation snapshot. Its
capture-source hash predates the missing-trait import correction; both build
attempts retain their exact source snapshots, and the successful
`builds/attempt-2/result.json` is the authoritative compiled-source binding.
The first compile failure, unexecuted preparation drafts, and failed host-only
preflights remain visible. Python bytecode, executables, and the 119 MB raw trace
are excluded from retention; the deterministic 27 MB trace gzip is retained.
`verify.py` checks the complete file seal and independently reruns the frozen
row/provenance/statistical analyzer without executing numerical binaries.
