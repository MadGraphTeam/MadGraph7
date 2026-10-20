# Diagnostic sample of corrected-original QCD4 CutTools warmup

Root requested this ten-second, read-only sample of the already running
corrected-original reference worker, PID 3177493 on CPU 207. It belongs outside
the collector's sealed run directory. It is not an acceptance timing, complete
point-stream result, or current candidate profile.

The exact `perf record` invocation, process CPU/elapsed time and file descriptors
before/after sampling are in `sample.json`. `perf` attached with 99 Hz user
CPU-clock sampling and DWARF stacks, without stopping or changing the worker.
The executable hash was unchanged afterward. Raw recording, decoded stacks and
record/decode logs are retained. The collector's per-phase timeout was 7,200 s.

Of 997 samples, 989 (99.20%) belong to generated quadruple-precision numerator
construction. Binary128 multiply/add/subtract account for 431/356/115 leaf
samples. The principal updater owners are rank 3 × 1 (353 samples), rank 2 × 1
(214) and rank 4 × 1 (187). All three coefficient-construction batches appear.
The interval shows legitimate generated arithmetic rather than a spinning or
sleeping routine or a stalled scalar-master call.

The worker was still in initial warming before and after this interval. These
samples do not establish its phase-space-point progress or a reliable remaining
duration. No source, input, threshold, worker command or collector evidence was
changed. The short sampling overhead is part of this diagnostic reference run;
its timings were already explicitly excluded from acceptance.
