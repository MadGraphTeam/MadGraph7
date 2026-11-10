# Materialized standalone tensor complement — source review only

**No compiler, linker, generated executable, or Rust routine has been run.**
The two model source sets, observer, schedule and analyzer are implemented.
They require review before compilation. Nothing in this directory authorizes
native execution. The frozen parent proposal remains unchanged.

`materialized/ew/` and `materialized/qcd/` contain the exact private Fortran
sources, includes and frozen input copies. The current campaign objects are
referenced read-only, not rebuilt. `build-plan.json` lists every compile/link
argument and object path. The commands are JSON data, not an executable build
script. Compile output paths are under `.tirust-build/generated-rank6-poles-1/`.
There are five small Fortran compilation units and one C observer per model,
then one link. Existing numerical objects number 37 (EW) and 24 (QCD).

The preparation script imports production Python rendering functions only;
its synthetic plan declarations request tensor capacity, not physical HELAS
amplitudes. All four actual kinematic/order helper bodies are copied with
prefix changes only. `CAP6_INITCOLLIER` uses the actual initialization body
with a private rank-six header. The only observation inserted into the
production COLLIER template is a read-only integer-copy hook after invariant
construction and before its first numerical call. Exact mapping generation
uses the production exporter and `FortranPolynomialRoutines` methods.

The runtime schedule contains 84 base TIRust calls, 18 cache/check/budget calls,
three actual group capture calls, and 21 COLLIER adapter calls. Capture is an
actual generated `LOOP_6` invocation with a unit scalar test numerator; it is
never presented as a physical amplitude. Fixtures101/102 use the original
generated adapter at ranks 4/5. Fixtures 103–107 use the separate CAP6 adapter.
Every call copies PL/M2L from immutable fixture storage. Both entry and exit
words are recorded, including the expected direction reversal.

The observer forwards exactly once. It retains actual DP bytes or actual QP
64-byte decimal ABI records, native status/error, precision/mode/cache/check
flags and IEEE state. The observer has fixed buffers of approximately 120 KiB
and performs no allocation, I/O or floating-point arithmetic inside numerical
wrappers. Fortran-controlled flushes write one JSON object per scheduled action.
Overflow, repeated native evaluation in one action, schema anomalies and
observer changes to flags/rounding fail closed. The callback observing COLLIER
copies its actual complex momenta, cyclic invariants and squared masses.

The Fortran record format is versioned and ordered:

* `HEADER version model original_capacity`.
* Three total `CAPTURE action ordinal stable capture_count` frames (two EW,
  one QCD), each with PI/MI inputs, all V coefficient rows, and CAPTURE_END.
* Each CASE contains its 11 schedule integers plus coefficient count; SCALE
  contains a binary64 word. Each PI/PO row has leg index and 8 binary128 words;
  each MI/MO row has leg index and 4 words, representing exact promotion of
  the actual DP squared masses. CACHE_PRE/POST each contain 9 aggregate and 6
  QP counters. STATE holds entry flags/rounding, exit flags/rounding, stable,
  COLLIER error flag and COLLIER accuracy flag. V rows contain coefficient,
  Laurent slot, four binary128 words and the two corresponding decimal values.
  END contains action and observer health.
* FINAL contains clear/finalization status, generation before/after destruction,
  and the 15 empty-cache counters. DONE follows observer closure and reports health.

`analyze.py` requires exact coverage of this schedule, all coefficient rows,
input immutability/fresh copies, correct direction mutation, healthy observers,
native option/precision agreement, successful finalization and empty final
caches. It compares full/partial tensors, UV/IR reconstruction, selector-finite
invariance, analytical scaleless residues, scale-induced pole mixing, selected
cache values/counters and exact captured full tensor replay. The cached check
direction has a separately computed uncached reference. The parser uses
integer binary128 decoding and nearest-even 113-bit parsing of MP input records;
it never loads a numerical library.

Independent COLLIER comparisons first compare the *actual* effective inputs,
including cyclic invariant correspondence. Exact differences are retained as
`input_policy_mismatch`; no invariant is repaired or replaced. Consequently,
ordinary floating-point construction differences may leave an independent
comparison unqualified. No new sample is selected in response. Separate QP
comparisons require the same effective inputs too. The report retains
dimensionless, ordinary relative and symmetric residuals. Independent acceptance
uses MadLoop's `abs(a-b)/abs(a+b)` criterion and same-coefficient finite-relative
small-pole exception at 3e-6, and records the 1e-8 target separately. Full/partial
comparisons use the parent proposal's dimensionless 1e-10/1e-24 bounds.

Native and host refusals, nonzero bad IEEE flags and independent-input
mismatches remain open items. IEEE flags are observed, never cleared: an
inherited invalid/divide-by-zero/overflow flag therefore also leaves that
later frame unqualified. The diagnostic does not manufacture a checked
physical-amplitude return code. Current generated finite-mode dispatch plus
the separately pinned core constructor/arithmetic/scalar-root tests remain
the finite-work omission evidence; there is no new dynamic finite-operation
counter or instrumented Rust build.

The 17 tests in `synthetic-tests-2.txt` use fabricated text, bit patterns and
source strings only. They cover binary64→binary128 promotion, nearest-even
rounding, malformed/missing/extra records, duplicate JSON keys, observer IEEE
mutation, dimensional bounds, final context/cache failures and exact schedule
counts. The earlier 13-test log is preserved. They are not native qualification
or evidence that the uncompiled Fortran/C code builds successfully.

Read-only checks, from the repository root:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/verify.py --sources
```

After separate build/run authorization, use the exact `models.*.commands`
argument arrays from `build-plan.json`, in order, under their declared cwd.
Use the recorded compiler paths/environment, two compiler processes at most,
and a 240-second compile cap. The two numerical workers are serialized under
the existing numerical-test lock, on the CPU assigned by root, with 900 seconds
and 4 GiB address space each. No automatic retry or extra native call is allowed.
Use the recorded `runtime_library_paths`; preserve stdout, stderr, exit, timeout,
dependency hashes, link maps and every runtime file, including `param.log`.
If linking requires any additional model build, stop for a resource review.

Analyze retained output only (after authorization and actual execution):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/analyze.py \
  .tirust-build/generated-rank6-poles-1/ew .tirust-build/generated-rank6-poles-1/qcd
```

The current review package contains no numerical `results.json` and no claim
that the remaining rank-six generated capacity acceptance item is closed.
