# Owned snapshot module: standalone qualification

The unchanged Fortran snapshot module passes **37 cases in all 20 explicit
floating-point environment cohorts**, using checked and optimized builds.
The two transcripts are byte-identical. Together they contain 1,480 observations,
3,680 module API outcomes and 94,720 complete INTEGER128 state words. Every
observed environment field is preserved and restored; no cohort is skipped.

This unit calls the real reset, prepare and commit APIs with separate DP and QP
buffers. Its matching-width TRANSFER operations and complete output words check
DP zero extension and binary128 storage, including signed zero, subnormals,
finite extremes, low QP bits and nonfinite refusal. The 27 bounded scenarios
cover pending publication, commit/reset/duplicate behavior, false genuine-QP
selection, malformed legs and views, dirty storage, invalid pending words and
separation of the two precision contexts. Partial invalidation expectations
describe only the slices actually passed. They do not extend to inaccessible
storage or unrecorded intermediate states.

The module compiles with `-O0 -fcheck=all -ftrapv` and separately with
`-O3 -fno-fast-math`; both use the same checked driver. All 57 recorded BUILD
commands succeed. The executable definitions uniquely bind the six real module
APIs. Recorded linker LOADs, actual PT_INTERP listings and initialized worker
maps bind the same five-provider runtime. Separate BUILD and RUN owners close
within their recorded 180-second windows, retaining a 60-second cleanup reserve,
4 GiB address limit, nonexclusive CPU12 and no surviving owned processes.
No constructor, ALOHA, geometry helper, q, dot product, reducer or amplitude runs.

The component analyzer deliberately keeps `analysis_only` true and its runtime
qualification fields false: parsing text alone cannot attest execution. This
capsule combines that exact replay with authenticated closed receipts, compiled
API definitions, executable identities and runtime maps to verify the bounded
observed unit behavior. It does not change those component flags into broader
qualification claims.

Physical constructor completion, canonical current-point origin, q-at-use
lowering, final-dot arithmetic, full caller/numerator lifetime and performance
remain unqualified. The module has no runtime precision tag. Distinct buffers
and matching precision APIs remain caller obligations; the unit does not
establish safe DP/QP interchange on the same buffer. It does not qualify the
`2f07cecc…` composite-input prototype. Only its small ancestry receipt is retained,
without bundling the CMS source family. Scientific dispatch remains disabled.

The packet retains the exact 72-payload controller source and its 31 unchanged
parent payloads, fixtures, expectations, all BUILD/RUN text, raw observations,
maps and six independent/parent reviews. Native executables, objects and compiler
modules are excluded with their identities retained. Larger text uses
deterministic gzip. The root's corrected pure-review PHASE omission and the
retention inventory's corrected SHA-sidecar assumption remain explicit history;
neither was a scientific-source or native execution failure.

Portable replay authenticates all text and reconstructs the frozen corpus and
integer reference expectations. Only import/file bootstrap is supplied from
the authenticated in-memory sources, and generated text is captured in memory;
the corpus and reference algorithms are unchanged. Arithmetic replay runs with
external filesystem, process, library-loading and mutation APIs blocked. Runtime
checks replay recorded identities without opening live libraries. Historical
controllers and review scripts are inert provenance.

```sh
python3 -B replay.py --index-sha256 EXPECTED_INDEX_SHA256
python3 -B test_replay.py
python3 -O -B test_replay.py
```

This capsule changes no production code, default, benchmark PDF or NinjaRS report.
