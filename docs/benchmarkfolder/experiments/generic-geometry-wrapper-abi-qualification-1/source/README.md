# Phase A: emitted provenance-wrapper/module ABI fixture

SOURCE/CONTROLLER ONLY. Nothing in this packet has been compiled or run natively.
The sealed source3 parent is
`c99ce0ffb80a1dabe90a109cc958305efabe1af7968b97ec0adc65a16b009dc2`.

The fixture compiles the exact emitted ML5_0_LOOP_3_PROVENANCE routine from
source3's actual MG object-generated Z geometry fixture, the exact generated
invariant_bounds module/data/C token reader, and the actual ALOHA_OBJECT module
extracted byte-for-byte from generated ALOHA source. Complete original source
snapshots and extraction coordinates are retained. The module's source-window
gates are true for this Born geometry; global whole-producer support remains
false. The source fixture has NSQUAREDSO=0: this is deliberately a direct ABI
fixture, not a complete CLI physical amplitude.

A test-only recording provenance seam preserves the emitted legacy call and
argument order. The legacy target is a recording stub, with no tensor, scalar
integral, numerator, reducer, physical amplitude or scientific DSO. It records
chord bounds, classes, availability, passed token, exact integer/mass argument
words and the two independent DP/QP W memories. The driver establishes its
requested IEEE settings, snapshots actual flags/halts/rounding/control tokens
before and after the emitted wrapper, then restores the original environment.
Recording occurs in memory; I/O is outside that measured call window.

The deterministic corpus has exactly240 calls:12 distinct-slot positive cases,
124 additional sticky masks (all32 masks across modes1/2/4/5 overall),44 direct
freshness/route/WE/token/nonfinite rejections,12 actual nonnearest controls,
20 single-halting controls,4 invalid modes and24 zero/subnormal/precision-tail
controls. The integer-only reference predicts180 available-bound cases and60
conservative rejections. QP payloads include the last binary128 significand bit;
no expected value passes through binary64. A Python-generated golden protocol
is only parser-test data, never native evidence.

Proposed native stages, after root's source/build/run review:

```sh
python3 -B .tirust-build/generic-geometry-wrapper-abi-proposal-1/control.py build --index-sha256 SOURCE_INDEX_SHA
python3 -B .tirust-build/generic-geometry-wrapper-abi-proposal-1/control.py run --index-sha256 SOURCE_INDEX_SHA
```

Use the pinned GNU15.2 paths in index.json and the pinned Python3.13.12 environment.
Build uses seven serial compiler/link commands,90seconds total with15seconds
reserved for finalization; runtime uses one240case process,15seconds total with
5seconds reserve. Both use4GiB address space,512MiB file limit and core0. No CPU
is reserved and no timing/performance claim is made. There is no automatic retry.
Work directories are one-shot and failures remain recorded.

The owned-group supervisor is copied unchanged from the successful standalone
caller-bound fixture. The mock-only map adapter waits at a READY/FIFO barrier,
records and hashes every observed file-backed ELF before releasing one RUN line,
and checks those files again after process cleanup. PT_INTERP is parsed directly
from the built ELF and must match the reviewed loader. No LD_* or inherited
compiler/library override enters the child environment. The mapping proof is a
single pre-case snapshot, not continuous monitoring or kernel-vDSO provenance.

The source/protocol mocks check exact extraction, frozen corpus/reference,
complete rows, flags, token/slot/bound/class/mass/word/forwarding corruption and
strict helper-only scope. They do not qualify the Fortran parser or compiler.

**Full caller lifetime coverage remains false.** A direct FRESH=false argument
does not execute SKIP_LOOPNUM, MP_DONE, rotations, FILTER_SO or narrowing. The
concrete larger preparation is in [PHASE-B.md](PHASE-B.md). Neither phase-A
success nor its source tests enable a scientific correction or production
promotion. Complex-mass amplitude truth, generic checked rescue/cache coverage
and precise local performance remain separate open requirements.
