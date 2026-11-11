# Private n1 Fortran DP profiling evidence

This text-only capsule is prepared for review. Nothing has been published or
promoted. It preserves source1's missing-make failure, the unchanged source2
scientific payload, closed BUILD/PARITY/PROFILE records, and independent audits.

Run `python3 -B verify_evidence.py` from this directory, or call
`verify(root, expected_index_sha256=...)`. An in-memory reader can instead pass
`read_bytes=reader`; its `paths` attribute must contain the exact physical index
closure. The callback receives only safe capsule-relative names. Historical
absolute paths remain provenance strings and are never opened by verification.

Eight full learning traces, four pre-timing parity gates and all 28 timed
batches are retained. Each batch contains 12,800 physical calls; status and
counter coverage spans all 358,400 timed calls, while exact-word checks cover
each batch's final 100-point traversal. Both output orders and actual IEEE
observations are preserved.

The 122 recorded BUILD actions exclude four additional compiler-policy captures
(12 make dry runs and four cached compiler-version queries). Make also invokes
internal shell queries; individual descendants are not exhaustively recorded.

The seven shared-host cycles describe instrumentation overhead. Stage3 contains
stage4, so these intervals must not be added. Empty-probe costs are never
subtracted to claim savings. Sticky IEEE equality observes state and does not
identify newly raised true-to-true flags. There is no speedup claim, precise
runtime acceptance, or current NinjaRS comparison. Hua-Sheng's precise local
timing campaign remains separate.

Large UTF8 payloads use deterministic gzip. No ELF, object, module, shared or
static library bytes are included. Compiler/runtime hashes are historical
identity evidence, not fresh checks of libraries on a receiving machine.
