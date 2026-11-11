# Current kinematic trace: closed build qualification

This packet preserves the successful **build-only** closure of the current-source
kinematic trace experiment: 258 owned commands, four executables (CMS/EW ×
unmodified/instrumented), and four explicit dynamic-loader `--list` queries with
15 resolved providers each. The terminal exit code was zero; protected inputs
matched before and after, and the post-terminal check found none of the 661
recorded process identities alive and acquired/released the common lock.
**The closed build stage performed zero amplitude evaluations.**

The CMS export was freshly generated from source anchor
`0fcd2cd873d61d134837e1d9e87afb13a47824fd`. The reused current EW export is bound
to the same numerical inventory by the retained source receipts. Trace changes
are insertion-only instrumentation with byte-stripping proofs, target-group
identities and signed-momentum mappings. Four private driver/process variants
were built with pinned compiler, object/archive, link-map and provider identities.
This establishes compilation and loader-resolution evidence. It does not
establish trace-enabled/disabled observational parity, numerical accuracy,
stability, or runtime performance.

The exact closed inputs are:

| Record | SHA-256 prefix |
|---|---|
| Terminal build closure | `564688cfb10fa7a0` |
| Sealed source packet | `7ea709b48d8de462` |
| Fresh generated/derived packet | `25c2114756ed3eb9` |
| Additive build admission | `c30fefa33c13dc04` |
| Post-terminal ownership check | `e4849a7dbce603f4` |
| Independent root build review | `36c3ec42425ee8cd` |

Original pre-admission README/index fields such as `compiled:false` and
`native_stage_admitted:false` are preserved as historical source-review records.
The later terminal closure and root review record the admitted build result.
The additive guard checks the actual ELF interpreter against the compiler's
declared loader before each owned `--list` query, with precisely pinned loader
and quadmath supplements. These queries resolve libraries; they are not
amplitude calls or evidence of runtime initialization during a physics worker.

Retention reads only the closure's explicit 258 command-receipt paths and their
matching logs, four named build/guard receipt sets, sealed source maps, and named
provenance/helper text. It never enumerates the executor's execution directory or
reads ordinary/stress worker records. Later ordinary runs are a separate stage
and are excluded from this packet. The original symbolic prototype, ancestry,
source packets and external binaries remain unchanged.

Large text and command receipts/logs use deterministic gzip (`mtime=0`), following
the existing portable qualification packets. `evidence-index.json` records both
stored-byte and decompressed-byte identities plus original path provenance.
The four ELF hashes, archive/object hashes, runtime-provider hashes and full link
closures are retained as recorded identities; their omitted external bytes are
not rehashed by portable verification. No executable, native library, object,
module or source archive is committed here. Elapsed values are incidental build
diagnostics, not performance measurements.

From the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/current-kinematic-trace-build-qualification-1/verify.py
```

This checks all retained byte identities and source seals, the 258 closed command
results/logs and cleanup records, four build/prelink/provider records, protected
input equality and root/ownership closure. It opens only this packet, executes no
retained controller and launches no native process. `capture.py` records the
explicit original selection and is not needed for portable replay.
