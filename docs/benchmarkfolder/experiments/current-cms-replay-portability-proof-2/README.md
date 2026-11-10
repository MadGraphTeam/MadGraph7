# CMS2 copied-capsule portability proof

The sealed CMS2 capsule was copied byte-for-byte into a fresh directory under `/tmp`, then verified from that copied directory through an in-memory byte callback exposing the complete `.paths` inventory. The reviewed verifier was compiled from the copied, hash-pinned source. Every one of the 1,181 copied files was read through the callback; verification attempted no filesystem, subprocess or dynamic-library access.

The harness initializes Python's standard-library modules before enabling its gates. During verification, filesystem entrypoints, process/exec/spawn entrypoints and ctypes loaders are patched to reject access; an active CPython audit hook also rejects file opens, process starts and dynamic-library events. Thirteen negative probes demonstrate that these gates are active. This exercises the exact reviewed verifier under Python entrypoint/audit gates; it is not a general hostile-code operating-system sandbox.

The run returned0 and produced `receipt.json`. Source and copied capsule bytes were checked identical before and after. The temporary copy remains at the receipt's recorded path for review. No original qualification/work namespace or live binary was read, no scientific native execution was launched, and the sealed source capsule was not modified. Its test receipts and source remain unchanged and are pinned by the capsule index and this proof index.

The verified result remains **incomplete**: six completed orchestrations and zero qualified final streams. The portability check does not change numerical qualification or scientific scope.

The executed command, from the repository root, was:

```bash
source tools/tirust_env.sh
python -B docs/benchmarkfolder/experiments/current-cms-replay-portability-proof-2/prove.py
```

The harness rejects optimized Python and refuses to overwrite an existing receipt. A later run requires a separate proof namespace. `proof-index.json` seals this harness, receipt, review and document, including the original capsule and retained-test hashes.
