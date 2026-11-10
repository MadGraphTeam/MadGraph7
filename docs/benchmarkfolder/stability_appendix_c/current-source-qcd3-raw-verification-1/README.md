# Closed current-source raw pilot verification

Actual capture session 10449 terminated with exit 0. Actual guarded verification session 66082 terminated with exit 0. This packet retains only the exact runner and passed proof; the isolated 14.8 MB data copy stays in ignored work space.

The exact data seal is `b2589cfcd71f9940a6dea98376be16b93e2b056aae4596d86a6fd6d5747dce30`. Three complete pure replays (original capsule, isolated copy, byte-reader callback) gave identical verification results and summaries. Read opens were 123, 119, and zero; no denied action, compiler, loader, amplitude, or lock was attempted. The guard permits read-only standard-library loading; the callback opens no files.

`guarded_verify.py` is the exact historical host runner, with the mandated checkout path bound. Its passed proof is historical evidence, not a new numerical result. The capsule's own verifier is portable and never follows recorded host paths:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-qcd3-raw-1/verify.py
```

For an independent relocated replay, copy the entire capsule to a new directory and call its `verify.py`; or supply its `verify(root=None, read_bytes=reader)` API with every indexed physical payload and `reader.paths`. The original five failed TIRust points and all 100 denominator slots remain unchanged. No accuracy or performance acceptance, historical reclassification, or hardware IEEE observation is implied.
