# Compact publication of the closed first-failure diagnostic

This packet is an additive lossless representation of original closed inventory
`d85a79ce185660512334c2f51e01cfa1734304149a29188fd62c782625a7092e`.
Its exact index is retained as `original-closed-1-index.json`. All logical
payload bytes remain identical, including failed capture/preadmission history,
the six captured Zgg inputs, six missing ZZgg inputs and twelve failed
standalone replay evaluations. No amplitude, tensor replay, guard reset or
production change is performed by compaction or verification.

Only the two complete GNU link maps use deterministic gzip (mtime0). The new
index binds both compressed bytes and the original decompressed size/hash.
The verifier bounds expansion to each original size, rejects extra gzip
members/trailing data/truncation, reconstructs the original logical reader,
and invokes the exact original portable verifier in memory. It never follows
historical host paths or calls a native library. The large original map files
are not duplicated in this publication.

Verify from the repository:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/retention-source-2/verify.py docs/benchmarkfolder/experiments/tirust-first-failure-capture/closed-2
```

After relocation, use `compact-source/verify.py` with the relocated packet
root. The returned summary preserves all12 intended inputs, six missing
requests, the original failed capture, and the fresh/repeated-context status1
Gram-conditioning rejections. The original `README.md` remains byte-identical
historical documentation of the uncompressed inventory; this file gives the
current compact verification command.

These are C-boundary status/error diagnostics, without independent numerical
truth or timing acceptance. LAST_ERROR identifies the reported Gram gate;
the exact eventual inner fallback cause is not fully observed. Software latch
history is distinct from hardware IEEE exceptions, which were not established.
