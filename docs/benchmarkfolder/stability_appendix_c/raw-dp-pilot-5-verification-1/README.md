# Closed raw stability verification

This packet preserves the exact successful relocated-verification runner and
its proof for data capsule `raw-dp-pilot-5-1`, index
`ebd8f1faf5228f205b1038a490329037066988dc03e7e774210d648b38dbbb6e`.
The proof records full raw-parser, physical-point and metric replay after
relocation: 557 capsule byte reads, 210 directory-handle opens, zero external
byte reads, zero mutations and zero native execution/loading events. Negative
read, write, child and loader probes were rejected before their operations.
Directory handles on ancestors are allowed only to reach the relocated tree.
Historical runtime observer receipts are checked without reopening their host
binaries. This establishes portable replay consistency, not independent true
error, observed reducer-internal precision or performance acceptance.

The first harness attempt correctly rejected a Python standard-library
`textwrap` import because it had not been preloaded. Its exact runner and
failure note are preserved under `first-harness-failure`. Adding that preload
to the successful runner changed no data, diagnostic policy or numerical code.
The large temporary relocated capsule is deliberately excluded.

To repeat, copy `guarded_verify.py` into a new ignored working directory, then
run it from the repository working directory (replace `NEW` with a fresh name):

```sh
source tools/tirust_env.sh
mkdir .tirust-build/NEW
cp docs/benchmarkfolder/stability_appendix_c/raw-dp-pilot-5-verification-1/guarded_verify.py .tirust-build/NEW/
python3 -B .tirust-build/NEW/guarded_verify.py docs/benchmarkfolder/stability_appendix_c/raw-dp-pilot-5-1 --expected-index ebd8f1faf5228f205b1038a490329037066988dc03e7e774210d648b38dbbb6e
```

The repeated run writes its relocated copy and new proof only in that ignored
working directory. It does not modify this packet or the admitted data capsule.
