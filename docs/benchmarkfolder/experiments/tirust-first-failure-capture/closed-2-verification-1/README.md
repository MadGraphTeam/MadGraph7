# Portable verification of the closed compact diagnostic

Tool session50512 actually terminated0. The retained runner verified the exact
compact index `1e26738f…` at its original location, copied only that packet into
a fresh ignored namespace, then verified the relocated packet and an in-memory
byte-reader under a guard denying external file reads, writes, child processes,
loaders and numerical lock acquisition. All returned results were identical;
there were zero denied attempts and zero native evaluations.

The proof binds the exact verifier3097d96a… and source runner. No relocated
13.9MB payload copy is duplicated here. The original failed capture, failed
preadmission, six missing ZZgg inputs and twelve status1 replay evaluations
remain unchanged in the compact packet. This proves portable evidence checks,
without numerical accuracy or performance acceptance.

Reproduce with a fresh copy of this runner in an ignored empty directory:

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/NEW-EMPTY-DIRECTORY/guarded_verify.py --verifier docs/benchmarkfolder/experiments/tirust-first-failure-capture/retention-source-2/verify.py --verifier-sha256 3097d96a5e90cde3aa437a1760384b05a25f2cf532b759f1376833a61169b8ec --capsule docs/benchmarkfolder/experiments/tirust-first-failure-capture/closed-2 --index-sha256 1e26738f4489b55e85ac059c963fec0c75dbc33f8cb4bd208f59a8676477cc06
```
