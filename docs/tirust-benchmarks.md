# MadLoop TIRust benchmark report

The report, its Typst sources, every benchmark campaign, the physical cards,
frozen phase-space points, stability data and earlier experiments are
distributed in the
[benchmark release](https://github.com/huashengshao/TIRust/releases/tag/madloop-tirust-benchmarks-v2),
not in the MadGraph source tree. The
[PDF](https://github.com/huashengshao/TIRust/releases/download/madloop-tirust-benchmarks-v2/benchmark.pdf)
is available separately for reading.

Download
[`madloop-tirust-benchmark-e0f51afcef.tar.xz`](https://github.com/huashengshao/TIRust/releases/download/madloop-tirust-benchmarks-v2/madloop-tirust-benchmark-e0f51afcef.tar.xz).
Its SHA256 is:

```text
313659fcfa5ca2c66b8c096de6f36c99ffb4b09b85863885f7d35474df3e8e8b
```

Version 2 is
[version 1](https://github.com/huashengshao/TIRust/releases/tag/madloop-tirust-benchmarks-v1)
plus campaign `tirust-1c130a81-1`. It was built from the restored workspace at
source revision `e0f51afcef`. Version 1 is byte-identical to source revision
[`5298fd3066`](https://github.com/MadGraphTeam/MadGraph7/tree/5298fd3066fe31cf27cd9005c21ca65c26eaab8e/docs/benchmarkfolder),
which the tag `madloop_tirust-evidence-2026-11-17` keeps. Older evidence
capsules and report builders are in the tag
`madloop_tirust-evidence-2026-11-13`. The bundle's `RELEASE.md` describes its
contents, and `RELEASE-PROVENANCE.json` lists the files that differ from
version 1 and the files it adds.

## Restoring the workspace

The benchmark tools (`tools/tirust_*`) read and write the workspace
`docs/benchmarkfolder`, which git ignores. Restore it before a requalification,
a timing run or a report build:

```sh
python3 -B tools/tirust_benchmark_bundle.py restore
python3 -B -m unittest discover -s docs/benchmarkfolder/tests
```

`restore` downloads the pinned release (or takes `--archive PATH`), checks its
SHA256 against `tools/tirust_benchmark_bundle.json`, keeps identical files and
refuses to overwrite changed ones. Over a version-1 workspace, add `--force`
for the seven files that version 2 changes. `verify` checks the workspace
against its `SHA256SUMS`. A new campaign is written into the workspace and
published as a new bundle version (`build --workspace`).

## Current results

Campaign `tirust-1c130a81-1` (TIRust `main` `1c130a81`, bridge method choice
off, shifted DP check pass), measured on a shared host. `CURRENT_STATUS.md` in
the bundle has the details.

- **Ordinary matrix:** 66/66 lanes pass `3e-6`; 54 also meet `1e-8`. TIRust
  needs no genuine-QP rescue in any process.
- **Double-double gate (headroom 26):** 1,289,214 captured bridge calls replay
  bit for bit, and every DD-tier call agrees with 256-bit evaluation within
  1.7e-16. Qualified.
- **Caches:** cache-off and a 1 MiB budget reproduce every bit at all 1,100
  ordinary points.
- **Stress (38 points):** TIRust accepts 28 and refuses 10. No accepted TIRust
  value is wrong. At the EW t t̄ γ threshold, where the binary64 kinematics fix
  the point only to about 1%, the bridge's shifted check pass now makes MadLoop
  rescue the point in quadruple precision. The previous campaign accepted a
  double-precision value 1.1% off there.
- **Cost:** with method choice off, TIRust takes 5.5% (stock `check profile`)
  to 22% (default card) more time per Zggg point, measured on paired replays
  on the shared host. A precise local timing is optional and not a merge
  condition; the bundle's `LOCAL_FINAL_VALIDATION_TODO.md` describes it.
