# MadLoop TIRust benchmark report

The report, its Typst sources, every benchmark campaign, the physical cards,
frozen phase-space points, stability data and earlier experiments are
distributed in the
[benchmark release](https://github.com/huashengshao/TIRust/releases/tag/madloop-tirust-benchmarks-v1),
not in the MadGraph source tree. The
[PDF](https://github.com/huashengshao/TIRust/releases/download/madloop-tirust-benchmarks-v1/benchmark.pdf)
is available separately for reading.

Download
[`madloop-tirust-benchmark-5298fd3066.tar.xz`](https://github.com/huashengshao/TIRust/releases/download/madloop-tirust-benchmarks-v1/madloop-tirust-benchmark-5298fd3066.tar.xz).
Its SHA256 is:

```text
27e36fa6ac0be45514e23d637f384a281e9be5a54aa0fc5b0cdd66cdefa6d570
```

Every archived file is byte-identical to source revision
[`5298fd3066`](https://github.com/MadGraphTeam/MadGraph7/tree/5298fd3066fe31cf27cd9005c21ca65c26eaab8e/docs/benchmarkfolder),
which the tag `madloop_tirust-evidence-2026-11-17` keeps. Older evidence
capsules and report builders are in the tag
`madloop_tirust-evidence-2026-11-13`. The bundle's `RELEASE.md` describes its
contents, and `RELEASE-PROVENANCE.json` records the source revision and the
files it adds.

## Restoring the workspace

The benchmark tools (`tools/tirust_*`) read and write the workspace
`docs/benchmarkfolder`, which git ignores. Restore it before a requalification,
the local timing or a report build:

```sh
python3 -B tools/tirust_benchmark_bundle.py restore
python3 -B -m unittest discover -s docs/benchmarkfolder/tests
```

`restore` downloads the pinned release (or takes `--archive PATH`), checks its
SHA256 against `tools/tirust_benchmark_bundle.json`, keeps identical files and
refuses to overwrite changed ones. `verify` checks the workspace against its
`SHA256SUMS`. A new campaign is written into the workspace and published as a
new bundle version.

## Current results

Campaign `tirust-2b9ba961-1` (TIRust `main` `2b9ba961`, bridge method choice
off), measured on a shared host. `CURRENT_STATUS.md` in the bundle has the
details.

- **Ordinary matrix:** 66/66 lanes pass `3e-6`; 54 also meet `1e-8`. TIRust
  needs no genuine-QP rescue in any process.
- **Double-double gate (headroom 26):** 1,289,214 captured bridge calls replay
  bit for bit, and every DD-tier call agrees with 256-bit evaluation within
  1.7e-16. Qualified.
- **Caches:** cache-off and a 1 MiB budget reproduce every bit at all 1,100
  ordinary points.
- **Stress (38 points):** TIRust accepts 28 and refuses 10. One accepted TIRust
  value is wrong (below).

## Open before merge

1. **TIRust at the EW t t̄ γ threshold.** TIRust returns 213 with a claimed
   accuracy of 6.5e-9, but the value is 1.1% off. The binary64 inputs fix this
   point only to about 1%, and both loop directions see the same inputs, so
   MadLoop's check passes. A bridge fix shifts the invariants of MadLoop's
   double-precision check pass by 1–2 ulps, so the check measures that
   sensitivity and the point is rescued in quadruple precision. It will be
   applied with the next TIRust pin
   ([huashengshao/TIRust#43](https://github.com/huashengshao/TIRust/pull/43),
   multi-precision rounding at the working precision), followed by one combined
   requalification and a new bundle version.
2. **Local timing acceptance** on a quiet machine, using the combined campaign.
