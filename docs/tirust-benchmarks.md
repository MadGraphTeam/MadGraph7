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

Campaign `tirust-6ba4b531-1` (TIRust `main` `6ba4b53`, bridge method choice
off, shifted DP check pass), measured on a shared host. It requalifies the pin
move from `1c130a81`, which brings the triangle PV error estimate
(huashengshao/TIRust#64) and the estimate-only refusal split
(huashengshao/TIRust#72). This campaign is in the workspace but not yet in a
benchmark release; version 2 holds the previous pin's campaign
`tirust-1c130a81-1`, whose `CURRENT_STATUS.md` describes the protocol.

- **Ordinary matrix:** 66/66 lanes pass `3e-6`; 54 also meet `1e-8`, the same
  lanes at the same points as at `1c130a81`. TIRust needs no genuine-QP rescue
  in any process. Per-point TIRust finite parts change by at most 5.7e-12
  relative (`u u~ > z z g g`); seven of the eleven processes are
  bit-identical.
- **Double-double gate (headroom 26):** 1,289,214 captured bridge calls replay
  bit for bit, and every DD-tier call (28,753, up from 23,123) agrees with
  256-bit evaluation within 1.7e-16. Qualified.
- **Pin-move replay:** 16 recorded traces (about 2.0 million calls, the
  ordinary and stress captures of `tirust-1c130a81-1` and a Zggg trace)
  replayed at `6ba4b53`. No call changes between served and refused. In the
  ordinary traces and the `u u~ > z z g g` stress capture, each of the 52
  served calls that moved by more than 1e-9 is closer to a 1024-bit reference
  than before; the worst was 5.1e-4 off and is now 3e-23 off. Inside the
  refused soft-collinear stress points, served calls improve from up to 0.40
  off to up to 1.05e-4 off, and two are slightly worse (up to 2.8e-6). Those
  values never reach MadLoop, because the point is refused as a whole.
- **Caches:** cache-off and a 1 MiB budget reproduce every bit at all 1,100
  ordinary points.
- **Stress (38 points):** the same classification as at `1c130a81`, number for
  number. TIRust accepts 28 and refuses 10, and no accepted TIRust value is
  wrong. Two wrong values in the TIRust-first chain come from its CutTools
  fallback, which accepts them without TIRust too.
- **Cost:** points served by binary64 alone drop from 66 to 18 of 100 in
  `u u~ > z g g`, from 97 to 86 in `e+ e- > t t~ a` and from 12 to 3 in
  `e+ e- > mu+ mu- a a`; the rest use the DD or MPFR tiers. Replaying a
  recorded Zggg trace's TIRust calls, alternating the two pins on one core,
  takes 12% longer at `6ba4b53` (4.36 s against 3.90 s, three runs each). Of
  the 0.46 s, 0.35 s is in DD-tier pentagons and hexagons that were already in
  the tier, and 0.02 s is in the 323 triangles that newly reach it. At
  `1c130a81`, with method choice off, TIRust took 5.5% (stock `check profile`)
  to 22% (default card) more time per Zggg point; that end-to-end figure was
  not remeasured. A precise local timing is optional and not a merge
  condition; the bundle's `LOCAL_FINAL_VALIDATION_TODO.md` describes it.
