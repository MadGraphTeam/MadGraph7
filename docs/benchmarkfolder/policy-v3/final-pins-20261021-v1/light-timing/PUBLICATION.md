This directory retains the completed light timing study. The authoritative result is
`runs/run-v1/result.json`; `terminal-summary-v1.json` is a compact summary. It covers
six processes, ten fixed phase-space points per process, and seven adjacent
alternating pairs of original Ninja 6 and final NinjaRS 8. The existing protocol
uses a calibration to target 0.5 seconds per measured command. Actual commands
took 0.326–0.519 seconds; 0.5 seconds is not an achieved lower bound.

The larger eight-process and 100-point loop timing queues were withdrawn at the
user's request. Their retained specifications provide original source and point
provenance. They must not be executed as part of this study. The completed
eight-process, 100-point correctness campaign is separate evidence. The later
stock MG7 CLI profiles are also separate measurements.

The retained reader can be run from the repository root after activating the
documented build environment:

```sh
python3 -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/light-timing/light_pair.py validate --result docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/light-timing/runs/run-v1/result.json
```

Verification requires the recorded Git revisions `856e76e669618ec8d0f799b9c8e6f86819dcd986`
and `235901b925688282c61172d5695fe4f9bb2be93d` and the exact installed NinjaRS
artifact and metadata listed in `text-prerequisite-inventory-v1.json`. In
particular, `HEPTools/ninjars/lib/libninjars.a` must have SHA-256
`b0d69deb8c16deb13293176296afc5e18dbeeebe778a20fa39b61804fba89d4c`.
The installed archive, Cargo manifest and lockfile, and configuration remain
explicit local prerequisites. The archive is not committed. A fresh build with
different artifact bytes needs a new measured cohort; it cannot replace the
recorded artifact while keeping its identity.

All selected generation, compilation, source, physical-card, point, raw-result,
and dispatch bindings have public retained paths. The 65 small `.mod` files are
hash-bound, gzip-compressed GFortran interface descriptions retained as ABI audit
metadata. They are not an installation or an executable build cache.

The original measured quiet handoff also names one historical private native-DP
terminal receipt. Its exact bytes are retained publicly as
`release-evidence/native-dp-terminal.json`. The mapping and unchanged handoff
hashes are in `release-evidence/public-copy-provenance-v1.json`. No observed
handoff was rewritten. The retained scientific reader does not reopen this
historical quiet-release evidence. If auditing its original path explicitly,
restore that path from this exact public copy first; the mapping records both
the source path and SHA-256.

`independent-actual-timing-audit-v1.json` and
`independent-authority-validation-v1.json` retain the independent checks of the
raw coefficients, paired confidence intervals, source checkpoints, and recorded
terminal custody. Their original audit paths remain descriptive provenance;
the public copies preserve the original bytes. The mathematical and timing
claims are limited to the actual sample and its reported uncertainty.
