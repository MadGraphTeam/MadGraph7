# Held ABI3 / tensor-factor candidate representative plan

This plan is **not launched**. Use it after final CLI generation, installation
and clean builds have frozen under:

- outputs: `.benchmark-work/policy-v2/generation-abi3-tensor-factor-v1/outputs/candidate`;
- generation evidence: `docs/benchmarkfolder/policy-v2/generation-abi3-tensor-factor-v1`.

The minimum refresh is eight candidate targets, assuming all final candidate
exports share one identical full runtime signature, including IREGI, ABI3
NinjaRS, scalar libraries, module files and link metadata.

| Specification | Complete retained stream | Backends | Coverage |
|---|---|---|---|
| `ggh.json` | All 10 `control-gg-h` points | 1, 6, 7, 8 | DP for every backend and loop-induced amplitudes |
| `mixed.json` | All 10 `control-mixed` points | 8 | Multiple physical split orders |
| `forced-qp.json` | All 10 ordinary `control-forced-qp` points | 1, 8 | Actual CutTools and NinjaRS QP rescue |
| `ninja-near-singular.json` | All 10 retained near-singular `gg → tt~` points | 6 | Actual Ninja QP rescue |

The last specification uses the already retained, byte-identical point-root
alias at `../representative-point-streams/near-singular`; it reuses the same
anchored `gg → tt~` physical export. The ordinary strict forced-Ninja failures
remain historical evidence and are not filtered or relabelled. The forced
representatives retain their explicit `1e-20` threshold and two DP/QP rotations;
core numerical settings stay unchanged.

Every target performs fresh-process parity plus two persistent commands over
the entire stream. CPU 5 is for representative validation, not acceptance
timing. Reserve the Symbolica slot for any target selecting backend 8. A failed
status, missing order, failed numerical parity, changed source/runtime identity
or incomplete stream rejects proof; retain the failed attempt in a new run
namespace before retrying. If any final target has another runtime signature,
add complete DP and actual QP coverage for that exact group rather than combining
groups or excluding a linked library.

`commands.sh` lists the four frozen-collector invocations and their ordinary
role/runtime bindings. It finishes by sealing a new three-role proof using the
12 unchanged eligible legacy bindings in `../legacy-coverage-status.json`.
It does not publish a dataset or run the eight-case performance matrix. The
ordinary full core correctness comparison and every outstanding focused control
in `../final-controls-checklist.md` remain required.

No qcd1 representative is needed for this minimum set: the complete ggH stream
provides DP coverage for all four backends under the same exact runtime. QCD1
still participates in the final 100-point core validation and timing.

This new namespace follows the QP tensor-factor source commit and successful development NLO validation. Earlier completed candidate proofs remain historical; eligible legacy bindings are reused without alteration.
