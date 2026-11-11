# Closed ordinary campaign extension: source-only report proposal

This packet extends the current-source supplement of published report commit
`75590e978f3a1c45231d6ed1e74a616960ede6be`. It extends the explicit numerical
checkpoint whitelist, binds the reviewed combined-closure replay API, and adds
baseline/continuation provenance notes. The existing target-miss table gains a
B/F/single-pole/double-pole breakdown from its already retained column counts.
Production code, numerical criteria, the raw stability study and its plot are
unchanged.

The selected QCD n4 closure is
`55649478b7aa8c2916765c5389ff2dcc1ec2e3359cc3c212f363ec6bf462a53a`.
Its six 100-point lanes all pass the `3e-6` regression criterion. The `1e-8`
finite target misses remain 88/80/80/78 for candidate backends 6/7/8/9.
Accepted DP/QP counts are 88/12 (baseline CutTools), 88/12 (candidate CutTools),
97/3 (Ninja), 100/0 (COLLIER), 96/4 (NinjaRS), and 14/86 (TIRust).
The original baseline full-build receipt remains unlocated; the retained proof
covers 155 inherited objects, 206 source pins and donor compiler flags.

The selected combined EW n4 closure is
`2819df51b7dc3816ce869a8a5b08b7a7bccd05e6a8d67b0e629112645d3aa0ae`,
with exact verifier source index
`1188b11951bcfb78d2c02768f5d36db37bd6b610e5c3da83f0a2291dc2a10b06`.
Its six complete selected lanes pass `3e-6`; candidate backends 6/7/8/9 retain
12/10/14/8 target misses. Their B/F/single/double breakdowns are 0/10/2/0,
0/8/0/2, 0/10/2/2 and 0/8/0/0. TIRust accepts 40 DP and 60 QP points.
The original attempt remains interrupted after saved point 74, with its five
completed lanes selected and its partial TIRust stream excluded. A separate
full 100-point TIRust attempt supplies the final lane; no partial streams were
concatenated. The combined case has no single returncode, and the consumer
validates that schema-specific absence while its pure verifier checks actual
stages and workers. The EW n4 baseline proof covers 320 inherited objects and
371 source pins; its original full-build receipt remains unlocated.

This frozen selection covers all 11 ordinary cases and 66 complete lanes:
all pass `3e-6`, 54 pass `1e-8`, and the 6,600 selected final outcomes comprise
6,396 DP and 204 QP results. Retained interrupted work is additional evidence,
not part of those selected-outcome totals.

`selection.json` selects a new `progress-3/report-context.json`; the prior
progress-1 and progress-2 selections and contexts remain unchanged. The historical
dossier, cache measurements and 66-lane fresh local qualification TODO also remain
unchanged. Exact data pins and outcomes, including unavailable raw stability
points, are retained. No paired timing or overall acceptance is established.

## Pure preparation and focused validation

Run from `/shared/external/madloop_tirust/madgraph7_madloop_tirust`:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/current-source-final-campaign-supplement-proposal-1/prepare_preview.py
python3 -B docs/benchmarkfolder/experiments/current-source-final-campaign-supplement-proposal-1/test_campaign_extension.py -v
```

Preparation maps only the proposed helper and new selection into the existing
report collector. All admitted numerical, raw stability, figure and cache
verifiers replay with process launches and foreign-library loading blocked.
`preview.json`, `preview-inputs.json` and `preview-receipt.json` retain this actual
preparation. The eleven focused tests cover previous-context preservation, exact
new outcomes and column breakdowns, payload binding, old selections, fail-closed
admission, unchanged canonical/raw-study inputs, baseline and continuation
provenance, and the narrow Typst change. The already reviewed consumer readiness
packet retains the synthetic interruption/refusal checks; this final preparation
replays the actual admitted combined closure in an indexed-files-only relocation.
The combined inventory includes seven `tools/` protocol identities. Each is
materialized only in the temporary replay root from its size/hash-identical
retained `lane/` copy, with an exact seven-name mapping. The report input boundary
remains the docs directory; no working `tools/` file is read or parsed.

This packet is a source-only proposal. No canonical context or PDF was written,
and no Typst render, compiler, amplitude call or Git operation was performed.
Pagination is unqualified until the root performs the actual render and visual
review. The existing raw stability page implementation is byte-identical.

## Promotion and reproduction

Verify `index.json` and `promotion-manifest.json` against every indexed byte first.
Require every `before` identity in `promotion-manifest.json` to match the current
canonical files. Apply `existing-files.patch`, or copy the identical proposed
helper and Typst fragment to their exact canonical destinations. Copy `selection.json` to the new
canonical selection destination specified in the promotion manifest. Do not
replace either historical progress selection or context.

Then run the existing canonical builder explicitly:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/build_report.py --campaign invariant-symmetry-1 \
  --campaign-report campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json \
  --supplement campaigns/entry-latch-cache-guard-1/report-supplements/progress-3/selection.json
python3 -B docs/benchmarkfolder/build_report.py --campaign invariant-symmetry-1 \
  --campaign-report campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json \
  --supplement campaigns/entry-latch-cache-guard-1/report-supplements/progress-3/selection.json --verify-only
```

The builder alone writes the new canonical context, `report-build.json` and PDF.
Check the generated context equals this packet's preview byte for byte, inspect
pagination and captions, and preserve the published 34-page predecessor via its
Git commit. No prediction of page count substitutes for that render.
