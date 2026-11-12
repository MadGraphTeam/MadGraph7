# Original QCD Zggg timing appendix

This one-page appendix reports the actual `loop_sm` CLI process
`u u~ > z g g g [virt=QCD]`, a QCD 2→4 timing control at 1 TeV.
The first ten points of a deterministic 100-point fixture were frozen before
amplitudes; all ten remain in each timed traversal. The original five-reducer
schedule contains ten Latin/reversed cycles. TIRust's unavailable slots are
skipped, leaving near-balanced positions for the four live reducers.

The [scientific packet](../../zggg-five-backend-timing-1/index.json) has index
SHA256 `e401cf5d4efb7708c277172eaa3be9f29bcadf54c5b3b484e7311109cf9a26cb`.
It retains all 40 timed batches, both calibrations, initialization, all coefficient
checks and target misses, frozen point streams/cards, build/generation commands
and receipts, protocol/source archives, loader/runtime identities, and clean
native termination/release. All 736 payload identities are authenticated before
pure replay. Compressed archives preserve the exact original source-tar bytes.

The signed original source is MG `cd0145922977fcfdc4e3cb3183ddc974c31a7897`
and TIRust `10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1`, DSO SHA256
`85ee0cdf860e5ad4c7a5c26163802673de22404690a10c7c53fb73ebe1ff5bb0`.
This appendix does not substitute any later initialization repair for that source.
GNU Fortran 15.2.0 and exact effective flags/runtime hashes are retained. CPU12
is an affinity choice on the shared EPYC9754 host, without a reserved CPU or
exclusive-host claim. The minimum whole ten-point batch is already much longer
than the nominal 1.1-second target: actual batches span 33.644750731–41.909685993 s.

The four complete warmed lanes use accepted DP output calls (100 per reducer).
Their median milliseconds per point are CutTools3707.5097158999997,
Ninja3489.7237694, COLLIER3950.0743433, NinjaRS3564.49005165. The CSV retains
full precision, unscaled MAD, separate READY time and exact accepted-use counts.
Ratios use matched-cycle NinjaRS8 denominators, with descriptive percentile
bootstrap intervals (10,000 whole-cycle resamples; seed20261112).
They are not candidate/baseline performance-acceptance intervals.

TIRust did not complete READY within its original 600-second budget. Its worker
was terminated cleanly; its warmed timing, READY time, MAD, ratio and confidence
interval remain unavailable. Four partial steady warmup calls (one DP, three QP)
are preserved separately and are not a complete ten-point qualification or timed
usage. Four READY lanes have before/after live runtime-map witnesses; failed
TIRust startup has executable, actual loader/ldd and unchanged shared-file binding,
without a live-map claim.

All four completed lanes pass MadLoop's `3e-6` regression gate against same-source
CutTools DP. Reducers6/7/8 miss the independent ordinary `1e-8` target in finite
terms at physical points1,2,4,5, maximum raw relative residual about4.2473e-8
at point4. Overall/sole-order duplicate rows do not add physical points or timed
calls. The partial TIRust point4 QP value agrees more closely with6/7/8 than with
CutTools DP; that corroboration assigns neither absolute truth nor automatic
fault. The full five-backend attempt retains overall `physics_regression_pass=false`
because TIRust's requested lane is incomplete. No failure is converted into a pass.

## Reproduction

The report scripts run no native physics. Reproduce JSON/CSV into a fresh folder:

```sh
python3 -B docs/benchmarkfolder/experiments/zggg-five-backend-timing-report-1/rendered-1/derive_zggg_timings.py \
  --root docs/benchmarkfolder --output /tmp/zggg-original-derived-1
```

The source binding authenticates the capsule, pure summarizer and controller,
checks all ten cycles/13 comparisons per measured lane and the exact ten-point
prefix, and recomputes the partial diagnostic from raw records. It fails closed
for live/incomplete runs or a changed source/cohort. Failed target rows remain
in `zggg-timing-data.json` and the scientific packet.

Render the exact selected Typst source using Typst0.15.1 and pinned fonts:

```sh
RAYON_NUM_THREADS=1 /nix/store/2szzba1xi60y8i3qcyxwvgq2i6w9iy5s-typst-0.15.1/bin/typst compile \
  docs/benchmarkfolder/experiments/zggg-five-backend-timing-report-1/rendered-1/zggg-timings.typ \
  /tmp/zggg-original-replayed.pdf --creation-timestamp 1794441600 \
  --font-path /nix/store/fsknhsx8k14l62da4d07pilsnxsvkdc1-libertinus-7.051/share/fonts \
  --font-path /nix/store/0lhf2h2fnjqvpzmk92s5axr2ybvs981s-dejavu-fonts-2.37/share/fonts
```

`render-receipt.json` records the successful commands, exact input identities,
font paths and one-page PDF. The style is unchanged from the existing timing
appendix, attributed to pyAmpliCol revision
`7266c1600f6ec4e39be39986e6503a515c380c52`. The navy headings, A4 geometry,
fonts, panels and green–white–red ratio palette are preserved.

The matching composite owner is `build_zggg_timing_report.py`. It authenticates
this sealed render packet and scientific evidence, archives the exact original
44-page PDF/ledger/owner, and proves all44 prefix pages plus this tail unchanged
in paired144dpi RGB pixels, text, fonts and geometry. This is not PDF object-graph
or annotation/link equivalence. The historical44 chain remains verifiable from
`history/report-before-zggg-timing-1`; canonical gains one page.

Hua-Sheng's precise local source-matched baseline/rescue-chain timing acceptance,
main-process 100-point campaign and broader stress/precision-cohort qualifications
remain separate TODOs. Later repaired-source results must retain their own phase
and source bindings while this original timeout remains preserved.
