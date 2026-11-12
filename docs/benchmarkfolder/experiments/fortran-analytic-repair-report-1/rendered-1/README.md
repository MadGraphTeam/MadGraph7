# Private analytic Fortran repair supplement

This one-page supplement documents the completed same-flag 2→1
`u u~ > z [virt=QCD]`, `loop_sm`, backend 9 comparison against the current
`0fcd2cd873d61d134837e1d9e87afb13a47824fd` control. Its loop topology has
three propagators and rank two. The candidate source patch is unapplied.
The starting-revision default-chain/eight-process acceptance and Hua-Sheng’s
precise local timing campaign remain separate open requirements.

Data are derived only from capsule index
`f8bc6ce06201665abc0da7d79bcb55697a215d99f2258e9ba623453ebc33e6e9`
and its 14 authenticated raw timing/status batches and closed independent review
`9e10f47d4599235efef6fb7908c5a710282a9ffbb89cbe53793e0e2cfa780306`.
The median paired ratio and ratio of arm medians have distinct JSON fields.
MAD is unscaled median absolute deviation of the seven batch times per point,
not a standard error. All 179,200 timed statuses are recorded; coefficient
records are batch-final 100-point sets, not every timed call.

From the MadGraph repository, pure data reproduction/check (no model, reducer,
MG CLI, compiler or scientific native evaluation):

```sh
python3 -B .tirust-build/fortran-analytic-repair-report-2/derive_timing.py \
  --capsule docs/benchmarkfolder/experiments/fortran-analytic-repair-qualification-1 \
  --check
```

When this source is relocated, run the same `derive_timing.py` beside its data
and supply the exact capsule path. Omitting `--check` creates the two JSON
inputs exclusively in that script’s directory; existing data are never
overwritten. The private original capsule path is
`.tirust-build/fortran-analytic-repair-retention-1/package`.

After root publishes the documentary tail, its repository command is:

```sh
python3 -B docs/benchmarkfolder/experiments/fortran-analytic-repair-report-1/rendered-1/derive_timing.py \
  --capsule docs/benchmarkfolder/experiments/fortran-analytic-repair-qualification-1 \
  --check
```

The standalone page uses the current rendered style unchanged: Libertinus
Serif 10.5pt, DejaVu Sans tables, A4 margins 1.6cm horizontal/1.7cm top/1.6cm
bottom, navy headings and the green–white–red log-ratio palette. Attribution
and original pyAmpliCol revision 7266c160 are in `style-provenance.json`.

Pinned document-only rendering command, with creation time 2026-11-12 UTC:

```sh
/nix/store/2szzba1xi60y8i3qcyxwvgq2i6w9iy5s-typst-0.15.1/bin/typst --version
/nix/store/2szzba1xi60y8i3qcyxwvgq2i6w9iy5s-typst-0.15.1/bin/typst compile \
  .tirust-build/fortran-analytic-repair-report-2/analytic-repair.typ \
  .tirust-build/fortran-analytic-repair-report-2/analytic-repair-supplement.pdf \
  --creation-timestamp 1794441600 \
  --font-path /nix/store/fsknhsx8k14l62da4d07pilsnxsvkdc1-libertinus-7.051/share/fonts \
  --font-path /nix/store/0lhf2h2fnjqvpzmk92s5axr2ybvs981s-dejavu-fonts-2.37/share/fonts
```

Root owns any later exact 42-page archive, append, canonical PDF, ledger,
public copy and Git publication. This private preparation performs none of
those writes and records no new scientific/native reducer execution.
