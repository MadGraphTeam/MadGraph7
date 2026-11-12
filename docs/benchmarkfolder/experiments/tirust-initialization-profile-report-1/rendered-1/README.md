# Final-source timing and initialization appendix

The two-page supplement separates final source 77eb/d427 observations from
prototype 607/470 initialization evidence. The full report preserves the exact
45-page predecessor before appending these two pages. No native physics is run
by the document owner.

Measured final first-point medians ± unscaled MAD, ms/point: CutTools 3695.587±57.175,
Ninja 3575.994±84.260, COLLIER 3837.506±127.229, NinjaRS 3526.731±82.681,
TIRust 5426.217±192.886. The matched-cycle median TIRust/NinjaRS ratio is 1.5532738353,
not a ratio of medians. This is an explicitly selected diagnosed point from the
original 100-point fixture: 35 batches, seven cycles, 45 retained numerical checks.
Every 3e-6 comparison passes; strict 1e-8 finite misses for 6/7/8/9 remain. All seven
TIRust dispatches transport DP inputs, with local 106/256 tensor rescues and zero
genuine Fortran-QP dispatches; they are not a native-DP-only cohort.

Corrected stock `check profile` gives NinjaRS FULL/SKIP/estimated difference
0.077810915/0.065434926/0.012375989ms and TIRust
0.124895949/0.064708981/0.060186969ms. FULL uses monotonic child-completion timing;
other stages retain wall clocks. Unequal short single runs give descriptive
ratio 1.605121207, no confidence interval. End-of-batch all-helicity Born/finite
results are nonzero; per-timed-point status/tensor routes are unavailable.
TIRust reused its export and fresh initialization remains null. Separate
10-point stability probes are not independent physical-error measurements.

The quiet 255.241→15.346s initialization pair remains bound to prototype 470,
with its separate functional observer and same-engine 512-bit reference. All
50 individual near-zero component-relative misses are retained in CSV. Original
instrumented attribution, unpromoted O2 control, old coarse stock controls and
original 600s ten-point timeout keep their original source identities.

The current source-matched 66 numerical lanes and eight main local runtime gates
are pending. TIRust remains slower in both final warmed comparisons. These pilots
supply no process-average, full campaign or upper 95% baseline-chain acceptance.
Use `../../../LOCAL_FINAL_VALIDATION_TODO.md` for Hua-Sheng’s local commands.

## Pure replay

From the repository root:

```sh
python3 -B docs/benchmarkfolder/experiments/tirust-initialization-profile-report-1/rendered-1/derive_initialization_profile.py --root docs/benchmarkfolder --check
```

The replayer authenticates all five scientific packets, replays the retained
prototype phase/precision metric, final artifact chain, all 35 warmed batches/
45 checks and corrected stock records. It compares all derived JSON and nine
CSVs byte for byte. It invokes no reducer, compiler or controller.

The current document owner is `../../../build_initialization_profile_report.py`.
After canonical promotion, verify PDF/ledger/source selection with:

```sh
python3 -B docs/benchmarkfolder/build_initialization_profile_report.py --root docs/benchmarkfolder --verify-only
```

For a fresh composite output, use the exact archive/report arguments in the
current `report-build.json`, changing only `--output`; the root README includes
the pinned command. `--verify-pages` additionally repeats paired 144 dpi RGB,
per-page text/font/geometry checks for all 45 prefix and both appended pages.
It does not claim PDF object-graph, annotation or hyperlink equivalence.

Exact Typst/font paths and successful render command are in `render-receipt.json`;
`initialization-and-profile.typ` plus `phase-bars.svg` reproduce the supplement.
The existing 16 MB MG source archive is reused by hash. The preserved style is
attributed to pyAmpliCol 7266c1600f6ec4e39be39986e6503a515c380c52 in `style-provenance.json`.
No earlier failure is converted into a pass and no prototype timing is relabeled.
