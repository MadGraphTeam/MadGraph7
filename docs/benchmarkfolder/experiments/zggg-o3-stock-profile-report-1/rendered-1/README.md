# Actual stock O3 Zggg profile

This one-page appendix reports actual MadGraph `check profile` for
`loop_sm; u u~ > z g g g [virt=QCD]`, explicit backends 1/6/7/8/9, generated
MODEL/DHELAS/process/driver objects at O3, normal learned filters, HFL2,
LoopPoleMode−1, CTModeRun/Init1 and COLLIER UV/IR true. Installed release reducer
libraries are unchanged. All five FULL/SKIP stages contain 50 points, with 48/96
learned contributing helicities. No `check timing` or custom timing-driver
substitution is used. FULL ms/point is the stock helicity-sum result itself.

The dataset retains 174 unique base-object hashes and observed O3 compiler
commands, including the initial model-dependency-race failure and successful
retry. Quiet subsequent stock recompiles have persisted O3 make_opts/environment/
recipe evidence rather than directly emitted compiler argv. No unsafe-math flags
or older O0 objects survive in that object proof. Exact raw recipes are retained.

FULL values, ms/point: CutTools 1579.818810, Ninja 1504.547799, COLLIER 1771.917629,
NinjaRS 1450.696761, TIRust 2260.844975. TIRust/NinjaRS ratio 1.55845 is descriptive,
without a paired interval or main-campaign performance acceptance. FULL−SKIP is
only an estimated reduction cost. Stock/filter initialization is separate from
compiler time and from the earlier one-point prototype initialization experiment.

These are configured-backend timings with unchecked endpoint RETCODE 110.
Per-timed-point accepted statuses/tensor routes are unavailable. Identical saved
six-momentum/normalization words permit ten pairwise endpoint comparisons; all
pass 3e-6 and 1e-8, maximum relative residual 1.7591068225e-9. This is one retained
endpoint, not accuracy qualification of all 50 timed points; native output
freshness was not independently instrumented. All 66 local numerical and eight
main runtime gates remain pending.

## Actual commands

The source-pinned capture used these five actual stock commands (backend-specific
compiler/library settings and every original command file are retained in
`../zggg-o3-stock-profile-1/runs/ID/commands.mg5`):

```text
import model loop_sm
check profile -reuse 1 u u~ > z g g g [virt=QCD] --reduction=1 --loop_pole_mode=-1 --split_orders=-1 --energy=1000 --seed=20261112
check profile -reuse 1 u u~ > z g g g [virt=QCD] --reduction=6 --loop_pole_mode=-1 --split_orders=-1 --energy=1000 --seed=20261112
check profile -reuse 1 u u~ > z g g g [virt=QCD] --reduction=7 --loop_pole_mode=-1 --split_orders=-1 --energy=1000 --seed=20261112
check profile -reuse 1 u u~ > z g g g [virt=QCD] --reduction=8 --loop_pole_mode=-1 --split_orders=-1 --energy=1000 --seed=20261112
check profile -reuse 1 u u~ > z g g g [virt=QCD] --reduction=9 --loop_pole_mode=-1 --split_orders=-1 --energy=1000 --seed=20261112
```

Each command ran sequentially against its isolated rebuilt export, on CPU 12 of a
shared unreserved host. To repeat the native campaign use the generation/rebuild
commands and O3 settings in the retained packet and current manuals; `-reuse 1`
expects the corresponding prepared export. The FULL driver's own energy path,
external masses and physical card establish 1 TeV, independently of the CLI option
used by the separate stability phase. Run `nix develop` before document work.

## Pure verification and document reproduction

From the repository root:

```sh
python3 -B docs/benchmarkfolder/experiments/zggg-o3-stock-profile-1/preparation/verify.py docs/benchmarkfolder/experiments/zggg-o3-stock-profile-1
python3 -B docs/benchmarkfolder/experiments/zggg-o3-stock-profile-report-1/rendered-1/derive_stock_o3_profile.py --root docs/benchmarkfolder --check
```

Both commands use only the retained packet; no reducer/compiler is run. The
matching append owner is `../../../build_stock_o3_profile_report.py`. It pins
and authenticates the preceding 47 PDF/ledger/owner and its page proof, then uses
the exact existing DocumentTools helper to compare all 47 old pages and the new
page in paired 144 dpi RGB pixels, text, fonts and geometry. It does not replay
old scientific campaigns or relabel their qualification. The old 47 source
archive is `../../../history/report-before-stock-o3-profile-1`.

After promotion, verify the current canonical pair with:

```sh
python3 -B docs/benchmarkfolder/build_stock_o3_profile_report.py --root docs/benchmarkfolder --verify-only
```

The root README gives the exact pinned composite reproduction command. The
render receipt contains exact Typst/font paths and timestamp. Source, numerical
values, CSV and raw packet inventory are authenticated by the report capsule.
No PDF object-graph/annotation equivalence is claimed. Earlier failures and
strict-target misses remain preserved in the 47-page prefix.
