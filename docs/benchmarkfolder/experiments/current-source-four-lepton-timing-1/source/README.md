# Current-source four-lepton descriptive timing

This separate source-only package targets the newly generated
`u u~ > e+ e- mu+ mu- [virt=QCD]` process in `loop_sm`, case
`qcd_n4_uux_epem_mupmum`. It does not modify the first three timing cases, the
registered eleven-process campaign, production source, or historical failures.
No native timing or loader query is admitted by this source package.

The input is the closed `current-source-four-lepton-preparation-1` export:
current numerical source inventory `afb6e0be...` (registered at `0fcd2cd8...`,
actually generated at documentary revision `b443c529...`), installed eb3d TIRust,
full fresh MODEL/DHELAS/process build, and 100 saved physical points at 1 TeV,
seed **20261109**, with six zero external masses. The export's maximum loop
propagator count is four and maximum tensor rank is three. Exact build, card,
point and closure hashes are checked before copying and before/after execution.
`CAMPAIGN_ID=None` routes all new data to this diagnostic namespace; explicit
source, object, provider and configuration guards remain required.

The controller narrowly adapts the cleared first-three controller. Exact copied
source hashes are in `copy-provenance.json`. The descriptive analyzer and its
tests are byte-identical. The ten balanced cycles, two calibration traversals
(targeting two seconds per batch), full100 traversals, independent stability
checks, numerical tolerances, selected-backend DP/QP parsing, provider checks,
resource limits, observed ownership/subreaper cleanup and absolute alarms are
retained. There is one nonexclusive CPU12 case, with a **600-second total cap**
including **60 seconds reserved for cleanup**. Expired/failed work remains
unavailable, with raw evidence retained and no retry or performance acceptance.

Each of backends **1, 6, 7, 8 and 9** receives its own private export clone.
Only the immutable library directory is shared. Cards, resources, scratch,
filters, generation/compilation records and source/object inventories are
checked; no compilation or relink occurs. Existing compiler-identity/make
inspection in the standard worker start remains a runtime provenance check.

There is **no historical CutTools observation for this new process**. Backend1
must first finish checked full100 learning; the strict unchanged warmup parser
validates statuses, finite coefficients, point count and full answer dimensions.
That complete retained learning matrix becomes the fresh comparison reference.
This acquisition is not an independent accuracy test of CutTools. Other
backends' learning, both calibrations, and every timed final traversal compare
with that reference under the unchanged `3e-6` regression rule; `1e-8` target
misses remain visible. The first backend's subsequent traversals also compare
against its frozen learning reference. The full learning matrix and provenance
are retained in `fresh-cuttools-reference.json` after actual acquisition.

No prior four-lepton runtime observation exists either. Preparation borrows a
hash-pinned expected provider set from the closed current-source n1 executable,
while binding this four-lepton executable's own generated sources, objects,
static archives and metadata. Pure ELF parsing checks the actual new executable's
PT_INTERP. This is only a **prebound expected provider set**, not evidence that
the four-lepton executable used it. The admitted run must query the actual
PT_INTERP with `--list` and match that exact set, then match `/proc/PID/maps`
before and after the measurements. Native eb3d and every mapped dependency stay
pinned. The historical ldd-only loader alias is recorded separately.

Commands run from the repository root, with the exact reviewed source-index
SHA substituted as `SOURCE_SHA`:

```sh
python3 -B .tirust-build/current-source-four-lepton-timing-controller-1/run.py --preflight --source SOURCE_SHA
python3 -B .tirust-build/current-source-four-lepton-timing-controller-1/run.py --prepare --source SOURCE_SHA
```

Both commands are file-only: hashing, pure ELF parsing and private copies, no
loader/compiler/native evaluation. Preparation is append-only and prints the
specification SHA for review. A later root-reviewed execution would require
both exact source and prepared specification SHA:

```sh
python3 -B .tirust-build/current-source-four-lepton-timing-controller-1/run.py --execute --source SOURCE_SHA --prepared PREPARED_SHA
```

The output namespace is
`docs/benchmarkfolder/experiments/current-source-four-lepton-timing-1`; private
clones use `.tirust-build/current-source-four-lepton-timing-work-1`.
No numerical values or runtime results are supplied until that separate
execution completes. Results are descriptive current-candidate five-backend
measurements, with no starting-revision runtime comparison or acceptance claim.

Synthetic tests launch no workload:

```sh
python3 -B -m unittest discover -s .tirust-build/current-source-four-lepton-timing-controller-1 -p 'test_*.py'
```
