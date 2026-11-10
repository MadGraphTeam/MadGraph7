# Private Born/effective-point implementation and legacy qualification

This milestone contains a concrete private implementation, complete generated
sources and completed legacy-backend tests. It does **not** qualify the pending
NinjaRS/TIRust runtime cases, retained QCD/EW stress cases, or performance. No
production source was changed by this experiment.

| Evidence | Result and scope |
|---|---|
| Four processes × four implementation arms | All 16 generated, compiled and linked |
| Controlled Fortran lifecycle/repair fixtures | 51/51 passed; no integral reducer |
| Ordinary real/CMS, original/full, backends 1/6/7 | 12/12 runs, six comparisons; every coefficient residual exactly zero |
| Fixed-QP real/CMS, original/full, backends 1/6, two integrand policies | 16/16 runs; 48 observed calls returned **120, unchecked CT4** |
| Full-arm Born and adopted coordinate consistency | All 24 fixed-QP observations match the requested Born view exactly; all 24 DP coordinate views equal binary64 rounding of C |

The ordinary comparisons also preserve status, MP state, IEEE flags and rounding
mode. The fixed-QP results establish transport and Born-policy consistency, not
independent numerical stability or amplitude accuracy. The original code matches
the requested Born reference in 12/24 fixed-QP observations (the DP-integrand
policy). For example, the first real CutTools observation publishes binary64
`405BB42F484780A5` in the original arm for either policy. The full arm retains that
value for the DP-integrand policy and publishes `405BB42F484780A7`, the correctly
rounded MP Born(C), for the MP-integrand policy. This two-ULP difference is a
precision-policy change; it is not an accuracy verdict.

The four processes are `u u~ > z [virt=QCD]`, CMS
`e+ e- > t t~ [virt=QCD]`, pinned Gmu-model `e+ e- > mu+ mu- [virt=QED]`, and
`u u~ > z g g [virt=QCD]`. Only the first two were executed here, at two saved
ordinary points followed by repetition of the first point. The 83 definitions in
`runtime-specifications.json` are the original **prepared** workload: only the
28 legacy runs listed above were executed in this milestone. Gmu, QCD n3,
NinjaRS/TIRust, and the mass/graph arms remain compiled-only here.

## Implementation scope

The arms are `original`, `mass`, `graph`, and `full`. `mass` supplies genuine MP
masses through guarded callers to QP backends 8/9. `graph` additionally uses
integer graph classification in the QP kinematic consumer. `full` adds checked
momentum repair, a generated stateless Born aggregate, transactional Born
learning and canonical-C adoption with matching Born/interference normalization.
The original arm uses the captured current base exporter through the private
subclass and its dense DP path; it is not the exact-type production terminal
dispatch optimization.

The full host's supported scalar optimized Born QP path covers backends 1/6/8/9;
it is not restricted to 8/9. With `UseQPIntegrand=true`, the Born and numerator
use C with actual MP model parameters. With false, they use `REAL(C,8)` and the
DP model; genuine QP tensor precision remains separately identified. The graph
consumer's policy 1 uses actual MP shell masses independently of this integrand
flag and is not itself a coordinate/model consistency certificate. Ordinary
successful DP behavior is preserved. Unsupported polarized/vector/other modes
retain explicit legacy behavior. No repair flag is substituted for `MP_DONE_ONCE`.

The scalar `vector.inc` emission needed by the full exporter is retained as an
explicit source addendum with a generation assertion; it is not merely an
out-of-band materialization repair. Model/ALOHA sources and the actual generated
Fortran for every arm are included as text.

## Retained execution and build history

`build/complete.json` records 37 successful owned build jobs in 477.029 seconds
under CPU14, one compiler job, a 600-second cap; no native amplitude was executed.
`build/exact-driver.json` records the separately authorized 90-second continuation
(20 jobs, 45.985 seconds) that completed the exact-word trace drivers, reusing six
already completed byte-identical outputs. The earlier include-path failure and
the earlier 600-second deadline are retained, not overwritten. The prior full
build attempts 1–7 and their failures are preserved in the prior-build bundle.

Two subsequent runtime fixture defects are retained separately. The first failed
with taskset exit 126 before entry because copied executables lacked executable
permission. The successor copies retained exact bytes and inherited the original
link output's mode. The second entered the model reader but stopped before any
WARM/RESULT record because `Cards/ident_card.dat` was absent. The successful
successor adds standard Cards paths with exact existing card bytes. Neither
failure is classified as an amplitude or reducer failure. Original paths and
records remain unchanged.

The exact-word driver records C as binary128 words, its DP view, Born values and
Laurent coefficients. `POST_OBSERVATION_FLAGS` is sampled **before** the final
`IEEE_SET_FLAG`; equality to the saved flags establishes that reference
calculation/formatting added no flags. Restoration is source-bound, not directly
sampled after that call.

The runtime controller used the executable's actual ELF `PT_INTERP --list`,
verified the pinned loader and captured actual process mappings. It used the
reviewed owned-process supervisor (WNOWAIT/deferred cleanup), exact one-entry
legacy chains, CPU14, 120 seconds per case and 300 seconds per stage. The two
successful stage wall times, 144.100 and 181.031 seconds, are protocol durations,
not performance measurements. All bound inputs, sources and native identities
were rechecked after each run and at stage completion.

## Portable contents and verification

Run from any directory with Python 3; no compiler, reducer, ignored build tree,
network access, or import of generated scientific code is required:

```sh
python3 -B /path/to/born-effective-point-native/verify.py
```

`evidence-index.json` binds the complete physical package. `sources.json.gz` is a
deduplicated JSON container of exact UTF-8 source/evidence bytes keyed by original
repository-relative path. `source-inventory.json` binds each path and blob.
`prior-build-history.json.gz` preserves the earlier build source/history, bound
by the unchanged `prior-build-index.json`. `controls/` has its own portable index
and verifier. Native artifacts are represented only by identity/size/mode;
original symlinks are represented only by target/identity metadata. No binaries,
pickles, bytecode or alias directories are shipped.

To inspect or restore selected source text without executing it:

```sh
python3 -B /path/to/born-effective-point-native/extract.py --list --prefix .tirust-build/born-effective-implementation/
python3 -B /path/to/born-effective-point-native/extract.py --destination /new/empty/text-root
```

The extractor refuses to overwrite different bytes and never restores native
files or links. Both source bundles contain original build/materialization,
source-test, controller and execution commands. The main lineage is:
`born-effective-implementation` → `born-effective-exporter-addendum` →
`born-effective-runtime/attempt1` → `driver-hex-addendum-attempt3` →
`legacy-controller4`. The frozen canonical-source record includes both Git-head
and working-tree identities; these must not be conflated. The actual generated
sources are authoritative for the compiled fixtures.

For a new scientific rerun, use an isolated checkout, restore the captured source
paths, rebuild the complete exported MODEL/DHELAS/process sources with the
captured compiler recipes and matching external dependencies, then prepare a new
runtime namespace and bind its rebuilt artifacts. The original optimized build
scripts reused hash-bound earlier artifacts; the text-only archive deliberately
does not supply those binaries. Therefore blindly running the old build script
without rebuilding its donors is not a supported portable reproduction. Retain
new build identities and re-review runtime manifests rather than rewriting the
historical hashes. Actual current TIRust dependency was the private `eb3d67fd…`
DSO; the old `49af12f0…` installed prefix must not be silently substituted.
