# Invariant-symmetry campaign numerical supervisor

This source package prepares an untimed numerical qualification for campaign
`invariant-symmetry-1`, manifest
`ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0`.
Creating and testing this package launches no native worker or compiler. The
default command prints the eleven scheduled commands. Only an explicit
`--execute`, after all five sealed `preparation-1` stages have passed, admits
numerical work. CPU 9 is the default and is not reserved or assumed exclusive.

From the repository root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-qualification-1/run.py
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-qualification-1/test_supervisor.py
```

The coordinator may subsequently launch the reviewed numerical phase with:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-qualification-1/run.py --cpu 9 --execute
```

The unchanged numerical tool runs baseline CutTools and candidate backends
1, 6, 7, 8 and 9 for each case. The order is QCD n1, EW n1, loop-induced h,
QCD n2, EW n2, mixed ga, loop-induced hh, QCD n3, EW n3, QCD n4 and EW n4.
Each command uses `--warmup-only` and its new campaign-local label
`full100-warmup-<case>-1`. The existing actual source worker supplies the final
complete steady 100-point traversal, including every Born and loop split-order
column. No timed RUN or extra classification traversal is requested. The
manifest's unchanged administrative worker limits are 3,600 seconds except
14,400 seconds for the two n4 cases.

Before admission and before/after every case, the supervisor checks all 22
builds: explicitly declared predecessor card ancestry and actual Fortran card
resolution, exact saved points and masses, campaign generation/build tags,
generated sources, executable bytes, current effective compiler policy, static
link inputs, and the complete recorded runtime library closure. TIRust's
library/archive/config/link files join by their exact installed prefix paths;
Cargo/header pins remain campaign source evidence, not runtime observations.
Baseline source identity must match the manifest-pinned baseline record and
starting revision `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`. Its exact retained
object-relink proof and predecessor generation/compilation hashes must agree,
and its non-worker numerical objects must remain unchanged. Both sources must
match their retained generation, compilation and physical-card copies. These
checks may inspect compiler recipes and dynamic dependencies during authorized
execution; none is executed by the default dry-run or synthetic tests.

Each completed lane is reparsed from its raw warmup/dimensions/READY files and
checked against its initialization copy, observation, admitted inputs and
postcheck. Comparisons are recomputed against the same case's retained baseline
CutTools rows. All six lanes must pass the existing `3e-6` criterion for a case
to pass. The ordinary `1e-8` target is reported separately and is not silently
converted into the regression criterion. A failed baseline leaves later
comparisons unavailable and the case failed. Failed and partial controller
results stay retained; no earlier failure is overwritten or promoted. Numerical
failures allow later scheduled cases; source, input, protocol, cleanup or
evidence inconsistency stops further admission.

The source seal also pins the completed preparation's source seal, shared
protocols, manifest and cases. At admission the supervisor records the exact
passed preparation status/specification hashes, all 22 build identities,
commands, allowed CPU information, configuration snapshots, worker limits and
host state. It does not call `summarize` or change the canonical pending summary,
report, shared tools, manifest or handoff. The supervisor and per-case labels
are append-only, with no overwrite, automatic retry or resume path.

The copied cleanup helper is byte-identical to the prior reviewed supervisor.
It records SIGINT/SIGTERM until child ownership is established, asks the
controller to close its worker, and then acts only on observed descendants
with matching PID/start/session identities. Observed survivors block further
admission. This remains an observed-descendant guarantee, not proof against an
unobserved child that is spawned and reparented between samples. Backend 8/9
workers retain the existing common flock; the supervisor does not take that
same lock around their parent controller.

Every observation is performance-ineligible. Its `cold_seconds` includes all
startup/warmup work and supplies no warmed runtime. FPE flags and internal
retry counts are not observed and remain null. Accepted DP/QP status describes
the returned result only. A successful numerical phase authorizes no timing,
performance acceptance or overall-goal completion. The focused tests use only
temporary text fixtures and mocked build/dependency inspection; they launch
no subprocesses or native code.
