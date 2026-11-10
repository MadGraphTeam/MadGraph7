# Proposed current-source QCD n3 raw stability pilot

This is a source-only proposal for `u u~ > z g g [virt=QCD]` in `loop_sm`,
using the `entry-latch-cache-guard-1` source. The benchmark owner has now closed
its QCD n3 checkpoint, and root has permitted read-only donor admission.
No private diagnostic driver has been built and no current-source raw worker
has been qualified or executed. Read-only permission grants no compiler,
loader or amplitude execution.

The controller must authenticate the separate `closed-input.json` source
payload through the reviewed source seal, verify its exact owner-checkpoint
SHA256 and require `status: closed` **before opening any export path**. That
payload pins the canonical export and its generation, compilation and
executable identities. Root reviews the exact source seal and closed-case
binding before compiler, loader or amplitude execution. The binding preserves
numerical failures from the closed benchmark; closure itself is not numerical
acceptance. A successful build alone would not permit reading an export whose
numerical stage remains active.

The proposed run attempts five fresh private workers once, in order:
CutTools `1`, Ninja `6`, COLLIER `7`, NinjaRS `8`, and TIRust `9`. Each worker
uses all 100 saved points and the exact sealed source5 diagnostic driver. Its
three probes are the original numerator/loop direction in mode 1, reversed
loop direction in mode 2, and the exact signed spatial permutation
`(px, py, pz) -> (-py, px, pz)` in mode 1. The complete schedule contains
1,500 primary public calls, plus at most 16 bootstrap calls per worker
(80 total). Partial, failed and unexecuted calls retain their places in the
fixed population.

The point stream is reused byte for byte from
`raw-dp-pilot-5-1/points/qcd_n3_uux_zgg.dat`, SHA256
`b46ce200bb330826d2a4cff69f61f828d3105c0566444fa548b1de75622ee8dd`.
Its metadata SHA256 is
`e4a262966bc0de149e3b16a56f85058c56ba068ead2b73f3586101ff38c320e4`.
These are 100 previously frozen physical points at 1 TeV, with final-state
`pT > 50 GeV` and pairwise `Delta R > 0.5`, using strict cut boundaries.
Incoming masses are zero; final-state masses are `[91.188, 0, 0] GeV`.
The current export's resolved physical card must match the metadata's
`5a535b7044777bd87c7088dc14bf9605d165ebae843dbcbde91a02f545c05433`
digest and its compiled mass probe must agree. No new points, selected
subsets, replacement points or resampling are proposed.

The source binding pins the current campaign manifest to
`4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076`,
the source5 worker to
`371f8ccb72a5ee3dc1bad628cc855e239c7d0c26d5f30b5ff652f15afc564cc9`,
and the additive classifier's index to
`7bc7a08c2a96483100ea47c2d652694a34a4f426af2b5ba9a8317798d6003032`.
Admission must join the complete exporter and numerical source inventories,
generation and full rebuild receipts, actual generated matrix, process
prefix/dimensions, initialized coefficient layout, exact rendered driver,
private executable, runtime closure and observed application mappings. It
must also authenticate the private tree, finite external link closure,
physical-card search, all runtime resources and preservation of the canonical
donor and production sources. The classifier's static checks require the
actual generated public-entry reset and qualified wrapper; the live template
hash alone cannot qualify a worker.

The controller reuses reviewed low-level source5 helpers for private copies
and link guards, object/include prerequisites, group layout, explicit card
write/readback, physical-card resolution, bounded process ownership and
runtime maps. It must use a fresh work namespace. It does not reuse source5's
old campaign admission, donor selection, source3 inheritance, four-process
protocol or high-level prepare/run stages. The pinned link-closure and
process-cleanup helpers remain unchanged. A source-bound adapter changes only
the supervisor's taskset CPU and matching recorded CPU from 12 to 13; it
requires exactly those two AST constants and preserves the launch gates,
cleanup and environment. Numerical objects come from the newly closed,
authenticated full build. Preparation compiles only the private
`check_sa.f` diagnostic driver and links it against those objects; no library
or generated-amplitude rebuild is proposed.

The explicit cards retain finite terms and all Laurent coefficients:
`LoopPoleMode=-1`, `TIRustComputeFinite=.TRUE.` and
`UseQPIntegrandForTIRust=.TRUE.`. Initialization uses mode 1; primary calls
use only modes 1 and 2, with exactly one selected backend and its zero
sentinel. Forced stability checks and automatic rescue are disabled.
`TIRustGlobalCache=0` and `COLLIERGlobalCache=0` avoid cross-call integral
cache reuse in this diagnostic. COLLIER UV/IR poles are enabled, its internal
stability test is disabled, and its required accuracy is `1e-13`.
Helicities and colors remain summed using the canonical normalization.

The additive classifier can ignore an incoming software failure latch only
for the authenticated current-source worker when both outgoing latches are
clear and its group journal has no parse issues. It retains the raw flags,
legacy decisions, active-group failures, nonfinite coefficient/result bits,
coverage, status, warnings, bootstrap readiness, split-order consistency and
completeness gates. A failed tensor remains unavailable. Existing source5
journals, figures and unknown bounds remain unchanged.

An unchecked host return code `110` is not a checked accuracy certificate or
an observation of the internal reducer or precision. The proposed result is
a host forced-DP internal consistency diagnostic based on the relative spread
of finite/Born across the three probes. It retains Born, finite and pole
coefficients, complete console output, failed groups and unavailable reasons.
No hardware IEEE exception state is observed or reset. No independent
true-error or runtime acceptance claim follows from this pilot.

The fixed execution limits are:

- One diagnostic-only make invocation, with at most its compile and link
  commands, capped at 120 seconds including cleanup; one exact `PT_INTERP`
  loader-list query capped at 30 seconds.
- Five native workers, each capped at 180 seconds including cleanup, using
  CPU 13 without reservation and a 4 GiB address-space limit. Read-only
  admission must confirm CPU 13 is in the allowed affinity set.
- Backends 8 and 9 acquire `.tirust-build/numerical-tests.lock`, waiting at
  most 1,800 seconds outside their execution caps. Native library search
  variables are applied only immediately before the target `execve`.
- No automatic retry, in-place restart, extra worker, point replacement,
  implicit precision rescue or numerical rebuild fallback. Every timeout,
  nonzero exit, guard failure and unexecuted request is retained.

The intended CLI, from the required checkout, is:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-pilot-proposal-1/controller.py plan
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-pilot-proposal-1/controller.py readonly
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-pilot-proposal-1/controller.py prepare --execute
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-pilot-proposal-1/controller.py dryrun
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-pilot-proposal-1/controller.py run --execute
```

These commands describe the proposed interface and record no execution.
`plan` is the inert default. `readonly` validates the sealed closed-input
binding and donor without launching children or creating the work namespace.
After a matching preparation, `dryrun` exercises the five private worker
copies, cards, physical-card lookup and guards without launching a child.
`run` requires the matching sealed preparation and prelaunch checks.

Any later dataset or figure must be an additive current-source artifact,
retain all 100 points and unavailable bounds, and carry the source and runtime
binding. Canonical report promotion requires separate root review. Precise
performance timings remain Hua-Sheng's local-machine TODO.
