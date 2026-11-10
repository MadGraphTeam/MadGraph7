# Hua-Sheng's local precise timing handoff — launcher correction

Precise performance acceptance is **TODO on Hua-Sheng's local machine** for
`invariant-symmetry-1`, the new source-only campaign. Its registration and pending
summary are not numerical or timing observations. All eleven cases still need
fresh local numerical qualification. The summary lists eight pending main cases;
the three controls are also unmeasured. GNU Fortran on Linux is the supported
qualification target.

The preserved [handoff 2](../../split-dp-terminal-1/timing-handoff-2/LOCAL_TIMING_TODO.md)
belongs to the older `split-dp-terminal-1` source. Its ordinary numerical results
and shared-host provisional measurements remain old-source evidence. They are
not successor results, and neither they nor any new local result erase earlier
failures. Shared-host provisional timings cannot pass the 1.05 runtime bound.
This handoff creates no new shared-host timing command.

This additive handoff 4 corrects the local installation launcher to invoke the
tracked, canonical-source-bound `bin/madgraph` entrypoint. Its SHA256 is
`375feb36e7064526010e4dd024307977f597d923ae62f9bf0b14629dbbf401c0`.
The previous [handoff 3](../timing-handoff-3/LOCAL_TIMING_TODO.md) is preserved
as historical source-review evidence; its frozen launcher referenced an absent
`bin/mg5_aMC`. Use the corrected delivered tool and this successor's commands.
The generator inventory and installation receipt schema retain their identities.
This source-only correction performs no installation, compilation or timing.

## Current source and input ancestry

The canonical registration binds source commit
`2c13246ffa63aa357ad1e29e5e79aff63d88dc7e`, including the promoted invariant-symmetry
correction and installer metadata changes. The exact generator source inventory
is `a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19`;
its numerical identity is
`f217762db032b8b755c0f62171bedcb0e9f6e09eac8128d7f8529d15e5dac8c8`.
Later documentation-only commits may retain the same source inventory. Changed
generator or installer bytes cannot use this registration.

The [manifest](../manifest.json) SHA256 is
`ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0`.
The initial [pending summary](../summary.json) SHA256 is
`02fe70b84fa3512adadf079b276e3b34ee25f9230bc855f21a3d430e6a0c2e3d`.
The explicit `physical_card_origins` field binds the immediate predecessor's
actual eleven physical cards through 66 observed input records and 22 retained
baseline/candidate card files, plus the unchanged model, policy, point streams
and metadata. This is **physical-input ancestry only**. No predecessor result,
compiled process, initialization or numerical prerequisite is imported as a
measurement for the new source. Actual generated local physical cards must match
those declared hashes. No fallback to undeclared ancestor cards is allowed.

## Prepare the local software and immutable inputs

Use a fresh clone of the delivered `madloop_tirust` branch with full history.
It must contain this campaign manifest and match its exact generator inventory.
Keep all checked-in benchmark evidence, model pins and saved points. Run every
command below from that clone's repository root, `madgraph7_madloop_tirust`.
Do not copy the shared machine's ignored object files or patch absolute paths
inside its campaign manifest.

On supported `x86_64-linux`, install Nix with flakes enabled, then enter the
repository's pinned `flake.nix`/`flake.lock` development shell. This self-contained
shell supplies Rust, GNU Fortran, Python, Typst and fonts without reference-tree
paths. `tools/tirust_env.sh` configures the shell and offers fallback discovery
of already-installed tools; it does not install missing dependencies.

```bash
nix develop
source tools/tirust_env.sh
python3 -B tools/tirust_benchmark_local.py --help
```

First reproduce the comparison libraries using the exact archive hashes and
commands in
[`qualification/comparison-backends/README.md`](../../../qualification/comparison-backends/README.md#reproduce-the-installations):
CutTools 1.9.3, Ninja 1.2.1 with OneLOop 3.6, COLLIER 1.2.9, and the existing
pinned NinjaRS dependency chain. The command sequence targets a fresh checkout.
NinjaRS's oneloopmaster revision must remain independent of TIRust's revision.
Keep its installation smoke results and runtime metadata.

Build TIRust through the new explicit launcher command. This runs the actual MG
CLI through `python3 -B bin/madgraph` into a new prefix and retains its source
inventory, exact subprocess command, log and native
artifact receipt. It never installs implicitly during measurement preparation.

```bash
python3 -B tools/tirust_benchmark_local.py install \
  --campaign invariant-symmetry-1 --installation hs-symmetry-native-1 \
  --tirust-prefix HEPTools/local-symmetry-native-1/tirust --jobs 4
```

The launcher checks the exact staged Cargo lock, TIRust revision
`854b79dcb7fffcb35e92b4a8a3dd443369b798c4`, patch hash, and all Git dependencies,
including oneloopmaster `bd703d09a78f7849a6b86a07096e6f222fc01193`. It also
checks the installed Fortran/header/Cargo sources, required ABI/precision/pole
capabilities and module/artifact hashes. New native bytes need the actual CLI
receipt; ABI-compatible metadata alone does not establish the bridge source.
The receipt is observational build provenance, not cryptographic attestation.

Select and record an allowed CPU on the local machine. The example chooses the
first allowed CPU; Hua-Sheng can replace it with a chosen physical core before
registration. Avoid changing the registered affinity halfway through the run.
The exported thread limits keep each worker single threaded. Record any local
background load and initialization costs separately.

```bash
TIRUST_LOCAL_CPU=$(python3 -B -c 'import os; print(min(os.sched_getaffinity(0)))')
python3 -B tools/tirust_benchmark_local.py prepare \
  --campaign invariant-symmetry-1 --measurement hs-symmetry-local-1 \
  --tirust-prefix HEPTools/local-symmetry-native-1/tirust \
  --install-receipt docs/benchmarkfolder/local-installations/hs-symmetry-native-1/receipt.json \
  --cpu "$TIRUST_LOCAL_CPU"
python3 -B tools/tirust_benchmark_local.py verify --measurement hs-symmetry-local-1
```

Preparation freezes a local deployment and copies the canonical inputs,
explicitly declared physical-card ancestry and failure evidence. It does not execute amplitudes or compile numerical objects.
The registry rejects a new label for the same source/policy/host/build. The old
qualified c91 QCD 2→1 failure remains retained: ratio 1.1210500947672746,
95% paired interval [1.1034925280600616, 1.138887018281026]. A different local
measurement neither erases that result nor relabels it as a pass.

## Rebuild and validate all eleven processes

```bash
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-symmetry-local-1 init
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-symmetry-local-1 model
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-symmetry-local-1 build --jobs 4
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-symmetry-local-1 numerical
python3 -B tools/tirust_benchmark_local.py verify --measurement hs-symmetry-local-1
```

`init` materializes the exact baseline
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f` and prepares its build dependencies.
`model` downloads the pinned EW archive and rejects changed bytes. `build`
generates the actual MG CLI processes with `PYTHONHASHSEED=0` and
configuration-changing `set --no_save` commands, then compiles both sources.
Baseline and candidate effective numerical compiler settings must agree.

`numerical` runs all eleven cases, baseline CutTools and candidate reducers
1/6/7/8/9, using every saved point. The warmup-only route accepts only its final
complete steady 100-point traversal; it emits no warmed timing. The frozen
worker limits are 14,400 seconds for each n4 case and 3,600 seconds otherwise.
It preserves refusals, incomplete initialization, coefficient failures and
ordinary 1e-8 target misses. The 3e-6 regression criterion remains unchanged.
No failed or incomplete case is automatically rerun under another label.

The launcher reparses all raw numerical streams and checks current compiled
sources, executable, runtime, physical cards, numerical policy and before/after
bindings before admitting precise timing. Source/compiler/dependency changes
require investigation and a new explicit deployment record, not a hidden
resume. All results remain visible.

## Perform the precise performance campaign

```bash
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-symmetry-local-1 campaign
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-symmetry-local-1 summarize
```

The timing stage uses the existing full-100-point protocol: at least seven
alternating pairs, sustained calibration with a two-second target, and the
registered 0.95-second observed batch floor. Every batch contains an integer
number of complete 100-point traversals. Cards, UV/IR poles, helicity/color sums,
rescue policy and `MLStabThres=1e-6` remain matched. Initialization plus warmup
is reported separately from steady timing.

All eight main TIRust-first chain comparisons run before expensive regressions.
For each main process the upper 95% paired candidate/baseline bound must be
<=1.05. A failed or ineligible chain remains reported; the command returns
nonzero and defers regressions. Once all eight chains pass, it runs existing
backend 1/6/7/8 regressions and descriptive backend9/control comparisons.
Main existing-backend regressions use the same speed bound. Backend9 versus
CutTools and controls introduce no extra speed gate.

Do not select a favorable rerun after an eligible failure. A timeout, numerical
refusal or failed criterion remains an open item. Accepted DP/QP histograms
count final amplitude results, not every internal reduction attempt. Separate
native-DP/rescue-cohort diagnostics must be labeled as such and do not replace
the full-stream campaign.

## Render the local evidence

The local renderer is a separate adapter; it does not run the old shared-host
report command or overwrite the canonical `docs/benchmarkfolder/benchmark.pdf`.
It reparses local numerical and paired evidence, retains missing/failed rows,
and writes an append-only local PDF with the same preserved report style.

```bash
python3 -B docs/benchmarkfolder/build_local_report.py \
  --measurement hs-symmetry-local-1 --render-label render-1
```

The output is
`docs/benchmarkfolder/local-measurements/hs-symmetry-local-1/render-1/benchmark.pdf`,
with its Typst sources, verified dossier, raw-input hash ledger and build receipt.
Use a new `render-N` label for a later refresh; existing render directories are
never overwritten. `--data-only` freezes the same verified dossier without
invoking Typst. The `forward summarize` command produces machine-readable JSON
and does not compile a PDF.

The renderer requires the sealed complete numerical preparation, but it can
render pending, failed or ineligible timing rows. Every retained local attempt
and inherited qualified failure remains visible. Main-chain and existing-backend
speed gates are distinct from descriptive backend9/control ratios. A local PDF
never declares the overall integration goal complete.

## Remaining work

- Complete the local software installation and source/deployment registration.
- Complete all eleven six-lane numerical qualifications on the local build.
- Complete all eight main chain performance gates and required existing-backend
  regressions, preserving all failures and actual precision usage.
- Render and independently review the local raw-evidence report. Report local
  runtime acceptance separately from any still-open physical stress or backend
  qualification limitations; no timing-only result completes the whole goal.
