# Hua-Sheng's local precise timing handoff

Precise performance acceptance is **TODO on Hua-Sheng's local machine**. Shared
host measurements are provisional and cannot pass the 1.05 runtime bound. The
older `timing-qualification-1` quiet-window controller remains unrun. This
handoff uses a separate local registry and does not alter historical evidence.
The supported qualification target is GNU Fortran on Linux.

## Prepare the local software and immutable inputs

Use a fresh clone of the delivered `madloop_tirust` branch with full history.
Keep all checked-in benchmark evidence, model pins and saved points. Run every
command below from that clone's repository root, `madgraph7_madloop_tirust`.
Do not copy the shared machine's ignored object files or patch absolute paths
inside its campaign manifest.

```bash
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
CLI into a new prefix and retains its source inventory, command, log and native
artifact receipt. It never installs implicitly during measurement preparation.

```bash
python3 -B tools/tirust_benchmark_local.py install \
  --campaign split-dp-terminal-1 --installation hs-native-1 \
  --tirust-prefix HEPTools/local-native-1/tirust --jobs 4
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
  --campaign split-dp-terminal-1 --measurement hs-local-1 \
  --tirust-prefix HEPTools/local-native-1/tirust \
  --install-receipt docs/benchmarkfolder/local-installations/hs-native-1/receipt.json \
  --cpu "$TIRUST_LOCAL_CPU"
python3 -B tools/tirust_benchmark_local.py verify --measurement hs-local-1
```

Preparation freezes a local deployment and copies the canonical inputs and
failure ancestry. It does not execute amplitudes or compile numerical objects.
The registry rejects a new label for the same source/policy/host/build. The old
qualified c91 QCD 2→1 failure remains retained: ratio 1.1210500947672746,
95% paired interval [1.1034925280600616, 1.138887018281026]. A different local
measurement neither erases that result nor relabels it as a pass.

## Rebuild and validate all eleven processes

```bash
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 init
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 model
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 build --jobs 4
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 numerical
python3 -B tools/tirust_benchmark_local.py verify --measurement hs-local-1
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
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 campaign
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 summarize
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
  --measurement hs-local-1 --render-label render-1
```

The output is
`docs/benchmarkfolder/local-measurements/hs-local-1/render-1/benchmark.pdf`,
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
