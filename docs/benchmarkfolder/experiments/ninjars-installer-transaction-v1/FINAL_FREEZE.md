# Final source freeze and validation handoff

This is a source-only checklist, prepared during the allocation timing window.
It starts no compiler, reduction engine, installer or benchmark. The assigned
goal remains active until the numerical/performance gates and final report
pass. Root alone selects proposals, commits, pushes and declares completion;
opening a PR awaits the user's later instruction.

## Freeze one production candidate

1. Finish the two existing proposal decisions: the exact tensor allocation
   patch and the pinned-upstream native Float kernel policy. Promote only the
   measured, reviewed choices. Preserve negative and inconclusive experiments.
   Do not independently advance Symbolica to `9cdc2949`: NinjaRS and
   oneloopmaster still explicitly require `e679e8c1` at this checkpoint.
2. Include the engine-free panic regression, the unavailable-backend diagnostic
   guard, and the independently reviewed installer publication transaction fix.
   The installer transaction passed its engine-free red/green proof on
   2026-10-21: three regressions reproduce the old failures, and all 37 focused
   tests pass for source `e2b83749` and tests `0c0a2913`. Evidence is retained
   under `docs/benchmarkfolder/experiments/ninjars-installer-transaction-v1`.
   The actual final-source CLI installation and relocated export gates below
   remain required.
3. Commit all production and durable test inputs on `madloop_ninjars`. Keep
   the original `27147b3300f556a75661a886e92bdbe00558ee25` archive unchanged.
   Record the full final commit, tree, Cargo manifest/lock, toolchain versions,
   effective build flags and bridge source inventory. Continue read-only
   upstream checks between campaign checkpoints; stop and review a NinjaRS or
   oneloopmaster update rather than moving a floating pin.
4. Freeze generator inputs separately from the Rust numerical library. The
   generator fingerprint includes `madgraph`, `aloha`, `Template`, `bin`,
   `models` and `input`. The diagnostic and installer fixes change that
   fingerprint even if the generated Fortran happens to be identical. Record
   the actual final CLI-generation revision; never relabel an f4b generation
   with the current HEAD. Rust source is additionally bound through its full
   source inventory and the actual archive/interface/link identities.

## One final Rust test executable

Use a new namespace, proposed as
`.benchmark-work/policy-v2/final-rust-abi3-v1`, with evidence under the matching
`docs/benchmarkfolder/policy-v2/final-rust-abi3-v1`. Compile the final committed
bridge once with its ordinary test modules and the `fenv-test-support` feature.
Any fixture/replay additions must be test-only code present before this build;
the final numerical code must match the production source inventory. Root's
explicit MP handoff is required for execution, one executable at a time.

The existing build command is reusable after setting the new isolated target
and build directories and choosing an allowed compiler CPU:

```sh
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
export CARGO_TARGET_DIR="$PWD/.benchmark-work/policy-v2/final-rust-abi3-v1/target"
export CARGO_BUILD_BUILD_DIR="$PWD/.benchmark-work/policy-v2/final-rust-abi3-v1/build"
export CARGO_BUILD_JOBS=1
taskset -c "$NINJARS_FINAL_CPU" cargo test \
  --manifest-path vendor/ninjars/Cargo.toml --locked --offline --release --lib \
  --features fenv-test-support --no-run --message-format=json
```

Save Cargo JSON and stderr. Resolve the sole test executable from Cargo's
`compiler-artifact` message, record its hash/shared libraries, and run that
exact path for the entire suite:

```sh
taskset -c "$NINJARS_FINAL_CPU" "$NINJARS_TEST_BINARY" --list
taskset -c "$NINJARS_FINAL_CPU" "$NINJARS_TEST_BINARY" --test-threads=1 --nocapture
```

The final executable must cover ordinary ABI/error/rebinding/fenv tests,
binary128 dyadics and signed-zero branch lips, the new panic regression,
retained analytic and one-loop fixture controls (including loop-induced
fixtures), all 1,392 fixed requests and all 11,136 formatted fields, plus
256/384-bit convergence for difficult cases. Retain every difference in value,
precision, sign, status and output; a selected native policy may intentionally
change arithmetic and must be assessed against convergence rather than silently
rewriting controls. Preserve the documented distinction between the input's
113 bits and arithmetic's working precision.

The production ignored replay already has an exact entry point:

```sh
export NINJARS_TRANSPORT_REPLAY="$PWD/.benchmark-work/experiments/ninjars-native-float256-v1/actual-mp-requests.jsonl"
export NINJARS_TRANSPORT_REPLAY_OUTPUT="$PWD/.benchmark-work/policy-v2/final-rust-abi3-v1/state-replay.jsonl"
taskset -c "$NINJARS_FINAL_CPU" "$NINJARS_TEST_BINARY" --ignored --exact \
  transport_replay_tests::all_stored_requests_retain_outputs_after_transport_fix \
  --test-threads=1 --nocapture
```

Resolve any other ignored fixture/convergence test names from the saved final
`--list`. Do not assume that the historical private tensor/native research
test names exist in the final production test binary. The combined-correctness
runner at `experiments/ninjars-mp-tensor-allocation-transport-v1/run.py` is a
useful model, but its hardcoded 22/18 counts and private capture module names
must be replaced with the actual final test list. Avoid building separate
example binaries or reusing an older ordinary-test binary as final proof.

## Actual install, fresh export, relocated execution

The existing, successful harness is
`docs/benchmarkfolder/policy-v2/install-export-abi3-zero-sign-v1`:

- `prepare.py` archives a committed source and executes the actual repository
  `bin/madgraph` command card containing `install ninjars --prefix=... --jobs=1`.
  The installation prefix stays inside the private archived checkout, so no
  user's configuration or original baseline is changed.
- `export.py` uses actual CLI cards to generate standalone and NLO outputs with
  internal dependencies from that source and installation.
- `runtime_validator.py` relocates both outputs, hides the private installation,
  checks GNU binary128 helper linkage, executes standalone DP110/QP120/stable218,
  and compiles NLO with twenty pole cancellations and the soft/collinear check.
  This is a fixed-order validation; it does not claim shower validation.
- `focused_validator.py` connects source-bound bridge tests to the installed
  artifact and focused MG7 Python tests. `verify.py` seals retained outputs and
  can rehash the still-preserved executables without running an engine.

**Do not run these immutable old scripts in place.** Copy the five scripts into
a new namespace, proposed as
`docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1`, then make these
bounded adaptations before any final command:

1. Replace the hardcoded `zero-sign-v1` work/report namespace and old default
   revision with the actual final commit.
2. In `prepare.py`, replace the old transport-only proof and `proposed_files`
   identities with the complete final Rust proof's production build inputs.
   Its old source-equality guard intentionally rejects later Rust changes.
3. In `focused_validator.py`, bind the new one-executable final Rust evidence;
   replace literal `18 passed`, `16 Python tests` and the old source map. Add
   the new MG7 boundary/upstream tests and transaction tests to the focused
   command. Determine counts from successful discovery/output, and retain the
   exact selected test names. Keep the fresh installed smoke and relocated QP
   execution; a test binary is not the installed release archive.
4. In `verify.py`, update the expected revision, exact source/proof inventory,
   final observed test counts and final archive/interface identities. Keep
   the numerical, dependency-copy, absence and linking checks intact.
5. Keep installer release invocation faithful to `vendor/ninjars/Makefile`:
   `cargo rustc --release --locked --lib -- --print native-static-libs`.
   The allocation study caught a different Cargo profile when substituting
   `cargo build`; do not equate such artifacts. Retain effective flags and
   Cargo fingerprints. A cache may be reused only with the verified source,
   features and dependency bindings.

After that adaptation and a committed final candidate, the existing harness
invocation sequence is:

```sh
source .ninjars-build/dev-env.sh
taskset -c "$NINJARS_FINAL_CPU" python -B \
  docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/prepare.py \
  --revision "$NINJARS_FINAL_REVISION" --cpu "$NINJARS_FINAL_CPU"
taskset -c "$NINJARS_FINAL_CPU" python -B \
  docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/export.py
taskset -c "$NINJARS_FINAL_CPU" python -B \
  docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/runtime_validator.py \
  --build --hide-installation --run --collect
taskset -c "$NINJARS_FINAL_CPU" python -B \
  docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/focused_validator.py
python -B docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/verify.py --live
```

These paths are proposed destinations, not a claim that the adapted final
harness already exists. Its command cards are retained by the harness.

## Reuse and final measurement limits

The old corrected installation archive `832fc0d1...` and its 63-file export
proof remain historical evidence. Neither that archive, the initial `1199bc84`
archive, nor either unselected experimental archive is the final runtime after
promotion. Preserve them; do not overwrite their seals. Refresh final candidate
builds and copied dependency modes against the newly installed archive.

Reuse existing original/corrected-original legacy reference campaigns only
when their inputs, correction, complete point streams, source/library signatures
and execution policy match the frozen role. No duplicate reference campaign is
requested by this checklist. Refresh final candidate core and supplemental
workers using actual final CLI generation and record the true generation
provenance before compiling or measuring. Revalidate representative persistent
and fresh driver agreement, full numerical library signatures, source/interface
and link inputs; bind any immutable validator copies rather than a report tool
that is still being edited.

Complete the eight core processes, loop-induced measurements, numerical
stability comparisons, changed scales/masses, complex masses, directions,
rotations, forced rescue and near-singular controls. Report pairwise observed
disagreements separately from MadLoop's own accuracy estimates and actual
backend/rescue counts. All final performance samples use the retained fixed
streams, normal stability/rescue and common pole settings. Every core process
must meet the predeclared original-Ninja 1.05 gate including its confidence
interval. A failed or unresolved gate stays outstanding work.

Finalize the machine-readable dataset/provenance and the attributed Typst
`benchmark.typ`/`benchmark.pdf`; compile reproducibly and visually inspect the
PDF. Remove accidental build products, ELF/object/module files and unrelated
temporary files from Git staging while retaining intentional, sealed research
inputs/logs and ignored working data needed by still-running campaigns. Push
the validated milestones on `madloop_ninjars`. Mark the goal achieved only if
all required gates pass, then wait for the user's instruction to open a PR.
