# Current-source internal relocation and NLO deployment proposal

Prepared for root review only. No CLI generation, compiler, linker, native
library load or amplitude has run. This is a small repeat of the historical
`qualification/madloop-adapter/release` check using the unchanged generated
`check_sa.f`, not a new benchmark harness.

`bindings.json` binds all 2,664 registered exporter paths at revision
`0fcd2cd873d61d134837e1d9e87afb13a47824fd`, inventory
`afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6`.
The native DSO is the installed `eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9`.
NinjaRS and all installed libraries remain unchanged. Source/tool/library and
1,311 protected file/symlink/absence records precede execution.

## Source and configuration isolation

`prepare_source.py` obtains an exact Git archive of the seven exporter roots
plus the auxiliary CLI roots listed in that script. It materializes regular
files and safe in-archive symlinks into the private `mg` directory. The 291
registered optional IREGI files are retained under `omitted-optional/IREGI`,
with their original names and hashes inventoried, rather than deployed under
`mg/vendor/IREGI`. All other existing source content remains byte-identical.
The two registered dangling generated symlinks remain explicit omissions.

The IREGI exclusion is necessary because its legacy internal build wrapper
addresses `$(PWD)/vendor/IREGI/src` and can rebuild the original installation.
The ordinary private CLI therefore disables that absent optional backend.
This is a backend 1/8/9 deployment test. Ninja/COLLIER are unavailable in
internal exports by the existing exporter policy. No exporter monkeypatch,
IREGI fix, or claim of all-backend internal availability is introduced.

The private CLI receives a private installation configuration and private XDG
directories. Every CLI setting uses `--no_save`; `MADGRAPH_BASE`, `PYTHONPATH`
and runtime-library overrides are unset. Only exact private copies of the
TIRust and NinjaRS prefixes are exposed to it. CutTools builds from the private
archived source; no original native object is rebuilt or modified.

## Exact bounded work

All orchestration commands run from the original repository root.
`commands.json` supplies actual argument arrays, paths, environment and bounds.
Build/generation: CPUs 20–21, at most two jobs, 16 GiB child address-space cap,
core dumps disabled, 2,400 seconds total, each step capped separately and
15 seconds reserved for process-group cleanup. No retries. A source/config
change, failed step or uncertain child cleanup stops qualification and is
retained. CLI `output` can compile libraries, so generation is part of build
admission.

1. `python3 -B .../proposal/prepare_source.py` performs pure source/prefix
   copying and pin verification; it does not run the CLI or compiler.
2. Actual private `bin/madgraph --plain` exports `u u~ > z [virt=QCD]` standalone
   and `u u~ > z [QCD]` FKS, both with internal dependencies.
3. Verify copied native archives/modules/DSO/metadata/notices. Remove only the
   exact inventoried private `source-prefixes` directory, then move both outputs
   under `relocated/` before the final process build. Preserve original output
   paths as absent. Build using the normal generated makefiles.
4. Standalone: `make -C .../relocated/standalone/SubProcesses/P0_uux_z -j2 check`.
   NLO: separate `libcuttools` with `-j1`, `libmodel`, `libdhelas`, virtual
   `libMadLoop.a`, then the tiny
   historical `adapter-check.mk` target, which includes the generated makefile
   and uses its actual `LINKLIBS`. No PDF/Monte Carlo target is built.
5. Retain effective compile/link recipes, generated Fortran/model/card hashes,
   archives, executable hashes, `readelf` dependencies/RPATH and resolved runtime
   files before any numerical call. Verify both changed generated procedures
   came from the current templates. No source retouch after source capture.
6. Only after root releases the final main EW priority, acquire the existing
   `.tirust-build/numerical-tests.lock` (120-second cap). Run `smoke.py` once on
   CPU 20, 16 GiB cap, 900-second total deadline and 15-second cleanup reserve.
   Each of nine successful-mode workers has 120 seconds; the partial guard has
   60 seconds. Lock timeout or any worker failure remains visible and stops.
   Each process uses a new process group that must be terminated/reaped on
   timeout; only that group may be signalled.

Standalone modes are `(1,1),(8,1),(9,1),(9,2),(9,4),(9,5)`. NLO modes are
`(1,1),(9,1),(9,4)`. The final NLO request sets backend 9, fixed mode 1,
`LoopPoleMode=0`, and `TIRustComputeFinite=False`: exit 1 and the established
incomplete-amplitude message are required. Cards are restored byte-for-byte.

The existing driver evaluates its deterministic on-shell 2→1 point four times,
but only the final `result.dat` is independently recorded. Expected final
return codes are unchecked 110/120 with accuracy −1. Compare finite Born and
Laurent coefficients with each export's own same-input CutTools answer using
`3e-6`, retaining the separate `1e-8` target. Retain all normalized and split-order
fields; do not use the historical coefficients as independent truth.

Runtime loader diagnostics use `LD_DEBUG=libs,files` with a private output path;
`LD_LIBRARY_PATH` and `LD_PRELOAD` remain unset. The actual loader must resolve
`libmadloop_tirust.so` to the relocated `eb3...` copy. Every resolved system
provider is bound; no runtime-timing claim is made. Preserve both relocated
notice/provenance bundles, all logs, cards, outputs and failures. Recheck all
protected installation/configuration/source pins afterward.

This qualifies fixed-mode internal deployment, relocation and the FKS virtual
entry-point guard. It does not qualify independent stability, automatic rescue,
general physical accuracy, Monte Carlo integration or runtime performance.

`build.py --execute-build` is the thin build-only controller. Its child ownership
implementation is imported unchanged from the pinned historical
`process_cleanup.py`; its stage timer is the exact `StageDeadline` class extracted
from the pinned historical preparation controller, with the new declared caps.
`build-admitted.json` must inventory the fresh generated numerical/model sources,
objects, archives, modules, make recipes and runtime cards, plus prebound dynamic
providers. `smoke.py` validates this receipt before/after the nine fixed-mode
workers and partial guard. Loader observations must agree with those prebound
provider hashes, and the relocated DSO must be a regular file inside its export.

Pure preparation already completed: 2,860 private source files and 291 separately
retained IREGI files. `preparation.json` records the archive and copied-prefix
identities. Original and copied NinjaRS `libninjars.a` are 298,293,528 bytes with
SHA256 `0edfa89d7d83161a6ce3fdf0bb1fbca928b73d2b5ec44211a3b2efa186a155d6`.
No build/native command has run. The parser/loader/failure-retention tests use
historical text, fabricated bytes and mocked processes only.
