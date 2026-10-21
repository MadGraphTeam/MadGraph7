# Publication and cleanup audit

This is a bounded publication review, not numerical or performance validation.
The initial read-only snapshot was taken at
`45375023e96b8fce0340ce14c93609ed4dcda01f`, before the allocation process-timing
quiet window. No tests, compilers, numerical engines, deletions or process signals
were run for this audit. Files owned by active agents can change after this
snapshot; their final sealed manifests decide publication scope.

## Findings from the initial inventory

- All 26,714 tracked paths were inspected by metadata and an eight-byte header
  read: 26,669 regular files and 45 symlinks. No tracked ELF executable, object,
  static archive, Python bytecode or cache path was found.
- The sole tracked build-extension exception is the 1,986-byte `mninjars.mod`
  under the sealed copied-installation proof. It is intentional compiler-module
  evidence and must remain with that proof.
- One tracked file exceeds 50 MB: the 65,567,980-byte
  `policy-v2/reference-diagnostics/qcd4-ct-warmup-20261020/profile.perf`. Its existing
  manifest records it as profiling evidence. It is not a disposable executable;
  do not remove it or rewrite its seal during general cleanup. Compression would
  require its own reviewed provenance-preserving change.
- There were 947 untracked files outside ignore rules. None had ELF/archive
  magic or an object/library/module/bytecode/cache path. Most belong to active
  research, reference, supplement or report namespaces. This count is a snapshot,
  not a claim that every untracked file is temporary.
- There are 195 existing ELF experiment executables protected only by local
  `.git/info/exclude`, which is not published. Their basenames are `fallback`
  (9), `traps` (36), `timing` (8), `oracle-check` (48), `guards` (21), `sticky`
  (13), `zero-work` (12), and `kernel` (48). Publish portable ignore rules for
  these outputs, or their exact anchored paths, before a broad documentation add.

Two existing binding namespaces already reference 136 content-addressed runtime
metadata assets that were untracked at the initial snapshot: 68 per namespace,
416,123 bytes each. All were present, every content hash matched, and no extra
asset was found. Each set includes 65 Fortran `.mod` metadata records and three
other runtime records. Their content-addressed filenames hide the original
module suffix; they are intentional provenance rather than redistributable build
libraries. A clone lacking these bytes cannot verify the already committed
bindings. Root approved explicitly staging these 136 exact assets with this
inventory while retaining the existing bindings unchanged. Do not sweep them
away as untracked build products.

On explicit resume, `inventory_runtime_assets.py` completed on CPU 4 at
revision `956b4e10584e0d18d34547558f2c38808d4b2703`. The exact inventory is
`runtime-assets.json`; the 136 paths approved for explicit publication are also
listed in `runtime-asset-paths.txt`. All 136 were still untracked and unindexed,
all hashes matched their already committed binding records, and their combined
size is 832,246 bytes. The helper checked only those bindings/assets and the 195
explicit local executable exclusions. It did not rescan the full source tree.
`local-executable-paths.txt` retains those exact ELF paths;
`portable-ignore-rules.txt` retains the eight proposed anchored rules.

## Bounded disposition of current untracked work

| Scope | Recommendation |
|---|---|
| `experiments/ninjars-mp-tensor-allocation-process-v1` (271 initial files) | Preserve through final measurements and root acceptance. The owner identifies `source`, `exact-build`, `package-exact`, `processes-final`, `bound-support`, final preparation/comparison/timing/verifier scripts, frozen bindings and independent audits as definitive. Earlier build/preparation/readiness attempts explain pre-sample failures and must not be silently removed. Publish only the final manifest's explicit files. |
| `experiments/ninjars-native-float256-v1` (125 initial files) | Preserve until the native study seals. The owner will retain all failed attempts and numerical classifications, compress both raw streams and two large deterministic analyses with compressed/plaintext hashes, and provide a materializer. Keep the required plaintext copies ignored locally. No cleanup before that handoff. |
| `policy-v2/reference-runs/*` (322 initial files) | Preserve both interrupted and fresh reference namespaces. Partial or pause-spanning intervals remain diagnostics; do not convert them into accepted timings. The reference owner decides the final retention seal. |
| `policy-v2/runtime-bindings-*/runtime-assets` (136 initial files) | Publish only the exact approved asset inventory described above. |
| `policy-v2/final-supplement-{plan,generation}-v1` | Active reproducible generation/provenance; retain and publish its final explicit manifests after source/runtime binding. |
| `policy-v2/final-report-pipeline-v1`, report source/test edits | Active report implementation/proof, not general cleanup candidates. Wait for the report owner to seal and identify the final PDF. |
| `final-numerical-audit-v1` | Independent audit evidence and held test patch; retain through root review. |
| New supplement/precision tools and unit tests | Intended source additions, not build output. Publish after their owners' test/review milestone. |
| `vendor/StdHEP/fail` | Preexisting user/workspace file. Leave untouched. |

The default ignored roots `.benchmark-work`, `.ninjars-build`, `HEPTools`,
`UNITTEST_proc`, Cargo targets, objects, archives, modules and Python caches remain
local. Never blanket-delete these roots: they contain the literal baseline,
required private source copies, active process trees, cached dependencies and
provenance inputs. Confirm each owner's final seal and that no process uses a
specific subtree before selecting it for removal.

The three scratch namespaces owned by the completed broad-regression review
(`.benchmark-work/final-regression-review-v1`,
`.benchmark-work/final-regression-fks-review-v1`, and
`.benchmark-work/final-regression-fks-review-v2-absent`) contain generated private
comparison copies and replay output. Their required evidence is sealed and
committed in `policy-v2/final-regressions-v1`; no live child remains. Root may
remove these three exact scratch subtrees after validating that milestone. No
other private source or test temporary directory is included in that permission.

## Source-review test gaps

The current public interface already has compiled Fortran ABI/status/reset
controls, staged installer smoke/failure/retry tests, source-override isolation,
missing-backend diagnostics, and raw evidence/report checks. Broad MG7 regression
results remain the separate 90-case source-bound campaign; this audit does not
rerun or extend it during timing.

One concrete installer gap was found at the initial source checkpoint:
`ninjars_install.install` rolled back primary destination publication, but its
transaction guard ended before publishing common-prefix aliases and deleting the
stage. The existing test injected failure at the primary destination replacement,
not the subsequent alias replacement/directory creation. Root confirmed the gap
and assigned the narrow fix plus failure/retry tests to the integration owner.
That owner's final passing evidence is required before considering the gap
closed; this audit performs no installer edit.

The supplement's `point_metadata` tests cover nonfinite momentum components and
conservation/shell residuals. Static inspection found no direct nonfinite-mass or
negative-energy input control in that function/test pair. The frozen generated
benchmark cards and streams do not use those inputs. A focused parser test for
nonfinite masses and negative external energies would clarify that public helper's
boundary after the quiet window; this observation does not invalidate measured
fixed inputs or justify changing any numerical acceptance threshold.

Report tests already distinguish estimated stability from observed discrepancies,
reject forged summaries/source/runtime mismatches, and keep missing supplement or
runtime/precision proofs pending. The final report still needs its actual complete
core/loop-induced/threshold data, exact final runtime proofs, reproducible compile
and visual inspection. Unit tests alone do not close those campaign gates.

## Portable ignore proposal

These eight rules preserve all currently known executable exclusions without
hiding source files with extensions. Review them against the exact executable
inventory before adding them to the repository `.gitignore`:

```gitignore
/docs/benchmarkfolder/experiments/**/fallback
/docs/benchmarkfolder/experiments/**/traps
/docs/benchmarkfolder/experiments/**/timing
/docs/benchmarkfolder/experiments/**/oracle-check
/docs/benchmarkfolder/experiments/**/guards
/docs/benchmarkfolder/experiments/**/sticky
/docs/benchmarkfolder/experiments/**/zero-work
/docs/benchmarkfolder/experiments/**/kernel
```

No ignore file, production source, existing evidence package or user file was
changed by this audit. The targeted exact inventory and this audit are now sealed. Root must review and
apply the eight rules and exact asset list; this audit does not change ignore
rules or stage any files. `verify.py` checks the audit file set, the existing
committed binding snapshots, the 136 asset bytes, and that the proposed rules
cover all 195 recorded executable paths. It does not require local disposable
ELFs to survive later cleanup.
