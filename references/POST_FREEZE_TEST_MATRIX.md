# Post-freeze validation matrix

This source-only handoff selects existing campaigns; it starts no test, build,
installer or engine. Root freezes the numerical choices first, then records the
actual final commit, generator tree and installed runtime. The detailed command
handoff remains `references/FINAL_FREEZE.md`. Preserve earlier seals and failed
attempts; create a new namespace for any refreshed proof.

| Evidence / selection | Current demonstrated scope | Required final disposition |
|---|---|---|
| `policy-v2/final-regressions-v1/validation.json` | 90 MG7 cases: 89 passed, one explicit existing GoSam Python 3 skip. Includes 69 polynomial/compiled Fortran/IEEE/HELAS cases, two loop-exporter IO cases and 19 FKS cases. | Compare all 102 tested production-source hashes with the final candidate generator map via `report_snapshot`. If they match, reuse the proof with its exact scope and skip. A mismatch needs assessment and a new relevant test proof; never relabel the old source. |
| `experiments/ninjars-installer-transaction-v1` | 37 focused installer/statistics/boundary/upstream tests passed; three original transactional failures retained. Compiler/engine calls mocked; filesystem publication and rollback real. | Confirm final installer and selected test hashes match this sealed proof. Include the real final-source CLI install and relocation checks below; mock tests cannot certify a compiled installation. |
| `policy-v2/report-stress-extension-v1` | 40 report tests passed, including synthetic Typst layout and PDF reproducibility. This supersedes the earlier renderer's software scope without rewriting its 36-test seal. | Retain the independent combined report/stress/precision-reader run when its owner completes it, with exact tested module/template hashes and observed counts. After source changes, refresh only affected suites. Synthetic layout tests establish no numerical gate. |
| Supplement / precision / stress reader tests | `test_ninjars_supplement`, `test_ninjars_precision`, plus the stress reader tests selected by their owner; prior independent report/supplement/precision run passed 55 cases at its own hashes. | Bind the final reader versions and combined run. Keep missing-data, stale-runtime, nonfinite/error, identity, rotation and scale-restoration assertions. Do not add unrequested research or count overlapping suites twice. |
| Proposed `policy-v2/final-rust-abi3-v1` | Not yet built: ordinary final Rust tests, engine-free panic control, all seven tensor fixtures/21 lanes, full 1,392-request replay, binary128/sign/branch controls and difficult-point 256/384 convergence. | Build one exact final committed test executable, save its discovered test list and hashes, and execute serially under the explicit MP handoff. Counts follow actual final discovery. Numerical code must match installed production sources; test-only additions are distinguished. Keep all raw differences/statuses/precision/signs. |
| Proposed `policy-v2/install-export-abi3-final-v1` | Not yet created: actual `install ninjars`, compiled installed Fortran smoke, fresh SA/NLO CLI exports, relocation and hidden-installation runtime checks. | Adapt the existing zero-sign harness into this new namespace. Match the final committed source, normal release profile, Cargo lock, archive, module/header/config/native-link metadata. Verify standalone DP/QP/stable calls and twenty NLO pole checks plus soft/collinear checks. |
| Final core and supplemental datasets | Full eight-process acceptance, loop-induced ggH/ggHH, threshold/precision/stability and two additional 2-to-4 stress campaigns remain independent numerical work. | Use final CLI generation/runtime bindings. Preserve ordinary input streams, tolerances, poles, stability/rescue and all failed/unresolved gates. Reuse original references only after their frozen role/input signatures match. |
| Final `benchmark.typ` / `benchmark.pdf` | Historical top-level files are not final certification. | Feed independently validated final datasets, source/runtime proofs and regression adapter into the renderer. Reproduce PDF bytes and visually inspect it. Report actual usage/rescues, observed discrepancies separately from estimates, and every mandatory pending/failed gate. |

MG7 suite selectors already used by these proofs are:

- `tests.unit_tests.various.test_q_polynomial`, `test_q_polynomial_compiled`,
  `test_q_polynomial_factorized`; `tests.unit_tests.iolibs.test_helas_call_writers`;
  `tests.unit_tests.loop.test_loop_exporters`; and the retained supported-FKS
  harness with `nb_core=2` and its existing explicit GoSam bypass.
- `tests.unit_tests.various.test_ninjars_install`, `test_ninjars_statistics`,
  `test_ninjars_integration_boundaries`, `test_ninjars_upstream`, plus the
  seven transaction tests selected by their retained collector.
- `tests.unit_tests.loop.test_ninjars_report_v2` and the supplement/precision/stress
  reader modules selected by their owners' final combined proof.

The 90-case, 37-case and report/reader counts overlap other focused historical
runs; they must not be added into an invented unique test count. Software tests,
physics correctness, installed-library controls and timing gates stay distinct.
Only the predeclared core and ggH/ggHH runtime gates apply; the new 2-to-4 stress
costs are diagnostic. A timing or correctness failure remains outstanding work.

Root retains cleanup authority. The three completed regression comparison scratch
subtrees remain ignored and untouched; no process scan or deletion was performed
for this handoff. The published audit at `publication-audit-v1` records the exact
intentional metadata assets and portable executable exclusions. Stay available to
compare the 102 tested production paths after the final commit, without rerunning
unchanged suites merely because a later commit exists.
