# Proposed classification for current-source raw stability workers

This is a **source-only proposal**, without new amplitude evaluations, new
plots, historical reclassification, or canonical promotion. The published
source5 raw pilot, its failures and unknown bounds remain unchanged.

The historical worker records software flags immediately before and after
`SLOOPMATRIX_THRES`. Its classifier rejects a set incoming flag even if both
outgoing flags are clear. The qualified current matrix template
`3223f38d2e61243f2b2849d08982d17eadf2a93ae1dc2edaccaa44515467494b`
clears both software reduction latches unconditionally on the public-call
path, before all reduction attempts. For the exact sealed source5 driver,
positive modes1/2 exclude automatic backend/precision transition resets, and
`FORCE_STABILITY_CHECK(.FALSE.)` prevents a second public call in the wrapper.

The sole proposed classification change is therefore: for an authenticated
worker from this exact source, an incoming DP/QP latch may be ignored when
**both outgoing latches are clear** and the full group journal has no parse
issues. The wrapper removes only the exact legacy sticky-latch reason, before
the unchanged `parse_diagnostics` function derives row and point eligibility.
It retains the incoming/outgoing bits, original delta fields, legacy reason
and eligibility, failed/invalid groups, nonfinite counts, coverage, statuses,
warnings, bootstrap checks, split-order consistency and completeness gates.
Any outgoing set flag or active-group failure remains ineligible. Clearing
flags does not make a failed tensor contribution available.

New annotations distinguish a source-qualified incoming flag cleared at the
public entry from an outgoing software failure in the current call. They do
not claim a directly observed post-entry snapshot. The original
`fpe_*_became_set` fields remain raw before/after deltas; new
`fpe_*_set_during_current_call` fields reflect the qualified reset lifetime.
No hardware IEEE exception state is observed or reset. Unchecked110 remains
an unchecked host result; no internal reducer precision, actual backend,
independent numerical truth, or timing acceptance is inferred.

`qualify_sources` requires byte-supplied receipts and sources. It pins the
`entry-latch-cache-guard-1` manifest (`4a27dda7…`) and current template, joins
the complete generated exporter inventory and campaign binding, requires a
full numerical source build, and verifies the actual generated
`loop_matrix.f` against its compilation receipt. It also checks the exact
rendered source5 worker (`371f8ccb…`) against the private prepared-case
driver/matrix hashes, successful private build command and executable digest.
Compilation, private donor, subprocess, dimensions and prefix must agree.
The actual generated matrix must contain the unconditional qualified Born
reference/reset/initialization block. NinjaRS exports additionally require the
exact qualified public bookkeeping wrapper and its single implementation
call. A stale matrix with recomputed receipt hashes is rejected. The static
check is exercised on the existing closed lifecycle candidate and rejects its
original source fixture. The live template alone cannot qualify a historical
worker.

These supplied records must first be authenticated by the future controller's
existing source/tree/build/runtime admission and joined to the raw worker's
executable. This pure classifier does not establish execution from receipts
alone. No new current-source raw worker has been admitted or run by this
proposal. In particular, the mocked receipts used by unit tests are not an
actual numerical qualification. Retain the supplied source/receipt bytes and
recompute qualification during portable verification; an unverified stored
flag is insufficient.

The pure API returns the executable/source binding for that future join:

```python
qualification = qualify_sources(
    manifest=manifest_bytes, template=matrix_template_bytes,
    generation=generation_receipt_bytes, compilation=compilation_receipt_bytes,
    matrix=actual_generated_matrix_bytes, worker_template=source5_worker_bytes,
    driver=actual_private_driver_bytes, private_build=prepared_case_receipt_bytes,
    matrix_source_key="SubProcesses/P0_uux_zgg/loop_matrix.f",
    worker_parameters={"prefix": "ML5_0_", "nexternal": 5, "nborn": 1},
)
parser = parsers(frozen_source5_pilot_bytes, qualification=qualification)
result = parser["parse_diagnostics"](dimensions, values, stdout, groups, layout)
```

Omitting `qualification` executes the exact historical source5 parser without
new annotations or changed decisions. Only pinned pure function AST bodies
are loaded; no controller import, external path lookup or native call occurs.

Source anchors:

- Current matrix template849–850 resets both latches before CT-mode selection
  at928, cache/reset and initial reduction branch967–991. Automatic-policy
  resets1843/1910/1923 are gated by automatic mode at1715.
- Source5 worker91 disables forced stability checks;134–135 selects modes1/2;
  174 assigns the selected mode;183–190 surrounds the public call with flag
  and group observations;229–230 checks initialization mode, selected backend
  and zero sentinel.
- Historical source5 `parse_groups`1202–1305 retains group/bit observations;
  1283 applies the incoming-or-outgoing latch rule. `parse_diagnostics`1308+
  derives all later eligibility and fixed100-point outcomes.

Run the pure proposal tests from the required checkout:

```sh
source tools/tirust_env.sh
python3 -B -m unittest discover -s docs/benchmarkfolder/stability_appendix_c/current-source-classifier-proposal-1 -p 'test_*.py'
```

Before a future raw pilot: root must review this exact source seal, retain the
new generated/compiled/private-driver binding, repeat the existing full
prelaunch checks and source/runtime admission, and separately admit numerical
execution. No historical curve may be relabeled using this proposal.
