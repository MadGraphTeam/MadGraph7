# Physical stress diagnostics

This package evaluates four selected real-mass stress points from the frozen
benchmark inputs. It is separate from the ordinary 100-point campaign, the
complex-mass study and all performance acceptance measurements. Each two-point
stream contains an ordinary control/initialization anchor followed by one stress
point. No runtime comparison is qualified here.

| Process | Stress selection | Parent index |
| --- | --- | ---: |
| `g g > t t~ [virt=QCD]` | Pair threshold | 1 |
| `u u~ > z g g [virt=QCD]` | Soft gluon collinear to the incoming beam | 2 |
| `u u~ > z z g g [virt=QCD]` | Nearly planar, small Gram determinant | 3 |
| `e+ e- > t t~ a [virt=QED]` | Threshold with a very soft photon | 1 |

Primary policy is `MLStabThres=1e-6`, identical for baseline and candidate.
The baseline uses explicit CutTools; the candidate uses each of `1,6,7,8,9`
independently. The retained comparator uses the frozen campaign's `3e-6`
coefficient tolerance and records raw relative errors as well. Passing a host
status alone does not establish agreement between reducers. Status rejections,
nonfinite outputs and timeouts remain distinct outcomes.
The comparator uses `abs(a-b)/abs(a+b)`, allowing poles whose magnitude is below
the tolerance times the finite coefficient. Thus a comparator pass does not
necessarily mean small raw relative error for every pole; the full raw residuals
and the separate `1e-8` ordinary target result are retained.

The 24 primary requests produced 12 complete checked streams, 11 rejected or
nonfinite streams, and one initialization/warmup timeout. These four extreme
points do **not** qualify as a successful five-backend stress suite.

| Stress point | Primary observations |
| --- | --- |
| QCD pair threshold | TIRust and CutTools auto-QP agree to `2.20e-16` maximum raw relative error. Across all checked reducers, the largest finite residual is `2.15e-7`, passing the regression gate; Ninja/NinjaRS and COLLIER disagree on the single pole despite accepted statuses. |
| QCD soft-collinear | Every reducer has a NaN Born coefficient from the common generated tree calculation; loop outputs are also rejected. Baseline and candidate CutTools behave identically. See `born-path-audit.json`. |
| QCD small-Gram | CutTools, Ninja and NinjaRS agree within about `2.4e-8`; COLLIER rejects; TIRust does not reach READY within the initialization deadline. |
| EW threshold | Ninja and NinjaRS accept auto-QP results but disagree in the finite coefficient by `1.68e-5`; CutTools, COLLIER and TIRust reject under the strict host policy. |

The initial `CTModeRun=4, CTModeInit=4` follow-ups exposed a protocol mismatch:
the ordinary worker forces stability checks, which remaps fixed-QP return codes
so its warmup cannot reach READY. One timeout and two interrupted requests are
preserved in `fixed-qp-protocol-audit.json`; the six remaining requests were
cancelled. These attempts provide no fixed-QP coefficient or numerical-cost
evidence. A compatible forced-QP diagnostic must disable forced stability
checks, retain status `120` and accuracy `-1`, and explicitly mark coefficients
as unchecked. Such results never count as primary numerical qualification.
The corrected requests in `diagnostics/` use DP initialization and fixed-QP
evaluation with forced stability checks disabled. Each has a separate clone;
only its driver object is recompiled and relinked. Build records verify that
all other numerical objects, library inputs and effective compiler settings
remain unchanged. At the QCD pair threshold, corrected Ninja and NinjaRS QP
coefficients match CutTools auto-QP exactly, resolving their primary DP pole
discrepancy without changing the original checked results.
The nine corrected requests produced six complete unchecked streams, two
nonfinite streams, and one warmup timeout. A final separately recorded trace
replay of the QCD small-Gram request then completed: bootstrap, ordinary warmup
and stress warmup all returned `120`, as did the measured traversal. Its stress
coefficients agree with checked CutTools to `9.17e-10` finite and `3.17e-13` poles
in raw relative error. No stability check was performed for this replay.
See `trace-observation.json` for its source and result bindings.
Each worker phase has a 180-second deadline; a startup or traversal timeout
does not imply an incorrect numerical answer. The final trace recorded 173.3 s
of cold initialization and 122.3 s for its two-point traversal, each within
the separate deadline. These are diagnostic observations without a controlled
timing lane or performance comparison.
The primary TIRust small-Gram timeout also occurred during initialization/warmup;
its untraced record cannot distinguish a long native call from repeated host
rejections. The successful derived trace does not retrospectively establish
that original timeout's cause or replace the original primary outcome.

The EW threshold discrepancy remains unresolved: unchecked CutTools and Ninja
QP finite coefficients differ by `1.32e-5`. TIRust's unchecked QP finite result
agrees with the rejected primary CutTools result to `6.43e-7`, but large relative
differences remain in the tiny pole coefficients. QCD soft-collinear QP
follow-ups retain the nonfinite Born output. None of these diagnostics converts
a rejected primary point into a checked acceptance.

`points/` contains exact copied parent DAT/JSON files and derivative streams;
momentum rows are copied byte-for-byte rather than regenerated from JSON.
`inputs/` records physical cards, original compilation/generation metadata,
copied-executable hashes, source equality, symlinks and dynamic-library identities.
Generated processes were copied into ignored private scratch directories. Their
cards, filters, FIFOs and observations are independent; original immutable
library directories are shared read-only. Primary runs required no compilation.

`runs/` and `diagnostics/` retain each requested card, input binding, stdout, raw coefficients,
statuses, dimensions and result record. `summary.json` compares complete
coefficient columns only. `adjudication.json` additionally compares complete
finite columns from rejected or unchecked requests, marking those comparisons
as diagnostic only. Failed runs remain in the inventories. The initial
QCD threshold runs used the first retained controller revision;
`protocol-amendment.json` documents the later evidence-handling fixes and a
concurrent main-harness change restricted to timing/report code. Subsequent
controllers import the frozen protocol copies and retain their own exact bytes.

The 200-digit Decimal calculation in `kinematics/` uses authoritative printed
DAT tokens. All four points have conservation residuals below `2.3e-16` and
mass-shell residuals below `8.9e-17` in the recorded global normalizations.
Local sensitivity is nevertheless substantial: the QCD soft-collinear invariant
is about `5e-14 GeV^2`, and the EW top-pair excess is only `9.62e-13 GeV`.
Input mass-shell rounding changes the latter inferred excess by roughly 3%.
The small-Gram point has
`abs(det(2*p_i.p_j))/s^4 = 1.2008e-16` for its four final-state vectors.
These limitations accompany the numerical results; the inputs are not silently
projected onto different kinematics.

Reproduction uses `tools/tirust_physical_stress.py` from the MG root after
`source tools/tirust_env.sh`. Preserve this append-only evidence directory and
the ignored private output directory elsewhere before a new `prepare` run.
`prepare` requires the existing frozen compiled benchmark workers and checks
their source/library identities; `run` executes the primary matrix with optional
`--case`, `--source` and `--backend` selections. Corrected QP followups use
`tools/tirust_physical_stress_qp.py --lane non-symbolica` and `--lane symbolica`;
the ordinary runner deliberately rejects fixed-QP mode.
The final trace command is
`python3 docs/benchmarkfolder/experiments/physical-stress/protocol-sources/trace-runner.py --lane symbolica`.
It also requires a fresh append-only report/private-output destination.
Use `PYTHONDONTWRITEBYTECODE=1` when replaying retained
Python sources. The kinematic calculation runs directly with
`python3 docs/benchmarkfolder/experiments/physical-stress/kinematics/diagnostics.py`.

`execution-summary.json` separates the primary matrix, the incompatible initial
fixed-QP attempts, the corrected requests and the final trace. `execution-logs/`
retains their outer controller logs. `evidence-index.json` hashes every portable
file except itself; numerical binaries are identified by the original input
and compilation records and are not included in this text-only package.
`evidence-check-final.json` independently checks 375 core identities, all 180
original audit identities, ten driver-only compilation proofs, and 248
coefficient comparisons. Its index snapshot predates adding the audit itself;
the final inventory includes that record and its retained audit script.
