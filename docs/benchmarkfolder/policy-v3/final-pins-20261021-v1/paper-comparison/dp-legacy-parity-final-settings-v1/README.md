# Legacy forced-DP recorder control at final settings

This capsule retains an actual private qcd1 (`u u~ > z [virt=QCD]`) recorder
nonperturbation control for CutTools1, Ninja6 and COLLIER7. It is partial control
evidence: NinjaRS8 was held, the full four-tool/process campaign is incomplete,
and no timing or performance acceptance is claimed. The control uses the release
explicitly bound by `evidence/started.json`, not whichever upstream branch is
latest when the reader runs.

Each backend evaluated the same100 fixed points in forward, reverse and z90
forward modes, full helicity/color sum and all Laurent/split orders, in both
plain and observed copies. All2,400 physical coefficients and600 internal
accuracy records per backend compare exactly as float64 bits. The sample had
no physical zero coefficients; zero, failed-zero and nonfinite controls are
covered separately by the compiled recorder and software fixtures. Each
instrumented copy recorded300 requested-backend reductions and no rejected,
nonfinite, QP, compensated, or reduction-FPE observations. Both copies retain
H1/ACC−1 as expected for these explicit modes; that status is not a stability
acceptance result.

The cards use MLStabThres1e−9 and explicit COLLIERRequiredAccuracy1e−12, full
poles, forced DP/no controller rescue and declared COLLIER cache0/internal
acceptance off. All other numerical settings match the qualified CLI origin.
The earlier default0.001 control remains narrower evidence and the reader
rejects it for publication (retained under earlier-default-control).

All six observations ran with one evaluation thread on CPU12. A diagnostic
link guard proves no NinjaRS context creation or numerical evaluation was
entered; the existing teardown forwards exactly one destroy_context(NULL).
The controller and all19 owned child groups exited successfully and were
reaped. Compiler executables, object files, generated output trees and dependency
archives are deliberately excluded; their exact hashes/source manifests,
commands, native/module/link/runtime tuple and raw outputs are retained.

Reproduction: use the repository toolchain environment and the explicit cohort
in evidence/started.json. Preserve the source freezes in review-request.json;
copy the retained run.py and guard.c to a new private work directory inside this
repository before executing the CPU12 legacy-only control. The runner refuses
to reuse an already started directory. The normalized manifest is produced by
sources/normalize-parity-v1.py, then re-read by tools/ninjars_dp_stability.py.
Semantic qualification checks live original/private generated sources, complete
runtime, cards, points, executable/selector and both raw streams; it does not
trust retained passed flags. Existing v3 cohort/origin/runtime readers are reused.
The independently revalidated legacy entries qualify, while overall complete
and physical_qualified remain false because backend8 is missing.

Software controls run via `python3 -m unittest
 tests.unit_tests.loop.test_ninjars_dp_stability -v` (one command line). Thirty
checks passed without a compiler or numerical engine. Their mock filesystem
fixtures are not physical observations. See the earlier immutable
../dp-software-controls-v3 capsule for the exact compiled recorder helper,
including its preserved failed hardware-flag control and corrected caller-level
flag capture. Source observer and Fortran driver mechanics are unchanged.

Current reader limitation: truncated/abnormally exited streams retain missing
slots diagnostically but require a concrete matched-prefix failure control
before publication qualification. Such an attempted tool must remain visible
as a failed/pending cell, distinct from an unattempted tool. This control had no
such exits. `physical_qualified` refers to observation provenance, not a claim
that a result passes the independent numerical correctness/performance gates.
