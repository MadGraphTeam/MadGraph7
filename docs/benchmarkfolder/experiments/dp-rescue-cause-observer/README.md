# Private DP-rescue cause observer — source proposal

This package prepares a diagnostic against the registered
`split-dp-terminal-1` implementation and installed TIRust DSO
`eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9`.
**No observer compiler, native worker, numerical test or timing has run.**
Production, reference sources, the installed libraries, registered outputs,
cards, and campaign protocol remain unchanged. The files here are private
source proposals; successful preparation does not qualify their numerical
neutrality or authorize execution.

The retained ordinary QCD n3 traversal has 87 DP / 13 QP accepted points; EW
n3 has 82 DP / 18 QP. These are final warmup calls 1–100 following bootstrap
call 0, rather than a post-warmup timed traversal. They do not reveal the
reason for rescue. This observer preserves the full stream and labels the
previously rescued points; it never selects a smaller warmup subset.

## Recorded boundaries

`prepare_host.py` derives three numerical-file patches and a common driver
patch for each exact generated process. All additions have numbered markers;
removing them reproduces the original eight files byte for byte. Original
source hashes must match the retained compilation records before any output
is written. `prepared/preparation.json` binds inputs, derived files and diffs.

* `TIR_interface.f`: DP group, squared order, CTMODE, N/rank, adapter result,
  and post-contraction stability/FPE state. A failed tensor never causes the
  observer to read its output buffer.
* `TIRUST_interface.f`: DP and QP diagnostic entry, guard/stage milestones,
  public status after the existing wrapper call, and successful completion.
  A final stage without a subsequent stage identifies a particular early
  return range; it is not automatically a native failure.
* `loop_matrix.f`: raw DP accuracy immediately after the existing calculation,
  its already-filled direction/rotation result array, effective accuracy
  after the FPE override, the unchanged acceptance threshold, and entry to
  QP before resetting FPE flags. QP accuracy is recorded separately.
* `check_sa.f`: identical bootstrap/warmup calls and phase labels, raw input
  momentum/scale, result/accuracy bits, all five IEEE flags and halting modes,
  and rounding mode immediately before and after every public amplitude.

`host_observer.f90` transports only integers and `TRANSFER` bit words. It does
not clear or restore IEEE state, evaluate a stability predicate, call a cache
API, or change a threshold. A detected observer flag/mode change is an explicit
health failure. Journal and parity-file output occurs after complete public
calls. Every arm writes the same public parity format. Entry-state parity on
the next call is required too, so output-side effects cannot be ignored.

The private native journal in `native/` surrounds the original ABI guard and
copies `LAST_ERROR` immediately on return, without a second guarded API. It
captures the existing embedded workspace diagnostics, route and execution
statistics before failure cleanup can discard the workspace. A Rust-level
`Err` does not reuse the previous invocation's `last_*` data. The copied
diagnostic history matters: a failed method-choice retry can leave the
original direct-engine failure as the public error.

Native hooks never enable profiling, evaluate a second integral, change
options, change cache attachment, format floating values, or read failed
output buffers. The helper reads host IEEE flags/rounding/traps and MPFR flags
around observation operations; any change stays visible and fails health.
The future link proof must establish that it reads the numerical DSO's actual
MPFR instance. A separately interposed MPFR would not qualify this check.

## Interpretation limits

Capture default options only: method choice enabled, partial rescue and
precision tiers disabled, unchanged gates and master policy. The observer
records accepted native routes and existing tier-attempt counts, plus public
QP input precision 113 and configured working precision 256. QP cache hits
are distinguished from new evaluations. It does not infer pure DP from the
`Direct` route: that route can include partial DD work under other options.
There is no complete public counter for those partial injections. Enabling
partial/tier options fails this proposal's policy gate; a future measurement
would need a new source freeze and appropriate actual-work instrumentation.

The observed host event tags are:

| Tags | Contents |
|---|---|
| 10, 11 | Public sequence/point/phase/loop; raw input words |
| 12–14 | Public status/shape; answer words; accuracy words |
| 20, 22, 21 | DP group entry; diagnostic-return stable/FPE; final group stable/FPE |
| 30 | Diagnostic precision, CTMODE, N and rank |
| 31 | Milestone: 2 rank passed, 3 support passed, 6 closure guard, 7 closure passed, 8 native call, 9 native success before raw scan |
| 34 | Initialization stage 4 or budget stage 5, with actual status |
| 32, 33 | Wrapper status; successful diagnostic completion |
| 40–42 | Backend/evaluation count/squared-order count/FPE; raw ACC; DP direction results |
| 43, 44, 49 | Backend/FPE; effective ACC after override; threshold bit word |
| 45 | Advanced library index, previous rejected backend, next slot/sentinel before QP selection |
| 46, 47 | QP backend/evaluation count/squared-order count/FPE; raw QP ACC |

Tag 45 describes the already-advanced library index; for the sole-backend-9
chain the next slot is zero and the previous rejected backend is 9. Do not
mistake the zero sentinel for the failing backend.

Keep native refusal, adapter guard/transport rejection, contraction failure,
and host consistency rejection as separate, potentially simultaneous causes.
An all-native-success history is insufficient on its own to declare a host
consistency failure: adapter stability and reduction FPE must agree with that
interpretation. Conversely, a final 323 status does not identify a specific
failing tensor or prove that a proposed DD option would help.

## Later bounded qualification

After the main campaign, obtain a separate root-approved quiet/availability
window. Follow `protocol-design.md`: original library and numerical sources
with only the common public driver, the private observer executable disabled,
and that identical private executable enabled, for each of the two processes.
The original arm links `observer_noop.c`; it must never be linked in the two
private arms. No process uses a policy change. Complete source/recipe/object,
card, dependency and runtime bindings precede execution and are checked after.

Use six sequential processes, the shared numerical lock, owned process-group
cleanup and a preregistered deadline. Preserve the existing complete warmup
order and filter-reset behavior in fresh private copies. Stop after the
unchanged 100-point warmup/READY protocol; no `RUN` command or timing claim is
needed. Retain every refusal, failure, timeout and partial trace; no automatic
retry. Require 101 complete public records, clean journal health/caps, native
and public close trailers, BYE, exit zero, expected point order, unchanged
historical statuses, and exact all-arm input/result/accuracy/IEEE-state parity.
The finite result, both poles, Born columns and all squared orders participate.

Only after those gates pass may an offline classifier attribute the rescued
points. A mismatching or incomplete observer is unqualified; its apparent
failure frequencies must not motivate a numerical-policy change. A later DD
or precision-tier experiment would be separate, use one justified candidate
policy, and retain independent MadLoop checks and genuine QP fallback.

`protocol.py` provides read-only input verification and optional source-only
snapshot creation. `analyze.py` is an unexecuted retained-text parser proposal.
Neither launches compilers or workers. A future execution controller and its
exact commands still require review before the qualification above.
