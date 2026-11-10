# Private AVH absolute-classifier refusal prototype

Source preparation only. No compiler, linker, loader, mock executable or actual
scalar kernel has run. Production AVH/CutTools sources and all reference inputs
remain unchanged. `generated/avh_olo_checked.f90` is an isolated additive copy.

The declared scope is exactly the no-`rmu`, real-invariant overloads `a0_c`,
`a0_r`, `b11rc`, `b11rr`, `c0rc`, `c0rr`, `d0rc`, `d0rr` in the pinned AVH DP
and genuine binary128 QP modules: 16 overloads. These cover actual CutTools
`scaloop=2` calls plus their real-mass counterparts. Complex-invariant overloads,
explicit-scale overloads, B0/dB0/BN/AN and MPFUN precision are unsupported. No
claim covers deeper analytic kernel branches, on-shell provenance or a general
numerical error bound.

`generate.py` extracts each actual wrapper's classification prefix before its
first analytic dispatch. Native real/complex norms, mass sign normalization,
the triangle mass-case permutation and the box soft/mass permutations are
preserved. The original public routines remain byte-identical. There are 64
static erasure assignments: A0 has one event per call, B11 three, C0 at most
nine and D0 at most thirteen. Box invariants 5/6 participate in soft permutation
but do not have a direct erasure assignment in this source. They are not
misreported as such. Difference sites retain their original source indices
through both permutations.

Additive precision-module entrypoints are `cms_preflight_<name>(report,inputs)`
and `cms_checked_<name>(original_outputs,original_inputs,report)`. Callers import
named DP/QP procedures explicitly. The common type `cms_scalar_report` lives
in `cms_scalar_observer`; direction and outer CutTools subtopology are caller
context fields to set after return. Each call resets its report. No observer
global state, allocation, threshold setter, I/O or mutation of original input
arguments is introduced.

The preflight reads the exact existing AVH absolute threshold and scale; it
does not install a new tolerance. Uninitialized state and default/non-absolute
mode return explicit unsupported statuses. Exact all-zero C0/D0 returns are
also unsupported, while exact-zero A0/B11 inputs may be delegated. Nonfinite
inputs and derived metrics/scales refuse. A detected erasure records the actual
native metric, its norm, topology/overload/precision, source line and indices,
original complex value, locally classified complex value, cutoff and local
scale. The record uses binary128 storage, with exact conversion of finite DP
values. Original and classified values remain distinct when the source flips a
positive mass imaginary part. Exact zero is ignored only when both components
of the actual classified value are zero; a nonzero value with an underflowed
zero metric remains a refusal event.

Classification continues on local copies after recording so all reached
erasure assignments on that exact source path are observed. Any event refuses
analytic evaluation. The fixed 32-event array exceeds the proved 13-event
scope; if it nevertheless fills, dropped events increment and status remains
`CMS_EVENT_OVERFLOW`, never success. `count` is retained records, `dropped`
means the event report is incomplete, and neither state permits publication.

Checked entrypoints preserve caller output buffers on preflight refusal. An
accepted call evaluates the unchanged original wrapper into local scratch,
checks every real and imaginary Laurent coefficient for finiteness, and only
then publishes buffers. `CMS_NONFINITE_RESULT=8` retains caller buffers and
reports `native_called=true`, `result_valid=false`. `result_valid` means a
completed finite scalar result after this limited classifier gate; it is not
physical accuracy, successful handling of every analytic singularity, or a
substitute for CutTools/MadLoop independent stability and rescue checks.

The preflight repeats native metric arithmetic, and event construction can
repeat complex subtraction. Floating-point flags, enabled traps, rounding mode
interactions and the runtime cost are **not qualified** by this source work.
No transparency or performance claim is made. An absolute classifier may
deliberately change refusals under common rescaling; this is not a scale-aware
repair or proof that a small transfer is zero.

The pure Fortran mock fixture uses these exact generated preflight/checked
bodies and the exact source permutation constants. Only analytic callbacks
are recording stubs, allowing assertions that refused calls never invoke them,
that accepted calls pass original arguments and that buffers/input storage bits
remain unchanged on refusal. Sources cover every overload, all event sites in
the maximally small case, strict threshold neighbors, complex modulus versus L1
norms, imaginary masses, C-versus-D difference norms, subnormals, zero/default/
uninitialized modes, finite-input metric overflow, event overflow and nonfinite
callback outputs. Exact retained point20 transfer/residual literals appear as
QP fixtures, with the historical binary64 cutoff promoted to QP; their historical
labels are input features, not asserted new preflight outcomes. These Fortran
assertions have not run.

`compile-plan.json` proposes five compiles and one mock link under a single
120-second cap, CPU 8 and one job. It also compiles the full isolated actual AVH
source using a private hash-bound copy of the existing CutTools module/include
closure. Actual, mock and shared observer module directories are separate; the
actual compilation searches its own output, the observer and private dependencies,
and never searches mock module outputs. The real AVH object is not linked into
the mock and no actual analytic
run is proposed. A separately authorized mock run would have a 15-second cap.
Production integration and actual analytic linking/numerical validation remain
separate work. Early source-only drafts are retained without implying that they
were compiled or executed.
