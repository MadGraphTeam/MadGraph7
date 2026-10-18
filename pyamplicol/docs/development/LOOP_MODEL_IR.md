# Exact loop-model preparation

`pyamplicol.loops.model.build_loop_model(raw_sidecar)` prepares exact scalar
definitions and individual tree/UV/R2 colour × Lorentz × loop-group entries.
It consumes the existing raw UFO sidecar, including its exact-source contract;
it does not import a second source manifest. Scalar definitions are indexed by
`(collection, raw_object_id)`. Vertex display names need not be unique.

`parse_exact_ufo` uses standard Python AST/tokenization to validate arithmetic
syntax and retain numeric lexemes. `Fraction` converts decimals into exact
rationals before the existing UFO parser runs. The parser and Symbolica own
complex arithmetic, model-function calls and expression transformations. No
Python expression evaluation or binary64 conversion occurs on this path.
Unknown model functions retain their raw definitions and remain symbolic;
arbitrary Python helper bodies are not interpreted.

`ScalarDefinition.value` is a `LaurentExpression`. CT dependencies, including
ordinary parameter or coupling aliases containing CT/Laurent definitions, are
substituted before requesting a coefficient. The full expression remains
available. `series(maximum_power)` returns Symbolica's native series and its
remainder order; `coefficient(power)` extracts a native exact coefficient.
For example, an O(epsilon) term multiplying a pole is retained in the finite
part. Ordinary epsilon-independent parameter definitions remain separate,
so input values and formal colour constants are not prematurely substituted.

`CovariantKernelCatalogue(ir, structural_model).lower(entry)` produces separate
native `TensorExpression` objects for colour and Lorentz structure. The
structural input is a native HepKit model or its public normalized JSON mapping.
Only species and propagator metadata are reused. Cached numerical scalar values
from the normalized model are never used for the entry's coefficient. Each
component is localized by a public HepKit single-vertex diagram with unit scalar
coupling and the other tensor domain set to one. Component results are cached.
This reuses native UFO index, momentum, gamma and colour lowering directly.

The local momenta are all incoming native `q_edge` symbols. `leg_half_edges`
identifies the UFO leg order. Each tensor retains its native axis order, which
can differ: `Metric(1,4)*Metric(2,3)` has axes `(0,3,1,2)`. Consumers must use the
native slots/index labels and native axis permutation/rebinding operations.
The coefficient stays separate from both tensors.

The initial `loop_sm` controls cover every pure-QCD component, all 57 active
four-gluon R2 basis entries, and two-point UV mass insertions. A synthetic
one-point UV rule checks native scalar lowering without extending process
support. A mixed QCD/QED quark-photon R2 control checks the exact model formula.
Run these controls with `PYAMPLICOL_NLO_UFO_REFERENCE` pointing to `loop_sm` and
a licensed Symbolica runtime. No reference files are changed.

This preparation does not certify a scheme, UV/R2 completeness, arbitrary
precision physics, or formal-Nc continuation. Loop-group alternatives and
particle counterterm metadata are retained; process-dependent matching,
external-wavefunction insertions, restrictions and scalar-kernel compilation
remain separate work. The active four-gluon `loop_sm` R2 coefficients contain
fixed SU(3) constants. Preserving them does not reconstruct their Nc dependence.
`FormalColorIdentities` states explicit SU(Nc) CA/CF/TR/TF/Ncol substitutions
with fixed Nf, but never applies them automatically or certifies LC/NLC.
Legacy inconsistent non-QCD Lorentz spin signatures fail when requested.
