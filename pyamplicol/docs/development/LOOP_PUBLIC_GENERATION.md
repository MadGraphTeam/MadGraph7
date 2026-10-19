<!-- SPDX-License-Identifier: 0BSD -->

# Public model-rule loop generation

`Generator.plan` and `Generator.generate` accept virtual requests through
`process.loop_coupling_orders`. The same entry points used for tree generation now
publish source-owned loop PACBIN instructions in the ordinary transactional
artifact. `Runtime.load` reads that artifact through its checked native loader;
Python does not reconstruct the graphs or run a second numerical implementation.

The source-model lane uses model-supplied R2 plus UV, inclusive or resolved
helicities, and JIT coefficient kernels. Double, DoubleFloat and arbitrary
precision use the saved exact source expressions. Full colour needs no formal
continuation. LC/NLC requests require explicit source colour data in the model
bundle, as described in `LOOP_SOURCE_COLOR_DATA.md`; missing coverage fails.
Full-colour QCD colour histories use the grouped complex virtual path described
below. Rectangular FFT, spin-source admission and flavour aliases remain separate
integration work. The compiler has no
process-name dispatch or artificial multiplicity limit; actual tensor, scalar,
rank and reducer domain checks can still reject a request.

Generation uses the existing loop coefficient JIT, projector and reduction
settings. Nondefault tree-only JIT optimization/compression, Horner/common-pair
settings, recurrence tile/workspace settings and output chunking are rejected
with their setting paths until those controls are connected. Shared resource
cores, validation, model inputs and backend selection retain their normal checks.
Changing these options is not a way to set a limit on process size.

## A usable card

Save this as `virtual.toml`, pointing `model.source` at a prepared compatible NLO
UFO directory or its compiled model JSON. The scheme strings below describe the
supplied QCD `loop_sm` rule convention used in the current controls. They are
provider/user declarations, not a certificate inferred from the model name or
from finding UV/R2 vertices. Other models must supply their actual conventions.

```toml
schema_version = 1
action = "generate"

[model]
source = "models/prepared-loop-sm.pyAmplicol-model.json"

[process]
entries = [{ expression = "u u~ > e- e+", name = "dy" }]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"

[color]
accuracy = "full"
contraction = "direct"

[loop]
r2_strategy = "model"

[loop.model]
loop_coupling_orders = ["QCD"]
uv_complete = true
model_r2_complete = true

[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_sm lhv=1 supplied QCD rules"

[generation]
output = "artifacts/dy-virtual"
emit_api_bundle = true

[generation.validation]
post_build_validation = true

[evaluator]
execution_mode = "auto"
backend = "jit"
```

```sh
pyamplicol generate --card virtual.toml --dry-run --json
pyamplicol generate --card virtual.toml
pyamplicol evaluate artifacts/dy-virtual --process dy \
  --momenta points.json --mu-r2 1000000 --arithmetic double --json
```

`points.json` is a batch of physical incoming/outgoing four-momenta with shape
`[point][leg][E, px, py, pz]`. `mu_r2` is the positive renormalization scale squared,
not the numerator's extra-dimensional variable. It is currently required for
public virtual evaluation. Its value is a caller choice; the example does not
imply automatic coupling running.

The identical card drives Python:

```python
from pyamplicol.api import Generator, Runtime
from pyamplicol.config import resolve_config

configuration = resolve_config("virtual.toml")
generator = Generator(configuration)
plan = generator.plan("u u~ > e- e+")
if plan.unsupported_features:
    raise RuntimeError(plan.unsupported_features)
artifact = generator.generate("u u~ > e- e+", "artifacts/dy-virtual")
runtime = Runtime.load(artifact.output, arithmetic="arbitrary", precision=70)
values = runtime.evaluate(points, mu_r2="1000000")
print(values.finite, values.single_pole, values.double_pole)
print(values.sectors)  # squared coupling orders and contributing Born/loop pairs
resolved = runtime.evaluate_resolved(points, mu_r2="1000000")
print(resolved.total().finite)  # sum helicities; preserve orders and poles
```

Full-colour resolved output carries the physical helicity axis and one contracted
component named `full`; NLC uses `nlc`. LC uses the saved canonical Born-tensor
partition and can have several signed, potentially overcomplete components. Every helicity keeps the incoming spin average from the
inclusive result. Components are signed Born–loop interferences; they are not
probabilities. Summing helicities and colour preserves each squared order and all
three Laurent coefficients.

Use decimal strings or decimal values for exact high-precision inputs. Runtime
load chooses the arithmetic before binding parameter overrides. Omitting explicit
arithmetic retains the runtime's configured defaults.

With `generation.emit_api_bundle=true`, the same transaction publishes examples
for Python, C, C++, Fortran, Rust and the explicitly versioned pyAmpliCol virtual
UMAMI extension under `API/`. They use the ordinary installed SDK and one native
bulk evaluation. Run a native example with `make run ARGS='dy 1000000 resolved'`
from its language directory; Python accepts `--resolved` instead. The scale is
always supplied at evaluation. Every example retains coupling-order catalogues,
all three Laurent coefficients and lane diagnostics. Resolved signed components
are not sampling weights. See the emitted `API/README.md` for exact-input and
explicit verification options. Tree UMAMI provider headers are not generated for
virtual artifacts.

## Source preparation and declarations

The released `ufo-model-loader` preserves complete ordinary and counterterm source
records, exact scalar expressions and Laurent coefficients before constructing
its normalized tree view. pyAmpliCol retains this source snapshot in the existing
compiled-model bundle. An older tree-only bundle without that snapshot cannot
supply loop generation.

A prepared model can carry the same declaration under its existing raw-source
metadata (`pyamplicol_loop`), using `with_loop_model_declaration`. An explicit
`[loop.model]` block replaces that whole declaration. Fields are never silently
merged into a fabricated complete declaration. A joint QCD/QED declaration covers
mixed loops; two independent single-correction claims are not combined. Likewise,
a request for `["QCD", "QED"]` is sent once to the joint graph-selection machinery,
rather than evaluated as the union of the two single requests.

The supplied MadLoop `loop_sm` has a known source reference mismatch: the four-G0
vertex `V_1001` points at the three-slot constant tensor `SSS1`. A broad prepared
scope therefore requires the existing explicit
`loops.model_compatibility.prepare_loop_sm_scalar_declarations` view before
serialization. It verifies that exact mismatch, replaces only that tensor
reference with `SSSS1`, and records the derivation while preserving the source
snapshot. The public service does not repair a model by recognizing its name.
Unprepared mismatches identify the vertex, Lorentz tensor, actual spin slots and
required particle slots. The current Drell–Yan and four-gluon public controls use
this explicitly prepared view.

Public `max_coupling_orders` are inclusive ceilings. They become `(0, maximum)`
Born bounds; only a selected correction's bounded coordinate is raised by two
for loop construction. An omitted coordinate remains unbounded. The existing
minimal-order policy supplies its usual Born envelope first. Graph eligibility
and exact interference pairing remain authoritative afterward.

## Dry-run, validation and publication

Dry-run prepares source/local kernels and graph orders and reports Born, loop and
family counts, squared-order catalogues, conventions and pending public features.
It does not compile numerical numerator programs. Its coverage explicitly marks
numerator and reducer checks as deferred. Formal-colour process coverage and LC
canonicalization are also deferred, so a successful dry-run is not a claim that
every later tensor or integral domain is supported.

With `generation.validation.post_build_validation=true`, generation evaluates the
source program on the configured sample points, then loads and evaluates the
staged public artifact before the transaction publishes it. The check compares
Born orders and every finite, single-pole and double-pole value. A malformed or
numerically inconsistent artifact aborts publication and preserves an existing
target. These are transport/execution checks, not an independent physics oracle;
provider, scheme, precision and physical identity comparisons remain separate.

All model, graph, tensor and reducer failures retain their explicit diagnostics.
Unsupported virtual correlation compositions fail at the public dispatch
boundary instead of being routed into the tree correlated compiler. Production `auto` R2 currently
selects the only complete published branch, model rules; forcing direct R2 fails.


## Directed colour correlations

Pass the existing `CorrelatorConfig` to `Generator.generate`. For example, with a
full-colour QCD card and a compatible declared loop model:

```python
from pyamplicol import CorrelatedRequest, CorrelatorConfig
from pyamplicol.color.correlator_matrices import ColorCorrelator

declarations = CorrelatorConfig(color_correlations=(
    ColorCorrelator.dipole("T12", 1, 2),
))
artifact = generator.generate(
    "u u~ > d d~", "artifacts/uudd-correlated", correlators=declarations,
)
runtime = Runtime.load(artifact.output, arithmetic="arbitrary", precision=70)
values = runtime.evaluate_correlated_many(
    points, {"dipole": CorrelatedRequest("T12"), "identity": CorrelatedRequest("born")},
    mu_r2="1000000",
)
print(values["dipole"].finite, values["dipole"].single_pole,
      values["dipole"].double_pole)
```

Physical legs use the existing one-based public labels. Negative auxiliary labels
are preserved. The implicit `born` history is the identity operator; ordered
multi-emission histories and `CorrelatorConfig.all_color` use the same existing
catalogue, without a fixed order cap. Results retain complex values, every
squared coupling order and every Laurent coefficient. The two integrated virtual
cross terms are kept distinct until their final sum. Grouped evaluation runs all
saved histories in one native batch and selects named outputs afterward.

The same declaration serializes with `declarations.to_json_dict()` and is accepted
by `pyamplicol generate --card virtual.toml --correlators correlators.json`.
`inspect --process uudd --full-physics --json` includes the authenticated virtual
scheme, order catalogue and public colour histories under `virtual`. The ordinary
CLI `evaluate` continues to evaluate the ordinary interference; neither the tree
nor virtual CLI currently exposes the Python grouped-correlated evaluation
surface. The existing shared CLI restriction also disallows combining
`--correlators` with `--dry-run`.

The initial public correlated domain is full colour, QCD corrections, model R2
plus UV and inclusive physical helicities. Spin sources, resolved correlated
output and LC/NLC correlations fail explicitly. Ordinary evaluation remains
available on the same artifact. Transactional validation checks both ordinary
and every prepared correlated output before publication when enabled; these are
source-versus-saved execution checks, not independent physics certification.
The fresh quark-scattering dipole control additionally compares the Born and all
three poles/finite coefficients with Recola under matched conventions. It does
not use Recola to certify non-Hermitian multi-emission imaginary contributions.
