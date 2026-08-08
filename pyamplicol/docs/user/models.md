<!-- SPDX-License-Identifier: 0BSD -->

# Models And Processes

Commands using the packaged `models/` paths assume the installed environment
is active and the current directory is a workspace created by
`pyamplicol examples copy ./pyamplicol-examples`.

## Serialized JSON

Serialized JSON is the primary portable model input. It is deterministic to
copy and inspect and does not execute model code while loading:

```toml
[model]
source = "models/json/sm/sm.json"
restriction = "default"
```

Materialize the packaged model assets into a copied example workspace:

```console
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples
pyamplicol generate_pp_zjj_from_ufo_sm.toml
```

The distribution contains `sm`, `scalars`, and `scalar_gravity` in JSON and UFO
forms. The public model interface currently accepts filesystem paths rather
than those package names, so `examples copy` writes wheel resources to ordinary
`models/json/...` and `models/ufo/...` paths. The copied
`python/copy_packaged_models.py` helper can populate a separate empty model
workspace when only the models are wanted.

## Trusted UFO

A UFO directory is accepted through the same configuration:

```toml
[model]
source = "models/ufo/sm"
restriction = "default"
```

UFO modules are Python and execute while loading. Treat a UFO directory as
code and use it only from a trusted source. Prefer its serialized JSON form for
portable generation and automated deployment.

The typed interface resolves explicit sources without importing Symbolica:

```python
from pyamplicol import ModelSource

json_source = ModelSource.from_path("models/json/sm/sm.json")
ufo_source = ModelSource.from_path(
    "models/ufo/sm",
    restriction="restrict_default.dat",
)
compiled = json_source.compile()
```

`compiled` is an immutable, opaque `CompiledModel` handle. Generation accepts
it directly, while stable source, capability, parameter, diagnostic, and phase
metadata are available through `compiled.info`; compiler-owned tensor and
expression IR remains private. Use `compiled.write(path)` to retain a compiled
model or `compiled.write_parameter_card(path)` to create an editable JSON card
containing its external parameter defaults.

That parameter card has the same flat shape as the serialized JSON form of a
UFO restriction: parameter names map to finite real numbers or explicit
`[real, imaginary]` pairs. It can be passed as `--model-parameters` to the
runtime CLI and every generated standalone API driver. Direct
`--set-parameter NAME REAL IMAG` updates are applied after the card and take
precedence; unknown, immutable, or invalid values reject the entire update.

With `ModelSource.from_path`, a relative restriction filename is resolved from
the model directory and validated. In a TOML card, use the restriction name
(`default`, `no_widths`, `none`) because the loader derives the conventional
`restrict_<name>.json` or `restrict_<name>.dat` filename.

Compiled models are content-addressed by source, restriction, compiler/schema
versions, and normalization/tensor policies. A compiled model can be supplied
to `Generator.plan()` without writing a process artifact:

```console
pyamplicol model compile models/json/sm/sm.json models/sm.pyAmplicol-model.json \
  --restriction default
pyamplicol model processes "p p > Z j j" \
  --model models/sm.pyAmplicol-model.json \
  --multiparticle 'p=d,d~,g' --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 --max-quark-lines 2
```

## Prepared Kernel Bundles

The JSON file above is portable model IR only. A path ending in
`.pyamplicol-model` is instead a self-contained prepared bundle containing the
same IR, exact expressions, and one compiled local-kernel backend. Wheels ship
the portable built-in-SM `built-in-sm-jit-o2` bundle, which is selected
automatically by eager, recurrence, and on-the-fly execution. For example:

```console
pyamplicol generate "d d~ > z g g g" artifacts/ddbar_z3g_eager \
  --model built-in-sm \
  --execution-mode eager --color-accuracy nlc
```

Prepare an explicit bundle for an external model or a different built-in
backend:

```console
pyamplicol model compile models/json/sm/sm.json models/ufo-sm-jit-o2.pyamplicol-model \
  --backend jit --jit-optimization-level 2 --jit-compress
pyamplicol generate "d d~ > z g g g" artifacts/ddbar_z3g_ufo_eager \
  --model models/ufo-sm-jit-o2.pyamplicol-model --execution-mode eager
```

Process generation from this bundle copies only the referenced kernels into
the standalone process artifact. Eager mode writes compact invocation tables,
recurrence mode writes compact current schedules, and on-the-fly mode writes a
compact process seed from which query-local recurrence schedules are built.
None of these lanes compiles missing kernels during process generation. Their
symbolic generation layer still uses Symbolica and follows the normal
license/concurrency policy. A saved JIT application's post-generation `f64`
runtime is Symbolica-free; higher precision continues to use Symbolica for
eager and recurrence execution. On-the-fly execution supports native `f64`
only. LC retains physical flow selection; NLC and full colour expose a
singleton contracted component and intentionally do not expose a colour-flow
selector. Higher precision is not available for OTF.

JIT bundles retain SymJIT application/MIR state and rebuild executable code for
the receiving CPU when loaded. SymJIT storage-v3 prepared state is portable
across supported `x86_64` and `aarch64` hosts at optimization level 2.
pyAmpliCol therefore forces O2 for prepared JIT kernels, including when a
different level was requested. Process-local compiled JIT artifacts also
default to O2 and use the same portable outer target when every evaluator is
O2 JIT. Explicit O1/O3 JIT and C++/ASM artifacts remain target-native. C++ and
ASM receive batched inputs but do not gain SIMD from pyAmpliCol; SymJIT may
auto-vectorize its JIT applications.

## Multiprocess Expansion

One list-valued field covers single and multiple requests. The primary card
uses one inclusive request with explicit aliases:

```toml
[process]
entries = [{ expression = "p p > Z j j" }]
flavor_scheme = 2
max_quark_lines = 2

[process.multiparticles]
p = ["d", "d~", "g"]
j = ["d", "d~", "g"]
```

The UFO SM uses its declared particle names when parsing generation requests.
The primary expansion produces 19 ordered candidates, then retains one
representative for each incoming/outgoing permutation class. Seven of the
eight representatives have tree-level amplitudes. Candidate 19,
`g g > Z g g`, is loop-induced in the Standard Model and is reported and
omitted. A runtime may select a retained process in any side-preserving order
or by a stable name such as `p_p_to_z_j_j_4`; neither is an output directory
name.

Explicit process sets are also supported:

```python
from pyamplicol import ProcessRequest, ProcessSet

processes = ProcessSet(
    requests=(
        ProcessRequest.parse("u u~ > Z g", name="uubar_Zg"),
        ProcessRequest.parse("u u~ > Z g g", name="uubar_Zgg"),
    )
)
```

Names must be unique. Repetition such as `3*scalar_0` is available for model
particle names. Every model also receives the generic multiparticle `all`, in
model declaration order. It contains every valid propagating physical external
state and excludes ghosts, Goldstones, non-propagating records, and auxiliary
states. Explicit multiparticle definitions are merged over defaults, so
defining `p` or `j` does not remove `all`; an explicit `all` replaces the
default.

For example, the following is valid for the built-in SM and compiled JSON/UFO
models:

```console
pyamplicol model processes "p p > all all" \
  --model built-in-sm \
  --flavor-scheme 1 --max-quark-lines 0
```

`all` is intentionally broad. Products such as `p p > all all` may expand to
many candidates for a large UFO model; define a narrower multiparticle label
for routine production generation. External models do not inherit the complete
legacy alias table of the built-in SM.

## Built-In Compatibility Model

The hand-coded built-in Standard Model remains available for compatibility and
parity tests:

```toml
[model]
source = "built-in-sm"

[process]
entries = [{ expression = "u u~ > g g", name = "uubar_gg" }]
```

Its typed source is `ModelSource.built_in_sm()`. Built-in aliases and optimized
kernels are isolated compatibility behavior, not the reference taxonomy for
external models. Generic compilation uses model-declared spin, statistics,
color representation, mass/width, propagators, interactions, and exact quantum
numbers rather than absolute SM PDG ranges. Default and model-supplied UFO
propagators are distinguished from normalized expressions, independently of
their object names. Implementation parity and model-hardening gates for the
`0.1.1` release are tracked in [Release Status](release-status.md).

## Supported UFO Subset

The generic path currently supports:

- scalar, Dirac-like fermion, vector, and the proven massless spin-2 source and
  propagator paths;
- SU(3) singlet, fundamental, antifundamental, and adjoint particles;
- the implemented UFO Lorentz basis built from identity, gamma/gamma5,
  projectors, sigma, metric, and momentum tensors;
- trilinear kernels, color-singlet contact trees, and the proof-gated colored
  contacts used by the packaged models;
- LC flows and the current NLC/full contracted SU(3) calculations.

Model preflight rejects unsupported theory features before process generation,
including Majorana/FNV fermions, spin 3/2, sextet generation, epsilon color
tensors, multiple or non-SU(3) color groups, form factors/unknown functions,
unsupported custom propagator tensors, and general colored higher-point
contacts without a proven decomposition. Massive spin-2 remains experimental
and reports a warning; the packaged massless `scalar_gravity` model is a tested
capability.

`full` does not promise a general arbitrary-representation UFO color basis.
Unsupported input produces a structured diagnostic identifying the model
feature instead of falling back to built-in SM assumptions.
