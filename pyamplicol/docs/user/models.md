<!-- SPDX-License-Identifier: 0BSD -->

# Models And Processes

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

## Prepared Eager Bundles

The JSON file above is portable model IR only. A path ending in
`.pyamplicol-model` is instead a self-contained prepared bundle containing the
same IR, exact expressions, and one compiled local-kernel backend. Wheels ship
built-in-SM JIT O3 bundles for `x86_64` and `aarch64`; the matching host bundle
is selected automatically by:

```console
pyamplicol generate "d d~ > z g g g" artifacts/ddbar_z3g_eager \
  --model built-in-sm \
  --execution-mode eager --color-accuracy nlc
```

Prepare an explicit bundle for an external model or a different built-in
backend:

```console
pyamplicol model compile models/json/sm/sm.json models/ufo-sm-jit-o3.pyamplicol-model \
  --backend jit --jit-optimization-level 3
pyamplicol generate "d d~ > z g g g" artifacts/ddbar_z3g_ufo_eager \
  --model models/ufo-sm-jit-o3.pyamplicol-model --execution-mode eager
```

Process generation from this bundle writes compact invocation tables and
copies only the referenced kernels into the standalone process artifact. It
does not construct evaluators or invoke a compiler. Eager generation still
uses Symbolica for the symbolic generation layer and follows the normal
license/concurrency policy. A saved JIT application's post-generation f64
runtime is Symbolica-free; higher precision continues to use Symbolica.

JIT bundles retain SymJIT application/MIR state and rebuild executable code for
the receiving CPU when loaded. SymJIT storage-v3 state remains scoped to one
architecture class: transfer between supported operating systems on the same
architecture is tested, but transfer between `x86_64` and `aarch64` is rejected
before DAG construction or SymJIT loading. The model IR inside the bundle
remains portable, and a future SymJIT storage ABI may allow prepared packs to
cross architecture classes. C++ and ASM bundles are target-native. C++ and ASM
receive batched inputs but do not gain SIMD from pyAmpliCol; SymJIT may
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

The UFO SM uses its declared particle names when parsing generation
requests. The current generation filter retains 19 concrete processes. A
runtime may select one through the concrete expression recorded by artifact
inspection or a stable name such as `p_p_to_z_j_j_4`; neither is an output
directory name.

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
particle names. External models receive generic model-derived particle
catalogs and may define `p`/`j`; they do not inherit the complete legacy alias
table of the built-in SM. Define multiparticles explicitly whenever exact
expansion is part of a reproducible workflow.

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
`0.1.0` release are tracked in [Release Status](release-status.md).

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
