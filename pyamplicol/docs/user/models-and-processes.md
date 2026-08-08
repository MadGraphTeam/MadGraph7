---
title: "Models and Processes"
nav_order: 1
parent: "Configuration"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Models and Processes

pyAmpliCol separates **model compilation** from **process generation**. A model
describes particles, parameters, interactions, propagators, Lorentz structures,
and color data. A process request selects external states and asks pyAmpliCol to
construct the supported tree amplitudes and executable runtime artifact.

## Model source choices

| Source | Typical use | Executes Python while loading? |
| --- | --- | --- |
| `built-in-sm` | Fast Standard Model compatibility workflows and prepared recurrence/eager/OTF kernels | No |
| Serialized JSON | Recommended portable external-model input | No |
| UFO directory | Trusted development input | **Yes** |
| Compiled model JSON | Reuse normalized model IR | No |
| `.pyamplicol-model` | Reuse model IR plus a prepared local-kernel backend | No |

## Built-in Standard Model

```toml
[model]
source = "built-in-sm"

[process]
entries = [{ expression = "u u~ > g g", name = "uubar_gg" }]
```

Direct CLI:

```console
pyamplicol generate 'u u~ > g g' artifacts/uubar_gg \
  --model built-in-sm
```

The built-in model selects the wheel-owned prepared JIT O2 bundle for default
recurrence or eager execution. It also retains compatibility aliases that an
arbitrary external model does not automatically inherit.

## Serialized JSON model

Serialized JSON is the recommended portable external-model form:

```toml
[model]
source = "models/json/sm/sm.json"
restriction = "default"
```

Copy the packaged JSON/UFO examples into an editable workspace:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
pyamplicol model inspect models/json/sm/sm.json
```

The wheel includes serialized `sm`, `scalars`, and `scalar_gravity` examples.
They become ordinary filesystem paths under the copied workspace; no code needs
to know the package installation directory.

If only the packaged models are wanted, the copied
`python/copy_packaged_models.py` helper can populate a separate empty model
workspace:

```console
python python/copy_packaged_models.py ../pyamplicol-models
```

## Trusted UFO model

A UFO directory uses the same source field:

```toml
[model]
source = "/path/to/MyUFO"
restriction = "default"

[process]
entries = [{ expression = "u u~ > z g" }]

[generation]
output = "artifacts/my_ufo_process"

[evaluator]
execution_mode = "compiled"
backend = "jit"
```

> UFO modules are Python code and execute during loading. Only load a UFO model
> from a source you trust. Serialize it for portable or automated workflows.

In a TOML card, `restriction` is normally a loader name such as `default`,
`no_widths`, or `none`. With the typed Python API, an explicit restriction file
may be supplied. Relative typed-API restriction paths are resolved from the
model directory; card names select the conventional `restrict_<name>.json` or
`restrict_<name>.dat` file.

## Model commands

Inspect model capabilities and diagnostics:

```console
pyamplicol model inspect models/json/sm/sm.json --restriction default
```

Compile portable normalized model IR:

```console
pyamplicol model compile \
  models/json/sm/sm.json models/sm.pyAmplicol-model.json \
  --restriction default
```

Enumerate concrete processes without generating a process artifact:

```console
pyamplicol model processes 'p p > Z j j' \
  --model models/sm.pyAmplicol-model.json \
  --multiparticle 'p=d,d~,g' \
  --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 --max-quark-lines 2
```

Unsupported model features are reported during model preflight, before an
expensive process build.

## Prepared kernel bundles

A file ending in `.pyamplicol-model` contains normalized model IR, exact
expressions, and one prepared local-kernel backend. It enables recurrence,
eager, or on-the-fly process generation without compiling each local kernel
again.

Create a JIT O2 prepared bundle:

```console
pyamplicol model compile \
  models/json/sm/sm.json models/ufo-sm-jit-o2.pyamplicol-model \
  --backend jit --jit-optimization-level 2 --jit-compress
```

Use it for eager generation:

```console
pyamplicol generate 'd d~ > z g g g' artifacts/ddbar_z3g_eager \
  --model models/ufo-sm-jit-o2.pyamplicol-model \
  --execution-mode eager
```

The three prepared-kernel lanes store different process-level structures:
eager writes compact invocation tables, recurrence writes compact current
schedules, and on-the-fly writes a compact process seed from which the selected
query-local recurrence family is built. Each completed process artifact carries
the referenced kernels and is standalone; the source prepared bundle is not
needed at runtime.

At native `f64`, OTF supports LC helicity/flow selection and the singleton
contracted color axis used by NLC and full colour. Contracted color rejects LC
flow selectors and is intended as a low-multiplicity correctness capability,
not a high-multiplicity practicality promise. OTF is binary64-only; eager and
recurrence can retain the Symbolica-backed exact path when their artifact does.

Prepared JIT state uses the exact-O2 portability contract across supported
64-bit little-endian macOS and Linux hosts. Process-local compiled JIT O1 and
O2 are also portable; C++/ASM and explicit O0/O3 JIT remain target-native. See
[Artifacts and Portability](artifacts-and-portability.md).
See [Generation Modes and Evaluators](generation-modes-and-evaluators.md) for
the generation and warm-state trade-offs.

## Typed model API

```python
from pyamplicol import ModelSource

json_source = ModelSource.from_path("models/json/sm/sm.json")
ufo_source = ModelSource.from_path(
    "/path/to/MyUFO",
    restriction="restrict_default.dat",
)

compiled = json_source.compile()
print(compiled.name)
print(compiled.capabilities.supported_color_accuracies)
print(compiled.supported)

compiled.write("models/sm.pyAmplicol-model.json")
compiled.write_parameter_card("data/sm_parameters.json")
```

`CompiledModel` is an immutable public handle. Stable metadata is available
through `compiled.info`; compiler-owned expression/tensor IR remains private.
Compiled models are content-addressed by their source, restriction,
compiler/schema versions, and normalization/tensor policies.

See [Python API](python-api.md) for generation with a compiled handle.

## Parameter restrictions and runtime cards

Model compilation restrictions determine the model baked into the artifact.
Mutable external UFO parameters may then be updated at runtime.

A generated parameter card is a flat JSON mapping:

```json
{
  "aS": 0.118,
  "MZ": 91.1876,
  "complex_parameter": [1.0, 0.0]
}
```

Each value is a finite real number or `[real, imaginary]`. Apply it through the
runtime CLI:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --model-parameters data/model_parameters.json \
  --momenta data/pp_zjj_momenta.json
```

The generated standalone API drivers also accept repeated
`--set-parameter NAME REAL IMAG` options; those overrides are applied after the
card and win over its values. The main `pyamplicol evaluate` command uses the
`--model-parameters` card, while Python callers can use
`Runtime.set_model_parameter()` or `Runtime.set_model_parameters()`. Updates
are atomic: unknown, immutable, invalid, or non-finite input rejects the whole
batch.

## Process expressions

A process has incoming and outgoing particles separated by `>`:

```text
d d~ > Z g g
```

Named requests provide stable IDs:

```toml
[process]
entries = [
  { expression = "u u~ > Z g", name = "uubar_Zg" },
  { expression = "u u~ > Z g g", name = "uubar_Zgg" },
]
```

Repetition syntax such as `3*scalar_0` is supported for particle names.

## Multiparticles

```toml
[process]
entries = [{ expression = "p p > Z j j" }]
flavor_scheme = 2
max_quark_lines = 2

[process.multiparticles]
p = ["d", "d~", "g"]
j = ["d", "d~", "g"]
```

Every model supplies `all`, in declaration order, containing its valid
propagating physical external states. Ghosts, Goldstones, non-propagating
records, and auxiliary particles are excluded. User labels merge over defaults;
an explicit `all` overrides the generic one.

Example enumeration:

```console
pyamplicol model processes 'p p > all all' \
  --model built-in-sm --flavor-scheme 1 --max-quark-lines 0
```

`all` is intentionally broad and may produce a combinatorial expansion for a
large UFO model. Prefer a focused custom label when possible.

## Permutation-equivalent processes

Multiprocess expansion stores one representative for candidates that differ
only by reordering within the incoming side and/or within the outgoing side.
At runtime, this is valid:

```console
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > g z g'
```

even when the representative is stored as `d d~ > Z g g`. Rusticol maps the
public particle order centrally across momenta, particle metadata, helicities,
LC flow words, selectors, reductions, and resolved values.

Rules:

- incoming particles may only permute with incoming particles;
- outgoing particles may only permute with outgoing particles;
- repeated identical particles use deterministic first-unused matching;
- ambiguous representative matches fail with candidate stable IDs;
- stable process and explicit alias IDs take precedence over inferred matching.

## Tree-level support and omitted candidates

Multiparticle expansion may find a syntactically valid process with no
model-supported tree-level amplitude. pyAmpliCol reports and omits it rather
than creating a zero or broken artifact. In the primary SM example,
`g g > Z g g` is loop-induced and therefore omitted.

This is distinct from an unsupported model feature: model preflight rejects
unsupported theory structures with an actionable diagnostic.

## Current generic UFO support

The generic path covers the proven subset used by shipped tests and examples,
including:

- scalar, Dirac-like fermion, vector, and proven massless spin-2 paths;
- SU(3) singlet, fundamental, antifundamental, and adjoint particles;
- implemented identity, gamma/gamma5, projector, sigma, metric, and momentum
  Lorentz structures;
- trilinear kernels, color-singlet contact trees, and proof-gated colored
  contacts;
- LC flows and contracted NLC/full SU(3) calculations.

Preflight rejects Majorana/FNV fermions, spin 3/2, sextets, epsilon color
tensors, multiple/non-SU(3) color groups, unknown form-factor functions,
unsupported custom propagator tensors, and unproven general colored
higher-point contacts. Massive spin-2 is experimental; the packaged massless
`scalar_gravity` model is a tested path.

For the validated release boundary, see
[Release and Support](release-and-support.md).
