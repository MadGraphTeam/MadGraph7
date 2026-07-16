<!-- SPDX-License-Identifier: 0BSD -->

# Models And Processes

## Built-In Standard Model

Use the built-in SM without a filesystem path:

```toml
[model]
source = "built-in-sm"
```

The equivalent typed source is `ModelSource.built_in_sm()`.

## External UFO And JSON

The public interface currently accepts a UFO directory, a serialized model
JSON file, or a compiled-model file by path:

```python
from pyamplicol import ModelSource

ufo = ModelSource.from_path("models/ufo/sm", restriction="restrict_default.dat")
serialized = ModelSource.from_path("models/json/scalars/scalars.json")
```

Relative restrictions are resolved from the model directory. TOML model paths
are resolved relative to the card.

UFO modules are Python and execute while loading. Treat them as code and load
only trusted UFO sources. Serialized JSON is preferable for portable workflows
that should not execute model code.

The distribution contains assets named `sm`, `scalars`, and `scalar_gravity`
in UFO and JSON forms. A stable public name resolver for those packaged assets
has not been implemented, so `model.source = "scalars"` is currently treated
as a relative path, not a package identifier. The copied examples can
materialize those wheel-owned assets without knowing the installation path:

```console
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples
python python/copy_packaged_models.py
pyamplicol external_json_scalars.toml
```

The resulting cards use paths such as `models/json/scalars/scalars.json`,
resolved relative to the copied card.

## Process Sets

One list-valued field represents single and multiple requests:

```toml
[process]
entries = [
  { expression = "d d~ > z g", name = "ddbar_zg" },
  { expression = "d d~ > z g g", name = "ddbar_zgg" },
]
```

Names are optional but must be unique when present. Mixed final-state
multiplicities may share one output artifact.

External-model process expansion supports model-defined particle names,
repetition such as `2*scalar_0`, default SM `p`/`j` multiparticles, and custom
definitions:

```toml
[process]
entries = [{ expression = "p p > z j" }]

[process.multiparticles]
p = ["d", "d~", "u", "u~", "g"]
j = ["d", "d~", "u", "u~", "g"]
```

`flavor_scheme`, `max_quark_lines`, and coupling-order policy constrain
expansion. With `coupling_order_policy = "explicit"`, set non-negative limits
under `[process.max_coupling_orders]`.

## Typed Requests

```python
from pyamplicol import ProcessRequest, ProcessSet

processes = ProcessSet(
    requests=(
        ProcessRequest.parse("d d~ > z g", name="ddbar_zg"),
        ProcessRequest.parse("d d~ > z g g", name="ddbar_zgg"),
    )
)
```

`Generator.plan(processes, model=...)` is the public non-writing validation
surface. `Generator.generate(...)` writes a transactional schema-v3 artifact
and emits its root multi-language `API/` bundle by default.
