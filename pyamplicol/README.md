<!-- SPDX-License-Identifier: 0BSD -->

# pyAmpliCol

pyAmpliCol generates optimized matrix-element evaluators and loads them through
the Rusticol runtime from Python, C++17, or Fortran 2008. It accepts the
built-in Standard Model and external UFO or serialized JSON models.

## Install And Run

The installed-wheel workflow for the `0.1.0` release is:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
pyamplicol examples copy ./pyamplicol-examples
pyamplicol generate "d d~ > z g" outputs/ddbar_zg
pyamplicol evaluate outputs/ddbar_zg \
  --momenta ./pyamplicol-examples/data/ddbar_zg_momenta.json
```

**Release status:** `pyamplicol==0.1.0` is not published yet. In this source
snapshot, schema-v3 generation and its generated root `API/` bundle are
integrated, as are the Python-to-Rusticol runtime adapter and benchmark service.
The commands above are the intended installed-wheel workflow; use the
contributor setup while exact dependencies and release wheels are still gated.
See [Release Status](docs/user/release-status.md) for the remaining integration
and publication gates.

A TOML card uses the same fields and can be overridden without editing it:

```console
pyamplicol examples copy ./pyamplicol-examples
pyamplicol ./pyamplicol-examples/builtin_sm_lc.toml \
  --set generation.output=outputs/ddbar_zg \
  --set evaluator.jit.optimization_level=2
```

The card parser and resolver are available now. Paths in a card, including
path-valued `--set` overrides applied to that card, are resolved relative to
the card. Dedicated path arguments in a direct command are resolved relative
to the current directory.

## Installation Choices

| Goal | Command | Dependency source |
| --- | --- | --- |
| Install a released binary wheel | `python -m pip install pyamplicol` | PyPI; no Rust compiler |
| Build and install this checkout | `python -m pip install .` | Published packages only |
| Keep a local wheel | `just wheel` | Published packages only; writes `dist/` |
| Install the matching retained wheel | `just install-wheel PYTHON=/path/to/python` | Existing or newly built wheel |
| Prepare a contributor environment | `just dev-install` | Pinned candidate revisions and checked patches |

Source and wheel builds require Python 3.11 or newer, Rust 1.89 or newer, and
a C/C++ toolchain. A Fortran compiler is needed only to compile Fortran
consumers. Contributor candidates are deliberately non-publishable:

```console
just dev-install
PYTHON=.venv/bin/python just dev-test
just --list
```

Strict release-equivalent builds remain gated by the dependency compatibility
state in `dependencies/release-lock.toml`. See
[Installation](docs/user/installation.md) for the distinction between release
and candidate mode.

## Steering

### TOML

```toml
schema_version = 1
action = "generate"

[model]
source = "built-in-sm"

[process]
entries = [{ expression = "d d~ > z g", name = "ddbar_zg" }]

[color]
accuracy = "lc"

[generation]
output = "outputs/ddbar_zg"
```

`examples/all_options.toml` documents every current schema field. Unknown
fields are rejected. Resolution precedence is defaults, card, dedicated CLI
flags, ordered `--set dotted.path=value` overrides, then recorded
license/resource clamps.

### Direct CLI

```console
pyamplicol generate "d d~ > z g" outputs/mixed \
  --process "d d~ > z g g" \
  --name ddbar_zg --name ddbar_zgg \
  --color-accuracy nlc --workers 2
```

The CLI exposes `generate`, `evaluate`, `benchmark`, `inspect`, model commands,
the two Symbolica license-request commands, and these local utilities:

```console
pyamplicol examples list
pyamplicol examples copy ./pyamplicol-examples
pyamplicol config template run.toml
pyamplicol config resolve run.toml --format json
pyamplicol doctor
pyamplicol self-test
pyamplicol generate "d d~ > z g" --dry-run
```

`generate --dry-run` and programmatic `Generator.plan()` validate and describe
the concrete process plan without writing an artifact. `examples run` refreshes
a versioned user-cache copy before execution, so it never writes into the
installed package; use `examples copy` for an editable visible workspace.
External-model cards can materialize the wheel-owned UFO/JSON examples without
referring to this checkout:

```console
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples
python python/copy_packaged_models.py
pyamplicol external_json_scalars.toml
```

### Typed Python

```python
from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config

resolution = resolve_config(
    {
        "schema_version": 1,
        "action": "generate",
        "process": {
            "entries": [
                {"expression": "d d~ > z g", "name": "ddbar_zg"},
                {"expression": "d d~ > z g g", "name": "ddbar_zgg"},
            ],
        },
        "color": {"accuracy": "lc"},
    }
)
processes = ProcessSet(
    tuple(
        ProcessRequest.parse(entry.expression, name=entry.name)
        for entry in resolution.effective.process.entries
    )
)
generator = Generator(resolution)
plan = generator.plan(processes, model=ModelSource.built_in_sm())
result = generator.generate(processes, "outputs/mixed")
print(result.output)
```

The package root is intentionally lightweight: importing `pyamplicol` does
not import Symbolica or execute UFO Python modules.

## Build And Physics Choices

- `color.accuracy = "lc"` keeps physical leading-color flows. `"nlc"` and
  `"full"` return a single contracted color component per helicity.
- LC generation always includes every physical leading-color flow. Runtime
  `evaluation.color_flow_ids` and `benchmark.color_flow_ids` select subsets
  without limiting artifact coverage. NLC/full expose one contracted color
  component per helicity and reject color-flow selectors.
- `evaluator.backend = "jit"` is the default at optimization level 3. `"asm"`
  selects Symbolica assembly evaluation; `"cpp"` emits/compiles C++ with its
  own compiler settings.
- JIT generation embeds a self-contained SymJIT application for f64 execution.
  Loading and evaluating that artifact through Rusticol does not import
  Symbolica, require a Symbolica license, or inherit restricted-mode generation
  limits. Arbitrary-precision evaluation and non-JIT evaluator artifacts remain
  Symbolica-backed. Decimal kinematics preserve their supplied digits; values
  originating as binary64 are upcast with trailing zeros, not treated as if
  additional input information were available. The requested decimal precision
  controls arithmetic and output rounding; it is not a certification of that
  many physically accurate digits.
- `generation.mode` is `"error"`, `"append"`, or `"replace"`. Schema-v3
  generation writes transactionally and validates payload hashes and runtime
  metadata before reporting success.
- `Runtime.evaluate()` returns one summed value per phase-space point.
  `evaluate_resolved()` retains helicity and color dimensions, and its
  `total()` must reproduce the summed result.

External UFO models execute trusted Python while loading. Use UFO only from a
trusted source; prefer serialized JSON when portability and non-execution are
important. The bundled model assets are named `sm`, `scalars`, and
`scalar_gravity`, but the current public interface accepts filesystem paths,
not those package names. The external-model cards therefore use paths relative
to each card, populated from wheel-owned assets by the example copy helper.
See [Models And Processes](docs/user/models.md).

## Runtime Use

```python
from pyamplicol import Runtime

runtime = Runtime.load("outputs/ddbar_zg", process="ddbar_zg")
runtime.set_model_parameter("normalization.alpha_s_me_check", 0.118)

momenta = [[
    [500.0, 0.0, 0.0, 500.0],
    [500.0, 0.0, 0.0, -500.0],
    [504.157625672, -304.1084262865, 208.7602652353, 331.3561179451],
    [495.842374328, 304.1084262865, -208.7602652353, -331.3561179451],
]]
total = runtime.evaluate(momenta)
resolved = runtime.evaluate_resolved(momenta)
assert resolved.total() == total
```

Schema-v3 generation emits one root `API/` bundle with Python, C++, and Fortran
drivers. The bundle includes deterministic validation points for concrete
processes and aliases when such a point is available. Python can run the
generated driver directly:

```console
python outputs/ddbar_zg/API/python/check_standalone.py \
  --process ddbar_zg --json
```

Native consumers locate the installed headers, static library, Fortran module
source, CycloneDX dependency SBOM, and platform link flags through
`rusticol-config`; they do not depend on a checkout:

```console
rusticol-config --json
make -C outputs/ddbar_zg/API/cpp run ARGS='--process ddbar_zg --json'
make -C outputs/ddbar_zg/API/fortran run ARGS='--process ddbar_zg --json'
```

The generated Makefiles place native build products in a sibling
`.pyamplicol-api-build/` directory, outside the integrity-checked process
artifact.

The SDK metadata is present only in a built wheel. See
[Runtime](docs/user/runtime.md) and [Native SDK](docs/user/native-sdk.md).

Process artifacts contain evaluator programs that Rusticol compiles or loads
for execution. Manifest hashes detect accidental modification but do not prove
who produced an artifact. Load artifacts only from a trusted source, just as
you would a native library.

## Symbolica License

Generation can continue in Symbolica's restricted mode. pyAmpliCol queries
`symbolica.is_licensed()` at first generation use instead of inferring license
state from an environment variable. If no valid license is active, it suggests
the appropriate request command once and clamps generation to one process
worker and one Symbolica core. This does not impose a global limit on
independent Rusticol runtime handles; one mutable handle must not be called
concurrently.

Symbolica is a generation and high-precision dependency, not an f64 runtime
dependency for the default JIT artifact format. A generated JIT process can be
deployed with the Rusticol Python or native APIs and evaluated at f64 precision
without `SYMBOLICA_LICENSE`. ASM and C++ evaluator artifacts retain their
explicit Symbolica runtime capability and are not accepted by the lightweight
native SDK.

```console
pyamplicol request-symbolica-trial-license
pyamplicol request-symbolica-hobbyist-license
export SYMBOLICA_LICENSE='issued-key'
```

For noninteractive requests, provide all requested identity fields and `--yes`.
pyAmpliCol does not retain those fields or print the key. Use
`--no-symbolica-suggestion` or `[symbolica] suggest_license = false` to hide
the acquisition reminder and set `SYMBOLICA_HIDE_BANNER=1` before import.
JSON output suppresses the banner automatically.

Symbolica emails requested keys. Set the received value in
`SYMBOLICA_LICENSE` before generation. Trial licenses are intended for
professional evaluation; hobbyist licenses are for eligible unaffiliated use.
Consult [Symbolica's current licensing guidance](https://symbolica.io/docs/get_started.html)
for the applicable terms. See [Symbolica Licensing](docs/user/symbolica.md) for
interactive and noninteractive pyAmpliCol examples.

## Documentation And License

- [User Guide](docs/user/index.md)
- [Examples](examples/README.md)
- [Configuration](docs/user/configuration.md)
- [Symbolica Licensing](docs/user/symbolica.md)

Original pyAmpliCol and Rusticol source is licensed under the BSD Zero Clause
License (`0BSD`). Symbolica and bundled third-party model material retain
their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
