---
title: "UMAMI and MadSpace"
nav_order: 3
parent: "Python API"
---
<!-- SPDX-License-Identifier: 0BSD -->
# UMAMI and MadSpace

The generated `API/umami` SDK exports a matrix-element shared library implementing
the [Unified MAtrix eleMent Interface (UMAMI)](http://docs.madgraph.org/latest/madspace/umami-api.html).
It lets a C/C++ executable or MadSpace evaluate a pyAmpliCol artifact through the
same Rusticol runtime used by the other [native APIs](native-apis.md).
The library needs no Python interpreter at evaluation time.

This interface is being developed on the `umami` branch. Use that branch's
matching generation code and installed native SDK; it is not present in an
older released wheel.

## Generate and build

Ordinary generation includes the API bundle by default:

```console
pyamplicol generate 'g g > g g g' ./artifacts/gg_ggg \
  --model built-in-sm --color-accuracy full --umami-grouping exact
make -C artifacts/gg_ggg/API/umami
make -C artifacts/gg_ggg/API/umami run
```

The generated directory contains `umami.h`, the adapter and provider tables, a
Makefile, `umami_driver.c`, `metadata.json`, and usage/licensing information.
The Makefile obtains headers and native link arguments from the active
pyAmpliCol environment:

```console
rusticol-config --cflags
rusticol-config --libs
```

If necessary, select the installation and build directory explicitly:

```console
make -C /absolute/artifact/API/umami \
  RUSTICOL_CONFIG=/absolute/venv/bin/rusticol-config \
  BUILD_DIR=/absolute/build/umami
/absolute/build/umami/umami_driver /absolute/artifact
```

The default build directory is the sibling
`.pyamplicol-api-build/<artifact-name>/umami`, outside the immutable artifact.
A single provider produces **`libumami.so`** and **`umami_driver`** on both Linux
and macOS; the `.so` suffix is retained on macOS. Rusticol is statically included
in the provider and its symbols are hidden. The standalone driver links only
the UMAMI library, not another copy of Rusticol. It reads a saved validation
point and prints the matrix element as JSON. Optional arguments select a
zero-based channel and flavour:

```console
/absolute/build/umami/umami_driver /absolute/artifact 0 0
```

### Link your executable

Include the generated `umami.h` and link your executable to the generated
library. The generated Makefile and driver are complete examples. For a
single-provider library and paths without shell quoting requirements:

```sh
# Linux: place the executable beside libumami.so.
cc -I/absolute/artifact/API/umami my_driver.c \
  -L/absolute/build/umami -lumami -Wl,-rpath,'$ORIGIN' \
  -o /absolute/build/umami/my_driver

# macOS: the filename remains libumami.so.
cc -I/absolute/artifact/API/umami my_driver.c \
  /absolute/build/umami/libumami.so -Wl,-rpath,@loader_path \
  -o /absolute/build/umami/my_driver
```

Keep the executable and provider together when relocating them, and retain the
whole generated artifact and any native evaluator dependencies. `rusticol-config`
is needed to **build the provider**, not to add a second Rusticol library to its
consumer. Distribute the generated `UMAMI_LICENSE` notice with the provider;
the upstream public header retains its license, independently of the adapter's
0BSD license.

## Initialization, inputs and normalization

This adapter intentionally gives `param_card_path` a different meaning from the
usual UMAMI convention: pass the **absolute artifact-directory path**, not a
parameter-card file.

```c
#include "umami.h"

UmamiHandle handle = NULL;
UmamiStatus status = umami_initialize(&handle, "/absolute/artifact");
if (status != UMAMI_SUCCESS) {
    /* Report the error; do not evaluate this handle. */
}
/* Evaluate using umami_matrix_element, then release the handle. */
umami_free(handle);
```

Each handle starts from the saved model defaults and owns independent mutable
state. Use a separate handle per thread. The provider checks the artifact's
external states against its generated tables. Metadata is available before
initialization because it is compiled into the library.

Momenta have components `(E, px, py, pz)` and UMAMI's column-major layout:

```text
momenta[event + stride * (particle + particle_count * component)]
```

`count` events are evaluated beginning at `offset`, with
`offset + count <= stride`. Scalar input and output arrays use the same event
indices. Values outside the requested output slice are untouched. Inputs must
be finite and indices valid. Check every returned `UmamiStatus`.

| Input | Meaning |
| --- | --- |
| `UMAMI_IN_MOMENTA` | Required momenta in the physical external order from metadata. |
| `UMAMI_IN_CHANNEL_INDEX` | Zero-based index into `channels`; defaults to zero. |
| `UMAMI_IN_FLAVOR_INDEX` | Zero-based index into that channel's `processes`; defaults to zero. |
| `UMAMI_IN_ALPHA_S` | Optional per-event strong coupling, only when the model supplies an unambiguous declaration. |
| `UMAMI_IN_RANDOM_HELICITY` | A number in `[0,1)` for an informational helicity label; not a sampled matrix-element estimator. |

The matrix-element output sums the artifact's **available helicities**. A
generation restricted to one helicity is therefore not an unpolarized provider;
consult `runtime_processes[].coverage` and `helicity_ids` in the metadata.
The result already includes Rusticol's incoming spin/colour averaging and
identical-final-state normalization. Do not apply these factors again.

LC channels select additive physical colour contributions. NLC/full-colour
channels return the already contracted colour result, not a separately sampled
colour flow. In particular, the absence of an LHE colour-flow reference for a
contracted result is intentional. Metadata integration multiplicities are
**not** silently included in `UMAMI_OUT_MATRIX_ELEMENT`: the integrator applies
them once when assembling physical contributions.

The adapter reports supported and required inputs and supported outputs through
`umami_supported_inputs`, `umami_required_inputs` and
`umami_supported_outputs`. Besides matrix elements, it supplies informational
colour/helicity indices. Explicit helicity selection, random colour selection,
diagram-resolved outputs and GPU execution are currently unsupported. Query
capabilities rather than assuming every UMAMI operation is implemented.

### Parameters

`umami_set_parameter` and `umami_get_parameter` use the model's parameter names.
Updates use Rusticol's validated setter and refresh dependent parameters;
immutable or invalid values are rejected. Event-wise αs overrides are temporary
and restore the persistent handle value after evaluation.

An αs input is advertised only for an explicit model mapping or a standard UFO
external `SMINPUTS[3]` declaration. It is not guessed from particle names or
coupling powers. The built-in demonstration model has no such declaration, so
its integration uses the saved coupling defaults; a suitable UFO model can
support the optional running-coupling workflow.

`UMAMI_META_MASSES` contains the **saved default** external masses and is not
handle-specific. An integrator using these fixed masses must not change mass
parameters without also updating its kinematics and checking the grouping
assumptions. Regenerating the provider at the desired masses is the simplest
consistent workflow.

## Providers and generated metadata

UMAMI metadata is fixed for a library, so one library cannot transparently
switch between incompatible external sizes or mass patterns. Compatible
processes share a provider; otherwise the SDK emits `libumami_p0.so`,
`libumami_p1.so`, and corresponding `umami_driver_p0`, etc. The root
`metadata.json` is then a provider index pointing to
`providers/p0/metadata.json`, etc. Select a matching library and metadata entry;
do not sum alternative providers without checking their process selections.

For example:

```console
make -C /absolute/artifact/API/umami run PROVIDER=p0
```

A provider's metadata contains:

- `provider`: fixed external size, incoming count, masses, colour accuracy and
  optional αs parameter.
- `channels[].processes`: the channel/flavour selectors consumed by UMAMI, with
  physical PDGs and integration weights referenced by `matrix_elements`.
- `pdg_ids`, `color_orders`, `color_flows` and `helicities`: shared physical
  configuration tables. Indices are zero-based and colour-flow references can
  be absent for contracted colour.
- `runtime` and `members`: the representative evaluator, certified momentum
  permutations and covered physical contributions.
- `grouping` and `runtime_processes`: the assumptions and available coverage
  needed to interpret those contributions correctly.

Physical incoming PDGs remain oriented for PDF evaluation. Aliases name the
same computation; they are not automatically extra integrable contributions.
Each exported contribution uses one phase-space ordering, rather than several
alternative maps whose naive sum would double count it.

## Grouping modes

Choose the export setting with `--umami-grouping`, or set
`generation.umami_grouping` in TOML/Python:

| Mode | Meaning |
| --- | --- |
| `exact` (default) | Reuse proven computations while retaining physical contributions and their explicit momentum maps. No flavour-blind-cut or PDF symmetry assumption. |
| `flavour_blind_observables` | Additionally compress proven integration orbits under compatible final-state permutation/flavour-blind cuts and observables. Multiplicities are explicit in metadata. |
| `none` | No UMAMI-level grouping; existing evaluator optimizations remain active. |

Equivalence is derived from model expressions and generated computation, not
equal default parameters, particle categories or a hard-coded Standard Model
process list. Unproved relations remain separate. Compression never assumes
equal PDFs, exchanges the two beams, or identifies unequal mass parameters.

In the compressed mode a representative's weighted integral reproduces its
orbit **only under the recorded assumptions**; it need not equal the
uncompressed pointwise sum at a given labelled momentum point. Use `exact` or
`none` for flavour-sensitive cuts or observables, and apply them to the physical
contribution in question. Changing this export setting does not change the
numerical evaluator or invalidate a native build.

## MadSpace example

The repository's
[`umami_example`](https://github.com/mg5amcnlo/pyamplicol/tree/umami/umami_example)
contains close adaptations of AmpliCol's loading and integration examples. They
use the real MadSpace matrix-element loader, phase-space mappings and
differential cross section, with VEGAS from MadNIS. The default is a bounded
partonic `g g > g g g` run for LC and full colour, with no PDF downloads:

```console
python -m pip install -r umami_example/requirements.txt
python umami_example/run.py
```

The orchestrator generates both artifacts, compiles the libraries and linked
drivers, checks batched UMAMI/native numerical agreement, and writes integrated
cross sections with statistical uncertainties. It disables numerical
current-relation discovery during generation. Optional PDF and neural-flow
workflows, provider selection, cuts and all commands are explained in the
example README. Those dependencies are not part of pyAmpliCol's core install.
