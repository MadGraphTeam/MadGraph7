<!-- SPDX-License-Identifier: 0BSD -->

# Packaged Examples

All TOML cards use schema version 1 and resolve paths relative to themselves.
Create an editable, installation-independent workspace with:

```console
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples
```

The primary example uses the serialized external Standard Model and generates a
19-subprocess `p p > Z j j` artifact:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
pyamplicol evaluate_resolved.toml
pyamplicol benchmark.toml
```

`evaluate_total.toml` selects the concrete `d d~ > z g g` process by its
readable expression. The equivalent stable ID is `p_p_to_z_j_j_4`. Its
parameter card updates the genuine UFO external inputs `aS` and `MZ`.

## Run Cards

| File | Coverage |
| --- | --- |
| `generate_pp_zjj_from_ufo_sm.toml` | Generate a multiprocess `p p > Z j j` artifact from the portable serialized UFO SM |
| `evaluate_total.toml` | Optimized total for one `pp_zjj` subprocess |
| `evaluate_resolved.toml` | Helicity/color-resolved evaluation and explicit sum |
| `benchmark.toml` | Short benchmark of the same selected subprocess |
| `benchmark_z6g_single_flow_helicity_sum.toml` | Profile the default topology-replay layout for runtime flow selection with a helicity sum in `u u~ > Z + 6g` |
| `benchmark_z6g_all_flows_single_helicity.toml` | Profile the explicit all-flow-union layout for an all-flow sum at one runtime-selected helicity in `u u~ > Z + 6g` |
| `benchmark_z6g_generation_specialized_flow_helicity_sum.toml` | Generation-specialized flow baseline for the reusable-selector comparison |
| `benchmark_z6g_generation_specialized_all_flows_single_helicity.toml` | Generation-specialized helicity baseline for the reusable-selector comparison |
| `process_set_mixed_multiplicity.toml` | Named UFO-SM 2-to-2 and 2-to-3 requests |
| `external_ufo_sm.toml` | Trusted UFO execution path |
| `external_json_scalars.toml` | Scalar contact model and repeated particles |
| `external_json_scalar_gravity.toml` | Proven massless spin-2 model path |
| `builtin_sm_lc.toml` | Built-in compatibility SM, `u u~ > g g`, LC |
| `builtin_sm_nlc.toml` | Built-in compatibility SM, contracted NLC |
| `builtin_sm_full.toml` | Built-in compatibility SM, contracted full color |
| `builtin_sm_eager.toml` | Built-in SM LC generation using the wheel-owned prepared JIT O3 pack |
| `all_options.toml` | Every current schema field, active and commented |

## Reproduce The Z-Ladder Workloads

These acceptance cards use two separately generated compiled-mode artifacts so
their generation and profile logs remain independent. The first keeps the
default `topology-replay` LC layout, selects flow ordinal `1` only at profile
time, and sums all helicities. The second explicitly sets
`color.lc_flow_layout = "all-flow-union"`, sums all flows, and selects one
stable helicity ID only at profile time. The union shares currents across the
physical flows and is optimized for this second workload.

Both artifacts retain all physical LC flows and all physical helicities. Either
can select any retained flow or helicity globally or per point at runtime;
neither card fixes a selector at generation time. Generate and profile them
independently:

```console
pyamplicol generate --card benchmark_z6g_single_flow_helicity_sum.toml
pyamplicol profile --card benchmark_z6g_single_flow_helicity_sum.toml

pyamplicol generate --card benchmark_z6g_all_flows_single_helicity.toml
pyamplicol profile --card benchmark_z6g_all_flows_single_helicity.toml
```

Both profiles use native Rusticol wall timing, JIT O3, 64-point batches, two
warmups, at least five samples, and a twenty-second target. The cards use
`u u~ > Z g g g g g g` as requested; the PDF's literal reference family uses
`d d~ > Z + (n-1)g`, so replace `u u~` with `d d~` when reproducing that exact
row rather than the equivalent up-quark topology.

The runtime selector can be changed without regenerating. For example:

```console
pyamplicol profile \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --color-flow 2

pyamplicol profile \
  --card benchmark_z6g_all_flows_single_helicity.toml \
  --helicity h:-1,+1,-1,+1,+1,-1,+1,-1,+1
```

Python's `Runtime.evaluate(..., color_flow_by_point=..., helicity_by_point=...)`
accepts one physical selector per phase-space point. Homogeneous, alternating,
and randomized selector batches all use the same complete artifact.

The layout is also a dedicated CLI option. For example, this creates a union
artifact from the otherwise topology-replay card:

```console
pyamplicol generate \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --lc-flow-layout all-flow-union \
  --set generation.output=artifacts/uubar_z6g_all_flows_override
```

`topology-replay` remains the global default because it is optimized for a
single runtime-selected flow with a helicity sum. `all-flow-union` is LC-only;
configuration rejects it for NLC/full and for LC requests with a
generation-selected color sector, a generation-selected helicity, or truncated
color coverage.

The execution mode can be overridden directly at invocation time. Use a
different output so the eager artifact cannot collide with the compiled one:

```console
pyamplicol generate \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --execution-mode eager \
  --set generation.output=artifacts/uubar_z6g_single_flow_helicity_sum_eager
```

The eager override keeps the same complete reusable-selector contract and the
card's LC flow layout. Eager generation and execution apply flow and helicity
choices through Rusticol's selector schedule rather than generation-time
specialization.

`examples copy` also materializes wheel-owned `sm`, `scalars`, and
`scalar_gravity` resources into `models/`. The included
`python/copy_packaged_models.py` helper performs the resource-only operation for
a separate workspace and refuses to merge into a non-empty destination unless
`--force` is supplied. The public model API accepts filesystem paths, so no
example relies on an installation directory.

## Typed Python

Plan or generate the primary process through the typed API:

```console
python python/typed_generation.py artifacts/pp_zjj --plan-only
python python/typed_generation.py artifacts/pp_zjj
```

The script compiles the external JSON model before planning, carries explicit
`p`/`j` definitions in the resolved configuration, and writes the same root
API bundle as the CLI.

Evaluate with a parameter card plus a direct override:

```console
python python/runtime_evaluation.py \
  artifacts/pp_zjj data/pp_zjj_momenta.json \
  --process 'd d~ > z g g' \
  --parameters data/model_parameters.json \
  --set-parameter aS=0.1165
```

The JSON output includes the resolved tensor `shape`, flattened row-major
`values`, its explicit `resolved_sum`, and the optimized
`compatibility_total`, in addition to the selected physics-axis IDs.

Benchmark the selected process:

```console
python python/benchmark.py artifacts/pp_zjj \
  --process 'd d~ > z g g' \
  --momenta data/pp_zjj_momenta.json
```

`python/external_models.py` demonstrates explicit JSON and trusted-UFO
`ModelSource` construction:

```console
python python/external_models.py models/json/sm/sm.json models/ufo/sm
```

## Generated Python, C, Rust, C++, And Fortran

Every generated artifact contains one `API/` bundle. All drivers select a
process, accept JSON/direct model-parameter updates, evaluate resolved values,
sum them, and compare with the optimized total:

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > z g g' --set-parameter aS 0.117 0 --json
make -C artifacts/pp_zjj/API/c run \
  ARGS='--process "d d~ > z g g" --set-parameter aS 0.117 0 --json'
make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > z g g" --set-parameter aS 0.117 0 --precision 16 --json'
make -C artifacts/pp_zjj/API/cpp run \
  ARGS='--process p_p_to_z_j_j_4 --set-parameter aS 0.117 0 --json'
make -C artifacts/pp_zjj/API/fortran run \
  ARGS='--process p_p_to_z_j_j_4 --set-parameter aS 0.117 0 --json'
```

The generated Rust source includes the wheel-owned safe wrapper located by
`rusticol-config --rust-source` and is compiled directly with `rustc` plus
`rusticol-config --rustflags`; no Rust crate dependency is needed. The Makefile
also has an optional `run-script` target for separately installed
`rust-script`, using `rusticol-config --cargo-rustflags`. C, Rust, C++, and
Fortran support f64 (`--precision 16`) only. At f64, direct SymJIT,
target-compatible ASM/C++, and eager JIT artifacts run without a Symbolica
runtime. The Python driver also exposes precision-controlled Symbolica
evaluation when exact expressions are available.

## Hand-Written Native Examples

`native/runtime.cpp`, `native/runtime.f90`, and `native/Makefile` consume only
the SDK discovered from an installed wheel:

```console
make -C native
native/runtime_cpp artifacts/pp_zjj p_p_to_z_j_j_4 \
  data/model_parameters.json
native/runtime_fortran artifacts/pp_zjj p_p_to_z_j_j_4 \
  data/model_parameters.json
```

Both examples apply a direct `aS` override, evaluate the five-particle
validation point, and verify that resolved components reproduce the total.

Current publication gates are listed in
[`docs/user/release-status.md`](../docs/user/release-status.md).
