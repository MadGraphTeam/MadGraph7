<!-- SPDX-License-Identifier: 0BSD -->

# Packaged Examples

All TOML cards use schema version 1 and resolve paths relative to themselves.
List or copy the wheel-owned examples without referring to an installation or
source-checkout path:

```console
pyamplicol examples list
pyamplicol examples copy ./pyamplicol-examples
```

Run a copied card by passing it as the first argument:

```console
pyamplicol pyamplicol-examples/builtin_sm_lc.toml \
  --set generation.output=artifacts/ddbar_zg
```

Schema-v3 generation is integrated and emits one root multi-language `API/`
bundle. Evaluation and benchmark cards use the lazy Rusticol runtime adapter
shipped in a built wheel. Remaining publication gates are listed in
[`docs/user/release-status.md`](../docs/user/release-status.md).

## Run Cards

| File | Coverage |
| --- | --- |
| `builtin_sm_lc.toml` | Built-in SM, LC, JIT defaults |
| `builtin_sm_nlc.toml` | Built-in SM, NLC contracted color |
| `builtin_sm_full.toml` | Built-in SM, full contracted color |
| `process_set_mixed_multiplicity.toml` | Named process set with 2-to-2 and 2-to-3 requests |
| `external_ufo_sm.toml` | Trusted external SM UFO directory |
| `external_json_sm.toml` | External serialized SM JSON |
| `external_json_scalars.toml` | Scalar contact model and model parameters |
| `external_json_scalar_gravity.toml` | Spin-2 scalar-gravity model |
| `evaluate_total.toml` | Summed runtime evaluation from JSON momenta |
| `evaluate_resolved.toml` | Resolved runtime evaluation and selectors |
| `benchmark.toml` | Short evaluator benchmark |
| `all_options.toml` | Every current schema field, active and commented |

The external-model cards deliberately use editable `models/...` paths relative
to the copied cards. Populate them from the wheel-owned model resources with:

```console
cd pyamplicol-examples
python python/copy_packaged_models.py
pyamplicol external_json_sm.toml
```

Use `--force` only to merge over an existing model workspace. The helper uses
`importlib.resources`, so it works from an installed wheel and never assumes a
checkout layout. The public model API still expects a filesystem path rather
than a packaged model name.

## Python And Native APIs

`python/typed_generation.py` covers typed config resolution, process sets,
planning, built-in/external models, and transactional schema-v3 generation.
Use `--plan-only` when only the non-writing plan is wanted.

`python/runtime_evaluation.py` loads model parameters, computes summed and
resolved values, and verifies their agreement. `python/benchmark.py` uses the
typed benchmark service. `python/external_models.py` demonstrates explicit
UFO and JSON `ModelSource` construction.

`native/runtime.cpp`, `native/runtime.f90`, and `native/Makefile` consume only
the SDK discovered from an installed wheel by `rusticol-config`. Native APIs
support f64; Python exposes f64 and precision-controlled evaluation through the
public runtime contract.

Every generated artifact also includes equivalent drivers under `API/`. For a
single process:

```console
python artifacts/builtin_sm_lc/API/python/check_standalone.py --json
make -C artifacts/builtin_sm_lc/API/cpp run
make -C artifacts/builtin_sm_lc/API/fortran run
```
