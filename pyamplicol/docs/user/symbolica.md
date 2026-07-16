<!-- SPDX-License-Identifier: 0BSD -->

# Symbolica Licensing

pyAmpliCol imports Symbolica only when model compilation or generation first
needs it, then asks `symbolica.is_licensed()`. The presence of a
`SYMBOLICA_LICENSE` environment variable alone is not treated as proof that the
license is valid.

## Restricted Generation

Without a valid license, pyAmpliCol suggests a request command once. For users
whose work is eligible under Symbolica's current terms, generation can continue
in restricted mode. Symbolica limits that mode to non-commercial work, one
instance, and one core per device; commercial work requires the professional
license path. pyAmpliCol clamps effective generation resources to one process
worker and one Symbolica core, but that technical clamp does not grant
eligibility or replace the upstream terms. Requested and effective
configuration remain separately recorded with the reason for each adjustment.

Suppress the reminder and Symbolica startup banner with either interface:

```console
pyamplicol external_json_sm.toml --no-symbolica-suggestion
```

```toml
[symbolica]
suggest_license = false
```

JSON CLI output also suppresses the banner so stdout remains machine-readable.

## Symbolica-Independent SymJIT f64 Runtime

The default JIT backend embeds a direct SymJIT application in the schema-v3
artifact. Rusticol loads and evaluates that payload at f64 precision without
importing Symbolica, reading `SYMBOLICA_LICENSE`, or applying Symbolica's
generation-time resource clamp. SymJIT is a separate MIT-licensed runtime
dependency. This runtime capability does not change the terms governing
Symbolica use during model compilation or process generation.

This is the deployment path for the Python, Rust, C++, and Fortran APIs. Rust,
C++, and Fortran are f64-only. Independent runtime handles can execute
concurrently even when the artifact was generated in restricted mode; one
mutable handle itself must not be called concurrently.

The distinction is capability-based. Python precision requests other than 16
load retained Symbolica evaluator state. Artifacts generated with the ASM or
C++ evaluator backends also retain an explicit Symbolica runtime requirement
and are rejected by the lightweight native SDK.

## Request A License

The interactive helpers collect the required fields, ask for confirmation, and
submit through Symbolica's Python API:

```console
pyamplicol request-symbolica-trial-license
pyamplicol request-symbolica-hobbyist-license
```

Complete noninteractive requests require every field and `--yes`:

```console
pyamplicol request-symbolica-trial-license \
  --name "Ada Lovelace" \
  --email ada@example.org \
  --organization "Example Institute" \
  --yes

pyamplicol request-symbolica-hobbyist-license \
  --name "Ada Lovelace" \
  --email ada@example.org \
  --yes
```

pyAmpliCol does not retain the submitted identity fields or print the returned
key. Symbolica emails the issued key; export it before generation:

```console
export SYMBOLICA_LICENSE='your-issued-key'
```

Eligibility, restricted-mode terms, and license duration are controlled by
Symbolica, not pyAmpliCol. Consult the
[official Symbolica installation and licensing guide](https://symbolica.io/docs/get_started.html)
before choosing a request type.

## Licensed Resource Sharing

With a valid license, `generation.workers = "auto"` and
`evaluator.optimization.cores = "auto"` share one affinity-aware CPU budget.
Concurrent process builds receive disjoint evaluator budgets. Explicit
requests are clamped when their product exceeds the available budget, and the
adjustment is included in the resolved configuration.
