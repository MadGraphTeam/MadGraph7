<!-- SPDX-License-Identifier: 0BSD -->

# Symbolica Licensing

pyAmpliCol imports Symbolica only when generation or model compilation first
needs it, then queries `symbolica.is_licensed()`. It does not infer validity
from the presence of `SYMBOLICA_LICENSE` alone.

## Restricted Mode

Without a valid license, pyAmpliCol suggests a license request once and
continues in Symbolica's restricted mode. Effective generation resources are
clamped to one process worker and one Symbolica core. Requested and effective
configuration remain separately recorded, including the reason for each
clamp. This policy limits generation; it does not serialize independent
Rusticol runtime handles evaluating a direct SymJIT f64 artifact.

## After Generation

The default JIT backend embeds a self-contained SymJIT application in the
schema-v3 artifact. Rusticol can load and evaluate that payload at f64
precision without importing Symbolica, reading `SYMBOLICA_LICENSE`, or applying
restricted-mode generation limits. This is the supported deployment path for
the Python, C++, and Fortran runtime APIs.

The separation is deliberately capability-based rather than inferred from
precision alone. Arbitrary-precision Python evaluation and artifacts generated
with the Symbolica ASM or C++ evaluator backends retain an explicit Symbolica
runtime requirement.

Suppress the suggestion and Symbolica startup banner with either interface:

```console
pyamplicol generate "d d~ > z g" artifacts/ddbar_zg \
  --no-symbolica-suggestion
```

```toml
[symbolica]
suggest_license = false
```

JSON CLI output also suppresses the banner so stdout remains machine-readable.

## Request A License

The interactive helpers collect the required fields, ask for confirmation,
submit through Symbolica's Python API, and retain neither personal data nor
the returned key:

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

Symbolica emails the key. Export it before generation:

```console
export SYMBOLICA_LICENSE='your-issued-key'
```

Trial and hobbyist eligibility, restricted-mode terms, and license durations
are controlled by Symbolica. Consult the
[official Symbolica installation and licensing guide](https://symbolica.io/docs/get_started.html)
before choosing a request type.

## Licensed Resource Sharing

With a valid license, `generation.workers = "auto"` and
`evaluator.optimization.cores = "auto"` share one affinity-aware CPU budget.
For example, several independent process builds may run concurrently while
each Symbolica evaluator receives a disjoint core budget. Explicit requests
are still clamped when their product would exceed the available budget, and
the adjustment is recorded in the resolved configuration.
