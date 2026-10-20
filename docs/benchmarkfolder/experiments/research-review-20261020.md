# Completed reduction and Fortran performance experiments

These experiments retain unsuccessful approaches and profiles as well as
numerical checks. None of the cache, color-contraction, or tensor-plan prototypes
listed here is promoted into production. They use their recorded, historical
generated processes; they do not certify the latest model-precision and QP
polynomial changes. Concurrent-host exploratory timings are not acceptance
measurements.

| Evidence directory | Finding | Decision |
|---|---|---|
| `master-cache` | Exact scalar reuse gives high hit rates, but the first prototype slows DP and gives only a small, unresolved MP benefit. | Do not promote. |
| `master-cache-v2` | Fixed DP keys and borrowed MP lookup preserve all 48 process rows. Scheduler drift prevents a useful timing conclusion. | Do not promote. |
| `generic-tensor-plan` | Dense-oracle tests and captured expansion requests agree exactly; three process blocks preserve all 48 rows. The measured median ratio is 1.213, with concurrent computation. | Do not promote. |
| `color-contraction` | A bounded contraction cache passes 2,560 DP/QP bit and cumulative IEEE-flag checks. Version 1 slows both processes. Version 2's ratios, 0.9989 and 0.9725, do not establish a benefit. | Do not promote. |
| `mp-qcd3-point44` | A warmed actual rescue profile assigns about 44.1% to Fortran QP numerator work, 18.3% to tensor expansion, 15.2% to generic reduction/bookkeeping, and 7.7% to scalar masters. | Prioritize numerator and reduction work. |
| `mp-qcd4-point81`, `normal-qcd4-point81-*`, `repeat-qcd4-point81-*` | Earlier forced and normal rescue profiles retain their exact drivers, settings, numerical outputs and repetition identities. | Historical cost diagnostics only. |
| `qcd3-point67-dp` | All 232 reductions succeed. The additional rescue arises from the aggregate direction check. Genuine 256-bit same-input replay shows that original Ninja's smaller direction difference is not uniformly smaller per-reduction error. | Investigate reduction accuracy; retain the existing stability test. |

The generic tensor README starts with its initial pre-measurement scope. Its
later “Generated-process exploratory comparison” and captured-request sections
record the completed experiment and negative timing outcome. The final
`process-evidence.json` is retained unchanged, including the exact Python
bytecode artifact it bound. That artifact is historical evidence, not an
executable dependency of the installed backend.

The color cache's IEEE-flag contract is narrower than an arbitrary low-level
replacement: it preserves cumulative flags within a coefficient batch, but
does not re-raise flags if external code clears them between cached calls.
The failed adversarial test is retained. This constraint and the unresolved
benefit both weigh against production use.

`research-review-20261020.json` inventories this completed collection. It was
created after the experiments and does not retroactively supply missing
generation provenance. Each existing manifest was independently rehashed
before packaging. Files and raw logs are retained even when they record a
failed build or license-limited concurrent-worker attempt. New point-67 scalar
isolation work is separate and is not covered by this frozen collection.
