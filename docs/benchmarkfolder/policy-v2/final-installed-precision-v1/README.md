# Final published-library precision controls

The main prefix now contains the byte-exact successful private CLI installation
from numerical revision `d05b59f3277ce0d727cb2816d8b7ad45107e58c4`. The archive is
`800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42`.
`installation.json` is an explicitly derived record for `HEPTools/ninjars`.
It preserves every original private CLI field except the installed path and adds
`publication_provenance` binding that original record, the root publication
transition and the passed final Rust proof. The command, elapsed time and smoke
result still describe the original private CLI invocation; this record claims
**zero** additional CLI invocations, compiler calls or engine evaluations.
The exact original `install.mg5` and `install-cli.log` are copied alongside it.
Neither the private proof nor its hidden installation is modified.

`derive_installation.py` checks all ten published asset hashes, source/revision
agreement and original CLI evidence before writing this fresh derived record.
It ran on CPU 6 without a compiler or numerical executable. It refuses an
existing derived record. The prior final Rust proof remains the one successful
binary: 24 ordinary tests plus the separately executed fixture and full-history
convergence tests. No duplicate Rust test suite is proposed here.

## Exact pending 64-call campaign

The existing `tools/ninjars_precision_v2.py` prepares the seven unchanged
actual-MadLoop tensor fixtures (21 lanes total), with native DP, compensated DP
and genuine 256-bit public C ABI calls for every lane: **63 calls**. Call 64 is
the final Rust proof's complex-mass pentagon with numerator information beyond
double precision. It must reproduce the final-source 256-bit ASCII outputs and
pass the independent 384-bit result using the physical Laurent scale.
The C driver contains no reduction mathematics. It retains every raw status and
output plus actual loaded shared-library paths/hashes. MP basis-dependent R1 is
reported separately from the original fast-basis oracle; it cannot enlarge the
physical convergence tolerance.

`precision-spec.template.json` is deliberately held with null runtime fields.
After the fresh core exports exist, fill them from the **complete sealed actual
backend-8 reducer descriptor** in `final-core-generation-v1/runtime-inventory.json`
(or the identical descriptor from final measured bindings). Its archive and all
four module/config/link metadata hashes must describe this published installation.
Do not replace an old identity string or infer a descriptor from the archive
alone. The existing reader recomputes and checks the full descriptor.

With the real specification written as `precision-spec.json`, the exact commands
from the repository root are:

```sh
source .ninjars-build/dev-env.sh
# Root must release compiler resources; this prepares but does not evaluate.
taskset -c 190 python -B tools/ninjars_precision_v2.py prepare \
  docs/benchmarkfolder/policy-v2/final-installed-precision-v1/precision-spec.json
# Root must separately grant the serialized numerical slot.
taskset -c 190 python -B tools/ninjars_precision_v2.py execute \
  docs/benchmarkfolder/policy-v2/final-installed-precision-v1/run-v1/prepared.json \
  --cpu 190 --execute-numerical
python -B tools/ninjars_precision_v2.py validate \
  docs/benchmarkfolder/policy-v2/final-installed-precision-v1/run-v1/result.json
```

Preparation compiles one C adapter linked to the **actual main archive** and
checks its installed header/module/config/link assets against the source proof,
original actual-CLI record and final descriptor. It does not rebuild Rust.
Execution remains pending; no 64-call result or physical acceptance is claimed
by this readiness package. A failed or incomplete call remains in the result.
The successful result will feed the final supplement/report through the existing
precision adapter; its source tests and historical evidence stay unchanged.

## Representative protocol refresh dependencies

The final candidate runtime needs the same eight representative targets as the
last accepted terminal-fixed proof: ggH on backends 1/6/7/8; mixed QCD/QED split
orders on backend 8; forced rescue on backends 1/8; and the retained near-singular
Ninja stream on backend 6. Each target needs two complete persistent commands
and fresh-process parity over all ten fixed points. Ordinary forced-Ninja
failures remain historical failed observations; the distinct near-singular
stream supplies successful Ninja QP coverage without changing its inputs.

Fresh candidate control exports must use the published main archive and actual
final generator. Their complete reducer signatures must match the new core
runtime groups, even for legacy backends linked beside NinjaRS. Existing
original/corrected-original roles can be retained only after their full source,
raw parity and runtime bindings revalidate. The final role-aware proof requires
DP for all supported backends, QP for 1/6/8 in the candidate role and 1/6 in both
original roles, zero Born columns for loop-induced amplitudes, multiple split
orders and separate DP/QP coverage for **every** distinct runtime group.
The 64 C ABI calls do not replace these Fortran amplitude/protocol controls.
