# Retained unlaunched n2 preparations

The two initial namespaces `historical-c91-qcd-n2-chain-1` and
`historical-c91-ew-n2-chain-1` were prepared for CPU2 but never launched. Their
premeasurement scheduling records at 22:19 UTC identify overlap with the
installer's known compiler affinity on CPUs2–5. Execution under these labels was
therefore canceled before native measurement, and separate CPU9 labels were
prepared. This was not selection between measured outcomes.

The frozen supplement `unlaunched-n2-cpu2-index.json` inventories all 22 original
files (11 per namespace): specifications, preparation and scheduling records,
frozen points/policy inputs and protocol sources. It verifies each specification's
input/protocol digest, the selected CPU, the unlaunched scheduling status and
the recorded absence of compilation or relinking. The exact file set contains
no raw numerical observations, timing output or result summary. It makes no
physical-correctness or performance claim and does not retrospectively
revalidate the ignored cloned executable bytes.

The CPU9 successor measurements are separately retained in
`historical-c91-qcd-n2-chain-cpu9-1` and
`historical-c91-ew-n2-chain-cpu9-1`; their diagnostics and severe shared-host
context remain bound by `n2-diagnostics-index.json` and `audit-n2-probes.json`.
They were performance-ineligible by design. This supplement neither changes
those observations nor turns the two initial preparations into measurements.

Replay this audit without modifying the original folders:

```sh
python3 docs/benchmarkfolder/experiments/chain-probes/audit-unlaunched-n2-cpu2.py
```
