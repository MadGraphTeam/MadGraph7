# Historical chain diagnostics

`tools/tirust_benchmark_chain_probe.py` checks whether the original TIRust-first
runtime difference also appears in the saved QCD and electroweak 2→2 processes.
These are optimization diagnostics, excluded from formal acceptance regardless
of the measured ratio. They preserve the qualified original 2→1 runtime failure.

Preparation copies the existing generated executable, numerical objects, and
sources into a private ignored directory. It verifies their original build
provenance, retains the exact historical worker template, and uses the original
immutable library directory. It performs no compilation or amplitude evaluation.
Numerical cards and helicity filters are private. The points, physical card,
automatic stability policy, compiler flags, and common native dependencies must
match between baseline and candidate.

Run from the repository after sourcing `tools/tirust_env.sh`. Choose an allowed
CPU and a fresh label; every label is append-only. For example:

```sh
python3 tools/tirust_benchmark_chain_probe.py prepare \
  --case qcd_n2_gg_ttx --label historical-qcd-n2-chain-1 --cpu 9
python3 tools/tirust_benchmark_chain_probe.py run \
  --label historical-qcd-n2-chain-1
```

The separate `run` command acquires the shared numerical lock. Its defaults
preregister seven alternating pairs, two-second calibration targets, a
120-second protocol-wait limit, and a cooperative 600-second elapsed budget after
lock acquisition. The budget is checked before worker startup and each batch;
preflight checks and bounded worker shutdown can extend it. It is not a hard
total process-lifetime cap. Elapsed time through shutdown is recorded separately.
The selected CPU is descriptive; these probes do not require
an otherwise idle shared host. Host-load counters, every calibration and timed
batch, numerical comparisons, accepted backend/precision histograms, and
post-run input hashes are retained in `experiments/chain-probes/<label>/`.

The geometric mean paired ratio and its interval are descriptive. Every result
has `performance_acceptance_eligible: false`; the tool does not publish a runtime
pass/fail decision or modify the campaign summary. Backend histograms count the
final accepted amplitude result, not every internal reduction attempt. The
initialization interval covers the first amplitude and saved-stream warmup,
excluding process startup and `SETPARA`.

Historical template overrides are restricted to explicit diagnostics and are
rejected by the formal retained-result validator. Reproduction therefore needs
the retained historical protocol, exported build, and original native library
bytes. A new optimized candidate must use its registered campaign and regenerated
numerical sources rather than relabeling these historical probes.
