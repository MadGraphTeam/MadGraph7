# Held generated-process controls after the NLO declaration fix

This plan is **not launched**. It uses the complete ten-point streams for
`control-gg-h`, `control-gg-hh`, `control-mixed` and `control-complex`, with all
four corrected candidate backends and all three helper-corrected original
backends. Every Born/Laurent/split-order row and status is retained.

Reuse already required final representative data rather than reevaluating it:

- Candidate ggH backends 1/6/7/8 from the fixed final ggH representative.
- Candidate mixed backend 8 from the fixed final mixed representative.
- Helper-corrected original ggH and mixed backends 1/6/7 from the unchanged,
  validated `representative-reference-controls-20261020` runtime binding.

The remaining work is eleven candidate targets (ggHH/complex × four backends
and mixed × three legacy backends) and six helper-corrected original targets
(ggHH/complex × three backends). `commands.sh` collects them in new namespaces
on CPU 4, with two persistent commands over every point. Candidate-only
collection uses `representative:true` and fresh parity because that is the
frozen adapter's supported mechanism; this does not promote those runs as
acceptance timing or replace the required final representative proof.
Reference collection reuses independently proven runtime coverage and does
not request a redundant fresh-process traversal.

Root must coordinate CPU/MP availability before launch. These are numerical
controls and diagnostic timings; they can share the host with other correctness
work but cannot overlap an exclusive acceptance timing window. Backend 8 owns
the single Symbolica slot while active. All candidate inputs come from
`generation-abi3-terminal-fixed`; source outputs and previous runs are unchanged.

`compare.py` validates the refreshed proof, all source/runtime bindings and
complete raw series, then compares all 21 corrected-cohort pairs at both
repetitions for each of the four controls. It requires the same point stream,
physical card, numerical settings and answer dimensions within each case.
It deliberately allows different diagnostic CPUs for reused runs. The
required tolerance remains
`abs(a-b) <= 1e-7 max(abs(a),abs(b)) + 1e-10 S`; no point/order is omitted.
It writes failures as evidence and exits nonzero if any comparison fails.

The ordinary strict forced-Ninja failures remain failures in their original
namespaces. The separate final forced-QP/near-singular representative proof
does not relabel those failed ordinary points. Rotations, direction reversal,
changed masses/scales, context rebinding and binary128/convergence controls
are separately assigned and remain on `../final-controls-checklist.md`.

After explicit coordination:

```sh
bash docs/benchmarkfolder/policy-v2/final-controls-plan-fixed/commands.sh \
  > .benchmark-work/logs/generated-controls-abi3-terminal-fixed-20261020.log 2>&1
```

Never overwrite an existing run or log; a retry needs new retained namespaces.
