# Private generic invariant geometry-window proposal 3

This is a source-only successor to private source2. Production and installed
libraries are unchanged. It does not calculate a corrected invariant or alter
an amplitude: the original metadata-only dispatch still calls legacy LOOP_N.

The proposal freshly joins source2's ordered momentum-window proof to generated
source bytes and source1's existing token/freshness plumbing. It emits separate
DP/QP geometry-window source gates while leaving whole-producer environment
support false. Runtime freshness, route, kind and admissible unchanged-token
checks remain conjunctive. The lifetime transformations are byte-exact copies
of source1, not newly qualified runtime behavior. See [DESIGN.md](DESIGN.md).

The final 24 pure tests pass in normal and optimized Python. Seven actual MG
MODEL/DHELAS/source-generation tests pass with subprocess/exec/ctypes calls
blocked during generation and writes restricted to this proposal. Five Born
variants yield ten supported source windows; two loop-induced variants retain
four unsupported windows because suffix call arities remain unresolved. All
fourteen still have `runtime_abi_qualified=false` and scientific dispatch false.
No compiler, reducer or scientific native driver ran.

The pinned interpreter used by the final tests is:

```text
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python3
```

From the repository root, with that interpreter on PATH:

```sh
python3 -B -m unittest discover -s .tirust-build/generic-invariant-provenance-proposal-3 -p test_geometry_window.py -v
python3 -B -O -m unittest discover -s .tirust-build/generic-invariant-provenance-proposal-3 -p test_geometry_window.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B .tirust-build/generic-invariant-provenance-proposal-3/test_generation.py -v
```

Generation is an historical source receipt, not portable numerical evidence.
It copied the frozen loop_sm model into this proposal. Keep the exact original
source1/source2 parents for these workspace reproduction commands. Do not infer
scientific qualification from old result JSON, matching source hashes, Python
tests, or the new narrow constants alone.

The next bounded qualification should first compile recording fixtures that
exercise the actual emitted caller and module ABI. Observe entry→same-kind
producer→use controls, slots and integer bounds, including rotations, FILTER_SO,
coefficient skip/reuse, MP_DONE and genuine-QP versus widened-DP paths. Fresh
qualified paths may expose source-bound bounds; every negative path must retain
false support. This remains independent of replacing the scientific dispatch.
Only after that can a private generated CMS amplitude integrate the previously
tested integer certificate, run normal DP→QP rescue with cache0/−1 controls, and
compare full Laurent coefficients with the retained matched-input/analytic pole
evidence. Independent finite truth, general stress behavior and Hua-Sheng's
precise local performance campaign remain open.

Initial test logs from the delayed environment-discovery command are retained
separately. That command finished successfully; its exact discovered Python was
not captured, so the final direct-interpreter receipts are the claimed tests.
No pending process remains from those source-only commands.
