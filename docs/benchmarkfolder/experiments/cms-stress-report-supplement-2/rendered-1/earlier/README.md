# Private source-only report proposal

No canonical or tracked report file is modified. No PDF/PNG is rendered and
no native amplitude, compiler, loader, timing run or network operation is
performed. `PLAN.md` states the proposed append-only delivery and limits.

`prepare.py` authenticates exactly these four closed public packet indexes:

- `cms-universal-qed-pole-audit-1`: `1565ac55fe6aeeefbd4f660455c422c4b954737a434d27138cbe8433d9534b77`
- `cms-universal-qed-single-pole-audit-1`: `99603d41d662e90f071b8052eb4303bcba8af451c5c96654febc276e620ad11e`
- `kinematic-invariant-amplitude-qualification-1`: `c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4`
- `kinematic-legacy-reference-adjudication-1`: `248bd61e987ac97c81e5397489eb4322ecfbf96cf85a42c84607aef2c013500e`

It verifies every outer payload and the decompressed identities of selected
records. It executes no retained calculation/controller or sibling verifier
and opens no other scientific packet, original generated tree or native
binary. Thus it authenticates already closed evidence; it does not newly
replay their broader source/build/dependency qualification.

The generated `data.json` contains eight collapsed eligible pole observations,
explicit failed legacy status rows, selected source pins and all qualification
limits. `supplement.typ` reuses the unchanged report style. It contains two
pole comparison tables and an optional future figure. `render_poles.py` is
unexecuted plotting source. `README.patch` is an unapplied documentation
proposal bound to its original README hash in `preserved-inputs.json`.

Pure verification, from the repository directory:

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/stress-report-supplement-proposal-1/prepare.py --verify-only
```

An explicit `--packets-root` supports a relocated directory containing only
the four admitted packets. To regenerate without overwriting the reviewed
data, supply `--output /tmp/new-stress-report-data.json`. The generator
rejects an existing output unless `--verify-only` is requested.

The following commands describe a **future reviewed rendering step** and
were not run for this proposal:

```sh
python3 -B .tirust-build/stress-report-supplement-proposal-1/render_poles.py --data .tirust-build/stress-report-supplement-proposal-1/data.json --output .tirust-build/stress-report-supplement-proposal-1/pole-comparison
typst compile .tirust-build/stress-report-supplement-proposal-1/supplement.typ /tmp/new-stress-supplement.pdf --input include-pole-plot=true
```

The canonical PDF's 35 pages and exact SHA-256
`2ecf83e01bd6367a7a39ecd7ea23dec40024d2ef5072dc7a583e26ebf9a37cc8`
remain the preserved source. Any subsequent merge must append pages and
verify original-page preservation. `doc/benchmarkfolder/benchmark.pdf`
and its Typst source remain independently preserved.
