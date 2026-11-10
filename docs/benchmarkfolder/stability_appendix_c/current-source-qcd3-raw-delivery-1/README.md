# Current-source QCD n3 stability comparison

This is a new, separately qualified current-source study of `u u~ > z g g [virt=QCD]` using `loop_sm`, the same saved 100 cut-qualified physical points and explicit reducers 1/6/7/8/9. Native session 29683 exited 0 after 1,500 primary and 25 bootstrap calls. It preserves historical evidence and does not reclassify or replace the older plots.

The standalone [PDF](../current-source-qcd3-raw-figure-1/current-source-qcd-n3-raw-dp-stability.pdf), [SVG](../current-source-qcd3-raw-figure-1/current-source-qcd-n3-raw-dp-stability.svg), and [PNG](../current-source-qcd3-raw-figure-1/current-source-qcd-n3-raw-dp-stability.png) use the frozen Appendix-C-style renderer and the retained data. All five tails, unavailable bounds and ratios over Ninja are reproduced in the adjacent statistical JSON. The full precision/interpretation caption is retained as `caption.txt`.

CutTools, Ninja, COLLIER and NinjaRS each have 100 available diagnostics. TIRust has 95; points 25, 34, 45, 70 and 84 remain unavailable, with 15 rejected probes and 27 failed active-group observations. Five subsequent healthy calls clear incoming software flags at the qualified public entry; both qualified and exact legacy eligibility annotations remain in the evidence. This does not qualify the failed calls.

These are host forced-DP/card-selected backend internal-consistency diagnostics, not independent true errors or performance acceptance. Return 110 does not identify the actual reducer or internal precision. Hardware IEEE state was not observed. All 100 points stay in the denominator; dotted curves are unavailable-outcome upper bounds, not measured failure fractions. At threshold 1e-3 the TIRust tail remains unknown within [0,0.05], and its ratio to Ninja is masked. With 100 points, empirical tail resolution is 1%.

The [data capsule](../current-source-qcd3-raw-1/index.json) contains all raw group/value records, warnings and failure evidence, exact cards and points, current/private source qualification, original runtime observations, and no numerical binaries. Its seal is `b2589cfc…`. Independent full raw replay passed; guarded original/isolated-copy/byte-only replays were identical ([proof](../current-source-qcd3-raw-verification-1/proof.json)). The figure seal is `cac6a230…`; original render session 39740 exited 0. The full original source/code pins remain in their separate source-only proposal packets.

From the MadGraph checkout, verify the closed data and figure without numerical execution:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-qcd3-raw-1/verify.py
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-qcd3-raw-delivery-1/verify_figure.py
```

To reproduce the figure into a fresh ignored directory after the data verifier passes:

```sh
source tools/tirust_env.sh
python3 -B - <<'PYFIGURE'
import importlib.util
from pathlib import Path
base = Path('docs/benchmarkfolder/stability_appendix_c')
spec = importlib.util.spec_from_file_location('current_figure', base/'current-source-pilot-figure-proposal-1/figures.py')
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)
output = Path('.tirust-build/local-current-source-qcd3-figure-1')
result = figure.render(output,
    point_metrics_file=base/'current-source-qcd3-raw-1/point-metrics.json',
    expected_point_metrics_sha256='f16806af788775fc42d5104092799d7d1d5879a5785c2b2f4760d5e4c3a6ddff',
    plotter_root=base/'plotter-source-6')
print(result['index_sha256'])
PYFIGURE
```

The output directory must not already exist. Its image/PDF/SVG byte hashes can depend on the plotting environment; statistical JSON must reproduce exactly. No native amplitude call, compiler, loader query, numerical lock, production edit or canonical report edit is needed for either verification or plotting.
