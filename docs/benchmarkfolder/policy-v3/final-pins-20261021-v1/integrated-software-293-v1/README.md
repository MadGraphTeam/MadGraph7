# Final integrated software proof

Twelve current NinjaRS software modules ran from a fresh selected-source archive: **293 tests, 292 passed, one unchanged conditional skip, zero failures/errors**. The skipped check requires an actual generated QCD2 source tree, absent from this source-only image. Physical campaigns are separate evidence.

`overlays.json` binds the five final production files and current tests/helpers. `source-state-before-v4.json` binds all1,389 selected source files and symlinks. No ignored `.benchmark-work` or `.ninjars-build` parent existed before each module. All selected source files, symlinks and live overlays remained unchanged. Compiler, Cargo, shell and reducer subprocesses were blocked. Python ownership-control children and read-only Git/fontconfig queries were allowed.

`actual/` contains exact logs/results. Earlier preparation failures are losslessly retained under `retained-preparation-failures/`: missing `vendor.ply`, a blocked font lookup, missing `mg5decay`, and shared scratch destinations. These required source/runner closure corrections; no product code or test assertion was changed. Original source paths remain for attribution.

From repository root, with Python/NumPy/Matplotlib matching `closure.json`, reproduce using:

```sh
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/integrated-software-293-v1/reproduce.py --work .benchmark-work/integrated-software-reproduction-v1
```

The script reconstructs pinned Git source plus hash-checked repository overlays and runs the exact observed runner on CPU4. It fetches no dependency and compiles/evaluates no physics library. Retained historical source commits are required. Test logs and timing values may vary; assertions/counts/source identities are the reproducible contracts.
