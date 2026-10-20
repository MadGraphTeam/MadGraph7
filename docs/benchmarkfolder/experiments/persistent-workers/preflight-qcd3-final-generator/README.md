# Preliminary strict QCD 2→3 measurement

This is exploratory evidence, not a promoted acceptance series. Three targets
were run on CPU 4 while independent export builds and diagnostic profiles were
finishing elsewhere on the shared host:

```sh
python tools/ninjars_benchmark_workers.py measure \
  --source both --backends 6,8 --cases qcd-3 --cpu 4 \
  --repetitions 7 --timeout 21600 \
  --run-id preflight-qcd3-final-generator
python docs/benchmarkfolder/experiments/persistent-workers/preflight-qcd3-final-generator/analyze.py
```

The candidate uses the expanded and compact Fortran traversal changes and the
GNU compiler-runtime linkage correction. Source, executable, library, card,
point and protocol identities are retained with the records. All targets use
the same physical inputs and `MLStabThres=1e-9`.

All 100 points, both physical output rows, and all four coefficients agree
within the specified tolerance. Original and optimized Ninja coefficients are
bit-for-bit identical. NinjaRS's worst pairwise difference uses 0.102 of the
allowed tolerance. Repeated rows and rescue counts are exact across all seven
commands.

Ninja rescues points 44 and 80; NinjaRS rescues 44, 67 and 80. NinjaRS's median
time is 2.269 times original Ninja, with a paired 95% interval [2.259, 2.725].
This clearly exposes outstanding performance work. The thresholds and rescue
behavior have not been relaxed. The original and optimized Ninja time series
also show a host-load change during the run; their median ratio is 0.931 but
its interval crosses the performance boundary. These data do not establish an
acceptance result for that comparison.

`summary.json` is regenerated from the hash-bound raw records by `analyze.py`.
The complete four-backend and optimized-legacy comparison remains required.

`additional-data/` also retains the earlier strict-setting original CutTools
and Collier calibrations, with the same point, parameter-card and common
numerical-setting identities. The analyzer includes them in the coefficient
comparison only; their timings are not part of this paired series. This adds
four-backend numerical evidence while the optimized CutTools/Collier exports
still require their own final measurements.
