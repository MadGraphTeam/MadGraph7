# Final native Float256 study result

Root retained interpreted production kernels under the rule declared before
sampling. Neither candidate demonstrated a request-scope gain, and no native
policy, feature or dependency-pin change is being promoted.

| Comparison | Ratio of medians | Paired bootstrap 95% interval | Demonstrated gain |
|---|---:|---:|---|
| Denominator-only / interpreted | 1.021698071 | [0.973591667, 1.045665774] | No |
| Full native / interpreted | 1.007070304 | [0.981059834, 1.040109539] | No |
| Full native / denominator-only | 0.985682887 | [0.957748429, 1.027689956] | No |

Each policy's median covers one full ordered stream of 1,392 private MP
requests: interpreted 19.853261741 seconds, denominator-only 20.284039222
seconds and full native 19.993630338 seconds. Input recovery, tensor work,
reduction, scalar masters and ASCII output formatting are inside the timer.
Fortran numerator/geometry construction, MadLoop stability control and the
public MP manager's channel round trip are outside this scope. These numbers
do not establish a full-process performance result.

The fixed seven adjacent triplets used the original declared order, inputs,
controls and sample count. Statistics retain all 21 measurements and use
100,000 paired triplet bootstrap draws with seed 20261021 and linear percentile
interpolation. No sample was omitted, replaced or followed by extra adaptive
repetitions. The incremental full/denominator analysis was declared separately
before the first sample and does not rewrite the original two comparisons.

CPU190 qualified with three consecutive three-second intervals at 0.332%, 0%
and 0% busy. The worker retained one CPU affinity and identity throughout;
the controller and host monitor used CPU2. The allowed legacy reference on
CPU207 and read-only upstream watcher were recorded in the root handoff.
Allocation's complete-process series finished before this screen started.

All three complete warmup streams and every one of the 29,232 measured request
outputs matched that policy's own bound original control. Input bytes were
unchanged; the worker and controller exited successfully. Warmup and untimed
lifecycle intervals remain separately reported. Full controller wall time was
484.840709717 seconds; it is not subtracted from or substituted for samples.

The extensive numerical findings, including raw precision differences,
binary128 return equality, 256/384 convergence, analytic and retained-state
controls, microscopic common history sensitivity and build recovery, remain
documented in `README.md` and the raw/independent evidence. Successful numerical
controls did not justify promoting an unmeasured or unresolved speedup.

The final branch's selected interpreted backend still needs its final combined
source tests and the full generated-process correctness, numerical-stability
and original-Ninja performance campaign. This negative study closes the native
optimization candidate; it does not close those project gates.
