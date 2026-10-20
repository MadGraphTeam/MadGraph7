# Warmed NinjaRS QCD3 profile

The fresh production-port correctness diagnostic agrees with Ninja at all100
points, including two QP rescues. Its single warmed traversal took0.13189s/point
with NinjaRS and0.06578s/point with Ninja. These are diagnostic observations on
CPU5, not paired acceptance measurements. They motivate this profile without
changing points, stability thresholds, contributions or precision.

`profile.py --tag full-profile-v2 --cpu 5 --loops 3` clones that exact export,
changes only the driver's pre-timer stdin gate, and checks that every other
generated source and numerical library remains unchanged. Initialization and
the numerical output traversal finish before profiling starts. All100 original
point/order rows are byte-identical to the retained NinjaRS diagnostic; the three
warmed traversals return294 DP and six QP statuses. The profiler exits naturally
after its target, allowing the original data header to finish before analysis.

The7,363 retained samples attribute20.78% to the MP tensor numerator,14.45% to
MP reduction,8.80% to MP scalar masters and6.22% to MP geometry. Fortran QP and
DP numerator construction account for17.45% and21.85%. Categories are sampled
call-stack ownership using the existing classifier, with9.21% unmatched; they
are not exclusive stage timers. Transport is only0.45%. The profile directs
further research toward the MP work. It does not establish a new speedup or
qualify any final performance gate.

`analyze.py` recomputes the partition and retains a lossless gzip of the original
118MiB perf stream. All samples remain included. `full-profile-v1` preserves an
initial setup failure: its disposable destination parent did not exist, so no
compiler, evaluator or profiler was launched. The v2 script creates that parent;
all numerical inputs remain unchanged.
