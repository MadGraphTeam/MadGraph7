# Independent 21-pair QP confirmation

The fixed confirmation supports a small full-process benefit for the isolated
FFV derivative-row factorization: the factor/original median runtime ratio is
**0.96809491**, with paired 95% bootstrap interval **[0.95849113, 0.98213763]**.
The preregistered retention rule was an upper interval limit at or below 1.0.
Original and factored medians were 0.0685765739 and 0.0663886319 seconds per point,
respectively, a 3.19% reduction in this full QCD3 evaluation.

All 21 alternating pairs used every one of the same 100 points, original Ninja6,
the unchanged strict stability settings, and the byte-identical Fortran variants
from the earlier seven-pair experiment. All 16,800 compared printed binary64
coefficients agree exactly. Accuracy estimates and statuses also agree, and
every timed traversal reports 98 DP points and two QP rescues. This sampled
external equality does not turn the changed-association algorithm into a
general internal-bit identity claim.

The earlier seven-pair result remains separate and inconclusive. This result
uses only the preregistered independent 21 pairs: no observations were excluded,
no pooled estimate is used, and no further adaptive rerun is authorized without
a source change. The prototype remains isolated and `acceptance_eligible=false`;
it has not been promoted into production or the final benchmark.

CPU208 was selected before timing by the retained idle-core survey, after the
old CPU211 preflight failed. It has one online thread and is on socket1/NUMA
node6. Other task-owned computation was held except the corrected-original
reference collector, its preparation/build children and evaluations, all
confined to CPU207 on the same NUMA node. The 911.97-second window retains 184
host samples: CPU208 was 99.37% busy and CPU207 99.83% busy over compilation,
initialization and evaluations together. This was a coordinated local window
on a shared host, not exclusive machine ownership. The preflight, topology,
affinity-context description and unfiltered five-second samples remain in the
package.

From the repository root, after activating the recorded development environment:

```sh
python docs/benchmarkfolder/experiments/symmetric-factorization/confirmatory21/analyze_completed.py --run-id qcd3-factor-confirmatory21-v1
python docs/benchmarkfolder/experiments/symmetric-factorization/confirmatory21/retain_sources.py --run-id qcd3-factor-confirmatory21-v1
```

These checks are read-only: they rehash the raw values, verify complete rows and
timed statuses, recalculate the fixed paired bootstrap, and confirm the exact
compiled source/driver/recipe equivalence with the earlier sealed campaign.
The retained `plan.md`, controller, runner, command, full generated-source
manifests, build logs, per-traversal logs/values/times/statuses and numerical
summary describe the complete experiment.
