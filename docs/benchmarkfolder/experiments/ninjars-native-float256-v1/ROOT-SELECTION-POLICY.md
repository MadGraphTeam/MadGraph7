# Root selection policy declared before native sampling

This note records root's production-selection rule before any native screening
sample has run. It supplements the immutable `SCREENING-PLAN.md`; it does not
change that plan, its controllers, the three arms, input stream, seven fixed
triplets, two original reported comparisons or timing scope.

An arm demonstrates a gain over interpreted evaluation only when the upper
endpoint of its preregistered 95% ratio interval is strictly below 1. The
denominator-only and full-native arms each require that gate independently.

Selection follows these fixed rules:

1. If neither arm demonstrates a gain, select neither for production.
2. If denominator-only demonstrates a gain and full native does not, prefer
   denominator-only.
3. If both demonstrate a gain, prefer denominator-only unless full native also
   demonstrates a resolved incremental gain over denominator-only.
4. If only full native demonstrates a gain, it may be selected subject to the
   same numerical conditions below.

The additional incremental comparison is declared here before sampling. Its
point estimate is the median of the seven full-native elapsed times divided
by the median of the seven denominator-only elapsed times. Resample the same
seven triplet indices together 100,000 times using `random.Random(20261021)`;
for each draw compute that ratio of medians. Report the linearly interpolated
2.5% and 97.5% percentiles. A resolved incremental gain requires the upper
endpoint strictly below 1. No sample is excluded, replaced or added after
inspection, and no extra repetitions or adaptive sampling are authorized.

Any selected arm must retain all numerical requirements: successful finite
status, original input and scalar thresholds, genuine 256-bit reduction,
binary128 return values and zero-sign comparisons, explicit treatment of raw
precision/ASCII differences, 256/384 convergence, analytic poles, and retained
state/error controls. Full native's already disclosed raw value and tracked
precision differences must remain within the validated conditions; a gain
does not excuse a new unexplained numerical regression.

Selection is provisional until the final tests and complete generated-process
correctness, stability and performance campaign pass on the selected source.
This request screening excludes Fortran numerator/controller work and the MP
manager round trip. Its selection outcome establishes neither full MadLoop
performance nor the original-Ninja per-process 5% acceptance gate.

Allocation's complete-process study retains priority. Native sampling remains
held until allocation releases its quiet window and root explicitly grants
the native quiet/MP handoff.
