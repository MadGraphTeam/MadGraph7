# Preliminary QCD three-body failure at default stability settings

The unchanged 100-point `u u~ > z g g [virt=QCD]` input stream exposed a mismatch
between default MadLoop stability acceptance and the required cross-backend
correctness tolerance. Original CutTools disagreed with original Ninja, original
Collier and candidate NinjaRS in the finite coefficient at points
6, 31, 34, 43, 44, 49, 59, 67, 80 and 100. Ninja, Collier and NinjaRS agreed with
one another. The summed and sole split-order columns duplicate these results,
giving 60 failing coefficient comparisons across three reducer pairs.

The original common settings were `MLStabThres=1e-3` and
`COLLIERRequiredAccuracy=1e-8`. CutTools's reported DP accuracy at the failing
points ranged from `3.9871e-8` to `1.73112e-5`; these estimates passed its default
stability threshold but did not ensure the required coefficient agreement.
`agreement.json` independently recomputes the failures from the retained raw
records using the original, unchanged correctness formula.

For the final benchmark, the common settings are tightened to
`MLStabThres=1e-9` and `COLLIERRequiredAccuracy=1e-10` for every original and
candidate backend. Phase-space points, seeds, model parameters, coefficient
tolerances and performance thresholds remain unchanged. Any additional
backend-local rescue is included in measured runtime and its status histogram.
The top-level configuration applies these settings; focused numerical controls
can explicitly override them.

This directory preserves the failed preliminary records, calibration records,
original configuration, exact point stream and available input/build metadata.
They are diagnostic evidence, excluded from final timing aggregation.
