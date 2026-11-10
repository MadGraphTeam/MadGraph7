# Precision-counter wording correction

The frozen numerical package README calls the QCD count “454 higher-precision attempts”. That wording is too strong. The measured fields are **454 method-choice/expansion rows, 320 successes and 134 refusals**, excluding bootstrap. They do not certify arithmetic precision. The historical README, index, raw journals, analyses and measured counts remain unchanged.

The [source audit of the pinned embedded options](../../embedded-dp-rescue-design/README.md) establishes that method choice was already enabled and retries at the same 53-bit precision; partial rescue and precision tiers were disabled. Neither a `Direct` route nor the aggregate `master_rows_all_precisions` field proves exclusive binary64 work. Genuine 113-bit-input QP rescue is a separate public path. Future reports must use “method-choice/expansion rows” for these counters and must not reproduce the stronger phrase.

`observations.json` binds the historical package, exact supplemental result, and source audit. It retains every exact ordinary-QCD upstream diagnostic string and count; in particular, `a priori method choice: accepted (unchecked)` occurs 320 times. The immediate native Gram error can retain the initial direct refusal while later method-choice diagnostics describe small-Gram nonconvergence or alternative-PV cancellation. These distinctions preserve the observed retry history without claiming an internal precision or a unique cause.

This is an additive source/data interpretation correction. It runs no compiler, reducer, amplitude or timing job and does not alter the numerical or acceptance result.
