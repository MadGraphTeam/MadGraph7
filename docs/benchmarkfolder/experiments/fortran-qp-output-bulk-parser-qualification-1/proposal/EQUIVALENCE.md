# Proposed qualification contract — no execution yet

Admission is sequential: source review, bounded private parser build, closed
build review, parser execution, actual-amplitude parity, then descriptive timing.
A failed stage is retained and stops dependent stages. There is no automatic
retry, production patch, native-library rebuild or changed numerical policy.

1. **Compile/link and source scope.** Build original and candidate private
   MTIRUST modules with the same actual compiler/flags. Bind all source/object,
   archive/provider/PT_INTERP bytes before invoking tools/loaders. For public
   parser fixtures, use the same recording ABI implementation to return exact
   64-byte records; for physical tests use the unchanged eb3d native DSO and
   frozen QCD n3 numerical objects. Explicit private MTIRUST precedes its archive,
   with link-map/nm proof of one definition and no old archive module extraction.
   No Rust compilation. Originals and closed profile capsules remain read-only.

2. **Public decoder corpus.** Compare original scalar versus candidate public
   tensor_mp outputs and every observable: status, exact local error, all real
   and complex binary128 words (including signed zero), cache clear calls,
   post-failure cache statistics and following valid-call behavior. Cover all
   admitted coefficient counts for ranks0..6 and N1..6. Place the first bad
   record at the first, middle and last positions of each batch. Failed public
   calls must retain zero coefficients and the exact invalid-output error and
   cache invalidation; successful records must preserve all113 significand bits.

   Eligible corpus: exact +/-zero; 39-significant-digit normal values at and
   around tie/last-bit boundaries; alternating signs; canonical exponents0,
   +/-1, +/-4899, +/-4900; real and imaginary signed zero. Use binary128 word
   fixtures and at least 40-digit decimal references; never generate expected
   QP values through binary64. Include real wire records captured separately
   from the unchanged native provider, recording fast/scalar eligibility counts.

   Mandatory scalar fallback corpus: exponent +/-4901, min/max normal,
   subnormal/tiny/underflow/overflow boundaries; exponent padding/plus/upper-case
   variants; leading, embedded and trailing tabs; NUL at each meaningful
   position; non-ASCII; NaN/sNaN/Inf spellings; truncated records; extra tokens;
   comma, slash, repeated/null values, blank records and cross-record strings.
   Follow every malformed field with a distinct sentinel record to detect
   accidental token/record shifts. Full/UV/IR/none and finite=false combinations
   must take the explicit original path where the gate requires it.

   Some legacy list-directed slash/null cases can return success without
   assigning an intent(out) real. Their old value is undefined, so never claim
   a defined numeric result or use allocation coincidence as truth. Prove that
   these batches take the unchanged scalar path. A separate helper fixture may
   use identical seeded storage to inspect parser mechanics, explicitly labeled
   as such; public behavior remains governed by the unchanged scalar routine.

3. **IEEE and failures.** Retain actual pre/post rounding, all supported five
   exception flags and halting settings, and actual exit/signal. Test nearest,
   up/down/toward-zero and any additional supported mode, with initially clear
   flags and independently set sticky flags. Candidate must not set/clear
   environment settings. Only nearest/nonhalting enters bulk; other modes
   exercise fallback. Run each halting case in its own bounded process, comparing
   original/candidate terminal behavior. Underflow/overflow and nonfinite fields
   remain scalar. Force the candidate's unexpected bulk-failure branch in a
   separately labeled test-only decoder hook, retaining the untouched candidate
   as the scientific variant; verify no scalar reparse occurs, and the original
   public error/cache-clear/zero-coefficient path executes exactly once. A
   test-only forced branch is not proof that real eligible READ can fail or that
   it would match scalar conversion. Any observed parity difference blocks
   promotion and remains an open result.

4. **Actual amplitude parity.** Clone the exact closed QCD n3 source/objects,
   cards, native providers and saved100 momenta. Only private MTIRUST differs.
   Use fresh original/candidate workers with unchanged full100 learning and
   the original stability policy, then the same ordinary anchors1(DP219) and
   16(auto-rescued QP323), followed by the full100 observation if admitted.
   Compare all coefficient/accuracy words, statuses and actual FP observations;
   bind before/after sources/cards/model/points/runtime and clear all private
   filters initially. No fixed-QP substitute or shortened learning. Keep full
   versus pole/partial diagnostic tests separate and retain native cache
   counters/transport failure cleanup in a public adapter fixture.

5. **Timing only after parity.** Use uninstrumented original/candidate private
   full-amplitude executables, the same full100 learning/history, and a frozen
   common point cohort. Predeclare and retain alternating paired repetitions,
   target about1s per batch, actual durations, point counts, accepted DP/QP
   usage, cache settings and shared-host load. Include initialization separately
   and a fixed total administrative bound. Keep parser microbench and stage
   clocks separate from full-amplitude measurements; an apparent stage gain
   alone does not establish end-to-end benefit. No best-of-retry selection or
   performance acceptance claim on this shared host. Precise local acceptance
   remains a separate, unmodified handoff to Hua-Sheng.

Initial bounded build/runtime values will be inherited from the reviewed private
profile controller where suitable (nonexclusiveCPU12, shared8/9 mutex,16GiB AS,
512MiB file/core0, identity-checked cleanup reserve). Exact commands, corpus sizes
and per-stage caps must be fixed in a later source/controller seal before any
compiler/native execution; this proposal does not authorize them.
