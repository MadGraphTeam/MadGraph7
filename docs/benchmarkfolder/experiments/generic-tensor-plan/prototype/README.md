# Exact dependency plan for the generic tensor numerator

This is an isolated prototype. It has not been installed, benchmarked, or used
for accepted physics results. Production `vendor/ninjars/src/tensor.rs` and the
frozen benchmark protocol remain unchanged.

The dense oracle is retained verbatim as
`bridge/src/tensor_dense_reference.rs`. Its original SHA256 is
`f5b153e179ac56e4a833f8d83b011a667e61cff1bcb567c2f1508a1d94013e55`.
The bridge is copied from production, with a distinct package/library name.
Only the copied tensor implementation and its tests use the prototype.

## Algorithm

1. Key a plan by the expansion shape, numerator rank, explicit mu powers, and
   the exact ordered list of requested Laurent powers. Coefficients and
   numerical kinematics are deliberately absent: each row still recomputes
   all required numerical values.
2. Seed the requested dense recurrence slots using the original contraction
   selection rules. For the Mu convolution, retain every original `j1` term.
3. Walk backward through the original recurrence, retaining every predecessor
   operation in its original factor order. Do not simplify a product with
   zero. Do not skip terms because a coefficient or factor is small.
4. Sort required nodes in original monomial/`j`/`a`/`l` order and assign compact
   scratch addresses. Slots that the original algorithm initializes to its
   prototype zero and never computes share a read-only zero. The constant
   monomial's unit slot shares the original one. Computed zeros remain
   separate computed nodes.
5. Overwrite each retained node once per row, in the original arithmetic
   order. Reuse compact scratch storage without clearing every computed slot.
   Within a row, identical `factor[component][term] * prototype_zero`
   subexpressions are computed at their first use through the original complex
   multiplication and then cloned. At most 28 such products are retained.
   Their full values, tracked precision and signs survive; every addition is
   still performed in its original order. This is exact common subexpression
   reuse, not replacement by a presumed zero. The small cache resets for each
   row. The original contraction loops, coefficient-zero test, factor
   construction, and Mu `g_powers` recurrence remain in place.
6. Use the untouched dense path for nonfinite coefficients, vectors,
   parameters or constructed factors, or when plan resource limits are
   exceeded. The prototype does not use algebraic zero pruning, so even a
   nonfinite intermediate on a retained path is evaluated in original order.

`tensor_plan.rs` contains integer indices only. It rejects more than 1,048,576
dense addresses, 65,536 retained recurrence nodes, 2,048 requested columns, or
rank above 64. A checked conservative budget of 4,194,304 root visits bounds
seeding even for repeated powers and long Mu convolutions. The per-numerator
FIFO retains at most eight plans and 32 MiB
of accounted plan allocations. These limits bound the added planning work;
they do not change the existing dense fallback's supported inputs or memory
requirements. The compact numeric buffer retains at most 65,538 entries.

## Precision audit

Pinned Numerica `Float * &Float` first reduces the left operand to the minimum
operand precision, including when its mathematical value is zero. Addition
has explicit zero shortcuts, while adding two zeros can retain different
precision/sign information. Complex multiplication evaluates four scalar
products followed by a subtraction and an addition. Consequently, replacing
a computed structural zero by the prototype zero, pruning a zero product,
or starting a sum from the first nonzero term is not justified.

The dependency plan does none of those algebraic transformations. It only
reuses exactly repeated products with the same immutable operands. An induction over
retained nodes gives the same operand values, precisions, signs and operation
order as the dense oracle. Every value read by a contraction is either a
retained node or the exact initial zero/one that the dense implementation
would read. Contraction order is unchanged. This is an argument to test,
the exact-oracle tests below check this argument for a finite sample of inputs.

Reviewed primary sources are the pinned Numerica `multiprecision.rs` and
`complex.rs` at Symbolica commit
`e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8`.
Their SHA256 values are respectively
`ed49b16eff4521a7fadd61bf623c4753a6d60522eb0b8dee9856814d892464d1`
and `97ab86b7514906abdd56802ca9c749633be96053fd81a90a7f178ebeee08790a`.

## Integer-only findings

The actual planner compiled with standalone `rustc` and inspected 176 layouts
without loading NinjaRS, Symbolica, or a numeric carrier. All retained parent
addresses precede their destination and preserve factor order.
`index-counts.csv` and `index-compile.log` retain the results.

For complete T2/T3 layouts at rank equal to the propagator count, backward
closure retains **all** dense recurrence nodes. It reduces scratch clearing,
not recurrence arithmetic. Exact repeated-product reuse saves the following
complex multiplications; additions remain unchanged:

| Rank | Shape | Original products | Reused products | Fraction |
|---|---|---:|---:|---:|
| 3 | T3 | 578 | 126 | 21.8% |
| 3 | T2 | 816 | 258 | 31.6% |
| 5 | T3 | 2,125 | 399 | 18.8% |
| 5 | T2 | 3,000 | 895 | 29.8% |
| 8 | T3 | 8,398 | 1,506 | 17.9% |
| 8 | T2 | 11,856 | 3,478 | 29.3% |

These are exact static operation counts, not measured runtime reductions.
Clone/lookup overhead, other reduction work and scalar integrals may limit or
erase a speedup.

## Evidence required before any production change

- Compile and run the integer-only `inspect_plan.rs` against the actual
  planner. Record node counts and memory sizes for the actual T2/T3 power
  layouts across ranks 0–8 and explicit mu blocks. Check all parent addresses
  precede their destinations and retain factor order.
- Run the copied bridge tests plus the dense-oracle comparison tests at
  113, 256 and 384 bits. Compare exact rational values, tracked precision,
  signed zero and nonfinite classification, not only decimal tolerances.
  Include all four expansion kinds, rank-zero tensors, explicit mu blocks,
  duplicate/reordered powers, sparse/dense coefficients, changed lanes,
  changed kinematics, rebinding, cancellation, and dense fallback.
- Add targeted captured expansion requests from actual MadLoop QP reductions,
  especially qcd3 rescue and qcd4 point81. Compare every expansion output
  before testing full amplitudes. Keep original and planned code in the same
  oracle executable where possible.
- Check complete forced-rescue/near-singular and analytic pole controls,
  plus 256/384-bit convergence, using isolated generated outputs and the same
  numerical thresholds. Preserve every status and rescue count.
- Only after numerical evidence, measure dense/planned recurrence cost and
  full warmed MadLoop cost with alternating order on one pinned CPU. Include
  cold plan construction separately. No speed claim follows from node counts.
- An independent review must verify the dependency closure, tracked precision,
  scratch reset, plan eviction and fallback paths. Promotion would require
  new library/executable provenance, refreshed representative parity, and
  all affected focused controls/core measurements.

The release prototype compiled on 2026-10-20. All three planner tests passed
in 2.95 seconds, and the full twelve-test bridge suite passed in 6.48 seconds.
The exact tensor oracle covers 36 rank/mu-block/precision configurations and
360 planned expansion rows, including two-row batches after rebinding. It
compares each component's exact stored rational, tracked precision and sign.
The inherited tests include binary128 transport, 256/384-bit complex-pentagon
convergence, MP rebinds and transactional failures. Logs and source hashes are
retained in `numerical-evidence.json`. Actual MadLoop request capture and
process comparison remain separate outstanding evidence; no production
promotion or runtime gate follows from these unit tests.
