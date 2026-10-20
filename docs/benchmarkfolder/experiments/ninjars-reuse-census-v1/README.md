# Retained-input reuse census

This completed offline census identifies repeated input bundles, not a demonstrated cache speedup. No engine, compiler, new capture, cache prototype, or production change was used. `PAUSED.md` preserves the earlier pause boundary. The resumed computation retains every record and has no nonfinite or failed record to exclude: both datasets contain 1,392 status-zero records.

| Retained scope | Requests | Exact geometry keys | Complete input keys | Most recent geometry per shape hits |
|---|---:|---:|---:|---:|
| Untagged historical MP stream | 1,392 | 540 | 624 | 16 |
| Tagged QP stage, points 44/80 | 464 | 353 | 416 | 8 |

The historical stream has 78 geometry groups with different numerator coefficients; the tagged QP stage has 57. A hypothetical global 128-entry LRU would hit 312/1,392 historical requests and 111/464 tagged QP requests. These simulations assume an initially empty cache for the stated stream. They do not establish process/context identity or an implementable geometry cache. The historical capture lacks point/direction/stage/context labels; none are inferred from order or shape. Cross-capture matches retain every possible semantic tag and explicitly cross different historical runtime/source identities.

Keys contain n, rank, ordered S/P/M/U components, backend policy, input precision, working precision and zero signs. Historical decimal strings are recovered by exact rational nearest-even rounding to 113 bits before keying; the tagged stream is decoded from binary64/binary128 words. MP boundary inputs are widened to 256 bits. This does not claim knowledge of the tracked precision of subsequently computed cut geometry or master arguments. The historical zero-canonicalized view is separately named; no geometry component in these captures has a negative zero, although numerator coefficients do.

The historical coefficients are dense: only 104 of 43,860 complex coefficients are zero, with 16 additional nonzero coefficients having just one nonzero real/imaginary component. All 42,468 nonconstant monomial parent-recurrence nodes are in the ancestor closure of nonzero coefficients. Thus removing whole unused monomial nodes offers no saving on this stream. This is a monomial-node result, not a count of Laurent subindices or a proof that any zero pruning is numerically safe. Existing expansion contractions already skip zero coefficients; ordinary value contractions do not. No new saving is credited for the existing skip.

## Machinery caches versus numerical results

The bound source snapshots support these distinctions:

- `sources/bridge-lib.rs`, `MpState::evaluate`: retains amplitude/tensor machinery per `(n,rank)`, rebinds coefficients, resets the amplitude batch, and evaluates fresh inputs. Numerator-dependent residues and complete amplitudes cannot be keyed by geometry alone.
- `sources/masters.rs`, `SymbolicaMasters::prepare` and `stage`: maps retain compiled evaluators by family and, for native tensors, working precision. Each stage validates arguments, applies the on-shell policy, and evaluates every valid row. They do not memoize numerical master values. `tensor_row_bits` depends on tracked precision and input magnitudes. The bounded evaluator map is not a result cache.
- `sources/cuts.rs` and `high_cuts.rs`: kernels retain evaluator code and reusable input/output arrays; evaluation still runs for each request.
- `sources/oneloopmaster-masters.rs` and `oneloopmaster-native.rs`: evaluator caches and constants are reused, while argument rows are evaluated anew.

No master-request stream was captured. Therefore this census cannot count duplicate individual master calls or quantify the benefit of a master-result cache. Top-level geometry equality is an opportunity to investigate, not evidence that every derived master request is identical.

Any later cache experiment would need a separate authorization and full correctness controls. A conservative numerical master key includes family, all ordered invariants/masses/scales, value **and tracked precision and zero sign**, working precision, effective row precision, on-shell threshold/snapping policy, and native/interpreted backend identity. A cut-geometry key additionally needs partition/request kind, supplied S policy, sampling scale, chart/tolerance choices and all relevant momenta. Validation must precede lookup; statuses, finite checks, error recovery and transactional output semantics must survive. Cached values must preserve represented values/precision/sign rather than round through binary64 or rational reconstruction. Per-context bounded storage and lifecycle cleanup are required. Any config/precision change must create a different key or invalidate the cache. These are source constraints, not an implementation proposal for this final round.

## Relation to the tensor-recursion paper

[Hirschi and Peraro, sections 4.2 and 5.1](https://arxiv.org/pdf/1604.01363), explain reuse of lower-rank recursive tensors and shared projectors. The existing generic tensor already constructs each monomial from a lower-rank parent. The present support census finds no unused whole nodes in the observed MP requests. It does not rule out cheaper arithmetic/allocation within those nodes, finer Laurent-state reuse, or a different validated algorithm.

## Reproduce and verify

From the repository root, with the established Python environment:

```sh
python -B docs/benchmarkfolder/experiments/ninjars-reuse-census-v1/verify.py
```

This read-only verifier checks retained source hashes and recomputes the exact census against both sealed raw inputs. The script's focused controls check ties-to-even, alternate decimal spellings, signed zero, precision/rank separation, a hand-constructed ancestor closure and an explicit LRU sequence. `result.json` retains per-record input hashes/support counts and every ambiguous cross-capture match. `provenance.json` binds 12 source/build snapshots and their upstream revisions. The separate reviewer inspected key construction and parent closure without running an engine.

The first verification log retains a serialization-only failure: the recomputed candidate-tag tuples were compared with their JSON list representation. The script now constructs lists explicitly; the existing result bytes/counts were not changed. `verification-v2.stdout` records the successful replay. No numerical capture was repeated.
