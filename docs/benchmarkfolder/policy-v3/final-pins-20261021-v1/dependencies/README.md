# Final dependency pins

The user identified both upstream main branches as final on 2026-10-21. Both
official repositories were cloned inside this repository and updated with
`git pull --ff-only origin main`. Sibling reference clones were left unchanged.
This capsule records source provenance and a static review. It contains no
new numerical, installation or performance result.

| Source | Exact main revision | Source tree |
|---|---|---|
| NinjaRS | `aa6a6307dd081dc3714f9accebc0df9e5561ae73` | `b0a6671ca67f026ba7c230e23328a64164d4f9d9` |
| oneloopmaster | `8ec256fc26cb361b20e10f59c19beb35717ef32c` | `f0f47b6c22c5d09c36e50e1fadf39cafb0b4ca12` |

NinjaRS now pins oneloopmaster at this revision. Both use Symbolica, numerica
and graphica from `alphal00p/symbolica` revision
`9cdc2949cdb153478c1eca0d40690dc0abfc616e`; the consuming bridge must supply all
three matching Cargo overrides. The gammaloop/feynkit packages advance from
`1e8063a763d3aa515749c33616ee59aa827705be` to
`d1904bf2bd5bf4f26969a73b415095e941a55610`. Source pins remain fixed for this
campaign; no watcher was started. The installed library and bridge Cargo lock
must be validated and bound separately before any new science claim.

NinjaRS advanced by 68 reachable commits from the preceding `0d0d7e7a` pin;
oneloopmaster advanced by 41 from `09b2c749`. The latter is the same `8ec256fc`
revision inspected before the pause. NinjaRS tracks a workspace Cargo lock;
oneloopmaster does not. Its workspace lock is not a replacement for the
bridge's separately resolved, committed lockfile.

The compatibility review found these material changes:

- The final generic Float master provider applies `regular_d0_labelling`
  after on-shell snapping. This fixes the previously missing relabelling for
  soft boxes, so a separate bridge adapter is unnecessary. The upstream test
  source covers all 24 propagator labellings and single/batch equivalence.
- B0 now uses the stable native bubble family in every carrier, including
  arbitrary precision. Other Float scalar families retain interpreted
  expressions. Input precision is retained.
- Float master results now fail when their inferred complex tracked
  precision is below `min(bits/2,max(53,bits/4))`, which is 64 at 256 working bits.
  The bridge must propagate this failure without publishing retained output.
- Generic bubble reference selection and tadpole bases are better conditioned.
  The fast engine's new `conditioned_cuts` option defaults to false; the
  existing bridge uses that default. Rank n+2 retains its conditioned engine.
- The strengthened generic cancellation test measures terms inside cuts as
  well as their sum. It requires `with_stability_test()`; a pin update alone
  does not activate it in the existing bridge. Enabling it requires an explicit
  integration change and validation.
- New fast numerator rank-lowering methods have defaults. The ordinary native
  tensor supports them; the local guarded wrapper does not forward that
  optimization. No obvious bridge API break was found statically. Compilation
  and numerical tests are still required.

The final oneloopmaster status explicitly leaves an adversarial tail open,
including some finite wrong values at exceptional near-singular or
vanishing-Gram points. Final NinjaRS rejects low-tracked-precision results, but
its own comments identify six exceptional rows whose precision estimates can
overstate accuracy. Successful status is not an accuracy certificate. The
oneloopmaster statement that NinjaRS checks only finiteness predates this final
NinjaRS pin. Conversely, the NinjaRS README's statement that no upstream
failure remains applies to its named 333-lane suite. Neither statement waives
this task's actual numerical or stability gates.

Upstream also documents a separate COLLIER 1.2.10 N=7/R=9 global-cache MRE in
`COLLIER_CACHE_BUG.md`, with an unconfirmed causal hypothesis and no external
issue. It was not executed by this review and must not be confused with our
COLLIER 1.2.9 qcd4 observations.

`pins.json` and `pull-and-ref-log.json` retain exact origins, clean-tree checks,
fast-forward ancestry and primary command output. `upstream-cargo/` contains
byte-identical manifests. `source-excerpts.json` gives pinned URLs, line numbers,
full-file hashes and selected exact excerpts, with upstream attribution. The
full private source review is sealed by manifest SHA256
`4f81cf0c99fabf72bb30e1396823fcae3995efcea679592dc503beb1504e6d36`;
the public capsule can be verified without that private directory.
