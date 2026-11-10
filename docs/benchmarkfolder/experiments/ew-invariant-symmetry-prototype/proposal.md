# Private invariant symmetry proposal

The retained Python fixture demonstrates one finite binary64 rounding mechanism that can trigger the bridge's strict symmetry rejection. It uses the EW export's card values `mt = 173.3` and `mw = 80.419`, zero widths, and synthetic shifts `(0,0,0,0)` and `(1,0,0,0)`. **These shifts and this mass pair are not a captured failing physical input.** This fixture does not establish the origin of all 531 EW native errors in the separate cause-observer experiment.

`fixture.py` reads the hash-bound retained card payloads and source references, computes separate binary64 scalar operations, and checks each rounded result against exact rational arithmetic. The trace completes the shared routine's entire matrix in I-major order before the adapter reconstructs invariants in J-major order. Default invocation only verifies the retained JSON:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/fixture.py
```

The real-component model follows explicit left association: `S_ij = (DIFFSQ - mi²) - mj²`, followed by `I_ij = (S_ij + mi²) + mj²`. Here `DIFFSQ` is exactly 1. The two massive on-shell predicates and the zero-invariant predicate are false in both directions at the card's `OSThres = 1e-8`; no override is removed or bypassed.

| Quantity | Binary64 hexadecimal word | Exact floating representation |
|---|---|---|
| `mt²` | `40dd5438f5c28f5d` | `0x1.d5438f5c28f5dp+14` |
| `mw²` | `40b943372f01754a` | `0x1.943372f01754ap+12` |
| upper `I(1,2)` | `3fefffffffffc000` | `0x1.fffffffffc000p-1` |
| lower `I(2,1)` | `3fefffffffff8000` | `0x1.fffffffff8000p-1` |

The upper-minus-lower difference is exactly `2^-39`. Both diagonal entries are positive zero. Mirroring the upper entry repairs this fixture's exact symmetry predicate while retaining the upper entry's rounding error relative to 1; this is not an accuracy improvement claim.

## Proposed private change

Keep both existing full-matrix computations, including all lower-triangle arithmetic and shared on-shell overrides. Immediately after the current reconstruction loop in each DP/QP TIRust diagnostic, insert:

```fortran
      IF(ALL(IEEE_IS_FINITE(INVARIANTS))) THEN
        DO J=2,NLOOPLINE
          DO I=1,J-1
            IF(INVARIANTS(J,I).NE.INVARIANTS(I,J)) THEN
              INVARIANTS(J,I)=INVARIANTS(I,J)
            ENDIF
          ENDDO
        ENDDO
      ENDIF
```

The whole-matrix finite guard prevents an original lower-only NaN or infinity from being overwritten and thereby hidden from the bridge's existing finite-input validation. Nonfinite matrices pass to that validation unchanged. The pairwise inequality guard avoids writing already numerically equal entries, including opposite signed zeros. This matters because the QP tensor cache key includes all raw invariant records, even though the numerical batch consumes only one triangle. Keep the bridge's strict symmetry guard unchanged; introduce no tolerance, averaging, recomputation from shifts, or modification to shared CT routines.

For Fortran column-major storage, Rust's `s[i + n*j]` with `i < j` is `INVARIANTS(I,J)` with `I < J`: the upper triangle. Both DP and QP batch builders use that entry and use its opposite only for the exact symmetry check. The proposed direction therefore preserves the existing selected numerical inputs. Diagonal entries, masses, shifts, scale, pole/finite mode and cache controls are unchanged. For finite matrices already accepted by the equality guard, the conditional copy performs no writes. This is a source-level argument, not compiled IEEE or cache-parity evidence.

## Source bindings and boundaries

`fixture.json` binds these repository sources and the archived card bytes with full SHA256 values:

- `madgraph/iolibs/template_files/loop_optimized/CT_interface.inc`: DP construction at lines 163–233, subtraction at 203, on-shell guards at 206–225; QP construction at 235–305, subtraction at 275, corresponding guards at 278–297.
- `madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc`: existing full reconstruction at 68–74 and 163–169.
- `vendor/tirust/src/lib.rs`: `dp_batch` at 459–485 and `mp_batch` at 514–549, including finite validation and exact symmetry comparison; `qp_cache_key` at 337 retains the full input records.
- Frozen generated adapter `dp-rescue-cause-observer/prepared/ew_n3_epem_ttxa/TIRUST_interface.f.original`, SHA256 `e60570ec599263807165f2f289add2076a4c5b71b9e09fe0de83306e9ab2f7ff`.
- Frozen preparation snapshot `dp-rescue-cause-observer/preparation/source-1/snapshot.json`, SHA256 `2aa5e87ead00c163d0f987dbef257654a8a22ec748f7d3f71be68aeaa64e7759`, binds the generated CT source SHA256 `ac240d6ffc2f7e1a15a72e61a325a53d0acf776f65b9329e4e9518cd85f3da0a`. Its generated construction is at lines 168–326. The fixture does not open the snapshot's recorded absolute paths.
- Retained `param_card.dat` raw SHA256 `891e4b89d522804e1f21c3f8b19b8cc2209b29cbfb7809189a5e7062c1363ea8`, mass tokens at lines 13 and 15; retained `MadLoopParams.dat` raw SHA256 `5300bd6d7f3c4553c3402ab7a044e8fbca3c646f451a5f586215933bfa113b85`, threshold at lines 219–220. The payloads are content-addressed gzip files in the cause observer's numerical qualification package.

Python real-component arithmetic does not reproduce compiled complex exponentiation/division, compiler reassociation, directed rounding, IEEE flags, halting modes, signed-zero behavior of every expression, or binary128 arithmetic. No compiler, Fortran driver, reducer, or physical amplitude was run for this fixture. In particular, executing the previous arithmetic at source level does not prove that compiled flag/trap behavior remains identical after adding `IEEE_IS_FINITE`, `ALL`, and comparisons.

Later qualification must separately check the actual generated routines with the installed compiler settings; finite symmetric inputs including signed zeros and QP cache keys; unequal finite entries; lower-only and upper-only nonfinite entries; complex widths; active massive/zero on-shell overrides; rounding and IEEE entry/exit state; and physical original/private comparisons with exact input provenance. Changes to DP rejection/rescue rates, accepted precision, coefficients, poles, or runtime require those measurements. The private DP/QP insertion and this binary64 witness alone qualify none of them.
