# Source-only CMS caller metadata and bound factory

This packet implements the missing structural/error metadata and runtime
integer factory for the private CMS export, not a production change or a
numerical qualification. Neither Fortran nor C was compiled or executed here.
The original certificate kernel at general-proposal1665eff3 is unchanged.
There is no point, event-value, or selected-group eligibility list: all26
source routes and modes1/2/4/5 are emitted. Other exports need their own
generator-emitted metadata; these tables must not be reused by another process.

`derive_dag.py` hashes all269 original files from derived25c21147 before parsing.
It reuses unchanged `saved_source.py`/`routing.py` from the existing private
source consumer. It recognizes the actual generated ALOHA P storage, real input
kinds, external sign assignments, internal linear momentum expressions,
zero-seeded corner sums, check reversal, closure sum, and prefixes. Unsupported
wavefunction kinds/expressions make dependent routes ineligible. Changed input
hashes or an unrecognized fundamental adapter sequence abort derivation.

External W1:4 and MP_W1:4 contain the actual numerator P/MP_P after existing
repair, copied with source-proven signs. Their real-to-complex same-kind copies
are exact and their imaginary parts are zero. The bound reference starts at
these actual leaves, not at an imagined infinitely precise event and not before
the nonlinear phase-space repair. Widened DP-to-QP integrands are unsupported:
the caller must pass the actual MP producer's COMPUTE_INTEGRAND_IN_QP result.
The source verifies DOUBLE COMPLEX P for ALOHA, COMPLEX*32 P for MP_ALOHA, and
corresponding REAL*8/DOUBLE PRECISION or REAL*16 external P declarations.

W5:8 each use one actual two-term addition per component. Corner j uses the
source's zero seed followed by one subtraction per WE entry. The first chord
uses the actual zero-seeded closure accumulator, not literal zero. Other chords
use the actual zero-seeded prefix sequence. Modes2/5 with N>2 negate/reverse
corners; other cases preserve them. Negation/sign copies are exact. Each chord
stores signed coefficients, unsigned leaf occurrences, and an expanded source
add/sub count. Reused DAG nodes are counted repeatedly when expressions are
expanded: this overestimates executed work conservatively and retains error
when signed leaves cancel. Chord counts range through8; final-difference
metadata additionally records both expanded chord trees plus the subtraction.

The read-only source audit found no named environment setter among269 files.
That spelling audit is not proof of the behavior of an uninspected runtime
library. The private build must bind compiler flags, actual providers and the
entry/use environment protocol, with no unsafe reassociation or custom producer.

## Concrete Fortran interface

Compile `caller_bounds.f90` with its emitted `cms_bounds_data.inc` and the
separate tiny `environment_token.c` only after review. Module name:
`ml_private_cms_caller_bounds`. It exports `ik=selected_int_kind(33)` and
`zero_bound=-1000000`. Each bound e means error<=2**e; zero_bound means exact0.
The module contains only integer/logical operations. The C routine uses only
read-only STMXCSR/FNSTCW stores; unsupported compiler/architecture returns0.

```
route_matches(route,n,nw,we(nw),pairing(n)) -> logical

build_chord_bounds(route,mode,width,n,
  external_bits(2,0:3,4), environment_valid,
  chord_error(0:3,n), pair_class(n,n), have_chord(n))

finish_difference_bounds(n,width,shift_bits(0:3,n),
  chord_error(0:3,n),pair_class(n,n),have_chord(n),environment_valid,
  q_error(0:3,n,n),have_pair(n,n))
```

All bits are INTEGER(ik). Width64 words must be zero-extended, never signed
INTEGER*8 extensions. Width128 words are full binary128 TRANSFER patterns.
external_bits(1,mu,e) is the real part and (2,mu,e) the imaginary part of
actual W(e)%P or MP_W(e)%P. Either sign of imaginary zero is accepted; any
other imaginary value or any nonfinite real word rejects the whole contract.
No floating-point comparison is used, including for signaling NaN encodings.

Use `route_matches` on actual WE/PAIRING before building. Fresh but incorrect
route IDs are not valid metadata. Mode1/2 requires width64; mode4/5 requires
width128. `producer_kind_ok(mode,width,genuine_qp)` is exported to make the
widened-DP rescue fallback explicit. The caller must combine it with its
route and environment checks. The factory accepts the resulting boolean;
it does not infer a genuine numerator producer from CTMODE or a return code.

The amplitude caller captures leaves before the existing reducer call and
passes only local per-call arrays/IDs through private adapters. After the
adapter constructs its actual SHIFTS, finish_difference_bounds uses those
bits to complete subtraction bounds. The shared helper captures the actual
rounded q component bits and raw DIFFSQ under the original expression order
and passes q_error to the unchanged1665 kernel. There is no saved or COMMON
context, no inherited cache mask, and no public diagnostic read of a prior
amplitude ID. Unannotated diagnostics remain legacy/unknown.

Pair classes are0unknown,1protected shell/structural zero,2composite. Error
bounds can be valid for protected pairs, but that never authorizes a bypass.
The certificate still requires composite/no-mass-snap. The first narrow
candidate may produce direct invariants only inside the original massless-zero
branch when that overwrite is vetoed; other rows retain legacy reconstruction.

## Bound construction

The actual leaf reference has zero copy error under the source/kind contract.
For a chord with expanded add/sub count L and unsigned multiplicities m_e,
S=sum(m_e*abs(actual external leaf_e)), u=2^-p and eta=2^(emin-p), use

```
E_chord <= 2*L*u*S + 2*L*eta     (L*u <=1/2).
e_difference <= E_i+E_j+u*(abs(actual shift_i)+abs(actual shift_j))+eta.
```

The factory computes only power-of-two upper bounds by decoding exponents and
adding integer exponents. It uses actual component magnitudes separately;
there is no sqrt(s), MU_R or LSCALE cap. Large boosts enlarge the bounds.
Constants2 derive from the gamma_L bound and accumulated underflow errors,
not from tuning the CMS inputs. L is source-derived; the1e6 sanity limit is
far below the condition L*u<=1/2 for either supported format.

Unknown/nonlinear/mixed-kind momentum construction is not modeled by an
invented operation count. Such paths return no bound. Even when a mathematical
bound is available, the spacelike certificate can conservatively refuse.
Nonfinite shift inputs and malformed bounds also return no bound.

## Environment protocol

`read_kinematic_environment_token()` returns INTEGER(c_int64_t), bound to
`ml_private_kinematic_environment_token`. Invalid is0; valid tokens are
positive and have marker0x4B45000000000000. Bits16..31 contain MXCSR&0xFFC0;
low16 contain x87CW&0x1F3F. Volatile IEEE status flags are excluded. All relevant
rounding/FTZ/DAZ/precision/exception-mask controls are retained.

`environment_ok(token)` requires both units nearest, FTZ/DAZ clear, and the
qualified x87 extended precision setting. `environment_unchanged(token)` reads
the controls again and compares the token exactly. A widening shim must reject
negative/out-of-c_int64-range values before conversion; use environment_ok for
marker/control validation. The query never loads control words, clears flags,
changes rounding, masks traps, or restores an earlier state.

Capture a fresh token at the amplitude entry before the actual DP/MP momentum
producer, not just at LOOP_CT_CALLS after W5:8 exist. Pass that token explicitly
through the private call chain. Recheck at factory use and before certificate
evaluation. The source-bound intervening momentum path must have no mode
setter/custom callback. A current token alone does not prove the earlier W
rounding contract. If entry provenance, genuine-QP production, compiler/runtime
contract or a token check is absent, environment_valid=false and behavior
remains legacy. Source-only tests cannot establish runtime token correctness.

## Validation and remaining work

Fourteen pure tests pass. They replay269 source hashes and all104 records,
independently check source-operation counts/multiplicities, test reverse closure
and prefixes, execute reference factories across all routes/modes, and cover
wrong IDs/WE/pairing, boost scaling, zero/subnormal/NaN/Inf/sign-extension words,
unknown/mixed paths, environment controls and widened-DP fallback. Fortran/C
compilation, ABI word transport, actual environment and IEEE preservation are
still untested in this packet; the parent owns any later admission.

`tests-attempt1.log` retains one pure replay failure: Python tuples in a proof
field compared unequal with their JSON lists. Normalizing that proof field to
lists fixed replay without changing numerical counts, masks or emitted data.
`tests-attempt2.log` and the final14-test log are retained. No numerical worker
or native attempt occurred.

The separate requested closed-trace calculation will evaluate eligibility only
conditional on the new entry-environment/genuine-producer contract; old traces
contain neither token. EW mass-square policy and model arguments remain unchanged.
