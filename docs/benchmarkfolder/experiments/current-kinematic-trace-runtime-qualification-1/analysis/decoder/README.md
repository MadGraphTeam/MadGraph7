# Private source-only kinematic decoder proposal

This small decoder is specific to the pinned current `KT_SCHEMA_V2` emitter,
four propagators, CMS groups 3/4/8/12/24 and EW group 5, modes 1/4. It does not
compile, load libraries, evaluate integrals, modify thresholds/model inputs, or
read a native trace by default. No native trace has been analyzed in this source
proposal. Nothing here qualifies numerical accuracy or successful reductions.

`decode_trace.py` verifies derived index
`25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184`
and the selected family's CT/TIRust sources and routing/derivation metadata.
Analysis requires an explicit trace filename/hash and `--qp-word-order low-first`.
The caller must independently authenticate the GNU x86_64, ELF64 little-endian
build. Storage-size headers alone do not establish word order. This is recorded
as an assumption, not silently inferred.

Each finite binary64/binary128 word is decoded exactly as a significand times a
power of two, without passing through a Python float. Signed zeros and NaN/Inf
payload bits remain visible. Derived rational values use hex numerator/denominator
plus an explicitly approximate 36-digit decimal display. Nonfinite values stop
the affected rational analysis, while original decoded records remain present.

The source-bound layouts differ: `PL`/`SHIFTS` are component-fast;
`P_LOOP` is propagator-fast; invariants are Fortran column-major. QP scale records
contain a QP normalization followed by one **DP** OSTHRES word. QP masses widen
already-squared DP complex inputs. The comparison reports the captured values;
it does not infer unknown off-shell model flags or replace mass definitions.

For CMS, pair (2,4) is `p4-p2 = p1-p3-r`; for EW, pair (1,2) is `p3+r`.
Here `r` remains the signed conservation residual. Captured wavefunction signs
come from the frozen routing metadata. Exact rational algebra compares these
relations with the actual captured shifts and the native invariant. Differences
include real floating-point construction/summation effects; they are not an
independent truth or a claim that a reducer is wrong.

Original BEFORE/AFTER branch records are authoritative. Exact rational squared
threshold comparisons are labeled **not a replay of Fortran floating-point
arithmetic**; they cannot override actual decisions, especially near boundaries.
Missing residual/matrix/native-return boundaries stay unresolved. A native
status is recorded as its integer; no success is inferred from fixed 110/120
amplitude codes, and this decoder does not consume those codes.

From the MG repository root:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python -B .tirust-build/kinematic-provenance-analysis-1/decode_trace.py
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s .tirust-build/kinematic-provenance-analysis-1 -p test_decode.py -v
```

Only after root admits a closed trace and authenticates its build:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B .tirust-build/kinematic-provenance-analysis-1/decode_trace.py \
  --case cms --trace EXPLICIT_CLOSED_TRACE --trace-sha256 EXACT_SHA256 \
  --qp-word-order low-first --output NEW_ANALYSIS_JSON
```

The output path must be new. The trace hash is checked before analysis and bytes
are checked again afterward. The decoder never discovers or follows active runs.
