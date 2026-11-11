# CMS group 3: private one-pair causal replay proposal

This source-only experiment asks whether changing only the native symmetric invariant pair S(2,4)/S(4,2) changes the original installed TIRust call's status/error for CMS points20/21. It does not fix production, define a fixture exception or input allowlist, establish the cause of the complete amplitude failure, or make an accuracy/performance claim. Existing stock inputs and all failed evidence remain unchanged.

The closed stress analysis `66fab0dde170020c83840364845d4413f7130f1c09dab1105fd2e95a0f13611d` found selected native refusals after the original matrix logic projected a spacelike critical invariant to zero. This proposal narrows to actual group3, N=4/rank=2, source amplitude5. The target call requires15 tensor moments with four Laurent slots. Source identities and captured data are independently bound; no physical model/offshell assumption is introduced.

The ten prospective independent contexts/calls are:

| Captured CMS point | Modes | Variants | Calls |
|---|---|---|---:|
| 1 | 1 (DP), 4 (QP) | Original stock control | 2 |
| 20 | 1,4 | Original stock; one RAW24 intervention | 4 |
| 21 | 1,4 | Original stock; one RAW24 intervention | 4 |

Each intervention replaces only the real native S(2,4) and S(4,2) words by the corresponding **captured original RAW_DIFFSQ real words**. Both entries are zero in the stress stock inputs. No other invariant, shift, complex squared mass, scale or option changes. In the Fortran column-major payload the affected one-based INTEGER*8 positions are32/38 for DP and63/64/75/76 for QP. The six stock captures and four derivatives are saved in `inputs/`; every word is restored directly through INTEGER*8 `TRANSFER`.

`input-certificate.json` binds raw trace hashes/line numbers, actual point/serial/group/mode/rank/split-order identity, the signed routing and exact rational negative bounds. For the source route, pair(2,4) is p1−p3−r, with r retained as the first cumulative shift; it is not silently set to zero. The captured source shifts and native shifts agree bit-for-bit after the known array transpose. Captured RAW and the exact square of the captured shift-pair are both strictly negative. For both precisions:

| Point | Captured RAW approximately | Certified exact dyadic enclosure |
|---|---:|---|
| 1 control | −365723.659143 | −2^19 < RAW ≤ −2^18 |
| 20 | −2.5×10^-7 | −2^-21 < RAW ≤ −2^-22 |
| 21 | −2.5×10^-11 | −2^-35 < RAW ≤ −2^-36 |

Approximate values above are descriptive only. Replay files contain exact hexadecimal words; no Python float or decimal conversion forms the native input. Other projections in the matrix remain untouched, so this intervention does not certify physical consistency of the entire supplied invariant matrix.

`replay.f90` uses the original installed `MTIRUST` module and its generic **TIRUST_TENSOR_EVALUATE**. It links the existing `libtirust.a` (0c86349a…), `mtirust.mod` (d7838e75…) and original `libmadloop_tirust.so` (eb3d67fd…). It does not compile the module, patch the library or build Rust. QP input restoration uses the original113-bit Fortran kind; the installed archive alone supplies its64-byte ES64.39E4 transport to the256-bit implementation. In particular, captured QP masses and scale are not recomputed from higher-precision constants.

All optional evaluator settings are explicit: full finite calculation, pole_mode0, check_passfalse, cache_enabledfalse. Omitting cache_enabled would be wrong because the module defaults it to true. Initialization mirrors the original CMS plan-registration sequence (1,0),(2,0),(2,2),(3,2),(4,2), then cache budget268435456. One fresh process/context makes one generic tensor call. There is no amplitude bootstrap or numerator contraction in this standalone experiment. The existing Convention::MadLoop/component weights already apply, so the driver adds no sign, normalization, tensor permutation or multinomial factor.

The driver echoes every reconstructed input word, then pauses at READY before evaluation. A future thin owned controller must verify the live executable/provider maps before sending EVALUATE. It captures the original Fortran status/error immediately after the call, before any operation that could clear last_error; then it re-echoes the actual working shifts, invariants, masses and scale through integer TRANSFER and requires every post-call word to match the original input; its bounded8192-character error buffer reports whether capacity is filled. **Coefficient buffers are never inspected, including on success.** DONE pauses for post-call map verification; EXIT then finalizes the sole context. A status change establishes a response to this controlled input intervention only. Status0 is not an accuracy proof, and a stock status mismatch makes the reproduction incomplete.

`stage-plan.json` specifies two future gates. Build-only may compile/link the single driver against pinned installed artifacts and perform one actual declared-loader `--list` query; it runs no Fortran main. Root then reviews and freezes that standalone executable/provider subset. The separate run gate covers exactly10 fresh workers,240s of work plus60s cleanup reserve,4GiB per process, single nonexclusive CPU12, the existing shared mutex, strict output limits and no retry. It reuses the existing OwnedChild/deadline, linker binding, PT_INTERP/provider admission and live-map logic. The concrete run.py entrypoint and narrow interactive worker compose those existing helpers; their source must be reviewed before either native gate. No new generic supervisor is proposed.

After source review, the separately admitted commands are `run.py build --reviewed-source-index <SOURCE_INDEX> --execute`, then `run.py run --reviewed-source-index <SOURCE_INDEX> --reviewed-build-sha256 <CLOSED_BUILD_SHA> --execute`. Without `--execute`, run.py performs file-only inspection. The build closure includes protected postchecks and must pass before its hash can admit the run. Neither gate automatically continues.

The pure extractor first authenticates frozen indices, closed worker trace pins and source/library files. It reuses the sealed decoder without altering it. The21 pure/mock regressions comprise six exact-input tests, six controller-admission tests and nine protocol/worker/limiter tests. They cover exact symmetric offsets, QP low-word order with an unchanged half-word, shape/no-op rejection, strict rational negativity, hash-before-parse failure, all ten saved files, changed executable/provider rejection, the actual retained loader-helper stage namespace, cancellation, status1 failure retention, protocol framing, pre/post input mutations, call/context counts and bounded mocked cleanup. Independent source review checked the Fortran API/layout/handshake and reconstructed all ten input files from their six bound traces. Compilation remains unqualified until the separate build gate.

Recheck source-only evidence from the repository root:

```
source tools/tirust_env.sh
python3 -B .tirust-build/kinematic-projection-replay-proposal-1/extract_inputs.py --verify-only
python3 -B -m unittest discover -s .tirust-build/kinematic-projection-replay-proposal-1 -p 'test_*.py' -v
```

No compiler, loader, Fortran main or tensor evaluation has run for this proposal. Existing sealed packets, production/reference repositories and installed libraries remain unchanged.
