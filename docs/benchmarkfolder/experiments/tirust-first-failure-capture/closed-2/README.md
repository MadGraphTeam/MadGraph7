# Closed TIRust first-failure capture and partial standalone replay

These diagnostics preserve the original raw-pilot plots and their unknown
bounds. They identify observed TIRust C-boundary status/error pairs; they do
not measure independent numerical truth or certify performance.

The qualified private amplitude capture watched Zgg point 25, groups 59/61
(N5, rank4), and ZZgg point 2, groups 408/422 (N6, rank5), with original,
reversed and signed-rotation probes. No flags were reset. Complete C inputs
were recorded only for the six Zgg primary requests. Every failed call omits
tensor output words; thread-local LAST_ERROR was copied immediately.

For all six Zgg primary requests, the boundary returned status1 with:

```
TIRust DP: Unstable("ill-conditioned Gram matrix for [0, 1, 2, 3]: conditioning 4.697307002016101e-4")
```

Reversed probes name subset `[0, 2, 3, 4]` and conditioning
`4.697307002016169e-4`. Entry→pre-call→status1→false-adapter→false-group stages
are retained. No selected request was rejected by the early Fortran guards or
the raw-tensor finite check.

ZZgg reached 288 watched bootstrap C-boundary calls, all status1 with subset
`[0, 2, 3, 4]`, conditioning `6.538097347992788e-4`. It hit the bounded capture
deadline during initialization. All six intended primary inputs remain missing;
none was reconstructed or replayed.

The original capture terminal remains **failed**. Zgg's numerical process
returned0, but its custom product guard omitted a normal624-byte private
`ML5_0_HelFilter.dat`. Additive closed-tree proof `230d1fc5…` applies the exact
existing source5 filter policy to both closed trees and passes. ZZgg's timeout
remains a timeout. The first replay admission, session1059, failed before native
work because two subsequently approved production template hashes changed;
its complete console output and confirmation remain in source2.

Root admitted a separate exact-two-hash inventory overlay. It changes neither
the standalone executable nor its native providers. Final replay session94497
terminated0: six fresh contexts, six registrations, twelve evaluations (one
fresh and one repeat per request), six destructions. Every evaluation returned
status1 and the exact error of its captured request. Creation, registration
and destruction returned0. Runtime binding, private guards and cleanup passed;
there were no surviving owned processes. Protocol completion is distinct from
the twelve numerical rejection statuses.

The selected Zgg boundary rejection therefore reproduces without the earlier
amplitude workspace history. This is not a complete inner-algorithm diagnosis:
LAST_ERROR describes the reported Gram gate, and method-choice fallbacks may
retain an earlier lane status. The printed conditioning is not the host
MadLoop accuracy target; its rank-dependent internal test is a separate
policy. No claim is made that the gate is physically necessary, that every
inner fallback failed for exactly that reason, or that independent true error
is large. Software-latch history and hardware IEEE exception state remain
distinct; this observer did not establish a hardware exception.

`summary.json` retains all12 intended slots, six captured and six missing.
Complete amplitude/group journals, warnings, stage events, immediate error
copies, input words, cards, frozen physical points, source transformations,
link map, every new build/loader/capture/replay command and sampled maps are
indexed. No executable, archive or shared-library binary is packaged.
Absolute original paths in receipts are historical identifiers. The portable
verifier resolves them only through `retention.json` and never opens them.

From the repository, verify the complete retained packet without native work:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/retention-source-1/verify.py docs/benchmarkfolder/experiments/tirust-first-failure-capture/closed-1
```

The same command works after relocating the packet, using its retained
`retention-source/verify.py`. Verification recomputes input shapes, event
lifecycles, status/error/output rules, command/stdout/map joins, all payload
hashes, complete source inventories and the unchanged source1 replay-summary
logic. It does not rerun an amplitude, load a library or relax a Gram guard.
