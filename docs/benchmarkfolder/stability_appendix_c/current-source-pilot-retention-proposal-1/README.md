# Current-source QCD n3 raw stability evidence

This source captures and replays the one closed five-reducer study admitted by root: source `c071ec1c…`, preparation `9c1b45cd…`, dryrun `5bac8cc4…`, and native run session 29683, actual exit 0. It preserves every raw group/value record, warning, command/map receipt, card identity, software latch observation, source qualification, and all 100 intended points. It packages no executable, object, library, or reference repository.

The study is host forced-DP/card-selected backend internal consistency. Return 110 is unchecked and does not observe the actual reducer or its internal precision. Hardware IEEE flags were not observed. Source-qualified entry-reset semantics distinguish incoming software flags from failures during the current call; the exact legacy parser decisions remain alongside the new decisions. Original historical data and figures are unchanged.

The fixed population is five reducers (1/6/7/8/9), 100 saved cut-qualified physical points each, three probes per point, and five bootstrap calls per reducer: 1,500 primary plus 25 bootstrap public calls. All five workers completed with original source/runtime/card/input guards. The four established reducers have 100 available diagnostics each. TIRust has 95 available and five unavailable at points 25, 34, 45, 70, and 84, including 15 rejected probes and 27 failed active-group observations. Five subsequent healthy calls clear their incoming software flags at the qualified public entry; this does not qualify the failed calls. These are diagnostic outcomes, not independent accuracy or performance acceptance.

For any threshold, the tail is bounded by `known exceedances / 100` and `(known exceedances + unavailable) / 100`. The 1% empirical resolution is too coarse for rare-tail claims. Ratios to Ninja retain the existing paired-population and zero-denominator masks. Do not interpret unavailable outcomes as measured failures or zero error.

`capture.py` follows only fixed, explicitly closed evidence paths. Group journals use the unchanged bounded deterministic lossless gzip codec (mtime zero). The index binds physical compressed bytes and decompressed original hashes and sizes; `retention.json` binds and rechecks every original logical input. The portable verifier never follows documentary host paths. It verifies source seals, exact entry-reset qualification, all five raw parser results and both legacy/new eligibility annotations, point/card/mass/cut bindings, maps against historical runtime receipts, protected-guard aggregation, saved metrics, and the fixed five-row summary. Runtime maps document the closed observation; replay does not reopen host binaries.

From `/shared/external/madloop_tirust/madgraph7_madloop_tirust`, after root reviews this source seal:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-pilot-retention-proposal-1/capture.py docs/benchmarkfolder/stability_appendix_c/current-source-qcd3-raw-1 --expected-terminal-sha256 4e6cc378cf14b9f9bab387efb02c63307a3f1d6dc2a1cbff03f4905a4d022fac --root-reported-exit-code 0
python3 -B docs/benchmarkfolder/stability_appendix_c/current-source-qcd3-raw-1/verify.py
```

The pure API is `verify(root=None, *, read_bytes=None) -> {verified, index_sha256, summary, scope}`. A supplied byte-reader must expose its complete physical inventory as `reader.paths`; no filesystem access is needed in that mode. Verify a copied capsule identically in an isolated temporary directory before publication. Do not run a compiler, loader, amplitude, or numerical lock during capture or verification.
