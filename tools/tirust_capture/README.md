# TIRust bridge call capture and replay

These tools record every call MadLoop makes into the TIRust bridge and replay
them against bridge builds that differ in one setting. They qualify the bridge's
double-double (DD) gate (`double_double_headroom_bits`) at tensor level, and
turn refused stress points into exact reproducers. They never modify
`vendor/tirust`, an installation or a campaign export.

The shim and replay are adapted from scratch tools that the Claude session
"Speed up TIRust in MadLoop and its initialization" wrote on 2026-11-13 for the
TIRust PR #35 work. The trace record format is unchanged, so its traces remain
readable.

| File | Purpose |
|---|---|
| `shim.c` | `LD_PRELOAD` shim that records `ml_tirust_*` plan, clear and DP/QP evaluation calls |
| `tirust_trace.py` | Streaming trace reader and summary (segments = phase-space points) |
| `capture_lane.py` | Reruns one campaign numerical lane (same `start_worker` call) with the shim |
| `prepare_replay.py` | Builds a replay binary for a bridge variant (optionally another DD headroom) |
| `replay/src/main.rs` | Replays a trace and compares every tensor bit by bit; per-call JSON lines |
| `dd_gate.py` | Joins the production and reference replays and decides the DD gate |

## Workflow

Run from the MadGraph root with `source tools/tirust_env.sh`.

```bash
cc -O2 -fPIC -shared tools/tirust_capture/shim.c -o $W/libtirust_capture.so -ldl
# 1. Capture: the campaign's candidate-9 lane, all warm-up calls with exact inputs.
python3 -B tools/tirust_capture/capture_lane.py --campaign $C --case qcd_n3_uux_zgg \
  --shim $W/libtirust_capture.so --out $W/capture/qcd_n3_uux_zgg --cpu $CPU
# 2. Replay binaries: production settings and reference DD headrooms.
#    --sources is the installer-patched TIRust tree; --fetch rebuilds it from the pin.
for v in 26 40 105; do
  python3 -B tools/tirust_capture/prepare_replay.py --workspace $W/replay --fetch \
    --variant h$v --dd-headroom $v --build
done
# 3. Replay every trace with each variant and decide the gate.
for v in h26 h40 h105; do
  $W/replay/replay-$v.bin $TRACE --calls-jsonl $W/$v.jsonl --report-json $W/$v.json
done
python3 -B tools/tirust_capture/dd_gate.py --production $W/h26.jsonl \
  --reference h40=$W/h40.jsonl --reference h105=$W/h105.jsonl --json $W/dd-gate.json
```

`--fetch` is needed only once per workspace. Later variants reuse
`$W/replay/sources/tirust`. The installer deletes its own staging tree, so either
fetch a tree or snapshot the stage during installation.

## What the comparison means

- **Production (h26).** The installed settings must reproduce every recorded
  status and bit. A failure means nondeterminism or a different build, and the
  study is not valid. Per-call route deltas (`[Direct, MethodChoice, Tier106,
  Tier256, Tier512, Refused, unreported, attempted106, attempted256,
  attempted512]` of the call's primary or check block) label the calls that
  went through DD.
- **h40.** This restores the previous 2^-66 gate. Lanes in the newly admitted
  [2^-80, 2^-66] window move to 256 bits, so the difference measures the
  headroom change itself.
- **h105.** No lane stays in DD (`tier_limit = 2^-1`). This is a 256-bit
  reference for every DD lane, including those the old gate already admitted.
- **Metric.** `|difference| / max|value|` over the components of the same
  tensor degree, across all four Laurent slots. An analytically vanishing slot
  is measured against the finite part, as MadLoop does for small poles. The
  per-tensor-scale figure is reported as `max_rel`.
- **Criterion.** Any DD-labelled call above 1e-12 is flagged. The gate is not
  qualified if any call exceeds 1e-9 or changes status.

Replay builds use `profile.replay` (release without LTO). Rust neither contracts
nor reassociates floating point, and the production replay checks bit-identity
explicitly.

## Record format

The format is documented at the top of `shim.c`. Kinds 1 to 4 are unchanged
from the original shim. Kind 5 is a QP call with its exact 64-byte ASCII
binary128 records (`TIRUST_TRACE_MP_INPUTS=1`).
