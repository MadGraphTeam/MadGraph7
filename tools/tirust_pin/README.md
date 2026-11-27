# Bumping MadGraph's TIRust pin

A pin bump touches:

- `vendor/tirust/Cargo.toml` and `Cargo.lock` (the git revision of `iregi-rust`);
- `madgraph/various/tirust_install.py` (`UPSTREAM_REVISION`);
- `vendor/tirust/patches/0001-direct-poles-only.patch`;
- `vendor/tirust/patches/manifest.json` (`upstream_revision`, `patch_sha256` and
  the `source_sha256` of every patched upstream file).

The installer refuses a build if any of these hashes is wrong. The scripts are
adapted from the procedure written on 2026-11-13 by the Claude session that
maintains TIRust upstream ("Speed up TIRust in MadLoop and its initialization").

## Steps

1. **Export the target revision twice.** Run
   `git -C <tirust clone> archive <REV> | tar -x -C <dir>` into two
   directories. In the second, apply the current patch with
   `patch -p1 --batch --forward --fuzz=0 -i vendor/tirust/patches/0001-direct-poles-only.patch`.
2. **Hand-merge any `*.rej` hunks.** The patch touches `direct/exec.rs`,
   `direct/exec/gram.rs` and nearby files. Earlier merges:
   - 6e4bf34: the per-use precision, stability-threshold and shared-subset
     calls moved next to the pooled lease.
   - 49dc836: a `POLES` branch before `match masters`, a `pole_masters` field,
     and lazily created pole masters for each plan variant.
3. **Regenerate the patch:**
   `tools/tirust_pin/regen_poles_patch.sh <tirust clone> <REV> <old patch> <patched tree> <new patch>`.
   It diffs the same file set, then checks that the new patch applies to a clean
   export with zero fuzz and offset and reproduces the patched tree.
4. **Rewrite the pin:**
   `python3 tools/tirust_pin/pinbump.py <MG root> <clean export> <REV> <new patch>`.
   It reads the old revision from `manifest.json`.
5. **Build and test.**
   - Run `install tirust --prefix=<fresh>` through the real CLI. This fetches,
     checks hashes, patches, builds and runs `smoke --require-poles`.
   - Run the bridge tests with `cargo test` in a staged bridge (16 tests).
   - Run `vendor/tirust/check_fixtures.py` for DP and `--qp`.
6. **Requalify.**
   - Register a new campaign `tirust-<pin8>-1` and rerun
     [`tools/tirust_requalify.sh`](../tirust_requalify.sh).
   - Replays of an earlier trace (`tools/tirust_capture`) must be bit-identical
     unless the TIRust change is meant to move values.

## Bridge settings

`vendor/tirust/src/lib.rs` sets
`EmbedderOptions { precision_tiers: true, tier_margin_bits: 0, double_double_headroom_bits: 26, .. }`.
Both of the last two options exist only from TIRust PR #35 (`9a781998`) onward,
so a pin to a TIRust revision without them does not compile. On TIRust's side
both are opt-in, with defaults 8 and 40.
