#!/usr/bin/env python3
"""Prepare (and optionally build) a replay binary for one bridge variant.

The replay links this checkout's bridge crate (vendor/tirust) as a library
against a patched TIRust source tree: the same tree `install tirust` builds
(upstream revision + bundled pole kernels + poles-only patch). A variant may
override the bridge's double-double headroom, which is the only setting the
double-double gate study changes; nothing in vendor/tirust is modified.

  prepare_replay.py --workspace W --sources W/sources/tirust --variant h40 --dd-headroom 40 --build

Without --sources, --fetch fetches UPSTREAM_REVISION from GitHub into
W/sources/tirust and prepares it exactly like the installer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "tirust"
HEADROOM = re.compile(r"double_double_headroom_bits: (\d+)")
IGNORED = shutil.ignore_patterns("target", ".git", "*.o", "*.mod", "*.a", "*.so", "*.dylib", "build.log",
                                 "tirust-native-libs.txt", "tirust-link-before.txt", "smoke", ".cargo")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_headroom(source, headroom):
    """Return lib.rs text with the single embedder headroom replaced."""
    matches = HEADROOM.findall(source)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one double_double_headroom_bits setting, found {len(matches)}")
    if not 0 <= headroom <= 105:
        raise ValueError("double-double headroom must lie in [0, 105] bits")
    return HEADROOM.sub(f"double_double_headroom_bits: {headroom}", source), int(matches[0])


def cargo_manifest(bridge, sources, symbolica_rev):
    bridge, sources = Path(bridge).resolve(), Path(sources).resolve()
    git = "https://github.com/ValentinHirschi/symbolica.git"
    lines = [
        "[package]", 'name = "tirust-replay"', 'version = "0.1.0"', 'edition = "2024"', "publish = false", "",
        "[[bin]]", 'name = "replay"', f"path = {json.dumps(str(HERE / 'replay' / 'src' / 'main.rs'))}", "",
        "[workspace]", "",
        "[dependencies]",
        f"madloop-tirust = {{ path = {json.dumps(str(bridge))}, features = [\"poles-only\"] }}", "",
        '[patch."https://github.com/huashengshao/TIRust"]',
        f"iregi-rust = {{ path = {json.dumps(str(sources / 'crates' / 'iregi-rust'))} }}",
        f"iregi-masters = {{ path = {json.dumps(str(sources / 'crates' / 'iregi-masters'))} }}", "",
        "[patch.crates-io]",
    ]
    lines += [f'{name} = {{ git = "{git}", rev = "{symbolica_rev}" }}' for name in ("symbolica", "numerica", "graphica")]
    # Unlike the installed release build, skip LTO: Rust does not contract or
    # reassociate floating point, and the production replay is checked bit for bit.
    lines += ["", "[profile.replay]", 'inherits = "release"', "lto = false", "codegen-units = 16", "incremental = true", ""]
    return "\n".join(lines)


def symbolica_revision():
    text = (VENDOR / "Cargo.toml").read_text()
    revisions = set(re.findall(r'symbolica\.git", rev = "([0-9a-f]{40})"', text))
    if len(revisions) != 1:
        raise ValueError("cannot determine the pinned Symbolica revision")
    return revisions.pop()


def fetch_sources(workspace):
    sys.path.insert(0, str(ROOT))
    from madgraph.various import tirust_install as installer
    target = workspace / "sources" / "tirust"
    if target.exists():
        raise ValueError(f"{target} exists; pass it with --sources")
    upstream = workspace / "upstream"
    upstream.mkdir(parents=True)
    for command in (["git", "init", "--quiet"], ["git", "fetch", "--depth=1", installer.UPSTREAM_URL, installer.UPSTREAM_REVISION],
                    ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"]):
        subprocess.run(command, cwd=upstream, check=True)
    shutil.copytree(upstream, target, ignore=shutil.ignore_patterns(".git"))
    bridge = workspace / "pristine-bridge"
    shutil.copytree(VENDOR, bridge, ignore=IGNORED)
    patch, _ = installer._validate_pole_patch(bridge, target)
    installer._copy_source(bridge / "pole-kernels", target / "madloop-pole-kernels",
                           ["Cargo.toml", "src/lib.rs"] + [f"src/generated_{f}.rs" for f in ("a0", "b0", "c0", "d0", "constants")]
                           + ["src/generated_manifest.txt"])
    subprocess.run(["patch", "-p1", "--batch", "--forward", "--fuzz=0", "-i", str(patch)], cwd=target, check=True)
    return target


def prepare(workspace, sources, variant, headroom=None, build=False, jobs=4):
    workspace, sources = Path(workspace).resolve(), Path(sources).resolve()
    if not (sources / "crates" / "iregi-rust" / "Cargo.toml").is_file():
        raise ValueError(f"{sources} is not a TIRust source tree")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", variant):
        raise ValueError("variant names use letters, digits, '.', '_' and '-'")
    bridge = workspace / f"bridge-{variant}"
    if bridge.exists():
        shutil.rmtree(bridge)
    shutil.copytree(VENDOR, bridge, ignore=IGNORED)
    library = bridge / "src" / "lib.rs"
    text, production = set_headroom(library.read_text(), production_headroom(library.read_text()) if headroom is None else headroom)
    library.write_text(text)
    crate = workspace / f"replay-{variant}"
    crate.mkdir(parents=True, exist_ok=True)
    (crate / "Cargo.toml").write_text(cargo_manifest(bridge, sources, symbolica_revision()))
    if not (crate / "Cargo.lock").exists():
        shutil.copy2(VENDOR / "Cargo.lock", crate / "Cargo.lock")
    record = {"variant": variant, "dd_headroom": int(HEADROOM.search(text).group(1)), "production_dd_headroom": production,
              "bridge_lib_rs_sha256": sha256(library), "vendor_lib_rs_sha256": sha256(VENDOR / "src" / "lib.rs"),
              "replay_main_rs_sha256": sha256(HERE / "replay" / "src" / "main.rs"), "sources": str(sources),
              "patch_manifest": json.loads((VENDOR / "patches" / "manifest.json").read_text())["upstream_revision"]}
    binary = workspace / "target" / "replay" / "replay"
    if build:
        env = dict(os.environ, CARGO_TARGET_DIR=str(workspace / "target"), CARGO_BUILD_JOBS=str(jobs))
        env.pop("CARGO_BUILD_BUILD_DIR", None)
        subprocess.run(["cargo", "build", "--profile", "replay", "--bin", "replay"], cwd=crate, env=env, check=True)
        destination = workspace / f"replay-{variant}.bin"
        shutil.copy2(binary, destination)
        record["binary"] = str(destination)
        record["binary_sha256"] = sha256(destination)
        record["cargo_lock_sha256"] = sha256(crate / "Cargo.lock")
    (workspace / f"variant-{variant}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def production_headroom(source):
    matches = HEADROOM.findall(source)
    if len(matches) != 1:
        raise ValueError("expected exactly one double_double_headroom_bits setting")
    return int(matches[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sources", type=Path, help="patched TIRust tree (e.g. a snapshot of the installer stage)")
    source.add_argument("--fetch", action="store_true", help="fetch and patch the pinned upstream revision")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dd-headroom", type=int, help="override double_double_headroom_bits (default: production)")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    args.workspace.mkdir(parents=True, exist_ok=True)
    sources = fetch_sources(args.workspace.resolve()) if args.fetch else args.sources
    record = prepare(args.workspace, sources, args.variant, args.dd_headroom, args.build, args.jobs)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
