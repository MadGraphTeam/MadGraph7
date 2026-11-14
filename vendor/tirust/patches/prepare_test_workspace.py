#!/usr/bin/env python3
"""Create a lean test workspace for the patched TIRust core, without host tools."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, help="staged, patched TIRust source directory")
parser.add_argument("output", type=Path, help="new test workspace directory")
parser.add_argument("--oneloop", type=Path, help="optional pinned local oneloopmaster checkout")
args = parser.parse_args()
source = args.source.resolve()
output = args.output.resolve()
core = source / "crates/iregi-rust"
if not (core / "src/direct/poles.rs").is_file():
    parser.error("source does not contain the pole-only patch")
if (output / "Cargo.toml").exists():
    parser.error("output already contains Cargo.toml; choose a new test directory")
q = lambda value: json.dumps(str(value))
revision = "0e2ad04c4f281e28bd29d5ad8b1bebf2764379b7"
manifest = f'''[package]
name = "madloop-pole-patch-validation"
version = "0.1.0"
edition = "2024"
[workspace]
[lib]
name = "iregi_rust"
path = {q(core / "src/lib.rs")}
[features]
default = []
native-f64-abc = ["iregi-masters/native-f64-abc"]
[dependencies]
iregi-masters = {{ path = {q(source / "crates/iregi-masters")} }}
madloop-pole-kernels = {{ path = {q(source / "madloop-pole-kernels")} }}
symbolica = {{ version = "3.0.1", default-features = false, features = ["integer-gmp", "float-mpfr", "native_code_generation", "bincode", "serde", "tracing_max_level_info"] }}
oneloop = {{ git = "https://github.com/alphal00p/oneloopmaster.git", rev = "2e7afabb5903814196d7ab39c695ed9cac777aa4", default-features = false, features = ["generated-evaluators", "rust-codegen"] }}
thiserror = "2"
serde = {{ version = "1", features = ["derive"] }}
[dev-dependencies]
serde_json = {{ version = "1", features = ["arbitrary_precision"] }}
sha2 = "=0.10.9"
[[test]]
name = "pole_only"
path = {q(core / "tests/pole_only.rs")}
[patch.crates-io]
'''
for name in ("symbolica", "numerica", "graphica"):
    manifest += f'{name} = {{ git = "https://github.com/symbolica-dev/symbolica.git", rev = "{revision}" }}\n'
if args.oneloop:
    manifest += f'\n[patch."https://github.com/alphal00p/oneloopmaster.git"]\noneloop = {{ path = {q(args.oneloop.resolve())} }}\n'
manifest += '\n[profile.dev]\nopt-level = 2\n'
output.mkdir(parents=True, exist_ok=True)
(output / "Cargo.toml").write_text(manifest)
print(f"Prepared {output / 'Cargo.toml'}")
print("Run cargo test --manifest-path <manifest> --test pole_only, then --lib pole_.")
