#!/usr/bin/env python3
"""Link isolated bridge timers against the exact cached release dependencies.

Run only outside benchmark windows. This does not invoke Cargo or mutate the
production source, installed prefix, baseline, or generated benchmark builds.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[5]
BUILD = ROOT / '.tirust-build/n1-profile/bridge'
assert Path.cwd().resolve() == ROOT
DEPS = ROOT / '.tirust-build/cargo-build/release/deps'
NATIVE = ROOT / '.tirust-build/cargo-build/release/build/gmp-mpfr-sys-e3a3f2f14732793a/out/lib'
inputs = {
    DEPS / 'libiregi_rust-d256505083284e99.rlib': '8b465be336966fb74b7d00d2daa81b47dee54e72df2024bf07c0463fc765dde0',
    DEPS / 'libsymbolica-bfa2826cb07859dd.rlib': '372431b6cc8eae82e338b234883f75caac7e11245937c77cebfecbdae072c192',
    ROOT / 'HEPTools/tirust/lib/libmadloop_tirust.so': '49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


for path, expected in inputs.items():
    assert digest(path) == expected, f'Cached production identity changed: {path}'
command = [os.environ.get('RUSTC', 'rustc'), '--crate-name', 'madloop_tirust', '--edition=2024',
    str(BUILD / 'src/lib.rs'), '--crate-type', 'cdylib', '--emit=dep-info,link',
    '-C', 'opt-level=3', '-C', 'lto=thin', '-C', 'codegen-units=1',
    '--cfg', 'feature="default"', '--cfg', 'feature="poles-only"',
    '--check-cfg', 'cfg(feature, values("default", "poles-only"))',
    '--check-cfg', 'cfg(test)',
    '--out-dir', str(BUILD), '-L', 'dependency=' + str(DEPS),
    '--extern', 'iregi_rust=' + str(DEPS / 'libiregi_rust-d256505083284e99.rlib'),
    '--extern', 'symbolica=' + str(DEPS / 'libsymbolica-bfa2826cb07859dd.rlib'),
    '-L', 'native=' + str(NATIVE), '-C', 'link-arg=-Wl,-soname,libmadloop_tirust.so',
    '-C', 'link-arg=-Wl,--exclude-libs,ALL']
with (BUILD / 'build.log').open('w') as log:
    log.write(json.dumps(command) + '\n'); log.flush()
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(f'Compiler failed ({result.returncode}); inspect {BUILD / "build.log"}')
library = BUILD / 'libmadloop_tirust.so'
metadata = {
    'command': command,
    'rustc': subprocess.check_output([command[0], '--version'], text=True).strip(),
    'input_sha256': {str(p.relative_to(ROOT)): value for p, value in inputs.items()},
    'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                      [Path(__file__).resolve(), BUILD / 'src/lib.rs', BUILD / 'src/profile.rs']},
    'library_sha256': digest(library),
    'scope': 'Instrumented diagnostic DSO; current production dependencies and release optimization flags',
    'limitation': 'Timer instrumentation changes execution cost; compare disabled/enabled levels and pristine production control'}
(BUILD / 'build-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(library)
