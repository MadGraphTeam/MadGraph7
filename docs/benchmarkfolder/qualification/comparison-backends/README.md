# Comparison-backend native qualification

This directory retains the native installation and smoke evidence captured on
2026-11-09 with GNU Fortran 15.2.0. It qualifies the comparison libraries and
their coexistence. It does not establish process-level numerical agreement,
TIRust correctness, or benchmark performance acceptance.

## Evidence index

| Evidence | Retained files | Outcome |
|---|---|---|
| Dependency pins, source and library SHA-256s | [comparison-dependencies.json](comparison-dependencies.json) | CutTools 1.9.3; Ninja 1.2.1; OneLOop 3.6; COLLIER 1.2.9; NinjaRS `41a4c00e…` |
| CutTools rank-six example | [cuttools-smoke.log.txt](cuttools-smoke.log.txt) | `stable=T`, zero unstable points |
| Ninja double/quadruple examples | [DP](ninja-dp-smoke.log.txt), [QP](ninja-qp-smoke.log.txt) | Both native example executables completed |
| COLLIER static/shared tensor fixtures | [comparison](collier-comparison.json), [measured coefficients](collier-reference.json), [static provenance](collier-static-provenance.json), [shared provenance](collier-shared-provenance.json) | Eight cases, 455 components; all status flags zero; maximum relative difference from the stored historical fixture `4.86e-15`; static/shared results identical |
| NinjaRS transactional installation | [install log](ninjars-install.log.txt), [installed metadata](ninjars-config.json) | ABI4 DP, binary128-input rescue, separated UV/IR poles and rounding smoke passed |
| Compiler-runtime regression | [original failure](ninjars-runtime-failure.log.txt), [corrected smoke](ninjars-staticgcc-smoke.log.txt), [installer tests](ninjars-runtime-unit.log.txt) | Original `STOP 8` fixed; 22 installer tests passed |
| Four-backend coexistence | [driver](comparison-coexistence.f90), [build](comparison-coexistence-build.log.txt), [output](comparison-coexistence.log.txt), [linked artifact hashes](comparison-coexistence-provenance.json) | CutTools, Ninja and COLLIER initialized together; Ninja scalar evaluation and the complete NinjaRS smoke passed in the same executable |

[evidence-index.json](evidence-index.json) identifies original evidence paths,
byte sizes and SHA-256s. The copied logs and metadata are byte-for-byte original;
their absolute paths identify the qualification machine and may refer to removed
installation staging directories. [SHA256SUMS](SHA256SUMS) covers all retained
files except itself. No libraries or source archives are retained here.

## Compiler-runtime correction

The initial NinjaRS installer stopped at the existing directed-rounding check
before evaluating any integral. A preceding `-lgcc_s` did not prevent Rust's
archive definitions of binary128 helpers from entering the executable on this
GNU/Nix toolchain. Those helpers failed the Fortran upward-rounding assertion.
The failure is preserved, including its link command.

The corrected link order places the actual archive reported by
`gfortran -print-file-name=libgcc.a`, followed by `-lgcc_s`, before `libninjars.a`.
The installer stores this ordering in `ninjars-link-before.txt` and
`ninjars-config.json`; exported processes consume the same metadata. Dependency
revisions and Rust release settings were unchanged. The real installation smoke
and four-backend coexistence driver passed with that ordering.

## Reproduce the installations

Run every command from the MadGraph checkout. This sequence uses the pinned
archives rather than Ninja's mutable `latest` URL. Existing installations are
replaced by the legacy HEPTools commands below; use a fresh checkout for a clean
reproduction.

```bash
source tools/tirust_env.sh
python3 - <<'PY'
import hashlib, json, urllib.request
from pathlib import Path
evidence = Path('docs/benchmarkfolder/qualification/comparison-backends')
data = json.loads((evidence/'comparison-dependencies.json').read_text())
archives = Path('.tirust-build/dependency-sources')
archives.mkdir(parents=True, exist_ok=True)
for name in ('installer', 'oneloop', 'ninja', 'collier'):
    entry = data[name]
    url = entry.get('source', 'https://madgraph.mi.infn.it/Downloads/HEPToolsInstaller/HEPToolsInstaller_V168.tar.gz')
    target = archives/Path(entry['source_archive']).name
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == entry['source_sha256'], target
PY
mkdir -p HEPTools
tar -xzf .tirust-build/dependency-sources/HEPToolsInstaller_V168.tar.gz -C HEPTools
```

The qualification host has no `/bin/bash`. Adapt only the downloaded installer's
shell scripts to the available Bash executable:

```bash
python3 - <<'PY'
from pathlib import Path
import shutil
for path in Path('HEPTools/HEPToolsInstallers').glob('*.sh'):
    text = path.read_text()
    if text.startswith('#!/bin/bash'):
        path.write_text(text.replace('#!/bin/bash', '#!'+shutil.which('bash'), 1))
PY
python3 HEPTools/HEPToolsInstallers/HEPToolInstaller.py ninja \
  --prefix="$PWD/HEPTools" --mg5_path="$PWD" \
  --fortran_compiler=gfortran --cpp_compiler=g++ --force \
  --ninja_tarball="$PWD/.tirust-build/dependency-sources/ninja-1.2.1.tar.gz" \
  --oneloop_tarball="$PWD/.tirust-build/dependency-sources/OneLOop-3.6.tgz"
python3 HEPTools/HEPToolsInstallers/HEPToolInstaller.py collier \
  --prefix="$PWD/HEPTools" --mg5_path="$PWD" --fortran_compiler=gfortran --force \
  --collier_tarball="$PWD/.tirust-build/dependency-sources/collier-1.2.9.tar.gz"

mkdir -p .tirust-build/cuttools
cp -a vendor/CutTools/. .tirust-build/cuttools/
make -C .tirust-build/cuttools -j1 FC=gfortran
mkdir -p HEPTools/cuttools vendor/CutTools/includects
cp -a .tirust-build/cuttools/. HEPTools/cuttools/
cp -a .tirust-build/cuttools/includects/. vendor/CutTools/includects/

cat > .tirust-build/install-comparison.mg5 <<'EOF'
set auto_update 0
set ninja HEPTools/ninja/lib
set collier HEPTools/collier
install ninjars --jobs=4 --update
quit
EOF
CARGO_TARGET_DIR="$PWD/.tirust-build/ninjars-target" \
CARGO_BUILD_BUILD_DIR="$PWD/.tirust-build/ninjars-build" \
  python3 bin/madgraph .tirust-build/install-comparison.mg5
```

CutTools is built serially because its makefile does not declare all Fortran
module dependencies. NinjaRS uses the checked-in Cargo manifest/lockfile and its
existing oneloopmaster `5b710437ada83cbc6e8c6d51e56f7d6ee6a0edb5` dependency;
TIRust's newer oneloopmaster pin must not replace it. Original source-archive
hashes are reproducible identities; installed library hashes can vary with the
compiler, build paths and archive metadata.

## Reproduce the native checks

```bash
gfortran -O2 -IHEPTools/cuttools/includects \
  vendor/CutTools/examples/example0.f vendor/CutTools/examples/numerators.f \
  vendor/CutTools/examples/mpnumerators.f90 HEPTools/cuttools/includects/libcts.a \
  -o .tirust-build/cuttools-smoke
.tirust-build/cuttools-smoke

TIRUST_FC_LIB=$(dirname "$(gfortran -print-file-name=libgfortran.so)")
make -C HEPTools/ninja/Ninja/examples -j2 simple_test quadninja_test \
  "LIBS=-L$TIRUST_FC_LIB -Wl,-rpath,$TIRUST_FC_LIB -lgfortran -lquadmath -lstdc++"
HEPTools/ninja/Ninja/examples/simple_test
HEPTools/ninja/Ninja/examples/quadninja_test

python3 vendor/ninjars/tests/poles/generate_collier_reference.py \
  --collier HEPTools/collier --work-dir .tirust-build/collier-reproduction \
  --output .tirust-build/collier-reproduction/reference.json
python3 -m unittest tests.unit_tests.various.test_ninjars_install
```

The successful NinjaRS installation already executes its complete native smoke.
Run the retained coexistence driver using the installed metadata:

```bash
python3 - <<'PY'
import json, subprocess
from pathlib import Path
root = Path.cwd()
lib = root/'HEPTools/ninjars/lib'
cfg = json.loads((lib/'ninjars-config.json').read_text())
output = root/'.tirust-build/comparison-coexistence'
command = ['gfortran', '-I'+str(lib), '-IHEPTools/ninja/include',
           '-IHEPTools/collier/include',
           'docs/benchmarkfolder/qualification/comparison-backends/comparison-coexistence.f90']
command += cfg['link_before'] + [str(lib/'libninjars.a'),
           'HEPTools/cuttools/includects/libcts.a', '-LHEPTools/ninja/lib',
           '-Wl,-rpath,'+str(root/'HEPTools/ninja/lib'), '-lninja',
           '-LHEPTools/collier', '-Wl,-rpath,'+str(root/'HEPTools/collier'), '-lcollier']
command += cfg['native_libs'] + ['-lstdc++', '-o', str(output)]
subprocess.run(command, check=True)
subprocess.run([str(output)], check=True)
PY
```

Run Symbolica-dependent numerical checks serially within the applicable runtime
allowance. The retained coexistence result concerns backends 1, 6, 7 and 8;
TIRust/NinjaRS mixed-link qualification is recorded separately by the integration
campaign. [Source notice inventory](source-notices.md) records the inspected
release notices and their provenance.
