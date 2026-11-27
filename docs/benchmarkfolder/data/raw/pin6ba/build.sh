#!/usr/bin/env bash
# TIRust 6ba4b53 (PR MadGraphTeam/MadGraph7#557's pin) with the bridge of
# onia_loop + the pin commit b49e9ea7df cherry-picked (tree 6041706465).
set -euo pipefail
source /shared/external/madloop_onia/madgraph7_madloop_onia/tools/tirust_env.sh
SP=/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad
PW=$SP/pin6ba
REV=6ba4b531
TREE=604170646584009241fe758b4630dea329c262ac
cd $PW
mkdir -p src bridge
git -C /shared/external/iregi/iregi-rust archive $REV | tar -x -C src
git -C /shared/external/madloop_onia/madgraph7_madloop_onia archive $TREE vendor/tirust | tar -x -C bridge --strip-components=2
mkdir -p src/madloop-pole-kernels
cp -r bridge/pole-kernels/. src/madloop-pole-kernels/
(cd src && patch -p1 --batch --forward --fuzz=0 -i $PW/bridge/patches/0001-direct-poles-only.patch)
mkdir -p bridge/.cargo
cat > bridge/.cargo/config.toml <<CFG
[patch."https://github.com/huashengshao/TIRust"]
iregi-rust = { path = "$PW/src/crates/iregi-rust" }
iregi-masters = { path = "$PW/src/crates/iregi-masters" }
CFG
export CARGO_TARGET_DIR=$PW/target
export CARGO_BUILD_JOBS=32
cd bridge
cargo metadata --format-version=1 > /dev/null
make install PREFIX=$PW/install FC=gfortran CARGO_FEATURES=poles-only
python3 - <<PY
import json
c = json.load(open('/shared/external/madloop_onia/madgraph7_madloop_onia/HEPTools/tirust/lib/tirust-config.json'))
c['raised_powers'] = True
json.dump(c, open('$PW/install/lib/tirust-config.json', 'w'), indent=1)
PY
gfortran -I$PW/install/lib smoke.f90 -lgcc_s $PW/install/lib/libtirust.a -L$PW/install/lib -Wl,-rpath,$PW/install/lib -lmadloop_tirust -o $PW/smoke
python3 $SP/pw/fix_config.py $PW/install/lib
$PW/smoke --require-poles
echo BUILD_OK
