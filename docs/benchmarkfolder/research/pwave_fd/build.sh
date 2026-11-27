#!/usr/bin/env bash
set -euo pipefail
source /shared/external/madloop_onia/madgraph7_madloop_onia/tools/tirust_env.sh
PW=${PW:-/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad/pw}
REV=${REV:-f12b322}
cd $PW
rm -rf bridge install src
mkdir -p src
git -C /shared/external/iregi/iregi-rust archive $REV | tar -x -C src
rsync -a --exclude target --exclude .git --exclude '*.o' --exclude '*.mod' --exclude '*.a' --exclude '*.so' --exclude smoke --exclude 'tirust-native-libs.txt' --exclude 'tirust-link-before.txt' /shared/external/madloop_onia/madgraph7_madloop_onia/vendor/tirust/ bridge/
cp /shared/external/madloop_onia/madgraph7_madloop_onia/HEPTools/tirust/Cargo.lock bridge/Cargo.lock
# optional patch of the bridge copy (paths a/vendor/tirust/...), e.g. madloop-tirust-relation-tiers.patch
if [ -n "${BRIDGE_PATCH:-}" ]; then patch -p3 --batch --forward --fuzz=0 -d bridge -i "$BRIDGE_PATCH"; fi
# optional oneloop revision for the bridge copy's pole kernels, to match a TIRust revision that
# re-pins oneloop (8b436f9: cdbc912); vendor/tirust itself is not changed
if [ -n "${ONELOOP_REV:-}" ]; then
  sed -i -E 's#(oneloopmaster\.git", rev = ")[0-9a-f]+#\1'"$ONELOOP_REV"'#' bridge/pole-kernels/Cargo.toml
  grep -c "$ONELOOP_REV" bridge/pole-kernels/Cargo.toml
fi
mkdir -p src/madloop-pole-kernels
cp -r bridge/pole-kernels/. src/madloop-pole-kernels/
(cd src && patch -p1 --batch --forward --fuzz=0 -i $PW/bridge/patches/0001-direct-poles-only.patch)
# FOLLOWUP=none when the bridge's poles-only patch already carries the follow-up (59aee45 and later)
if [ "${FOLLOWUP:-}" != none ]; then (cd src && patch -p1 --batch --forward --fuzz=0 -i ${FOLLOWUP:-/shared/external/iregi/pwave-patches/0001-followup-soft-line-poles.patch}); fi
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
python3 /tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad/pw/fix_config.py $PW/install/lib
$PW/smoke --require-poles
echo BUILD_OK
