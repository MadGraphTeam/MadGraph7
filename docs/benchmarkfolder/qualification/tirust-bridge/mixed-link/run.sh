#!/usr/bin/env bash
set -euo pipefail
source tools/tirust_env.sh
tirust_mix_dir="$TIRUST_MG_ROOT/.tirust-build/mixed-link"
tirust_mix_ninja="${TIRUST_MIXED_NINJARS_LIB:-$TIRUST_MG_ROOT/HEPTools/ninjars/lib}"
tirust_mix_dso="${TIRUST_MIXED_DSO_DIR:-$TIRUST_MG_ROOT/.tirust-build/cargo-build/debug/deps}"
tirust_mix_tir="$tirust_mix_dir/tirust-wrapper"
for file in libninjars.a mninjars.mod ninjars-native-libs.txt ninjars-link-before.txt; do
    test -f "$tirust_mix_ninja/$file" || { echo "NinjaRS artifact is not ready: $file"; exit 2; }
done
test -f "$tirust_mix_dso/libmadloop_tirust.so" || { echo 'TIRust shared library is not ready'; exit 2; }
nm -D --defined-only "$tirust_mix_dso/libmadloop_tirust.so" | \
    awk '$3=="ml_tirust_qp_cache_statistics" {found=1} END {exit !found}' || \
    { echo 'TIRust shared library does not yet expose the QP cache ABI'; exit 2; }
mkdir -p "$tirust_mix_tir"
gfortran -O3 -fPIC -J"$tirust_mix_tir" -c vendor/tirust/mtirust.f90 -o "$tirust_mix_tir/mtirust.o"
ar rcs "$tirust_mix_tir/libtirust.a" "$tirust_mix_tir/mtirust.o"
mapfile -t tirust_mix_native < <(python3 -c \
    'import pathlib,shlex,sys; print("\n".join(shlex.split(pathlib.Path(sys.argv[1]).read_text())))' \
    "$tirust_mix_ninja/ninjars-native-libs.txt")
mapfile -t tirust_mix_before < <(python3 -c \
    'import pathlib,shlex,sys; print("\n".join(shlex.split(pathlib.Path(sys.argv[1]).read_text())))' \
    "$tirust_mix_ninja/ninjars-link-before.txt")
gfortran -O3 -fcheck=all -I"$tirust_mix_tir" -I"$tirust_mix_ninja" \
    -c "$tirust_mix_dir/smoke.f90" -o "$tirust_mix_dir/smoke.o"
gfortran "$tirust_mix_dir/smoke.o" "${tirust_mix_before[@]}" \
    -L"$tirust_mix_tir" -ltirust -L"$tirust_mix_dso" -lmadloop_tirust \
    -L"$tirust_mix_ninja" -lninjars "${tirust_mix_native[@]}" \
    -Wl,-rpath,"$tirust_mix_dso" -o "$tirust_mix_dir/tirust-first"
gfortran "$tirust_mix_dir/smoke.o" "${tirust_mix_before[@]}" \
    -L"$tirust_mix_ninja" -lninjars "${tirust_mix_native[@]}" \
    -L"$tirust_mix_tir" -ltirust -L"$tirust_mix_dso" -lmadloop_tirust \
    -Wl,-rpath,"$tirust_mix_dso" -o "$tirust_mix_dir/ninjars-first"
python3 - <<'PY'
from pathlib import Path
import subprocess
root = Path('.tirust-build/mixed-link')
failed = False
for linked in ('tirust-first', 'ninjars-first'):
    for initialized in ('tirust-first', 'ninjars-first'):
        try:
            result = subprocess.run([str(root / linked), initialized], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=180)
            output, code = result.stdout, result.returncode
        except subprocess.TimeoutExpired as error:
            output, code = error.stdout or b'', 124
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
            output += '\nTimed out after 180 seconds.\n'
        (root / f'{linked}-{initialized}.log').write_text(output)
        print(linked, initialized, 'exit', code, flush=True)
        print(output, flush=True)
        failed |= code != 0
raise SystemExit(int(failed))
PY
