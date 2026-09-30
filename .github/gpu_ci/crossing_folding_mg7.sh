#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job crossing_folding of gpu_runner_ci.yml.
# One generation of p p > w+ j --use_crossing=True, written twice by `output mg7`: FOLD
# (the crossed subprocesses are subprocesses.json entries evaluated by their base's
# library at an extended flavor id) and EXP (--use_crossing=False, every subprocess its
# own). Same seed everywhere, so the cpu runs of the two agree to the last digit.
#
# Checks, in this order (the gpu runs build the gpu libraries the gridpacks then ship):
#   exp_gpu      EXP on $BACKEND                                       runs
#   exp_cpu      EXP on cpu, saving a gridpack                         = exp_gpu
#   fold_gpu     FOLD on $BACKEND        GPU_CROSSING=0: refused, exit != 0, says why
#                                        GPU_CROSSING=1: runs, = exp_gpu
#   fold_cpu     FOLD on cpu, saving a gridpack                        = exp_cpu
#   gp_exp_gpu   the cpu-made EXP gridpack on $BACKEND                 = exp_cpu
#   gp_fold_gpu  the cpu-made FOLD gridpack on $BACKEND
#                                        GPU_CROSSING=0: refused before any run directory
#                                        GPU_CROSSING=1: runs, = fold_cpu
# "=" is agreement within 4 combined standard deviations.
# Environment: as pp_ttx_mg7.sh (BACKEND, MODULES, GPU_ARCH, VENV, MADSPACE_PREFIX,
# WORKDIR, NEVENTS, PDF_SET, CACHE_DIR), plus
#   GPU_CROSSING  1 once the GPU backend evaluates crossed flavor ids (Phase 1b), else 0
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
section() { echo; echo "=================== $* ($(date +%T))"; }
GPU_CROSSING=${GPU_CROSSING:-0}
SEED=4242

section "Environment"
source "$HERE/mg7_run_env.sh"

section "Generating p p > w+ j (folded and expanded)"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
cat > wj.mg7 << EOF
generate p p > w+ j --use_crossing=True
output mg7 FOLD
output mg7 EXP --use_crossing=False
EOF
cat wj.mg7
python3 "$REPO/bin/madgraph" wj.mg7 2>&1 | tee output.log
NCROSSED=$(python3 -c 'import json, sys; print(sum(1 for e in json.load(open(sys.argv[1])) if e.get("crossing")))' \
               FOLD/SubProcesses/subprocesses.json)
echo "FOLD: $NCROSSED crossed subprocess entries"
if [ "$NCROSSED" -eq 0 ]; then
    echo "::error::output mg7 folded no crossing: nothing to test"
    exit 1
fi

STATUS=0
RESULTS=results.txt
: > "$RESULTS"
# record NAME VERDICT DETAIL: one line of the job summary; FAIL makes the job fail
record() {
    printf '%s=%s %s\n' "$1" "$2" "$3" >> "$RESULTS"
    echo "--> $1: $2 $3"
    if [ "$2" = FAIL ]; then STATUS=1; echo "::error::$1: $3"; fi
}

# xsec DIR RUN_NAME: "mean error" of the latest info.json of that run, or nothing
xsec() {
    local info
    info=$(ls -t "$1"/Events/"$2"_*/info.json 2> /dev/null | sed -n 1p || true)
    [ -n "$info" ] || return 0
    python3 - "$info" << 'EOF' || true
import json, math, sys
proc = json.load(open(sys.argv[1]))["process"]
mean, err = float(proc["mean"]), float(proc.get("error") or 0.0)
if mean > 0 and math.isfinite(mean):
    print(mean, err)
EOF
}

# agree "m1 e1" "m2 e2": within 4 combined standard deviations
agree() {
    python3 - $1 $2 << 'EOF'
import math, sys
m1, e1, m2, e2 = map(float, sys.argv[1:5])
sys.exit(0 if abs(m1 - m2) <= 4 * math.hypot(e1, e2) + 1e-12 * abs(m1) else 1)
EOF
}

# run DIR DEVICE RUN_NAME SAVE_GRIDPACK: bin/generate_events -f; prints its exit status
run() {
    # (errexit does not reach into the $(run ...) of the callers: check by hand)
    python3 "$HERE/set_run_card.py" "$1/Cards/run_card.toml" \
        run.device="[\"$2\"]" run.seed=$SEED run.run_name="\"$3\"" \
        generation.events=$NEVENTS beam.pdf="\"$PDF_SET\"" \
        systematics.enable=false gridpack.save_gridpack=$4 > "$3.log" 2>&1 \
        || { echo 99; return; }
    local ret=0
    (cd "$1" && python3 bin/generate_events -f) >> "$3.log" 2>&1 || ret=$?
    echo $ret
}

# run_gridpack GRIDPACK RUN_NAME: the gridpack on $BACKEND; prints its exit status
run_gridpack() {
    local ret=0
    (cd "$1" && python3 bin/generate_events --device "$BACKEND" --seed $SEED \
         --run_name "$2" --events "$NEVENTS") > "$2.log" 2>&1 || ret=$?
    echo $ret
}

# expect_runs NAME EXIT DIR RUN_NAME [REFERENCE_NAME REFERENCE]: sets X_<NAME>
expect_runs() {
    local name=$1 ret=$2 x
    x=$(xsec "$3" "$4")
    eval "X_$name=\"$x\""
    if [ "$ret" -ne 0 ] || [ -z "$x" ]; then
        record "$name" FAIL "exit $ret, cross section '${x:-none}' (see $name.log)"
    elif [ -n "$5" ] && ! agree "$x" "$6"; then
        record "$name" FAIL "$x differs from $5 ($6)"
    else
        record "$name" PASS "$x${5:+ (agrees with $5)}"
    fi
}

# expect_refused NAME EXIT LOG [DIR_THAT_MUST_STAY_EMPTY]
expect_refused() {
    if [ "$2" -eq 0 ]; then
        record "$1" FAIL "exit 0: the $BACKEND run of folded crossings was not refused"
    elif ! grep -q 'does not support' "$3" || ! grep -q -- '--use_crossing=False' "$3"; then
        record "$1" FAIL "exit $2 without the crossing refusal message (see $3)"
    elif [ -n "$4" ] && [ -n "$(ls -A "$4" 2> /dev/null)" ]; then
        record "$1" FAIL "refused, but made a run directory in $4"
    else
        record "$1" PASS "refused (exit $2)"
    fi
}

START=$SECONDS
section "EXP on $BACKEND"
ret=$(run EXP "$BACKEND" exp_gpu false)
expect_runs exp_gpu "$ret" EXP exp_gpu

section "EXP on cpu (saving a gridpack)"
ret=$(run EXP cpu exp_cpu true)
expect_runs exp_cpu "$ret" EXP exp_cpu exp_gpu "$X_exp_gpu"

section "FOLD on $BACKEND"
ret=$(run FOLD "$BACKEND" fold_gpu false)
if [ "$GPU_CROSSING" = 1 ]; then
    expect_runs fold_gpu "$ret" FOLD fold_gpu exp_gpu "$X_exp_gpu"
else
    expect_refused fold_gpu "$ret" fold_gpu.log
fi

section "FOLD on cpu (saving a gridpack)"
ret=$(run FOLD cpu fold_cpu true)
expect_runs fold_cpu "$ret" FOLD fold_cpu exp_cpu "$X_exp_cpu"

section "cpu-made EXP gridpack on $BACKEND"
GP=$(ls -d EXP/Events/exp_cpu_*/gridpack 2> /dev/null | sed -n 1p || true)
if [ -z "$GP" ]; then
    record gp_exp_gpu FAIL "no gridpack saved by exp_cpu"
else
    ret=$(run_gridpack "$GP" gp_exp_gpu)
    expect_runs gp_exp_gpu "$ret" "$GP" gp_exp_gpu exp_cpu "$X_exp_cpu"
fi

section "cpu-made FOLD gridpack on $BACKEND"
GP=$(ls -d FOLD/Events/fold_cpu_*/gridpack 2> /dev/null | sed -n 1p || true)
if [ -z "$GP" ]; then
    record gp_fold_gpu FAIL "no gridpack saved by fold_cpu"
else
    ret=$(run_gridpack "$GP" gp_fold_gpu)
    if [ "$GPU_CROSSING" = 1 ]; then
        expect_runs gp_fold_gpu "$ret" "$GP" gp_fold_gpu fold_cpu "$X_fold_cpu"
    else
        expect_refused gp_fold_gpu "$ret" gp_fold_gpu.log "$GP/Events"
    fi
fi
WALLTIME=$((SECONDS - START))

cat > summary.txt << EOF
node=$(hostname)
gpu=${GPU_NAME:-unknown}
backend=$BACKEND
gpu_crossing=$GPU_CROSSING
crossed_entries=$NCROSSED
events=$NEVENTS
walltime=${WALLTIME}s
EOF
cat "$RESULTS" >> summary.txt
section "Summary"
cat summary.txt
exit $STATUS
