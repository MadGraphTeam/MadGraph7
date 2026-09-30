#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job crossing_folding of gpu_runner_ci.yml.
# Per process (p p > w+ j, p p > j j, p p > w+ j j): one generation --use_crossing=True,
# written twice by `output mg7`: FOLD_<tag> (the crossed subprocesses are
# subprocesses.json entries evaluated by their base's library at an extended flavor id)
# and EXP_<tag> (--use_crossing=False, every subprocess its own). Same seed everywhere,
# so the runs of the two on one device agree to the last digit.
#
# Checks, per process <tag>, in this order (the gpu runs build the gpu libraries the
# gridpacks then ship; the gridpacks for the first process only):
#   <tag>_exp_gpu      EXP on $BACKEND
#   <tag>_exp_cpu      EXP on cpu                                  = <tag>_exp_gpu
#   <tag>_fold_gpu     FOLD on $BACKEND                            = <tag>_exp_gpu
#   <tag>_fold_cpu     FOLD on cpu                                 = <tag>_exp_cpu
#   <tag>_gp_exp_gpu   the cpu-made EXP gridpack on $BACKEND       = <tag>_exp_cpu
#   <tag>_gp_fold_gpu  the cpu-made FOLD gridpack on $BACKEND      = <tag>_fold_cpu
# "=" is agreement within 4 combined standard deviations; each run must succeed.
# Environment: as pp_ttx_mg7.sh (BACKEND, MODULES, GPU_ARCH, VENV, MADSPACE_PREFIX,
# WORKDIR, NEVENTS, PDF_SET, CACHE_DIR).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
section() { echo; echo "=================== $* ($(date +%T))"; }
SEED=4242

section "Environment"
source "$HERE/mg7_run_env.sh"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

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

# check_process TAG PROCESS GRIDPACKS: the checks above for one process
check_process() {
    local tag=$1 proc=$2 gridpacks=$3 fold=FOLD_$1 exp=EXP_$1 ret n gp
    section "Generating $proc (folded and expanded)"
    cat > "$tag.mg7" << EOF
generate $proc --use_crossing=True
output mg7 $fold
output mg7 $exp --use_crossing=False
EOF
    cat "$tag.mg7"
    python3 "$REPO/bin/madgraph" "$tag.mg7" > "${tag}_output.log" 2>&1 || true
    n=$(python3 -c 'import json, sys; print(sum(1 for e in json.load(open(sys.argv[1])) if e.get("crossing")))' \
            "$fold/SubProcesses/subprocesses.json" 2> /dev/null || echo 0)
    CROSSED="$CROSSED $tag:$n"
    if [ "$n" -eq 0 ]; then
        record "${tag}_output" FAIL "output mg7 folded no crossing (see ${tag}_output.log)"
        return
    fi
    echo "$fold: $n crossed subprocess entries"

    section "$tag: EXP on $BACKEND"
    ret=$(run "$exp" "$BACKEND" "${tag}_exp_gpu" false)
    expect_runs "${tag}_exp_gpu" "$ret" "$exp" "${tag}_exp_gpu"
    section "$tag: EXP on cpu"
    ret=$(run "$exp" cpu "${tag}_exp_cpu" "$gridpacks")
    expect_runs "${tag}_exp_cpu" "$ret" "$exp" "${tag}_exp_cpu" "${tag}_exp_gpu" "$(eval echo \$X_${tag}_exp_gpu)"
    section "$tag: FOLD on $BACKEND"
    ret=$(run "$fold" "$BACKEND" "${tag}_fold_gpu" false)
    expect_runs "${tag}_fold_gpu" "$ret" "$fold" "${tag}_fold_gpu" "${tag}_exp_gpu" "$(eval echo \$X_${tag}_exp_gpu)"
    section "$tag: FOLD on cpu"
    ret=$(run "$fold" cpu "${tag}_fold_cpu" "$gridpacks")
    expect_runs "${tag}_fold_cpu" "$ret" "$fold" "${tag}_fold_cpu" "${tag}_exp_cpu" "$(eval echo \$X_${tag}_exp_cpu)"
    [ "$gridpacks" = true ] || return 0

    local kind dir
    for kind in exp fold; do
        if [ "$kind" = exp ]; then dir=$exp; else dir=$fold; fi
        section "$tag: cpu-made $dir gridpack on $BACKEND"
        gp=$(ls -d "$dir"/Events/"${tag}_${kind}_cpu"_*/gridpack 2> /dev/null | sed -n 1p || true)
        if [ -z "$gp" ]; then
            record "${tag}_gp_${kind}_gpu" FAIL "no gridpack saved by ${tag}_${kind}_cpu"
            continue
        fi
        ret=$(run_gridpack "$gp" "${tag}_gp_${kind}_gpu")
        expect_runs "${tag}_gp_${kind}_gpu" "$ret" "$gp" "${tag}_gp_${kind}_gpu" \
            "${tag}_${kind}_cpu" "$(eval echo \$X_${tag}_${kind}_cpu)"
    done
}

START=$SECONDS
CROSSED=
check_process wj 'p p > w+ j' true
check_process jj 'p p > j j' false
check_process wjj 'p p > w+ j j' false
WALLTIME=$((SECONDS - START))

cat > summary.txt << EOF
node=$(hostname)
gpu=${GPU_NAME:-unknown}
backend=$BACKEND
crossed_entries=${CROSSED# }
events=$NEVENTS
walltime=${WALLTIME}s
EOF
cat "$RESULTS" >> summary.txt
section "Summary"
cat summary.txt
exit $STATUS
