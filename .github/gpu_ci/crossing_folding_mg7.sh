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
# and, per process <tag> of STANDALONE (p p > w+ j j, p p > j j), at one fixed phase-space
# point with no Monte-Carlo noise: every matrix element check_sa.exe prints (every base
# flavor, every crossing folded in) must agree between its $BACKEND and cpu builds, in
# double precision, to 1e-10, and none may be NaN -- for the folded and the expanded
# standalone output, after checking that the folded one did fold (fewer P dirs, crossings
# to show):
#   <tag>_sa_output, <tag>_sa_fold_<P dir>, <tag>_sa_exp_<P dir>
# "=" is agreement within 4 combined standard deviations; each run must succeed, within
# RUN_TIMEOUT seconds (default 1200): a hung run fails alone instead of taking the
# whole allocation with it.
# Environment: as pp_ttx_mg7.sh (BACKEND, MODULES, GPU_ARCH, VENV, MADSPACE_PREFIX,
# WORKDIR, NEVENTS, PDF_SET, CACHE_DIR), plus RUN_TIMEOUT.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
section() { echo; echo "=================== $* ($(date +%T))"; }
SEED=4242
RUN_TIMEOUT=${RUN_TIMEOUT:-1200}

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
    (cd "$1" && timeout "$RUN_TIMEOUT" python3 bin/generate_events -f) >> "$3.log" 2>&1 || ret=$?
    echo $ret
}

# run_gridpack GRIDPACK RUN_NAME: the gridpack on $BACKEND; prints its exit status
run_gridpack() {
    local ret=0
    (cd "$1" && timeout "$RUN_TIMEOUT" python3 bin/generate_events --device "$BACKEND" --seed $SEED \
         --run_name "$2" --events "$NEVENTS") > "$2.log" 2>&1 || ret=$?
    echo $ret
}

# expect_runs NAME EXIT DIR RUN_NAME [REFERENCE_NAME REFERENCE]: sets X_<NAME>
expect_runs() {
    local name=$1 ret=$2 x
    x=$(xsec "$3" "$4")
    eval "X_$name=\"$x\""
    if [ "$ret" -eq 124 ]; then
        record "$name" FAIL "timed out after ${RUN_TIMEOUT}s (see $name.log)"
    elif [ "$ret" -ne 0 ] || [ -z "$x" ]; then
        record "$name" FAIL "exit $ret, cross section '${x:-none}' (see $name.log)"
    elif [ -n "$5" ] && [ -z "$6" ]; then
        # the reference run failed, and is recorded as such
        record "$name" PASS "$x (not compared: $5 has no cross section)"
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

# check_sa_devices TAG KIND DIR: check_sa.exe of every P dir of the standalone output DIR,
# built for $BACKEND and for cpu (FPTYPE=d), must print the same matrix elements
check_sa_devices() {
    local tag=$1 kind=$2 out=$3 pdir name dev ret worst gpu=$BACKEND
    # (BACKEND=cpu, the laptop try-out of this script, is no madmatrix backend)
    if [ "$gpu" = cpu ]; then gpu=scalar; fi
    for pdir in "$out"/SubProcesses/P*/; do
        name=${tag}_sa_${kind}_$(basename "$pdir")
        for dev in "$gpu" scalar; do
            ret=0
            (cd "$pdir" && make -j"${SLURM_CPUS_PER_TASK:-4}" BACKEND=$dev FPTYPE=d check_sa.exe \
                 && timeout "$RUN_TIMEOUT" ./check_sa.exe) > "$name.$dev.log" 2>&1 || ret=$?
            if [ "$ret" -ne 0 ]; then
                record "$name" FAIL "check_sa.exe on $dev: exit $ret (see $name.$dev.log)"
                continue 2
            fi
        done
        # a folded P dir must also show its crossings (check_sa reads crossing_demo.dat)
        if [ -s "$pdir/crossing_demo.dat" ] \
               && ! grep -q 'Crossed processes folded into this matrix element' "$name.$gpu.log"; then
            record "$name" FAIL "check_sa.exe on $gpu shows none of the crossings of crossing_demo.dat"
            continue
        fi
        worst=$(python3 - "$name.$gpu.log" "$name.scalar.log" << 'EOF'
import math, re, sys
def values(path):
    return [float(v) for v in re.findall(r'^ *Matrix element = (\S+) GeV', open(path).read(), re.M)]
gpu, cpu = values(sys.argv[1]), values(sys.argv[2])
if not gpu or len(gpu) != len(cpu):
    print('count %d vs %d' % (len(gpu), len(cpu)))
elif not all(math.isfinite(v) for v in gpu + cpu):
    # max() would drop a NaN after the first term, and report it as agreement
    print('nonfinite %d of %d' % (sum(not math.isfinite(v) for v in gpu + cpu), 2 * len(gpu)))
else:
    print('%.3g %d' % (max(abs(a - b) / max(abs(a), abs(b), 1e-300) for a, b in zip(gpu, cpu)), len(gpu)))
EOF
)
        case $worst in
            count*) record "$name" FAIL "matrix elements printed: $worst" ;;
            nonfinite*) record "$name" FAIL "matrix elements that are NaN or infinite: ${worst#nonfinite }" ;;
            *) if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 1e-10 else 1)" "${worst% *}"; then
                   record "$name" PASS "${worst#* } matrix elements, max relative difference ${worst% *}"
               else
                   record "$name" FAIL "${worst#* } matrix elements, max relative difference ${worst% *} > 1e-10"
               fi ;;
        esac
    done
}

# check_standalone TAG PROCESS: folded and expanded standalone outputs, $BACKEND vs cpu
check_standalone() {
    local tag=$1 proc=$2
    section "$tag: standalone check_sa, $BACKEND vs cpu"
    cat > "${tag}_sa.mg7" << EOF
generate $proc --use_crossing=True
output standalone SA_FOLD_$tag
output standalone SA_EXP_$tag --use_crossing=False
EOF
    ret=0
    python3 "$REPO/bin/madgraph" "${tag}_sa.mg7" > "${tag}_sa_output.log" 2>&1 || ret=$?
    # The folded output must actually fold: fewer P dirs than the expanded one, and
    # crossings for check_sa to show. Otherwise its checks would pass on the base
    # flavors alone.
    local nfold nexp ndemo
    nfold=$(ls -d SA_FOLD_"$tag"/SubProcesses/P*/ 2> /dev/null | wc -l)
    nexp=$(ls -d SA_EXP_"$tag"/SubProcesses/P*/ 2> /dev/null | wc -l)
    ndemo=$(ls SA_FOLD_"$tag"/SubProcesses/P*/crossing_demo.dat 2> /dev/null | wc -l)
    if [ "$ret" -ne 0 ] || [ "$nfold" -eq 0 ] || [ "$nexp" -eq 0 ]; then
        record "${tag}_sa_output" FAIL "output standalone: exit $ret, $nfold folded and $nexp expanded P dirs (see ${tag}_sa_output.log)"
        return 0
    elif [ "$nfold" -ge "$nexp" ] || [ "$ndemo" -eq 0 ]; then
        record "${tag}_sa_output" FAIL "the folded output did not fold: $nfold P dirs (expanded: $nexp), $ndemo with crossings"
    else
        record "${tag}_sa_output" PASS "$nfold folded P dirs ($ndemo with crossings), $nexp expanded"
    fi
    check_sa_devices "$tag" fold "SA_FOLD_$tag"
    check_sa_devices "$tag" exp "SA_EXP_$tag"
}

START=$SECONDS
CROSSED=
check_process wj 'p p > w+ j' true
check_process jj 'p p > j j' false
check_process wjj 'p p > w+ j j' false
check_standalone wjj 'p p > w+ j j'
check_standalone jj 'p p > j j'
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
