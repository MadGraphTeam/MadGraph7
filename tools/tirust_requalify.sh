#!/usr/bin/env bash
# Requalify the current TIRust pin end to end on this host (shared-host evidence, never timing
# acceptance). Phases are resumable: completed outputs are skipped.
#
#   tools/tirust_requalify.sh CAMPAIGN [PHASE...]
#
# CAMPAIGN   new campaign ID, e.g. tirust-<pin8>-1
# PHASES     install register build numerical dd-gate stress stability provenance summaries
#            (default: all, in this order)
# Environment:
#   PREDECESSOR  campaign whose physical cards are inherited (default entry-latch-cache-guard-1)
#   WORKDIR      ignored work directory (default .tirust-build/q-<CAMPAIGN>)
#   CPU0         first CPU for pinned workers (default 200); JOBS (default 4)
#   REASON       registration reason (required for 'register')
#
# Run from the MadGraph root. Uses tools/tirust_env.sh (gfortran 15.2, Rust 1.97.1, Python 3.13)
# outside `nix develop`, and FFLAGS='-w -fPIC -O3' for every export.
set -u
CAMPAIGN=${1:?campaign ID}; shift
PHASES=${*:-install register build numerical dd-gate stress stability provenance summaries}
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
source tools/tirust_env.sh >/dev/null
export FFLAGS='-w -fPIC -O3'
PREDECESSOR=${PREDECESSOR:-entry-latch-cache-guard-1}
W=${WORKDIR:-$ROOT/.tirust-build/q-$CAMPAIGN}; mkdir -p "$W/logs"
CPU0=${CPU0:-200}; JOBS=${JOBS:-4}
PREFIX=$W/HEPTools/tirust
REPORT=docs/benchmarkfolder/campaigns/$CAMPAIGN
CASES="qcd_n1_uux_z ew_n1_udx_wp qcd_n2_gg_ttx ew_n2_epem_mupmum mixed_uux_ga li_gg_h li_gg_hh qcd_n3_uux_zgg ew_n3_epem_ttxa qcd_n4_uux_zzgg ew_n4_epem_mupmumaa"
STRESS_CASES="qcd_n2_gg_ttx qcd_n3_uux_zgg qcd_n4_uux_zzgg ew_n2_epem_mupmum ew_n3_epem_ttxa ew_n4_epem_mupmumaa mixed_uux_ga li_gg_hh cms_ew_n2_epem_mupmum"
log() { echo "[$(date -u +%T)] $*" | tee -a "$W/logs/requalify.log"; }
py() { python3 -B "$@"; }

phase_install() {
  [ -e "$PREFIX/lib/tirust-config.json" ] && { log "install: exists"; return; }
  printf 'install tirust --prefix=%s --jobs=%s\nquit\n' "$W/HEPTools" "$JOBS" > "$W/install.mg5"
  py bin/madgraph "$W/install.mg5" > "$W/logs/install-console.log" 2>&1
  grep -q 'TIRust successfully installed' "$W/logs/install-console.log" || { log "install FAILED"; exit 1; }
  log "install: $(py -c "import json;c=json.load(open('$PREFIX/lib/tirust-config.json'));print(c['local_sources']['tirust']['revision'],c['artifact_sha256']['libmadloop_tirust.so'])")"
}

phase_register() {
  [ -e "$REPORT/manifest.json" ] && { log "register: exists"; return; }
  py tools/tirust_benchmark.py register-campaign --campaign "$CAMPAIGN" --tirust-prefix "$PREFIX" \
    --predecessor "$PREDECESSOR" --inherit-physical-cards --reason "${REASON:?set REASON}" > "$W/logs/register.log" 2>&1 \
    || { log "register FAILED"; exit 1; }
  for stage in init model; do
    py tools/tirust_benchmark.py $stage --campaign "$CAMPAIGN" > "$W/logs/$stage.log" 2>&1 || { log "$stage FAILED"; exit 1; }
  done
  log "register: done"
}

phase_build() {
  local b="py tools/tirust_benchmark.py build --campaign $CAMPAIGN --jobs $JOBS --cases"
  # One case per model first, so concurrent generations find warm model caches.
  $b qcd_n1_uux_z ew_n1_udx_wp > "$W/logs/build-g0.log" 2>&1 || { log "build g0 FAILED"; exit 1; }
  $b qcd_n4_uux_zzgg > "$W/logs/build-g1.log" 2>&1 &
  $b ew_n4_epem_mupmumaa > "$W/logs/build-g2.log" 2>&1 &
  $b qcd_n3_uux_zgg ew_n3_epem_ttxa > "$W/logs/build-g3.log" 2>&1 &
  $b qcd_n2_gg_ttx ew_n2_epem_mupmum mixed_uux_ga li_gg_h li_gg_hh > "$W/logs/build-g4.log" 2>&1 &
  wait; log "build: $(grep -l Traceback "$W"/logs/build-g*.log | wc -l) failed groups"
}

numerical_cases() {
  for c in "$@"; do
    [ -e "$REPORT/experiments/numerical/ordinary-$c/results.json" ] && continue
    py tools/tirust_benchmark_numerical.py --campaign "$CAMPAIGN" --case $c --label ordinary-$c --warmup-only \
      > "$W/logs/numerical-$c.log" 2>&1
    log "numerical $c exit $?"
  done
}
phase_numerical() {
  numerical_cases qcd_n4_uux_zzgg &
  numerical_cases ew_n4_epem_mupmumaa &
  numerical_cases qcd_n3_uux_zgg qcd_n2_gg_ttx li_gg_h li_gg_hh qcd_n1_uux_z &
  numerical_cases ew_n3_epem_ttxa ew_n2_epem_mupmum mixed_uux_ga ew_n1_udx_wp &
  wait
}

phase_dd_gate() {
  local G=$W/dd-gate R=$W/replay; mkdir -p "$G"
  gfortran -O2 -fPIC -shared tools/tirust_capture/shim.c -o "$W/libtirust_capture.so"
  [ -d "$R/sources/tirust" ] || py tools/tirust_capture/prepare_replay.py --workspace "$R" --fetch --variant h26 \
      --build --jobs "$JOBS" > "$W/logs/replay-h26.log" 2>&1
  for v in 26 40 105; do
    [ -e "$R/replay-h$v.bin" ] || py tools/tirust_capture/prepare_replay.py --workspace "$R" --sources "$R/sources/tirust" \
      --variant h$v --dd-headroom $v --build --jobs "$JOBS" > "$W/logs/replay-h$v.log" 2>&1
  done
  local cpu=$CPU0
  for c in $CASES; do
    ( [ -e "$G/$c/capture.json" ] || { rm -rf "$G/$c"; py tools/tirust_capture/capture_lane.py --campaign "$CAMPAIGN" \
        --case $c --shim "$W/libtirust_capture.so" --out "$G/$c" --cpu $cpu > "$G/$c.capture.log" 2>&1; }
      [ -e "$G/$c/capture.json" ] || exit 0
      trace=$G/$c/$(py -c "import json,sys;print(json.load(open(sys.argv[1]))['trace'])" "$G/$c/capture.json")
      for v in h26 h40 h105; do
        taskset -c $cpu "$R/replay-$v.bin" "$trace" --calls-jsonl "$G/$c/$v.jsonl" --report-json "$G/$c/$v.json" > "$G/$c/$v.txt" 2>&1
      done
      py tools/tirust_capture/dd_gate.py --production "$G/$c/h26.jsonl" --reference h40="$G/$c/h40.jsonl" \
        --reference h105="$G/$c/h105.jsonl" --json "$G/$c/dd-gate.json" > "$G/$c/dd-gate.txt" 2>&1 ) &
    cpu=$((cpu+1))
  done
  wait; log "dd-gate: $(grep -l 'NOT qualified' "$G"/*/dd-gate.txt 2>/dev/null | wc -l) cases not qualified"
}

stress_lanes() {  # cpu case lane:mode...
  local cpu=$1 c=$2; shift 2
  for item in "$@"; do
    [ -e "$REPORT/experiments/stress/$c/${item%%:*}-${item##*:}" ] && continue
    py tools/tirust_stress.py run --campaign "$CAMPAIGN" --case $c --lane ${item%%:*} --mode ${item##*:} --cpu $cpu \
      >> "$W/logs/stress-$c.log" 2>&1
  done
}
phase_stress() {
  [ -e "$REPORT/stress-points/cms_ew_n2_epem_mupmum-stress.dat" ] || py tools/tirust_cms_stress.py prepare \
    --campaign "$CAMPAIGN" --jobs "$JOBS" > "$W/logs/cms-prepare.log" 2>&1
  local cpu=$CPU0
  for c in $STRESS_CASES; do
    stress_lanes $cpu $c baseline-1:auto candidate-1:auto candidate-6:auto candidate-7:auto candidate-9:auto \
      chain-9671:auto candidate-1:qp candidate-6:qp candidate-9:qp &
    cpu=$((cpu+1))
  done
  # NinjaRS lanes share the unlicensed-Symbolica lock and run one at a time anyway.
  ( for c in $STRESS_CASES; do stress_lanes $cpu $c candidate-8:auto candidate-8:qp; done ) &
  wait; py tools/tirust_stress.py summarize --campaign "$CAMPAIGN" > "$W/logs/stress-summary.log" 2>&1
  log "stress: $(tail -1 "$W/logs/stress-summary.log")"
}

phase_stability() {
  py tools/tirust_stability.py prepare --campaign "$CAMPAIGN"
  local cpu=$CPU0
  for c in qcd_n2_gg_ttx qcd_n3_uux_zgg qcd_n4_uux_zzgg ew_n2_epem_mupmum; do
    ( for v in raw rescued; do for b in 1 6 7 9 8; do
        [ -e "$REPORT/experiments/stability/$v/$c/candidate-$b" ] && continue
        py tools/tirust_stability.py run --campaign "$CAMPAIGN" --case $c --lane candidate-$b --variant $v --cpu $cpu \
          >> "$W/logs/stability-$c.log" 2>&1
      done; done ) &
    cpu=$((cpu+1))
  done
  wait; py tools/tirust_stability.py analyse --campaign "$CAMPAIGN" > "$W/logs/stability-analyse.log" 2>&1; log "stability: done"
}

phase_provenance() {
  local cpu=$CPU0
  for v in default cache-off small-budget; do
    ( for c in $CASES; do
        dir=provenance; [ $v = default ] || dir=provenance-$v
        [ -e "$REPORT/experiments/$dir/$c" ] && continue
        py tools/tirust_provenance.py run --campaign "$CAMPAIGN" --case $c --cpu $cpu --variant $v >> "$W/logs/provenance-$v.log" 2>&1
      done ) &
    cpu=$((cpu+1))
  done
  wait; py tools/tirust_provenance.py summarize --campaign "$CAMPAIGN" > "$W/logs/provenance-summary.log" 2>&1; log "provenance: done"
}

phase_summaries() {
  py tools/tirust_numerical_summary.py --campaign "$CAMPAIGN" --predecessor "$PREDECESSOR" --out "$REPORT/summaries" \
    > "$W/logs/numerical-summary.log" 2>&1
  py tools/tirust_current_report.py collect --campaign "$CAMPAIGN" --dd-gate "$W/dd-gate"
  log "summaries: $(head -1 "$W/logs/numerical-summary.log")"
  log "report preview: copy the previous campaign's report/current-source.typ, then"
  log "  python3 -B tools/tirust_current_report.py render --campaign $CAMPAIGN   (add --publish to replace benchmark.pdf)"
}

for phase in $PHASES; do
  log "phase $phase"
  "phase_${phase//-/_}"
done
