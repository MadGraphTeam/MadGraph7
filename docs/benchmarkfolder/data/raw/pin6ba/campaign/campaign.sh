#!/usr/bin/env bash
# Rerun of the v16all campaign on TIRust 6ba4b53 + the onia_loop bridge with the pin commit cherry-picked (pin6ba).
source /shared/external/madloop_onia/madgraph7_madloop_onia/tools/tirust_env.sh
R=/shared/external/madloop_onia/madgraph7_madloop_onia
RS=$R/docs/benchmarkfolder/research
SP=/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad
W=$SP/v6ba; LIB=$SP/pin6ba/install/lib; LBL='6ba4b53 (PR 557 pin) + onia_loop bridge'
cd $W; mkdir -p pw psi qpf ee bc pol logs
fin() { echo "exit $?" >> "$1"; }
# validation and reducer cases (TIRust only)
( python3 $R/tools/onia_loop_validate.py --work $W/val --libs 9 --npoints 10 --timeout 120 \
    --tirust $LIB --tirust-label "$LBL" --json $W/validate_v16.json > logs/val.log 2>&1; fin logs/val.log ) &
# 2.5.5 comparison and J/psi polarisations
( python3 $W/mg255_v16.py $LIB $W/mg255 > logs/mg255.log 2>&1; fin logs/mg255.log ) &
( python3 $RS/mg255_swave/pol_jpsi.py $W/pol/out $LIB -1,4 $W/pol_v16.json > logs/pol.log 2>&1; fin logs/pol.log ) &
# P-wave checks
for c in aa_chic0 aa_chic2 ag_3p08 ag_3p28 gg_1p18 gg_3p08 gg_3p28 gg_chic0 gg_chic2 \
         ee_1p18g ee_3p08g ee_3p18g ee_3p28g gg_chic0g gu_chic0u uu_chic0g; do
  case $c in ee_*) E=10;; gg_chic0g|gu_chic0u|uu_chic0g) E=20;; *) E=1000;; esac
  ( cd $W/pw && RUNS=$W/pw TIRUST_LIB=$LIB REGEN=1 ENERGY=$E python3 $RS/pwave_fd/pw_check.py $c \
      > $W/logs/pw_$c.log 2>&1; fin $W/logs/pw_$c.log ) &
done
# J/psi chi_cJ
for J in 0 1 2; do
  ( python3 $RS/pwave_fd/psi_chic.py --work $W/psi/J$J --json $W/psi_chic_$J.json --J $J --tirust $LIB \
      --tirust-label "$LBL" > logs/psi$J.log 2>&1 && \
    python3 $RS/pwave_fd/psi_chic.py --work $W/psi/J$J --json $W/psi_chic_exact_$J.json --J $J --exact \
      --modes=-1,4,5 --tirust-label "$LBL" >> logs/psi$J.log 2>&1; fin logs/psi$J.log ) &
done
# QP floor (b, t in the loops)
( python3 $W/gen_out.py $LIB loop_sm_onia-c_mass 'e+ e- > Jpsi(1|3S11) chic2(1|3P21) / z QED=2 [virt=QCD]' \
    $W/qpf/chic2 '{"4": 1.5}' > logs/qpf_chic2.log 2>&1 && \
  python3 $RS/pwave_fd/qp_floor.py $W/qpf/chic2 $W/qp_floor_chic2.json 2 >> logs/qpf_chic2.log 2>&1; fin logs/qpf_chic2.log ) &
( python3 $W/gen_out.py $LIB loop_sm_onia-c_mass 'e+ e- > Jpsi(1|3S11) etac(1|1S01) / z QED=2 [virt=QCD]' \
    $W/qpf/etac '{"4": 1.5}' > logs/qpf_etac.log 2>&1 && \
  python3 $RS/pwave_fd/qp_floor.py $W/qpf/etac $W/qp_floor_etac.json etac >> logs/qpf_etac.log 2>&1; fin logs/qpf_etac.log ) &
# two P-wave states: e+ e- > chic0 hc (qp_check; appendix B table 2 on its own output)
( python3 $W/gen_out.py $LIB loop_sm_onia-c_mass 'e+ e- > chic0(1|3P01) hc(1|1P11) [virt=QCD]' $W/ee/out \
    > logs/ee.log 2>&1 && \
  python3 $RS/two_pwave/qp_check.py $W/ee/out $W/ee_dp.json 7:1 10.58:2 30:1 --modes 1,2 >> logs/ee.log 2>&1 && \
  python3 $RS/two_pwave/qp_check.py $W/ee/out $W/ee_qp.json 10.58:1 --modes 4 >> logs/ee.log 2>&1; fin logs/ee.log ) &
( python3 $W/gen_out.py $LIB loop_sm_onia-c_mass 'e+ e- > chic0(1|3P01) hc(1|1P11) [virt=QCD]' $W/ee/out_t2 \
    > logs/ee_t2.log 2>&1 && \
  python3 $RS/two_pwave/appendix_b.py table2 $W/ee/out_t2 $W/appb_table2_dp.json 1 \
    1.5:6.5:2 1.5:10.58:2 1.5:30.0:2 1.5:100.0:2 1.5:1000.0:2 1.55:10.6:2 >> logs/ee_t2.log 2>&1 && \
  python3 $RS/two_pwave/appendix_b.py table2 $W/ee/out_t2 $W/appb_table2_qp.json 4 \
    1.5:100.0:2 1.5:1000.0:2 1.5:10.58:1 >> logs/ee_t2.log 2>&1; fin logs/ee_t2.log ) &
# unequal constituent masses (B_c P waves)
for spec in 'bc1p1|Bc1L+(1|1P11)' 'bc3p1|Bc1H+(1|3P11)' 'bc1p8|Bc+(1|1P18)'; do
  tag=${spec%%|*}; st=${spec#*|}
  ( python3 $W/gen_out.py $LIB loop_sm_onia-c_mass "e+ e- > $st b c~ / z t t~ QED=2 [virt=QCD]" $W/bc/$tag \
      > logs/bc_$tag.log 2>&1 && \
    python3 $RS/pwave_fd/unequal_mass.py $W/bc/$tag $W/appb_unequal_$tag.json 20:2 30:2 --modes 1,2 \
      >> logs/bc_$tag.log 2>&1; fin logs/bc_$tag.log ) &
done
wait
echo ALLDONE
