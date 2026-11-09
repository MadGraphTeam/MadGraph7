# Run from the MadGraph checkout. Numerical executions acquire the shared lock.
source tools/tirust_env.sh
python3 -m unittest tests.unit_tests.loop.test_tirust_benchmark_poles -v
python3 tools/tirust_benchmark_poles.py .benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z --label final-qcd-n1-poles-r2 --precision both --loops 1 --fixtures docs/benchmarkfolder/experiments/poles-diagnostics/tensor-cases-r2.json
python3 tools/tirust_benchmark_poles.py .benchmark-work/tirust/outputs/candidate/qcd_n4_uux_zzgg --label final-qcd-n4-poles-r4 --precision both --loops 1
python3 -m unittest tests.unit_tests.loop.test_tirust_benchmark_cache -v
python3 tools/tirust_benchmark_cache.py .benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z --label final-qcd-n1-cache
# No numerical execution is needed to repeat the retained-output analysis.
python3 docs/benchmarkfolder/experiments/poles-diagnostics/analyze-retained.py
