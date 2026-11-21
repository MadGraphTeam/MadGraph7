#!/usr/bin/env bash
# fd_setup.sh <case>: patch the generated output of the case for the FD test
set -e
SP=/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad
source /shared/external/madloop_onia/madgraph7_madloop_onia/tools/tirust_env.sh
D=$(ls -d ${RUNS:-$SP/pw/runs}/$1/SubProcesses/P*/)
cd $D
[ -f onia_loop.f.gen ] || cp onia_loop.f onia_loop.f.gen
cp onia_loop.f.gen onia_loop.f
NB=$(grep -m1 'NBORNAMPS=' onia_loop.f | sed 's/.*NBORNAMPS=\([0-9]*\).*/\1/')
NC=$(grep -m1 'NCOMB=' onia_loop.f | sed 's/.*NCOMB=\([0-9]*\).*/\1/')
python3 $SP/pw/fd_patch.py onia_loop.f $NB $NC
python3 $SP/pw/fd_patch2.py onia_loop.f
python3 $SP/pw/fd_patch3.py onia_loop.f
cp $SP/pw/drivers/pw_fd_driver.f $SP/pw/drivers/pw_fd.mk .
make SHELL=$(command -v bash) -f pw_fd.mk pw_fd_driver > mk_fd.log 2>&1 || { grep -v '^/nix' mk_fd.log | grep -B2 Error | head; exit 1; }
echo built $D
