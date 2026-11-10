# Exact emitted GNU make target check

The sealed supervisor1 already defines all three targets in run.py lines151–160. Its source and index are unchanged. This additive fixture executes the exact `MAKE_ADDENDUM` with GNU make against a tiny synthetic makefile; no correction or successor is needed.

Three actual bounded commands passed: `make -s --no-print-directory -f stress.mk stress-info`, `make -s --no-print-directory -f stress.mk stress-list`, and `make --always-make -j4 -f stress.mk stress-objects`. The existing owned-process helper adds nonexclusive taskset CPUs12–15 and the pinned shell. The aggregate bound is30seconds, each command at most10seconds. All exitcodes are0 and all owned cleanup receipts have no survivors/errors.

The info target produced FC, FFLAGS, LDFLAGS, LINKLIBS and LOOP_INCLUDE fields with their expected token lists. The list target returned `check_sa.o tensor.o denominator.o`. Object recipes only use shell printf to emit synthetic text; no compiler exists in this fixture. The actual info output then passed through unchanged q.external_inputs with compiler path queries mocked. It correctly bound the synthetic -L archive and -I module paths. These files are test data, never linked or loaded. Raw stdout and effective owned commands are retained under execution/ and result.json.

Actual command: `source tools/tirust_env.sh` then `python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-1-make-fixture/run_fixture.py`, from repository root. The append-only work namespace is `.tirust-build/stress-make-fixture-1`. There were three real GNU make processes; compiler, CLI export, loader, reducer and amplitude counts are zero. Supervisor1 seal dbfa3ca37a05f7f7e7f2f91872902adf249106968314192a273ab804a24ff1bf passed before and after.
