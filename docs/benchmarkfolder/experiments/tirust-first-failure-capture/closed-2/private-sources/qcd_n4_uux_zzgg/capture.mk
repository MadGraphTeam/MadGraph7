include makefile
.PHONY: capture-build
CAPTURE_PROCESS = $(filter-out TIRUST_interface.o TIR_interface.o,$(PROCESS))
capture-build:
	$(FC) $(FFLAGS) -c observer.f90 -o capture-observer-f.o $(LOOP_INCLUDE)
	$(CC) -std=c11 -O2 -fPIC -I. -I$(CAPTURE_ROOT) -c observer.c -o capture-observer-c.o
	$(FC) $(FFLAGS) -c TIRUST_interface.f -o TIRUST_interface.o $(LOOP_INCLUDE)
	$(FC) $(FFLAGS) -c TIR_interface.f -o TIR_interface.o $(LOOP_INCLUDE)
	$(FC) $(FFLAGS) -c check_sa.f -o check_sa.o $(LOOP_INCLUDE)
	$(FC) $(FFLAGS) -o stability-check check_sa.o TIRUST_interface.o TIR_interface.o capture-observer-f.o capture-observer-c.o $(CAPTURE_PROCESS) $(LINKLIBS) -Wl,--wrap=ml_tirust_evaluate_dp,-Map,capture-link.map,--cref
	$(CC) -std=c11 -O2 -fPIC -I. -I$(CAPTURE_ROOT) -c replay.c -o capture-replay.o
	$(FC) $(FFLAGS) -o capture-replay/stability-check capture-replay.o $(LINKLIBS)
