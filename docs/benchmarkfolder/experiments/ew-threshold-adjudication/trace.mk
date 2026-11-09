include makefile
LINKLIBS := $(subst -lmadloop_tirust,/shared/external/madloop_tirust/madgraph7_madloop_tirust/.tirust-build/ew-threshold-adjudication/private-libraries/libmadloop_tirust.so,$(LINKLIBS))
trace-build: check_sa.o TIRUST_interface.o ew_trace.o
	$(FC) $(FFLAGS) -o check check_sa.o $(PROCESS) ew_trace.o $(LINKLIBS)
