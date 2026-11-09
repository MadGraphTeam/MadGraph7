include makefile
LINKLIBS := $(subst -lmadloop_tirust,/shared/external/madloop_tirust/madgraph7_madloop_tirust/.tirust-build/ew-threshold-geometry/private-libraries/libmadloop_tirust.so,$(LINKLIBS))
geometry-probe:
	$(FC) $(FFLAGS) -c geometry_probe.f -o geometry_probe.o $(LOOP_INCLUDE)
	$(FC) $(FFLAGS) -o geometry-probe geometry_probe.o $(PROCESS) $(LINKLIBS)
