include makefile
LINKLIBS := cts_cutroutines_qpinput.o $(subst -lmadloop_tirust,/shared/external/madloop_tirust/madgraph7_madloop_tirust/.tirust-build/ew-threshold-ct-qp-transport/private-libraries/libmadloop_tirust.so,$(LINKLIBS))
ct-transport-build:
	$(FC) $(FFLAGS) -c CT_interface.f -o CT_interface.o $(LOOP_INCLUDE)
	cd qpinput && $(FC) -w -fPIC -I/shared/external/madloop_tirust/madgraph7_madloop_tirust/vendor/CutTools/includects -c cts_cutroutines_qpinput.f90 -o ../cts_cutroutines_qpinput.o
	$(FC) $(FFLAGS) -o check check_sa.o $(PROCESS) $(LINKLIBS)
