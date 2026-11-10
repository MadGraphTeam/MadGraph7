include makefile

adapter-check: check_sa.o
	$(FC) $(FFLAGS) -o check check_sa.o ../../../Source/mg5_citation.f ../libMadLoop.a $(LINKLIBS) -lstdc++
