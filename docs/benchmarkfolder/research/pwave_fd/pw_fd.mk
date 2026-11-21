include makefile
pw_fd_driver: pw_fd_driver.o $(PROCESS) $(LIBS)
	$(FC) $(FFLAGS) -o pw_fd_driver pw_fd_driver.o $(PROCESS) $(LINKLIBS)
