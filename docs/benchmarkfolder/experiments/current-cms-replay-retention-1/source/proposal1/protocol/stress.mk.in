include makefile
.PHONY: stress-info stress-list stress-objects
stress-info:
	@echo FC $(FC)
	@echo FFLAGS $(FFLAGS)
	@echo LDFLAGS $(LDFLAGS)
	@echo LINKLIBS $(LINKLIBS)
	@echo LOOP_INCLUDE $(LOOP_INCLUDE)
stress-list:
	@echo check_sa.o $(PROCESS)
stress-objects: check_sa.o $(PROCESS)
