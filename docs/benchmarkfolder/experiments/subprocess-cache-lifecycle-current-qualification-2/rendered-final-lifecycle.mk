include makefile
lifecycle_plain: lifecycle_driver.o observer_plain.o $(OLP_PROCESS) $(LIBS) mg5_citation.o
	$(FC) $(FFLAGS) -o $@ lifecycle_driver.o observer_plain.o $(OLP_PROCESS) mg5_citation.o $(LINKLIBS) $(RPATH_LIBS) -Wl,-Map,lifecycle_plain.map
lifecycle_observed: lifecycle_driver.o observer_observed.o $(OLP_PROCESS) $(LIBS) mg5_citation.o
	$(FC) $(FFLAGS) -o $@ lifecycle_driver.o observer_observed.o $(OLP_PROCESS) mg5_citation.o $(LINKLIBS) $(RPATH_LIBS) -Wl,--wrap=ml_tirust_context_create -Wl,--wrap=ml_tirust_context_destroy -Wl,--wrap=ml_tirust_clear_cache -Wl,--wrap=ml_tirust_register_plan -Wl,--wrap=ml_tirust_evaluate_dp -Wl,--wrap=ml_tirust_evaluate_mp -Wl,-Map,lifecycle_observed.map
observer_plain.o: observer.c tirust.h
	$(CC) -std=c11 -O2 -fno-fast-math -I. -c observer.c -o $@
observer_observed.o: observer.c tirust.h
	$(CC) -std=c11 -O2 -fno-fast-math -DLC_OBSERVED=1 -I. -c observer.c -o $@
lifecycle-objects: lifecycle_driver.o observer_plain.o observer_observed.o $(OLP_PROCESS) $(LIBS) mg5_citation.o
lifecycle-list:
	@echo $(OLP_PROCESS) mg5_citation.o
lifecycle-link-info:
	@echo LIBS $(LIBS)
	@echo LINKLIBS $(LINKLIBS)
	@echo INCLUDES $(LOOP_INCLUDE)
	@echo FC $(FC)
	@echo CC $(CC)
	@echo FFLAGS $(FFLAGS)
	@echo LDFLAGS $(LDFLAGS)

lifecycle_card_smoke: lifecycle_card_smoke.o observer_observed.o $(OLP_PROCESS) $(LIBS) mg5_citation.o
	$(FC) $(FFLAGS) -o $@ lifecycle_card_smoke.o observer_observed.o $(OLP_PROCESS) mg5_citation.o $(LINKLIBS) $(RPATH_LIBS) -Wl,--wrap=ml_tirust_context_create -Wl,--wrap=ml_tirust_context_destroy -Wl,--wrap=ml_tirust_clear_cache -Wl,--wrap=ml_tirust_register_plan -Wl,--wrap=ml_tirust_evaluate_dp -Wl,--wrap=ml_tirust_evaluate_mp -Wl,-Map,lifecycle_card_smoke.map
lifecycle-objects: lifecycle_card_smoke.o
