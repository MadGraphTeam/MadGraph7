!====================================================================
!
!  Define common block with all general parameters used by MadLoop
!  See their definitions in the file MadLoopParams.dat
!
!====================================================================
! Nine library entries plus the terminating zero used during fallback.
!
	  integer CTModeInit,CTModeRun,CheckCycle,MaxAttempts,
     &CTLoopLibrary,NRotations_DP,NRotations_QP,ImprovePSPoint,
     &MLReductionLib(10),IREGIMODE,HelicityFilterLevel,COLLIERMode,
     &COLLIERGlobalCache,LoopPoleMode,TIRustGlobalCache

      real*8 MLStabThres,CTStabThres,ZeroThres,OSThres,COLLIERRequiredAccuracy

      logical UseLoopFilter,LoopInitStartOver,DoubleCheckHelicityFilter,
     &COLLIERComputeIRpoles,COLLIERComputeUVpoles,COLLIERCanOutput
	  logical HelInitStartOver,IREGIRECY,WriteOutFilters
      logical UseQPIntegrandForNinja, UseQPIntegrandForCutTools
      integer*8 TIRustCacheMaxBytes
      logical TIRustComputeFinite,UseQPIntegrandForTIRust
      logical UseQPIntegrandForNinjaRS
	  logical COLLIERUseCacheForPoles,COLLIERUseInternalStabilityTest

      common /MADLOOP/CTModeInit,CTModeRun,NRotations_DP,NRotations_QP,
     &COLLIERMode,COLLIERGlobalCache,LoopPoleMode,TIRustGlobalCache,
     &ImprovePSPoint,CheckCycle, MaxAttempts,UseLoopFilter,MLStabThres,
     &COLLIERRequiredAccuracy,
     &CTStabThres,CTLoopLibrary,LoopInitStartOver,
     &COLLIERComputeIRpoles,COLLIERComputeUVpoles,COLLIERCanOutput,
     &COLLIERUseCacheForPoles,COLLIERUseInternalStabilityTest,
     &DoubleCheckHelicityFilter,ZeroThres,OSThres,HelInitStartOver,
     &MLReductionLib,IREGIMODE,HelicityFilterLevel,IREGIRECY,
     &WriteOutFilters,UseQPIntegrandForNinja,UseQPIntegrandForCutTools,
     &UseQPIntegrandForNinjaRS,TIRustComputeFinite,
     &UseQPIntegrandForTIRust,TIRustCacheMaxBytes
