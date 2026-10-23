      subroutine MadLoopParamReader(filename, printParam) 

      implicit none

      CHARACTER(512) fileName, buff, buff2, mode
      CHARACTER*20 MLReductionLib_str,MLReductionLib_str_save
      CHARACTER*2  MLReductionLib_char
      INTEGER MLRed,i,j,k

      include "MadLoopParams.inc"

      logical printParam, couldRead, paramPrinted, find
      data paramPrinted/.FALSE./
      couldRead=.False.
!     Default parameters

      open(666, file=fileName, err=676, status='OLD', action='READ')
      do
         read(666,*,end=999) buff
         if(index(buff,'#').eq.1) then

           if (buff .eq. '#CTModeInit') then
             read(666,*,end=999) CTModeInit
             if (CTModeInit .lt. 0 .or.
     &           CTModeInit .gt. 6 ) then
               stop 'CTModeInit must be >= 0 and <=6.'
             endif
           
           else if (buff .eq. '#CTModeRun') then
             read(666,*,end=999) CTModeRun
             if (CTModeRun .lt. -1 .or.
     &           CTModeRun .gt. 6 ) then
               stop 'CTModeRun must be >= -1 and <=6.'
             endif 

           else if (buff .eq. '#LoopPoleMode') then
             read(666,*,end=999) LoopPoleMode
             if (LoopPoleMode.lt.-1.or.LoopPoleMode.gt.3) then
               write(*,*) 'LoopPoleMode must be -1, 0, 1, 2 or 3.'
               stop 1
             endif

           else if (buff .eq. '#COLLIERGlobalCache') then
             read(666,*,end=999) COLLIERGlobalCache
             if (COLLIERGlobalCache .lt. -1) then
               stop 'COLLIERGlobalCache must be >= -1'
             endif

           else if (buff .eq. '#NRotations_DP') then
             read(666,*,end=999) NRotations_DP
             if (NRotations_DP .lt. 0 .or.
     &           NRotations_DP .gt. 2 ) then
               stop 'NRotations_DP must be >= 0 and <=2.'
             endif

           else if (buff .eq. '#NRotations_QP') then
             read(666,*,end=999) NRotations_QP
             if (NRotations_QP .lt. 0 .or.
     &           NRotations_QP .gt. 2 ) then
               stop 'NRotations_QP must be >= 0 and <=2.'
             endif

           else if (buff .eq. '#MLStabThres') then
             read(666,*,end=999) MLStabThres
             if (MLStabThres.lt.0.0d0) then
                 stop 'MLStabThres must be >= 0'
             endif

           else if (buff .eq. '#COLLIERRequiredAccuracy') then
             read(666,*,end=999) COLLIERRequiredAccuracy
             if (COLLIERRequiredAccuracy.le.0.0d0.and.
     &           COLLIERRequiredAccuracy.ne.-1.0d0) then
                 stop 'COLLIERRequiredAccuracy must be > 0 or = -1.0'
             endif

           else if (buff .eq. '#CTLoopLibrary') then
             read(666,*,end=999) CTLoopLibrary
             if (CTLoopLibrary.lt.2 .or.
     &           CTLoopLibrary.gt.3) then
                 stop 'CTLoopLibrary must be >= 2 and <=3.'
             endif

           else if (buff .eq. '#CTStabThres') then
             read(666,*,end=999) CTStabThres
             if (CTStabThres.le.0.0d0) then
                 stop 'CTStabThres must be > 0'
             endif

           else if (buff .eq. '#ZeroThres') then
             read(666,*,end=999) ZeroThres
             if (ZeroThres.le.0.0d0) then
                 stop 'ZeroThres must be > 0'
             endif

           else if (buff .eq. '#OSThres') then
              read(666,*,end=999) OSThres
              if (OSThres.le.0.0d0) then
                 stop 'OSThres must be > 0'
              endif

           else if (buff .eq. '#CheckCycle') then
             read(666,*,end=999) CheckCycle
             if (CheckCycle.lt.1) then
                 stop 'CheckCycle must be >= 1'
             endif

           else if (buff .eq. '#MaxAttempts') then
             read(666,*,end=999) MaxAttempts
             if (MaxAttempts.lt.1) then
                 stop 'MaxAttempts must be >= 1'
             endif

          else if (buff .eq. '#COLLIERComputeUVpoles') then
             read(666,*,end=999) COLLIERComputeUVpoles

          else if (buff .eq. '#COLLIERComputeIRpoles') then
             read(666,*,end=999) COLLIERComputeIRpoles

		  else if (buff .eq. '#COLLIERUseInternalStabilityTest') then
             read(666,*,end=999) COLLIERUseInternalStabilityTest

          else if (buff .eq. '#COLLIERUseCacheForPoles') then
             read(666,*,end=999) COLLIERUseCacheForPoles

          else if (buff .eq. '#COLLIERCanOutput') then
             read(666,*,end=999) COLLIERCanOutput

          else if (buff .eq. '#UseLoopFilter') then
             read(666,*,end=999) UseLoopFilter

          else if (buff .eq. '#DoubleCheckHelicityFilter') then
             read(666,*,end=999) DoubleCheckHelicityFilter

          else if (buff .eq. '#LoopInitStartOver') then
             read(666,*,end=999) LoopInitStartOver

          else if (buff .eq. '#HelInitStartOver') then
             read(666,*,end=999) HelInitStartOver

          else if (buff .eq. '#WriteOutFilters') then
             read(666,*,end=999) WriteOutFilters

          else if (buff .eq. '#UseQPIntegrandForNinja') then
             read(666,*,end=999) UseQPIntegrandForNinja

          else if (buff .eq. '#UseQPIntegrandForNinjaRS') then
             read(666,*,end=999) UseQPIntegrandForNinjaRS

          else if (buff .eq. '#UseQPIntegrandForCutTools') then
             read(666,*,end=999) UseQPIntegrandForCutTools

          else if (buff .eq. '#ImprovePSPoint') then
             read(666,*,end=999) ImprovePSPoint
             if (ImprovePSPoint .lt. -1 .or.
     &           ImprovePSPoint .gt. 2 ) then
               stop 'ImprovePSPoint must be >= -1 and <=2.'
             endif

          else if (buff .eq. '#HelicityFilterLevel') then
             read(666,*,end=999) HelicityFilterLevel
             if (HelicityFilterLevel .lt. 0 .or.
     &           HelicityFilterLevel .gt. 2 ) then
               stop 'HelicityFilterLevel must be >= 0 and <=2.'
             endif

          else if (buff .eq. '#MLReductionLib') then
             read(666,*,end=999) MLReductionLib_str
             MLReductionLib(1:9)=0
             MLReductionLib_str_save=MLReductionLib_str
             j=0
             DO
                i=index(MLReductionLib_str,'|')
                IF(i.EQ.0)THEN
                   MLReductionLib_char=MLReductionLib_str
                ELSE
                   MLReductionLib_char=MLReductionLib_str(:i-1)
                ENDIF
                IF(MLReductionLib_char.EQ.'1 ')THEN
                   MLRed=1
                ELSEIF(MLReductionLib_char.EQ.'2 ')THEN
                   MLRed=2
                ELSEIF(MLReductionLib_char.EQ.'3 ')THEN
                   MLRed=3
                ELSEIF(MLReductionLib_char.EQ.'4 ')THEN
                   MLRed=4
                ELSEIF(MLReductionLib_char.EQ.'5 ')THEN
                   MLRed=5
                ELSEIF(MLReductionLib_char.EQ.'6 ')THEN
                   MLRed=6
                ELSEIF(MLReductionLib_char.EQ.'7 ')THEN
                   MLRed=7
                ELSEIF(MLReductionLib_char.EQ.'8 ')THEN
                   MLRed=8
                ELSE
                   PRINT *, 'MLReductionLib is wrong: '//
     $                  TRIM(MLReductionLib_str_save)
                   STOP
                ENDIF
                find=.FALSE.
                DO k=1,j
                   IF(MLReductionLib(k).EQ.MLRed)THEN
                      find=.TRUE.
                      EXIT
                   ENDIF
                ENDDO
                IF(.NOT.find)THEN
                   j=j+1
                   MLReductionLib(j)=MLRed
                ENDIF
                IF(i.EQ.0)THEN
                   EXIT
                ELSE
                   MLReductionLib_str=MLReductionLib_str(i+1:)
                ENDIF
             ENDDO
          else if (buff .eq. '#COLLIERMode') then
             read(666,*,end=999) COLLIERMode
             if (COLLIERMode .lt. 1 .or.
     &            COLLIERMode .gt.3) then
                stop 'COLLIERMode must be >=1 and <=3.'
             endif
          else if (buff .eq. '#IREGIRECY') then
             read(666,*,end=999) IREGIRECY
          else if (buff .eq. '#IREGIMODE') then
             read(666,*,end=999) IREGIMODE
             if (IREGIMODE .lt. 0 .or.
     &            IREGIMODE .gt.2) then
                stop 'IREGIMODE must be >=0 and <=2.'
             endif
          else
             write(*,*) 'The parameter name ',buff(2:),
     &' is not reckognized.'
             stop
           endif

         endif
      enddo
  999 continue
      couldRead=.True.
      goto 998      

  676 continue
      write(*,*) '##E00 Error:: MadLoop parameter file ',fileName,
     &' could not be found or is malformed. Please specify it.'
      stop 
C     Below is the code if one desires to let the code continue with
C     a non existing or malformed parameter file
      write(*,*) '##I01 INFO :: The file ',fileName,' could not be ',
     & ' open or did not contain the necessary information. The ',
     & ' default MadLoop parameters will be used.'
      call DefaultParam()
      goto 999

  998 continue

      if(printParam.and..not.paramPrinted) then
      write(*,*)
     & '==============================================================='
      if (couldRead) then      
        write(*,*) 'INFO: MadLoop read these parameters from '
     &,filename
      else
        write(*,*) 'INFO: MadLoop used the default parameters.'
      endif
      write(*,*)
     & '==============================================================='
      write(*,*) ' > MLReductionLib            = '
     $     //TRIM(MLReductionLib_str_save)
      write(*,*) ' > CTModeRun                 = ',CTModeRun
      write(*,*) ' > LoopPoleMode              = ',LoopPoleMode
      write(*,*) ' > MLStabThres               = ',MLStabThres
      write(*,*) ' > NRotations_DP             = ',NRotations_DP
      write(*,*) ' > NRotations_QP             = ',NRotations_QP
      write(*,*) ' > CTStabThres               = ',CTStabThres
      write(*,*) ' > CTLoopLibrary             = ',CTLoopLibrary
      write(*,*) ' > CTModeInit                = ',CTModeInit
      write(*,*) ' > CheckCycle                = ',CheckCycle
      write(*,*) ' > MaxAttempts               = ',MaxAttempts
      write(*,*) ' > UseLoopFilter             = ',UseLoopFilter
      write(*,*) ' > HelicityFilterLevel       = ',HelicityFilterLevel
      write(*,*) ' > ImprovePSPoint            = ',ImprovePSPoint
      write(*,*) ' > DoubleCheckHelicityFilter = ',
     &DoubleCheckHelicityFilter
      write(*,*) ' > LoopInitStartOver         = ',LoopInitStartOver
      write(*,*) ' > HelInitStartOver          = ',HelInitStartOver
      write(*,*) ' > ZeroThres                 = ',ZeroThres
      write(*,*) ' > OSThres                   = ',OSThres
      write(*,*) ' > WriteOutFilters           = ',WriteOutFilters
      write(*,*) ' > UseQPIntegrandForNinja    = ',
     &UseQPIntegrandForNinja
      write(*,*) ' > UseQPIntegrandForNinjaRS  = ',
     &UseQPIntegrandForNinjaRS
      write(*,*) ' > UseQPIntegrandForCutTools = ',
     &UseQPIntegrandForCutTools
      write(*,*) ' > IREGIMODE                 = ',IREGIMODE
      write(*,*) ' > IREGIRECY                 = ',IREGIRECY
      write(*,*) ' > COLLIERMode               = ',COLLIERMode
      write(*,*) ' > COLLIERRequiredAccuracy   = ',
     $COLLIERRequiredAccuracy
      write(*,*) ' > COLLIERCanOutput          = ',COLLIERCanOutput
      write(*,*) ' > COLLIERComputeUVpoles     = ',COLLIERComputeUVpoles
      write(*,*) ' > COLLIERComputeIRpoles     = ',COLLIERComputeIRpoles
      write(*,*) ' > COLLIERGlobalCache        = ',COLLIERGlobalCache
      write(*,*) ' > COLLIERUseCacheForPoles   = ',
     &COLLIERUseCacheForPoles
      write(*,*) ' > COLLIERUseInternalStabilityTest = ',
     &COLLIERUseInternalStabilityTest
      write(*,*)
     & '==============================================================='
      paramPrinted=.TRUE.
      endif

      close(666)

      end

      subroutine DefaultParam() 

      implicit none
      
      include "MadLoopParams.inc"

      MLReductionLib(1)=6
      MLReductionLib(2)=7
      MLReductionLib(3)=1
      MLReductionLib(4:9)=0
      LoopPoleMode=-1
      IREGIMODE=2
      IREGIRECY=.TRUE.
      COLLIERComputeIRpoles = .TRUE.
      COLLIERComputeUVpoles = .TRUE.
      COLLIERUseCacheForPoles = .FALSE.
      COLLIERCanOutput = .FALSE.
      COLLIERGlobalCache = -1
      COLLIERMode=1
      COLLIERRequiredAccuracy=1.0d-8
      COLLIERUseInternalStabilityTest = .TRUE.
      CTModeInit=0
      CTModeRun=-1
      NRotations_DP=0
      NRotations_QP=0
      MLStabThres=1.0d-3
      CTStabThres=1.0d-2
      CTLoopLibrary=3
      CheckCycle=3
      MaxAttempts=10
      HelicityFilterLevel=2
      UseLoopFilter=.False.
      DoubleCheckHelicityFilter=.True.
      LoopInitStartOver=.False.
      HelInitStartOver=.False.
      WriteOutFilters=.True.
      ZeroThres=1.0d-9
      OSThres=1.0d-13
      ImprovePSPoint=2
      UseQPIntegrandForCutTools=.True.
      UseQPIntegrandForNinja=.True.
      UseQPIntegrandForNinjaRS=.True.

      end

      SUBROUTINE ML_SET_LOOP_POLE_MODE(MODE)
C     Pending requests have the same global scope as /MADLOOP/ and ML_INIT.
C     A setter from any subprocess overrides the card for all subprocesses.
      IMPLICIT NONE
      INTEGER MODE,REQUESTED_LOOP_POLE_MODE
      INCLUDE 'MadLoopParams.inc'
      COMMON/ML_LOOP_POLE_REQUEST/REQUESTED_LOOP_POLE_MODE
      IF(MODE.LT.-1.OR.MODE.GT.3) THEN
        WRITE(*,*) 'LoopPoleMode must be -1, 0, 1, 2 or 3.'
        STOP 1
      ENDIF
      REQUESTED_LOOP_POLE_MODE=MODE
      LoopPoleMode=MODE
      END

      SUBROUTINE ML_APPLY_LOOP_POLE_MODE()
C     Restore the most recent request after the first global card read.
      IMPLICIT NONE
      INTEGER REQUESTED_LOOP_POLE_MODE
      INCLUDE 'MadLoopParams.inc'
      COMMON/ML_LOOP_POLE_REQUEST/REQUESTED_LOOP_POLE_MODE
      IF(REQUESTED_LOOP_POLE_MODE.NE.-2) THEN
        LoopPoleMode=REQUESTED_LOOP_POLE_MODE
      ENDIF
      END

      BLOCK DATA ML_LOOP_POLE_REQUEST_INIT
C     The shared reader is linked once, even in a multi-subprocess library.
      IMPLICIT NONE
      INTEGER REQUESTED_LOOP_POLE_MODE
      COMMON/ML_LOOP_POLE_REQUEST/REQUESTED_LOOP_POLE_MODE
      DATA REQUESTED_LOOP_POLE_MODE/-2/
      END

      SUBROUTINE ML_LOOP_POLE_FLAGS(COMPUTE_UV,COMPUTE_IR)
C     The generic selector affects bare loop integrals only.  The legacy
C     COLLIER switches remain effective when LoopPoleMode is -1.
      IMPLICIT NONE
      LOGICAL COMPUTE_UV,COMPUTE_IR
      INCLUDE 'MadLoopParams.inc'
      COMPUTE_UV=COLLIERComputeUVpoles
      COMPUTE_IR=COLLIERComputeIRpoles
      IF(LoopPoleMode.GE.0) THEN
        COMPUTE_UV=LoopPoleMode.EQ.0.OR.LoopPoleMode.EQ.1
        COMPUTE_IR=LoopPoleMode.EQ.0.OR.LoopPoleMode.EQ.2
      ENDIF
      END

      SUBROUTINE ML_CHECK_LOOP_POLE_MODE()
C     Reject unsupported rescue chains rather than silently returning
C     combined poles after a UV-only or IR-only reduction fails.
      IMPLICIT NONE
      INTEGER I
      INCLUDE 'MadLoopParams.inc'
      IF(LoopPoleMode.LT.-1.OR.LoopPoleMode.GT.3) THEN
        WRITE(*,*) 'LoopPoleMode must be -1, 0, 1, 2 or 3.'
        STOP 1
      ENDIF
      IF(LoopPoleMode.LE.0) RETURN
      DO I=1,9
        IF(MLReductionLib(I).EQ.0) EXIT
        IF(MLReductionLib(I).NE.7.AND.MLReductionLib(I).NE.8) THEN
          WRITE(*,*) 'LoopPoleMode=1, 2 or 3 requires NinjaRS/COLLIER.'
          WRITE(*,*) 'Set MLReductionLib to 8, 7 or 8|7, including',
     &               ' only supported rescue backends.'
          STOP 1
        ENDIF
      ENDDO
      END
