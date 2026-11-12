      SUBROUTINE ML5_0_HELAS_CALLS_UVCT_1(P,NHEL,H,IC)
C     
C     Modules
C     
      USE ML5_0_POLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=6)
      INTEGER    NCOMB
      PARAMETER (NCOMB=96)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=54)
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=2871, NLOOPGROUPS=1160, NCTAMPS=2982)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=5853)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=227,NLOOPWAVEFUNCS=4000)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      REAL*16     MP__ZERO
      PARAMETER (MP__ZERO=0.0E0_16)
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=1, NSQUAREDSO=1, NAMPSO=2)
C     
C     ARGUMENTS
C     
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER H
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      COMPLEX*16 COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)

      LOGICAL DUMMYFALSE
      DATA DUMMYFALSE/.FALSE./
C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'coupl.inc'
      INCLUDE 'mp_coupl.inc'

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      LOGICAL CHECKPHASE
      LOGICAL HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER SQSO_TARGET
      COMMON/ML5_0_SOCHOICE/SQSO_TARGET

      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE
     $ ,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE
     $ ,CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/ML5_0_SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE
     $ ,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE
     $ ,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

      INTEGER I_SO
      COMMON/ML5_0_I_SO/I_SO
      INTEGER I_LIB
      COMMON/ML5_0_I_LIB/I_LIB

      COMPLEX*16 AMP(NBORNAMPS)
      COMMON/ML5_0_AMPS/AMP
      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      COMPLEX*16 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,
     $ -1:NLOOPWAVEFUNCS)
      COMPLEX*16 PL(0:3,-1:NLOOPWAVEFUNCS)
      COMMON/ML5_0_WL/WL,PL

      COMPLEX*16 AMPL(3,NCTAMPS)
      COMMON/ML5_0_AMPL/AMPL

C     
C     ----------
C     BEGIN CODE
C     ----------

C     The target squared split order contribution is already reached
C      if true.
      IF (FILTER_SO.AND.UVCT_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

C     Amplitude(s) for UVCT diagram with ID 2438
      CALL FFV1_0(W(9),W(8),W(6),GC_5,AMPL(2,2875))
      AMPL(2,2875)=AMPL(2,2875)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2439
      CALL FFV1_0(W(9),W(8),W(6),GC_5,AMPL(1,2876))
      AMPL(1,2876)=AMPL(1,2876)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2440
      CALL FFV1_0(W(10),W(8),W(5),GC_5,AMPL(2,2877))
      AMPL(2,2877)=AMPL(2,2877)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2441
      CALL FFV1_0(W(10),W(8),W(5),GC_5,AMPL(1,2878))
      AMPL(1,2878)=AMPL(1,2878)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2442
      CALL FFV1_0(W(7),W(8),W(11),GC_5,AMPL(2,2879))
      AMPL(2,2879)=AMPL(2,2879)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2443
      CALL FFV1_0(W(7),W(8),W(11),GC_5,AMPL(1,2880))
      AMPL(1,2880)=AMPL(1,2880)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2444
      CALL FFV1_0(W(13),W(12),W(6),GC_5,AMPL(2,2881))
      AMPL(2,2881)=AMPL(2,2881)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2445
      CALL FFV1_0(W(13),W(12),W(6),GC_5,AMPL(1,2882))
      AMPL(1,2882)=AMPL(1,2882)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2446
      CALL FFV1_0(W(10),W(12),W(4),GC_5,AMPL(2,2883))
      AMPL(2,2883)=AMPL(2,2883)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2447
      CALL FFV1_0(W(10),W(12),W(4),GC_5,AMPL(1,2884))
      AMPL(1,2884)=AMPL(1,2884)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2448
      CALL FFV1_0(W(7),W(12),W(14),GC_5,AMPL(2,2885))
      AMPL(2,2885)=AMPL(2,2885)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2449
      CALL FFV1_0(W(7),W(12),W(14),GC_5,AMPL(1,2886))
      AMPL(1,2886)=AMPL(1,2886)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2450
      CALL FFV1_0(W(13),W(15),W(5),GC_5,AMPL(2,2887))
      AMPL(2,2887)=AMPL(2,2887)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2451
      CALL FFV1_0(W(13),W(15),W(5),GC_5,AMPL(1,2888))
      AMPL(1,2888)=AMPL(1,2888)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2452
      CALL FFV1_0(W(9),W(15),W(4),GC_5,AMPL(2,2889))
      AMPL(2,2889)=AMPL(2,2889)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2453
      CALL FFV1_0(W(9),W(15),W(4),GC_5,AMPL(1,2890))
      AMPL(1,2890)=AMPL(1,2890)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2454
      CALL FFV1_0(W(7),W(15),W(16),GC_5,AMPL(2,2891))
      AMPL(2,2891)=AMPL(2,2891)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2455
      CALL FFV1_0(W(7),W(15),W(16),GC_5,AMPL(1,2892))
      AMPL(1,2892)=AMPL(1,2892)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2456
      CALL VVV1_0(W(16),W(6),W(17),GC_4,AMPL(2,2893))
      AMPL(2,2893)=AMPL(2,2893)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2457
      CALL VVV1_0(W(16),W(6),W(17),GC_4,AMPL(1,2894))
      AMPL(1,2894)=AMPL(1,2894)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2458
      CALL FFV1_0(W(10),W(2),W(16),GC_5,AMPL(2,2895))
      AMPL(2,2895)=AMPL(2,2895)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2459
      CALL FFV1_0(W(10),W(2),W(16),GC_5,AMPL(1,2896))
      AMPL(1,2896)=AMPL(1,2896)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2460
      CALL VVV1_0(W(14),W(5),W(17),GC_4,AMPL(2,2897))
      AMPL(2,2897)=AMPL(2,2897)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2461
      CALL VVV1_0(W(14),W(5),W(17),GC_4,AMPL(1,2898))
      AMPL(1,2898)=AMPL(1,2898)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2462
      CALL FFV1_0(W(9),W(2),W(14),GC_5,AMPL(2,2899))
      AMPL(2,2899)=AMPL(2,2899)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2463
      CALL FFV1_0(W(9),W(2),W(14),GC_5,AMPL(1,2900))
      AMPL(1,2900)=AMPL(1,2900)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2464
      CALL VVV1_0(W(4),W(11),W(17),GC_4,AMPL(2,2901))
      AMPL(2,2901)=AMPL(2,2901)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2465
      CALL VVV1_0(W(4),W(11),W(17),GC_4,AMPL(1,2902))
      AMPL(1,2902)=AMPL(1,2902)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2466
      CALL FFV1_0(W(13),W(2),W(11),GC_5,AMPL(2,2903))
      AMPL(2,2903)=AMPL(2,2903)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2467
      CALL FFV1_0(W(13),W(2),W(11),GC_5,AMPL(1,2904))
      AMPL(1,2904)=AMPL(1,2904)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2468
      CALL FFV1_0(W(7),W(2),W(18),GC_5,AMPL(2,2905))
      AMPL(2,2905)=AMPL(2,2905)*(6.0D0*UVWFCT_G_1_1EPS)
      CALL FFV1_0(W(7),W(2),W(19),GC_5,AMPL(2,2906))
      AMPL(2,2906)=AMPL(2,2906)*(6.0D0*UVWFCT_G_1_1EPS)
      CALL FFV1_0(W(7),W(2),W(20),GC_5,AMPL(2,2907))
      AMPL(2,2907)=AMPL(2,2907)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2469
      CALL FFV1_0(W(7),W(2),W(18),GC_5,AMPL(1,2908))
      AMPL(1,2908)=AMPL(1,2908)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
      CALL FFV1_0(W(7),W(2),W(19),GC_5,AMPL(1,2909))
      AMPL(1,2909)=AMPL(1,2909)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
      CALL FFV1_0(W(7),W(2),W(20),GC_5,AMPL(1,2910))
      AMPL(1,2910)=AMPL(1,2910)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2470
      CALL FFV1_0(W(23),W(22),W(6),GC_5,AMPL(2,2911))
      AMPL(2,2911)=AMPL(2,2911)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2471
      CALL FFV1_0(W(23),W(22),W(6),GC_5,AMPL(1,2912))
      AMPL(1,2912)=AMPL(1,2912)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2472
      CALL FFV1_0(W(24),W(22),W(5),GC_5,AMPL(2,2913))
      AMPL(2,2913)=AMPL(2,2913)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2473
      CALL FFV1_0(W(24),W(22),W(5),GC_5,AMPL(1,2914))
      AMPL(1,2914)=AMPL(1,2914)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2474
      CALL FFV1_0(W(21),W(22),W(11),GC_5,AMPL(2,2915))
      AMPL(2,2915)=AMPL(2,2915)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2475
      CALL FFV1_0(W(21),W(22),W(11),GC_5,AMPL(1,2916))
      AMPL(1,2916)=AMPL(1,2916)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2476
      CALL FFV1_0(W(25),W(12),W(6),GC_5,AMPL(2,2917))
      AMPL(2,2917)=AMPL(2,2917)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2477
      CALL FFV1_0(W(25),W(12),W(6),GC_5,AMPL(1,2918))
      AMPL(1,2918)=AMPL(1,2918)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2478
      CALL FFV2_5_0(W(24),W(12),W(3),GC_22,GC_23,AMPL(2,2919))
      AMPL(2,2919)=AMPL(2,2919)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2479
      CALL FFV2_5_0(W(24),W(12),W(3),GC_22,GC_23,AMPL(1,2920))
      AMPL(1,2920)=AMPL(1,2920)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2480
      CALL FFV1_0(W(25),W(15),W(5),GC_5,AMPL(2,2921))
      AMPL(2,2921)=AMPL(2,2921)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2481
      CALL FFV1_0(W(25),W(15),W(5),GC_5,AMPL(1,2922))
      AMPL(1,2922)=AMPL(1,2922)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2482
      CALL FFV2_5_0(W(23),W(15),W(3),GC_22,GC_23,AMPL(2,2923))
      AMPL(2,2923)=AMPL(2,2923)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2483
      CALL FFV2_5_0(W(23),W(15),W(3),GC_22,GC_23,AMPL(1,2924))
      AMPL(1,2924)=AMPL(1,2924)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2484
      CALL FFV1_0(W(25),W(2),W(11),GC_5,AMPL(2,2925))
      AMPL(2,2925)=AMPL(2,2925)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2485
      CALL FFV1_0(W(25),W(2),W(11),GC_5,AMPL(1,2926))
      AMPL(1,2926)=AMPL(1,2926)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2486
      CALL FFV1_0(W(27),W(22),W(6),GC_5,AMPL(2,2927))
      AMPL(2,2927)=AMPL(2,2927)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2487
      CALL FFV1_0(W(27),W(22),W(6),GC_5,AMPL(1,2928))
      AMPL(1,2928)=AMPL(1,2928)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2488
      CALL FFV1_0(W(28),W(22),W(4),GC_5,AMPL(2,2929))
      AMPL(2,2929)=AMPL(2,2929)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2489
      CALL FFV1_0(W(28),W(22),W(4),GC_5,AMPL(1,2930))
      AMPL(1,2930)=AMPL(1,2930)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2490
      CALL FFV1_0(W(26),W(22),W(14),GC_5,AMPL(2,2931))
      AMPL(2,2931)=AMPL(2,2931)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2491
      CALL FFV1_0(W(26),W(22),W(14),GC_5,AMPL(1,2932))
      AMPL(1,2932)=AMPL(1,2932)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2492
      CALL FFV1_0(W(29),W(8),W(6),GC_5,AMPL(2,2933))
      AMPL(2,2933)=AMPL(2,2933)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2493
      CALL FFV1_0(W(29),W(8),W(6),GC_5,AMPL(1,2934))
      AMPL(1,2934)=AMPL(1,2934)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2494
      CALL FFV2_5_0(W(28),W(8),W(3),GC_22,GC_23,AMPL(2,2935))
      AMPL(2,2935)=AMPL(2,2935)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2495
      CALL FFV2_5_0(W(28),W(8),W(3),GC_22,GC_23,AMPL(1,2936))
      AMPL(1,2936)=AMPL(1,2936)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2496
      CALL FFV1_0(W(29),W(15),W(4),GC_5,AMPL(2,2937))
      AMPL(2,2937)=AMPL(2,2937)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2497
      CALL FFV1_0(W(29),W(15),W(4),GC_5,AMPL(1,2938))
      AMPL(1,2938)=AMPL(1,2938)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2498
      CALL FFV2_5_0(W(27),W(15),W(3),GC_22,GC_23,AMPL(2,2939))
      AMPL(2,2939)=AMPL(2,2939)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2499
      CALL FFV2_5_0(W(27),W(15),W(3),GC_22,GC_23,AMPL(1,2940))
      AMPL(1,2940)=AMPL(1,2940)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2500
      CALL FFV1_0(W(29),W(2),W(14),GC_5,AMPL(2,2941))
      AMPL(2,2941)=AMPL(2,2941)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2501
      CALL FFV1_0(W(29),W(2),W(14),GC_5,AMPL(1,2942))
      AMPL(1,2942)=AMPL(1,2942)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2502
      CALL FFV1_0(W(31),W(22),W(5),GC_5,AMPL(2,2943))
      AMPL(2,2943)=AMPL(2,2943)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2503
      CALL FFV1_0(W(31),W(22),W(5),GC_5,AMPL(1,2944))
      AMPL(1,2944)=AMPL(1,2944)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2504
      CALL FFV1_0(W(32),W(22),W(4),GC_5,AMPL(2,2945))
      AMPL(2,2945)=AMPL(2,2945)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2505
      CALL FFV1_0(W(32),W(22),W(4),GC_5,AMPL(1,2946))
      AMPL(1,2946)=AMPL(1,2946)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2506
      CALL FFV1_0(W(30),W(22),W(16),GC_5,AMPL(2,2947))
      AMPL(2,2947)=AMPL(2,2947)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2507
      CALL FFV1_0(W(30),W(22),W(16),GC_5,AMPL(1,2948))
      AMPL(1,2948)=AMPL(1,2948)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2508
      CALL FFV1_0(W(33),W(8),W(5),GC_5,AMPL(2,2949))
      AMPL(2,2949)=AMPL(2,2949)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2509
      CALL FFV1_0(W(33),W(8),W(5),GC_5,AMPL(1,2950))
      AMPL(1,2950)=AMPL(1,2950)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2510
      CALL FFV2_5_0(W(32),W(8),W(3),GC_22,GC_23,AMPL(2,2951))
      AMPL(2,2951)=AMPL(2,2951)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2511
      CALL FFV2_5_0(W(32),W(8),W(3),GC_22,GC_23,AMPL(1,2952))
      AMPL(1,2952)=AMPL(1,2952)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2512
      CALL FFV1_0(W(33),W(12),W(4),GC_5,AMPL(2,2953))
      AMPL(2,2953)=AMPL(2,2953)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2513
      CALL FFV1_0(W(33),W(12),W(4),GC_5,AMPL(1,2954))
      AMPL(1,2954)=AMPL(1,2954)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2514
      CALL FFV2_5_0(W(31),W(12),W(3),GC_22,GC_23,AMPL(2,2955))
      AMPL(2,2955)=AMPL(2,2955)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2515
      CALL FFV2_5_0(W(31),W(12),W(3),GC_22,GC_23,AMPL(1,2956))
      AMPL(1,2956)=AMPL(1,2956)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2516
      CALL FFV1_0(W(33),W(2),W(16),GC_5,AMPL(2,2957))
      AMPL(2,2957)=AMPL(2,2957)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2517
      CALL FFV1_0(W(33),W(2),W(16),GC_5,AMPL(1,2958))
      AMPL(1,2958)=AMPL(1,2958)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2518
      CALL VVV1_0(W(16),W(6),W(34),GC_4,AMPL(2,2959))
      AMPL(2,2959)=AMPL(2,2959)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2519
      CALL VVV1_0(W(16),W(6),W(34),GC_4,AMPL(1,2960))
      AMPL(1,2960)=AMPL(1,2960)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2520
      CALL FFV1_0(W(35),W(22),W(6),GC_5,AMPL(2,2961))
      AMPL(2,2961)=AMPL(2,2961)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2521
      CALL FFV1_0(W(35),W(22),W(6),GC_5,AMPL(1,2962))
      AMPL(1,2962)=AMPL(1,2962)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2522
      CALL VVV1_0(W(14),W(5),W(34),GC_4,AMPL(2,2963))
      AMPL(2,2963)=AMPL(2,2963)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2523
      CALL VVV1_0(W(14),W(5),W(34),GC_4,AMPL(1,2964))
      AMPL(1,2964)=AMPL(1,2964)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2524
      CALL FFV1_0(W(36),W(22),W(5),GC_5,AMPL(2,2965))
      AMPL(2,2965)=AMPL(2,2965)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2525
      CALL FFV1_0(W(36),W(22),W(5),GC_5,AMPL(1,2966))
      AMPL(1,2966)=AMPL(1,2966)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2526
      CALL VVV1_0(W(4),W(11),W(34),GC_4,AMPL(2,2967))
      AMPL(2,2967)=AMPL(2,2967)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2527
      CALL VVV1_0(W(4),W(11),W(34),GC_4,AMPL(1,2968))
      AMPL(1,2968)=AMPL(1,2968)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2528
      CALL FFV1_0(W(37),W(22),W(4),GC_5,AMPL(2,2969))
      AMPL(2,2969)=AMPL(2,2969)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2529
      CALL FFV1_0(W(37),W(22),W(4),GC_5,AMPL(1,2970))
      AMPL(1,2970)=AMPL(1,2970)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2530
      CALL FFV1_0(W(1),W(22),W(18),GC_5,AMPL(2,2971))
      AMPL(2,2971)=AMPL(2,2971)*(6.0D0*UVWFCT_G_1_1EPS)
      CALL FFV1_0(W(1),W(22),W(19),GC_5,AMPL(2,2972))
      AMPL(2,2972)=AMPL(2,2972)*(6.0D0*UVWFCT_G_1_1EPS)
      CALL FFV1_0(W(1),W(22),W(20),GC_5,AMPL(2,2973))
      AMPL(2,2973)=AMPL(2,2973)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2531
      CALL FFV1_0(W(1),W(22),W(18),GC_5,AMPL(1,2974))
      AMPL(1,2974)=AMPL(1,2974)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
      CALL FFV1_0(W(1),W(22),W(19),GC_5,AMPL(1,2975))
      AMPL(1,2975)=AMPL(1,2975)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
      CALL FFV1_0(W(1),W(22),W(20),GC_5,AMPL(1,2976))
      AMPL(1,2976)=AMPL(1,2976)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2532
      CALL FFV2_5_0(W(37),W(8),W(3),GC_22,GC_23,AMPL(2,2977))
      AMPL(2,2977)=AMPL(2,2977)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2533
      CALL FFV2_5_0(W(37),W(8),W(3),GC_22,GC_23,AMPL(1,2978))
      AMPL(1,2978)=AMPL(1,2978)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2534
      CALL FFV2_5_0(W(36),W(12),W(3),GC_22,GC_23,AMPL(2,2979))
      AMPL(2,2979)=AMPL(2,2979)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2535
      CALL FFV2_5_0(W(36),W(12),W(3),GC_22,GC_23,AMPL(1,2980))
      AMPL(1,2980)=AMPL(1,2980)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     Amplitude(s) for UVCT diagram with ID 2536
      CALL FFV2_5_0(W(35),W(15),W(3),GC_22,GC_23,AMPL(2,2981))
      AMPL(2,2981)=AMPL(2,2981)*(6.0D0*UVWFCT_G_1_1EPS)
C     Amplitude(s) for UVCT diagram with ID 2537
      CALL FFV2_5_0(W(35),W(15),W(3),GC_22,GC_23,AMPL(1,2982))
      AMPL(1,2982)=AMPL(1,2982)*(3.0D0*UVWFCT_G_1+3.0D0*UVWFCT_G_2)
C     At this point, all UVCT amps needed for (QCD=8), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 3000

      GOTO 1001
 3000 CONTINUE
      UVCT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

