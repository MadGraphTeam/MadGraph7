      SUBROUTINE ML5_0_LOOP_CT_CALLS_2(P,NHEL,H,IC)
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
      IF (FILTER_SO.AND.CTCALL_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

C     CutTools call for loop numbers 2532,2626
      CALL ML5_0_LOOP_3(3,6,167,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1001)
C     CutTools call for loop numbers 2533,2627
      CALL ML5_0_LOOP_4(3,6,5,165,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1002)
C     CutTools call for loop numbers 2534,2585
      CALL ML5_0_LOOP_3(3,4,169,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1003)
C     CutTools call for loop numbers 2535,2587
      CALL ML5_0_LOOP_4(3,4,6,168,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1004)
C     CutTools call for loop numbers
C      2536,2548,2549,2550,2559,2567,2629,2635,2636,2637
      CALL ML5_0_LOOP_3(3,6,170,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1005)
C     CutTools call for loop numbers 2537,2630
      CALL ML5_0_LOOP_4(3,6,4,168,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1006)
C     CutTools call for loop numbers
C      2538,2554,2555,2556,2588,2593,2594,2595,2598,2603
      CALL ML5_0_LOOP_3(3,4,172,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1007)
C     CutTools call for loop numbers 2539,2590
      CALL ML5_0_LOOP_4(3,4,5,171,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1008)
C     CutTools call for loop numbers
C      2540,2551,2552,2553,2571,2579,2616,2620,2621,2622
      CALL ML5_0_LOOP_3(3,5,173,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1009)
C     CutTools call for loop numbers 2541,2617
      CALL ML5_0_LOOP_4(3,5,4,171,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1010)
C     CutTools call for loop numbers 2542,2582
      CALL ML5_0_LOOP_5(3,4,6,5,164,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1011)
C     CutTools call for loop numbers 2543,2581
      CALL ML5_0_LOOP_5(3,4,5,6,164,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1012)
C     CutTools call for loop numbers 2544,2609
      CALL ML5_0_LOOP_5(3,5,6,4,164,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1013)
C     CutTools call for loop numbers 2545,2608
      CALL ML5_0_LOOP_5(3,5,4,6,164,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1014)
C     CutTools call for loop numbers 2546,2624
      CALL ML5_0_LOOP_5(3,6,5,4,164,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1015)
C     CutTools call for loop numbers 2547,2623
      CALL ML5_0_LOOP_5(3,6,4,5,164,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1016)
C     CutTools call for loop numbers 2557,2564
      CALL ML5_0_LOOP_3(3,164,65,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1017)
C     CutTools call for loop numbers 2558,2562
      CALL ML5_0_LOOP_4(3,164,6,16,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1018)
C     CutTools call for loop numbers 2560,2561
      CALL ML5_0_LOOP_3(3,16,171,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1019)
C     CutTools call for loop numbers 2563,2566
      CALL ML5_0_LOOP_4(3,6,16,164,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1020)
C     CutTools call for loop numbers 2565,2568
      CALL ML5_0_LOOP_4(3,6,164,16,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1021)
C     CutTools call for loop numbers 2569,2576
      CALL ML5_0_LOOP_3(3,164,66,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1022)
C     CutTools call for loop numbers 2570,2574
      CALL ML5_0_LOOP_4(3,164,5,14,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1023)
C     CutTools call for loop numbers 2572,2573
      CALL ML5_0_LOOP_3(3,14,168,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1024)
C     CutTools call for loop numbers 2575,2578
      CALL ML5_0_LOOP_4(3,5,14,164,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1025)
C     CutTools call for loop numbers 2577,2580
      CALL ML5_0_LOOP_4(3,5,164,14,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1026)
C     CutTools call for loop numbers 2583,2618
      CALL ML5_0_LOOP_5(3,4,164,6,5,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1027)
C     CutTools call for loop numbers 2584,2633
      CALL ML5_0_LOOP_5(3,4,164,5,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1028)
C     CutTools call for loop numbers 2586,2628
      CALL ML5_0_LOOP_4(3,4,168,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1029)
C     CutTools call for loop numbers 2589,2615
      CALL ML5_0_LOOP_4(3,4,171,5,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1030)
C     CutTools call for loop numbers 2591,2613
      CALL ML5_0_LOOP_5(3,4,6,164,5,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1031)
C     CutTools call for loop numbers 2592,2631
      CALL ML5_0_LOOP_5(3,4,5,164,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1032)
C     CutTools call for loop numbers 2596,2604
      CALL ML5_0_LOOP_4(3,4,11,164,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1033)
C     CutTools call for loop numbers 2597,2607
      CALL ML5_0_LOOP_4(3,4,164,11,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1034)
C     CutTools call for loop numbers
C      2599,2606,2638,2639,2640,2641,2642,2643
      CALL ML5_0_LOOP_3(3,164,67,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1035)
C     CutTools call for loop numbers 2600,2605
      CALL ML5_0_LOOP_4(3,164,4,11,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1036)
C     CutTools call for loop numbers 2601,2602
      CALL ML5_0_LOOP_3(3,11,165,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1037)
C     CutTools call for loop numbers 2611,2625
      CALL ML5_0_LOOP_4(3,5,165,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1038)
C     CutTools call for loop numbers 2614,2634
      CALL ML5_0_LOOP_5(3,5,164,4,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1039)
C     CutTools call for loop numbers 2619,2632
      CALL ML5_0_LOOP_5(3,5,4,164,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MT),5,I_SO,1040)
C     CutTools call for loop numbers 2644
      CALL ML5_0_LOOP_2(11,44,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1041)
C     CutTools call for loop numbers 2645,2646
      CALL ML5_0_LOOP_3(5,6,44,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1042)
C     CutTools call for loop numbers 2647
      CALL ML5_0_LOOP_2(14,53,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1043)
C     CutTools call for loop numbers 2648,2649
      CALL ML5_0_LOOP_3(4,6,53,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1044)
C     CutTools call for loop numbers 2650
      CALL ML5_0_LOOP_2(16,62,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1045)
C     CutTools call for loop numbers 2651,2652
      CALL ML5_0_LOOP_3(4,5,62,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1046)
C     CutTools call for loop numbers 2653
      CALL ML5_0_LOOP_2(17,65,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1047)
C     CutTools call for loop numbers 2654
      CALL ML5_0_LOOP_2(16,68,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1048)
C     CutTools call for loop numbers 2655,2656
      CALL ML5_0_LOOP_3(6,16,17,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1049)
C     CutTools call for loop numbers 2657
      CALL ML5_0_LOOP_2(16,70,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1050)
C     CutTools call for loop numbers 2658
      CALL ML5_0_LOOP_2(17,66,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1051)
C     CutTools call for loop numbers 2659
      CALL ML5_0_LOOP_2(14,73,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1052)
C     CutTools call for loop numbers 2660,2661
      CALL ML5_0_LOOP_3(5,14,17,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1053)
C     CutTools call for loop numbers 2662
      CALL ML5_0_LOOP_2(14,75,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1054)
C     CutTools call for loop numbers 2663,2671
      CALL ML5_0_LOOP_3(4,5,68,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1055)
C     CutTools call for loop numbers 2664,2673
      CALL ML5_0_LOOP_4(4,5,6,17,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1056)
C     CutTools call for loop numbers 2665,2676
      CALL ML5_0_LOOP_3(4,6,73,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1057)
C     CutTools call for loop numbers 2666,2677
      CALL ML5_0_LOOP_4(4,6,5,17,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1058)
C     CutTools call for loop numbers 2667,2678
      CALL ML5_0_LOOP_3(4,6,75,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1059)
C     CutTools call for loop numbers 2668,2674
      CALL ML5_0_LOOP_3(4,5,70,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1060)
C     CutTools call for loop numbers 2669,2670
      CALL ML5_0_LOOP_3(4,11,17,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1061)
C     CutTools call for loop numbers 2672,2675
      CALL ML5_0_LOOP_4(4,5,17,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1062)
C     CutTools call for loop numbers 2679,2686,2687,2688
      CALL ML5_0_LOOP_2(17,67,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1063)
C     CutTools call for loop numbers 2680
      CALL ML5_0_LOOP_2(11,79,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1064)
C     CutTools call for loop numbers 2681
      CALL ML5_0_LOOP_2(11,80,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1065)
C     CutTools call for loop numbers 2682,2684
      CALL ML5_0_LOOP_3(5,6,79,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1066)
C     CutTools call for loop numbers 2683,2685
      CALL ML5_0_LOOP_3(5,6,80,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1067)
C     CutTools call for loop numbers 2689
      CALL ML5_0_LOOP_2(11,92,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1068)
C     CutTools call for loop numbers 2690,2691
      CALL ML5_0_LOOP_3(5,6,92,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1069)
C     CutTools call for loop numbers 2692,2693
      CALL ML5_0_LOOP_3(3,6,186,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1070)
C     CutTools call for loop numbers 2694,2695
      CALL ML5_0_LOOP_3(3,5,187,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1071)
C     CutTools call for loop numbers 2696,2704
      CALL ML5_0_LOOP_3(3,5,189,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1072)
C     CutTools call for loop numbers 2697,2706
      CALL ML5_0_LOOP_4(3,5,6,188,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1073)
C     CutTools call for loop numbers 2698,2709
      CALL ML5_0_LOOP_3(3,6,190,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1074)
C     CutTools call for loop numbers 2699,2710
      CALL ML5_0_LOOP_4(3,6,5,188,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1075)
C     CutTools call for loop numbers 2700,2711,2860,2862
      CALL ML5_0_LOOP_3(3,6,191,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1076)
C     CutTools call for loop numbers 2701,2707,2864,2866
      CALL ML5_0_LOOP_3(3,5,192,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1077)
C     CutTools call for loop numbers 2702,2703
      CALL ML5_0_LOOP_3(3,11,188,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1078)
C     CutTools call for loop numbers 2705,2708
      CALL ML5_0_LOOP_4(3,5,188,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1079)
C     CutTools call for loop numbers 2712
      CALL ML5_0_LOOP_2(11,105,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1080)
C     CutTools call for loop numbers 2713,2714
      CALL ML5_0_LOOP_3(5,6,105,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1081)
C     CutTools call for loop numbers 2715
      CALL ML5_0_LOOP_2(14,112,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1082)
C     CutTools call for loop numbers 2716,2717
      CALL ML5_0_LOOP_3(4,6,112,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1083)
C     CutTools call for loop numbers 2718,2719
      CALL ML5_0_LOOP_3(3,6,193,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1084)
C     CutTools call for loop numbers 2720,2721
      CALL ML5_0_LOOP_3(3,4,194,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1085)
C     CutTools call for loop numbers 2722,2730
      CALL ML5_0_LOOP_3(3,4,196,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1086)
C     CutTools call for loop numbers 2723,2732
      CALL ML5_0_LOOP_4(3,4,6,195,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1087)
C     CutTools call for loop numbers 2724,2735
      CALL ML5_0_LOOP_3(3,6,197,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1088)
C     CutTools call for loop numbers 2725,2736
      CALL ML5_0_LOOP_4(3,6,4,195,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1089)
C     CutTools call for loop numbers 2726,2737
      CALL ML5_0_LOOP_3(3,6,198,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1090)
C     CutTools call for loop numbers 2727,2733,2868,2870
      CALL ML5_0_LOOP_3(3,4,199,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1091)
C     CutTools call for loop numbers 2728,2729
      CALL ML5_0_LOOP_3(3,14,195,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1092)
C     CutTools call for loop numbers 2731,2734
      CALL ML5_0_LOOP_4(3,4,195,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1093)
C     CutTools call for loop numbers 2738
      CALL ML5_0_LOOP_2(14,124,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1094)
C     CutTools call for loop numbers 2739,2740
      CALL ML5_0_LOOP_3(4,6,124,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1095)
C     CutTools call for loop numbers 2741
      CALL ML5_0_LOOP_2(16,130,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1096)
C     CutTools call for loop numbers 2742,2743
      CALL ML5_0_LOOP_3(4,5,130,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1097)
C     CutTools call for loop numbers 2744,2745
      CALL ML5_0_LOOP_3(3,5,200,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1098)
C     CutTools call for loop numbers 2746,2747
      CALL ML5_0_LOOP_3(3,4,201,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1099)
C     CutTools call for loop numbers 2748,2756
      CALL ML5_0_LOOP_3(3,4,203,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1100)
C     CutTools call for loop numbers 2749,2758
      CALL ML5_0_LOOP_4(3,4,5,202,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1101)
C     CutTools call for loop numbers 2750,2761
      CALL ML5_0_LOOP_3(3,5,204,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1102)
C     CutTools call for loop numbers 2751,2762
      CALL ML5_0_LOOP_4(3,5,4,202,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1103)
C     CutTools call for loop numbers 2752,2763
      CALL ML5_0_LOOP_3(3,5,205,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1104)
C     CutTools call for loop numbers 2753,2759
      CALL ML5_0_LOOP_3(3,4,206,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1105)
C     CutTools call for loop numbers 2754,2755
      CALL ML5_0_LOOP_3(3,16,202,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1106)
C     CutTools call for loop numbers 2757,2760
      CALL ML5_0_LOOP_4(3,4,202,5,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1107)
C     CutTools call for loop numbers 2764
      CALL ML5_0_LOOP_2(16,141,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1108)
C     CutTools call for loop numbers 2765,2766
      CALL ML5_0_LOOP_3(4,5,141,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1109)
C     CutTools call for loop numbers 2767
      CALL ML5_0_LOOP_2(34,65,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1110)
C     CutTools call for loop numbers 2768
      CALL ML5_0_LOOP_2(16,143,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1111)
C     CutTools call for loop numbers 2769,2770
      CALL ML5_0_LOOP_3(6,16,34,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1112)
C     CutTools call for loop numbers 2771
      CALL ML5_0_LOOP_2(16,145,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1113)
C     CutTools call for loop numbers 2772
      CALL ML5_0_LOOP_2(34,66,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1114)
C     CutTools call for loop numbers 2773
      CALL ML5_0_LOOP_2(14,147,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1115)
C     CutTools call for loop numbers 2774,2775
      CALL ML5_0_LOOP_3(5,14,34,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1116)
C     CutTools call for loop numbers 2776
      CALL ML5_0_LOOP_2(14,149,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1117)
C     CutTools call for loop numbers 2777,2785
      CALL ML5_0_LOOP_3(4,5,143,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1118)
C     CutTools call for loop numbers 2778,2787
      CALL ML5_0_LOOP_4(4,5,6,34,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1119)
C     CutTools call for loop numbers 2779,2790
      CALL ML5_0_LOOP_3(4,6,147,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1120)
C     CutTools call for loop numbers 2780,2791
      CALL ML5_0_LOOP_4(4,6,5,34,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1121)
C     CutTools call for loop numbers 2781,2792
      CALL ML5_0_LOOP_3(4,6,149,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1122)
C     CutTools call for loop numbers 2782,2788
      CALL ML5_0_LOOP_3(4,5,145,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1123)
C     CutTools call for loop numbers 2783,2784
      CALL ML5_0_LOOP_3(4,11,34,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1124)
C     CutTools call for loop numbers 2786,2789
      CALL ML5_0_LOOP_4(4,5,34,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1125)
C     CutTools call for loop numbers 2793,2800,2801,2802
      CALL ML5_0_LOOP_2(34,67,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1126)
C     CutTools call for loop numbers 2794
      CALL ML5_0_LOOP_2(11,151,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1127)
C     CutTools call for loop numbers 2795
      CALL ML5_0_LOOP_2(11,153,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1128)
C     CutTools call for loop numbers 2796,2798
      CALL ML5_0_LOOP_3(5,6,151,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1129)
C     CutTools call for loop numbers 2797,2799
      CALL ML5_0_LOOP_3(5,6,153,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1130)
C     CutTools call for loop numbers 2803,2811
      CALL ML5_0_LOOP_3(3,5,208,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1131)
C     CutTools call for loop numbers 2804,2813
      CALL ML5_0_LOOP_4(3,5,6,207,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1132)
C     CutTools call for loop numbers 2805,2816
      CALL ML5_0_LOOP_3(3,6,209,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1133)
C     CutTools call for loop numbers 2806,2817
      CALL ML5_0_LOOP_4(3,6,5,207,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1134)
C     CutTools call for loop numbers 2807,2818,2861,2863
      CALL ML5_0_LOOP_3(3,6,210,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1135)
C     CutTools call for loop numbers 2808,2814,2865,2867
      CALL ML5_0_LOOP_3(3,5,211,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1136)
C     CutTools call for loop numbers 2809,2810
      CALL ML5_0_LOOP_3(3,11,207,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1137)
C     CutTools call for loop numbers 2812,2815
      CALL ML5_0_LOOP_4(3,5,207,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1138)
C     CutTools call for loop numbers 2819
      CALL ML5_0_LOOP_2(11,159,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1139)
C     CutTools call for loop numbers 2820,2821
      CALL ML5_0_LOOP_3(5,6,159,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1140)
C     CutTools call for loop numbers 2822,2830
      CALL ML5_0_LOOP_3(3,4,213,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1141)
C     CutTools call for loop numbers 2823,2832
      CALL ML5_0_LOOP_4(3,4,6,212,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1142)
C     CutTools call for loop numbers 2824,2835
      CALL ML5_0_LOOP_3(3,6,214,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1143)
C     CutTools call for loop numbers 2825,2836
      CALL ML5_0_LOOP_4(3,6,4,212,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1144)
C     CutTools call for loop numbers 2826,2837
      CALL ML5_0_LOOP_3(3,6,215,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1145)
C     CutTools call for loop numbers 2827,2833,2869,2871
      CALL ML5_0_LOOP_3(3,4,216,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1146)
C     CutTools call for loop numbers 2828,2829
      CALL ML5_0_LOOP_3(3,14,212,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1147)
C     CutTools call for loop numbers 2831,2834
      CALL ML5_0_LOOP_4(3,4,212,6,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1148)
C     CutTools call for loop numbers 2838
      CALL ML5_0_LOOP_2(14,161,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1149)
C     CutTools call for loop numbers 2839,2840
      CALL ML5_0_LOOP_3(4,6,161,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1150)
C     CutTools call for loop numbers 2841,2849
      CALL ML5_0_LOOP_3(3,4,218,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1151)
C     CutTools call for loop numbers 2842,2851
      CALL ML5_0_LOOP_4(3,4,5,217,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1152)
C     CutTools call for loop numbers 2843,2854
      CALL ML5_0_LOOP_3(3,5,219,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1153)
C     CutTools call for loop numbers 2844,2855
      CALL ML5_0_LOOP_4(3,5,4,217,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1154)
C     CutTools call for loop numbers 2845,2856
      CALL ML5_0_LOOP_3(3,5,220,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1155)
C     CutTools call for loop numbers 2846,2852
      CALL ML5_0_LOOP_3(3,4,221,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1156)
C     CutTools call for loop numbers 2847,2848
      CALL ML5_0_LOOP_3(3,16,217,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1157)
C     CutTools call for loop numbers 2850,2853
      CALL ML5_0_LOOP_4(3,4,217,5,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),4,I_SO,1158)
C     CutTools call for loop numbers 2857
      CALL ML5_0_LOOP_2(16,163,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO
     $ ,1159)
C     CutTools call for loop numbers 2858,2859
      CALL ML5_0_LOOP_3(4,5,163,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,1160)
C     At this point, all reductions needed for (QCD=8), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 5000

      GOTO 1001
 5000 CONTINUE
      CTCALL_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

