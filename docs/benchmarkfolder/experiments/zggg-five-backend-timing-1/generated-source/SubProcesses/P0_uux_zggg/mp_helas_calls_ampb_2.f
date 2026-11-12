      SUBROUTINE ML5_0_MP_HELAS_CALLS_AMPB_2(P,NHEL,H,IC)
C     
      USE ML5_0_POLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
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
      REAL*16     ZERO
      PARAMETER (ZERO=0.0E0_16)
      COMPLEX*32     IZERO
      PARAMETER (IZERO=CMPLX(0.0E0_16,0.0E0_16,KIND=16))
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=1, NSQUAREDSO=1, NAMPSO=2)
C     
C     ARGUMENTS
C     
      REAL*16 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER H
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      COMPLEX*32 COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'mp_coupl_same_name.inc'

      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL

      INTEGER SQSO_TARGET
      COMMON/ML5_0_SOCHOICE/SQSO_TARGET

      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE
     $ ,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE
     $ ,CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/ML5_0_SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE
     $ ,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE
     $ ,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

      COMPLEX*32 AMP(NBORNAMPS)
      COMMON/ML5_0_MP_AMPS/AMP
      TYPE(MP_ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/W

      COMPLEX*32 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,
     $ -1:NLOOPWAVEFUNCS)
      COMPLEX*32 PL(0:3,-1:NLOOPWAVEFUNCS)
      COMMON/ML5_0_MP_WL/WL,PL

      COMPLEX*32 AMPL(3,NCTAMPS)
      COMMON/ML5_0_MP_AMPL/AMPL

C     
C     ----------
C     BEGIN CODE
C     ----------

C     The target squared split order contribution is already reached
C      if true.
      IF (FILTER_SO.AND.MP_CT_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

      CALL MP_FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1477))
      CALL MP_FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1478))
      CALL MP_FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1479))
      CALL MP_FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1480))
      CALL MP_FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1481))
      CALL MP_FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1482))
      CALL MP_FFV1_0(W(1),W(101),W(16),UV_GQQG_1EPS,AMPL(2,1483))
      CALL MP_FFV1_0(W(1),W(101),W(16),UV_GQQB,AMPL(1,1484))
      CALL MP_FFV1_0(W(1),W(101),W(16),UV_GQQT,AMPL(1,1485))
C     Counter-term amplitude(s) for loop diagram number 936
      CALL MP_FFV2_5_0(W(1),W(71),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1486)
     $ )
C     Counter-term amplitude(s) for loop diagram number 938
      CALL MP_FFV2_5_0(W(1),W(72),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1487)
     $ )
C     Counter-term amplitude(s) for loop diagram number 943
      CALL MP_FFV1_0(W(1),W(142),W(6),R2_GQQ,AMPL(1,1488))
      CALL MP_FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1489))
      CALL MP_FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1490))
      CALL MP_FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1491))
      CALL MP_FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1492))
      CALL MP_FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1493))
      CALL MP_FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1494))
      CALL MP_FFV1_0(W(1),W(142),W(6),UV_GQQG_1EPS,AMPL(2,1495))
      CALL MP_FFV1_0(W(1),W(142),W(6),UV_GQQB,AMPL(1,1496))
      CALL MP_FFV1_0(W(1),W(142),W(6),UV_GQQT,AMPL(1,1497))
C     Counter-term amplitude(s) for loop diagram number 951
      CALL MP_FFV2_5_0(W(1),W(76),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1498)
     $ )
C     Counter-term amplitude(s) for loop diagram number 953
      CALL MP_FFV2_5_0(W(1),W(77),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1499)
     $ )
C     Counter-term amplitude(s) for loop diagram number 958
      CALL MP_FFV1_0(W(1),W(125),W(5),R2_GQQ,AMPL(1,1500))
      CALL MP_FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1501))
      CALL MP_FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1502))
      CALL MP_FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1503))
      CALL MP_FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1504))
      CALL MP_FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1505))
      CALL MP_FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1506))
      CALL MP_FFV1_0(W(1),W(125),W(5),UV_GQQG_1EPS,AMPL(2,1507))
      CALL MP_FFV1_0(W(1),W(125),W(5),UV_GQQB,AMPL(1,1508))
      CALL MP_FFV1_0(W(1),W(125),W(5),UV_GQQT,AMPL(1,1509))
C     Counter-term amplitude(s) for loop diagram number 981
      CALL MP_FFV1_0(W(1),W(106),W(4),R2_GQQ,AMPL(1,1510))
      CALL MP_FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1511))
      CALL MP_FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1512))
      CALL MP_FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1513))
      CALL MP_FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1514))
      CALL MP_FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1515))
      CALL MP_FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1516))
      CALL MP_FFV1_0(W(1),W(106),W(4),UV_GQQG_1EPS,AMPL(2,1517))
      CALL MP_FFV1_0(W(1),W(106),W(4),UV_GQQB,AMPL(1,1518))
      CALL MP_FFV1_0(W(1),W(106),W(4),UV_GQQT,AMPL(1,1519))
C     Counter-term amplitude(s) for loop diagram number 984
      CALL MP_FFV2_5_0(W(1),W(81),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1520)
     $ )
C     Counter-term amplitude(s) for loop diagram number 986
      CALL MP_FFV2_5_0(W(1),W(82),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1521)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1019
      CALL MP_FFV2_5_0(W(1),W(83),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1522)
     $ )
      CALL MP_FFV2_5_0(W(1),W(84),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1523)
     $ )
      CALL MP_FFV2_5_0(W(1),W(85),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1524)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1030
      CALL MP_R2_GG_1_R2_GG_2_0(W(34),W(65),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1525))
C     Counter-term amplitude(s) for loop diagram number 1031
      CALL MP_VVV1_0(W(16),W(6),W(34),R2_3GG,AMPL(1,1526))
      CALL MP_VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1527))
      CALL MP_VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1528))
      CALL MP_VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1529))
      CALL MP_VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1530))
      CALL MP_VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1531))
      CALL MP_VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1532))
      CALL MP_VVV1_0(W(16),W(6),W(34),UV_3GG_1EPS,AMPL(2,1533))
      CALL MP_VVV1_0(W(16),W(6),W(34),UV_3GB,AMPL(1,1534))
      CALL MP_VVV1_0(W(16),W(6),W(34),UV_3GT,AMPL(1,1535))
      CALL MP_VVV1P0_1(W(6),W(34),GC_4,ZERO,ZERO,W(143))
C     Counter-term amplitude(s) for loop diagram number 1032
      CALL MP_R2_GG_1_R2_GG_2_0(W(16),W(143),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1536))
C     Counter-term amplitude(s) for loop diagram number 1035
      CALL MP_R2_QQ_1_0(W(35),W(86),R2_QQQ,AMPL(1,1537))
      CALL MP_FFV1_2(W(35),W(6),GC_5,ZERO,ZERO,W(144))
C     Counter-term amplitude(s) for loop diagram number 1036
      CALL MP_R2_QQ_1_0(W(144),W(22),R2_QQQ,AMPL(1,1538))
C     Counter-term amplitude(s) for loop diagram number 1037
      CALL MP_FFV1_0(W(35),W(22),W(6),R2_GQQ,AMPL(1,1539))
      CALL MP_FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1540))
      CALL MP_FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1541))
      CALL MP_FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1542))
      CALL MP_FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1543))
      CALL MP_FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1544))
      CALL MP_FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1545))
      CALL MP_FFV1_0(W(35),W(22),W(6),UV_GQQG_1EPS,AMPL(2,1546))
      CALL MP_FFV1_0(W(35),W(22),W(6),UV_GQQB,AMPL(1,1547))
      CALL MP_FFV1_0(W(35),W(22),W(6),UV_GQQT,AMPL(1,1548))
      CALL MP_FFV1P0_3(W(1),W(86),GC_5,ZERO,ZERO,W(145))
C     Counter-term amplitude(s) for loop diagram number 1038
      CALL MP_R2_GG_1_R2_GG_2_0(W(16),W(145),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1549))
      CALL MP_FFV1_2(W(1),W(65),GC_5,ZERO,ZERO,W(146))
C     Counter-term amplitude(s) for loop diagram number 1039
      CALL MP_R2_QQ_1_0(W(146),W(22),R2_QQQ,AMPL(1,1550))
C     Counter-term amplitude(s) for loop diagram number 1042
      CALL MP_R2_GG_1_R2_GG_2_0(W(34),W(66),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1551))
C     Counter-term amplitude(s) for loop diagram number 1043
      CALL MP_VVV1_0(W(14),W(5),W(34),R2_3GG,AMPL(1,1552))
      CALL MP_VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1553))
      CALL MP_VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1554))
      CALL MP_VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1555))
      CALL MP_VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1556))
      CALL MP_VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1557))
      CALL MP_VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1558))
      CALL MP_VVV1_0(W(14),W(5),W(34),UV_3GG_1EPS,AMPL(2,1559))
      CALL MP_VVV1_0(W(14),W(5),W(34),UV_3GB,AMPL(1,1560))
      CALL MP_VVV1_0(W(14),W(5),W(34),UV_3GT,AMPL(1,1561))
      CALL MP_VVV1P0_1(W(5),W(34),GC_4,ZERO,ZERO,W(147))
C     Counter-term amplitude(s) for loop diagram number 1044
      CALL MP_R2_GG_1_R2_GG_2_0(W(14),W(147),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1562))
C     Counter-term amplitude(s) for loop diagram number 1047
      CALL MP_R2_QQ_1_0(W(36),W(88),R2_QQQ,AMPL(1,1563))
      CALL MP_FFV1_2(W(36),W(5),GC_5,ZERO,ZERO,W(148))
C     Counter-term amplitude(s) for loop diagram number 1048
      CALL MP_R2_QQ_1_0(W(148),W(22),R2_QQQ,AMPL(1,1564))
C     Counter-term amplitude(s) for loop diagram number 1049
      CALL MP_FFV1_0(W(36),W(22),W(5),R2_GQQ,AMPL(1,1565))
      CALL MP_FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1566))
      CALL MP_FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1567))
      CALL MP_FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1568))
      CALL MP_FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1569))
      CALL MP_FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1570))
      CALL MP_FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1571))
      CALL MP_FFV1_0(W(36),W(22),W(5),UV_GQQG_1EPS,AMPL(2,1572))
      CALL MP_FFV1_0(W(36),W(22),W(5),UV_GQQB,AMPL(1,1573))
      CALL MP_FFV1_0(W(36),W(22),W(5),UV_GQQT,AMPL(1,1574))
      CALL MP_FFV1P0_3(W(1),W(88),GC_5,ZERO,ZERO,W(149))
C     Counter-term amplitude(s) for loop diagram number 1050
      CALL MP_R2_GG_1_R2_GG_2_0(W(14),W(149),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1575))
      CALL MP_FFV1_2(W(1),W(66),GC_5,ZERO,ZERO,W(150))
C     Counter-term amplitude(s) for loop diagram number 1051
      CALL MP_R2_QQ_1_0(W(150),W(22),R2_QQQ,AMPL(1,1576))
C     Counter-term amplitude(s) for loop diagram number 1054
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_44,AMPL(1
     $ ,1577))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_59,AMPL(1
     $ ,1578))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_44,AMPL(1
     $ ,1579))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_59,AMPL(1
     $ ,1580))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_59,AMPL(1
     $ ,1581))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_59,AMPL(1
     $ ,1582))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_141_51,R2GC_142_53,AMPL(1,1583))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_144_57,R2GC_138_46,R2GC_143_55,AMPL(1,1584))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_138_46,R2GC_144_57,R2GC_143_55,AMPL(1,1585))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_139_47,AMPL(1
     $ ,1586))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_49,AMPL(1
     $ ,1587))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_49,AMPL(1
     $ ,1588))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1589))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1590))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1591))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1592))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1593))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1594))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1595))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1596))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1597))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1598))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1599))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1600))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1601))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1602))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1603))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1604))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1605))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1606))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(34),UV_4GG_1EPS,AMPL(2,1607))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(34),UV_4GG_1EPS,AMPL(2,1608))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(34),UV_4GG_1EPS,AMPL(2,1609))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB,AMPL(1,1610))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB,AMPL(1,1611))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB,AMPL(1,1612))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(34),UV_4GGT,AMPL(1,1613))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(34),UV_4GGT,AMPL(1,1614))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(34),UV_4GGT,AMPL(1,1615))
C     Counter-term amplitude(s) for loop diagram number 1055
      CALL MP_VVV1_0(W(4),W(5),W(143),R2_3GG,AMPL(1,1616))
      CALL MP_VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1617))
      CALL MP_VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1618))
      CALL MP_VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1619))
      CALL MP_VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1620))
      CALL MP_VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1621))
      CALL MP_VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1622))
      CALL MP_VVV1_0(W(4),W(5),W(143),UV_3GG_1EPS,AMPL(2,1623))
      CALL MP_VVV1_0(W(4),W(5),W(143),UV_3GB,AMPL(1,1624))
      CALL MP_VVV1_0(W(4),W(5),W(143),UV_3GT,AMPL(1,1625))
C     Counter-term amplitude(s) for loop diagram number 1058
      CALL MP_VVV1_0(W(4),W(6),W(147),R2_3GG,AMPL(1,1626))
      CALL MP_VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1627))
      CALL MP_VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1628))
      CALL MP_VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1629))
      CALL MP_VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1630))
      CALL MP_VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1631))
      CALL MP_VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1632))
      CALL MP_VVV1_0(W(4),W(6),W(147),UV_3GG_1EPS,AMPL(2,1633))
      CALL MP_VVV1_0(W(4),W(6),W(147),UV_3GB,AMPL(1,1634))
      CALL MP_VVV1_0(W(4),W(6),W(147),UV_3GT,AMPL(1,1635))
C     Counter-term amplitude(s) for loop diagram number 1063
      CALL MP_VVV1_0(W(4),W(6),W(149),R2_3GG,AMPL(1,1636))
      CALL MP_VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1637))
      CALL MP_VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1638))
      CALL MP_VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1639))
      CALL MP_VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1640))
      CALL MP_VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1641))
      CALL MP_VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1642))
      CALL MP_VVV1_0(W(4),W(6),W(149),UV_3GG_1EPS,AMPL(2,1643))
      CALL MP_VVV1_0(W(4),W(6),W(149),UV_3GB,AMPL(1,1644))
      CALL MP_VVV1_0(W(4),W(6),W(149),UV_3GT,AMPL(1,1645))
C     Counter-term amplitude(s) for loop diagram number 1065
      CALL MP_VVV1_0(W(4),W(5),W(145),R2_3GG,AMPL(1,1646))
      CALL MP_VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1647))
      CALL MP_VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1648))
      CALL MP_VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1649))
      CALL MP_VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1650))
      CALL MP_VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1651))
      CALL MP_VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1652))
      CALL MP_VVV1_0(W(4),W(5),W(145),UV_3GG_1EPS,AMPL(2,1653))
      CALL MP_VVV1_0(W(4),W(5),W(145),UV_3GB,AMPL(1,1654))
      CALL MP_VVV1_0(W(4),W(5),W(145),UV_3GT,AMPL(1,1655))
C     Counter-term amplitude(s) for loop diagram number 1067
      CALL MP_VVV1_0(W(4),W(11),W(34),R2_3GG,AMPL(1,1656))
      CALL MP_VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1657))
      CALL MP_VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1658))
      CALL MP_VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1659))
      CALL MP_VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1660))
      CALL MP_VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1661))
      CALL MP_VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1662))
      CALL MP_VVV1_0(W(4),W(11),W(34),UV_3GG_1EPS,AMPL(2,1663))
      CALL MP_VVV1_0(W(4),W(11),W(34),UV_3GB,AMPL(1,1664))
      CALL MP_VVV1_0(W(4),W(11),W(34),UV_3GT,AMPL(1,1665))
C     Counter-term amplitude(s) for loop diagram number 1069
      CALL MP_FFV1_0(W(37),W(22),W(4),R2_GQQ,AMPL(1,1666))
      CALL MP_FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1667))
      CALL MP_FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1668))
      CALL MP_FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1669))
      CALL MP_FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1670))
      CALL MP_FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1671))
      CALL MP_FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1672))
      CALL MP_FFV1_0(W(37),W(22),W(4),UV_GQQG_1EPS,AMPL(2,1673))
      CALL MP_FFV1_0(W(37),W(22),W(4),UV_GQQB,AMPL(1,1674))
      CALL MP_FFV1_0(W(37),W(22),W(4),UV_GQQT,AMPL(1,1675))
C     Counter-term amplitude(s) for loop diagram number 1071
      CALL MP_R2_GG_1_R2_GG_2_0(W(34),W(67),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1676))
      CALL MP_VVV1P0_1(W(4),W(34),GC_4,ZERO,ZERO,W(151))
C     Counter-term amplitude(s) for loop diagram number 1073
      CALL MP_R2_GG_1_R2_GG_2_0(W(11),W(151),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1677))
C     Counter-term amplitude(s) for loop diagram number 1075
      CALL MP_R2_QQ_1_0(W(37),W(108),R2_QQQ,AMPL(1,1678))
      CALL MP_FFV1_2(W(37),W(4),GC_5,ZERO,ZERO,W(152))
C     Counter-term amplitude(s) for loop diagram number 1076
      CALL MP_R2_QQ_1_0(W(152),W(22),R2_QQQ,AMPL(1,1679))
      CALL MP_FFV1P0_3(W(1),W(108),GC_5,ZERO,ZERO,W(153))
C     Counter-term amplitude(s) for loop diagram number 1078
      CALL MP_R2_GG_1_R2_GG_2_0(W(11),W(153),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1680))
      CALL MP_FFV1_2(W(1),W(67),GC_5,ZERO,ZERO,W(154))
C     Counter-term amplitude(s) for loop diagram number 1079
      CALL MP_R2_QQ_1_0(W(154),W(22),R2_QQQ,AMPL(1,1681))
C     Counter-term amplitude(s) for loop diagram number 1081
      CALL MP_VVV1_0(W(5),W(6),W(151),R2_3GG,AMPL(1,1682))
      CALL MP_VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1683))
      CALL MP_VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1684))
      CALL MP_VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1685))
      CALL MP_VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1686))
      CALL MP_VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1687))
      CALL MP_VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1688))
      CALL MP_VVV1_0(W(5),W(6),W(151),UV_3GG_1EPS,AMPL(2,1689))
      CALL MP_VVV1_0(W(5),W(6),W(151),UV_3GB,AMPL(1,1690))
      CALL MP_VVV1_0(W(5),W(6),W(151),UV_3GT,AMPL(1,1691))
C     Counter-term amplitude(s) for loop diagram number 1085
      CALL MP_VVV1_0(W(5),W(6),W(153),R2_3GG,AMPL(1,1692))
      CALL MP_VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1693))
      CALL MP_VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1694))
      CALL MP_VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1695))
      CALL MP_VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1696))
      CALL MP_VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1697))
      CALL MP_VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1698))
      CALL MP_VVV1_0(W(5),W(6),W(153),UV_3GG_1EPS,AMPL(2,1699))
      CALL MP_VVV1_0(W(5),W(6),W(153),UV_3GB,AMPL(1,1700))
      CALL MP_VVV1_0(W(5),W(6),W(153),UV_3GT,AMPL(1,1701))
C     Counter-term amplitude(s) for loop diagram number 1093
      CALL MP_R2_GG_1_R2_GG_2_0(W(34),W(18),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1702))
      CALL MP_R2_GG_1_R2_GG_2_0(W(34),W(19),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1703))
      CALL MP_R2_GG_1_R2_GG_2_0(W(34),W(20),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1704))
      CALL MP_FFV1_2(W(1),W(18),GC_5,ZERO,ZERO,W(155))
      CALL MP_FFV1_2(W(1),W(19),GC_5,ZERO,ZERO,W(156))
      CALL MP_FFV1_2(W(1),W(20),GC_5,ZERO,ZERO,W(157))
C     Counter-term amplitude(s) for loop diagram number 1094
      CALL MP_R2_QQ_1_0(W(155),W(22),R2_QQQ,AMPL(1,1705))
      CALL MP_R2_QQ_1_0(W(156),W(22),R2_QQQ,AMPL(1,1706))
      CALL MP_R2_QQ_1_0(W(157),W(22),R2_QQQ,AMPL(1,1707))
C     Counter-term amplitude(s) for loop diagram number 1104
      CALL MP_R2_QQ_1_0(W(37),W(116),R2_QQQ,AMPL(1,1708))
      CALL MP_FFV2_5_2(W(37),W(3),GC_22,GC_23,ZERO,ZERO,W(158))
C     Counter-term amplitude(s) for loop diagram number 1105
      CALL MP_R2_QQ_1_0(W(158),W(8),R2_QQQ,AMPL(1,1709))
C     Counter-term amplitude(s) for loop diagram number 1106
      CALL MP_FFV2_5_0(W(37),W(8),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1710)
     $ )
      CALL MP_FFV1P0_3(W(1),W(116),GC_5,ZERO,ZERO,W(159))
C     Counter-term amplitude(s) for loop diagram number 1107
      CALL MP_R2_GG_1_R2_GG_2_0(W(11),W(159),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1711))
C     Counter-term amplitude(s) for loop diagram number 1108
      CALL MP_VVV1_0(W(5),W(6),W(159),R2_3GG,AMPL(1,1712))
      CALL MP_VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1713))
      CALL MP_VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1714))
      CALL MP_VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1715))
      CALL MP_VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1716))
      CALL MP_VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1717))
      CALL MP_VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1718))
      CALL MP_VVV1_0(W(5),W(6),W(159),UV_3GG_1EPS,AMPL(2,1719))
      CALL MP_VVV1_0(W(5),W(6),W(159),UV_3GB,AMPL(1,1720))
      CALL MP_VVV1_0(W(5),W(6),W(159),UV_3GT,AMPL(1,1721))
C     Counter-term amplitude(s) for loop diagram number 1112
      CALL MP_R2_QQ_1_0(W(36),W(96),R2_QQQ,AMPL(1,1722))
      CALL MP_FFV2_5_2(W(36),W(3),GC_22,GC_23,ZERO,ZERO,W(160))
C     Counter-term amplitude(s) for loop diagram number 1113
      CALL MP_R2_QQ_1_0(W(160),W(12),R2_QQQ,AMPL(1,1723))
C     Counter-term amplitude(s) for loop diagram number 1114
      CALL MP_FFV2_5_0(W(36),W(12),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1724))
      CALL MP_FFV1P0_3(W(1),W(96),GC_5,ZERO,ZERO,W(161))
C     Counter-term amplitude(s) for loop diagram number 1115
      CALL MP_R2_GG_1_R2_GG_2_0(W(14),W(161),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1725))
C     Counter-term amplitude(s) for loop diagram number 1116
      CALL MP_VVV1_0(W(4),W(6),W(161),R2_3GG,AMPL(1,1726))
      CALL MP_VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1727))
      CALL MP_VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1728))
      CALL MP_VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1729))
      CALL MP_VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1730))
      CALL MP_VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1731))
      CALL MP_VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1732))
      CALL MP_VVV1_0(W(4),W(6),W(161),UV_3GG_1EPS,AMPL(2,1733))
      CALL MP_VVV1_0(W(4),W(6),W(161),UV_3GB,AMPL(1,1734))
      CALL MP_VVV1_0(W(4),W(6),W(161),UV_3GT,AMPL(1,1735))
C     Counter-term amplitude(s) for loop diagram number 1120
      CALL MP_R2_QQ_1_0(W(35),W(101),R2_QQQ,AMPL(1,1736))
      CALL MP_FFV2_5_2(W(35),W(3),GC_22,GC_23,ZERO,ZERO,W(162))
C     Counter-term amplitude(s) for loop diagram number 1121
      CALL MP_R2_QQ_1_0(W(162),W(15),R2_QQQ,AMPL(1,1737))
C     Counter-term amplitude(s) for loop diagram number 1122
      CALL MP_FFV2_5_0(W(35),W(15),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1738))
      CALL MP_FFV1P0_3(W(1),W(101),GC_5,ZERO,ZERO,W(163))
C     Counter-term amplitude(s) for loop diagram number 1123
      CALL MP_R2_GG_1_R2_GG_2_0(W(16),W(163),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1739))
C     Counter-term amplitude(s) for loop diagram number 1124
      CALL MP_VVV1_0(W(4),W(5),W(163),R2_3GG,AMPL(1,1740))
      CALL MP_VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1741))
      CALL MP_VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1742))
      CALL MP_VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1743))
      CALL MP_VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1744))
      CALL MP_VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1745))
      CALL MP_VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1746))
      CALL MP_VVV1_0(W(4),W(5),W(163),UV_3GG_1EPS,AMPL(2,1747))
      CALL MP_VVV1_0(W(4),W(5),W(163),UV_3GB,AMPL(1,1748))
      CALL MP_VVV1_0(W(4),W(5),W(163),UV_3GT,AMPL(1,1749))
C     Counter-term amplitude(s) for loop diagram number 1128
      CALL MP_FFV2_5_0(W(144),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1750))
C     Counter-term amplitude(s) for loop diagram number 1130
      CALL MP_FFV1_0(W(162),W(2),W(6),R2_GQQ,AMPL(1,1751))
      CALL MP_FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1752))
      CALL MP_FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1753))
      CALL MP_FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1754))
      CALL MP_FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1755))
      CALL MP_FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1756))
      CALL MP_FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1757))
      CALL MP_FFV1_0(W(162),W(2),W(6),UV_GQQG_1EPS,AMPL(2,1758))
      CALL MP_FFV1_0(W(162),W(2),W(6),UV_GQQB,AMPL(1,1759))
      CALL MP_FFV1_0(W(162),W(2),W(6),UV_GQQT,AMPL(1,1760))
C     Counter-term amplitude(s) for loop diagram number 1132
      CALL MP_FFV2_5_0(W(146),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1761))
C     Counter-term amplitude(s) for loop diagram number 1135
      CALL MP_FFV2_5_0(W(148),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1762))
C     Counter-term amplitude(s) for loop diagram number 1137
      CALL MP_FFV1_0(W(160),W(2),W(5),R2_GQQ,AMPL(1,1763))
      CALL MP_FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1764))
      CALL MP_FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1765))
      CALL MP_FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1766))
      CALL MP_FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1767))
      CALL MP_FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1768))
      CALL MP_FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1769))
      CALL MP_FFV1_0(W(160),W(2),W(5),UV_GQQG_1EPS,AMPL(2,1770))
      CALL MP_FFV1_0(W(160),W(2),W(5),UV_GQQB,AMPL(1,1771))
      CALL MP_FFV1_0(W(160),W(2),W(5),UV_GQQT,AMPL(1,1772))
C     Counter-term amplitude(s) for loop diagram number 1139
      CALL MP_FFV2_5_0(W(150),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1773))
C     Counter-term amplitude(s) for loop diagram number 1143
      CALL MP_FFV1_0(W(158),W(2),W(4),R2_GQQ,AMPL(1,1774))
      CALL MP_FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1775))
      CALL MP_FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1776))
      CALL MP_FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1777))
      CALL MP_FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1778))
      CALL MP_FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1779))
      CALL MP_FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1780))
      CALL MP_FFV1_0(W(158),W(2),W(4),UV_GQQG_1EPS,AMPL(2,1781))
      CALL MP_FFV1_0(W(158),W(2),W(4),UV_GQQB,AMPL(1,1782))
      CALL MP_FFV1_0(W(158),W(2),W(4),UV_GQQT,AMPL(1,1783))
C     Counter-term amplitude(s) for loop diagram number 1144
      CALL MP_FFV2_5_0(W(152),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1784))
C     Counter-term amplitude(s) for loop diagram number 1148
      CALL MP_FFV2_5_0(W(154),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1785))
C     Counter-term amplitude(s) for loop diagram number 1149
      CALL MP_FFV2_5_0(W(155),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1786))
      CALL MP_FFV2_5_0(W(156),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1787))
      CALL MP_FFV2_5_0(W(157),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1788))
      CALL MP_FFV1P0_3(W(1),W(2),GC_5,ZERO,ZERO,W(164))
      CALL MP_VVV1P0_1(W(4),W(164),GC_4,ZERO,ZERO,W(165))
      CALL MP_VVV1P0_1(W(6),W(165),GC_4,ZERO,ZERO,W(166))
C     Counter-term amplitude(s) for loop diagram number 1254
      CALL MP_R2_GGZ_0(W(5),W(166),W(3),R2_GGZDOWN,AMPL(1,1789))
      CALL MP_R2_GGZ_0(W(5),W(166),W(3),R2_GGZDOWN,AMPL(1,1790))
C     Counter-term amplitude(s) for loop diagram number 1255
      CALL MP_R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1791))
      CALL MP_R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1792))
      CALL MP_R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1793))
      CALL MP_R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1794))
      CALL MP_VVV1P0_1(W(5),W(165),GC_4,ZERO,ZERO,W(167))
C     Counter-term amplitude(s) for loop diagram number 1256
      CALL MP_R2_GGZ_0(W(6),W(167),W(3),R2_GGZDOWN,AMPL(1,1795))
      CALL MP_R2_GGZ_0(W(6),W(167),W(3),R2_GGZDOWN,AMPL(1,1796))
      CALL MP_VVV1P0_1(W(5),W(164),GC_4,ZERO,ZERO,W(168))
      CALL MP_VVV1P0_1(W(6),W(168),GC_4,ZERO,ZERO,W(169))
C     Counter-term amplitude(s) for loop diagram number 1258
      CALL MP_R2_GGZ_0(W(4),W(169),W(3),R2_GGZDOWN,AMPL(1,1797))
      CALL MP_R2_GGZ_0(W(4),W(169),W(3),R2_GGZDOWN,AMPL(1,1798))
C     Counter-term amplitude(s) for loop diagram number 1259
      CALL MP_R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1799))
      CALL MP_R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1800))
      CALL MP_R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1801))
      CALL MP_R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1802))
      CALL MP_VVV1P0_1(W(4),W(168),GC_4,ZERO,ZERO,W(170))
C     Counter-term amplitude(s) for loop diagram number 1260
      CALL MP_R2_GGZ_0(W(6),W(170),W(3),R2_GGZDOWN,AMPL(1,1803))
      CALL MP_R2_GGZ_0(W(6),W(170),W(3),R2_GGZDOWN,AMPL(1,1804))
      CALL MP_VVV1P0_1(W(6),W(164),GC_4,ZERO,ZERO,W(171))
      CALL MP_VVV1P0_1(W(5),W(171),GC_4,ZERO,ZERO,W(172))
C     Counter-term amplitude(s) for loop diagram number 1262
      CALL MP_R2_GGZ_0(W(4),W(172),W(3),R2_GGZDOWN,AMPL(1,1805))
      CALL MP_R2_GGZ_0(W(4),W(172),W(3),R2_GGZDOWN,AMPL(1,1806))
C     Counter-term amplitude(s) for loop diagram number 1263
      CALL MP_R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1807))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1808))
      CALL MP_R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1809))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1810))
      CALL MP_VVV1P0_1(W(4),W(171),GC_4,ZERO,ZERO,W(173))
C     Counter-term amplitude(s) for loop diagram number 1264
      CALL MP_R2_GGZ_0(W(5),W(173),W(3),R2_GGZDOWN,AMPL(1,1811))
      CALL MP_R2_GGZ_0(W(5),W(173),W(3),R2_GGZDOWN,AMPL(1,1812))
      CALL MP_VVVV1P0_1(W(4),W(5),W(164),GC_6,ZERO,ZERO,W(174))
      CALL MP_VVVV3P0_1(W(4),W(5),W(164),GC_6,ZERO,ZERO,W(175))
      CALL MP_VVVV4P0_1(W(4),W(5),W(164),GC_6,ZERO,ZERO,W(176))
C     Counter-term amplitude(s) for loop diagram number 1272
      CALL MP_R2_GGZ_0(W(6),W(174),W(3),R2_GGZDOWN,AMPL(1,1813))
      CALL MP_R2_GGZ_0(W(6),W(174),W(3),R2_GGZDOWN,AMPL(1,1814))
      CALL MP_R2_GGZ_0(W(6),W(175),W(3),R2_GGZDOWN,AMPL(1,1815))
      CALL MP_R2_GGZ_0(W(6),W(175),W(3),R2_GGZDOWN,AMPL(1,1816))
      CALL MP_R2_GGZ_0(W(6),W(176),W(3),R2_GGZDOWN,AMPL(1,1817))
      CALL MP_R2_GGZ_0(W(6),W(176),W(3),R2_GGZDOWN,AMPL(1,1818))
      CALL MP_VVVV1P0_1(W(4),W(6),W(164),GC_6,ZERO,ZERO,W(177))
      CALL MP_VVVV3P0_1(W(4),W(6),W(164),GC_6,ZERO,ZERO,W(178))
      CALL MP_VVVV4P0_1(W(4),W(6),W(164),GC_6,ZERO,ZERO,W(179))
C     Counter-term amplitude(s) for loop diagram number 1273
      CALL MP_R2_GGZ_0(W(5),W(177),W(3),R2_GGZDOWN,AMPL(1,1819))
      CALL MP_R2_GGZ_0(W(5),W(177),W(3),R2_GGZDOWN,AMPL(1,1820))
      CALL MP_R2_GGZ_0(W(5),W(178),W(3),R2_GGZDOWN,AMPL(1,1821))
      CALL MP_R2_GGZ_0(W(5),W(178),W(3),R2_GGZDOWN,AMPL(1,1822))
      CALL MP_R2_GGZ_0(W(5),W(179),W(3),R2_GGZDOWN,AMPL(1,1823))
      CALL MP_R2_GGZ_0(W(5),W(179),W(3),R2_GGZDOWN,AMPL(1,1824))
      CALL MP_VVVV1P0_1(W(5),W(6),W(164),GC_6,ZERO,ZERO,W(180))
      CALL MP_VVVV3P0_1(W(5),W(6),W(164),GC_6,ZERO,ZERO,W(181))
      CALL MP_VVVV4P0_1(W(5),W(6),W(164),GC_6,ZERO,ZERO,W(182))
C     Counter-term amplitude(s) for loop diagram number 1274
      CALL MP_R2_GGZ_0(W(4),W(180),W(3),R2_GGZDOWN,AMPL(1,1825))
      CALL MP_R2_GGZ_0(W(4),W(180),W(3),R2_GGZDOWN,AMPL(1,1826))
      CALL MP_R2_GGZ_0(W(4),W(181),W(3),R2_GGZDOWN,AMPL(1,1827))
      CALL MP_R2_GGZ_0(W(4),W(181),W(3),R2_GGZDOWN,AMPL(1,1828))
      CALL MP_R2_GGZ_0(W(4),W(182),W(3),R2_GGZDOWN,AMPL(1,1829))
      CALL MP_R2_GGZ_0(W(4),W(182),W(3),R2_GGZDOWN,AMPL(1,1830))
C     Counter-term amplitude(s) for loop diagram number 1275
      CALL MP_R2_GGZ_0(W(164),W(65),W(3),R2_GGZDOWN,AMPL(1,1831))
      CALL MP_R2_GGZ_0(W(164),W(65),W(3),R2_GGZDOWN,AMPL(1,1832))
C     Counter-term amplitude(s) for loop diagram number 1276
      CALL MP_R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1833))
      CALL MP_R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALDOWN
     $ ,AMPL(1,1834))
      CALL MP_R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1835))
      CALL MP_R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALDOWN
     $ ,AMPL(1,1836))
      CALL MP_VVV1P0_1(W(16),W(164),GC_4,ZERO,ZERO,W(183))
C     Counter-term amplitude(s) for loop diagram number 1277
      CALL MP_R2_GGZ_0(W(6),W(183),W(3),R2_GGZDOWN,AMPL(1,1837))
      CALL MP_R2_GGZ_0(W(6),W(183),W(3),R2_GGZDOWN,AMPL(1,1838))
C     Counter-term amplitude(s) for loop diagram number 1278
      CALL MP_R2_GGZ_0(W(16),W(171),W(3),R2_GGZDOWN,AMPL(1,1839))
      CALL MP_R2_GGZ_0(W(16),W(171),W(3),R2_GGZDOWN,AMPL(1,1840))
C     Counter-term amplitude(s) for loop diagram number 1287
      CALL MP_R2_GGZ_0(W(164),W(66),W(3),R2_GGZDOWN,AMPL(1,1841))
      CALL MP_R2_GGZ_0(W(164),W(66),W(3),R2_GGZDOWN,AMPL(1,1842))
C     Counter-term amplitude(s) for loop diagram number 1288
      CALL MP_R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1843))
      CALL MP_R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALDOWN
     $ ,AMPL(1,1844))
      CALL MP_R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1845))
      CALL MP_R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALDOWN
     $ ,AMPL(1,1846))
      CALL MP_VVV1P0_1(W(14),W(164),GC_4,ZERO,ZERO,W(184))
C     Counter-term amplitude(s) for loop diagram number 1289
      CALL MP_R2_GGZ_0(W(5),W(184),W(3),R2_GGZDOWN,AMPL(1,1847))
      CALL MP_R2_GGZ_0(W(5),W(184),W(3),R2_GGZDOWN,AMPL(1,1848))
C     Counter-term amplitude(s) for loop diagram number 1290
      CALL MP_R2_GGZ_0(W(14),W(168),W(3),R2_GGZDOWN,AMPL(1,1849))
      CALL MP_R2_GGZ_0(W(14),W(168),W(3),R2_GGZDOWN,AMPL(1,1850))
C     Counter-term amplitude(s) for loop diagram number 1312
      CALL MP_R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1851))
      CALL MP_R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALDOWN
     $ ,AMPL(1,1852))
      CALL MP_R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,1853))
      CALL MP_R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALDOWN
     $ ,AMPL(1,1854))
      CALL MP_VVV1P0_1(W(11),W(164),GC_4,ZERO,ZERO,W(185))
C     Counter-term amplitude(s) for loop diagram number 1314
      CALL MP_R2_GGZ_0(W(4),W(185),W(3),R2_GGZDOWN,AMPL(1,1855))
      CALL MP_R2_GGZ_0(W(4),W(185),W(3),R2_GGZDOWN,AMPL(1,1856))
C     Counter-term amplitude(s) for loop diagram number 1315
      CALL MP_R2_GGZ_0(W(164),W(67),W(3),R2_GGZDOWN,AMPL(1,1857))
      CALL MP_R2_GGZ_0(W(164),W(67),W(3),R2_GGZDOWN,AMPL(1,1858))
C     Counter-term amplitude(s) for loop diagram number 1317
      CALL MP_R2_GGZ_0(W(11),W(165),W(3),R2_GGZDOWN,AMPL(1,1859))
      CALL MP_R2_GGZ_0(W(11),W(165),W(3),R2_GGZDOWN,AMPL(1,1860))
C     Counter-term amplitude(s) for loop diagram number 1350
      CALL MP_R2_GGZ_0(W(164),W(18),W(3),R2_GGZDOWN,AMPL(1,1861))
      CALL MP_R2_GGZ_0(W(164),W(18),W(3),R2_GGZDOWN,AMPL(1,1862))
      CALL MP_R2_GGZ_0(W(164),W(19),W(3),R2_GGZDOWN,AMPL(1,1863))
      CALL MP_R2_GGZ_0(W(164),W(19),W(3),R2_GGZDOWN,AMPL(1,1864))
      CALL MP_R2_GGZ_0(W(164),W(20),W(3),R2_GGZDOWN,AMPL(1,1865))
      CALL MP_R2_GGZ_0(W(164),W(20),W(3),R2_GGZDOWN,AMPL(1,1866))
C     Counter-term amplitude(s) for loop diagram number 1352
      CALL MP_R2_GG_1_0(W(11),W(44),R2_GGQ,AMPL(1,1867))
      CALL MP_R2_GG_1_0(W(11),W(44),R2_GGQ,AMPL(1,1868))
      CALL MP_R2_GG_1_0(W(11),W(44),R2_GGQ,AMPL(1,1869))
      CALL MP_R2_GG_1_0(W(11),W(44),R2_GGQ,AMPL(1,1870))
C     Counter-term amplitude(s) for loop diagram number 1353
      CALL MP_VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,1871))
      CALL MP_VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,1872))
      CALL MP_VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,1873))
      CALL MP_VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,1874))
C     Counter-term amplitude(s) for loop diagram number 1355
      CALL MP_R2_GG_1_0(W(14),W(53),R2_GGQ,AMPL(1,1875))
      CALL MP_R2_GG_1_0(W(14),W(53),R2_GGQ,AMPL(1,1876))
      CALL MP_R2_GG_1_0(W(14),W(53),R2_GGQ,AMPL(1,1877))
      CALL MP_R2_GG_1_0(W(14),W(53),R2_GGQ,AMPL(1,1878))
C     Counter-term amplitude(s) for loop diagram number 1356
      CALL MP_VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,1879))
      CALL MP_VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,1880))
      CALL MP_VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,1881))
      CALL MP_VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,1882))
C     Counter-term amplitude(s) for loop diagram number 1358
      CALL MP_R2_GG_1_0(W(16),W(62),R2_GGQ,AMPL(1,1883))
      CALL MP_R2_GG_1_0(W(16),W(62),R2_GGQ,AMPL(1,1884))
      CALL MP_R2_GG_1_0(W(16),W(62),R2_GGQ,AMPL(1,1885))
      CALL MP_R2_GG_1_0(W(16),W(62),R2_GGQ,AMPL(1,1886))
C     Counter-term amplitude(s) for loop diagram number 1359
      CALL MP_VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,1887))
      CALL MP_VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,1888))
      CALL MP_VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,1889))
      CALL MP_VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,1890))
C     Counter-term amplitude(s) for loop diagram number 1361
      CALL MP_R2_GG_1_0(W(17),W(65),R2_GGQ,AMPL(1,1891))
      CALL MP_R2_GG_1_0(W(17),W(65),R2_GGQ,AMPL(1,1892))
      CALL MP_R2_GG_1_0(W(17),W(65),R2_GGQ,AMPL(1,1893))
      CALL MP_R2_GG_1_0(W(17),W(65),R2_GGQ,AMPL(1,1894))
C     Counter-term amplitude(s) for loop diagram number 1362
      CALL MP_R2_GG_1_0(W(16),W(68),R2_GGQ,AMPL(1,1895))
      CALL MP_R2_GG_1_0(W(16),W(68),R2_GGQ,AMPL(1,1896))
      CALL MP_R2_GG_1_0(W(16),W(68),R2_GGQ,AMPL(1,1897))
      CALL MP_R2_GG_1_0(W(16),W(68),R2_GGQ,AMPL(1,1898))
C     Counter-term amplitude(s) for loop diagram number 1363
      CALL MP_VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,1899))
      CALL MP_VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,1900))
      CALL MP_VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,1901))
      CALL MP_VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,1902))
C     Counter-term amplitude(s) for loop diagram number 1365
      CALL MP_R2_GG_1_0(W(16),W(70),R2_GGQ,AMPL(1,1903))
      CALL MP_R2_GG_1_0(W(16),W(70),R2_GGQ,AMPL(1,1904))
      CALL MP_R2_GG_1_0(W(16),W(70),R2_GGQ,AMPL(1,1905))
      CALL MP_R2_GG_1_0(W(16),W(70),R2_GGQ,AMPL(1,1906))
C     Counter-term amplitude(s) for loop diagram number 1366
      CALL MP_R2_GG_1_0(W(17),W(66),R2_GGQ,AMPL(1,1907))
      CALL MP_R2_GG_1_0(W(17),W(66),R2_GGQ,AMPL(1,1908))
      CALL MP_R2_GG_1_0(W(17),W(66),R2_GGQ,AMPL(1,1909))
      CALL MP_R2_GG_1_0(W(17),W(66),R2_GGQ,AMPL(1,1910))
C     Counter-term amplitude(s) for loop diagram number 1367
      CALL MP_R2_GG_1_0(W(14),W(73),R2_GGQ,AMPL(1,1911))
      CALL MP_R2_GG_1_0(W(14),W(73),R2_GGQ,AMPL(1,1912))
      CALL MP_R2_GG_1_0(W(14),W(73),R2_GGQ,AMPL(1,1913))
      CALL MP_R2_GG_1_0(W(14),W(73),R2_GGQ,AMPL(1,1914))
C     Counter-term amplitude(s) for loop diagram number 1368
      CALL MP_VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,1915))
      CALL MP_VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,1916))
      CALL MP_VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,1917))
      CALL MP_VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,1918))
C     Counter-term amplitude(s) for loop diagram number 1370
      CALL MP_R2_GG_1_0(W(14),W(75),R2_GGQ,AMPL(1,1919))
      CALL MP_R2_GG_1_0(W(14),W(75),R2_GGQ,AMPL(1,1920))
      CALL MP_R2_GG_1_0(W(14),W(75),R2_GGQ,AMPL(1,1921))
      CALL MP_R2_GG_1_0(W(14),W(75),R2_GGQ,AMPL(1,1922))
C     Counter-term amplitude(s) for loop diagram number 1371
      CALL MP_VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,1923))
      CALL MP_VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,1924))
      CALL MP_VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,1925))
      CALL MP_VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,1926))
C     Counter-term amplitude(s) for loop diagram number 1372
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,1927))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1928))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,1929))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1930))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1931))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1932))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,1933))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,1934))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,1935))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,1936))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,1937))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,1938))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1939))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,1940))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1941))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1942))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1943))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,1944))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,1945))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,1946))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,1947))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,1948))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,1949))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1950))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,1951))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1952))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1953))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1954))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,1955))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,1956))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,1957))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,1958))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,1959))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,1960))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1961))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,1962))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1963))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1964))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,1965))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,1966))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,1967))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,1968))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,1969))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,1970))
C     Counter-term amplitude(s) for loop diagram number 1373
      CALL MP_VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,1971))
      CALL MP_VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,1972))
      CALL MP_VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,1973))
      CALL MP_VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,1974))
C     Counter-term amplitude(s) for loop diagram number 1375
      CALL MP_VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,1975))
      CALL MP_VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,1976))
      CALL MP_VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,1977))
      CALL MP_VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,1978))
C     Counter-term amplitude(s) for loop diagram number 1376
      CALL MP_VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,1979))
      CALL MP_VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,1980))
      CALL MP_VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,1981))
      CALL MP_VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,1982))
C     Counter-term amplitude(s) for loop diagram number 1377
      CALL MP_VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,1983))
      CALL MP_VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,1984))
      CALL MP_VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,1985))
      CALL MP_VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,1986))
C     Counter-term amplitude(s) for loop diagram number 1387
      CALL MP_R2_GG_1_0(W(17),W(67),R2_GGQ,AMPL(1,1987))
      CALL MP_R2_GG_1_0(W(17),W(67),R2_GGQ,AMPL(1,1988))
      CALL MP_R2_GG_1_0(W(17),W(67),R2_GGQ,AMPL(1,1989))
      CALL MP_R2_GG_1_0(W(17),W(67),R2_GGQ,AMPL(1,1990))
C     Counter-term amplitude(s) for loop diagram number 1388
      CALL MP_R2_GG_1_0(W(11),W(79),R2_GGQ,AMPL(1,1991))
      CALL MP_R2_GG_1_0(W(11),W(79),R2_GGQ,AMPL(1,1992))
      CALL MP_R2_GG_1_0(W(11),W(79),R2_GGQ,AMPL(1,1993))
      CALL MP_R2_GG_1_0(W(11),W(79),R2_GGQ,AMPL(1,1994))
C     Counter-term amplitude(s) for loop diagram number 1389
      CALL MP_R2_GG_1_0(W(11),W(80),R2_GGQ,AMPL(1,1995))
      CALL MP_R2_GG_1_0(W(11),W(80),R2_GGQ,AMPL(1,1996))
      CALL MP_R2_GG_1_0(W(11),W(80),R2_GGQ,AMPL(1,1997))
      CALL MP_R2_GG_1_0(W(11),W(80),R2_GGQ,AMPL(1,1998))
C     Counter-term amplitude(s) for loop diagram number 1390
      CALL MP_VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,1999))
      CALL MP_VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2000))
      CALL MP_VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2001))
      CALL MP_VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2002))
C     Counter-term amplitude(s) for loop diagram number 1391
      CALL MP_VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2003))
      CALL MP_VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2004))
      CALL MP_VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2005))
      CALL MP_VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2006))
C     Counter-term amplitude(s) for loop diagram number 1394
      CALL MP_R2_GG_1_0(W(17),W(18),R2_GGQ,AMPL(1,2007))
      CALL MP_R2_GG_1_0(W(17),W(18),R2_GGQ,AMPL(1,2008))
      CALL MP_R2_GG_1_0(W(17),W(18),R2_GGQ,AMPL(1,2009))
      CALL MP_R2_GG_1_0(W(17),W(18),R2_GGQ,AMPL(1,2010))
      CALL MP_R2_GG_1_0(W(17),W(19),R2_GGQ,AMPL(1,2011))
      CALL MP_R2_GG_1_0(W(17),W(19),R2_GGQ,AMPL(1,2012))
      CALL MP_R2_GG_1_0(W(17),W(19),R2_GGQ,AMPL(1,2013))
      CALL MP_R2_GG_1_0(W(17),W(19),R2_GGQ,AMPL(1,2014))
      CALL MP_R2_GG_1_0(W(17),W(20),R2_GGQ,AMPL(1,2015))
      CALL MP_R2_GG_1_0(W(17),W(20),R2_GGQ,AMPL(1,2016))
      CALL MP_R2_GG_1_0(W(17),W(20),R2_GGQ,AMPL(1,2017))
      CALL MP_R2_GG_1_0(W(17),W(20),R2_GGQ,AMPL(1,2018))
C     Counter-term amplitude(s) for loop diagram number 1395
      CALL MP_R2_GG_1_0(W(11),W(92),R2_GGQ,AMPL(1,2019))
      CALL MP_R2_GG_1_0(W(11),W(92),R2_GGQ,AMPL(1,2020))
      CALL MP_R2_GG_1_0(W(11),W(92),R2_GGQ,AMPL(1,2021))
      CALL MP_R2_GG_1_0(W(11),W(92),R2_GGQ,AMPL(1,2022))
C     Counter-term amplitude(s) for loop diagram number 1396
      CALL MP_VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2023))
      CALL MP_VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2024))
      CALL MP_VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2025))
      CALL MP_VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2026))
      CALL MP_FFV1P0_3(W(21),W(12),GC_5,ZERO,ZERO,W(186))
C     Counter-term amplitude(s) for loop diagram number 1398
      CALL MP_R2_GGZ_0(W(6),W(186),W(3),R2_GGZDOWN,AMPL(1,2027))
      CALL MP_R2_GGZ_0(W(6),W(186),W(3),R2_GGZDOWN,AMPL(1,2028))
      CALL MP_FFV1P0_3(W(21),W(15),GC_5,ZERO,ZERO,W(187))
C     Counter-term amplitude(s) for loop diagram number 1400
      CALL MP_R2_GGZ_0(W(5),W(187),W(3),R2_GGZDOWN,AMPL(1,2029))
      CALL MP_R2_GGZ_0(W(5),W(187),W(3),R2_GGZDOWN,AMPL(1,2030))
      CALL MP_FFV1P0_3(W(21),W(2),GC_5,ZERO,ZERO,W(188))
      CALL MP_VVV1P0_1(W(6),W(188),GC_4,ZERO,ZERO,W(189))
C     Counter-term amplitude(s) for loop diagram number 1402
      CALL MP_R2_GGZ_0(W(5),W(189),W(3),R2_GGZDOWN,AMPL(1,2031))
      CALL MP_R2_GGZ_0(W(5),W(189),W(3),R2_GGZDOWN,AMPL(1,2032))
C     Counter-term amplitude(s) for loop diagram number 1403
      CALL MP_R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2033))
      CALL MP_R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2034))
      CALL MP_R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2035))
      CALL MP_R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2036))
      CALL MP_VVV1P0_1(W(5),W(188),GC_4,ZERO,ZERO,W(190))
C     Counter-term amplitude(s) for loop diagram number 1404
      CALL MP_R2_GGZ_0(W(6),W(190),W(3),R2_GGZDOWN,AMPL(1,2037))
      CALL MP_R2_GGZ_0(W(6),W(190),W(3),R2_GGZDOWN,AMPL(1,2038))
      CALL MP_FFV1P0_3(W(23),W(2),GC_5,ZERO,ZERO,W(191))
C     Counter-term amplitude(s) for loop diagram number 1406
      CALL MP_R2_GGZ_0(W(6),W(191),W(3),R2_GGZDOWN,AMPL(1,2039))
      CALL MP_R2_GGZ_0(W(6),W(191),W(3),R2_GGZDOWN,AMPL(1,2040))
      CALL MP_FFV1P0_3(W(24),W(2),GC_5,ZERO,ZERO,W(192))
C     Counter-term amplitude(s) for loop diagram number 1407
      CALL MP_R2_GGZ_0(W(5),W(192),W(3),R2_GGZDOWN,AMPL(1,2041))
      CALL MP_R2_GGZ_0(W(5),W(192),W(3),R2_GGZDOWN,AMPL(1,2042))
C     Counter-term amplitude(s) for loop diagram number 1408
      CALL MP_R2_GGZ_0(W(11),W(188),W(3),R2_GGZDOWN,AMPL(1,2043))
      CALL MP_R2_GGZ_0(W(11),W(188),W(3),R2_GGZDOWN,AMPL(1,2044))
C     Counter-term amplitude(s) for loop diagram number 1418
      CALL MP_R2_GG_1_0(W(11),W(105),R2_GGQ,AMPL(1,2045))
      CALL MP_R2_GG_1_0(W(11),W(105),R2_GGQ,AMPL(1,2046))
      CALL MP_R2_GG_1_0(W(11),W(105),R2_GGQ,AMPL(1,2047))
      CALL MP_R2_GG_1_0(W(11),W(105),R2_GGQ,AMPL(1,2048))
C     Counter-term amplitude(s) for loop diagram number 1419
      CALL MP_VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2049))
      CALL MP_VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2050))
      CALL MP_VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2051))
      CALL MP_VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2052))
C     Counter-term amplitude(s) for loop diagram number 1421
      CALL MP_R2_GG_1_0(W(14),W(112),R2_GGQ,AMPL(1,2053))
      CALL MP_R2_GG_1_0(W(14),W(112),R2_GGQ,AMPL(1,2054))
      CALL MP_R2_GG_1_0(W(14),W(112),R2_GGQ,AMPL(1,2055))
      CALL MP_R2_GG_1_0(W(14),W(112),R2_GGQ,AMPL(1,2056))
C     Counter-term amplitude(s) for loop diagram number 1422
      CALL MP_VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2057))
      CALL MP_VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2058))
      CALL MP_VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2059))
      CALL MP_VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2060))
      CALL MP_FFV1P0_3(W(26),W(8),GC_5,ZERO,ZERO,W(193))
C     Counter-term amplitude(s) for loop diagram number 1424
      CALL MP_R2_GGZ_0(W(6),W(193),W(3),R2_GGZDOWN,AMPL(1,2061))
      CALL MP_R2_GGZ_0(W(6),W(193),W(3),R2_GGZDOWN,AMPL(1,2062))
      CALL MP_FFV1P0_3(W(26),W(15),GC_5,ZERO,ZERO,W(194))
C     Counter-term amplitude(s) for loop diagram number 1426
      CALL MP_R2_GGZ_0(W(4),W(194),W(3),R2_GGZDOWN,AMPL(1,2063))
      CALL MP_R2_GGZ_0(W(4),W(194),W(3),R2_GGZDOWN,AMPL(1,2064))
      CALL MP_FFV1P0_3(W(26),W(2),GC_5,ZERO,ZERO,W(195))
      CALL MP_VVV1P0_1(W(6),W(195),GC_4,ZERO,ZERO,W(196))
C     Counter-term amplitude(s) for loop diagram number 1428
      CALL MP_R2_GGZ_0(W(4),W(196),W(3),R2_GGZDOWN,AMPL(1,2065))
      CALL MP_R2_GGZ_0(W(4),W(196),W(3),R2_GGZDOWN,AMPL(1,2066))
C     Counter-term amplitude(s) for loop diagram number 1429
      CALL MP_R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2067))
      CALL MP_R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2068))
      CALL MP_R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2069))
      CALL MP_R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2070))
      CALL MP_VVV1P0_1(W(4),W(195),GC_4,ZERO,ZERO,W(197))
C     Counter-term amplitude(s) for loop diagram number 1430
      CALL MP_R2_GGZ_0(W(6),W(197),W(3),R2_GGZDOWN,AMPL(1,2071))
      CALL MP_R2_GGZ_0(W(6),W(197),W(3),R2_GGZDOWN,AMPL(1,2072))
      CALL MP_FFV1P0_3(W(27),W(2),GC_5,ZERO,ZERO,W(198))
C     Counter-term amplitude(s) for loop diagram number 1432
      CALL MP_R2_GGZ_0(W(6),W(198),W(3),R2_GGZDOWN,AMPL(1,2073))
      CALL MP_R2_GGZ_0(W(6),W(198),W(3),R2_GGZDOWN,AMPL(1,2074))
      CALL MP_FFV1P0_3(W(28),W(2),GC_5,ZERO,ZERO,W(199))
C     Counter-term amplitude(s) for loop diagram number 1433
      CALL MP_R2_GGZ_0(W(4),W(199),W(3),R2_GGZDOWN,AMPL(1,2075))
      CALL MP_R2_GGZ_0(W(4),W(199),W(3),R2_GGZDOWN,AMPL(1,2076))
C     Counter-term amplitude(s) for loop diagram number 1434
      CALL MP_R2_GGZ_0(W(14),W(195),W(3),R2_GGZDOWN,AMPL(1,2077))
      CALL MP_R2_GGZ_0(W(14),W(195),W(3),R2_GGZDOWN,AMPL(1,2078))
C     Counter-term amplitude(s) for loop diagram number 1444
      CALL MP_R2_GG_1_0(W(14),W(124),R2_GGQ,AMPL(1,2079))
      CALL MP_R2_GG_1_0(W(14),W(124),R2_GGQ,AMPL(1,2080))
      CALL MP_R2_GG_1_0(W(14),W(124),R2_GGQ,AMPL(1,2081))
      CALL MP_R2_GG_1_0(W(14),W(124),R2_GGQ,AMPL(1,2082))
C     Counter-term amplitude(s) for loop diagram number 1445
      CALL MP_VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2083))
      CALL MP_VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2084))
      CALL MP_VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2085))
      CALL MP_VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2086))
C     Counter-term amplitude(s) for loop diagram number 1447
      CALL MP_R2_GG_1_0(W(16),W(130),R2_GGQ,AMPL(1,2087))
      CALL MP_R2_GG_1_0(W(16),W(130),R2_GGQ,AMPL(1,2088))
      CALL MP_R2_GG_1_0(W(16),W(130),R2_GGQ,AMPL(1,2089))
      CALL MP_R2_GG_1_0(W(16),W(130),R2_GGQ,AMPL(1,2090))
C     Counter-term amplitude(s) for loop diagram number 1448
      CALL MP_VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2091))
      CALL MP_VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2092))
      CALL MP_VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2093))
      CALL MP_VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2094))
      CALL MP_FFV1P0_3(W(30),W(8),GC_5,ZERO,ZERO,W(200))
C     Counter-term amplitude(s) for loop diagram number 1450
      CALL MP_R2_GGZ_0(W(5),W(200),W(3),R2_GGZDOWN,AMPL(1,2095))
      CALL MP_R2_GGZ_0(W(5),W(200),W(3),R2_GGZDOWN,AMPL(1,2096))
      CALL MP_FFV1P0_3(W(30),W(12),GC_5,ZERO,ZERO,W(201))
C     Counter-term amplitude(s) for loop diagram number 1452
      CALL MP_R2_GGZ_0(W(4),W(201),W(3),R2_GGZDOWN,AMPL(1,2097))
      CALL MP_R2_GGZ_0(W(4),W(201),W(3),R2_GGZDOWN,AMPL(1,2098))
      CALL MP_FFV1P0_3(W(30),W(2),GC_5,ZERO,ZERO,W(202))
      CALL MP_VVV1P0_1(W(5),W(202),GC_4,ZERO,ZERO,W(203))
C     Counter-term amplitude(s) for loop diagram number 1454
      CALL MP_R2_GGZ_0(W(4),W(203),W(3),R2_GGZDOWN,AMPL(1,2099))
      CALL MP_R2_GGZ_0(W(4),W(203),W(3),R2_GGZDOWN,AMPL(1,2100))
C     Counter-term amplitude(s) for loop diagram number 1455
      CALL MP_R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2101))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2102))
      CALL MP_R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2103))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2104))
      CALL MP_VVV1P0_1(W(4),W(202),GC_4,ZERO,ZERO,W(204))
C     Counter-term amplitude(s) for loop diagram number 1456
      CALL MP_R2_GGZ_0(W(5),W(204),W(3),R2_GGZDOWN,AMPL(1,2105))
      CALL MP_R2_GGZ_0(W(5),W(204),W(3),R2_GGZDOWN,AMPL(1,2106))
      CALL MP_FFV1P0_3(W(31),W(2),GC_5,ZERO,ZERO,W(205))
C     Counter-term amplitude(s) for loop diagram number 1458
      CALL MP_R2_GGZ_0(W(5),W(205),W(3),R2_GGZDOWN,AMPL(1,2107))
      CALL MP_R2_GGZ_0(W(5),W(205),W(3),R2_GGZDOWN,AMPL(1,2108))
      CALL MP_FFV1P0_3(W(32),W(2),GC_5,ZERO,ZERO,W(206))
C     Counter-term amplitude(s) for loop diagram number 1459
      CALL MP_R2_GGZ_0(W(4),W(206),W(3),R2_GGZDOWN,AMPL(1,2109))
      CALL MP_R2_GGZ_0(W(4),W(206),W(3),R2_GGZDOWN,AMPL(1,2110))
C     Counter-term amplitude(s) for loop diagram number 1460
      CALL MP_R2_GGZ_0(W(16),W(202),W(3),R2_GGZDOWN,AMPL(1,2111))
      CALL MP_R2_GGZ_0(W(16),W(202),W(3),R2_GGZDOWN,AMPL(1,2112))
C     Counter-term amplitude(s) for loop diagram number 1470
      CALL MP_R2_GG_1_0(W(16),W(141),R2_GGQ,AMPL(1,2113))
      CALL MP_R2_GG_1_0(W(16),W(141),R2_GGQ,AMPL(1,2114))
      CALL MP_R2_GG_1_0(W(16),W(141),R2_GGQ,AMPL(1,2115))
      CALL MP_R2_GG_1_0(W(16),W(141),R2_GGQ,AMPL(1,2116))
C     Counter-term amplitude(s) for loop diagram number 1471
      CALL MP_VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2117))
      CALL MP_VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2118))
      CALL MP_VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2119))
      CALL MP_VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2120))
C     Counter-term amplitude(s) for loop diagram number 1473
      CALL MP_R2_GG_1_0(W(34),W(65),R2_GGQ,AMPL(1,2121))
      CALL MP_R2_GG_1_0(W(34),W(65),R2_GGQ,AMPL(1,2122))
      CALL MP_R2_GG_1_0(W(34),W(65),R2_GGQ,AMPL(1,2123))
      CALL MP_R2_GG_1_0(W(34),W(65),R2_GGQ,AMPL(1,2124))
C     Counter-term amplitude(s) for loop diagram number 1474
      CALL MP_R2_GG_1_0(W(16),W(143),R2_GGQ,AMPL(1,2125))
      CALL MP_R2_GG_1_0(W(16),W(143),R2_GGQ,AMPL(1,2126))
      CALL MP_R2_GG_1_0(W(16),W(143),R2_GGQ,AMPL(1,2127))
      CALL MP_R2_GG_1_0(W(16),W(143),R2_GGQ,AMPL(1,2128))
C     Counter-term amplitude(s) for loop diagram number 1475
      CALL MP_VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2129))
      CALL MP_VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2130))
      CALL MP_VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2131))
      CALL MP_VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2132))
C     Counter-term amplitude(s) for loop diagram number 1477
      CALL MP_R2_GG_1_0(W(16),W(145),R2_GGQ,AMPL(1,2133))
      CALL MP_R2_GG_1_0(W(16),W(145),R2_GGQ,AMPL(1,2134))
      CALL MP_R2_GG_1_0(W(16),W(145),R2_GGQ,AMPL(1,2135))
      CALL MP_R2_GG_1_0(W(16),W(145),R2_GGQ,AMPL(1,2136))
C     Counter-term amplitude(s) for loop diagram number 1478
      CALL MP_R2_GG_1_0(W(34),W(66),R2_GGQ,AMPL(1,2137))
      CALL MP_R2_GG_1_0(W(34),W(66),R2_GGQ,AMPL(1,2138))
      CALL MP_R2_GG_1_0(W(34),W(66),R2_GGQ,AMPL(1,2139))
      CALL MP_R2_GG_1_0(W(34),W(66),R2_GGQ,AMPL(1,2140))
C     Counter-term amplitude(s) for loop diagram number 1479
      CALL MP_R2_GG_1_0(W(14),W(147),R2_GGQ,AMPL(1,2141))
      CALL MP_R2_GG_1_0(W(14),W(147),R2_GGQ,AMPL(1,2142))
      CALL MP_R2_GG_1_0(W(14),W(147),R2_GGQ,AMPL(1,2143))
      CALL MP_R2_GG_1_0(W(14),W(147),R2_GGQ,AMPL(1,2144))
C     Counter-term amplitude(s) for loop diagram number 1480
      CALL MP_VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2145))
      CALL MP_VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2146))
      CALL MP_VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2147))
      CALL MP_VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2148))
C     Counter-term amplitude(s) for loop diagram number 1482
      CALL MP_R2_GG_1_0(W(14),W(149),R2_GGQ,AMPL(1,2149))
      CALL MP_R2_GG_1_0(W(14),W(149),R2_GGQ,AMPL(1,2150))
      CALL MP_R2_GG_1_0(W(14),W(149),R2_GGQ,AMPL(1,2151))
      CALL MP_R2_GG_1_0(W(14),W(149),R2_GGQ,AMPL(1,2152))
C     Counter-term amplitude(s) for loop diagram number 1483
      CALL MP_VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2153))
      CALL MP_VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2154))
      CALL MP_VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2155))
      CALL MP_VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2156))
C     Counter-term amplitude(s) for loop diagram number 1484
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2157))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2158))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2159))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2160))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2161))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2162))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,2163))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2164))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2165))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2166))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2167))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2168))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2169))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2170))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2171))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2172))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2173))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,2174))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2175))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2176))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2177))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2178))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2179))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2180))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2181))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2182))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2183))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2184))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,2185))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2186))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2187))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2188))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2189))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2190))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2191))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2192))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2193))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2194))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2195))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,2196))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2197))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2198))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2199))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2200))
C     Counter-term amplitude(s) for loop diagram number 1485
      CALL MP_VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2201))
      CALL MP_VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2202))
      CALL MP_VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2203))
      CALL MP_VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2204))
C     Counter-term amplitude(s) for loop diagram number 1487
      CALL MP_VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2205))
      CALL MP_VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2206))
      CALL MP_VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2207))
      CALL MP_VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2208))
C     Counter-term amplitude(s) for loop diagram number 1488
      CALL MP_VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2209))
      CALL MP_VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2210))
      CALL MP_VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2211))
      CALL MP_VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2212))
C     Counter-term amplitude(s) for loop diagram number 1489
      CALL MP_VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2213))
      CALL MP_VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2214))
      CALL MP_VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2215))
      CALL MP_VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2216))
C     Counter-term amplitude(s) for loop diagram number 1499
      CALL MP_R2_GG_1_0(W(34),W(67),R2_GGQ,AMPL(1,2217))
      CALL MP_R2_GG_1_0(W(34),W(67),R2_GGQ,AMPL(1,2218))
      CALL MP_R2_GG_1_0(W(34),W(67),R2_GGQ,AMPL(1,2219))
      CALL MP_R2_GG_1_0(W(34),W(67),R2_GGQ,AMPL(1,2220))
C     Counter-term amplitude(s) for loop diagram number 1500
      CALL MP_R2_GG_1_0(W(11),W(151),R2_GGQ,AMPL(1,2221))
      CALL MP_R2_GG_1_0(W(11),W(151),R2_GGQ,AMPL(1,2222))
      CALL MP_R2_GG_1_0(W(11),W(151),R2_GGQ,AMPL(1,2223))
      CALL MP_R2_GG_1_0(W(11),W(151),R2_GGQ,AMPL(1,2224))
C     Counter-term amplitude(s) for loop diagram number 1501
      CALL MP_R2_GG_1_0(W(11),W(153),R2_GGQ,AMPL(1,2225))
      CALL MP_R2_GG_1_0(W(11),W(153),R2_GGQ,AMPL(1,2226))
      CALL MP_R2_GG_1_0(W(11),W(153),R2_GGQ,AMPL(1,2227))
      CALL MP_R2_GG_1_0(W(11),W(153),R2_GGQ,AMPL(1,2228))
C     Counter-term amplitude(s) for loop diagram number 1502
      CALL MP_VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2229))
      CALL MP_VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2230))
      CALL MP_VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2231))
      CALL MP_VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2232))
C     Counter-term amplitude(s) for loop diagram number 1503
      CALL MP_VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2233))
      CALL MP_VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2234))
      CALL MP_VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2235))
      CALL MP_VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2236))
C     Counter-term amplitude(s) for loop diagram number 1506
      CALL MP_R2_GG_1_0(W(34),W(18),R2_GGQ,AMPL(1,2237))
      CALL MP_R2_GG_1_0(W(34),W(18),R2_GGQ,AMPL(1,2238))
      CALL MP_R2_GG_1_0(W(34),W(18),R2_GGQ,AMPL(1,2239))
      CALL MP_R2_GG_1_0(W(34),W(18),R2_GGQ,AMPL(1,2240))
      CALL MP_R2_GG_1_0(W(34),W(19),R2_GGQ,AMPL(1,2241))
      CALL MP_R2_GG_1_0(W(34),W(19),R2_GGQ,AMPL(1,2242))
      CALL MP_R2_GG_1_0(W(34),W(19),R2_GGQ,AMPL(1,2243))
      CALL MP_R2_GG_1_0(W(34),W(19),R2_GGQ,AMPL(1,2244))
      CALL MP_R2_GG_1_0(W(34),W(20),R2_GGQ,AMPL(1,2245))
      CALL MP_R2_GG_1_0(W(34),W(20),R2_GGQ,AMPL(1,2246))
      CALL MP_R2_GG_1_0(W(34),W(20),R2_GGQ,AMPL(1,2247))
      CALL MP_R2_GG_1_0(W(34),W(20),R2_GGQ,AMPL(1,2248))
      CALL MP_FFV1P0_3(W(1),W(8),GC_5,ZERO,ZERO,W(207))
      CALL MP_VVV1P0_1(W(6),W(207),GC_4,ZERO,ZERO,W(208))
C     Counter-term amplitude(s) for loop diagram number 1507
      CALL MP_R2_GGZ_0(W(5),W(208),W(3),R2_GGZDOWN,AMPL(1,2249))
      CALL MP_R2_GGZ_0(W(5),W(208),W(3),R2_GGZDOWN,AMPL(1,2250))
C     Counter-term amplitude(s) for loop diagram number 1508
      CALL MP_R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2251))
      CALL MP_R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2252))
      CALL MP_R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2253))
      CALL MP_R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2254))
      CALL MP_VVV1P0_1(W(5),W(207),GC_4,ZERO,ZERO,W(209))
C     Counter-term amplitude(s) for loop diagram number 1509
      CALL MP_R2_GGZ_0(W(6),W(209),W(3),R2_GGZDOWN,AMPL(1,2255))
      CALL MP_R2_GGZ_0(W(6),W(209),W(3),R2_GGZDOWN,AMPL(1,2256))
      CALL MP_FFV1P0_3(W(1),W(40),GC_5,ZERO,ZERO,W(210))
C     Counter-term amplitude(s) for loop diagram number 1511
      CALL MP_R2_GGZ_0(W(6),W(210),W(3),R2_GGZDOWN,AMPL(1,2257))
      CALL MP_R2_GGZ_0(W(6),W(210),W(3),R2_GGZDOWN,AMPL(1,2258))
      CALL MP_FFV1P0_3(W(1),W(38),GC_5,ZERO,ZERO,W(211))
C     Counter-term amplitude(s) for loop diagram number 1512
      CALL MP_R2_GGZ_0(W(5),W(211),W(3),R2_GGZDOWN,AMPL(1,2259))
      CALL MP_R2_GGZ_0(W(5),W(211),W(3),R2_GGZDOWN,AMPL(1,2260))
C     Counter-term amplitude(s) for loop diagram number 1513
      CALL MP_R2_GGZ_0(W(11),W(207),W(3),R2_GGZDOWN,AMPL(1,2261))
      CALL MP_R2_GGZ_0(W(11),W(207),W(3),R2_GGZDOWN,AMPL(1,2262))
C     Counter-term amplitude(s) for loop diagram number 1523
      CALL MP_R2_GG_1_0(W(11),W(159),R2_GGQ,AMPL(1,2263))
      CALL MP_R2_GG_1_0(W(11),W(159),R2_GGQ,AMPL(1,2264))
      CALL MP_R2_GG_1_0(W(11),W(159),R2_GGQ,AMPL(1,2265))
      CALL MP_R2_GG_1_0(W(11),W(159),R2_GGQ,AMPL(1,2266))
C     Counter-term amplitude(s) for loop diagram number 1524
      CALL MP_VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2267))
      CALL MP_VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2268))
      CALL MP_VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2269))
      CALL MP_VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2270))
      CALL MP_FFV1P0_3(W(1),W(12),GC_5,ZERO,ZERO,W(212))
      CALL MP_VVV1P0_1(W(6),W(212),GC_4,ZERO,ZERO,W(213))
C     Counter-term amplitude(s) for loop diagram number 1526
      CALL MP_R2_GGZ_0(W(4),W(213),W(3),R2_GGZDOWN,AMPL(1,2271))
      CALL MP_R2_GGZ_0(W(4),W(213),W(3),R2_GGZDOWN,AMPL(1,2272))
C     Counter-term amplitude(s) for loop diagram number 1527
      CALL MP_R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2273))
      CALL MP_R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2274))
      CALL MP_R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2275))
      CALL MP_R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2276))
      CALL MP_VVV1P0_1(W(4),W(212),GC_4,ZERO,ZERO,W(214))
C     Counter-term amplitude(s) for loop diagram number 1528
      CALL MP_R2_GGZ_0(W(6),W(214),W(3),R2_GGZDOWN,AMPL(1,2277))
      CALL MP_R2_GGZ_0(W(6),W(214),W(3),R2_GGZDOWN,AMPL(1,2278))
      CALL MP_FFV1P0_3(W(1),W(49),GC_5,ZERO,ZERO,W(215))
C     Counter-term amplitude(s) for loop diagram number 1530
      CALL MP_R2_GGZ_0(W(6),W(215),W(3),R2_GGZDOWN,AMPL(1,2279))
      CALL MP_R2_GGZ_0(W(6),W(215),W(3),R2_GGZDOWN,AMPL(1,2280))
      CALL MP_FFV1P0_3(W(1),W(47),GC_5,ZERO,ZERO,W(216))
C     Counter-term amplitude(s) for loop diagram number 1531
      CALL MP_R2_GGZ_0(W(4),W(216),W(3),R2_GGZDOWN,AMPL(1,2281))
      CALL MP_R2_GGZ_0(W(4),W(216),W(3),R2_GGZDOWN,AMPL(1,2282))
C     Counter-term amplitude(s) for loop diagram number 1532
      CALL MP_R2_GGZ_0(W(14),W(212),W(3),R2_GGZDOWN,AMPL(1,2283))
      CALL MP_R2_GGZ_0(W(14),W(212),W(3),R2_GGZDOWN,AMPL(1,2284))
C     Counter-term amplitude(s) for loop diagram number 1542
      CALL MP_R2_GG_1_0(W(14),W(161),R2_GGQ,AMPL(1,2285))
      CALL MP_R2_GG_1_0(W(14),W(161),R2_GGQ,AMPL(1,2286))
      CALL MP_R2_GG_1_0(W(14),W(161),R2_GGQ,AMPL(1,2287))
      CALL MP_R2_GG_1_0(W(14),W(161),R2_GGQ,AMPL(1,2288))
C     Counter-term amplitude(s) for loop diagram number 1543
      CALL MP_VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2289))
      CALL MP_VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2290))
      CALL MP_VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2291))
      CALL MP_VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2292))
      CALL MP_FFV1P0_3(W(1),W(15),GC_5,ZERO,ZERO,W(217))
      CALL MP_VVV1P0_1(W(5),W(217),GC_4,ZERO,ZERO,W(218))
C     Counter-term amplitude(s) for loop diagram number 1545
      CALL MP_R2_GGZ_0(W(4),W(218),W(3),R2_GGZDOWN,AMPL(1,2293))
      CALL MP_R2_GGZ_0(W(4),W(218),W(3),R2_GGZDOWN,AMPL(1,2294))
C     Counter-term amplitude(s) for loop diagram number 1546
      CALL MP_R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2295))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2296))
      CALL MP_R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2297))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2298))
      CALL MP_VVV1P0_1(W(4),W(217),GC_4,ZERO,ZERO,W(219))
C     Counter-term amplitude(s) for loop diagram number 1547
      CALL MP_R2_GGZ_0(W(5),W(219),W(3),R2_GGZDOWN,AMPL(1,2299))
      CALL MP_R2_GGZ_0(W(5),W(219),W(3),R2_GGZDOWN,AMPL(1,2300))
      CALL MP_FFV1P0_3(W(1),W(58),GC_5,ZERO,ZERO,W(220))
C     Counter-term amplitude(s) for loop diagram number 1549
      CALL MP_R2_GGZ_0(W(5),W(220),W(3),R2_GGZDOWN,AMPL(1,2301))
      CALL MP_R2_GGZ_0(W(5),W(220),W(3),R2_GGZDOWN,AMPL(1,2302))
      CALL MP_FFV1P0_3(W(1),W(56),GC_5,ZERO,ZERO,W(221))
C     Counter-term amplitude(s) for loop diagram number 1550
      CALL MP_R2_GGZ_0(W(4),W(221),W(3),R2_GGZDOWN,AMPL(1,2303))
      CALL MP_R2_GGZ_0(W(4),W(221),W(3),R2_GGZDOWN,AMPL(1,2304))
C     Counter-term amplitude(s) for loop diagram number 1551
      CALL MP_R2_GGZ_0(W(16),W(217),W(3),R2_GGZDOWN,AMPL(1,2305))
      CALL MP_R2_GGZ_0(W(16),W(217),W(3),R2_GGZDOWN,AMPL(1,2306))
C     Counter-term amplitude(s) for loop diagram number 1561
      CALL MP_R2_GG_1_0(W(16),W(163),R2_GGQ,AMPL(1,2307))
      CALL MP_R2_GG_1_0(W(16),W(163),R2_GGQ,AMPL(1,2308))
      CALL MP_R2_GG_1_0(W(16),W(163),R2_GGQ,AMPL(1,2309))
      CALL MP_R2_GG_1_0(W(16),W(163),R2_GGQ,AMPL(1,2310))
C     Counter-term amplitude(s) for loop diagram number 1562
      CALL MP_VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2311))
      CALL MP_VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2312))
      CALL MP_VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2313))
      CALL MP_VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2314))
      CALL MP_FFV1P0_3(W(35),W(2),GC_5,ZERO,ZERO,W(222))
C     Counter-term amplitude(s) for loop diagram number 1564
      CALL MP_R2_GGZ_0(W(6),W(222),W(3),R2_GGZDOWN,AMPL(1,2315))
      CALL MP_R2_GGZ_0(W(6),W(222),W(3),R2_GGZDOWN,AMPL(1,2316))
      CALL MP_FFV1P0_3(W(1),W(69),GC_5,ZERO,ZERO,W(223))
C     Counter-term amplitude(s) for loop diagram number 1565
      CALL MP_R2_GGZ_0(W(6),W(223),W(3),R2_GGZDOWN,AMPL(1,2317))
      CALL MP_R2_GGZ_0(W(6),W(223),W(3),R2_GGZDOWN,AMPL(1,2318))
      CALL MP_FFV1P0_3(W(36),W(2),GC_5,ZERO,ZERO,W(224))
C     Counter-term amplitude(s) for loop diagram number 1568
      CALL MP_R2_GGZ_0(W(5),W(224),W(3),R2_GGZDOWN,AMPL(1,2319))
      CALL MP_R2_GGZ_0(W(5),W(224),W(3),R2_GGZDOWN,AMPL(1,2320))
      CALL MP_FFV1P0_3(W(1),W(74),GC_5,ZERO,ZERO,W(225))
C     Counter-term amplitude(s) for loop diagram number 1569
      CALL MP_R2_GGZ_0(W(5),W(225),W(3),R2_GGZDOWN,AMPL(1,2321))
      CALL MP_R2_GGZ_0(W(5),W(225),W(3),R2_GGZDOWN,AMPL(1,2322))
      CALL MP_FFV1P0_3(W(37),W(2),GC_5,ZERO,ZERO,W(226))
C     Counter-term amplitude(s) for loop diagram number 1572
      CALL MP_R2_GGZ_0(W(4),W(226),W(3),R2_GGZDOWN,AMPL(1,2323))
      CALL MP_R2_GGZ_0(W(4),W(226),W(3),R2_GGZDOWN,AMPL(1,2324))
      CALL MP_FFV1P0_3(W(1),W(78),GC_5,ZERO,ZERO,W(227))
C     Counter-term amplitude(s) for loop diagram number 1573
      CALL MP_R2_GGZ_0(W(4),W(227),W(3),R2_GGZDOWN,AMPL(1,2325))
      CALL MP_R2_GGZ_0(W(4),W(227),W(3),R2_GGZDOWN,AMPL(1,2326))
C     Counter-term amplitude(s) for loop diagram number 1576
      CALL MP_R2_GGZ_0(W(5),W(166),W(3),R2_GGZUP,AMPL(1,2327))
      CALL MP_R2_GGZ_0(W(5),W(166),W(3),R2_GGZUP,AMPL(1,2328))
C     Counter-term amplitude(s) for loop diagram number 1577
      CALL MP_R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECUP,AMPL(1,2329)
     $ )
      CALL MP_R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2330))
      CALL MP_R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECUP,AMPL(1,2331)
     $ )
      CALL MP_R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2332))
C     Counter-term amplitude(s) for loop diagram number 1578
      CALL MP_R2_GGZ_0(W(6),W(167),W(3),R2_GGZUP,AMPL(1,2333))
      CALL MP_R2_GGZ_0(W(6),W(167),W(3),R2_GGZUP,AMPL(1,2334))
C     Counter-term amplitude(s) for loop diagram number 1580
      CALL MP_R2_GGZ_0(W(4),W(169),W(3),R2_GGZUP,AMPL(1,2335))
      CALL MP_R2_GGZ_0(W(4),W(169),W(3),R2_GGZUP,AMPL(1,2336))
C     Counter-term amplitude(s) for loop diagram number 1581
      CALL MP_R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECUP,AMPL(1,2337)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2338))
      CALL MP_R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECUP,AMPL(1,2339)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2340))
C     Counter-term amplitude(s) for loop diagram number 1582
      CALL MP_R2_GGZ_0(W(6),W(170),W(3),R2_GGZUP,AMPL(1,2341))
      CALL MP_R2_GGZ_0(W(6),W(170),W(3),R2_GGZUP,AMPL(1,2342))
C     Counter-term amplitude(s) for loop diagram number 1584
      CALL MP_R2_GGZ_0(W(4),W(172),W(3),R2_GGZUP,AMPL(1,2343))
      CALL MP_R2_GGZ_0(W(4),W(172),W(3),R2_GGZUP,AMPL(1,2344))
C     Counter-term amplitude(s) for loop diagram number 1585
      CALL MP_R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECUP,AMPL(1,2345)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2346))
      CALL MP_R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECUP,AMPL(1,2347)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2348))
C     Counter-term amplitude(s) for loop diagram number 1586
      CALL MP_R2_GGZ_0(W(5),W(173),W(3),R2_GGZUP,AMPL(1,2349))
      CALL MP_R2_GGZ_0(W(5),W(173),W(3),R2_GGZUP,AMPL(1,2350))
C     Counter-term amplitude(s) for loop diagram number 1594
      CALL MP_R2_GGZ_0(W(6),W(174),W(3),R2_GGZUP,AMPL(1,2351))
      CALL MP_R2_GGZ_0(W(6),W(174),W(3),R2_GGZUP,AMPL(1,2352))
      CALL MP_R2_GGZ_0(W(6),W(175),W(3),R2_GGZUP,AMPL(1,2353))
      CALL MP_R2_GGZ_0(W(6),W(175),W(3),R2_GGZUP,AMPL(1,2354))
      CALL MP_R2_GGZ_0(W(6),W(176),W(3),R2_GGZUP,AMPL(1,2355))
      CALL MP_R2_GGZ_0(W(6),W(176),W(3),R2_GGZUP,AMPL(1,2356))
C     Counter-term amplitude(s) for loop diagram number 1595
      CALL MP_R2_GGZ_0(W(5),W(177),W(3),R2_GGZUP,AMPL(1,2357))
      CALL MP_R2_GGZ_0(W(5),W(177),W(3),R2_GGZUP,AMPL(1,2358))
      CALL MP_R2_GGZ_0(W(5),W(178),W(3),R2_GGZUP,AMPL(1,2359))
      CALL MP_R2_GGZ_0(W(5),W(178),W(3),R2_GGZUP,AMPL(1,2360))
      CALL MP_R2_GGZ_0(W(5),W(179),W(3),R2_GGZUP,AMPL(1,2361))
      CALL MP_R2_GGZ_0(W(5),W(179),W(3),R2_GGZUP,AMPL(1,2362))
C     Counter-term amplitude(s) for loop diagram number 1596
      CALL MP_R2_GGZ_0(W(4),W(180),W(3),R2_GGZUP,AMPL(1,2363))
      CALL MP_R2_GGZ_0(W(4),W(180),W(3),R2_GGZUP,AMPL(1,2364))
      CALL MP_R2_GGZ_0(W(4),W(181),W(3),R2_GGZUP,AMPL(1,2365))
      CALL MP_R2_GGZ_0(W(4),W(181),W(3),R2_GGZUP,AMPL(1,2366))
      CALL MP_R2_GGZ_0(W(4),W(182),W(3),R2_GGZUP,AMPL(1,2367))
      CALL MP_R2_GGZ_0(W(4),W(182),W(3),R2_GGZUP,AMPL(1,2368))
C     Counter-term amplitude(s) for loop diagram number 1597
      CALL MP_R2_GGZ_0(W(164),W(65),W(3),R2_GGZUP,AMPL(1,2369))
      CALL MP_R2_GGZ_0(W(164),W(65),W(3),R2_GGZUP,AMPL(1,2370))
C     Counter-term amplitude(s) for loop diagram number 1598
      CALL MP_R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECUP,AMPL(1
     $ ,2371))
      CALL MP_R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2372))
      CALL MP_R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECUP,AMPL(1
     $ ,2373))
      CALL MP_R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2374))
C     Counter-term amplitude(s) for loop diagram number 1599
      CALL MP_R2_GGZ_0(W(6),W(183),W(3),R2_GGZUP,AMPL(1,2375))
      CALL MP_R2_GGZ_0(W(6),W(183),W(3),R2_GGZUP,AMPL(1,2376))
C     Counter-term amplitude(s) for loop diagram number 1600
      CALL MP_R2_GGZ_0(W(16),W(171),W(3),R2_GGZUP,AMPL(1,2377))
      CALL MP_R2_GGZ_0(W(16),W(171),W(3),R2_GGZUP,AMPL(1,2378))
C     Counter-term amplitude(s) for loop diagram number 1609
      CALL MP_R2_GGZ_0(W(164),W(66),W(3),R2_GGZUP,AMPL(1,2379))
      CALL MP_R2_GGZ_0(W(164),W(66),W(3),R2_GGZUP,AMPL(1,2380))
C     Counter-term amplitude(s) for loop diagram number 1610
      CALL MP_R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECUP,AMPL(1
     $ ,2381))
      CALL MP_R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2382))
      CALL MP_R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECUP,AMPL(1
     $ ,2383))
      CALL MP_R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2384))
C     Counter-term amplitude(s) for loop diagram number 1611
      CALL MP_R2_GGZ_0(W(5),W(184),W(3),R2_GGZUP,AMPL(1,2385))
      CALL MP_R2_GGZ_0(W(5),W(184),W(3),R2_GGZUP,AMPL(1,2386))
C     Counter-term amplitude(s) for loop diagram number 1612
      CALL MP_R2_GGZ_0(W(14),W(168),W(3),R2_GGZUP,AMPL(1,2387))
      CALL MP_R2_GGZ_0(W(14),W(168),W(3),R2_GGZUP,AMPL(1,2388))
C     Counter-term amplitude(s) for loop diagram number 1634
      CALL MP_R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECUP,AMPL(1
     $ ,2389))
      CALL MP_R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2390))
      CALL MP_R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECUP,AMPL(1
     $ ,2391))
      CALL MP_R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2392))
C     Counter-term amplitude(s) for loop diagram number 1636
      CALL MP_R2_GGZ_0(W(4),W(185),W(3),R2_GGZUP,AMPL(1,2393))
      CALL MP_R2_GGZ_0(W(4),W(185),W(3),R2_GGZUP,AMPL(1,2394))
C     Counter-term amplitude(s) for loop diagram number 1637
      CALL MP_R2_GGZ_0(W(164),W(67),W(3),R2_GGZUP,AMPL(1,2395))
      CALL MP_R2_GGZ_0(W(164),W(67),W(3),R2_GGZUP,AMPL(1,2396))
C     Counter-term amplitude(s) for loop diagram number 1639
      CALL MP_R2_GGZ_0(W(11),W(165),W(3),R2_GGZUP,AMPL(1,2397))
      CALL MP_R2_GGZ_0(W(11),W(165),W(3),R2_GGZUP,AMPL(1,2398))
C     Counter-term amplitude(s) for loop diagram number 1672
      CALL MP_R2_GGZ_0(W(164),W(18),W(3),R2_GGZUP,AMPL(1,2399))
      CALL MP_R2_GGZ_0(W(164),W(18),W(3),R2_GGZUP,AMPL(1,2400))
      CALL MP_R2_GGZ_0(W(164),W(19),W(3),R2_GGZUP,AMPL(1,2401))
      CALL MP_R2_GGZ_0(W(164),W(19),W(3),R2_GGZUP,AMPL(1,2402))
      CALL MP_R2_GGZ_0(W(164),W(20),W(3),R2_GGZUP,AMPL(1,2403))
      CALL MP_R2_GGZ_0(W(164),W(20),W(3),R2_GGZUP,AMPL(1,2404))
C     Counter-term amplitude(s) for loop diagram number 1674
      CALL MP_R2_GGZ_0(W(6),W(186),W(3),R2_GGZUP,AMPL(1,2405))
      CALL MP_R2_GGZ_0(W(6),W(186),W(3),R2_GGZUP,AMPL(1,2406))
C     Counter-term amplitude(s) for loop diagram number 1676
      CALL MP_R2_GGZ_0(W(5),W(187),W(3),R2_GGZUP,AMPL(1,2407))
      CALL MP_R2_GGZ_0(W(5),W(187),W(3),R2_GGZUP,AMPL(1,2408))
C     Counter-term amplitude(s) for loop diagram number 1678
      CALL MP_R2_GGZ_0(W(5),W(189),W(3),R2_GGZUP,AMPL(1,2409))
      CALL MP_R2_GGZ_0(W(5),W(189),W(3),R2_GGZUP,AMPL(1,2410))
C     Counter-term amplitude(s) for loop diagram number 1679
      CALL MP_R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECUP,AMPL(1,2411)
     $ )
      CALL MP_R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2412))
      CALL MP_R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECUP,AMPL(1,2413)
     $ )
      CALL MP_R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2414))
C     Counter-term amplitude(s) for loop diagram number 1680
      CALL MP_R2_GGZ_0(W(6),W(190),W(3),R2_GGZUP,AMPL(1,2415))
      CALL MP_R2_GGZ_0(W(6),W(190),W(3),R2_GGZUP,AMPL(1,2416))
C     Counter-term amplitude(s) for loop diagram number 1682
      CALL MP_R2_GGZ_0(W(6),W(191),W(3),R2_GGZUP,AMPL(1,2417))
      CALL MP_R2_GGZ_0(W(6),W(191),W(3),R2_GGZUP,AMPL(1,2418))
C     Counter-term amplitude(s) for loop diagram number 1683
      CALL MP_R2_GGZ_0(W(5),W(192),W(3),R2_GGZUP,AMPL(1,2419))
      CALL MP_R2_GGZ_0(W(5),W(192),W(3),R2_GGZUP,AMPL(1,2420))
C     Counter-term amplitude(s) for loop diagram number 1684
      CALL MP_R2_GGZ_0(W(11),W(188),W(3),R2_GGZUP,AMPL(1,2421))
      CALL MP_R2_GGZ_0(W(11),W(188),W(3),R2_GGZUP,AMPL(1,2422))
C     Counter-term amplitude(s) for loop diagram number 1694
      CALL MP_R2_GGZ_0(W(6),W(193),W(3),R2_GGZUP,AMPL(1,2423))
      CALL MP_R2_GGZ_0(W(6),W(193),W(3),R2_GGZUP,AMPL(1,2424))
C     Counter-term amplitude(s) for loop diagram number 1696
      CALL MP_R2_GGZ_0(W(4),W(194),W(3),R2_GGZUP,AMPL(1,2425))
      CALL MP_R2_GGZ_0(W(4),W(194),W(3),R2_GGZUP,AMPL(1,2426))
C     Counter-term amplitude(s) for loop diagram number 1698
      CALL MP_R2_GGZ_0(W(4),W(196),W(3),R2_GGZUP,AMPL(1,2427))
      CALL MP_R2_GGZ_0(W(4),W(196),W(3),R2_GGZUP,AMPL(1,2428))
C     Counter-term amplitude(s) for loop diagram number 1699
      CALL MP_R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECUP,AMPL(1,2429)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2430))
      CALL MP_R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECUP,AMPL(1,2431)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2432))
C     Counter-term amplitude(s) for loop diagram number 1700
      CALL MP_R2_GGZ_0(W(6),W(197),W(3),R2_GGZUP,AMPL(1,2433))
      CALL MP_R2_GGZ_0(W(6),W(197),W(3),R2_GGZUP,AMPL(1,2434))
C     Counter-term amplitude(s) for loop diagram number 1702
      CALL MP_R2_GGZ_0(W(6),W(198),W(3),R2_GGZUP,AMPL(1,2435))
      CALL MP_R2_GGZ_0(W(6),W(198),W(3),R2_GGZUP,AMPL(1,2436))
C     Counter-term amplitude(s) for loop diagram number 1703
      CALL MP_R2_GGZ_0(W(4),W(199),W(3),R2_GGZUP,AMPL(1,2437))
      CALL MP_R2_GGZ_0(W(4),W(199),W(3),R2_GGZUP,AMPL(1,2438))
C     Counter-term amplitude(s) for loop diagram number 1704
      CALL MP_R2_GGZ_0(W(14),W(195),W(3),R2_GGZUP,AMPL(1,2439))
      CALL MP_R2_GGZ_0(W(14),W(195),W(3),R2_GGZUP,AMPL(1,2440))
C     Counter-term amplitude(s) for loop diagram number 1714
      CALL MP_R2_GGZ_0(W(5),W(200),W(3),R2_GGZUP,AMPL(1,2441))
      CALL MP_R2_GGZ_0(W(5),W(200),W(3),R2_GGZUP,AMPL(1,2442))
C     Counter-term amplitude(s) for loop diagram number 1716
      CALL MP_R2_GGZ_0(W(4),W(201),W(3),R2_GGZUP,AMPL(1,2443))
      CALL MP_R2_GGZ_0(W(4),W(201),W(3),R2_GGZUP,AMPL(1,2444))
C     Counter-term amplitude(s) for loop diagram number 1718
      CALL MP_R2_GGZ_0(W(4),W(203),W(3),R2_GGZUP,AMPL(1,2445))
      CALL MP_R2_GGZ_0(W(4),W(203),W(3),R2_GGZUP,AMPL(1,2446))
C     Counter-term amplitude(s) for loop diagram number 1719
      CALL MP_R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECUP,AMPL(1,2447)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2448))
      CALL MP_R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECUP,AMPL(1,2449)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2450))
C     Counter-term amplitude(s) for loop diagram number 1720
      CALL MP_R2_GGZ_0(W(5),W(204),W(3),R2_GGZUP,AMPL(1,2451))
      CALL MP_R2_GGZ_0(W(5),W(204),W(3),R2_GGZUP,AMPL(1,2452))
C     Counter-term amplitude(s) for loop diagram number 1722
      CALL MP_R2_GGZ_0(W(5),W(205),W(3),R2_GGZUP,AMPL(1,2453))
      CALL MP_R2_GGZ_0(W(5),W(205),W(3),R2_GGZUP,AMPL(1,2454))
C     Counter-term amplitude(s) for loop diagram number 1723
      CALL MP_R2_GGZ_0(W(4),W(206),W(3),R2_GGZUP,AMPL(1,2455))
      CALL MP_R2_GGZ_0(W(4),W(206),W(3),R2_GGZUP,AMPL(1,2456))
C     Counter-term amplitude(s) for loop diagram number 1724
      CALL MP_R2_GGZ_0(W(16),W(202),W(3),R2_GGZUP,AMPL(1,2457))
      CALL MP_R2_GGZ_0(W(16),W(202),W(3),R2_GGZUP,AMPL(1,2458))
C     Counter-term amplitude(s) for loop diagram number 1734
      CALL MP_R2_GGZ_0(W(5),W(208),W(3),R2_GGZUP,AMPL(1,2459))
      CALL MP_R2_GGZ_0(W(5),W(208),W(3),R2_GGZUP,AMPL(1,2460))
C     Counter-term amplitude(s) for loop diagram number 1735
      CALL MP_R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECUP,AMPL(1,2461)
     $ )
      CALL MP_R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2462))
      CALL MP_R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECUP,AMPL(1,2463)
     $ )
      CALL MP_R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2464))
C     Counter-term amplitude(s) for loop diagram number 1736
      CALL MP_R2_GGZ_0(W(6),W(209),W(3),R2_GGZUP,AMPL(1,2465))
      CALL MP_R2_GGZ_0(W(6),W(209),W(3),R2_GGZUP,AMPL(1,2466))
C     Counter-term amplitude(s) for loop diagram number 1738
      CALL MP_R2_GGZ_0(W(6),W(210),W(3),R2_GGZUP,AMPL(1,2467))
      CALL MP_R2_GGZ_0(W(6),W(210),W(3),R2_GGZUP,AMPL(1,2468))
C     Counter-term amplitude(s) for loop diagram number 1739
      CALL MP_R2_GGZ_0(W(5),W(211),W(3),R2_GGZUP,AMPL(1,2469))
      CALL MP_R2_GGZ_0(W(5),W(211),W(3),R2_GGZUP,AMPL(1,2470))
C     Counter-term amplitude(s) for loop diagram number 1740
      CALL MP_R2_GGZ_0(W(11),W(207),W(3),R2_GGZUP,AMPL(1,2471))
      CALL MP_R2_GGZ_0(W(11),W(207),W(3),R2_GGZUP,AMPL(1,2472))
C     Counter-term amplitude(s) for loop diagram number 1750
      CALL MP_R2_GGZ_0(W(4),W(213),W(3),R2_GGZUP,AMPL(1,2473))
      CALL MP_R2_GGZ_0(W(4),W(213),W(3),R2_GGZUP,AMPL(1,2474))
C     Counter-term amplitude(s) for loop diagram number 1751
      CALL MP_R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECUP,AMPL(1,2475)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2476))
      CALL MP_R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECUP,AMPL(1,2477)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2478))
C     Counter-term amplitude(s) for loop diagram number 1752
      CALL MP_R2_GGZ_0(W(6),W(214),W(3),R2_GGZUP,AMPL(1,2479))
      CALL MP_R2_GGZ_0(W(6),W(214),W(3),R2_GGZUP,AMPL(1,2480))
C     Counter-term amplitude(s) for loop diagram number 1754
      CALL MP_R2_GGZ_0(W(6),W(215),W(3),R2_GGZUP,AMPL(1,2481))
      CALL MP_R2_GGZ_0(W(6),W(215),W(3),R2_GGZUP,AMPL(1,2482))
C     Counter-term amplitude(s) for loop diagram number 1755
      CALL MP_R2_GGZ_0(W(4),W(216),W(3),R2_GGZUP,AMPL(1,2483))
      CALL MP_R2_GGZ_0(W(4),W(216),W(3),R2_GGZUP,AMPL(1,2484))
C     Counter-term amplitude(s) for loop diagram number 1756
      CALL MP_R2_GGZ_0(W(14),W(212),W(3),R2_GGZUP,AMPL(1,2485))
      CALL MP_R2_GGZ_0(W(14),W(212),W(3),R2_GGZUP,AMPL(1,2486))
C     Counter-term amplitude(s) for loop diagram number 1766
      CALL MP_R2_GGZ_0(W(4),W(218),W(3),R2_GGZUP,AMPL(1,2487))
      CALL MP_R2_GGZ_0(W(4),W(218),W(3),R2_GGZUP,AMPL(1,2488))
C     Counter-term amplitude(s) for loop diagram number 1767
      CALL MP_R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECUP,AMPL(1,2489)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2490))
      CALL MP_R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECUP,AMPL(1,2491)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2492))
C     Counter-term amplitude(s) for loop diagram number 1768
      CALL MP_R2_GGZ_0(W(5),W(219),W(3),R2_GGZUP,AMPL(1,2493))
      CALL MP_R2_GGZ_0(W(5),W(219),W(3),R2_GGZUP,AMPL(1,2494))
C     Counter-term amplitude(s) for loop diagram number 1770
      CALL MP_R2_GGZ_0(W(5),W(220),W(3),R2_GGZUP,AMPL(1,2495))
      CALL MP_R2_GGZ_0(W(5),W(220),W(3),R2_GGZUP,AMPL(1,2496))
C     Counter-term amplitude(s) for loop diagram number 1771
      CALL MP_R2_GGZ_0(W(4),W(221),W(3),R2_GGZUP,AMPL(1,2497))
      CALL MP_R2_GGZ_0(W(4),W(221),W(3),R2_GGZUP,AMPL(1,2498))
C     Counter-term amplitude(s) for loop diagram number 1772
      CALL MP_R2_GGZ_0(W(16),W(217),W(3),R2_GGZUP,AMPL(1,2499))
      CALL MP_R2_GGZ_0(W(16),W(217),W(3),R2_GGZUP,AMPL(1,2500))
C     Counter-term amplitude(s) for loop diagram number 1782
      CALL MP_R2_GGZ_0(W(6),W(222),W(3),R2_GGZUP,AMPL(1,2501))
      CALL MP_R2_GGZ_0(W(6),W(222),W(3),R2_GGZUP,AMPL(1,2502))
C     Counter-term amplitude(s) for loop diagram number 1783
      CALL MP_R2_GGZ_0(W(6),W(223),W(3),R2_GGZUP,AMPL(1,2503))
      CALL MP_R2_GGZ_0(W(6),W(223),W(3),R2_GGZUP,AMPL(1,2504))
C     Counter-term amplitude(s) for loop diagram number 1786
      CALL MP_R2_GGZ_0(W(5),W(224),W(3),R2_GGZUP,AMPL(1,2505))
      CALL MP_R2_GGZ_0(W(5),W(224),W(3),R2_GGZUP,AMPL(1,2506))
C     Counter-term amplitude(s) for loop diagram number 1787
      CALL MP_R2_GGZ_0(W(5),W(225),W(3),R2_GGZUP,AMPL(1,2507))
      CALL MP_R2_GGZ_0(W(5),W(225),W(3),R2_GGZUP,AMPL(1,2508))
C     Counter-term amplitude(s) for loop diagram number 1790
      CALL MP_R2_GGZ_0(W(4),W(226),W(3),R2_GGZUP,AMPL(1,2509))
      CALL MP_R2_GGZ_0(W(4),W(226),W(3),R2_GGZUP,AMPL(1,2510))
C     Counter-term amplitude(s) for loop diagram number 1791
      CALL MP_R2_GGZ_0(W(4),W(227),W(3),R2_GGZUP,AMPL(1,2511))
      CALL MP_R2_GGZ_0(W(4),W(227),W(3),R2_GGZUP,AMPL(1,2512))
C     Counter-term amplitude(s) for loop diagram number 1794
      CALL MP_R2_GGZ_0(W(5),W(166),W(3),R2_GGZDOWN,AMPL(1,2513))
C     Counter-term amplitude(s) for loop diagram number 1795
      CALL MP_R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2514))
      CALL MP_R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2515))
C     Counter-term amplitude(s) for loop diagram number 1796
      CALL MP_R2_GGZ_0(W(6),W(167),W(3),R2_GGZDOWN,AMPL(1,2516))
C     Counter-term amplitude(s) for loop diagram number 1798
      CALL MP_R2_GGZ_0(W(4),W(169),W(3),R2_GGZDOWN,AMPL(1,2517))
C     Counter-term amplitude(s) for loop diagram number 1799
      CALL MP_R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2518))
      CALL MP_R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2519))
C     Counter-term amplitude(s) for loop diagram number 1800
      CALL MP_R2_GGZ_0(W(6),W(170),W(3),R2_GGZDOWN,AMPL(1,2520))
C     Counter-term amplitude(s) for loop diagram number 1802
      CALL MP_R2_GGZ_0(W(4),W(172),W(3),R2_GGZDOWN,AMPL(1,2521))
C     Counter-term amplitude(s) for loop diagram number 1803
      CALL MP_R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2522))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2523))
C     Counter-term amplitude(s) for loop diagram number 1804
      CALL MP_R2_GGZ_0(W(5),W(173),W(3),R2_GGZDOWN,AMPL(1,2524))
C     Counter-term amplitude(s) for loop diagram number 1812
      CALL MP_R2_GGZ_0(W(6),W(174),W(3),R2_GGZDOWN,AMPL(1,2525))
      CALL MP_R2_GGZ_0(W(6),W(175),W(3),R2_GGZDOWN,AMPL(1,2526))
      CALL MP_R2_GGZ_0(W(6),W(176),W(3),R2_GGZDOWN,AMPL(1,2527))
C     Counter-term amplitude(s) for loop diagram number 1813
      CALL MP_R2_GGZ_0(W(5),W(177),W(3),R2_GGZDOWN,AMPL(1,2528))
      CALL MP_R2_GGZ_0(W(5),W(178),W(3),R2_GGZDOWN,AMPL(1,2529))
      CALL MP_R2_GGZ_0(W(5),W(179),W(3),R2_GGZDOWN,AMPL(1,2530))
C     Counter-term amplitude(s) for loop diagram number 1814
      CALL MP_R2_GGZ_0(W(4),W(180),W(3),R2_GGZDOWN,AMPL(1,2531))
      CALL MP_R2_GGZ_0(W(4),W(181),W(3),R2_GGZDOWN,AMPL(1,2532))
      CALL MP_R2_GGZ_0(W(4),W(182),W(3),R2_GGZDOWN,AMPL(1,2533))
C     Counter-term amplitude(s) for loop diagram number 1815
      CALL MP_R2_GGZ_0(W(164),W(65),W(3),R2_GGZDOWN,AMPL(1,2534))
C     Counter-term amplitude(s) for loop diagram number 1816
      CALL MP_R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2535))
      CALL MP_R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALDOWN
     $ ,AMPL(1,2536))
C     Counter-term amplitude(s) for loop diagram number 1817
      CALL MP_R2_GGZ_0(W(6),W(183),W(3),R2_GGZDOWN,AMPL(1,2537))
C     Counter-term amplitude(s) for loop diagram number 1818
      CALL MP_R2_GGZ_0(W(16),W(171),W(3),R2_GGZDOWN,AMPL(1,2538))
C     Counter-term amplitude(s) for loop diagram number 1827
      CALL MP_R2_GGZ_0(W(164),W(66),W(3),R2_GGZDOWN,AMPL(1,2539))
C     Counter-term amplitude(s) for loop diagram number 1828
      CALL MP_R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2540))
      CALL MP_R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALDOWN
     $ ,AMPL(1,2541))
C     Counter-term amplitude(s) for loop diagram number 1829
      CALL MP_R2_GGZ_0(W(5),W(184),W(3),R2_GGZDOWN,AMPL(1,2542))
C     Counter-term amplitude(s) for loop diagram number 1830
      CALL MP_R2_GGZ_0(W(14),W(168),W(3),R2_GGZDOWN,AMPL(1,2543))
C     Counter-term amplitude(s) for loop diagram number 1852
      CALL MP_R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2544))
      CALL MP_R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALDOWN
     $ ,AMPL(1,2545))
C     Counter-term amplitude(s) for loop diagram number 1854
      CALL MP_R2_GGZ_0(W(4),W(185),W(3),R2_GGZDOWN,AMPL(1,2546))
C     Counter-term amplitude(s) for loop diagram number 1855
      CALL MP_R2_GGZ_0(W(164),W(67),W(3),R2_GGZDOWN,AMPL(1,2547))
C     Counter-term amplitude(s) for loop diagram number 1857
      CALL MP_R2_GGZ_0(W(11),W(165),W(3),R2_GGZDOWN,AMPL(1,2548))
C     Counter-term amplitude(s) for loop diagram number 1890
      CALL MP_R2_GGZ_0(W(164),W(18),W(3),R2_GGZDOWN,AMPL(1,2549))
      CALL MP_R2_GGZ_0(W(164),W(19),W(3),R2_GGZDOWN,AMPL(1,2550))
      CALL MP_R2_GGZ_0(W(164),W(20),W(3),R2_GGZDOWN,AMPL(1,2551))
C     Counter-term amplitude(s) for loop diagram number 1892
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(44),R2_GGQ,R2_GGB,AMPL(1,2552))
C     Counter-term amplitude(s) for loop diagram number 1893
      CALL MP_VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,2553))
C     Counter-term amplitude(s) for loop diagram number 1895
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(53),R2_GGQ,R2_GGB,AMPL(1,2554))
C     Counter-term amplitude(s) for loop diagram number 1896
      CALL MP_VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,2555))
C     Counter-term amplitude(s) for loop diagram number 1898
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(62),R2_GGQ,R2_GGB,AMPL(1,2556))
C     Counter-term amplitude(s) for loop diagram number 1899
      CALL MP_VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,2557))
C     Counter-term amplitude(s) for loop diagram number 1901
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(65),R2_GGQ,R2_GGB,AMPL(1,2558))
C     Counter-term amplitude(s) for loop diagram number 1902
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(68),R2_GGQ,R2_GGB,AMPL(1,2559))
C     Counter-term amplitude(s) for loop diagram number 1903
      CALL MP_VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,2560))
C     Counter-term amplitude(s) for loop diagram number 1905
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(70),R2_GGQ,R2_GGB,AMPL(1,2561))
C     Counter-term amplitude(s) for loop diagram number 1906
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(66),R2_GGQ,R2_GGB,AMPL(1,2562))
C     Counter-term amplitude(s) for loop diagram number 1907
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(73),R2_GGQ,R2_GGB,AMPL(1,2563))
C     Counter-term amplitude(s) for loop diagram number 1908
      CALL MP_VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,2564))
C     Counter-term amplitude(s) for loop diagram number 1910
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(75),R2_GGQ,R2_GGB,AMPL(1,2565))
C     Counter-term amplitude(s) for loop diagram number 1911
      CALL MP_VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,2566))
C     Counter-term amplitude(s) for loop diagram number 1912
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,2567))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,2568))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,2569))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,2570))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,2571))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,2572))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,2573))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2574))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2575))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,2576))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,2577))
C     Counter-term amplitude(s) for loop diagram number 1913
      CALL MP_VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,2578))
C     Counter-term amplitude(s) for loop diagram number 1915
      CALL MP_VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,2579))
C     Counter-term amplitude(s) for loop diagram number 1916
      CALL MP_VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,2580))
C     Counter-term amplitude(s) for loop diagram number 1917
      CALL MP_VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,2581))
C     Counter-term amplitude(s) for loop diagram number 1927
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(67),R2_GGQ,R2_GGB,AMPL(1,2582))
C     Counter-term amplitude(s) for loop diagram number 1928
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(79),R2_GGQ,R2_GGB,AMPL(1,2583))
C     Counter-term amplitude(s) for loop diagram number 1929
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(80),R2_GGQ,R2_GGB,AMPL(1,2584))
C     Counter-term amplitude(s) for loop diagram number 1930
      CALL MP_VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2585))
C     Counter-term amplitude(s) for loop diagram number 1931
      CALL MP_VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2586))
C     Counter-term amplitude(s) for loop diagram number 1934
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(18),R2_GGQ,R2_GGB,AMPL(1,2587))
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(19),R2_GGQ,R2_GGB,AMPL(1,2588))
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(20),R2_GGQ,R2_GGB,AMPL(1,2589))
C     Counter-term amplitude(s) for loop diagram number 1935
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(92),R2_GGQ,R2_GGB,AMPL(1,2590))
C     Counter-term amplitude(s) for loop diagram number 1936
      CALL MP_VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2591))
C     Counter-term amplitude(s) for loop diagram number 1938
      CALL MP_R2_GGZ_0(W(6),W(186),W(3),R2_GGZDOWN,AMPL(1,2592))
C     Counter-term amplitude(s) for loop diagram number 1940
      CALL MP_R2_GGZ_0(W(5),W(187),W(3),R2_GGZDOWN,AMPL(1,2593))
C     Counter-term amplitude(s) for loop diagram number 1942
      CALL MP_R2_GGZ_0(W(5),W(189),W(3),R2_GGZDOWN,AMPL(1,2594))
C     Counter-term amplitude(s) for loop diagram number 1943
      CALL MP_R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2595))
      CALL MP_R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2596))
C     Counter-term amplitude(s) for loop diagram number 1944
      CALL MP_R2_GGZ_0(W(6),W(190),W(3),R2_GGZDOWN,AMPL(1,2597))
C     Counter-term amplitude(s) for loop diagram number 1946
      CALL MP_R2_GGZ_0(W(6),W(191),W(3),R2_GGZDOWN,AMPL(1,2598))
C     Counter-term amplitude(s) for loop diagram number 1947
      CALL MP_R2_GGZ_0(W(5),W(192),W(3),R2_GGZDOWN,AMPL(1,2599))
C     Counter-term amplitude(s) for loop diagram number 1948
      CALL MP_R2_GGZ_0(W(11),W(188),W(3),R2_GGZDOWN,AMPL(1,2600))
C     Counter-term amplitude(s) for loop diagram number 1958
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(105),R2_GGQ,R2_GGB,AMPL(1,2601)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1959
      CALL MP_VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2602))
C     Counter-term amplitude(s) for loop diagram number 1961
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(112),R2_GGQ,R2_GGB,AMPL(1,2603)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1962
      CALL MP_VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2604))
C     Counter-term amplitude(s) for loop diagram number 1964
      CALL MP_R2_GGZ_0(W(6),W(193),W(3),R2_GGZDOWN,AMPL(1,2605))
C     Counter-term amplitude(s) for loop diagram number 1966
      CALL MP_R2_GGZ_0(W(4),W(194),W(3),R2_GGZDOWN,AMPL(1,2606))
C     Counter-term amplitude(s) for loop diagram number 1968
      CALL MP_R2_GGZ_0(W(4),W(196),W(3),R2_GGZDOWN,AMPL(1,2607))
C     Counter-term amplitude(s) for loop diagram number 1969
      CALL MP_R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2608))
      CALL MP_R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2609))
C     Counter-term amplitude(s) for loop diagram number 1970
      CALL MP_R2_GGZ_0(W(6),W(197),W(3),R2_GGZDOWN,AMPL(1,2610))
C     Counter-term amplitude(s) for loop diagram number 1972
      CALL MP_R2_GGZ_0(W(6),W(198),W(3),R2_GGZDOWN,AMPL(1,2611))
C     Counter-term amplitude(s) for loop diagram number 1973
      CALL MP_R2_GGZ_0(W(4),W(199),W(3),R2_GGZDOWN,AMPL(1,2612))
C     Counter-term amplitude(s) for loop diagram number 1974
      CALL MP_R2_GGZ_0(W(14),W(195),W(3),R2_GGZDOWN,AMPL(1,2613))
C     Counter-term amplitude(s) for loop diagram number 1984
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(124),R2_GGQ,R2_GGB,AMPL(1,2614)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1985
      CALL MP_VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2615))
C     Counter-term amplitude(s) for loop diagram number 1987
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(130),R2_GGQ,R2_GGB,AMPL(1,2616)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1988
      CALL MP_VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2617))
C     Counter-term amplitude(s) for loop diagram number 1990
      CALL MP_R2_GGZ_0(W(5),W(200),W(3),R2_GGZDOWN,AMPL(1,2618))
C     Counter-term amplitude(s) for loop diagram number 1992
      CALL MP_R2_GGZ_0(W(4),W(201),W(3),R2_GGZDOWN,AMPL(1,2619))
C     Counter-term amplitude(s) for loop diagram number 1994
      CALL MP_R2_GGZ_0(W(4),W(203),W(3),R2_GGZDOWN,AMPL(1,2620))
C     Counter-term amplitude(s) for loop diagram number 1995
      CALL MP_R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2621))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2622))
C     Counter-term amplitude(s) for loop diagram number 1996
      CALL MP_R2_GGZ_0(W(5),W(204),W(3),R2_GGZDOWN,AMPL(1,2623))
C     Counter-term amplitude(s) for loop diagram number 1998
      CALL MP_R2_GGZ_0(W(5),W(205),W(3),R2_GGZDOWN,AMPL(1,2624))
C     Counter-term amplitude(s) for loop diagram number 1999
      CALL MP_R2_GGZ_0(W(4),W(206),W(3),R2_GGZDOWN,AMPL(1,2625))
C     Counter-term amplitude(s) for loop diagram number 2000
      CALL MP_R2_GGZ_0(W(16),W(202),W(3),R2_GGZDOWN,AMPL(1,2626))
C     Counter-term amplitude(s) for loop diagram number 2010
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(141),R2_GGQ,R2_GGB,AMPL(1,2627)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2011
      CALL MP_VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2628))
C     Counter-term amplitude(s) for loop diagram number 2013
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(65),R2_GGQ,R2_GGB,AMPL(1,2629))
C     Counter-term amplitude(s) for loop diagram number 2014
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(143),R2_GGQ,R2_GGB,AMPL(1,2630)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2015
      CALL MP_VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2631))
C     Counter-term amplitude(s) for loop diagram number 2017
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(145),R2_GGQ,R2_GGB,AMPL(1,2632)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2018
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(66),R2_GGQ,R2_GGB,AMPL(1,2633))
C     Counter-term amplitude(s) for loop diagram number 2019
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(147),R2_GGQ,R2_GGB,AMPL(1,2634)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2020
      CALL MP_VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2635))
C     Counter-term amplitude(s) for loop diagram number 2022
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(149),R2_GGQ,R2_GGB,AMPL(1,2636)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2023
      CALL MP_VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2637))
C     Counter-term amplitude(s) for loop diagram number 2024
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2638))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2639))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2640))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2641))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2642))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2643))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,2644))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2645))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2646))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2647))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2648))
C     Counter-term amplitude(s) for loop diagram number 2025
      CALL MP_VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2649))
C     Counter-term amplitude(s) for loop diagram number 2027
      CALL MP_VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2650))
C     Counter-term amplitude(s) for loop diagram number 2028
      CALL MP_VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2651))
C     Counter-term amplitude(s) for loop diagram number 2029
      CALL MP_VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2652))
C     Counter-term amplitude(s) for loop diagram number 2039
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(67),R2_GGQ,R2_GGB,AMPL(1,2653))
C     Counter-term amplitude(s) for loop diagram number 2040
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(151),R2_GGQ,R2_GGB,AMPL(1,2654)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2041
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(153),R2_GGQ,R2_GGB,AMPL(1,2655)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2042
      CALL MP_VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2656))
C     Counter-term amplitude(s) for loop diagram number 2043
      CALL MP_VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2657))
C     Counter-term amplitude(s) for loop diagram number 2046
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(18),R2_GGQ,R2_GGB,AMPL(1,2658))
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(19),R2_GGQ,R2_GGB,AMPL(1,2659))
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(20),R2_GGQ,R2_GGB,AMPL(1,2660))
C     Counter-term amplitude(s) for loop diagram number 2047
      CALL MP_R2_GGZ_0(W(5),W(208),W(3),R2_GGZDOWN,AMPL(1,2661))
C     Counter-term amplitude(s) for loop diagram number 2048
      CALL MP_R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2662))
      CALL MP_R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2663))
C     Counter-term amplitude(s) for loop diagram number 2049
      CALL MP_R2_GGZ_0(W(6),W(209),W(3),R2_GGZDOWN,AMPL(1,2664))
C     Counter-term amplitude(s) for loop diagram number 2051
      CALL MP_R2_GGZ_0(W(6),W(210),W(3),R2_GGZDOWN,AMPL(1,2665))
C     Counter-term amplitude(s) for loop diagram number 2052
      CALL MP_R2_GGZ_0(W(5),W(211),W(3),R2_GGZDOWN,AMPL(1,2666))
C     Counter-term amplitude(s) for loop diagram number 2053
      CALL MP_R2_GGZ_0(W(11),W(207),W(3),R2_GGZDOWN,AMPL(1,2667))
C     Counter-term amplitude(s) for loop diagram number 2063
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(159),R2_GGQ,R2_GGB,AMPL(1,2668)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2064
      CALL MP_VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2669))
C     Counter-term amplitude(s) for loop diagram number 2066
      CALL MP_R2_GGZ_0(W(4),W(213),W(3),R2_GGZDOWN,AMPL(1,2670))
C     Counter-term amplitude(s) for loop diagram number 2067
      CALL MP_R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2671))
      CALL MP_R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2672))
C     Counter-term amplitude(s) for loop diagram number 2068
      CALL MP_R2_GGZ_0(W(6),W(214),W(3),R2_GGZDOWN,AMPL(1,2673))
C     Counter-term amplitude(s) for loop diagram number 2070
      CALL MP_R2_GGZ_0(W(6),W(215),W(3),R2_GGZDOWN,AMPL(1,2674))
C     Counter-term amplitude(s) for loop diagram number 2071
      CALL MP_R2_GGZ_0(W(4),W(216),W(3),R2_GGZDOWN,AMPL(1,2675))
C     Counter-term amplitude(s) for loop diagram number 2072
      CALL MP_R2_GGZ_0(W(14),W(212),W(3),R2_GGZDOWN,AMPL(1,2676))
C     Counter-term amplitude(s) for loop diagram number 2082
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(161),R2_GGQ,R2_GGB,AMPL(1,2677)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2083
      CALL MP_VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2678))
C     Counter-term amplitude(s) for loop diagram number 2085
      CALL MP_R2_GGZ_0(W(4),W(218),W(3),R2_GGZDOWN,AMPL(1,2679))
C     Counter-term amplitude(s) for loop diagram number 2086
      CALL MP_R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECDOWN,AMPL(1
     $ ,2680))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2681))
C     Counter-term amplitude(s) for loop diagram number 2087
      CALL MP_R2_GGZ_0(W(5),W(219),W(3),R2_GGZDOWN,AMPL(1,2682))
C     Counter-term amplitude(s) for loop diagram number 2089
      CALL MP_R2_GGZ_0(W(5),W(220),W(3),R2_GGZDOWN,AMPL(1,2683))
C     Counter-term amplitude(s) for loop diagram number 2090
      CALL MP_R2_GGZ_0(W(4),W(221),W(3),R2_GGZDOWN,AMPL(1,2684))
C     Counter-term amplitude(s) for loop diagram number 2091
      CALL MP_R2_GGZ_0(W(16),W(217),W(3),R2_GGZDOWN,AMPL(1,2685))
C     Counter-term amplitude(s) for loop diagram number 2101
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(163),R2_GGQ,R2_GGB,AMPL(1,2686)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2102
      CALL MP_VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2687))
C     Counter-term amplitude(s) for loop diagram number 2104
      CALL MP_R2_GGZ_0(W(6),W(222),W(3),R2_GGZDOWN,AMPL(1,2688))
C     Counter-term amplitude(s) for loop diagram number 2105
      CALL MP_R2_GGZ_0(W(6),W(223),W(3),R2_GGZDOWN,AMPL(1,2689))
C     Counter-term amplitude(s) for loop diagram number 2108
      CALL MP_R2_GGZ_0(W(5),W(224),W(3),R2_GGZDOWN,AMPL(1,2690))
C     Counter-term amplitude(s) for loop diagram number 2109
      CALL MP_R2_GGZ_0(W(5),W(225),W(3),R2_GGZDOWN,AMPL(1,2691))
C     Counter-term amplitude(s) for loop diagram number 2112
      CALL MP_R2_GGZ_0(W(4),W(226),W(3),R2_GGZDOWN,AMPL(1,2692))
C     Counter-term amplitude(s) for loop diagram number 2113
      CALL MP_R2_GGZ_0(W(4),W(227),W(3),R2_GGZDOWN,AMPL(1,2693))
C     Counter-term amplitude(s) for loop diagram number 2116
      CALL MP_R2_GGZ_0(W(5),W(166),W(3),R2_GGZUP,AMPL(1,2694))
C     Counter-term amplitude(s) for loop diagram number 2117
      CALL MP_R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECUP,AMPL(1,2695)
     $ )
      CALL MP_R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2696))
C     Counter-term amplitude(s) for loop diagram number 2118
      CALL MP_R2_GGZ_0(W(6),W(167),W(3),R2_GGZUP,AMPL(1,2697))
C     Counter-term amplitude(s) for loop diagram number 2120
      CALL MP_R2_GGZ_0(W(4),W(169),W(3),R2_GGZUP,AMPL(1,2698))
C     Counter-term amplitude(s) for loop diagram number 2121
      CALL MP_R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECUP,AMPL(1,2699)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2700))
C     Counter-term amplitude(s) for loop diagram number 2122
      CALL MP_R2_GGZ_0(W(6),W(170),W(3),R2_GGZUP,AMPL(1,2701))
C     Counter-term amplitude(s) for loop diagram number 2124
      CALL MP_R2_GGZ_0(W(4),W(172),W(3),R2_GGZUP,AMPL(1,2702))
C     Counter-term amplitude(s) for loop diagram number 2125
      CALL MP_R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECUP,AMPL(1,2703)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2704))
C     Counter-term amplitude(s) for loop diagram number 2126
      CALL MP_R2_GGZ_0(W(5),W(173),W(3),R2_GGZUP,AMPL(1,2705))
C     Counter-term amplitude(s) for loop diagram number 2134
      CALL MP_R2_GGZ_0(W(6),W(174),W(3),R2_GGZUP,AMPL(1,2706))
      CALL MP_R2_GGZ_0(W(6),W(175),W(3),R2_GGZUP,AMPL(1,2707))
      CALL MP_R2_GGZ_0(W(6),W(176),W(3),R2_GGZUP,AMPL(1,2708))
C     Counter-term amplitude(s) for loop diagram number 2135
      CALL MP_R2_GGZ_0(W(5),W(177),W(3),R2_GGZUP,AMPL(1,2709))
      CALL MP_R2_GGZ_0(W(5),W(178),W(3),R2_GGZUP,AMPL(1,2710))
      CALL MP_R2_GGZ_0(W(5),W(179),W(3),R2_GGZUP,AMPL(1,2711))
C     Counter-term amplitude(s) for loop diagram number 2136
      CALL MP_R2_GGZ_0(W(4),W(180),W(3),R2_GGZUP,AMPL(1,2712))
      CALL MP_R2_GGZ_0(W(4),W(181),W(3),R2_GGZUP,AMPL(1,2713))
      CALL MP_R2_GGZ_0(W(4),W(182),W(3),R2_GGZUP,AMPL(1,2714))
C     Counter-term amplitude(s) for loop diagram number 2137
      CALL MP_R2_GGZ_0(W(164),W(65),W(3),R2_GGZUP,AMPL(1,2715))
C     Counter-term amplitude(s) for loop diagram number 2138
      CALL MP_R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECUP,AMPL(1
     $ ,2716))
      CALL MP_R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2717))
C     Counter-term amplitude(s) for loop diagram number 2139
      CALL MP_R2_GGZ_0(W(6),W(183),W(3),R2_GGZUP,AMPL(1,2718))
C     Counter-term amplitude(s) for loop diagram number 2140
      CALL MP_R2_GGZ_0(W(16),W(171),W(3),R2_GGZUP,AMPL(1,2719))
C     Counter-term amplitude(s) for loop diagram number 2149
      CALL MP_R2_GGZ_0(W(164),W(66),W(3),R2_GGZUP,AMPL(1,2720))
C     Counter-term amplitude(s) for loop diagram number 2150
      CALL MP_R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECUP,AMPL(1
     $ ,2721))
      CALL MP_R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2722))
C     Counter-term amplitude(s) for loop diagram number 2151
      CALL MP_R2_GGZ_0(W(5),W(184),W(3),R2_GGZUP,AMPL(1,2723))
C     Counter-term amplitude(s) for loop diagram number 2152
      CALL MP_R2_GGZ_0(W(14),W(168),W(3),R2_GGZUP,AMPL(1,2724))
C     Counter-term amplitude(s) for loop diagram number 2174
      CALL MP_R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECUP,AMPL(1
     $ ,2725))
      CALL MP_R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2726))
C     Counter-term amplitude(s) for loop diagram number 2176
      CALL MP_R2_GGZ_0(W(4),W(185),W(3),R2_GGZUP,AMPL(1,2727))
C     Counter-term amplitude(s) for loop diagram number 2177
      CALL MP_R2_GGZ_0(W(164),W(67),W(3),R2_GGZUP,AMPL(1,2728))
C     Counter-term amplitude(s) for loop diagram number 2179
      CALL MP_R2_GGZ_0(W(11),W(165),W(3),R2_GGZUP,AMPL(1,2729))
C     Counter-term amplitude(s) for loop diagram number 2212
      CALL MP_R2_GGZ_0(W(164),W(18),W(3),R2_GGZUP,AMPL(1,2730))
      CALL MP_R2_GGZ_0(W(164),W(19),W(3),R2_GGZUP,AMPL(1,2731))
      CALL MP_R2_GGZ_0(W(164),W(20),W(3),R2_GGZUP,AMPL(1,2732))
C     Counter-term amplitude(s) for loop diagram number 2214
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(44),R2_GGQ,R2_GGT,AMPL(1,2733))
C     Counter-term amplitude(s) for loop diagram number 2215
      CALL MP_VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,2734))
C     Counter-term amplitude(s) for loop diagram number 2217
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(53),R2_GGQ,R2_GGT,AMPL(1,2735))
C     Counter-term amplitude(s) for loop diagram number 2218
      CALL MP_VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,2736))
C     Counter-term amplitude(s) for loop diagram number 2220
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(62),R2_GGQ,R2_GGT,AMPL(1,2737))
C     Counter-term amplitude(s) for loop diagram number 2221
      CALL MP_VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,2738))
C     Counter-term amplitude(s) for loop diagram number 2223
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(65),R2_GGQ,R2_GGT,AMPL(1,2739))
C     Counter-term amplitude(s) for loop diagram number 2224
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(68),R2_GGQ,R2_GGT,AMPL(1,2740))
C     Counter-term amplitude(s) for loop diagram number 2225
      CALL MP_VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,2741))
C     Counter-term amplitude(s) for loop diagram number 2227
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(70),R2_GGQ,R2_GGT,AMPL(1,2742))
C     Counter-term amplitude(s) for loop diagram number 2228
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(66),R2_GGQ,R2_GGT,AMPL(1,2743))
C     Counter-term amplitude(s) for loop diagram number 2229
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(73),R2_GGQ,R2_GGT,AMPL(1,2744))
C     Counter-term amplitude(s) for loop diagram number 2230
      CALL MP_VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,2745))
C     Counter-term amplitude(s) for loop diagram number 2232
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(75),R2_GGQ,R2_GGT,AMPL(1,2746))
C     Counter-term amplitude(s) for loop diagram number 2233
      CALL MP_VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,2747))
C     Counter-term amplitude(s) for loop diagram number 2234
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,2748))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,2749))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1
     $ ,2750))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,2751))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,2752))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1
     $ ,2753))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,2754))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2755))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2756))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,2757))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1
     $ ,2758))
C     Counter-term amplitude(s) for loop diagram number 2235
      CALL MP_VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,2759))
C     Counter-term amplitude(s) for loop diagram number 2237
      CALL MP_VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,2760))
C     Counter-term amplitude(s) for loop diagram number 2238
      CALL MP_VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,2761))
C     Counter-term amplitude(s) for loop diagram number 2239
      CALL MP_VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,2762))
C     Counter-term amplitude(s) for loop diagram number 2249
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(67),R2_GGQ,R2_GGT,AMPL(1,2763))
C     Counter-term amplitude(s) for loop diagram number 2250
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(79),R2_GGQ,R2_GGT,AMPL(1,2764))
C     Counter-term amplitude(s) for loop diagram number 2251
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(80),R2_GGQ,R2_GGT,AMPL(1,2765))
C     Counter-term amplitude(s) for loop diagram number 2252
      CALL MP_VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2766))
C     Counter-term amplitude(s) for loop diagram number 2253
      CALL MP_VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2767))
C     Counter-term amplitude(s) for loop diagram number 2256
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(18),R2_GGQ,R2_GGT,AMPL(1,2768))
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(19),R2_GGQ,R2_GGT,AMPL(1,2769))
      CALL MP_R2_GG_1_R2_GG_3_0(W(17),W(20),R2_GGQ,R2_GGT,AMPL(1,2770))
C     Counter-term amplitude(s) for loop diagram number 2257
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(92),R2_GGQ,R2_GGT,AMPL(1,2771))
C     Counter-term amplitude(s) for loop diagram number 2258
      CALL MP_VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2772))
C     Counter-term amplitude(s) for loop diagram number 2260
      CALL MP_R2_GGZ_0(W(6),W(186),W(3),R2_GGZUP,AMPL(1,2773))
C     Counter-term amplitude(s) for loop diagram number 2262
      CALL MP_R2_GGZ_0(W(5),W(187),W(3),R2_GGZUP,AMPL(1,2774))
C     Counter-term amplitude(s) for loop diagram number 2264
      CALL MP_R2_GGZ_0(W(5),W(189),W(3),R2_GGZUP,AMPL(1,2775))
C     Counter-term amplitude(s) for loop diagram number 2265
      CALL MP_R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECUP,AMPL(1,2776)
     $ )
      CALL MP_R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2777))
C     Counter-term amplitude(s) for loop diagram number 2266
      CALL MP_R2_GGZ_0(W(6),W(190),W(3),R2_GGZUP,AMPL(1,2778))
C     Counter-term amplitude(s) for loop diagram number 2268
      CALL MP_R2_GGZ_0(W(6),W(191),W(3),R2_GGZUP,AMPL(1,2779))
C     Counter-term amplitude(s) for loop diagram number 2269
      CALL MP_R2_GGZ_0(W(5),W(192),W(3),R2_GGZUP,AMPL(1,2780))
C     Counter-term amplitude(s) for loop diagram number 2270
      CALL MP_R2_GGZ_0(W(11),W(188),W(3),R2_GGZUP,AMPL(1,2781))
C     Counter-term amplitude(s) for loop diagram number 2280
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(105),R2_GGQ,R2_GGT,AMPL(1,2782)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2281
      CALL MP_VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2783))
C     Counter-term amplitude(s) for loop diagram number 2283
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(112),R2_GGQ,R2_GGT,AMPL(1,2784)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2284
      CALL MP_VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2785))
C     Counter-term amplitude(s) for loop diagram number 2286
      CALL MP_R2_GGZ_0(W(6),W(193),W(3),R2_GGZUP,AMPL(1,2786))
C     Counter-term amplitude(s) for loop diagram number 2288
      CALL MP_R2_GGZ_0(W(4),W(194),W(3),R2_GGZUP,AMPL(1,2787))
C     Counter-term amplitude(s) for loop diagram number 2290
      CALL MP_R2_GGZ_0(W(4),W(196),W(3),R2_GGZUP,AMPL(1,2788))
C     Counter-term amplitude(s) for loop diagram number 2291
      CALL MP_R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECUP,AMPL(1,2789)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2790))
C     Counter-term amplitude(s) for loop diagram number 2292
      CALL MP_R2_GGZ_0(W(6),W(197),W(3),R2_GGZUP,AMPL(1,2791))
C     Counter-term amplitude(s) for loop diagram number 2294
      CALL MP_R2_GGZ_0(W(6),W(198),W(3),R2_GGZUP,AMPL(1,2792))
C     Counter-term amplitude(s) for loop diagram number 2295
      CALL MP_R2_GGZ_0(W(4),W(199),W(3),R2_GGZUP,AMPL(1,2793))
C     Counter-term amplitude(s) for loop diagram number 2296
      CALL MP_R2_GGZ_0(W(14),W(195),W(3),R2_GGZUP,AMPL(1,2794))
C     Counter-term amplitude(s) for loop diagram number 2306
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(124),R2_GGQ,R2_GGT,AMPL(1,2795)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2307
      CALL MP_VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2796))
C     Counter-term amplitude(s) for loop diagram number 2309
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(130),R2_GGQ,R2_GGT,AMPL(1,2797)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2310
      CALL MP_VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2798))
C     Counter-term amplitude(s) for loop diagram number 2312
      CALL MP_R2_GGZ_0(W(5),W(200),W(3),R2_GGZUP,AMPL(1,2799))
C     Counter-term amplitude(s) for loop diagram number 2314
      CALL MP_R2_GGZ_0(W(4),W(201),W(3),R2_GGZUP,AMPL(1,2800))
C     Counter-term amplitude(s) for loop diagram number 2316
      CALL MP_R2_GGZ_0(W(4),W(203),W(3),R2_GGZUP,AMPL(1,2801))
C     Counter-term amplitude(s) for loop diagram number 2317
      CALL MP_R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECUP,AMPL(1,2802)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2803))
C     Counter-term amplitude(s) for loop diagram number 2318
      CALL MP_R2_GGZ_0(W(5),W(204),W(3),R2_GGZUP,AMPL(1,2804))
C     Counter-term amplitude(s) for loop diagram number 2320
      CALL MP_R2_GGZ_0(W(5),W(205),W(3),R2_GGZUP,AMPL(1,2805))
C     Counter-term amplitude(s) for loop diagram number 2321
      CALL MP_R2_GGZ_0(W(4),W(206),W(3),R2_GGZUP,AMPL(1,2806))
C     Counter-term amplitude(s) for loop diagram number 2322
      CALL MP_R2_GGZ_0(W(16),W(202),W(3),R2_GGZUP,AMPL(1,2807))
C     Counter-term amplitude(s) for loop diagram number 2332
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(141),R2_GGQ,R2_GGT,AMPL(1,2808)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2333
      CALL MP_VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2809))
C     Counter-term amplitude(s) for loop diagram number 2335
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(65),R2_GGQ,R2_GGT,AMPL(1,2810))
C     Counter-term amplitude(s) for loop diagram number 2336
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(143),R2_GGQ,R2_GGT,AMPL(1,2811)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2337
      CALL MP_VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2812))
C     Counter-term amplitude(s) for loop diagram number 2339
      CALL MP_R2_GG_1_R2_GG_3_0(W(16),W(145),R2_GGQ,R2_GGT,AMPL(1,2813)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2340
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(66),R2_GGQ,R2_GGT,AMPL(1,2814))
C     Counter-term amplitude(s) for loop diagram number 2341
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(147),R2_GGQ,R2_GGT,AMPL(1,2815)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2342
      CALL MP_VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2816))
C     Counter-term amplitude(s) for loop diagram number 2344
      CALL MP_R2_GG_1_R2_GG_3_0(W(14),W(149),R2_GGQ,R2_GGT,AMPL(1,2817)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2345
      CALL MP_VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2818))
C     Counter-term amplitude(s) for loop diagram number 2346
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2819))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2820))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1
     $ ,2821))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2822))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2823))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1
     $ ,2824))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_141_50,R2GC_142_52,AMPL(1,2825))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2826))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(34),R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2827))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2828))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1
     $ ,2829))
C     Counter-term amplitude(s) for loop diagram number 2347
      CALL MP_VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2830))
C     Counter-term amplitude(s) for loop diagram number 2349
      CALL MP_VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2831))
C     Counter-term amplitude(s) for loop diagram number 2350
      CALL MP_VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2832))
C     Counter-term amplitude(s) for loop diagram number 2351
      CALL MP_VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2833))
C     Counter-term amplitude(s) for loop diagram number 2361
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(67),R2_GGQ,R2_GGT,AMPL(1,2834))
C     Counter-term amplitude(s) for loop diagram number 2362
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(151),R2_GGQ,R2_GGT,AMPL(1,2835)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2363
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(153),R2_GGQ,R2_GGT,AMPL(1,2836)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2364
      CALL MP_VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2837))
C     Counter-term amplitude(s) for loop diagram number 2365
      CALL MP_VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2838))
C     Counter-term amplitude(s) for loop diagram number 2368
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(18),R2_GGQ,R2_GGT,AMPL(1,2839))
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(19),R2_GGQ,R2_GGT,AMPL(1,2840))
      CALL MP_R2_GG_1_R2_GG_3_0(W(34),W(20),R2_GGQ,R2_GGT,AMPL(1,2841))
C     Counter-term amplitude(s) for loop diagram number 2369
      CALL MP_R2_GGZ_0(W(5),W(208),W(3),R2_GGZUP,AMPL(1,2842))
C     Counter-term amplitude(s) for loop diagram number 2370
      CALL MP_R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECUP,AMPL(1,2843)
     $ )
      CALL MP_R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2844))
C     Counter-term amplitude(s) for loop diagram number 2371
      CALL MP_R2_GGZ_0(W(6),W(209),W(3),R2_GGZUP,AMPL(1,2845))
C     Counter-term amplitude(s) for loop diagram number 2373
      CALL MP_R2_GGZ_0(W(6),W(210),W(3),R2_GGZUP,AMPL(1,2846))
C     Counter-term amplitude(s) for loop diagram number 2374
      CALL MP_R2_GGZ_0(W(5),W(211),W(3),R2_GGZUP,AMPL(1,2847))
C     Counter-term amplitude(s) for loop diagram number 2375
      CALL MP_R2_GGZ_0(W(11),W(207),W(3),R2_GGZUP,AMPL(1,2848))
C     Counter-term amplitude(s) for loop diagram number 2385
      CALL MP_R2_GG_1_R2_GG_3_0(W(11),W(159),R2_GGQ,R2_GGT,AMPL(1,2849)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2386
      CALL MP_VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2850))
C     Counter-term amplitude(s) for loop diagram number 2388
      CALL MP_R2_GGZ_0(W(4),W(213),W(3),R2_GGZUP,AMPL(1,2851))
C     Counter-term amplitude(s) for loop diagram number 2389
      CALL MP_R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECUP,AMPL(1,2852)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2853))
C     Counter-term amplitude(s) for loop diagram number 2390
      CALL MP_R2_GGZ_0(W(6),W(214),W(3),R2_GGZUP,AMPL(1,2854))
C     Counter-term amplitude(s) for loop diagram number 2392
      CALL MP_R2_GGZ_0(W(6),W(215),W(3),R2_GGZUP,AMPL(1,2855))
C     Counter-term amplitude(s) for loop diagram number 2393
      CALL MP_R2_GGZ_0(W(4),W(216),W(3),R2_GGZUP,AMPL(1,2856))

      GOTO 1001
 2000 CONTINUE
      MP_CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

