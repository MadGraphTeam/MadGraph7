      SUBROUTINE ML5_0_HELAS_CALLS_AMPB_2(P,NHEL,H,IC)
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
      IF (FILTER_SO.AND.CT_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

      CALL FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1477))
      CALL FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1478))
      CALL FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1479))
      CALL FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1480))
      CALL FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1481))
      CALL FFV1_0(W(1),W(101),W(16),UV_GQQB_1EPS,AMPL(2,1482))
      CALL FFV1_0(W(1),W(101),W(16),UV_GQQG_1EPS,AMPL(2,1483))
      CALL FFV1_0(W(1),W(101),W(16),UV_GQQB,AMPL(1,1484))
      CALL FFV1_0(W(1),W(101),W(16),UV_GQQT,AMPL(1,1485))
C     Counter-term amplitude(s) for loop diagram number 936
      CALL FFV2_5_0(W(1),W(71),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1486))
C     Counter-term amplitude(s) for loop diagram number 938
      CALL FFV2_5_0(W(1),W(72),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1487))
C     Counter-term amplitude(s) for loop diagram number 943
      CALL FFV1_0(W(1),W(142),W(6),R2_GQQ,AMPL(1,1488))
      CALL FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1489))
      CALL FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1490))
      CALL FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1491))
      CALL FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1492))
      CALL FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1493))
      CALL FFV1_0(W(1),W(142),W(6),UV_GQQB_1EPS,AMPL(2,1494))
      CALL FFV1_0(W(1),W(142),W(6),UV_GQQG_1EPS,AMPL(2,1495))
      CALL FFV1_0(W(1),W(142),W(6),UV_GQQB,AMPL(1,1496))
      CALL FFV1_0(W(1),W(142),W(6),UV_GQQT,AMPL(1,1497))
C     Counter-term amplitude(s) for loop diagram number 951
      CALL FFV2_5_0(W(1),W(76),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1498))
C     Counter-term amplitude(s) for loop diagram number 953
      CALL FFV2_5_0(W(1),W(77),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1499))
C     Counter-term amplitude(s) for loop diagram number 958
      CALL FFV1_0(W(1),W(125),W(5),R2_GQQ,AMPL(1,1500))
      CALL FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1501))
      CALL FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1502))
      CALL FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1503))
      CALL FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1504))
      CALL FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1505))
      CALL FFV1_0(W(1),W(125),W(5),UV_GQQB_1EPS,AMPL(2,1506))
      CALL FFV1_0(W(1),W(125),W(5),UV_GQQG_1EPS,AMPL(2,1507))
      CALL FFV1_0(W(1),W(125),W(5),UV_GQQB,AMPL(1,1508))
      CALL FFV1_0(W(1),W(125),W(5),UV_GQQT,AMPL(1,1509))
C     Counter-term amplitude(s) for loop diagram number 981
      CALL FFV1_0(W(1),W(106),W(4),R2_GQQ,AMPL(1,1510))
      CALL FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1511))
      CALL FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1512))
      CALL FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1513))
      CALL FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1514))
      CALL FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1515))
      CALL FFV1_0(W(1),W(106),W(4),UV_GQQB_1EPS,AMPL(2,1516))
      CALL FFV1_0(W(1),W(106),W(4),UV_GQQG_1EPS,AMPL(2,1517))
      CALL FFV1_0(W(1),W(106),W(4),UV_GQQB,AMPL(1,1518))
      CALL FFV1_0(W(1),W(106),W(4),UV_GQQT,AMPL(1,1519))
C     Counter-term amplitude(s) for loop diagram number 984
      CALL FFV2_5_0(W(1),W(81),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1520))
C     Counter-term amplitude(s) for loop diagram number 986
      CALL FFV2_5_0(W(1),W(82),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1521))
C     Counter-term amplitude(s) for loop diagram number 1019
      CALL FFV2_5_0(W(1),W(83),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1522))
      CALL FFV2_5_0(W(1),W(84),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1523))
      CALL FFV2_5_0(W(1),W(85),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1524))
C     Counter-term amplitude(s) for loop diagram number 1030
      CALL R2_GG_1_R2_GG_2_0(W(34),W(65),R2_GGG_1,R2_GGG_2,AMPL(1,1525)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1031
      CALL VVV1_0(W(16),W(6),W(34),R2_3GG,AMPL(1,1526))
      CALL VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1527))
      CALL VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1528))
      CALL VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1529))
      CALL VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1530))
      CALL VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1531))
      CALL VVV1_0(W(16),W(6),W(34),UV_3GB_1EPS,AMPL(2,1532))
      CALL VVV1_0(W(16),W(6),W(34),UV_3GG_1EPS,AMPL(2,1533))
      CALL VVV1_0(W(16),W(6),W(34),UV_3GB,AMPL(1,1534))
      CALL VVV1_0(W(16),W(6),W(34),UV_3GT,AMPL(1,1535))
      CALL VVV1P0_1(W(6),W(34),GC_4,ZERO,ZERO,W(143))
C     Counter-term amplitude(s) for loop diagram number 1032
      CALL R2_GG_1_R2_GG_2_0(W(16),W(143),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1536))
C     Counter-term amplitude(s) for loop diagram number 1035
      CALL R2_QQ_1_0(W(35),W(86),R2_QQQ,AMPL(1,1537))
      CALL FFV1_2(W(35),W(6),GC_5,ZERO,ZERO,W(144))
C     Counter-term amplitude(s) for loop diagram number 1036
      CALL R2_QQ_1_0(W(144),W(22),R2_QQQ,AMPL(1,1538))
C     Counter-term amplitude(s) for loop diagram number 1037
      CALL FFV1_0(W(35),W(22),W(6),R2_GQQ,AMPL(1,1539))
      CALL FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1540))
      CALL FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1541))
      CALL FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1542))
      CALL FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1543))
      CALL FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1544))
      CALL FFV1_0(W(35),W(22),W(6),UV_GQQB_1EPS,AMPL(2,1545))
      CALL FFV1_0(W(35),W(22),W(6),UV_GQQG_1EPS,AMPL(2,1546))
      CALL FFV1_0(W(35),W(22),W(6),UV_GQQB,AMPL(1,1547))
      CALL FFV1_0(W(35),W(22),W(6),UV_GQQT,AMPL(1,1548))
      CALL FFV1P0_3(W(1),W(86),GC_5,ZERO,ZERO,W(145))
C     Counter-term amplitude(s) for loop diagram number 1038
      CALL R2_GG_1_R2_GG_2_0(W(16),W(145),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1549))
      CALL FFV1_2(W(1),W(65),GC_5,ZERO,ZERO,W(146))
C     Counter-term amplitude(s) for loop diagram number 1039
      CALL R2_QQ_1_0(W(146),W(22),R2_QQQ,AMPL(1,1550))
C     Counter-term amplitude(s) for loop diagram number 1042
      CALL R2_GG_1_R2_GG_2_0(W(34),W(66),R2_GGG_1,R2_GGG_2,AMPL(1,1551)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1043
      CALL VVV1_0(W(14),W(5),W(34),R2_3GG,AMPL(1,1552))
      CALL VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1553))
      CALL VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1554))
      CALL VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1555))
      CALL VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1556))
      CALL VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1557))
      CALL VVV1_0(W(14),W(5),W(34),UV_3GB_1EPS,AMPL(2,1558))
      CALL VVV1_0(W(14),W(5),W(34),UV_3GG_1EPS,AMPL(2,1559))
      CALL VVV1_0(W(14),W(5),W(34),UV_3GB,AMPL(1,1560))
      CALL VVV1_0(W(14),W(5),W(34),UV_3GT,AMPL(1,1561))
      CALL VVV1P0_1(W(5),W(34),GC_4,ZERO,ZERO,W(147))
C     Counter-term amplitude(s) for loop diagram number 1044
      CALL R2_GG_1_R2_GG_2_0(W(14),W(147),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1562))
C     Counter-term amplitude(s) for loop diagram number 1047
      CALL R2_QQ_1_0(W(36),W(88),R2_QQQ,AMPL(1,1563))
      CALL FFV1_2(W(36),W(5),GC_5,ZERO,ZERO,W(148))
C     Counter-term amplitude(s) for loop diagram number 1048
      CALL R2_QQ_1_0(W(148),W(22),R2_QQQ,AMPL(1,1564))
C     Counter-term amplitude(s) for loop diagram number 1049
      CALL FFV1_0(W(36),W(22),W(5),R2_GQQ,AMPL(1,1565))
      CALL FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1566))
      CALL FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1567))
      CALL FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1568))
      CALL FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1569))
      CALL FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1570))
      CALL FFV1_0(W(36),W(22),W(5),UV_GQQB_1EPS,AMPL(2,1571))
      CALL FFV1_0(W(36),W(22),W(5),UV_GQQG_1EPS,AMPL(2,1572))
      CALL FFV1_0(W(36),W(22),W(5),UV_GQQB,AMPL(1,1573))
      CALL FFV1_0(W(36),W(22),W(5),UV_GQQT,AMPL(1,1574))
      CALL FFV1P0_3(W(1),W(88),GC_5,ZERO,ZERO,W(149))
C     Counter-term amplitude(s) for loop diagram number 1050
      CALL R2_GG_1_R2_GG_2_0(W(14),W(149),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1575))
      CALL FFV1_2(W(1),W(66),GC_5,ZERO,ZERO,W(150))
C     Counter-term amplitude(s) for loop diagram number 1051
      CALL R2_QQ_1_0(W(150),W(22),R2_QQQ,AMPL(1,1576))
C     Counter-term amplitude(s) for loop diagram number 1054
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_44,AMPL(1,1577))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_59,AMPL(1,1578))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_44,AMPL(1,1579))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_59,AMPL(1,1580))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_59,AMPL(1,1581))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_59,AMPL(1,1582))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34),R2GC_141_51
     $ ,R2GC_142_53,AMPL(1,1583))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_144_57,R2GC_138_46,R2GC_143_55,AMPL(1,1584))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_138_46,R2GC_144_57,R2GC_143_55,AMPL(1,1585))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_139_47,AMPL(1,1586)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_49,AMPL(1,1587)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_49,AMPL(1,1588)
     $ )
      CALL VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1589))
      CALL VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1590))
      CALL VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1591))
      CALL VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1592))
      CALL VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1593))
      CALL VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1594))
      CALL VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1595))
      CALL VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1596))
      CALL VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1597))
      CALL VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1598))
      CALL VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1599))
      CALL VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1600))
      CALL VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1601))
      CALL VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1602))
      CALL VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1603))
      CALL VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1604))
      CALL VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1605))
      CALL VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB_1EPS,AMPL(2,1606))
      CALL VVVV1_0(W(4),W(5),W(6),W(34),UV_4GG_1EPS,AMPL(2,1607))
      CALL VVVV3_0(W(4),W(5),W(6),W(34),UV_4GG_1EPS,AMPL(2,1608))
      CALL VVVV4_0(W(4),W(5),W(6),W(34),UV_4GG_1EPS,AMPL(2,1609))
      CALL VVVV1_0(W(4),W(5),W(6),W(34),UV_4GB,AMPL(1,1610))
      CALL VVVV3_0(W(4),W(5),W(6),W(34),UV_4GB,AMPL(1,1611))
      CALL VVVV4_0(W(4),W(5),W(6),W(34),UV_4GB,AMPL(1,1612))
      CALL VVVV1_0(W(4),W(5),W(6),W(34),UV_4GGT,AMPL(1,1613))
      CALL VVVV3_0(W(4),W(5),W(6),W(34),UV_4GGT,AMPL(1,1614))
      CALL VVVV4_0(W(4),W(5),W(6),W(34),UV_4GGT,AMPL(1,1615))
C     Counter-term amplitude(s) for loop diagram number 1055
      CALL VVV1_0(W(4),W(5),W(143),R2_3GG,AMPL(1,1616))
      CALL VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1617))
      CALL VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1618))
      CALL VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1619))
      CALL VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1620))
      CALL VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1621))
      CALL VVV1_0(W(4),W(5),W(143),UV_3GB_1EPS,AMPL(2,1622))
      CALL VVV1_0(W(4),W(5),W(143),UV_3GG_1EPS,AMPL(2,1623))
      CALL VVV1_0(W(4),W(5),W(143),UV_3GB,AMPL(1,1624))
      CALL VVV1_0(W(4),W(5),W(143),UV_3GT,AMPL(1,1625))
C     Counter-term amplitude(s) for loop diagram number 1058
      CALL VVV1_0(W(4),W(6),W(147),R2_3GG,AMPL(1,1626))
      CALL VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1627))
      CALL VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1628))
      CALL VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1629))
      CALL VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1630))
      CALL VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1631))
      CALL VVV1_0(W(4),W(6),W(147),UV_3GB_1EPS,AMPL(2,1632))
      CALL VVV1_0(W(4),W(6),W(147),UV_3GG_1EPS,AMPL(2,1633))
      CALL VVV1_0(W(4),W(6),W(147),UV_3GB,AMPL(1,1634))
      CALL VVV1_0(W(4),W(6),W(147),UV_3GT,AMPL(1,1635))
C     Counter-term amplitude(s) for loop diagram number 1063
      CALL VVV1_0(W(4),W(6),W(149),R2_3GG,AMPL(1,1636))
      CALL VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1637))
      CALL VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1638))
      CALL VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1639))
      CALL VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1640))
      CALL VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1641))
      CALL VVV1_0(W(4),W(6),W(149),UV_3GB_1EPS,AMPL(2,1642))
      CALL VVV1_0(W(4),W(6),W(149),UV_3GG_1EPS,AMPL(2,1643))
      CALL VVV1_0(W(4),W(6),W(149),UV_3GB,AMPL(1,1644))
      CALL VVV1_0(W(4),W(6),W(149),UV_3GT,AMPL(1,1645))
C     Counter-term amplitude(s) for loop diagram number 1065
      CALL VVV1_0(W(4),W(5),W(145),R2_3GG,AMPL(1,1646))
      CALL VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1647))
      CALL VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1648))
      CALL VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1649))
      CALL VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1650))
      CALL VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1651))
      CALL VVV1_0(W(4),W(5),W(145),UV_3GB_1EPS,AMPL(2,1652))
      CALL VVV1_0(W(4),W(5),W(145),UV_3GG_1EPS,AMPL(2,1653))
      CALL VVV1_0(W(4),W(5),W(145),UV_3GB,AMPL(1,1654))
      CALL VVV1_0(W(4),W(5),W(145),UV_3GT,AMPL(1,1655))
C     Counter-term amplitude(s) for loop diagram number 1067
      CALL VVV1_0(W(4),W(11),W(34),R2_3GG,AMPL(1,1656))
      CALL VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1657))
      CALL VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1658))
      CALL VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1659))
      CALL VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1660))
      CALL VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1661))
      CALL VVV1_0(W(4),W(11),W(34),UV_3GB_1EPS,AMPL(2,1662))
      CALL VVV1_0(W(4),W(11),W(34),UV_3GG_1EPS,AMPL(2,1663))
      CALL VVV1_0(W(4),W(11),W(34),UV_3GB,AMPL(1,1664))
      CALL VVV1_0(W(4),W(11),W(34),UV_3GT,AMPL(1,1665))
C     Counter-term amplitude(s) for loop diagram number 1069
      CALL FFV1_0(W(37),W(22),W(4),R2_GQQ,AMPL(1,1666))
      CALL FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1667))
      CALL FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1668))
      CALL FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1669))
      CALL FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1670))
      CALL FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1671))
      CALL FFV1_0(W(37),W(22),W(4),UV_GQQB_1EPS,AMPL(2,1672))
      CALL FFV1_0(W(37),W(22),W(4),UV_GQQG_1EPS,AMPL(2,1673))
      CALL FFV1_0(W(37),W(22),W(4),UV_GQQB,AMPL(1,1674))
      CALL FFV1_0(W(37),W(22),W(4),UV_GQQT,AMPL(1,1675))
C     Counter-term amplitude(s) for loop diagram number 1071
      CALL R2_GG_1_R2_GG_2_0(W(34),W(67),R2_GGG_1,R2_GGG_2,AMPL(1,1676)
     $ )
      CALL VVV1P0_1(W(4),W(34),GC_4,ZERO,ZERO,W(151))
C     Counter-term amplitude(s) for loop diagram number 1073
      CALL R2_GG_1_R2_GG_2_0(W(11),W(151),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1677))
C     Counter-term amplitude(s) for loop diagram number 1075
      CALL R2_QQ_1_0(W(37),W(108),R2_QQQ,AMPL(1,1678))
      CALL FFV1_2(W(37),W(4),GC_5,ZERO,ZERO,W(152))
C     Counter-term amplitude(s) for loop diagram number 1076
      CALL R2_QQ_1_0(W(152),W(22),R2_QQQ,AMPL(1,1679))
      CALL FFV1P0_3(W(1),W(108),GC_5,ZERO,ZERO,W(153))
C     Counter-term amplitude(s) for loop diagram number 1078
      CALL R2_GG_1_R2_GG_2_0(W(11),W(153),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1680))
      CALL FFV1_2(W(1),W(67),GC_5,ZERO,ZERO,W(154))
C     Counter-term amplitude(s) for loop diagram number 1079
      CALL R2_QQ_1_0(W(154),W(22),R2_QQQ,AMPL(1,1681))
C     Counter-term amplitude(s) for loop diagram number 1081
      CALL VVV1_0(W(5),W(6),W(151),R2_3GG,AMPL(1,1682))
      CALL VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1683))
      CALL VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1684))
      CALL VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1685))
      CALL VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1686))
      CALL VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1687))
      CALL VVV1_0(W(5),W(6),W(151),UV_3GB_1EPS,AMPL(2,1688))
      CALL VVV1_0(W(5),W(6),W(151),UV_3GG_1EPS,AMPL(2,1689))
      CALL VVV1_0(W(5),W(6),W(151),UV_3GB,AMPL(1,1690))
      CALL VVV1_0(W(5),W(6),W(151),UV_3GT,AMPL(1,1691))
C     Counter-term amplitude(s) for loop diagram number 1085
      CALL VVV1_0(W(5),W(6),W(153),R2_3GG,AMPL(1,1692))
      CALL VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1693))
      CALL VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1694))
      CALL VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1695))
      CALL VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1696))
      CALL VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1697))
      CALL VVV1_0(W(5),W(6),W(153),UV_3GB_1EPS,AMPL(2,1698))
      CALL VVV1_0(W(5),W(6),W(153),UV_3GG_1EPS,AMPL(2,1699))
      CALL VVV1_0(W(5),W(6),W(153),UV_3GB,AMPL(1,1700))
      CALL VVV1_0(W(5),W(6),W(153),UV_3GT,AMPL(1,1701))
C     Counter-term amplitude(s) for loop diagram number 1093
      CALL R2_GG_1_R2_GG_2_0(W(34),W(18),R2_GGG_1,R2_GGG_2,AMPL(1,1702)
     $ )
      CALL R2_GG_1_R2_GG_2_0(W(34),W(19),R2_GGG_1,R2_GGG_2,AMPL(1,1703)
     $ )
      CALL R2_GG_1_R2_GG_2_0(W(34),W(20),R2_GGG_1,R2_GGG_2,AMPL(1,1704)
     $ )
      CALL FFV1_2(W(1),W(18),GC_5,ZERO,ZERO,W(155))
      CALL FFV1_2(W(1),W(19),GC_5,ZERO,ZERO,W(156))
      CALL FFV1_2(W(1),W(20),GC_5,ZERO,ZERO,W(157))
C     Counter-term amplitude(s) for loop diagram number 1094
      CALL R2_QQ_1_0(W(155),W(22),R2_QQQ,AMPL(1,1705))
      CALL R2_QQ_1_0(W(156),W(22),R2_QQQ,AMPL(1,1706))
      CALL R2_QQ_1_0(W(157),W(22),R2_QQQ,AMPL(1,1707))
C     Counter-term amplitude(s) for loop diagram number 1104
      CALL R2_QQ_1_0(W(37),W(116),R2_QQQ,AMPL(1,1708))
      CALL FFV2_5_2(W(37),W(3),GC_22,GC_23,ZERO,ZERO,W(158))
C     Counter-term amplitude(s) for loop diagram number 1105
      CALL R2_QQ_1_0(W(158),W(8),R2_QQQ,AMPL(1,1709))
C     Counter-term amplitude(s) for loop diagram number 1106
      CALL FFV2_5_0(W(37),W(8),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1710))
      CALL FFV1P0_3(W(1),W(116),GC_5,ZERO,ZERO,W(159))
C     Counter-term amplitude(s) for loop diagram number 1107
      CALL R2_GG_1_R2_GG_2_0(W(11),W(159),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1711))
C     Counter-term amplitude(s) for loop diagram number 1108
      CALL VVV1_0(W(5),W(6),W(159),R2_3GG,AMPL(1,1712))
      CALL VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1713))
      CALL VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1714))
      CALL VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1715))
      CALL VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1716))
      CALL VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1717))
      CALL VVV1_0(W(5),W(6),W(159),UV_3GB_1EPS,AMPL(2,1718))
      CALL VVV1_0(W(5),W(6),W(159),UV_3GG_1EPS,AMPL(2,1719))
      CALL VVV1_0(W(5),W(6),W(159),UV_3GB,AMPL(1,1720))
      CALL VVV1_0(W(5),W(6),W(159),UV_3GT,AMPL(1,1721))
C     Counter-term amplitude(s) for loop diagram number 1112
      CALL R2_QQ_1_0(W(36),W(96),R2_QQQ,AMPL(1,1722))
      CALL FFV2_5_2(W(36),W(3),GC_22,GC_23,ZERO,ZERO,W(160))
C     Counter-term amplitude(s) for loop diagram number 1113
      CALL R2_QQ_1_0(W(160),W(12),R2_QQQ,AMPL(1,1723))
C     Counter-term amplitude(s) for loop diagram number 1114
      CALL FFV2_5_0(W(36),W(12),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1724))
      CALL FFV1P0_3(W(1),W(96),GC_5,ZERO,ZERO,W(161))
C     Counter-term amplitude(s) for loop diagram number 1115
      CALL R2_GG_1_R2_GG_2_0(W(14),W(161),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1725))
C     Counter-term amplitude(s) for loop diagram number 1116
      CALL VVV1_0(W(4),W(6),W(161),R2_3GG,AMPL(1,1726))
      CALL VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1727))
      CALL VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1728))
      CALL VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1729))
      CALL VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1730))
      CALL VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1731))
      CALL VVV1_0(W(4),W(6),W(161),UV_3GB_1EPS,AMPL(2,1732))
      CALL VVV1_0(W(4),W(6),W(161),UV_3GG_1EPS,AMPL(2,1733))
      CALL VVV1_0(W(4),W(6),W(161),UV_3GB,AMPL(1,1734))
      CALL VVV1_0(W(4),W(6),W(161),UV_3GT,AMPL(1,1735))
C     Counter-term amplitude(s) for loop diagram number 1120
      CALL R2_QQ_1_0(W(35),W(101),R2_QQQ,AMPL(1,1736))
      CALL FFV2_5_2(W(35),W(3),GC_22,GC_23,ZERO,ZERO,W(162))
C     Counter-term amplitude(s) for loop diagram number 1121
      CALL R2_QQ_1_0(W(162),W(15),R2_QQQ,AMPL(1,1737))
C     Counter-term amplitude(s) for loop diagram number 1122
      CALL FFV2_5_0(W(35),W(15),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1738))
      CALL FFV1P0_3(W(1),W(101),GC_5,ZERO,ZERO,W(163))
C     Counter-term amplitude(s) for loop diagram number 1123
      CALL R2_GG_1_R2_GG_2_0(W(16),W(163),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1739))
C     Counter-term amplitude(s) for loop diagram number 1124
      CALL VVV1_0(W(4),W(5),W(163),R2_3GG,AMPL(1,1740))
      CALL VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1741))
      CALL VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1742))
      CALL VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1743))
      CALL VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1744))
      CALL VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1745))
      CALL VVV1_0(W(4),W(5),W(163),UV_3GB_1EPS,AMPL(2,1746))
      CALL VVV1_0(W(4),W(5),W(163),UV_3GG_1EPS,AMPL(2,1747))
      CALL VVV1_0(W(4),W(5),W(163),UV_3GB,AMPL(1,1748))
      CALL VVV1_0(W(4),W(5),W(163),UV_3GT,AMPL(1,1749))
C     Counter-term amplitude(s) for loop diagram number 1128
      CALL FFV2_5_0(W(144),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1750))
C     Counter-term amplitude(s) for loop diagram number 1130
      CALL FFV1_0(W(162),W(2),W(6),R2_GQQ,AMPL(1,1751))
      CALL FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1752))
      CALL FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1753))
      CALL FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1754))
      CALL FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1755))
      CALL FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1756))
      CALL FFV1_0(W(162),W(2),W(6),UV_GQQB_1EPS,AMPL(2,1757))
      CALL FFV1_0(W(162),W(2),W(6),UV_GQQG_1EPS,AMPL(2,1758))
      CALL FFV1_0(W(162),W(2),W(6),UV_GQQB,AMPL(1,1759))
      CALL FFV1_0(W(162),W(2),W(6),UV_GQQT,AMPL(1,1760))
C     Counter-term amplitude(s) for loop diagram number 1132
      CALL FFV2_5_0(W(146),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1761))
C     Counter-term amplitude(s) for loop diagram number 1135
      CALL FFV2_5_0(W(148),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1762))
C     Counter-term amplitude(s) for loop diagram number 1137
      CALL FFV1_0(W(160),W(2),W(5),R2_GQQ,AMPL(1,1763))
      CALL FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1764))
      CALL FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1765))
      CALL FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1766))
      CALL FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1767))
      CALL FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1768))
      CALL FFV1_0(W(160),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1769))
      CALL FFV1_0(W(160),W(2),W(5),UV_GQQG_1EPS,AMPL(2,1770))
      CALL FFV1_0(W(160),W(2),W(5),UV_GQQB,AMPL(1,1771))
      CALL FFV1_0(W(160),W(2),W(5),UV_GQQT,AMPL(1,1772))
C     Counter-term amplitude(s) for loop diagram number 1139
      CALL FFV2_5_0(W(150),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1773))
C     Counter-term amplitude(s) for loop diagram number 1143
      CALL FFV1_0(W(158),W(2),W(4),R2_GQQ,AMPL(1,1774))
      CALL FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1775))
      CALL FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1776))
      CALL FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1777))
      CALL FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1778))
      CALL FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1779))
      CALL FFV1_0(W(158),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1780))
      CALL FFV1_0(W(158),W(2),W(4),UV_GQQG_1EPS,AMPL(2,1781))
      CALL FFV1_0(W(158),W(2),W(4),UV_GQQB,AMPL(1,1782))
      CALL FFV1_0(W(158),W(2),W(4),UV_GQQT,AMPL(1,1783))
C     Counter-term amplitude(s) for loop diagram number 1144
      CALL FFV2_5_0(W(152),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1784))
C     Counter-term amplitude(s) for loop diagram number 1148
      CALL FFV2_5_0(W(154),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1785))
C     Counter-term amplitude(s) for loop diagram number 1149
      CALL FFV2_5_0(W(155),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1786))
      CALL FFV2_5_0(W(156),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1787))
      CALL FFV2_5_0(W(157),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1788))
      CALL FFV1P0_3(W(1),W(2),GC_5,ZERO,ZERO,W(164))
      CALL VVV1P0_1(W(4),W(164),GC_4,ZERO,ZERO,W(165))
      CALL VVV1P0_1(W(6),W(165),GC_4,ZERO,ZERO,W(166))
C     Counter-term amplitude(s) for loop diagram number 1254
      CALL R2_GGZ_0(W(5),W(166),W(3),R2_GGZDOWN,AMPL(1,1789))
      CALL R2_GGZ_0(W(5),W(166),W(3),R2_GGZDOWN,AMPL(1,1790))
C     Counter-term amplitude(s) for loop diagram number 1255
      CALL R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECDOWN,AMPL(1,1791))
      CALL R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1792))
      CALL R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECDOWN,AMPL(1,1793))
      CALL R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1794))
      CALL VVV1P0_1(W(5),W(165),GC_4,ZERO,ZERO,W(167))
C     Counter-term amplitude(s) for loop diagram number 1256
      CALL R2_GGZ_0(W(6),W(167),W(3),R2_GGZDOWN,AMPL(1,1795))
      CALL R2_GGZ_0(W(6),W(167),W(3),R2_GGZDOWN,AMPL(1,1796))
      CALL VVV1P0_1(W(5),W(164),GC_4,ZERO,ZERO,W(168))
      CALL VVV1P0_1(W(6),W(168),GC_4,ZERO,ZERO,W(169))
C     Counter-term amplitude(s) for loop diagram number 1258
      CALL R2_GGZ_0(W(4),W(169),W(3),R2_GGZDOWN,AMPL(1,1797))
      CALL R2_GGZ_0(W(4),W(169),W(3),R2_GGZDOWN,AMPL(1,1798))
C     Counter-term amplitude(s) for loop diagram number 1259
      CALL R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECDOWN,AMPL(1,1799))
      CALL R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1800))
      CALL R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECDOWN,AMPL(1,1801))
      CALL R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1802))
      CALL VVV1P0_1(W(4),W(168),GC_4,ZERO,ZERO,W(170))
C     Counter-term amplitude(s) for loop diagram number 1260
      CALL R2_GGZ_0(W(6),W(170),W(3),R2_GGZDOWN,AMPL(1,1803))
      CALL R2_GGZ_0(W(6),W(170),W(3),R2_GGZDOWN,AMPL(1,1804))
      CALL VVV1P0_1(W(6),W(164),GC_4,ZERO,ZERO,W(171))
      CALL VVV1P0_1(W(5),W(171),GC_4,ZERO,ZERO,W(172))
C     Counter-term amplitude(s) for loop diagram number 1262
      CALL R2_GGZ_0(W(4),W(172),W(3),R2_GGZDOWN,AMPL(1,1805))
      CALL R2_GGZ_0(W(4),W(172),W(3),R2_GGZDOWN,AMPL(1,1806))
C     Counter-term amplitude(s) for loop diagram number 1263
      CALL R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECDOWN,AMPL(1,1807))
      CALL R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1808))
      CALL R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECDOWN,AMPL(1,1809))
      CALL R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1810))
      CALL VVV1P0_1(W(4),W(171),GC_4,ZERO,ZERO,W(173))
C     Counter-term amplitude(s) for loop diagram number 1264
      CALL R2_GGZ_0(W(5),W(173),W(3),R2_GGZDOWN,AMPL(1,1811))
      CALL R2_GGZ_0(W(5),W(173),W(3),R2_GGZDOWN,AMPL(1,1812))
      CALL VVVV1P0_1(W(4),W(5),W(164),GC_6,ZERO,ZERO,W(174))
      CALL VVVV3P0_1(W(4),W(5),W(164),GC_6,ZERO,ZERO,W(175))
      CALL VVVV4P0_1(W(4),W(5),W(164),GC_6,ZERO,ZERO,W(176))
C     Counter-term amplitude(s) for loop diagram number 1272
      CALL R2_GGZ_0(W(6),W(174),W(3),R2_GGZDOWN,AMPL(1,1813))
      CALL R2_GGZ_0(W(6),W(174),W(3),R2_GGZDOWN,AMPL(1,1814))
      CALL R2_GGZ_0(W(6),W(175),W(3),R2_GGZDOWN,AMPL(1,1815))
      CALL R2_GGZ_0(W(6),W(175),W(3),R2_GGZDOWN,AMPL(1,1816))
      CALL R2_GGZ_0(W(6),W(176),W(3),R2_GGZDOWN,AMPL(1,1817))
      CALL R2_GGZ_0(W(6),W(176),W(3),R2_GGZDOWN,AMPL(1,1818))
      CALL VVVV1P0_1(W(4),W(6),W(164),GC_6,ZERO,ZERO,W(177))
      CALL VVVV3P0_1(W(4),W(6),W(164),GC_6,ZERO,ZERO,W(178))
      CALL VVVV4P0_1(W(4),W(6),W(164),GC_6,ZERO,ZERO,W(179))
C     Counter-term amplitude(s) for loop diagram number 1273
      CALL R2_GGZ_0(W(5),W(177),W(3),R2_GGZDOWN,AMPL(1,1819))
      CALL R2_GGZ_0(W(5),W(177),W(3),R2_GGZDOWN,AMPL(1,1820))
      CALL R2_GGZ_0(W(5),W(178),W(3),R2_GGZDOWN,AMPL(1,1821))
      CALL R2_GGZ_0(W(5),W(178),W(3),R2_GGZDOWN,AMPL(1,1822))
      CALL R2_GGZ_0(W(5),W(179),W(3),R2_GGZDOWN,AMPL(1,1823))
      CALL R2_GGZ_0(W(5),W(179),W(3),R2_GGZDOWN,AMPL(1,1824))
      CALL VVVV1P0_1(W(5),W(6),W(164),GC_6,ZERO,ZERO,W(180))
      CALL VVVV3P0_1(W(5),W(6),W(164),GC_6,ZERO,ZERO,W(181))
      CALL VVVV4P0_1(W(5),W(6),W(164),GC_6,ZERO,ZERO,W(182))
C     Counter-term amplitude(s) for loop diagram number 1274
      CALL R2_GGZ_0(W(4),W(180),W(3),R2_GGZDOWN,AMPL(1,1825))
      CALL R2_GGZ_0(W(4),W(180),W(3),R2_GGZDOWN,AMPL(1,1826))
      CALL R2_GGZ_0(W(4),W(181),W(3),R2_GGZDOWN,AMPL(1,1827))
      CALL R2_GGZ_0(W(4),W(181),W(3),R2_GGZDOWN,AMPL(1,1828))
      CALL R2_GGZ_0(W(4),W(182),W(3),R2_GGZDOWN,AMPL(1,1829))
      CALL R2_GGZ_0(W(4),W(182),W(3),R2_GGZDOWN,AMPL(1,1830))
C     Counter-term amplitude(s) for loop diagram number 1275
      CALL R2_GGZ_0(W(164),W(65),W(3),R2_GGZDOWN,AMPL(1,1831))
      CALL R2_GGZ_0(W(164),W(65),W(3),R2_GGZDOWN,AMPL(1,1832))
C     Counter-term amplitude(s) for loop diagram number 1276
      CALL R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECDOWN,AMPL(1,1833)
     $ )
      CALL R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1834))
      CALL R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECDOWN,AMPL(1,1835)
     $ )
      CALL R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1836))
      CALL VVV1P0_1(W(16),W(164),GC_4,ZERO,ZERO,W(183))
C     Counter-term amplitude(s) for loop diagram number 1277
      CALL R2_GGZ_0(W(6),W(183),W(3),R2_GGZDOWN,AMPL(1,1837))
      CALL R2_GGZ_0(W(6),W(183),W(3),R2_GGZDOWN,AMPL(1,1838))
C     Counter-term amplitude(s) for loop diagram number 1278
      CALL R2_GGZ_0(W(16),W(171),W(3),R2_GGZDOWN,AMPL(1,1839))
      CALL R2_GGZ_0(W(16),W(171),W(3),R2_GGZDOWN,AMPL(1,1840))
C     Counter-term amplitude(s) for loop diagram number 1287
      CALL R2_GGZ_0(W(164),W(66),W(3),R2_GGZDOWN,AMPL(1,1841))
      CALL R2_GGZ_0(W(164),W(66),W(3),R2_GGZDOWN,AMPL(1,1842))
C     Counter-term amplitude(s) for loop diagram number 1288
      CALL R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECDOWN,AMPL(1,1843)
     $ )
      CALL R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1844))
      CALL R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECDOWN,AMPL(1,1845)
     $ )
      CALL R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1846))
      CALL VVV1P0_1(W(14),W(164),GC_4,ZERO,ZERO,W(184))
C     Counter-term amplitude(s) for loop diagram number 1289
      CALL R2_GGZ_0(W(5),W(184),W(3),R2_GGZDOWN,AMPL(1,1847))
      CALL R2_GGZ_0(W(5),W(184),W(3),R2_GGZDOWN,AMPL(1,1848))
C     Counter-term amplitude(s) for loop diagram number 1290
      CALL R2_GGZ_0(W(14),W(168),W(3),R2_GGZDOWN,AMPL(1,1849))
      CALL R2_GGZ_0(W(14),W(168),W(3),R2_GGZDOWN,AMPL(1,1850))
C     Counter-term amplitude(s) for loop diagram number 1312
      CALL R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECDOWN,AMPL(1,1851)
     $ )
      CALL R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1852))
      CALL R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECDOWN,AMPL(1,1853)
     $ )
      CALL R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1854))
      CALL VVV1P0_1(W(11),W(164),GC_4,ZERO,ZERO,W(185))
C     Counter-term amplitude(s) for loop diagram number 1314
      CALL R2_GGZ_0(W(4),W(185),W(3),R2_GGZDOWN,AMPL(1,1855))
      CALL R2_GGZ_0(W(4),W(185),W(3),R2_GGZDOWN,AMPL(1,1856))
C     Counter-term amplitude(s) for loop diagram number 1315
      CALL R2_GGZ_0(W(164),W(67),W(3),R2_GGZDOWN,AMPL(1,1857))
      CALL R2_GGZ_0(W(164),W(67),W(3),R2_GGZDOWN,AMPL(1,1858))
C     Counter-term amplitude(s) for loop diagram number 1317
      CALL R2_GGZ_0(W(11),W(165),W(3),R2_GGZDOWN,AMPL(1,1859))
      CALL R2_GGZ_0(W(11),W(165),W(3),R2_GGZDOWN,AMPL(1,1860))
C     Counter-term amplitude(s) for loop diagram number 1350
      CALL R2_GGZ_0(W(164),W(18),W(3),R2_GGZDOWN,AMPL(1,1861))
      CALL R2_GGZ_0(W(164),W(18),W(3),R2_GGZDOWN,AMPL(1,1862))
      CALL R2_GGZ_0(W(164),W(19),W(3),R2_GGZDOWN,AMPL(1,1863))
      CALL R2_GGZ_0(W(164),W(19),W(3),R2_GGZDOWN,AMPL(1,1864))
      CALL R2_GGZ_0(W(164),W(20),W(3),R2_GGZDOWN,AMPL(1,1865))
      CALL R2_GGZ_0(W(164),W(20),W(3),R2_GGZDOWN,AMPL(1,1866))
C     Counter-term amplitude(s) for loop diagram number 1352
      CALL R2_GG_1_0(W(11),W(44),R2_GGQ,AMPL(1,1867))
      CALL R2_GG_1_0(W(11),W(44),R2_GGQ,AMPL(1,1868))
      CALL R2_GG_1_0(W(11),W(44),R2_GGQ,AMPL(1,1869))
      CALL R2_GG_1_0(W(11),W(44),R2_GGQ,AMPL(1,1870))
C     Counter-term amplitude(s) for loop diagram number 1353
      CALL VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,1871))
      CALL VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,1872))
      CALL VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,1873))
      CALL VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,1874))
C     Counter-term amplitude(s) for loop diagram number 1355
      CALL R2_GG_1_0(W(14),W(53),R2_GGQ,AMPL(1,1875))
      CALL R2_GG_1_0(W(14),W(53),R2_GGQ,AMPL(1,1876))
      CALL R2_GG_1_0(W(14),W(53),R2_GGQ,AMPL(1,1877))
      CALL R2_GG_1_0(W(14),W(53),R2_GGQ,AMPL(1,1878))
C     Counter-term amplitude(s) for loop diagram number 1356
      CALL VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,1879))
      CALL VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,1880))
      CALL VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,1881))
      CALL VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,1882))
C     Counter-term amplitude(s) for loop diagram number 1358
      CALL R2_GG_1_0(W(16),W(62),R2_GGQ,AMPL(1,1883))
      CALL R2_GG_1_0(W(16),W(62),R2_GGQ,AMPL(1,1884))
      CALL R2_GG_1_0(W(16),W(62),R2_GGQ,AMPL(1,1885))
      CALL R2_GG_1_0(W(16),W(62),R2_GGQ,AMPL(1,1886))
C     Counter-term amplitude(s) for loop diagram number 1359
      CALL VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,1887))
      CALL VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,1888))
      CALL VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,1889))
      CALL VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,1890))
C     Counter-term amplitude(s) for loop diagram number 1361
      CALL R2_GG_1_0(W(17),W(65),R2_GGQ,AMPL(1,1891))
      CALL R2_GG_1_0(W(17),W(65),R2_GGQ,AMPL(1,1892))
      CALL R2_GG_1_0(W(17),W(65),R2_GGQ,AMPL(1,1893))
      CALL R2_GG_1_0(W(17),W(65),R2_GGQ,AMPL(1,1894))
C     Counter-term amplitude(s) for loop diagram number 1362
      CALL R2_GG_1_0(W(16),W(68),R2_GGQ,AMPL(1,1895))
      CALL R2_GG_1_0(W(16),W(68),R2_GGQ,AMPL(1,1896))
      CALL R2_GG_1_0(W(16),W(68),R2_GGQ,AMPL(1,1897))
      CALL R2_GG_1_0(W(16),W(68),R2_GGQ,AMPL(1,1898))
C     Counter-term amplitude(s) for loop diagram number 1363
      CALL VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,1899))
      CALL VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,1900))
      CALL VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,1901))
      CALL VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,1902))
C     Counter-term amplitude(s) for loop diagram number 1365
      CALL R2_GG_1_0(W(16),W(70),R2_GGQ,AMPL(1,1903))
      CALL R2_GG_1_0(W(16),W(70),R2_GGQ,AMPL(1,1904))
      CALL R2_GG_1_0(W(16),W(70),R2_GGQ,AMPL(1,1905))
      CALL R2_GG_1_0(W(16),W(70),R2_GGQ,AMPL(1,1906))
C     Counter-term amplitude(s) for loop diagram number 1366
      CALL R2_GG_1_0(W(17),W(66),R2_GGQ,AMPL(1,1907))
      CALL R2_GG_1_0(W(17),W(66),R2_GGQ,AMPL(1,1908))
      CALL R2_GG_1_0(W(17),W(66),R2_GGQ,AMPL(1,1909))
      CALL R2_GG_1_0(W(17),W(66),R2_GGQ,AMPL(1,1910))
C     Counter-term amplitude(s) for loop diagram number 1367
      CALL R2_GG_1_0(W(14),W(73),R2_GGQ,AMPL(1,1911))
      CALL R2_GG_1_0(W(14),W(73),R2_GGQ,AMPL(1,1912))
      CALL R2_GG_1_0(W(14),W(73),R2_GGQ,AMPL(1,1913))
      CALL R2_GG_1_0(W(14),W(73),R2_GGQ,AMPL(1,1914))
C     Counter-term amplitude(s) for loop diagram number 1368
      CALL VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,1915))
      CALL VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,1916))
      CALL VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,1917))
      CALL VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,1918))
C     Counter-term amplitude(s) for loop diagram number 1370
      CALL R2_GG_1_0(W(14),W(75),R2_GGQ,AMPL(1,1919))
      CALL R2_GG_1_0(W(14),W(75),R2_GGQ,AMPL(1,1920))
      CALL R2_GG_1_0(W(14),W(75),R2_GGQ,AMPL(1,1921))
      CALL R2_GG_1_0(W(14),W(75),R2_GGQ,AMPL(1,1922))
C     Counter-term amplitude(s) for loop diagram number 1371
      CALL VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,1923))
      CALL VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,1924))
      CALL VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,1925))
      CALL VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,1926))
C     Counter-term amplitude(s) for loop diagram number 1372
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,1927))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1928))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,1929))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1930))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1931))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1932))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,1933))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,1934))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,1935))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,1936)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,1937)
     $ )
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,1938))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1939))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,1940))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1941))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1942))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1943))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,1944))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,1945))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,1946))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,1947)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,1948)
     $ )
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,1949))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1950))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,1951))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1952))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1953))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1954))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,1955))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,1956))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,1957))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,1958)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,1959)
     $ )
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,1960))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1961))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,1962))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1963))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1964))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,1965))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,1966))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,1967))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,1968))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,1969)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,1970)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1373
      CALL VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,1971))
      CALL VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,1972))
      CALL VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,1973))
      CALL VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,1974))
C     Counter-term amplitude(s) for loop diagram number 1375
      CALL VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,1975))
      CALL VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,1976))
      CALL VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,1977))
      CALL VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,1978))
C     Counter-term amplitude(s) for loop diagram number 1376
      CALL VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,1979))
      CALL VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,1980))
      CALL VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,1981))
      CALL VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,1982))
C     Counter-term amplitude(s) for loop diagram number 1377
      CALL VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,1983))
      CALL VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,1984))
      CALL VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,1985))
      CALL VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,1986))
C     Counter-term amplitude(s) for loop diagram number 1387
      CALL R2_GG_1_0(W(17),W(67),R2_GGQ,AMPL(1,1987))
      CALL R2_GG_1_0(W(17),W(67),R2_GGQ,AMPL(1,1988))
      CALL R2_GG_1_0(W(17),W(67),R2_GGQ,AMPL(1,1989))
      CALL R2_GG_1_0(W(17),W(67),R2_GGQ,AMPL(1,1990))
C     Counter-term amplitude(s) for loop diagram number 1388
      CALL R2_GG_1_0(W(11),W(79),R2_GGQ,AMPL(1,1991))
      CALL R2_GG_1_0(W(11),W(79),R2_GGQ,AMPL(1,1992))
      CALL R2_GG_1_0(W(11),W(79),R2_GGQ,AMPL(1,1993))
      CALL R2_GG_1_0(W(11),W(79),R2_GGQ,AMPL(1,1994))
C     Counter-term amplitude(s) for loop diagram number 1389
      CALL R2_GG_1_0(W(11),W(80),R2_GGQ,AMPL(1,1995))
      CALL R2_GG_1_0(W(11),W(80),R2_GGQ,AMPL(1,1996))
      CALL R2_GG_1_0(W(11),W(80),R2_GGQ,AMPL(1,1997))
      CALL R2_GG_1_0(W(11),W(80),R2_GGQ,AMPL(1,1998))
C     Counter-term amplitude(s) for loop diagram number 1390
      CALL VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,1999))
      CALL VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2000))
      CALL VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2001))
      CALL VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2002))
C     Counter-term amplitude(s) for loop diagram number 1391
      CALL VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2003))
      CALL VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2004))
      CALL VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2005))
      CALL VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2006))
C     Counter-term amplitude(s) for loop diagram number 1394
      CALL R2_GG_1_0(W(17),W(18),R2_GGQ,AMPL(1,2007))
      CALL R2_GG_1_0(W(17),W(18),R2_GGQ,AMPL(1,2008))
      CALL R2_GG_1_0(W(17),W(18),R2_GGQ,AMPL(1,2009))
      CALL R2_GG_1_0(W(17),W(18),R2_GGQ,AMPL(1,2010))
      CALL R2_GG_1_0(W(17),W(19),R2_GGQ,AMPL(1,2011))
      CALL R2_GG_1_0(W(17),W(19),R2_GGQ,AMPL(1,2012))
      CALL R2_GG_1_0(W(17),W(19),R2_GGQ,AMPL(1,2013))
      CALL R2_GG_1_0(W(17),W(19),R2_GGQ,AMPL(1,2014))
      CALL R2_GG_1_0(W(17),W(20),R2_GGQ,AMPL(1,2015))
      CALL R2_GG_1_0(W(17),W(20),R2_GGQ,AMPL(1,2016))
      CALL R2_GG_1_0(W(17),W(20),R2_GGQ,AMPL(1,2017))
      CALL R2_GG_1_0(W(17),W(20),R2_GGQ,AMPL(1,2018))
C     Counter-term amplitude(s) for loop diagram number 1395
      CALL R2_GG_1_0(W(11),W(92),R2_GGQ,AMPL(1,2019))
      CALL R2_GG_1_0(W(11),W(92),R2_GGQ,AMPL(1,2020))
      CALL R2_GG_1_0(W(11),W(92),R2_GGQ,AMPL(1,2021))
      CALL R2_GG_1_0(W(11),W(92),R2_GGQ,AMPL(1,2022))
C     Counter-term amplitude(s) for loop diagram number 1396
      CALL VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2023))
      CALL VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2024))
      CALL VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2025))
      CALL VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2026))
      CALL FFV1P0_3(W(21),W(12),GC_5,ZERO,ZERO,W(186))
C     Counter-term amplitude(s) for loop diagram number 1398
      CALL R2_GGZ_0(W(6),W(186),W(3),R2_GGZDOWN,AMPL(1,2027))
      CALL R2_GGZ_0(W(6),W(186),W(3),R2_GGZDOWN,AMPL(1,2028))
      CALL FFV1P0_3(W(21),W(15),GC_5,ZERO,ZERO,W(187))
C     Counter-term amplitude(s) for loop diagram number 1400
      CALL R2_GGZ_0(W(5),W(187),W(3),R2_GGZDOWN,AMPL(1,2029))
      CALL R2_GGZ_0(W(5),W(187),W(3),R2_GGZDOWN,AMPL(1,2030))
      CALL FFV1P0_3(W(21),W(2),GC_5,ZERO,ZERO,W(188))
      CALL VVV1P0_1(W(6),W(188),GC_4,ZERO,ZERO,W(189))
C     Counter-term amplitude(s) for loop diagram number 1402
      CALL R2_GGZ_0(W(5),W(189),W(3),R2_GGZDOWN,AMPL(1,2031))
      CALL R2_GGZ_0(W(5),W(189),W(3),R2_GGZDOWN,AMPL(1,2032))
C     Counter-term amplitude(s) for loop diagram number 1403
      CALL R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECDOWN,AMPL(1,2033))
      CALL R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2034))
      CALL R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECDOWN,AMPL(1,2035))
      CALL R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2036))
      CALL VVV1P0_1(W(5),W(188),GC_4,ZERO,ZERO,W(190))
C     Counter-term amplitude(s) for loop diagram number 1404
      CALL R2_GGZ_0(W(6),W(190),W(3),R2_GGZDOWN,AMPL(1,2037))
      CALL R2_GGZ_0(W(6),W(190),W(3),R2_GGZDOWN,AMPL(1,2038))
      CALL FFV1P0_3(W(23),W(2),GC_5,ZERO,ZERO,W(191))
C     Counter-term amplitude(s) for loop diagram number 1406
      CALL R2_GGZ_0(W(6),W(191),W(3),R2_GGZDOWN,AMPL(1,2039))
      CALL R2_GGZ_0(W(6),W(191),W(3),R2_GGZDOWN,AMPL(1,2040))
      CALL FFV1P0_3(W(24),W(2),GC_5,ZERO,ZERO,W(192))
C     Counter-term amplitude(s) for loop diagram number 1407
      CALL R2_GGZ_0(W(5),W(192),W(3),R2_GGZDOWN,AMPL(1,2041))
      CALL R2_GGZ_0(W(5),W(192),W(3),R2_GGZDOWN,AMPL(1,2042))
C     Counter-term amplitude(s) for loop diagram number 1408
      CALL R2_GGZ_0(W(11),W(188),W(3),R2_GGZDOWN,AMPL(1,2043))
      CALL R2_GGZ_0(W(11),W(188),W(3),R2_GGZDOWN,AMPL(1,2044))
C     Counter-term amplitude(s) for loop diagram number 1418
      CALL R2_GG_1_0(W(11),W(105),R2_GGQ,AMPL(1,2045))
      CALL R2_GG_1_0(W(11),W(105),R2_GGQ,AMPL(1,2046))
      CALL R2_GG_1_0(W(11),W(105),R2_GGQ,AMPL(1,2047))
      CALL R2_GG_1_0(W(11),W(105),R2_GGQ,AMPL(1,2048))
C     Counter-term amplitude(s) for loop diagram number 1419
      CALL VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2049))
      CALL VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2050))
      CALL VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2051))
      CALL VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2052))
C     Counter-term amplitude(s) for loop diagram number 1421
      CALL R2_GG_1_0(W(14),W(112),R2_GGQ,AMPL(1,2053))
      CALL R2_GG_1_0(W(14),W(112),R2_GGQ,AMPL(1,2054))
      CALL R2_GG_1_0(W(14),W(112),R2_GGQ,AMPL(1,2055))
      CALL R2_GG_1_0(W(14),W(112),R2_GGQ,AMPL(1,2056))
C     Counter-term amplitude(s) for loop diagram number 1422
      CALL VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2057))
      CALL VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2058))
      CALL VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2059))
      CALL VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2060))
      CALL FFV1P0_3(W(26),W(8),GC_5,ZERO,ZERO,W(193))
C     Counter-term amplitude(s) for loop diagram number 1424
      CALL R2_GGZ_0(W(6),W(193),W(3),R2_GGZDOWN,AMPL(1,2061))
      CALL R2_GGZ_0(W(6),W(193),W(3),R2_GGZDOWN,AMPL(1,2062))
      CALL FFV1P0_3(W(26),W(15),GC_5,ZERO,ZERO,W(194))
C     Counter-term amplitude(s) for loop diagram number 1426
      CALL R2_GGZ_0(W(4),W(194),W(3),R2_GGZDOWN,AMPL(1,2063))
      CALL R2_GGZ_0(W(4),W(194),W(3),R2_GGZDOWN,AMPL(1,2064))
      CALL FFV1P0_3(W(26),W(2),GC_5,ZERO,ZERO,W(195))
      CALL VVV1P0_1(W(6),W(195),GC_4,ZERO,ZERO,W(196))
C     Counter-term amplitude(s) for loop diagram number 1428
      CALL R2_GGZ_0(W(4),W(196),W(3),R2_GGZDOWN,AMPL(1,2065))
      CALL R2_GGZ_0(W(4),W(196),W(3),R2_GGZDOWN,AMPL(1,2066))
C     Counter-term amplitude(s) for loop diagram number 1429
      CALL R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECDOWN,AMPL(1,2067))
      CALL R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2068))
      CALL R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECDOWN,AMPL(1,2069))
      CALL R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2070))
      CALL VVV1P0_1(W(4),W(195),GC_4,ZERO,ZERO,W(197))
C     Counter-term amplitude(s) for loop diagram number 1430
      CALL R2_GGZ_0(W(6),W(197),W(3),R2_GGZDOWN,AMPL(1,2071))
      CALL R2_GGZ_0(W(6),W(197),W(3),R2_GGZDOWN,AMPL(1,2072))
      CALL FFV1P0_3(W(27),W(2),GC_5,ZERO,ZERO,W(198))
C     Counter-term amplitude(s) for loop diagram number 1432
      CALL R2_GGZ_0(W(6),W(198),W(3),R2_GGZDOWN,AMPL(1,2073))
      CALL R2_GGZ_0(W(6),W(198),W(3),R2_GGZDOWN,AMPL(1,2074))
      CALL FFV1P0_3(W(28),W(2),GC_5,ZERO,ZERO,W(199))
C     Counter-term amplitude(s) for loop diagram number 1433
      CALL R2_GGZ_0(W(4),W(199),W(3),R2_GGZDOWN,AMPL(1,2075))
      CALL R2_GGZ_0(W(4),W(199),W(3),R2_GGZDOWN,AMPL(1,2076))
C     Counter-term amplitude(s) for loop diagram number 1434
      CALL R2_GGZ_0(W(14),W(195),W(3),R2_GGZDOWN,AMPL(1,2077))
      CALL R2_GGZ_0(W(14),W(195),W(3),R2_GGZDOWN,AMPL(1,2078))
C     Counter-term amplitude(s) for loop diagram number 1444
      CALL R2_GG_1_0(W(14),W(124),R2_GGQ,AMPL(1,2079))
      CALL R2_GG_1_0(W(14),W(124),R2_GGQ,AMPL(1,2080))
      CALL R2_GG_1_0(W(14),W(124),R2_GGQ,AMPL(1,2081))
      CALL R2_GG_1_0(W(14),W(124),R2_GGQ,AMPL(1,2082))
C     Counter-term amplitude(s) for loop diagram number 1445
      CALL VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2083))
      CALL VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2084))
      CALL VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2085))
      CALL VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2086))
C     Counter-term amplitude(s) for loop diagram number 1447
      CALL R2_GG_1_0(W(16),W(130),R2_GGQ,AMPL(1,2087))
      CALL R2_GG_1_0(W(16),W(130),R2_GGQ,AMPL(1,2088))
      CALL R2_GG_1_0(W(16),W(130),R2_GGQ,AMPL(1,2089))
      CALL R2_GG_1_0(W(16),W(130),R2_GGQ,AMPL(1,2090))
C     Counter-term amplitude(s) for loop diagram number 1448
      CALL VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2091))
      CALL VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2092))
      CALL VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2093))
      CALL VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2094))
      CALL FFV1P0_3(W(30),W(8),GC_5,ZERO,ZERO,W(200))
C     Counter-term amplitude(s) for loop diagram number 1450
      CALL R2_GGZ_0(W(5),W(200),W(3),R2_GGZDOWN,AMPL(1,2095))
      CALL R2_GGZ_0(W(5),W(200),W(3),R2_GGZDOWN,AMPL(1,2096))
      CALL FFV1P0_3(W(30),W(12),GC_5,ZERO,ZERO,W(201))
C     Counter-term amplitude(s) for loop diagram number 1452
      CALL R2_GGZ_0(W(4),W(201),W(3),R2_GGZDOWN,AMPL(1,2097))
      CALL R2_GGZ_0(W(4),W(201),W(3),R2_GGZDOWN,AMPL(1,2098))
      CALL FFV1P0_3(W(30),W(2),GC_5,ZERO,ZERO,W(202))
      CALL VVV1P0_1(W(5),W(202),GC_4,ZERO,ZERO,W(203))
C     Counter-term amplitude(s) for loop diagram number 1454
      CALL R2_GGZ_0(W(4),W(203),W(3),R2_GGZDOWN,AMPL(1,2099))
      CALL R2_GGZ_0(W(4),W(203),W(3),R2_GGZDOWN,AMPL(1,2100))
C     Counter-term amplitude(s) for loop diagram number 1455
      CALL R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECDOWN,AMPL(1,2101))
      CALL R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2102))
      CALL R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECDOWN,AMPL(1,2103))
      CALL R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2104))
      CALL VVV1P0_1(W(4),W(202),GC_4,ZERO,ZERO,W(204))
C     Counter-term amplitude(s) for loop diagram number 1456
      CALL R2_GGZ_0(W(5),W(204),W(3),R2_GGZDOWN,AMPL(1,2105))
      CALL R2_GGZ_0(W(5),W(204),W(3),R2_GGZDOWN,AMPL(1,2106))
      CALL FFV1P0_3(W(31),W(2),GC_5,ZERO,ZERO,W(205))
C     Counter-term amplitude(s) for loop diagram number 1458
      CALL R2_GGZ_0(W(5),W(205),W(3),R2_GGZDOWN,AMPL(1,2107))
      CALL R2_GGZ_0(W(5),W(205),W(3),R2_GGZDOWN,AMPL(1,2108))
      CALL FFV1P0_3(W(32),W(2),GC_5,ZERO,ZERO,W(206))
C     Counter-term amplitude(s) for loop diagram number 1459
      CALL R2_GGZ_0(W(4),W(206),W(3),R2_GGZDOWN,AMPL(1,2109))
      CALL R2_GGZ_0(W(4),W(206),W(3),R2_GGZDOWN,AMPL(1,2110))
C     Counter-term amplitude(s) for loop diagram number 1460
      CALL R2_GGZ_0(W(16),W(202),W(3),R2_GGZDOWN,AMPL(1,2111))
      CALL R2_GGZ_0(W(16),W(202),W(3),R2_GGZDOWN,AMPL(1,2112))
C     Counter-term amplitude(s) for loop diagram number 1470
      CALL R2_GG_1_0(W(16),W(141),R2_GGQ,AMPL(1,2113))
      CALL R2_GG_1_0(W(16),W(141),R2_GGQ,AMPL(1,2114))
      CALL R2_GG_1_0(W(16),W(141),R2_GGQ,AMPL(1,2115))
      CALL R2_GG_1_0(W(16),W(141),R2_GGQ,AMPL(1,2116))
C     Counter-term amplitude(s) for loop diagram number 1471
      CALL VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2117))
      CALL VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2118))
      CALL VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2119))
      CALL VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2120))
C     Counter-term amplitude(s) for loop diagram number 1473
      CALL R2_GG_1_0(W(34),W(65),R2_GGQ,AMPL(1,2121))
      CALL R2_GG_1_0(W(34),W(65),R2_GGQ,AMPL(1,2122))
      CALL R2_GG_1_0(W(34),W(65),R2_GGQ,AMPL(1,2123))
      CALL R2_GG_1_0(W(34),W(65),R2_GGQ,AMPL(1,2124))
C     Counter-term amplitude(s) for loop diagram number 1474
      CALL R2_GG_1_0(W(16),W(143),R2_GGQ,AMPL(1,2125))
      CALL R2_GG_1_0(W(16),W(143),R2_GGQ,AMPL(1,2126))
      CALL R2_GG_1_0(W(16),W(143),R2_GGQ,AMPL(1,2127))
      CALL R2_GG_1_0(W(16),W(143),R2_GGQ,AMPL(1,2128))
C     Counter-term amplitude(s) for loop diagram number 1475
      CALL VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2129))
      CALL VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2130))
      CALL VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2131))
      CALL VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2132))
C     Counter-term amplitude(s) for loop diagram number 1477
      CALL R2_GG_1_0(W(16),W(145),R2_GGQ,AMPL(1,2133))
      CALL R2_GG_1_0(W(16),W(145),R2_GGQ,AMPL(1,2134))
      CALL R2_GG_1_0(W(16),W(145),R2_GGQ,AMPL(1,2135))
      CALL R2_GG_1_0(W(16),W(145),R2_GGQ,AMPL(1,2136))
C     Counter-term amplitude(s) for loop diagram number 1478
      CALL R2_GG_1_0(W(34),W(66),R2_GGQ,AMPL(1,2137))
      CALL R2_GG_1_0(W(34),W(66),R2_GGQ,AMPL(1,2138))
      CALL R2_GG_1_0(W(34),W(66),R2_GGQ,AMPL(1,2139))
      CALL R2_GG_1_0(W(34),W(66),R2_GGQ,AMPL(1,2140))
C     Counter-term amplitude(s) for loop diagram number 1479
      CALL R2_GG_1_0(W(14),W(147),R2_GGQ,AMPL(1,2141))
      CALL R2_GG_1_0(W(14),W(147),R2_GGQ,AMPL(1,2142))
      CALL R2_GG_1_0(W(14),W(147),R2_GGQ,AMPL(1,2143))
      CALL R2_GG_1_0(W(14),W(147),R2_GGQ,AMPL(1,2144))
C     Counter-term amplitude(s) for loop diagram number 1480
      CALL VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2145))
      CALL VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2146))
      CALL VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2147))
      CALL VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2148))
C     Counter-term amplitude(s) for loop diagram number 1482
      CALL R2_GG_1_0(W(14),W(149),R2_GGQ,AMPL(1,2149))
      CALL R2_GG_1_0(W(14),W(149),R2_GGQ,AMPL(1,2150))
      CALL R2_GG_1_0(W(14),W(149),R2_GGQ,AMPL(1,2151))
      CALL R2_GG_1_0(W(14),W(149),R2_GGQ,AMPL(1,2152))
C     Counter-term amplitude(s) for loop diagram number 1483
      CALL VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2153))
      CALL VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2154))
      CALL VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2155))
      CALL VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2156))
C     Counter-term amplitude(s) for loop diagram number 1484
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2157))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2158))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2159))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2160))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2161))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2162))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,2163))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2164))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2165))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2166)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2167)
     $ )
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2168))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2169))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2170))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2171))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2172))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2173))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,2174))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2175))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2176))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2177)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2178)
     $ )
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2179))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2180))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2181))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2182))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2183))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2184))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,2185))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2186))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2187))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2188)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2189)
     $ )
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2190))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2191))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2192))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2193))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2194))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2195))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,2196))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2197))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2198))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2199)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2200)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1485
      CALL VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2201))
      CALL VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2202))
      CALL VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2203))
      CALL VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2204))
C     Counter-term amplitude(s) for loop diagram number 1487
      CALL VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2205))
      CALL VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2206))
      CALL VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2207))
      CALL VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2208))
C     Counter-term amplitude(s) for loop diagram number 1488
      CALL VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2209))
      CALL VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2210))
      CALL VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2211))
      CALL VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2212))
C     Counter-term amplitude(s) for loop diagram number 1489
      CALL VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2213))
      CALL VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2214))
      CALL VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2215))
      CALL VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2216))
C     Counter-term amplitude(s) for loop diagram number 1499
      CALL R2_GG_1_0(W(34),W(67),R2_GGQ,AMPL(1,2217))
      CALL R2_GG_1_0(W(34),W(67),R2_GGQ,AMPL(1,2218))
      CALL R2_GG_1_0(W(34),W(67),R2_GGQ,AMPL(1,2219))
      CALL R2_GG_1_0(W(34),W(67),R2_GGQ,AMPL(1,2220))
C     Counter-term amplitude(s) for loop diagram number 1500
      CALL R2_GG_1_0(W(11),W(151),R2_GGQ,AMPL(1,2221))
      CALL R2_GG_1_0(W(11),W(151),R2_GGQ,AMPL(1,2222))
      CALL R2_GG_1_0(W(11),W(151),R2_GGQ,AMPL(1,2223))
      CALL R2_GG_1_0(W(11),W(151),R2_GGQ,AMPL(1,2224))
C     Counter-term amplitude(s) for loop diagram number 1501
      CALL R2_GG_1_0(W(11),W(153),R2_GGQ,AMPL(1,2225))
      CALL R2_GG_1_0(W(11),W(153),R2_GGQ,AMPL(1,2226))
      CALL R2_GG_1_0(W(11),W(153),R2_GGQ,AMPL(1,2227))
      CALL R2_GG_1_0(W(11),W(153),R2_GGQ,AMPL(1,2228))
C     Counter-term amplitude(s) for loop diagram number 1502
      CALL VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2229))
      CALL VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2230))
      CALL VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2231))
      CALL VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2232))
C     Counter-term amplitude(s) for loop diagram number 1503
      CALL VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2233))
      CALL VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2234))
      CALL VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2235))
      CALL VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2236))
C     Counter-term amplitude(s) for loop diagram number 1506
      CALL R2_GG_1_0(W(34),W(18),R2_GGQ,AMPL(1,2237))
      CALL R2_GG_1_0(W(34),W(18),R2_GGQ,AMPL(1,2238))
      CALL R2_GG_1_0(W(34),W(18),R2_GGQ,AMPL(1,2239))
      CALL R2_GG_1_0(W(34),W(18),R2_GGQ,AMPL(1,2240))
      CALL R2_GG_1_0(W(34),W(19),R2_GGQ,AMPL(1,2241))
      CALL R2_GG_1_0(W(34),W(19),R2_GGQ,AMPL(1,2242))
      CALL R2_GG_1_0(W(34),W(19),R2_GGQ,AMPL(1,2243))
      CALL R2_GG_1_0(W(34),W(19),R2_GGQ,AMPL(1,2244))
      CALL R2_GG_1_0(W(34),W(20),R2_GGQ,AMPL(1,2245))
      CALL R2_GG_1_0(W(34),W(20),R2_GGQ,AMPL(1,2246))
      CALL R2_GG_1_0(W(34),W(20),R2_GGQ,AMPL(1,2247))
      CALL R2_GG_1_0(W(34),W(20),R2_GGQ,AMPL(1,2248))
      CALL FFV1P0_3(W(1),W(8),GC_5,ZERO,ZERO,W(207))
      CALL VVV1P0_1(W(6),W(207),GC_4,ZERO,ZERO,W(208))
C     Counter-term amplitude(s) for loop diagram number 1507
      CALL R2_GGZ_0(W(5),W(208),W(3),R2_GGZDOWN,AMPL(1,2249))
      CALL R2_GGZ_0(W(5),W(208),W(3),R2_GGZDOWN,AMPL(1,2250))
C     Counter-term amplitude(s) for loop diagram number 1508
      CALL R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECDOWN,AMPL(1,2251))
      CALL R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2252))
      CALL R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECDOWN,AMPL(1,2253))
      CALL R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2254))
      CALL VVV1P0_1(W(5),W(207),GC_4,ZERO,ZERO,W(209))
C     Counter-term amplitude(s) for loop diagram number 1509
      CALL R2_GGZ_0(W(6),W(209),W(3),R2_GGZDOWN,AMPL(1,2255))
      CALL R2_GGZ_0(W(6),W(209),W(3),R2_GGZDOWN,AMPL(1,2256))
      CALL FFV1P0_3(W(1),W(40),GC_5,ZERO,ZERO,W(210))
C     Counter-term amplitude(s) for loop diagram number 1511
      CALL R2_GGZ_0(W(6),W(210),W(3),R2_GGZDOWN,AMPL(1,2257))
      CALL R2_GGZ_0(W(6),W(210),W(3),R2_GGZDOWN,AMPL(1,2258))
      CALL FFV1P0_3(W(1),W(38),GC_5,ZERO,ZERO,W(211))
C     Counter-term amplitude(s) for loop diagram number 1512
      CALL R2_GGZ_0(W(5),W(211),W(3),R2_GGZDOWN,AMPL(1,2259))
      CALL R2_GGZ_0(W(5),W(211),W(3),R2_GGZDOWN,AMPL(1,2260))
C     Counter-term amplitude(s) for loop diagram number 1513
      CALL R2_GGZ_0(W(11),W(207),W(3),R2_GGZDOWN,AMPL(1,2261))
      CALL R2_GGZ_0(W(11),W(207),W(3),R2_GGZDOWN,AMPL(1,2262))
C     Counter-term amplitude(s) for loop diagram number 1523
      CALL R2_GG_1_0(W(11),W(159),R2_GGQ,AMPL(1,2263))
      CALL R2_GG_1_0(W(11),W(159),R2_GGQ,AMPL(1,2264))
      CALL R2_GG_1_0(W(11),W(159),R2_GGQ,AMPL(1,2265))
      CALL R2_GG_1_0(W(11),W(159),R2_GGQ,AMPL(1,2266))
C     Counter-term amplitude(s) for loop diagram number 1524
      CALL VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2267))
      CALL VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2268))
      CALL VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2269))
      CALL VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2270))
      CALL FFV1P0_3(W(1),W(12),GC_5,ZERO,ZERO,W(212))
      CALL VVV1P0_1(W(6),W(212),GC_4,ZERO,ZERO,W(213))
C     Counter-term amplitude(s) for loop diagram number 1526
      CALL R2_GGZ_0(W(4),W(213),W(3),R2_GGZDOWN,AMPL(1,2271))
      CALL R2_GGZ_0(W(4),W(213),W(3),R2_GGZDOWN,AMPL(1,2272))
C     Counter-term amplitude(s) for loop diagram number 1527
      CALL R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECDOWN,AMPL(1,2273))
      CALL R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2274))
      CALL R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECDOWN,AMPL(1,2275))
      CALL R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2276))
      CALL VVV1P0_1(W(4),W(212),GC_4,ZERO,ZERO,W(214))
C     Counter-term amplitude(s) for loop diagram number 1528
      CALL R2_GGZ_0(W(6),W(214),W(3),R2_GGZDOWN,AMPL(1,2277))
      CALL R2_GGZ_0(W(6),W(214),W(3),R2_GGZDOWN,AMPL(1,2278))
      CALL FFV1P0_3(W(1),W(49),GC_5,ZERO,ZERO,W(215))
C     Counter-term amplitude(s) for loop diagram number 1530
      CALL R2_GGZ_0(W(6),W(215),W(3),R2_GGZDOWN,AMPL(1,2279))
      CALL R2_GGZ_0(W(6),W(215),W(3),R2_GGZDOWN,AMPL(1,2280))
      CALL FFV1P0_3(W(1),W(47),GC_5,ZERO,ZERO,W(216))
C     Counter-term amplitude(s) for loop diagram number 1531
      CALL R2_GGZ_0(W(4),W(216),W(3),R2_GGZDOWN,AMPL(1,2281))
      CALL R2_GGZ_0(W(4),W(216),W(3),R2_GGZDOWN,AMPL(1,2282))
C     Counter-term amplitude(s) for loop diagram number 1532
      CALL R2_GGZ_0(W(14),W(212),W(3),R2_GGZDOWN,AMPL(1,2283))
      CALL R2_GGZ_0(W(14),W(212),W(3),R2_GGZDOWN,AMPL(1,2284))
C     Counter-term amplitude(s) for loop diagram number 1542
      CALL R2_GG_1_0(W(14),W(161),R2_GGQ,AMPL(1,2285))
      CALL R2_GG_1_0(W(14),W(161),R2_GGQ,AMPL(1,2286))
      CALL R2_GG_1_0(W(14),W(161),R2_GGQ,AMPL(1,2287))
      CALL R2_GG_1_0(W(14),W(161),R2_GGQ,AMPL(1,2288))
C     Counter-term amplitude(s) for loop diagram number 1543
      CALL VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2289))
      CALL VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2290))
      CALL VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2291))
      CALL VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2292))
      CALL FFV1P0_3(W(1),W(15),GC_5,ZERO,ZERO,W(217))
      CALL VVV1P0_1(W(5),W(217),GC_4,ZERO,ZERO,W(218))
C     Counter-term amplitude(s) for loop diagram number 1545
      CALL R2_GGZ_0(W(4),W(218),W(3),R2_GGZDOWN,AMPL(1,2293))
      CALL R2_GGZ_0(W(4),W(218),W(3),R2_GGZDOWN,AMPL(1,2294))
C     Counter-term amplitude(s) for loop diagram number 1546
      CALL R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECDOWN,AMPL(1,2295))
      CALL R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2296))
      CALL R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECDOWN,AMPL(1,2297))
      CALL R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2298))
      CALL VVV1P0_1(W(4),W(217),GC_4,ZERO,ZERO,W(219))
C     Counter-term amplitude(s) for loop diagram number 1547
      CALL R2_GGZ_0(W(5),W(219),W(3),R2_GGZDOWN,AMPL(1,2299))
      CALL R2_GGZ_0(W(5),W(219),W(3),R2_GGZDOWN,AMPL(1,2300))
      CALL FFV1P0_3(W(1),W(58),GC_5,ZERO,ZERO,W(220))
C     Counter-term amplitude(s) for loop diagram number 1549
      CALL R2_GGZ_0(W(5),W(220),W(3),R2_GGZDOWN,AMPL(1,2301))
      CALL R2_GGZ_0(W(5),W(220),W(3),R2_GGZDOWN,AMPL(1,2302))
      CALL FFV1P0_3(W(1),W(56),GC_5,ZERO,ZERO,W(221))
C     Counter-term amplitude(s) for loop diagram number 1550
      CALL R2_GGZ_0(W(4),W(221),W(3),R2_GGZDOWN,AMPL(1,2303))
      CALL R2_GGZ_0(W(4),W(221),W(3),R2_GGZDOWN,AMPL(1,2304))
C     Counter-term amplitude(s) for loop diagram number 1551
      CALL R2_GGZ_0(W(16),W(217),W(3),R2_GGZDOWN,AMPL(1,2305))
      CALL R2_GGZ_0(W(16),W(217),W(3),R2_GGZDOWN,AMPL(1,2306))
C     Counter-term amplitude(s) for loop diagram number 1561
      CALL R2_GG_1_0(W(16),W(163),R2_GGQ,AMPL(1,2307))
      CALL R2_GG_1_0(W(16),W(163),R2_GGQ,AMPL(1,2308))
      CALL R2_GG_1_0(W(16),W(163),R2_GGQ,AMPL(1,2309))
      CALL R2_GG_1_0(W(16),W(163),R2_GGQ,AMPL(1,2310))
C     Counter-term amplitude(s) for loop diagram number 1562
      CALL VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2311))
      CALL VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2312))
      CALL VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2313))
      CALL VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2314))
      CALL FFV1P0_3(W(35),W(2),GC_5,ZERO,ZERO,W(222))
C     Counter-term amplitude(s) for loop diagram number 1564
      CALL R2_GGZ_0(W(6),W(222),W(3),R2_GGZDOWN,AMPL(1,2315))
      CALL R2_GGZ_0(W(6),W(222),W(3),R2_GGZDOWN,AMPL(1,2316))
      CALL FFV1P0_3(W(1),W(69),GC_5,ZERO,ZERO,W(223))
C     Counter-term amplitude(s) for loop diagram number 1565
      CALL R2_GGZ_0(W(6),W(223),W(3),R2_GGZDOWN,AMPL(1,2317))
      CALL R2_GGZ_0(W(6),W(223),W(3),R2_GGZDOWN,AMPL(1,2318))
      CALL FFV1P0_3(W(36),W(2),GC_5,ZERO,ZERO,W(224))
C     Counter-term amplitude(s) for loop diagram number 1568
      CALL R2_GGZ_0(W(5),W(224),W(3),R2_GGZDOWN,AMPL(1,2319))
      CALL R2_GGZ_0(W(5),W(224),W(3),R2_GGZDOWN,AMPL(1,2320))
      CALL FFV1P0_3(W(1),W(74),GC_5,ZERO,ZERO,W(225))
C     Counter-term amplitude(s) for loop diagram number 1569
      CALL R2_GGZ_0(W(5),W(225),W(3),R2_GGZDOWN,AMPL(1,2321))
      CALL R2_GGZ_0(W(5),W(225),W(3),R2_GGZDOWN,AMPL(1,2322))
      CALL FFV1P0_3(W(37),W(2),GC_5,ZERO,ZERO,W(226))
C     Counter-term amplitude(s) for loop diagram number 1572
      CALL R2_GGZ_0(W(4),W(226),W(3),R2_GGZDOWN,AMPL(1,2323))
      CALL R2_GGZ_0(W(4),W(226),W(3),R2_GGZDOWN,AMPL(1,2324))
      CALL FFV1P0_3(W(1),W(78),GC_5,ZERO,ZERO,W(227))
C     Counter-term amplitude(s) for loop diagram number 1573
      CALL R2_GGZ_0(W(4),W(227),W(3),R2_GGZDOWN,AMPL(1,2325))
      CALL R2_GGZ_0(W(4),W(227),W(3),R2_GGZDOWN,AMPL(1,2326))
C     Counter-term amplitude(s) for loop diagram number 1576
      CALL R2_GGZ_0(W(5),W(166),W(3),R2_GGZUP,AMPL(1,2327))
      CALL R2_GGZ_0(W(5),W(166),W(3),R2_GGZUP,AMPL(1,2328))
C     Counter-term amplitude(s) for loop diagram number 1577
      CALL R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECUP,AMPL(1,2329))
      CALL R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALUP,AMPL(1,2330)
     $ )
      CALL R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECUP,AMPL(1,2331))
      CALL R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALUP,AMPL(1,2332)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1578
      CALL R2_GGZ_0(W(6),W(167),W(3),R2_GGZUP,AMPL(1,2333))
      CALL R2_GGZ_0(W(6),W(167),W(3),R2_GGZUP,AMPL(1,2334))
C     Counter-term amplitude(s) for loop diagram number 1580
      CALL R2_GGZ_0(W(4),W(169),W(3),R2_GGZUP,AMPL(1,2335))
      CALL R2_GGZ_0(W(4),W(169),W(3),R2_GGZUP,AMPL(1,2336))
C     Counter-term amplitude(s) for loop diagram number 1581
      CALL R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECUP,AMPL(1,2337))
      CALL R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALUP,AMPL(1,2338)
     $ )
      CALL R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECUP,AMPL(1,2339))
      CALL R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALUP,AMPL(1,2340)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1582
      CALL R2_GGZ_0(W(6),W(170),W(3),R2_GGZUP,AMPL(1,2341))
      CALL R2_GGZ_0(W(6),W(170),W(3),R2_GGZUP,AMPL(1,2342))
C     Counter-term amplitude(s) for loop diagram number 1584
      CALL R2_GGZ_0(W(4),W(172),W(3),R2_GGZUP,AMPL(1,2343))
      CALL R2_GGZ_0(W(4),W(172),W(3),R2_GGZUP,AMPL(1,2344))
C     Counter-term amplitude(s) for loop diagram number 1585
      CALL R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECUP,AMPL(1,2345))
      CALL R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALUP,AMPL(1,2346)
     $ )
      CALL R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECUP,AMPL(1,2347))
      CALL R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALUP,AMPL(1,2348)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1586
      CALL R2_GGZ_0(W(5),W(173),W(3),R2_GGZUP,AMPL(1,2349))
      CALL R2_GGZ_0(W(5),W(173),W(3),R2_GGZUP,AMPL(1,2350))
C     Counter-term amplitude(s) for loop diagram number 1594
      CALL R2_GGZ_0(W(6),W(174),W(3),R2_GGZUP,AMPL(1,2351))
      CALL R2_GGZ_0(W(6),W(174),W(3),R2_GGZUP,AMPL(1,2352))
      CALL R2_GGZ_0(W(6),W(175),W(3),R2_GGZUP,AMPL(1,2353))
      CALL R2_GGZ_0(W(6),W(175),W(3),R2_GGZUP,AMPL(1,2354))
      CALL R2_GGZ_0(W(6),W(176),W(3),R2_GGZUP,AMPL(1,2355))
      CALL R2_GGZ_0(W(6),W(176),W(3),R2_GGZUP,AMPL(1,2356))
C     Counter-term amplitude(s) for loop diagram number 1595
      CALL R2_GGZ_0(W(5),W(177),W(3),R2_GGZUP,AMPL(1,2357))
      CALL R2_GGZ_0(W(5),W(177),W(3),R2_GGZUP,AMPL(1,2358))
      CALL R2_GGZ_0(W(5),W(178),W(3),R2_GGZUP,AMPL(1,2359))
      CALL R2_GGZ_0(W(5),W(178),W(3),R2_GGZUP,AMPL(1,2360))
      CALL R2_GGZ_0(W(5),W(179),W(3),R2_GGZUP,AMPL(1,2361))
      CALL R2_GGZ_0(W(5),W(179),W(3),R2_GGZUP,AMPL(1,2362))
C     Counter-term amplitude(s) for loop diagram number 1596
      CALL R2_GGZ_0(W(4),W(180),W(3),R2_GGZUP,AMPL(1,2363))
      CALL R2_GGZ_0(W(4),W(180),W(3),R2_GGZUP,AMPL(1,2364))
      CALL R2_GGZ_0(W(4),W(181),W(3),R2_GGZUP,AMPL(1,2365))
      CALL R2_GGZ_0(W(4),W(181),W(3),R2_GGZUP,AMPL(1,2366))
      CALL R2_GGZ_0(W(4),W(182),W(3),R2_GGZUP,AMPL(1,2367))
      CALL R2_GGZ_0(W(4),W(182),W(3),R2_GGZUP,AMPL(1,2368))
C     Counter-term amplitude(s) for loop diagram number 1597
      CALL R2_GGZ_0(W(164),W(65),W(3),R2_GGZUP,AMPL(1,2369))
      CALL R2_GGZ_0(W(164),W(65),W(3),R2_GGZUP,AMPL(1,2370))
C     Counter-term amplitude(s) for loop diagram number 1598
      CALL R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECUP,AMPL(1,2371))
      CALL R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2372))
      CALL R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECUP,AMPL(1,2373))
      CALL R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2374))
C     Counter-term amplitude(s) for loop diagram number 1599
      CALL R2_GGZ_0(W(6),W(183),W(3),R2_GGZUP,AMPL(1,2375))
      CALL R2_GGZ_0(W(6),W(183),W(3),R2_GGZUP,AMPL(1,2376))
C     Counter-term amplitude(s) for loop diagram number 1600
      CALL R2_GGZ_0(W(16),W(171),W(3),R2_GGZUP,AMPL(1,2377))
      CALL R2_GGZ_0(W(16),W(171),W(3),R2_GGZUP,AMPL(1,2378))
C     Counter-term amplitude(s) for loop diagram number 1609
      CALL R2_GGZ_0(W(164),W(66),W(3),R2_GGZUP,AMPL(1,2379))
      CALL R2_GGZ_0(W(164),W(66),W(3),R2_GGZUP,AMPL(1,2380))
C     Counter-term amplitude(s) for loop diagram number 1610
      CALL R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECUP,AMPL(1,2381))
      CALL R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2382))
      CALL R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECUP,AMPL(1,2383))
      CALL R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2384))
C     Counter-term amplitude(s) for loop diagram number 1611
      CALL R2_GGZ_0(W(5),W(184),W(3),R2_GGZUP,AMPL(1,2385))
      CALL R2_GGZ_0(W(5),W(184),W(3),R2_GGZUP,AMPL(1,2386))
C     Counter-term amplitude(s) for loop diagram number 1612
      CALL R2_GGZ_0(W(14),W(168),W(3),R2_GGZUP,AMPL(1,2387))
      CALL R2_GGZ_0(W(14),W(168),W(3),R2_GGZUP,AMPL(1,2388))
C     Counter-term amplitude(s) for loop diagram number 1634
      CALL R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECUP,AMPL(1,2389))
      CALL R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2390))
      CALL R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECUP,AMPL(1,2391))
      CALL R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2392))
C     Counter-term amplitude(s) for loop diagram number 1636
      CALL R2_GGZ_0(W(4),W(185),W(3),R2_GGZUP,AMPL(1,2393))
      CALL R2_GGZ_0(W(4),W(185),W(3),R2_GGZUP,AMPL(1,2394))
C     Counter-term amplitude(s) for loop diagram number 1637
      CALL R2_GGZ_0(W(164),W(67),W(3),R2_GGZUP,AMPL(1,2395))
      CALL R2_GGZ_0(W(164),W(67),W(3),R2_GGZUP,AMPL(1,2396))
C     Counter-term amplitude(s) for loop diagram number 1639
      CALL R2_GGZ_0(W(11),W(165),W(3),R2_GGZUP,AMPL(1,2397))
      CALL R2_GGZ_0(W(11),W(165),W(3),R2_GGZUP,AMPL(1,2398))
C     Counter-term amplitude(s) for loop diagram number 1672
      CALL R2_GGZ_0(W(164),W(18),W(3),R2_GGZUP,AMPL(1,2399))
      CALL R2_GGZ_0(W(164),W(18),W(3),R2_GGZUP,AMPL(1,2400))
      CALL R2_GGZ_0(W(164),W(19),W(3),R2_GGZUP,AMPL(1,2401))
      CALL R2_GGZ_0(W(164),W(19),W(3),R2_GGZUP,AMPL(1,2402))
      CALL R2_GGZ_0(W(164),W(20),W(3),R2_GGZUP,AMPL(1,2403))
      CALL R2_GGZ_0(W(164),W(20),W(3),R2_GGZUP,AMPL(1,2404))
C     Counter-term amplitude(s) for loop diagram number 1674
      CALL R2_GGZ_0(W(6),W(186),W(3),R2_GGZUP,AMPL(1,2405))
      CALL R2_GGZ_0(W(6),W(186),W(3),R2_GGZUP,AMPL(1,2406))
C     Counter-term amplitude(s) for loop diagram number 1676
      CALL R2_GGZ_0(W(5),W(187),W(3),R2_GGZUP,AMPL(1,2407))
      CALL R2_GGZ_0(W(5),W(187),W(3),R2_GGZUP,AMPL(1,2408))
C     Counter-term amplitude(s) for loop diagram number 1678
      CALL R2_GGZ_0(W(5),W(189),W(3),R2_GGZUP,AMPL(1,2409))
      CALL R2_GGZ_0(W(5),W(189),W(3),R2_GGZUP,AMPL(1,2410))
C     Counter-term amplitude(s) for loop diagram number 1679
      CALL R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECUP,AMPL(1,2411))
      CALL R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALUP,AMPL(1,2412)
     $ )
      CALL R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECUP,AMPL(1,2413))
      CALL R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALUP,AMPL(1,2414)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1680
      CALL R2_GGZ_0(W(6),W(190),W(3),R2_GGZUP,AMPL(1,2415))
      CALL R2_GGZ_0(W(6),W(190),W(3),R2_GGZUP,AMPL(1,2416))
C     Counter-term amplitude(s) for loop diagram number 1682
      CALL R2_GGZ_0(W(6),W(191),W(3),R2_GGZUP,AMPL(1,2417))
      CALL R2_GGZ_0(W(6),W(191),W(3),R2_GGZUP,AMPL(1,2418))
C     Counter-term amplitude(s) for loop diagram number 1683
      CALL R2_GGZ_0(W(5),W(192),W(3),R2_GGZUP,AMPL(1,2419))
      CALL R2_GGZ_0(W(5),W(192),W(3),R2_GGZUP,AMPL(1,2420))
C     Counter-term amplitude(s) for loop diagram number 1684
      CALL R2_GGZ_0(W(11),W(188),W(3),R2_GGZUP,AMPL(1,2421))
      CALL R2_GGZ_0(W(11),W(188),W(3),R2_GGZUP,AMPL(1,2422))
C     Counter-term amplitude(s) for loop diagram number 1694
      CALL R2_GGZ_0(W(6),W(193),W(3),R2_GGZUP,AMPL(1,2423))
      CALL R2_GGZ_0(W(6),W(193),W(3),R2_GGZUP,AMPL(1,2424))
C     Counter-term amplitude(s) for loop diagram number 1696
      CALL R2_GGZ_0(W(4),W(194),W(3),R2_GGZUP,AMPL(1,2425))
      CALL R2_GGZ_0(W(4),W(194),W(3),R2_GGZUP,AMPL(1,2426))
C     Counter-term amplitude(s) for loop diagram number 1698
      CALL R2_GGZ_0(W(4),W(196),W(3),R2_GGZUP,AMPL(1,2427))
      CALL R2_GGZ_0(W(4),W(196),W(3),R2_GGZUP,AMPL(1,2428))
C     Counter-term amplitude(s) for loop diagram number 1699
      CALL R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECUP,AMPL(1,2429))
      CALL R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALUP,AMPL(1,2430)
     $ )
      CALL R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECUP,AMPL(1,2431))
      CALL R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALUP,AMPL(1,2432)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1700
      CALL R2_GGZ_0(W(6),W(197),W(3),R2_GGZUP,AMPL(1,2433))
      CALL R2_GGZ_0(W(6),W(197),W(3),R2_GGZUP,AMPL(1,2434))
C     Counter-term amplitude(s) for loop diagram number 1702
      CALL R2_GGZ_0(W(6),W(198),W(3),R2_GGZUP,AMPL(1,2435))
      CALL R2_GGZ_0(W(6),W(198),W(3),R2_GGZUP,AMPL(1,2436))
C     Counter-term amplitude(s) for loop diagram number 1703
      CALL R2_GGZ_0(W(4),W(199),W(3),R2_GGZUP,AMPL(1,2437))
      CALL R2_GGZ_0(W(4),W(199),W(3),R2_GGZUP,AMPL(1,2438))
C     Counter-term amplitude(s) for loop diagram number 1704
      CALL R2_GGZ_0(W(14),W(195),W(3),R2_GGZUP,AMPL(1,2439))
      CALL R2_GGZ_0(W(14),W(195),W(3),R2_GGZUP,AMPL(1,2440))
C     Counter-term amplitude(s) for loop diagram number 1714
      CALL R2_GGZ_0(W(5),W(200),W(3),R2_GGZUP,AMPL(1,2441))
      CALL R2_GGZ_0(W(5),W(200),W(3),R2_GGZUP,AMPL(1,2442))
C     Counter-term amplitude(s) for loop diagram number 1716
      CALL R2_GGZ_0(W(4),W(201),W(3),R2_GGZUP,AMPL(1,2443))
      CALL R2_GGZ_0(W(4),W(201),W(3),R2_GGZUP,AMPL(1,2444))
C     Counter-term amplitude(s) for loop diagram number 1718
      CALL R2_GGZ_0(W(4),W(203),W(3),R2_GGZUP,AMPL(1,2445))
      CALL R2_GGZ_0(W(4),W(203),W(3),R2_GGZUP,AMPL(1,2446))
C     Counter-term amplitude(s) for loop diagram number 1719
      CALL R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECUP,AMPL(1,2447))
      CALL R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALUP,AMPL(1,2448)
     $ )
      CALL R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECUP,AMPL(1,2449))
      CALL R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALUP,AMPL(1,2450)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1720
      CALL R2_GGZ_0(W(5),W(204),W(3),R2_GGZUP,AMPL(1,2451))
      CALL R2_GGZ_0(W(5),W(204),W(3),R2_GGZUP,AMPL(1,2452))
C     Counter-term amplitude(s) for loop diagram number 1722
      CALL R2_GGZ_0(W(5),W(205),W(3),R2_GGZUP,AMPL(1,2453))
      CALL R2_GGZ_0(W(5),W(205),W(3),R2_GGZUP,AMPL(1,2454))
C     Counter-term amplitude(s) for loop diagram number 1723
      CALL R2_GGZ_0(W(4),W(206),W(3),R2_GGZUP,AMPL(1,2455))
      CALL R2_GGZ_0(W(4),W(206),W(3),R2_GGZUP,AMPL(1,2456))
C     Counter-term amplitude(s) for loop diagram number 1724
      CALL R2_GGZ_0(W(16),W(202),W(3),R2_GGZUP,AMPL(1,2457))
      CALL R2_GGZ_0(W(16),W(202),W(3),R2_GGZUP,AMPL(1,2458))
C     Counter-term amplitude(s) for loop diagram number 1734
      CALL R2_GGZ_0(W(5),W(208),W(3),R2_GGZUP,AMPL(1,2459))
      CALL R2_GGZ_0(W(5),W(208),W(3),R2_GGZUP,AMPL(1,2460))
C     Counter-term amplitude(s) for loop diagram number 1735
      CALL R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECUP,AMPL(1,2461))
      CALL R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALUP,AMPL(1,2462)
     $ )
      CALL R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECUP,AMPL(1,2463))
      CALL R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALUP,AMPL(1,2464)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1736
      CALL R2_GGZ_0(W(6),W(209),W(3),R2_GGZUP,AMPL(1,2465))
      CALL R2_GGZ_0(W(6),W(209),W(3),R2_GGZUP,AMPL(1,2466))
C     Counter-term amplitude(s) for loop diagram number 1738
      CALL R2_GGZ_0(W(6),W(210),W(3),R2_GGZUP,AMPL(1,2467))
      CALL R2_GGZ_0(W(6),W(210),W(3),R2_GGZUP,AMPL(1,2468))
C     Counter-term amplitude(s) for loop diagram number 1739
      CALL R2_GGZ_0(W(5),W(211),W(3),R2_GGZUP,AMPL(1,2469))
      CALL R2_GGZ_0(W(5),W(211),W(3),R2_GGZUP,AMPL(1,2470))
C     Counter-term amplitude(s) for loop diagram number 1740
      CALL R2_GGZ_0(W(11),W(207),W(3),R2_GGZUP,AMPL(1,2471))
      CALL R2_GGZ_0(W(11),W(207),W(3),R2_GGZUP,AMPL(1,2472))
C     Counter-term amplitude(s) for loop diagram number 1750
      CALL R2_GGZ_0(W(4),W(213),W(3),R2_GGZUP,AMPL(1,2473))
      CALL R2_GGZ_0(W(4),W(213),W(3),R2_GGZUP,AMPL(1,2474))
C     Counter-term amplitude(s) for loop diagram number 1751
      CALL R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECUP,AMPL(1,2475))
      CALL R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALUP,AMPL(1,2476)
     $ )
      CALL R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECUP,AMPL(1,2477))
      CALL R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALUP,AMPL(1,2478)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1752
      CALL R2_GGZ_0(W(6),W(214),W(3),R2_GGZUP,AMPL(1,2479))
      CALL R2_GGZ_0(W(6),W(214),W(3),R2_GGZUP,AMPL(1,2480))
C     Counter-term amplitude(s) for loop diagram number 1754
      CALL R2_GGZ_0(W(6),W(215),W(3),R2_GGZUP,AMPL(1,2481))
      CALL R2_GGZ_0(W(6),W(215),W(3),R2_GGZUP,AMPL(1,2482))
C     Counter-term amplitude(s) for loop diagram number 1755
      CALL R2_GGZ_0(W(4),W(216),W(3),R2_GGZUP,AMPL(1,2483))
      CALL R2_GGZ_0(W(4),W(216),W(3),R2_GGZUP,AMPL(1,2484))
C     Counter-term amplitude(s) for loop diagram number 1756
      CALL R2_GGZ_0(W(14),W(212),W(3),R2_GGZUP,AMPL(1,2485))
      CALL R2_GGZ_0(W(14),W(212),W(3),R2_GGZUP,AMPL(1,2486))
C     Counter-term amplitude(s) for loop diagram number 1766
      CALL R2_GGZ_0(W(4),W(218),W(3),R2_GGZUP,AMPL(1,2487))
      CALL R2_GGZ_0(W(4),W(218),W(3),R2_GGZUP,AMPL(1,2488))
C     Counter-term amplitude(s) for loop diagram number 1767
      CALL R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECUP,AMPL(1,2489))
      CALL R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALUP,AMPL(1,2490)
     $ )
      CALL R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECUP,AMPL(1,2491))
      CALL R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALUP,AMPL(1,2492)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1768
      CALL R2_GGZ_0(W(5),W(219),W(3),R2_GGZUP,AMPL(1,2493))
      CALL R2_GGZ_0(W(5),W(219),W(3),R2_GGZUP,AMPL(1,2494))
C     Counter-term amplitude(s) for loop diagram number 1770
      CALL R2_GGZ_0(W(5),W(220),W(3),R2_GGZUP,AMPL(1,2495))
      CALL R2_GGZ_0(W(5),W(220),W(3),R2_GGZUP,AMPL(1,2496))
C     Counter-term amplitude(s) for loop diagram number 1771
      CALL R2_GGZ_0(W(4),W(221),W(3),R2_GGZUP,AMPL(1,2497))
      CALL R2_GGZ_0(W(4),W(221),W(3),R2_GGZUP,AMPL(1,2498))
C     Counter-term amplitude(s) for loop diagram number 1772
      CALL R2_GGZ_0(W(16),W(217),W(3),R2_GGZUP,AMPL(1,2499))
      CALL R2_GGZ_0(W(16),W(217),W(3),R2_GGZUP,AMPL(1,2500))
C     Counter-term amplitude(s) for loop diagram number 1782
      CALL R2_GGZ_0(W(6),W(222),W(3),R2_GGZUP,AMPL(1,2501))
      CALL R2_GGZ_0(W(6),W(222),W(3),R2_GGZUP,AMPL(1,2502))
C     Counter-term amplitude(s) for loop diagram number 1783
      CALL R2_GGZ_0(W(6),W(223),W(3),R2_GGZUP,AMPL(1,2503))
      CALL R2_GGZ_0(W(6),W(223),W(3),R2_GGZUP,AMPL(1,2504))
C     Counter-term amplitude(s) for loop diagram number 1786
      CALL R2_GGZ_0(W(5),W(224),W(3),R2_GGZUP,AMPL(1,2505))
      CALL R2_GGZ_0(W(5),W(224),W(3),R2_GGZUP,AMPL(1,2506))
C     Counter-term amplitude(s) for loop diagram number 1787
      CALL R2_GGZ_0(W(5),W(225),W(3),R2_GGZUP,AMPL(1,2507))
      CALL R2_GGZ_0(W(5),W(225),W(3),R2_GGZUP,AMPL(1,2508))
C     Counter-term amplitude(s) for loop diagram number 1790
      CALL R2_GGZ_0(W(4),W(226),W(3),R2_GGZUP,AMPL(1,2509))
      CALL R2_GGZ_0(W(4),W(226),W(3),R2_GGZUP,AMPL(1,2510))
C     Counter-term amplitude(s) for loop diagram number 1791
      CALL R2_GGZ_0(W(4),W(227),W(3),R2_GGZUP,AMPL(1,2511))
      CALL R2_GGZ_0(W(4),W(227),W(3),R2_GGZUP,AMPL(1,2512))
C     Counter-term amplitude(s) for loop diagram number 1794
      CALL R2_GGZ_0(W(5),W(166),W(3),R2_GGZDOWN,AMPL(1,2513))
C     Counter-term amplitude(s) for loop diagram number 1795
      CALL R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECDOWN,AMPL(1,2514))
      CALL R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2515))
C     Counter-term amplitude(s) for loop diagram number 1796
      CALL R2_GGZ_0(W(6),W(167),W(3),R2_GGZDOWN,AMPL(1,2516))
C     Counter-term amplitude(s) for loop diagram number 1798
      CALL R2_GGZ_0(W(4),W(169),W(3),R2_GGZDOWN,AMPL(1,2517))
C     Counter-term amplitude(s) for loop diagram number 1799
      CALL R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECDOWN,AMPL(1,2518))
      CALL R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2519))
C     Counter-term amplitude(s) for loop diagram number 1800
      CALL R2_GGZ_0(W(6),W(170),W(3),R2_GGZDOWN,AMPL(1,2520))
C     Counter-term amplitude(s) for loop diagram number 1802
      CALL R2_GGZ_0(W(4),W(172),W(3),R2_GGZDOWN,AMPL(1,2521))
C     Counter-term amplitude(s) for loop diagram number 1803
      CALL R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECDOWN,AMPL(1,2522))
      CALL R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2523))
C     Counter-term amplitude(s) for loop diagram number 1804
      CALL R2_GGZ_0(W(5),W(173),W(3),R2_GGZDOWN,AMPL(1,2524))
C     Counter-term amplitude(s) for loop diagram number 1812
      CALL R2_GGZ_0(W(6),W(174),W(3),R2_GGZDOWN,AMPL(1,2525))
      CALL R2_GGZ_0(W(6),W(175),W(3),R2_GGZDOWN,AMPL(1,2526))
      CALL R2_GGZ_0(W(6),W(176),W(3),R2_GGZDOWN,AMPL(1,2527))
C     Counter-term amplitude(s) for loop diagram number 1813
      CALL R2_GGZ_0(W(5),W(177),W(3),R2_GGZDOWN,AMPL(1,2528))
      CALL R2_GGZ_0(W(5),W(178),W(3),R2_GGZDOWN,AMPL(1,2529))
      CALL R2_GGZ_0(W(5),W(179),W(3),R2_GGZDOWN,AMPL(1,2530))
C     Counter-term amplitude(s) for loop diagram number 1814
      CALL R2_GGZ_0(W(4),W(180),W(3),R2_GGZDOWN,AMPL(1,2531))
      CALL R2_GGZ_0(W(4),W(181),W(3),R2_GGZDOWN,AMPL(1,2532))
      CALL R2_GGZ_0(W(4),W(182),W(3),R2_GGZDOWN,AMPL(1,2533))
C     Counter-term amplitude(s) for loop diagram number 1815
      CALL R2_GGZ_0(W(164),W(65),W(3),R2_GGZDOWN,AMPL(1,2534))
C     Counter-term amplitude(s) for loop diagram number 1816
      CALL R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECDOWN,AMPL(1,2535)
     $ )
      CALL R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2536))
C     Counter-term amplitude(s) for loop diagram number 1817
      CALL R2_GGZ_0(W(6),W(183),W(3),R2_GGZDOWN,AMPL(1,2537))
C     Counter-term amplitude(s) for loop diagram number 1818
      CALL R2_GGZ_0(W(16),W(171),W(3),R2_GGZDOWN,AMPL(1,2538))
C     Counter-term amplitude(s) for loop diagram number 1827
      CALL R2_GGZ_0(W(164),W(66),W(3),R2_GGZDOWN,AMPL(1,2539))
C     Counter-term amplitude(s) for loop diagram number 1828
      CALL R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECDOWN,AMPL(1,2540)
     $ )
      CALL R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2541))
C     Counter-term amplitude(s) for loop diagram number 1829
      CALL R2_GGZ_0(W(5),W(184),W(3),R2_GGZDOWN,AMPL(1,2542))
C     Counter-term amplitude(s) for loop diagram number 1830
      CALL R2_GGZ_0(W(14),W(168),W(3),R2_GGZDOWN,AMPL(1,2543))
C     Counter-term amplitude(s) for loop diagram number 1852
      CALL R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECDOWN,AMPL(1,2544)
     $ )
      CALL R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2545))
C     Counter-term amplitude(s) for loop diagram number 1854
      CALL R2_GGZ_0(W(4),W(185),W(3),R2_GGZDOWN,AMPL(1,2546))
C     Counter-term amplitude(s) for loop diagram number 1855
      CALL R2_GGZ_0(W(164),W(67),W(3),R2_GGZDOWN,AMPL(1,2547))
C     Counter-term amplitude(s) for loop diagram number 1857
      CALL R2_GGZ_0(W(11),W(165),W(3),R2_GGZDOWN,AMPL(1,2548))
C     Counter-term amplitude(s) for loop diagram number 1890
      CALL R2_GGZ_0(W(164),W(18),W(3),R2_GGZDOWN,AMPL(1,2549))
      CALL R2_GGZ_0(W(164),W(19),W(3),R2_GGZDOWN,AMPL(1,2550))
      CALL R2_GGZ_0(W(164),W(20),W(3),R2_GGZDOWN,AMPL(1,2551))
C     Counter-term amplitude(s) for loop diagram number 1892
      CALL R2_GG_1_R2_GG_3_0(W(11),W(44),R2_GGQ,R2_GGB,AMPL(1,2552))
C     Counter-term amplitude(s) for loop diagram number 1893
      CALL VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,2553))
C     Counter-term amplitude(s) for loop diagram number 1895
      CALL R2_GG_1_R2_GG_3_0(W(14),W(53),R2_GGQ,R2_GGB,AMPL(1,2554))
C     Counter-term amplitude(s) for loop diagram number 1896
      CALL VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,2555))
C     Counter-term amplitude(s) for loop diagram number 1898
      CALL R2_GG_1_R2_GG_3_0(W(16),W(62),R2_GGQ,R2_GGB,AMPL(1,2556))
C     Counter-term amplitude(s) for loop diagram number 1899
      CALL VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,2557))
C     Counter-term amplitude(s) for loop diagram number 1901
      CALL R2_GG_1_R2_GG_3_0(W(17),W(65),R2_GGQ,R2_GGB,AMPL(1,2558))
C     Counter-term amplitude(s) for loop diagram number 1902
      CALL R2_GG_1_R2_GG_3_0(W(16),W(68),R2_GGQ,R2_GGB,AMPL(1,2559))
C     Counter-term amplitude(s) for loop diagram number 1903
      CALL VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,2560))
C     Counter-term amplitude(s) for loop diagram number 1905
      CALL R2_GG_1_R2_GG_3_0(W(16),W(70),R2_GGQ,R2_GGB,AMPL(1,2561))
C     Counter-term amplitude(s) for loop diagram number 1906
      CALL R2_GG_1_R2_GG_3_0(W(17),W(66),R2_GGQ,R2_GGB,AMPL(1,2562))
C     Counter-term amplitude(s) for loop diagram number 1907
      CALL R2_GG_1_R2_GG_3_0(W(14),W(73),R2_GGQ,R2_GGB,AMPL(1,2563))
C     Counter-term amplitude(s) for loop diagram number 1908
      CALL VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,2564))
C     Counter-term amplitude(s) for loop diagram number 1910
      CALL R2_GG_1_R2_GG_3_0(W(14),W(75),R2_GGQ,R2_GGB,AMPL(1,2565))
C     Counter-term amplitude(s) for loop diagram number 1911
      CALL VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,2566))
C     Counter-term amplitude(s) for loop diagram number 1912
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,2567))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,2568))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,2569))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,2570))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,2571))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,2572))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,2573))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2574))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2575))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,2576)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,2577)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1913
      CALL VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,2578))
C     Counter-term amplitude(s) for loop diagram number 1915
      CALL VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,2579))
C     Counter-term amplitude(s) for loop diagram number 1916
      CALL VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,2580))
C     Counter-term amplitude(s) for loop diagram number 1917
      CALL VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,2581))
C     Counter-term amplitude(s) for loop diagram number 1927
      CALL R2_GG_1_R2_GG_3_0(W(17),W(67),R2_GGQ,R2_GGB,AMPL(1,2582))
C     Counter-term amplitude(s) for loop diagram number 1928
      CALL R2_GG_1_R2_GG_3_0(W(11),W(79),R2_GGQ,R2_GGB,AMPL(1,2583))
C     Counter-term amplitude(s) for loop diagram number 1929
      CALL R2_GG_1_R2_GG_3_0(W(11),W(80),R2_GGQ,R2_GGB,AMPL(1,2584))
C     Counter-term amplitude(s) for loop diagram number 1930
      CALL VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2585))
C     Counter-term amplitude(s) for loop diagram number 1931
      CALL VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2586))
C     Counter-term amplitude(s) for loop diagram number 1934
      CALL R2_GG_1_R2_GG_3_0(W(17),W(18),R2_GGQ,R2_GGB,AMPL(1,2587))
      CALL R2_GG_1_R2_GG_3_0(W(17),W(19),R2_GGQ,R2_GGB,AMPL(1,2588))
      CALL R2_GG_1_R2_GG_3_0(W(17),W(20),R2_GGQ,R2_GGB,AMPL(1,2589))
C     Counter-term amplitude(s) for loop diagram number 1935
      CALL R2_GG_1_R2_GG_3_0(W(11),W(92),R2_GGQ,R2_GGB,AMPL(1,2590))
C     Counter-term amplitude(s) for loop diagram number 1936
      CALL VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2591))
C     Counter-term amplitude(s) for loop diagram number 1938
      CALL R2_GGZ_0(W(6),W(186),W(3),R2_GGZDOWN,AMPL(1,2592))
C     Counter-term amplitude(s) for loop diagram number 1940
      CALL R2_GGZ_0(W(5),W(187),W(3),R2_GGZDOWN,AMPL(1,2593))
C     Counter-term amplitude(s) for loop diagram number 1942
      CALL R2_GGZ_0(W(5),W(189),W(3),R2_GGZDOWN,AMPL(1,2594))
C     Counter-term amplitude(s) for loop diagram number 1943
      CALL R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECDOWN,AMPL(1,2595))
      CALL R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2596))
C     Counter-term amplitude(s) for loop diagram number 1944
      CALL R2_GGZ_0(W(6),W(190),W(3),R2_GGZDOWN,AMPL(1,2597))
C     Counter-term amplitude(s) for loop diagram number 1946
      CALL R2_GGZ_0(W(6),W(191),W(3),R2_GGZDOWN,AMPL(1,2598))
C     Counter-term amplitude(s) for loop diagram number 1947
      CALL R2_GGZ_0(W(5),W(192),W(3),R2_GGZDOWN,AMPL(1,2599))
C     Counter-term amplitude(s) for loop diagram number 1948
      CALL R2_GGZ_0(W(11),W(188),W(3),R2_GGZDOWN,AMPL(1,2600))
C     Counter-term amplitude(s) for loop diagram number 1958
      CALL R2_GG_1_R2_GG_3_0(W(11),W(105),R2_GGQ,R2_GGB,AMPL(1,2601))
C     Counter-term amplitude(s) for loop diagram number 1959
      CALL VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2602))
C     Counter-term amplitude(s) for loop diagram number 1961
      CALL R2_GG_1_R2_GG_3_0(W(14),W(112),R2_GGQ,R2_GGB,AMPL(1,2603))
C     Counter-term amplitude(s) for loop diagram number 1962
      CALL VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2604))
C     Counter-term amplitude(s) for loop diagram number 1964
      CALL R2_GGZ_0(W(6),W(193),W(3),R2_GGZDOWN,AMPL(1,2605))
C     Counter-term amplitude(s) for loop diagram number 1966
      CALL R2_GGZ_0(W(4),W(194),W(3),R2_GGZDOWN,AMPL(1,2606))
C     Counter-term amplitude(s) for loop diagram number 1968
      CALL R2_GGZ_0(W(4),W(196),W(3),R2_GGZDOWN,AMPL(1,2607))
C     Counter-term amplitude(s) for loop diagram number 1969
      CALL R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECDOWN,AMPL(1,2608))
      CALL R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2609))
C     Counter-term amplitude(s) for loop diagram number 1970
      CALL R2_GGZ_0(W(6),W(197),W(3),R2_GGZDOWN,AMPL(1,2610))
C     Counter-term amplitude(s) for loop diagram number 1972
      CALL R2_GGZ_0(W(6),W(198),W(3),R2_GGZDOWN,AMPL(1,2611))
C     Counter-term amplitude(s) for loop diagram number 1973
      CALL R2_GGZ_0(W(4),W(199),W(3),R2_GGZDOWN,AMPL(1,2612))
C     Counter-term amplitude(s) for loop diagram number 1974
      CALL R2_GGZ_0(W(14),W(195),W(3),R2_GGZDOWN,AMPL(1,2613))
C     Counter-term amplitude(s) for loop diagram number 1984
      CALL R2_GG_1_R2_GG_3_0(W(14),W(124),R2_GGQ,R2_GGB,AMPL(1,2614))
C     Counter-term amplitude(s) for loop diagram number 1985
      CALL VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2615))
C     Counter-term amplitude(s) for loop diagram number 1987
      CALL R2_GG_1_R2_GG_3_0(W(16),W(130),R2_GGQ,R2_GGB,AMPL(1,2616))
C     Counter-term amplitude(s) for loop diagram number 1988
      CALL VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2617))
C     Counter-term amplitude(s) for loop diagram number 1990
      CALL R2_GGZ_0(W(5),W(200),W(3),R2_GGZDOWN,AMPL(1,2618))
C     Counter-term amplitude(s) for loop diagram number 1992
      CALL R2_GGZ_0(W(4),W(201),W(3),R2_GGZDOWN,AMPL(1,2619))
C     Counter-term amplitude(s) for loop diagram number 1994
      CALL R2_GGZ_0(W(4),W(203),W(3),R2_GGZDOWN,AMPL(1,2620))
C     Counter-term amplitude(s) for loop diagram number 1995
      CALL R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECDOWN,AMPL(1,2621))
      CALL R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2622))
C     Counter-term amplitude(s) for loop diagram number 1996
      CALL R2_GGZ_0(W(5),W(204),W(3),R2_GGZDOWN,AMPL(1,2623))
C     Counter-term amplitude(s) for loop diagram number 1998
      CALL R2_GGZ_0(W(5),W(205),W(3),R2_GGZDOWN,AMPL(1,2624))
C     Counter-term amplitude(s) for loop diagram number 1999
      CALL R2_GGZ_0(W(4),W(206),W(3),R2_GGZDOWN,AMPL(1,2625))
C     Counter-term amplitude(s) for loop diagram number 2000
      CALL R2_GGZ_0(W(16),W(202),W(3),R2_GGZDOWN,AMPL(1,2626))
C     Counter-term amplitude(s) for loop diagram number 2010
      CALL R2_GG_1_R2_GG_3_0(W(16),W(141),R2_GGQ,R2_GGB,AMPL(1,2627))
C     Counter-term amplitude(s) for loop diagram number 2011
      CALL VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2628))
C     Counter-term amplitude(s) for loop diagram number 2013
      CALL R2_GG_1_R2_GG_3_0(W(34),W(65),R2_GGQ,R2_GGB,AMPL(1,2629))
C     Counter-term amplitude(s) for loop diagram number 2014
      CALL R2_GG_1_R2_GG_3_0(W(16),W(143),R2_GGQ,R2_GGB,AMPL(1,2630))
C     Counter-term amplitude(s) for loop diagram number 2015
      CALL VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2631))
C     Counter-term amplitude(s) for loop diagram number 2017
      CALL R2_GG_1_R2_GG_3_0(W(16),W(145),R2_GGQ,R2_GGB,AMPL(1,2632))
C     Counter-term amplitude(s) for loop diagram number 2018
      CALL R2_GG_1_R2_GG_3_0(W(34),W(66),R2_GGQ,R2_GGB,AMPL(1,2633))
C     Counter-term amplitude(s) for loop diagram number 2019
      CALL R2_GG_1_R2_GG_3_0(W(14),W(147),R2_GGQ,R2_GGB,AMPL(1,2634))
C     Counter-term amplitude(s) for loop diagram number 2020
      CALL VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2635))
C     Counter-term amplitude(s) for loop diagram number 2022
      CALL R2_GG_1_R2_GG_3_0(W(14),W(149),R2_GGQ,R2_GGB,AMPL(1,2636))
C     Counter-term amplitude(s) for loop diagram number 2023
      CALL VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2637))
C     Counter-term amplitude(s) for loop diagram number 2024
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2638))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2639))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2640))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2641))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2642))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2643))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,2644))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2645))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2646))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2647)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2648)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2025
      CALL VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2649))
C     Counter-term amplitude(s) for loop diagram number 2027
      CALL VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2650))
C     Counter-term amplitude(s) for loop diagram number 2028
      CALL VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2651))
C     Counter-term amplitude(s) for loop diagram number 2029
      CALL VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2652))
C     Counter-term amplitude(s) for loop diagram number 2039
      CALL R2_GG_1_R2_GG_3_0(W(34),W(67),R2_GGQ,R2_GGB,AMPL(1,2653))
C     Counter-term amplitude(s) for loop diagram number 2040
      CALL R2_GG_1_R2_GG_3_0(W(11),W(151),R2_GGQ,R2_GGB,AMPL(1,2654))
C     Counter-term amplitude(s) for loop diagram number 2041
      CALL R2_GG_1_R2_GG_3_0(W(11),W(153),R2_GGQ,R2_GGB,AMPL(1,2655))
C     Counter-term amplitude(s) for loop diagram number 2042
      CALL VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2656))
C     Counter-term amplitude(s) for loop diagram number 2043
      CALL VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2657))
C     Counter-term amplitude(s) for loop diagram number 2046
      CALL R2_GG_1_R2_GG_3_0(W(34),W(18),R2_GGQ,R2_GGB,AMPL(1,2658))
      CALL R2_GG_1_R2_GG_3_0(W(34),W(19),R2_GGQ,R2_GGB,AMPL(1,2659))
      CALL R2_GG_1_R2_GG_3_0(W(34),W(20),R2_GGQ,R2_GGB,AMPL(1,2660))
C     Counter-term amplitude(s) for loop diagram number 2047
      CALL R2_GGZ_0(W(5),W(208),W(3),R2_GGZDOWN,AMPL(1,2661))
C     Counter-term amplitude(s) for loop diagram number 2048
      CALL R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECDOWN,AMPL(1,2662))
      CALL R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2663))
C     Counter-term amplitude(s) for loop diagram number 2049
      CALL R2_GGZ_0(W(6),W(209),W(3),R2_GGZDOWN,AMPL(1,2664))
C     Counter-term amplitude(s) for loop diagram number 2051
      CALL R2_GGZ_0(W(6),W(210),W(3),R2_GGZDOWN,AMPL(1,2665))
C     Counter-term amplitude(s) for loop diagram number 2052
      CALL R2_GGZ_0(W(5),W(211),W(3),R2_GGZDOWN,AMPL(1,2666))
C     Counter-term amplitude(s) for loop diagram number 2053
      CALL R2_GGZ_0(W(11),W(207),W(3),R2_GGZDOWN,AMPL(1,2667))
C     Counter-term amplitude(s) for loop diagram number 2063
      CALL R2_GG_1_R2_GG_3_0(W(11),W(159),R2_GGQ,R2_GGB,AMPL(1,2668))
C     Counter-term amplitude(s) for loop diagram number 2064
      CALL VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2669))
C     Counter-term amplitude(s) for loop diagram number 2066
      CALL R2_GGZ_0(W(4),W(213),W(3),R2_GGZDOWN,AMPL(1,2670))
C     Counter-term amplitude(s) for loop diagram number 2067
      CALL R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECDOWN,AMPL(1,2671))
      CALL R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2672))
C     Counter-term amplitude(s) for loop diagram number 2068
      CALL R2_GGZ_0(W(6),W(214),W(3),R2_GGZDOWN,AMPL(1,2673))
C     Counter-term amplitude(s) for loop diagram number 2070
      CALL R2_GGZ_0(W(6),W(215),W(3),R2_GGZDOWN,AMPL(1,2674))
C     Counter-term amplitude(s) for loop diagram number 2071
      CALL R2_GGZ_0(W(4),W(216),W(3),R2_GGZDOWN,AMPL(1,2675))
C     Counter-term amplitude(s) for loop diagram number 2072
      CALL R2_GGZ_0(W(14),W(212),W(3),R2_GGZDOWN,AMPL(1,2676))
C     Counter-term amplitude(s) for loop diagram number 2082
      CALL R2_GG_1_R2_GG_3_0(W(14),W(161),R2_GGQ,R2_GGB,AMPL(1,2677))
C     Counter-term amplitude(s) for loop diagram number 2083
      CALL VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2678))
C     Counter-term amplitude(s) for loop diagram number 2085
      CALL R2_GGZ_0(W(4),W(218),W(3),R2_GGZDOWN,AMPL(1,2679))
C     Counter-term amplitude(s) for loop diagram number 2086
      CALL R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECDOWN,AMPL(1,2680))
      CALL R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,2681))
C     Counter-term amplitude(s) for loop diagram number 2087
      CALL R2_GGZ_0(W(5),W(219),W(3),R2_GGZDOWN,AMPL(1,2682))
C     Counter-term amplitude(s) for loop diagram number 2089
      CALL R2_GGZ_0(W(5),W(220),W(3),R2_GGZDOWN,AMPL(1,2683))
C     Counter-term amplitude(s) for loop diagram number 2090
      CALL R2_GGZ_0(W(4),W(221),W(3),R2_GGZDOWN,AMPL(1,2684))
C     Counter-term amplitude(s) for loop diagram number 2091
      CALL R2_GGZ_0(W(16),W(217),W(3),R2_GGZDOWN,AMPL(1,2685))
C     Counter-term amplitude(s) for loop diagram number 2101
      CALL R2_GG_1_R2_GG_3_0(W(16),W(163),R2_GGQ,R2_GGB,AMPL(1,2686))
C     Counter-term amplitude(s) for loop diagram number 2102
      CALL VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2687))
C     Counter-term amplitude(s) for loop diagram number 2104
      CALL R2_GGZ_0(W(6),W(222),W(3),R2_GGZDOWN,AMPL(1,2688))
C     Counter-term amplitude(s) for loop diagram number 2105
      CALL R2_GGZ_0(W(6),W(223),W(3),R2_GGZDOWN,AMPL(1,2689))
C     Counter-term amplitude(s) for loop diagram number 2108
      CALL R2_GGZ_0(W(5),W(224),W(3),R2_GGZDOWN,AMPL(1,2690))
C     Counter-term amplitude(s) for loop diagram number 2109
      CALL R2_GGZ_0(W(5),W(225),W(3),R2_GGZDOWN,AMPL(1,2691))
C     Counter-term amplitude(s) for loop diagram number 2112
      CALL R2_GGZ_0(W(4),W(226),W(3),R2_GGZDOWN,AMPL(1,2692))
C     Counter-term amplitude(s) for loop diagram number 2113
      CALL R2_GGZ_0(W(4),W(227),W(3),R2_GGZDOWN,AMPL(1,2693))
C     Counter-term amplitude(s) for loop diagram number 2116
      CALL R2_GGZ_0(W(5),W(166),W(3),R2_GGZUP,AMPL(1,2694))
C     Counter-term amplitude(s) for loop diagram number 2117
      CALL R2_GGVV_0(W(5),W(6),W(165),W(3),R2_GGGZVECUP,AMPL(1,2695))
      CALL R2_GGGVA_0(W(5),W(6),W(165),W(3),R2_GGGZAXIALUP,AMPL(1,2696)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2118
      CALL R2_GGZ_0(W(6),W(167),W(3),R2_GGZUP,AMPL(1,2697))
C     Counter-term amplitude(s) for loop diagram number 2120
      CALL R2_GGZ_0(W(4),W(169),W(3),R2_GGZUP,AMPL(1,2698))
C     Counter-term amplitude(s) for loop diagram number 2121
      CALL R2_GGVV_0(W(4),W(6),W(168),W(3),R2_GGGZVECUP,AMPL(1,2699))
      CALL R2_GGGVA_0(W(4),W(6),W(168),W(3),R2_GGGZAXIALUP,AMPL(1,2700)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2122
      CALL R2_GGZ_0(W(6),W(170),W(3),R2_GGZUP,AMPL(1,2701))
C     Counter-term amplitude(s) for loop diagram number 2124
      CALL R2_GGZ_0(W(4),W(172),W(3),R2_GGZUP,AMPL(1,2702))
C     Counter-term amplitude(s) for loop diagram number 2125
      CALL R2_GGVV_0(W(4),W(5),W(171),W(3),R2_GGGZVECUP,AMPL(1,2703))
      CALL R2_GGGVA_0(W(4),W(5),W(171),W(3),R2_GGGZAXIALUP,AMPL(1,2704)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2126
      CALL R2_GGZ_0(W(5),W(173),W(3),R2_GGZUP,AMPL(1,2705))
C     Counter-term amplitude(s) for loop diagram number 2134
      CALL R2_GGZ_0(W(6),W(174),W(3),R2_GGZUP,AMPL(1,2706))
      CALL R2_GGZ_0(W(6),W(175),W(3),R2_GGZUP,AMPL(1,2707))
      CALL R2_GGZ_0(W(6),W(176),W(3),R2_GGZUP,AMPL(1,2708))
C     Counter-term amplitude(s) for loop diagram number 2135
      CALL R2_GGZ_0(W(5),W(177),W(3),R2_GGZUP,AMPL(1,2709))
      CALL R2_GGZ_0(W(5),W(178),W(3),R2_GGZUP,AMPL(1,2710))
      CALL R2_GGZ_0(W(5),W(179),W(3),R2_GGZUP,AMPL(1,2711))
C     Counter-term amplitude(s) for loop diagram number 2136
      CALL R2_GGZ_0(W(4),W(180),W(3),R2_GGZUP,AMPL(1,2712))
      CALL R2_GGZ_0(W(4),W(181),W(3),R2_GGZUP,AMPL(1,2713))
      CALL R2_GGZ_0(W(4),W(182),W(3),R2_GGZUP,AMPL(1,2714))
C     Counter-term amplitude(s) for loop diagram number 2137
      CALL R2_GGZ_0(W(164),W(65),W(3),R2_GGZUP,AMPL(1,2715))
C     Counter-term amplitude(s) for loop diagram number 2138
      CALL R2_GGVV_0(W(164),W(6),W(16),W(3),R2_GGGZVECUP,AMPL(1,2716))
      CALL R2_GGGVA_0(W(164),W(6),W(16),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2717))
C     Counter-term amplitude(s) for loop diagram number 2139
      CALL R2_GGZ_0(W(6),W(183),W(3),R2_GGZUP,AMPL(1,2718))
C     Counter-term amplitude(s) for loop diagram number 2140
      CALL R2_GGZ_0(W(16),W(171),W(3),R2_GGZUP,AMPL(1,2719))
C     Counter-term amplitude(s) for loop diagram number 2149
      CALL R2_GGZ_0(W(164),W(66),W(3),R2_GGZUP,AMPL(1,2720))
C     Counter-term amplitude(s) for loop diagram number 2150
      CALL R2_GGVV_0(W(164),W(5),W(14),W(3),R2_GGGZVECUP,AMPL(1,2721))
      CALL R2_GGGVA_0(W(164),W(5),W(14),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2722))
C     Counter-term amplitude(s) for loop diagram number 2151
      CALL R2_GGZ_0(W(5),W(184),W(3),R2_GGZUP,AMPL(1,2723))
C     Counter-term amplitude(s) for loop diagram number 2152
      CALL R2_GGZ_0(W(14),W(168),W(3),R2_GGZUP,AMPL(1,2724))
C     Counter-term amplitude(s) for loop diagram number 2174
      CALL R2_GGVV_0(W(4),W(11),W(164),W(3),R2_GGGZVECUP,AMPL(1,2725))
      CALL R2_GGGVA_0(W(4),W(11),W(164),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,2726))
C     Counter-term amplitude(s) for loop diagram number 2176
      CALL R2_GGZ_0(W(4),W(185),W(3),R2_GGZUP,AMPL(1,2727))
C     Counter-term amplitude(s) for loop diagram number 2177
      CALL R2_GGZ_0(W(164),W(67),W(3),R2_GGZUP,AMPL(1,2728))
C     Counter-term amplitude(s) for loop diagram number 2179
      CALL R2_GGZ_0(W(11),W(165),W(3),R2_GGZUP,AMPL(1,2729))
C     Counter-term amplitude(s) for loop diagram number 2212
      CALL R2_GGZ_0(W(164),W(18),W(3),R2_GGZUP,AMPL(1,2730))
      CALL R2_GGZ_0(W(164),W(19),W(3),R2_GGZUP,AMPL(1,2731))
      CALL R2_GGZ_0(W(164),W(20),W(3),R2_GGZUP,AMPL(1,2732))
C     Counter-term amplitude(s) for loop diagram number 2214
      CALL R2_GG_1_R2_GG_3_0(W(11),W(44),R2_GGQ,R2_GGT,AMPL(1,2733))
C     Counter-term amplitude(s) for loop diagram number 2215
      CALL VVV1_0(W(5),W(6),W(44),R2_3GQ,AMPL(1,2734))
C     Counter-term amplitude(s) for loop diagram number 2217
      CALL R2_GG_1_R2_GG_3_0(W(14),W(53),R2_GGQ,R2_GGT,AMPL(1,2735))
C     Counter-term amplitude(s) for loop diagram number 2218
      CALL VVV1_0(W(4),W(6),W(53),R2_3GQ,AMPL(1,2736))
C     Counter-term amplitude(s) for loop diagram number 2220
      CALL R2_GG_1_R2_GG_3_0(W(16),W(62),R2_GGQ,R2_GGT,AMPL(1,2737))
C     Counter-term amplitude(s) for loop diagram number 2221
      CALL VVV1_0(W(4),W(5),W(62),R2_3GQ,AMPL(1,2738))
C     Counter-term amplitude(s) for loop diagram number 2223
      CALL R2_GG_1_R2_GG_3_0(W(17),W(65),R2_GGQ,R2_GGT,AMPL(1,2739))
C     Counter-term amplitude(s) for loop diagram number 2224
      CALL R2_GG_1_R2_GG_3_0(W(16),W(68),R2_GGQ,R2_GGT,AMPL(1,2740))
C     Counter-term amplitude(s) for loop diagram number 2225
      CALL VVV1_0(W(6),W(16),W(17),R2_3GQ,AMPL(1,2741))
C     Counter-term amplitude(s) for loop diagram number 2227
      CALL R2_GG_1_R2_GG_3_0(W(16),W(70),R2_GGQ,R2_GGT,AMPL(1,2742))
C     Counter-term amplitude(s) for loop diagram number 2228
      CALL R2_GG_1_R2_GG_3_0(W(17),W(66),R2_GGQ,R2_GGT,AMPL(1,2743))
C     Counter-term amplitude(s) for loop diagram number 2229
      CALL R2_GG_1_R2_GG_3_0(W(14),W(73),R2_GGQ,R2_GGT,AMPL(1,2744))
C     Counter-term amplitude(s) for loop diagram number 2230
      CALL VVV1_0(W(5),W(14),W(17),R2_3GQ,AMPL(1,2745))
C     Counter-term amplitude(s) for loop diagram number 2232
      CALL R2_GG_1_R2_GG_3_0(W(14),W(75),R2_GGQ,R2_GGT,AMPL(1,2746))
C     Counter-term amplitude(s) for loop diagram number 2233
      CALL VVV1_0(W(4),W(5),W(68),R2_3GQ,AMPL(1,2747))
C     Counter-term amplitude(s) for loop diagram number 2234
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,2748))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,2749))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_43,AMPL(1,2750))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,2751))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,2752))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_58,AMPL(1,2753))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,2754))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2755))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2756))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,2757)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_48,AMPL(1,2758)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2235
      CALL VVV1_0(W(4),W(6),W(73),R2_3GQ,AMPL(1,2759))
C     Counter-term amplitude(s) for loop diagram number 2237
      CALL VVV1_0(W(4),W(6),W(75),R2_3GQ,AMPL(1,2760))
C     Counter-term amplitude(s) for loop diagram number 2238
      CALL VVV1_0(W(4),W(5),W(70),R2_3GQ,AMPL(1,2761))
C     Counter-term amplitude(s) for loop diagram number 2239
      CALL VVV1_0(W(4),W(11),W(17),R2_3GQ,AMPL(1,2762))
C     Counter-term amplitude(s) for loop diagram number 2249
      CALL R2_GG_1_R2_GG_3_0(W(17),W(67),R2_GGQ,R2_GGT,AMPL(1,2763))
C     Counter-term amplitude(s) for loop diagram number 2250
      CALL R2_GG_1_R2_GG_3_0(W(11),W(79),R2_GGQ,R2_GGT,AMPL(1,2764))
C     Counter-term amplitude(s) for loop diagram number 2251
      CALL R2_GG_1_R2_GG_3_0(W(11),W(80),R2_GGQ,R2_GGT,AMPL(1,2765))
C     Counter-term amplitude(s) for loop diagram number 2252
      CALL VVV1_0(W(5),W(6),W(79),R2_3GQ,AMPL(1,2766))
C     Counter-term amplitude(s) for loop diagram number 2253
      CALL VVV1_0(W(5),W(6),W(80),R2_3GQ,AMPL(1,2767))
C     Counter-term amplitude(s) for loop diagram number 2256
      CALL R2_GG_1_R2_GG_3_0(W(17),W(18),R2_GGQ,R2_GGT,AMPL(1,2768))
      CALL R2_GG_1_R2_GG_3_0(W(17),W(19),R2_GGQ,R2_GGT,AMPL(1,2769))
      CALL R2_GG_1_R2_GG_3_0(W(17),W(20),R2_GGQ,R2_GGT,AMPL(1,2770))
C     Counter-term amplitude(s) for loop diagram number 2257
      CALL R2_GG_1_R2_GG_3_0(W(11),W(92),R2_GGQ,R2_GGT,AMPL(1,2771))
C     Counter-term amplitude(s) for loop diagram number 2258
      CALL VVV1_0(W(5),W(6),W(92),R2_3GQ,AMPL(1,2772))
C     Counter-term amplitude(s) for loop diagram number 2260
      CALL R2_GGZ_0(W(6),W(186),W(3),R2_GGZUP,AMPL(1,2773))
C     Counter-term amplitude(s) for loop diagram number 2262
      CALL R2_GGZ_0(W(5),W(187),W(3),R2_GGZUP,AMPL(1,2774))
C     Counter-term amplitude(s) for loop diagram number 2264
      CALL R2_GGZ_0(W(5),W(189),W(3),R2_GGZUP,AMPL(1,2775))
C     Counter-term amplitude(s) for loop diagram number 2265
      CALL R2_GGVV_0(W(5),W(6),W(188),W(3),R2_GGGZVECUP,AMPL(1,2776))
      CALL R2_GGGVA_0(W(5),W(6),W(188),W(3),R2_GGGZAXIALUP,AMPL(1,2777)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2266
      CALL R2_GGZ_0(W(6),W(190),W(3),R2_GGZUP,AMPL(1,2778))
C     Counter-term amplitude(s) for loop diagram number 2268
      CALL R2_GGZ_0(W(6),W(191),W(3),R2_GGZUP,AMPL(1,2779))
C     Counter-term amplitude(s) for loop diagram number 2269
      CALL R2_GGZ_0(W(5),W(192),W(3),R2_GGZUP,AMPL(1,2780))
C     Counter-term amplitude(s) for loop diagram number 2270
      CALL R2_GGZ_0(W(11),W(188),W(3),R2_GGZUP,AMPL(1,2781))
C     Counter-term amplitude(s) for loop diagram number 2280
      CALL R2_GG_1_R2_GG_3_0(W(11),W(105),R2_GGQ,R2_GGT,AMPL(1,2782))
C     Counter-term amplitude(s) for loop diagram number 2281
      CALL VVV1_0(W(5),W(6),W(105),R2_3GQ,AMPL(1,2783))
C     Counter-term amplitude(s) for loop diagram number 2283
      CALL R2_GG_1_R2_GG_3_0(W(14),W(112),R2_GGQ,R2_GGT,AMPL(1,2784))
C     Counter-term amplitude(s) for loop diagram number 2284
      CALL VVV1_0(W(4),W(6),W(112),R2_3GQ,AMPL(1,2785))
C     Counter-term amplitude(s) for loop diagram number 2286
      CALL R2_GGZ_0(W(6),W(193),W(3),R2_GGZUP,AMPL(1,2786))
C     Counter-term amplitude(s) for loop diagram number 2288
      CALL R2_GGZ_0(W(4),W(194),W(3),R2_GGZUP,AMPL(1,2787))
C     Counter-term amplitude(s) for loop diagram number 2290
      CALL R2_GGZ_0(W(4),W(196),W(3),R2_GGZUP,AMPL(1,2788))
C     Counter-term amplitude(s) for loop diagram number 2291
      CALL R2_GGVV_0(W(4),W(6),W(195),W(3),R2_GGGZVECUP,AMPL(1,2789))
      CALL R2_GGGVA_0(W(4),W(6),W(195),W(3),R2_GGGZAXIALUP,AMPL(1,2790)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2292
      CALL R2_GGZ_0(W(6),W(197),W(3),R2_GGZUP,AMPL(1,2791))
C     Counter-term amplitude(s) for loop diagram number 2294
      CALL R2_GGZ_0(W(6),W(198),W(3),R2_GGZUP,AMPL(1,2792))
C     Counter-term amplitude(s) for loop diagram number 2295
      CALL R2_GGZ_0(W(4),W(199),W(3),R2_GGZUP,AMPL(1,2793))
C     Counter-term amplitude(s) for loop diagram number 2296
      CALL R2_GGZ_0(W(14),W(195),W(3),R2_GGZUP,AMPL(1,2794))
C     Counter-term amplitude(s) for loop diagram number 2306
      CALL R2_GG_1_R2_GG_3_0(W(14),W(124),R2_GGQ,R2_GGT,AMPL(1,2795))
C     Counter-term amplitude(s) for loop diagram number 2307
      CALL VVV1_0(W(4),W(6),W(124),R2_3GQ,AMPL(1,2796))
C     Counter-term amplitude(s) for loop diagram number 2309
      CALL R2_GG_1_R2_GG_3_0(W(16),W(130),R2_GGQ,R2_GGT,AMPL(1,2797))
C     Counter-term amplitude(s) for loop diagram number 2310
      CALL VVV1_0(W(4),W(5),W(130),R2_3GQ,AMPL(1,2798))
C     Counter-term amplitude(s) for loop diagram number 2312
      CALL R2_GGZ_0(W(5),W(200),W(3),R2_GGZUP,AMPL(1,2799))
C     Counter-term amplitude(s) for loop diagram number 2314
      CALL R2_GGZ_0(W(4),W(201),W(3),R2_GGZUP,AMPL(1,2800))
C     Counter-term amplitude(s) for loop diagram number 2316
      CALL R2_GGZ_0(W(4),W(203),W(3),R2_GGZUP,AMPL(1,2801))
C     Counter-term amplitude(s) for loop diagram number 2317
      CALL R2_GGVV_0(W(4),W(5),W(202),W(3),R2_GGGZVECUP,AMPL(1,2802))
      CALL R2_GGGVA_0(W(4),W(5),W(202),W(3),R2_GGGZAXIALUP,AMPL(1,2803)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2318
      CALL R2_GGZ_0(W(5),W(204),W(3),R2_GGZUP,AMPL(1,2804))
C     Counter-term amplitude(s) for loop diagram number 2320
      CALL R2_GGZ_0(W(5),W(205),W(3),R2_GGZUP,AMPL(1,2805))
C     Counter-term amplitude(s) for loop diagram number 2321
      CALL R2_GGZ_0(W(4),W(206),W(3),R2_GGZUP,AMPL(1,2806))
C     Counter-term amplitude(s) for loop diagram number 2322
      CALL R2_GGZ_0(W(16),W(202),W(3),R2_GGZUP,AMPL(1,2807))
C     Counter-term amplitude(s) for loop diagram number 2332
      CALL R2_GG_1_R2_GG_3_0(W(16),W(141),R2_GGQ,R2_GGT,AMPL(1,2808))
C     Counter-term amplitude(s) for loop diagram number 2333
      CALL VVV1_0(W(4),W(5),W(141),R2_3GQ,AMPL(1,2809))
C     Counter-term amplitude(s) for loop diagram number 2335
      CALL R2_GG_1_R2_GG_3_0(W(34),W(65),R2_GGQ,R2_GGT,AMPL(1,2810))
C     Counter-term amplitude(s) for loop diagram number 2336
      CALL R2_GG_1_R2_GG_3_0(W(16),W(143),R2_GGQ,R2_GGT,AMPL(1,2811))
C     Counter-term amplitude(s) for loop diagram number 2337
      CALL VVV1_0(W(6),W(16),W(34),R2_3GQ,AMPL(1,2812))
C     Counter-term amplitude(s) for loop diagram number 2339
      CALL R2_GG_1_R2_GG_3_0(W(16),W(145),R2_GGQ,R2_GGT,AMPL(1,2813))
C     Counter-term amplitude(s) for loop diagram number 2340
      CALL R2_GG_1_R2_GG_3_0(W(34),W(66),R2_GGQ,R2_GGT,AMPL(1,2814))
C     Counter-term amplitude(s) for loop diagram number 2341
      CALL R2_GG_1_R2_GG_3_0(W(14),W(147),R2_GGQ,R2_GGT,AMPL(1,2815))
C     Counter-term amplitude(s) for loop diagram number 2342
      CALL VVV1_0(W(5),W(14),W(34),R2_3GQ,AMPL(1,2816))
C     Counter-term amplitude(s) for loop diagram number 2344
      CALL R2_GG_1_R2_GG_3_0(W(14),W(149),R2_GGQ,R2_GGT,AMPL(1,2817))
C     Counter-term amplitude(s) for loop diagram number 2345
      CALL VVV1_0(W(4),W(5),W(143),R2_3GQ,AMPL(1,2818))
C     Counter-term amplitude(s) for loop diagram number 2346
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2819))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2820))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_137_43,AMPL(1,2821))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2822))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2823))
      CALL R2RGA_VVVV1_0(W(4),W(5),W(6),W(34),R2GC_145_58,AMPL(1,2824))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(34),R2GC_141_50
     $ ,R2GC_142_52,AMPL(1,2825))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_144_56,R2GC_138_45,R2GC_143_54,AMPL(1,2826))
      CALL R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6),W(34)
     $ ,R2GC_138_45,R2GC_144_56,R2GC_143_54,AMPL(1,2827))
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2828)
     $ )
      CALL R2RGA_VVVV10_0(W(4),W(5),W(6),W(34),R2GC_140_48,AMPL(1,2829)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2347
      CALL VVV1_0(W(4),W(6),W(147),R2_3GQ,AMPL(1,2830))
C     Counter-term amplitude(s) for loop diagram number 2349
      CALL VVV1_0(W(4),W(6),W(149),R2_3GQ,AMPL(1,2831))
C     Counter-term amplitude(s) for loop diagram number 2350
      CALL VVV1_0(W(4),W(5),W(145),R2_3GQ,AMPL(1,2832))
C     Counter-term amplitude(s) for loop diagram number 2351
      CALL VVV1_0(W(4),W(11),W(34),R2_3GQ,AMPL(1,2833))
C     Counter-term amplitude(s) for loop diagram number 2361
      CALL R2_GG_1_R2_GG_3_0(W(34),W(67),R2_GGQ,R2_GGT,AMPL(1,2834))
C     Counter-term amplitude(s) for loop diagram number 2362
      CALL R2_GG_1_R2_GG_3_0(W(11),W(151),R2_GGQ,R2_GGT,AMPL(1,2835))
C     Counter-term amplitude(s) for loop diagram number 2363
      CALL R2_GG_1_R2_GG_3_0(W(11),W(153),R2_GGQ,R2_GGT,AMPL(1,2836))
C     Counter-term amplitude(s) for loop diagram number 2364
      CALL VVV1_0(W(5),W(6),W(151),R2_3GQ,AMPL(1,2837))
C     Counter-term amplitude(s) for loop diagram number 2365
      CALL VVV1_0(W(5),W(6),W(153),R2_3GQ,AMPL(1,2838))
C     Counter-term amplitude(s) for loop diagram number 2368
      CALL R2_GG_1_R2_GG_3_0(W(34),W(18),R2_GGQ,R2_GGT,AMPL(1,2839))
      CALL R2_GG_1_R2_GG_3_0(W(34),W(19),R2_GGQ,R2_GGT,AMPL(1,2840))
      CALL R2_GG_1_R2_GG_3_0(W(34),W(20),R2_GGQ,R2_GGT,AMPL(1,2841))
C     Counter-term amplitude(s) for loop diagram number 2369
      CALL R2_GGZ_0(W(5),W(208),W(3),R2_GGZUP,AMPL(1,2842))
C     Counter-term amplitude(s) for loop diagram number 2370
      CALL R2_GGVV_0(W(5),W(6),W(207),W(3),R2_GGGZVECUP,AMPL(1,2843))
      CALL R2_GGGVA_0(W(5),W(6),W(207),W(3),R2_GGGZAXIALUP,AMPL(1,2844)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2371
      CALL R2_GGZ_0(W(6),W(209),W(3),R2_GGZUP,AMPL(1,2845))
C     Counter-term amplitude(s) for loop diagram number 2373
      CALL R2_GGZ_0(W(6),W(210),W(3),R2_GGZUP,AMPL(1,2846))
C     Counter-term amplitude(s) for loop diagram number 2374
      CALL R2_GGZ_0(W(5),W(211),W(3),R2_GGZUP,AMPL(1,2847))
C     Counter-term amplitude(s) for loop diagram number 2375
      CALL R2_GGZ_0(W(11),W(207),W(3),R2_GGZUP,AMPL(1,2848))
C     Counter-term amplitude(s) for loop diagram number 2385
      CALL R2_GG_1_R2_GG_3_0(W(11),W(159),R2_GGQ,R2_GGT,AMPL(1,2849))
C     Counter-term amplitude(s) for loop diagram number 2386
      CALL VVV1_0(W(5),W(6),W(159),R2_3GQ,AMPL(1,2850))
C     Counter-term amplitude(s) for loop diagram number 2388
      CALL R2_GGZ_0(W(4),W(213),W(3),R2_GGZUP,AMPL(1,2851))
C     Counter-term amplitude(s) for loop diagram number 2389
      CALL R2_GGVV_0(W(4),W(6),W(212),W(3),R2_GGGZVECUP,AMPL(1,2852))
      CALL R2_GGGVA_0(W(4),W(6),W(212),W(3),R2_GGGZAXIALUP,AMPL(1,2853)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2390
      CALL R2_GGZ_0(W(6),W(214),W(3),R2_GGZUP,AMPL(1,2854))
C     Counter-term amplitude(s) for loop diagram number 2392
      CALL R2_GGZ_0(W(6),W(215),W(3),R2_GGZUP,AMPL(1,2855))
C     Counter-term amplitude(s) for loop diagram number 2393
      CALL R2_GGZ_0(W(4),W(216),W(3),R2_GGZUP,AMPL(1,2856))

      GOTO 1001
 2000 CONTINUE
      CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

