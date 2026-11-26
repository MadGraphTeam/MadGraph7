      INTEGER    NONIA
      INTEGER    NPWAVE
      INTEGER    DER_ORDER
      PARAMETER (NONIA=2)
      PARAMETER (NPWAVE=2)
      PARAMETER (DER_ORDER=1)
      INTEGER, PARAMETER :: ND_MAX(3) = (/1,1,2/)
      INTEGER, PARAMETER :: BN_ARRAY(2) = (/1,2/)
      INTEGER :: SN_ARRAY(NPWAVE,NPWAVE)
      INTEGER :: SPLIT(1:2**NPWAVE-1,2,NPWAVE)
      INTEGER :: FALLFACT_INT(-4:6,NPWAVE)
      REAL*8 :: FALLFACT_REAL(-1:2,NPWAVE)

      INTEGER    SN, FF
      DATA (SN_ARRAY(1, SN), SN = 1, 2) /1,0/
      DATA (SN_ARRAY(2, SN), SN = 1, 2) /1,1/
      DATA (SPLIT(1, 1, SN), SN = 1, 1) /1/
      DATA (SPLIT(2, 1, SN), SN = 1, 1) /2/
      DATA (SPLIT(3, 1, SN), SN = 1, 1) /3/
      DATA (SPLIT(3, 2, SN), SN = 1, 2) /1,2/

      DATA (FALLFACT_INT(-4, FF), FF = 1, 2) /-4,20/
      DATA (FALLFACT_INT(-3, FF), FF = 1, 2) /-3,12/
      DATA (FALLFACT_INT(-2, FF), FF = 1, 2) /-2,6/
      DATA (FALLFACT_INT(-1, FF), FF = 1, 2) /-1,2/
      DATA (FALLFACT_INT(0, FF), FF = 1, 2) /0,0/
      DATA (FALLFACT_INT(1, FF), FF = 1, 2) /1,0/
      DATA (FALLFACT_INT(2, FF), FF = 1, 2) /2,2/
      DATA (FALLFACT_INT(3, FF), FF = 1, 2) /3,6/
      DATA (FALLFACT_INT(4, FF), FF = 1, 2) /4,12/
      DATA (FALLFACT_REAL(-1, FF), FF = 1, 2) /-1.5,3.75/  !POW = -1.5
      DATA (FALLFACT_REAL(0, FF), FF = 1, 2) /-0.5,0.75/  !POW = -0.5
      DATA (FALLFACT_REAL(1, FF), FF = 1, 2) /0.5,-0.25/  !POW = 0.5
      DATA (FALLFACT_REAL(2, FF), FF = 1, 2) /1.5,0.75/  !POW = 1.5
