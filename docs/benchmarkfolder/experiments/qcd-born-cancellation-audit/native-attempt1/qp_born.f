      SUBROUTINE AUDIT_QP_GET_AMP(P,NHEL,IC,FLAV_IDX,AMP,MDL_MZ,GC_4,GC_5,GC_22,GC_23)
      USE MODEL_OBJECT
C     
C     Process: u u~ > z g g [ virt = QCD ]
C     
CF2PY INTENT(OUT) :: AMP
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: IC
CF2PY INTENT(IN) :: FLAV_IDX

      USE ALOHA_OBJECT
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=8)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=10, NCOLOR=2)
      REAL*16     ZERO
      PARAMETER (ZERO=0Q0)
C     
C     ARGUMENTS
C     
      REAL*16 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
      INTEGER FLAVOR(NEXTERNAL)
C     
C     LOCAL VARIABLES
C     
      COMPLEX*32 AMP(NGRAPHS)
      TYPE(MP_ALOHA) W(NWAVEFUNCS)
      COMPLEX*32 DUM0,DUM1
      DATA DUM0, DUM1/(0Q0, 0Q0), (1Q0, 0Q0)/
      REAL*16 BWCUTOFF
C     Flavor table for the FLAV_IDX -> FLAVOR rebuild.
      INTEGER NMASK_FLAV
      PARAMETER (NMASK_FLAV=1)
      INTEGER MASK_J
      INTEGER FLAV_TABLE(NEXTERNAL, NMASK_FLAV)
      DATA FLAV_TABLE / 1, 1, 1, 1, 1 /
C     
C     GLOBAL VARIABLES
C     
      REAL*16 MDL_MZ
      COMPLEX*32 GC_4,GC_5,GC_22,GC_23

C     
C     
      BWCUTOFF=15  ! use if $ syntax is defined in the process
C     Rebuild FLAVOR(NEXTERNAL) from the resolved flavor index.
      IF (FLAV_IDX .GE. 1 .AND. FLAV_IDX .LE. NMASK_FLAV) THEN
        DO MASK_J = 1, NEXTERNAL
          FLAVOR(MASK_J) = FLAV_TABLE(MASK_J, FLAV_IDX)
        ENDDO
      ELSE
        DO MASK_J = 1, NEXTERNAL
          FLAVOR(MASK_J) = FLAV_TABLE(MASK_J, 1)
        ENDDO
      ENDIF
      CALL MP_IXXXXX(P(0,1),ZERO,NHEL(1),+1, FLAVOR(1),W(1))
      CALL MP_OXXXXX(P(0,2),ZERO,NHEL(2),-1, FLAVOR(2),W(2))
      CALL MP_VXXXXX(P(0,3),MDL_MZ,NHEL(3),+1,W(3))
      CALL MP_VXXXXX(P(0,4),ZERO,NHEL(4),+1,W(4))
      CALL MP_VXXXXX(P(0,5),ZERO,NHEL(5),+1,W(5))
      CALL MP_FFV2_5_2(W(1),W(3),GC_22,GC_23,ZERO,ZERO,W(6))
      CALL MP_FFV1_1(W(2),W(4),GC_5,ZERO,ZERO,W(7))
C     Amplitude(s) for diagram number 1
      CALL MP_FFV1_0(W(6),W(7),W(5),GC_5,AMP(1))
      CALL MP_FFV1_1(W(2),W(5),GC_5,ZERO,ZERO,W(8))
C     Amplitude(s) for diagram number 2
      CALL MP_FFV1_0(W(6),W(8),W(4),GC_5,AMP(2))
      CALL MP_VVV1P0_1(W(4),W(5),GC_4,ZERO,ZERO,W(9))
C     Amplitude(s) for diagram number 3
      CALL MP_FFV1_0(W(6),W(2),W(9),GC_5,AMP(3))
      CALL MP_FFV1_2(W(1),W(4),GC_5,ZERO,ZERO,W(6))
      CALL MP_FFV2_5_1(W(2),W(3),GC_22,GC_23,ZERO,ZERO,W(10))
C     Amplitude(s) for diagram number 4
      CALL MP_FFV1_0(W(6),W(10),W(5),GC_5,AMP(4))
C     Amplitude(s) for diagram number 5
      CALL MP_FFV2_5_0(W(6),W(8),W(3),GC_22,GC_23,AMP(5))
      CALL MP_FFV1_2(W(1),W(5),GC_5,ZERO,ZERO,W(6))
C     Amplitude(s) for diagram number 6
      CALL MP_FFV1_0(W(6),W(10),W(4),GC_5,AMP(6))
C     Amplitude(s) for diagram number 7
      CALL MP_FFV2_5_0(W(6),W(7),W(3),GC_22,GC_23,AMP(7))
C     Amplitude(s) for diagram number 8
      CALL MP_FFV1_0(W(1),W(10),W(9),GC_5,AMP(8))

      END


      SUBROUTINE AUDIT_QP_GET_JAMP(AMP,JAMP)
C     
C     Process: u u~ > z g g [ virt = QCD ]
C     
CF2PY INTENT(OUT) :: JAMP
CF2PY INTENT(IN) :: AMP

      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=8)
      INTEGER    NCOLOR
      PARAMETER ( NCOLOR=2)
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=1, NSQAMPSO=1)
      COMPLEX*32 IMAG1
      PARAMETER (IMAG1=(0Q0,1Q0))
      COMPLEX*32 AMP(NGRAPHS), TMP_JAMP(1)
      COMPLEX*32 JAMP(NCOLOR,NAMPSO)

      JAMP(:,:) = 0Q0
C     JAMPs contributing to orders QCD=2
      TMP_JAMP(1) = AMP(3) +  AMP(8)  ! used 2 times
      JAMP(1,1) = AMP(1)+AMP(6)+AMP(7)+((0.000000000000000Q+00,
     $ -1.000000000000000Q+00))*TMP_JAMP(1)
      JAMP(2,1) = AMP(2)+AMP(4)+AMP(5)+((0.000000000000000Q+00
     $ ,1.000000000000000Q+00))*TMP_JAMP(1)
      END

      SUBROUTINE AUDIT_QP_GET_MATRIX(JAMP, RES)
C     
C     Process: u u~ > z g g [ virt = QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
CF2PY INTENT(OUT) :: MATRIX
CF2PY INTENT(IN) :: JAMP(NCOLOR,NAMPSO)


      INTEGER    NCOLOR
      PARAMETER (NCOLOR=2)
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=1, NSQAMPSO=1)
      REAL*16     ZERO,RES(NSQAMPSO)
      PARAMETER (ZERO=0Q0)
      INTEGER ML5_0_SQSOINDEX
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,M,N
      COMPLEX*32 ZTEMP

      INTEGER CF_INDEX
      INTEGER ML5_0_CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER ML5_0_DENOM
      COMMON /ML5_0_COLOR_MATRIX/ ML5_0_CF,ML5_0_DENOM
      COMPLEX*32 JAMP(NCOLOR,NAMPSO), TMP_JAMP(1)
      COMPLEX*32 DUM0,DUM1
      DATA DUM0, DUM1/(0Q0, 0Q0), (1Q0, 0Q0)/
C     
C     COLOR DATA
C     

      RES(:) = 0.Q0
      DO M = 1, NAMPSO
        CF_INDEX= 0
        DO I = 1, NCOLOR
          ZTEMP = (0.Q0,0.Q0)
          DO J = I, NCOLOR
            CF_INDEX = CF_INDEX +1
            ZTEMP = ZTEMP + ML5_0_CF(CF_INDEX)*JAMP(J,M)
          ENDDO
          DO N = 1, NAMPSO
            RES(ML5_0_SQSOINDEX(M,N)) = RES(ML5_0_SQSOINDEX(M,N)) +
     $        REAL(ZTEMP*CONJG(JAMP(I,N)))
          ENDDO
        ENDDO
      ENDDO
      RES(:) = RES(:)/ML5_0_DENOM
      END

