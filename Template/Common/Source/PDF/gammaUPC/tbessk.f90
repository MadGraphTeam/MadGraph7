!***********************************************************
!* Modified Bessel functions K0, K1, I0, I1, K_n (integer n)*
!* and the exponentially scaled K0e, K1e, I0e, I1e for real *
!* arguments, in double precision.                         *
!* ------------------------------------------------------- *
!* BESSK(N,X): K_N(X) by upward recursion from K0 and K1   *
!*   (from the Numath Library by Tuan Dang Trong, F90      *
!*   version by J-P Moreau, www.jpmoreau.fr).               *
!* BESSK0, BESSK1, BESSI0, BESSI1 and the scaled            *
!*   GUPC_BESSK0E(X) = exp(X)    K0(X)                      *
!*   GUPC_BESSK1E(X) = exp(X)    K1(X)                      *
!*   GUPC_BESSI0E(X) = exp(-|X|) I0(X)                      *
!*   GUPC_BESSI1E(X) = exp(-|X|) I1(X)                      *
!*   use the Chebyshev expansions of the Cephes Math       *
!*   Library (k0.c, k1.c, i0.c, i1.c, S. L. Moshier), same  *
!*   functional forms and number of terms; the coefficients *
!*   were recomputed with mpmath at 50 digits (script       *
!*   benchmark/unit/gen_cephes_coeffs.py). Relative error   *
!*   ~1e-16 (max ~1e-15), versus ~1e-7 for the Abramowitz-  *
!*   Stegun polynomial fits used before.                    *
!* K0, K1 at X=0 return 1D30, 1D32 as before; X<0 is not    *
!* defined for K (NaN). This file must not use modules.     *
!***********************************************************

      FUNCTION BESSK(N,X)
      IMPLICIT NONE
      INTEGER N,J
      REAL *8 X,BESSK,BESSK0,BESSK1,TOX,BK,BKM,BKP
! ------------------------------------------------------------------------
!     CE SOUS-PROGRAMME CALCULE LA FONCTION BESSEL MODIFIFIEE 3E ESPECE
!     D'ORDRE N ENTIER POUR TOUT X REEL POSITIF > 0.  ON UTILISE ICI LA
!     FORMULE DE RECURRENCE CLASSIQUE EN PARTANT DE BESSK0 ET BESSK1.
!
!     THIS ROUTINE CALCULATES THE MODIFIED BESSEL FUNCTION OF THE THIRD
!     KIND OF INTEGER ORDER, N FOR ANY POSITIVE REAL ARGUMENT, X. THE
!     CLASSICAL RECURSION FORMULA IS USED, STARTING FROM BESSK0 AND BESSK1.
! ------------------------------------------------------------------------
!     REFERENCE:
!     C.W.CLENSHAW, CHEBYSHEV SERIES FOR MATHEMATICAL FUNCTIONS,
!     MATHEMATICAL TABLES, VOL.5, 1962.
! ------------------------------------------------------------------------
      IF (N.EQ.0) THEN
      BESSK = BESSK0(X)
      RETURN
      ENDIF
      IF (N.EQ.1) THEN
      BESSK = BESSK1(X)
      RETURN
      ENDIF
      IF (X.EQ.0.D0) THEN
      BESSK = 1.D30
      RETURN
      ENDIF
      TOX = 2.D0/X
      BK  = BESSK1(X)
      BKM = BESSK0(X)
      DO 11 J=1,N-1
      BKP = BKM+DFLOAT(J)*TOX*BK
      BKM = BK
      BK  = BKP
   11 CONTINUE
      BESSK = BK
      RETURN
      END
! ----------------------------------------------------------------------
!     K0(X), modified Bessel function of the second kind of order 0, X>0.
! ----------------------------------------------------------------------
      FUNCTION BESSK0(X)
      IMPLICIT NONE
      REAL*8 X,BESSK0,GUPC_BESSK0_CORE
      BESSK0=GUPC_BESSK0_CORE(X,0)
      RETURN
      END
! ----------------------------------------------------------------------
!     K1(X), modified Bessel function of the second kind of order 1, X>0.
! ----------------------------------------------------------------------
      FUNCTION BESSK1(X)
      IMPLICIT NONE
      REAL*8 X,BESSK1,GUPC_BESSK1_CORE
      BESSK1=GUPC_BESSK1_CORE(X,0)
      RETURN
      END
! ----------------------------------------------------------------------
!     I0(X), modified Bessel function of the first kind of order 0.
! ----------------------------------------------------------------------
      FUNCTION BESSI0(X)
      IMPLICIT NONE
      REAL*8 X,BESSI0,GUPC_BESSI0_CORE
      BESSI0=GUPC_BESSI0_CORE(X,0)
      RETURN
      END
! ----------------------------------------------------------------------
!     I1(X), modified Bessel function of the first kind of order 1.
! ----------------------------------------------------------------------
      FUNCTION BESSI1(X)
      IMPLICIT NONE
      REAL*8 X,BESSI1,GUPC_BESSI1_CORE
      BESSI1=GUPC_BESSI1_CORE(X,0)
      RETURN
      END
! ----------------------------------------------------------------------
!     exp(X) K0(X), X>0
! ----------------------------------------------------------------------
      FUNCTION GUPC_BESSK0E(X)
      IMPLICIT NONE
      REAL*8 X,GUPC_BESSK0E,GUPC_BESSK0_CORE
      GUPC_BESSK0E=GUPC_BESSK0_CORE(X,1)
      RETURN
      END
! ----------------------------------------------------------------------
!     exp(X) K1(X), X>0
! ----------------------------------------------------------------------
      FUNCTION GUPC_BESSK1E(X)
      IMPLICIT NONE
      REAL*8 X,GUPC_BESSK1E,GUPC_BESSK1_CORE
      GUPC_BESSK1E=GUPC_BESSK1_CORE(X,1)
      RETURN
      END
! ----------------------------------------------------------------------
!     exp(-|X|) I0(X)
! ----------------------------------------------------------------------
      FUNCTION GUPC_BESSI0E(X)
      IMPLICIT NONE
      REAL*8 X,GUPC_BESSI0E,GUPC_BESSI0_CORE
      GUPC_BESSI0E=GUPC_BESSI0_CORE(X,1)
      RETURN
      END
! ----------------------------------------------------------------------
!     exp(-|X|) I1(X)
! ----------------------------------------------------------------------
      FUNCTION GUPC_BESSI1E(X)
      IMPLICIT NONE
      REAL*8 X,GUPC_BESSI1E,GUPC_BESSI1_CORE
      GUPC_BESSI1E=GUPC_BESSI1_CORE(X,1)
      RETURN
      END
! ----------------------------------------------------------------------
!     Chebyshev series (Cephes chbevl): with Y = 2*T, T in [-1,1],
!     returns C(1)*T_{N-1}(T) + ... + C(N-1)*T_1(T) + C(N)/2.
! ----------------------------------------------------------------------
      FUNCTION GUPC_CHBEVL(Y,C,N)
      IMPLICIT NONE
      INTEGER N,I
      REAL*8 GUPC_CHBEVL,Y,C(N),B0,B1,B2
      B0=C(1)
      B1=0.D0
      B2=0.D0
      DO I=2,N
         B2=B1
         B1=B0
         B0=Y*B1-B2+C(I)
      ENDDO
      GUPC_CHBEVL=0.5D0*(B0-B2)
      RETURN
      END
! ----------------------------------------------------------------------
!     K0 core: ISCALE=0 returns K0(X), ISCALE=1 returns exp(X) K0(X).
! ----------------------------------------------------------------------
      FUNCTION GUPC_BESSK0_CORE(X,ISCALE)
      IMPLICIT NONE
      REAL*8 X,GUPC_BESSK0_CORE,GUPC_CHBEVL,GUPC_BESSI0_CORE,V
      INTEGER ISCALE
!     Chebyshev coefficients for K0(x) + log(x/2) I0(x) in x^2 on [0,4]
!     (x in [0,2]); lim(x->0) = -Euler gamma.
      REAL*8,PARAMETER::A(10)=(/ &
      1.37446543588075089694D-16, 4.25981614279108257652D-14, 1.03496952576336245851D-11, &
      1.90451637722020885897D-9, 2.53479107902614945731D-7, 2.28621210311945178608D-5, &
      1.26461541144692592338D-3, 3.59799365153615016266D-2, 3.44289899924628486886D-1, &
      -5.3532739323390276872D-1 /)
!     Chebyshev coefficients for exp(x) sqrt(x) K0(x) in 8/x on [0,4]
!     (x in [2,inf)); lim(x->inf) = sqrt(pi/2).
      REAL*8,PARAMETER::B(25)=(/ &
      5.30043377117733577104D-18, -1.64758059398426328153D-17, 5.21039177764355411254D-17, &
      -1.67823112575490063832D-16, 5.51205599940433336489D-16, -1.84859337792090716941D-15, &
      6.34007647627664596613D-15, -2.22751332674629636045D-14, 8.03289077506837436945D-14, &
      -2.98009692314817835483D-13, 1.14034058820734423473D-12, -4.51459788337451917507D-12, &
      1.85594911495492655497D-11, -7.95748924447739703773D-11, 3.57739728140032844716D-10, &
      -1.69753450938906151564D-9, 8.57403401741422608582D-9, -4.66048989768794766556D-8, &
      2.76681363944501507614D-7, -1.83175552271911948478D-6, 1.39498137188764993641D-5, &
      -1.28495495816278026384D-4, 1.56988388573005337491D-3, -3.14481013119645005427D-2, &
      2.44030308206595545468D0 /)
      IF(X.EQ.0.D0) THEN
         GUPC_BESSK0_CORE=1.D30
         RETURN
      ELSEIF(X.LT.0.D0) THEN
         GUPC_BESSK0_CORE=LOG(X)
         RETURN
      ENDIF
      IF(X.LE.2.D0) THEN
         V=GUPC_CHBEVL(X*X-2.D0,A,10)-LOG(0.5D0*X)*GUPC_BESSI0_CORE(X,0)
         IF(ISCALE.NE.0) V=V*EXP(X)
      ELSE
         V=GUPC_CHBEVL(8.D0/X-2.D0,B,25)/SQRT(X)
         IF(ISCALE.EQ.0) V=V*EXP(-X)
      ENDIF
      GUPC_BESSK0_CORE=V
      RETURN
      END
! ----------------------------------------------------------------------
!     K1 core: ISCALE=0 returns K1(X), ISCALE=1 returns exp(X) K1(X).
! ----------------------------------------------------------------------
      FUNCTION GUPC_BESSK1_CORE(X,ISCALE)
      IMPLICIT NONE
      REAL*8 X,GUPC_BESSK1_CORE,GUPC_CHBEVL,GUPC_BESSI1_CORE,V
      INTEGER ISCALE
!     Chebyshev coefficients for x (K1(x) - log(x/2) I1(x)) in x^2 on [0,4]
!     (x in [0,2]); lim(x->0) = 1.
      REAL*8,PARAMETER::A(11)=(/ &
      -7.02386347938628759718D-18, -2.42744985051936593393D-15, -6.66690169419932900609D-13, &
      -1.4114883926335277611D-10, -2.21338763073472585583D-8, -2.43340614156596823496D-6, &
      -1.73028895751305206302D-4, -6.97572385963986435018D-3, -1.22611180822657148235D-1, &
      -3.53155960776544875667D-1, 1.52530022733894777053D0 /)
!     Chebyshev coefficients for exp(x) sqrt(x) K1(x) in 8/x on [0,4]
!     (x in [2,inf)); lim(x->inf) = sqrt(pi/2).
      REAL*8,PARAMETER::B(25)=(/ &
      -5.75674448207330245029D-18, 1.79405104788635729143D-17, -5.68946284919364837425D-17, &
      1.83809357524304542556D-16, -6.05704727064301782278D-16, 2.03870316623986087993D-15, &
      -7.01983708921476885131D-15, 2.47715442421959868133D-14, -8.97670518201014606915D-14, &
      3.3484196660522431201D-13, -1.2891739609498229352D-12, 5.1396396734823435404D-12, &
      -2.12996783842779102155D-11, 9.21831518760531412583D-11, -4.19035475934192558424D-10, &
      2.01504975519703461615D-9, -1.03457624656780970267D-8, 5.74108412545004929231D-8, &
      -3.5019606030878125421D-7, 2.40648494783721711706D-6, -1.93619797416608296002D-5, &
      1.95215518471351631108D-4, -2.8578168596227793868D-3, 1.03923736576817238437D-1, &
      2.72062619048444266945D0 /)
      IF(X.EQ.0.D0) THEN
         GUPC_BESSK1_CORE=1.D32
         RETURN
      ELSEIF(X.LT.0.D0) THEN
         GUPC_BESSK1_CORE=LOG(X)
         RETURN
      ENDIF
      IF(X.LE.2.D0) THEN
         V=LOG(0.5D0*X)*GUPC_BESSI1_CORE(X,0)+GUPC_CHBEVL(X*X-2.D0,A,11)/X
         IF(ISCALE.NE.0) V=V*EXP(X)
      ELSE
         V=GUPC_CHBEVL(8.D0/X-2.D0,B,25)/SQRT(X)
         IF(ISCALE.EQ.0) V=V*EXP(-X)
      ENDIF
      GUPC_BESSK1_CORE=V
      RETURN
      END
! ----------------------------------------------------------------------
!     I0 core: ISCALE=0 returns I0(X), ISCALE=1 returns exp(-|X|) I0(X).
! ----------------------------------------------------------------------
      FUNCTION GUPC_BESSI0_CORE(X,ISCALE)
      IMPLICIT NONE
      REAL*8 X,GUPC_BESSI0_CORE,GUPC_CHBEVL,Z,V,T
      INTEGER ISCALE
!     Chebyshev coefficients for exp(-x) I0(x) in x/2-2 (x in [0,8]);
!     lim(x->0) = 1.
      REAL*8,PARAMETER::A(30)=(/ &
      -4.4153416464793393795D-18, 3.33079451882223809783D-17, -2.43127984654795469359D-16, &
      1.71539128555513303061D-15, -1.16853328779934516808D-14, 7.67618549860493561688D-14, &
      -4.8564467831119294609D-13, 2.95505266312963983461D-12, -1.72682629144155570723D-11, &
      9.67580903537323691224D-11, -5.18979560163526290666D-10, 2.65982372468238665035D-9, &
      -1.30002500998624804212D-8, 6.04699502254191894932D-8, -2.67079385394061173391D-7, &
      1.11738753912010371815D-6, -4.41673835845875056359D-6, 1.64484480707288970893D-5, &
      -5.75419501008210370398D-5, 1.88502885095841655729D-4, -5.76375574538582365885D-4, &
      1.63947561694133579842D-3, -4.3243099950505759443D-3, 1.05464603945949983183D-2, &
      -2.37374148058994688156D-2, 4.93052842396707084878D-2, -9.4901097048047644421D-2, &
      1.71620901522208775349D-1, -3.04682672343198398683D-1, 6.76795274409476084995D-1 /)
!     Chebyshev coefficients for exp(-x) sqrt(x) I0(x) in 32/x-2
!     (x in [8,inf)); lim(x->inf) = 1/sqrt(2 pi).
      REAL*8,PARAMETER::B(25)=(/ &
      -7.23318048787475395456D-18, -4.83050448594418207126D-18, 4.46562142029675999901D-17, &
      3.4612228676974610931D-17, -2.82762398051658348494D-16, -3.42548561967721913462D-16, &
      1.7725601330565263836D-15, 3.81168066935262242075D-15, -9.5548466988283076487D-15, &
      -4.15056934728722208663D-14, 1.54008621752140982691D-14, 3.85277838274214270114D-13, &
      7.18012445138366623367D-13, -1.79417853150680611778D-12, -1.32158118404477131188D-11, &
      -3.14991652796324136454D-11, 1.18891471078464383424D-11, 4.9406023882249695891D-10, &
      3.39623202570838634515D-9, 2.26666899049817806459D-8, 2.04891858946906374183D-7, &
      2.89137052083475648297D-6, 6.88975834691682398426D-5, 3.3691164782556940899D-3, &
      8.04490411014108831608D-1 /)
      Z=ABS(X)
      IF(Z.LE.8.D0) THEN
         V=GUPC_CHBEVL(0.5D0*Z-2.D0,A,30)
      ELSE
         V=GUPC_CHBEVL(32.D0/Z-2.D0,B,25)/SQRT(Z)
      ENDIF
      IF(ISCALE.EQ.0) THEN
         IF(Z.LE.700.D0) THEN
            V=V*EXP(Z)
         ELSE
!           avoid the overflow of exp(Z) for 709.8 < Z < 713.9
            T=EXP(0.5D0*Z)
            V=(V*T)*T
         ENDIF
      ENDIF
      GUPC_BESSI0_CORE=V
      RETURN
      END
! ----------------------------------------------------------------------
!     I1 core: ISCALE=0 returns I1(X), ISCALE=1 returns exp(-|X|) I1(X).
! ----------------------------------------------------------------------
      FUNCTION GUPC_BESSI1_CORE(X,ISCALE)
      IMPLICIT NONE
      REAL*8 X,GUPC_BESSI1_CORE,GUPC_CHBEVL,Z,V,T
      INTEGER ISCALE
!     Chebyshev coefficients for exp(-x) I1(x)/x in x/2-2 (x in [0,8]);
!     lim(x->0) = 1/2.
      REAL*8,PARAMETER::A(29)=(/ &
      2.77791411276104637049D-18, -2.11142121435816607824D-17, 1.55363195773620046892D-16, &
      -1.10559694773538630803D-15, 7.60068429473540693407D-15, -5.04218550472791168711D-14, &
      3.22379336594557470981D-13, -1.9839743977649437152D-12, 1.17361862988909016308D-11, &
      -6.66348972350202774223D-11, 3.62559028155211703701D-10, -1.8872497517228292879D-9, &
      9.38153738649577178388D-9, -4.44505912879632808065D-8, 2.00329475355213526229D-7, &
      -8.56872026469545474066D-7, 3.47025130813767847674D-6, -1.32731636560394358279D-5, &
      4.78156510755005422638D-5, -1.61760815825896745588D-4, 5.12285956168575772895D-4, &
      -1.51357245063125314899D-3, 4.15642294431288815669D-3, -1.05640848946261981558D-2, &
      2.47264490306265168283D-2, -5.29459812080949914269D-2, 1.02643658689847095384D-1, &
      -1.76416518357834055153D-1, 2.52587186443633654823D-1 /)
!     Chebyshev coefficients for exp(-x) sqrt(x) I1(x) in 32/x-2
!     (x in [8,inf)); lim(x->inf) = 1/sqrt(2 pi).
      REAL*8,PARAMETER::B(25)=(/ &
      7.51729631084210480543D-18, 4.41434832307170794995D-18, -4.65030536848935832557D-17, &
      -3.20952592199342395878D-17, 2.96262899764595013907D-16, 3.30820231092092828273D-16, &
      -1.88035477551078244851D-15, -3.81440307243700780477D-15, 1.04202769841288027642D-14, &
      4.2724400167119513543D-14, -2.10154184277266431302D-14, -4.08355111109219731823D-13, &
      -7.19855177624590851209D-13, 2.03562854414708950722D-12, 1.41258074366137813316D-11, &
      3.25260358301548823856D-11, -1.8974958123505412345D-11, -5.58974346219658380687D-10, &
      -3.83538038596423702205D-9, -2.63146884688951950684D-8, -2.51223623787020892529D-7, &
      -3.88256480887769039346D-6, -1.10588938762623716291D-4, -9.76109749136146840777D-3, &
      7.78576235018280120474D-1 /)
      Z=ABS(X)
      IF(Z.LE.8.D0) THEN
         V=GUPC_CHBEVL(0.5D0*Z-2.D0,A,29)*Z
      ELSE
         V=GUPC_CHBEVL(32.D0/Z-2.D0,B,25)/SQRT(Z)
      ENDIF
      IF(ISCALE.EQ.0) THEN
         IF(Z.LE.700.D0) THEN
            V=V*EXP(Z)
         ELSE
            T=EXP(0.5D0*Z)
            V=(V*T)*T
         ENDIF
      ENDIF
      IF(X.LT.0.D0) V=-V
      GUPC_BESSI1_CORE=V
      RETURN
      END

! End of file Tbessk.f90
