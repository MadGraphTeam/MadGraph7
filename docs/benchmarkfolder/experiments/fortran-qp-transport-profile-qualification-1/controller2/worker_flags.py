"""Add read-only IEEE observations to the exact sealed profiling worker.

Only the two PROFILE anchors are observed. Sticky flags are never cleared:
pre/post values describe the actual environment, not newly raised exceptions.
The five flags are INVALID, DIVIDE_BY_ZERO, OVERFLOW, UNDERFLOW, INEXACT.
Unsupported observations are -1. Rounding is -1 if unavailable, 0 if an
unrecognized mode, 1 nearest, 2 toward zero, 3 upward, or 4 downward.
The source transformation performs no compilation or native execution.
"""

from __future__ import annotations

import hashlib

BASE_WORKER_SHA256 = "c085c319426aa2bdbe35ef85b8f4893492bd4f31c870fde3a0abc5353a3f2672"
FLAGS_FILENAME = "ieee-flags.dat"
FLAGS_COLUMNS = (
    "mode", "point", "phase", "rounding",
    "f_invalid", "f_divide_by_zero", "f_overflow", "f_underflow", "f_inexact",
    "h_invalid", "h_divide_by_zero", "h_overflow", "h_underflow", "h_inexact",
)

DECLARATIONS = """      INTEGER QP_FP_FLAGS(5,2),QP_FP_HALTING(5,2)
      INTEGER QP_FP_ROUND(2),QP_FP_PHASE
"""

OPEN_FLAGS = """         OPEN(77,FILE='ieee-flags.dat',STATUS='NEW',
     $        ACTION='WRITE')
         WRITE(77,'(A)')
     $ '# mode point phase rounding f_invalid f_divide_by_zero '//
     $ 'f_overflow f_underflow f_inexact h_invalid '//
     $ 'h_divide_by_zero h_overflow h_underflow h_inexact'
         WRITE(77,'(A)')
     $ '# phase 1=before 2=after; flags/halting -1=unsupported '//
     $ '0=false 1=true; sticky flags are not cleared'
         WRITE(77,'(A)')
     $ '# rounding -1=unavailable 0=other 1=nearest '//
     $ '2=to_zero 3=up 4=down'
"""

ANCHOR = """            CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
"""
OBSERVED_ANCHOR = """            CALL QP_CAPTURE_FP(QP_FP_FLAGS(:,1),
     $           QP_FP_HALTING(:,1),QP_FP_ROUND(1))
""" + ANCHOR + """            CALL QP_CAPTURE_FP(QP_FP_FLAGS(:,2),
     $           QP_FP_HALTING(:,2),QP_FP_ROUND(2))
            DO QP_FP_PHASE=1,2
               WRITE(77,'(14I12)') QP_MODE,QP_POINT,QP_FP_PHASE,
     $              QP_FP_ROUND(QP_FP_PHASE),
     $              QP_FP_FLAGS(:,QP_FP_PHASE),
     $              QP_FP_HALTING(:,QP_FP_PHASE)
            ENDDO
            FLUSH(77)
"""

CAPTURE = """
      SUBROUTINE QP_CAPTURE_FP(FLAGS,HALTING,ROUNDING)
C     Read-only queries; no floating arithmetic, formatting or setters.
      USE, INTRINSIC :: IEEE_EXCEPTIONS
      USE, INTRINSIC :: IEEE_ARITHMETIC
      IMPLICIT NONE
      INTEGER, INTENT(OUT) :: FLAGS(5),HALTING(5),ROUNDING
      TYPE(IEEE_FLAG_TYPE), PARAMETER :: KINDS(5)=
     $ [IEEE_INVALID,IEEE_DIVIDE_BY_ZERO,IEEE_OVERFLOW,
     $  IEEE_UNDERFLOW,IEEE_INEXACT]
      TYPE(IEEE_ROUND_TYPE) CURRENT_ROUNDING
      INTEGER I
      LOGICAL VALUE
      FLAGS=-1
      HALTING=-1
      ROUNDING=-1
      DO I=1,5
         IF(IEEE_SUPPORT_FLAG(KINDS(I))) THEN
            CALL IEEE_GET_FLAG(KINDS(I),VALUE)
            FLAGS(I)=MERGE(1,0,VALUE)
         ENDIF
         IF(IEEE_SUPPORT_HALTING(KINDS(I))) THEN
            CALL IEEE_GET_HALTING_MODE(KINDS(I),VALUE)
            HALTING(I)=MERGE(1,0,VALUE)
         ENDIF
      ENDDO
      IF(IEEE_SUPPORT_ROUNDING(IEEE_NEAREST)) THEN
         CALL IEEE_GET_ROUNDING_MODE(CURRENT_ROUNDING)
         ROUNDING=0
         IF(CURRENT_ROUNDING.EQ.IEEE_NEAREST) ROUNDING=1
         IF(CURRENT_ROUNDING.EQ.IEEE_TO_ZERO) ROUNDING=2
         IF(CURRENT_ROUNDING.EQ.IEEE_UP) ROUNDING=3
         IF(CURRENT_ROUNDING.EQ.IEEE_DOWN) ROUNDING=4
      ENDIF
      END
"""


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"sealed worker anchor is not unique: {old!r}")
    return text.replace(old, new, 1)


def instrument_worker(text: str) -> str:
    """Return the private worker; reject edited or already patched input."""
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual != BASE_WORKER_SHA256:
        raise ValueError(f"sealed profiling worker SHA256 mismatch: {actual}")
    text = _replace_once(text, "      CHARACTER*80 QP_NAME\n",
                         "      CHARACTER*80 QP_NAME\n" + DECLARATIONS)
    text = _replace_once(text, "         OPEN(76,FILE='profile-values.dat',STATUS='NEW')\n",
                         "         OPEN(76,FILE='profile-values.dat',STATUS='NEW')\n" + OPEN_FLAGS)
    text = _replace_once(text, ANCHOR, OBSERVED_ANCHOR)
    text = _replace_once(text, "         CLOSE(76)\n", "         CLOSE(76)\n         CLOSE(77)\n")
    return text + CAPTURE
