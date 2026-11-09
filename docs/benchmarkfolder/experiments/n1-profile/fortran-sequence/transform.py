#!/usr/bin/env python3
"""Fixed local buffers through explicit-shape dummy sequence association.

This isolated alternative avoids the pointer descriptors of fortran-scratch.
The actual arithmetic and runtime leading dimensions remain in the worker.
"""
import argparse
from pathlib import Path
import re


def transform(source):
    for real, complex_, prefix in ((8,16,''),(16,32,'MP_')):
        public='ML5_0_'+prefix+'TIRUST_TENSOR_DIAGNOSTIC'
        worker='ML5_0_'+prefix+'TIRUST_DIAGNOSTIC_WORK'
        pattern=r'      SUBROUTINE '+public+r'\([^\n]*\n(?:     \$[^\n]*\n)+'
        matches=list(re.finditer(pattern,source))
        if len(matches)!=1:
            raise ValueError('Expected one diagnostic header: '+public)
        match=matches[0]
        wrapper=(f'      RECURSIVE SUBROUTINE {public}(\n'
                 '     $ CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)\n'
                 '      IMPLICIT NONE\n'
                 "      INCLUDE 'loop_max_coefs.inc'\n"
                 '      INTEGER CTMODE,NLOOPLINE,RANK\n'
                 f'      REAL*{real} PL(0:3,NLOOPLINE),REAL_WORK(128)\n'
                 '      COMPLEX*16 M2L(NLOOPLINE)\n'
                 f'      COMPLEX*{complex_} TIRCOEFS(0:LOOPMAXCOEFS-1,3)\n'
                 f'      COMPLEX*{complex_} COMPLEX_WORK(42)\n'
                 '      LOGICAL STABLE\n'
                 'C     Guard before sequence association into the six-propagator storage.\n'
                 '      IF(NLOOPLINE.LT.1.OR.NLOOPLINE.GT.6) THEN\n'
                 '        STABLE=.FALSE.\n'
                 '        TIRCOEFS=0\n'
                 '        RETURN\n'
                 '      ENDIF\n'
                 f'      CALL {worker}(\n'
                 '     $ CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE,\n'
                 '     $ REAL_WORK(1),REAL_WORK(25),REAL_WORK(49),REAL_WORK(73),\n'
                 '     $ REAL_WORK(93),COMPLEX_WORK(1),COMPLEX_WORK(7))\n'
                 '      END\n\n'
                 f'      RECURSIVE SUBROUTINE {worker}(\n'
                 '     $ CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE,\n'
                 '     $ P_TMP,SHIFTS,P_S_MAT,PDEN_DUMMY,INVARIANTS,MASSES,S_MAT)\n')
        source=source[:match.start()]+wrapper+source[match.end():]
    return source


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    args.destination.write_text(transform(args.source.read_text()))
