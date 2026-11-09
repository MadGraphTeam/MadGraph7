#!/usr/bin/env python3
"""Replace seven automatic adapter arrays by local contiguous rank-remapped scratch.

This is an isolated experiment, not an edit to the production template.  The
bridge v1 limit n <= 6 bounds storage.  No numerical expression is changed.
"""
import argparse
from pathlib import Path


def transform(source):
    for real, complex_, prefix in ((8,16,''),(16,32,'MP_')):
        routine=f'      SUBROUTINE ML5_0_{prefix}TIRUST_TENSOR_DIAGNOSTIC('
        if source.count(routine)!=1:
            raise ValueError('Expected one '+routine.strip())
        source=source.replace(routine,routine.replace('SUBROUTINE','RECURSIVE SUBROUTINE'),1)
        old=(f'      REAL*{real} P_TMP(0:3,0:NLOOPLINE-1),ABSP(0:3),REF_P,PREFIX_SUM\n'
             f'      REAL*{real} SHIFTS(0:3,NLOOPLINE),P_S_MAT(NLOOPLINE,0:3)\n'
             f'      REAL*{real} PDEN_DUMMY(0:3,NLOOPLINE-1),INVARIANTS(NLOOPLINE\n'
             f'     $ ,NLOOPLINE),SCALE\n'
             f'      COMPLEX*{complex_} MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)\n')
        new=(f'      REAL*{real} ABSP(0:3),REF_P,PREFIX_SUM,SCALE\n'
             f'      REAL*{real}, TARGET :: REAL_WORK(128)\n'
             f'      COMPLEX*{complex_}, TARGET :: COMPLEX_WORK(42)\n'
             f'      REAL*{real}, POINTER, CONTIGUOUS :: P_TMP(:,:),SHIFTS(:,:)\n'
             f'      REAL*{real}, POINTER, CONTIGUOUS :: P_S_MAT(:,:)\n'
             f'      REAL*{real}, POINTER, CONTIGUOUS :: PDEN_DUMMY(:,:)\n'
             f'      REAL*{real}, POINTER, CONTIGUOUS :: INVARIANTS(:,:)\n'
             f'      COMPLEX*{complex_}, POINTER, CONTIGUOUS :: MASSES(:),S_MAT(:,:)\n')
        if source.count(old)!=1:
            raise ValueError('Expected original '+str(real)+'-byte declarations exactly once')
        source=source.replace(old,new,1)
    boundary=('      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE))\n'
              '     $  RETURN\n')
    if source.count(boundary)!=2:
        raise ValueError('Expected two support guards')
    associations=('C     Exact leading dimensions, local storage; bridge v1 supports N<=6.\n'
                  '      IF(NLOOPLINE.LT.1.OR.NLOOPLINE.GT.6) RETURN\n'
                  '      P_TMP(0:3,0:NLOOPLINE-1)=>REAL_WORK(1:4*NLOOPLINE)\n'
                  '      SHIFTS(0:3,1:NLOOPLINE)=>REAL_WORK(25:24+4*NLOOPLINE)\n'
                  '      P_S_MAT(1:NLOOPLINE,0:3)=>REAL_WORK(49:48+4*NLOOPLINE)\n'
                  '      PDEN_DUMMY(0:3,1:NLOOPLINE-1)=>\n'
                  '     $ REAL_WORK(73:72+4*(NLOOPLINE-1))\n'
                  '      INVARIANTS(1:NLOOPLINE,1:NLOOPLINE)=>\n'
                  '     $ REAL_WORK(93:92+NLOOPLINE*NLOOPLINE)\n'
                  '      MASSES(1:NLOOPLINE)=>COMPLEX_WORK(1:NLOOPLINE)\n'
                  '      S_MAT(1:NLOOPLINE,1:NLOOPLINE)=>\n'
                  '     $ COMPLEX_WORK(7:6+NLOOPLINE*NLOOPLINE)\n')
    return source.replace(boundary,boundary+associations)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    args.destination.write_text(transform(args.source.read_text()))
