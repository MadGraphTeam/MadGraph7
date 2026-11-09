#!/usr/bin/env python3
"""Isolate omission of a DP raw scan already guaranteed by bridge ABI v1.

The status guard, mapping arithmetic, QP checks, independent host stability
checks and final contraction checks remain unchanged.  No production edit.
"""
import argparse
from pathlib import Path


def transform(source):
    scan=('        DO J=1,4\n'
          '          IF(J.EQ.4.AND..NOT.TIRUSTCOMPUTEFINITE) CYCLE\n'
          '          IF(.NOT.IEEE_IS_FINITE(REAL(RAW(K,J),KIND=KI_TIR))\n'
          '     $     .OR..NOT.IEEE_IS_FINITE(AIMAG(RAW(K,J)))) RETURN\n'
          '        ENDDO\n')
    if source.count(scan)!=1:
        raise ValueError('Expected exactly one DP-only raw finiteness scan')
    return source.replace(scan,
        'C       ABI v1 dp_checked validates every packed DP component before\n'
        'C       SUCCESS. The status guard above and host checks remain.\n',1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    args.destination.write_text(transform(args.source.read_text()))
