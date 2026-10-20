#!/usr/bin/env python3
"""Isolated compact rank-one gather generator; never patches production."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0,str(ROOT))
from madgraph.various.q_polynomial import Polynomial, get_number_of_coefs_for_rank


def predecessors(rank):
    """P[m,n] is the unique L with monomial(L)*monomial(m)==monomial(n)."""
    source, output = Polynomial(rank), Polynomial(rank+1)
    result = []
    for n in range(get_number_of_coefs_for_rank(rank+1)):
        monomial = output.get_coef_at_position(n)
        row = []
        for m in range(5):
            predecessor = list(monomial)
            if m:
                if m-1 not in predecessor:
                    row.append(-1)
                    continue
                predecessor.remove(m-1)
            row.append(source.get_coef_position(predecessor) if len(predecessor)<=rank else -1)
        result.append(row)
    # Every existing scatter operation occurs once, with no invented zero terms.
    actual = sorted((l,m,n) for n,row in enumerate(result) for m,l in enumerate(row) if l>=0)
    expected = sorted((l,m,output.get_coef_position(source.get_coef_at_position(l)+
                        ([] if m==0 else [m-1])))
                      for l in range(get_number_of_coefs_for_rank(rank)) for m in range(5))
    assert actual == expected
    return result


def generate(generator, rank, terminal=False):
    """Keep full preparation/fallback and change only the compact traversal."""
    assert rank>=2
    source = generator.write_compact_wl_updater(rank,1,terminal=terminal)
    name = generator.sub_prefix+('TERMINAL' if terminal else 'UPDATE')+f'_WL_{rank}_1'
    # The public generator also emits a dense fallback. The caller supplies the
    # unmodified public routines separately, so keep only the prototype entry.
    if not terminal:
        source = source.split('\n\nSUBROUTINE '+name+'_DENSE(',1)[0]
    source = source.replace('SUBROUTINE '+name+'(','SUBROUTINE '+name+'_GATHER(',1)
    if 'USE, INTRINSIC :: IEEE_EXCEPTIONS' not in source:
        source = source.replace('  implicit none',
            '  USE, INTRINSIC :: IEEE_EXCEPTIONS\n  implicit none',1)
    rows = predecessors(rank)
    values = ','.join(str(l) for row in rows for l in row)
    declarations = f"""  INTEGER N
  INTEGER, PARAMETER :: PREDECESSOR(0:4,0:{len(rows)-1})=RESHAPE((/{values}/),(/5,{len(rows)}/))
  LOGICAL ACTIVE(0:4,MAXLWFSIZE),SKIP_A(0:4,MAXLWFSIZE)
  LOGICAL HALT_FLAGS(SIZE(IEEE_ALL))
  {generator.coef_format} ACC
"""
    source = source.replace('  INTEGER I,J,K,L,M\n','  INTEGER I,J,K,L,M\n'+declarations,1)
    # Reordering independent output operations is restricted to disabled traps.
    # All operations for one output preserve the original M,K order. Exceptional
    # QP values already use the unmodified dense fallback in its preparation.
    extra = ',WL_ENTRY_INEXACT' if terminal else ''
    kind = 16 if generator.coef_format=='complex*32' else 8
    na = get_number_of_coefs_for_rank(rank)-1
    trap_guard = f"""CALL IEEE_GET_HALTING_MODE(IEEE_ALL,HALT_FLAGS)
IF(ANY(HALT_FLAGS)) THEN
  CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT{extra})
  RETURN
ENDIF
IF(.NOT.ALL(GATHER_BOUNDED(REAL(A(1:IN_SIZE,0:{na},1:LCUT_SIZE),KIND={kind}))).OR..NOT.ALL(GATHER_BOUNDED(AIMAG(A(1:IN_SIZE,0:{na},1:LCUT_SIZE)))).OR..NOT.ALL(GATHER_BOUNDED(REAL(B(1:OUT_SIZE,0:4,1:IN_SIZE),KIND={kind}))).OR..NOT.ALL(GATHER_BOUNDED(AIMAG(B(1:OUT_SIZE,0:4,1:IN_SIZE))))) THEN
  CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT{extra})
  RETURN
ENDIF
"""
    source = source.replace('C Welcome to the computational heart of MadLoop...',
                            'C Experimental coefficient-major gather\n'+trap_guard,1)
    start = source.index('OUT(:,:,:)=',source.index('C Experimental'))
    end = source.index('\nCONTAINS',start) if '\nCONTAINS' in source[start:] else source.rindex('\nEND')
    sparse = generator.coef_format=='complex*32'
    zero_mask = ('SKIP_A(M,K)=WL_NEAREST\n' if sparse else 'SKIP_A(M,K)=.FALSE.\n')
    zero_guard = '          IF(SKIP_A(M,K).AND.WL_AZERO(K,L,I)) CYCLE\n' if sparse else ''
    iloop = 'DO I=J,MIN(J,LCUT_SIZE)' if terminal else 'DO I=1,LCUT_SIZE'
    body = f"""OUT(:,:,:)={generator.czero}
DO J=1,OUT_SIZE
  DO M=0,4
    DO K=1,IN_SIZE
      ACTIVE(M,K)=B(J,M,K).NE.{generator.czero}
      {zero_mask}
    ENDDO
  ENDDO
  {iloop}
    DO N=0,{len(rows)-1}
      ACC={generator.czero}
      DO M=0,4
        L=PREDECESSOR(M,N)
        IF(L.LT.0) CYCLE
        DO K=1,IN_SIZE
          IF(.NOT.ACTIVE(M,K)) CYCLE
{zero_guard}          ACC=ACC+A(K,L,I)*B(J,M,K)
        ENDDO
      ENDDO
      OUT(J,N,I)=ACC
    ENDDO
  ENDDO
ENDDO
"""
    result = source[:start]+body+source[end:]
    if kind==16:
        helper = """ELEMENTAL LOGICAL FUNCTION GATHER_BOUNDED(X)
  REAL*16, INTENT(IN) :: X
  INTEGER*8, PARAMETER :: ONE(2)=TRANSFER(1.0_16,(/0_8,0_8/))
  INTEGER, PARAMETER :: HI=MERGE(1,2,ONE(1).NE.0_8)
  GATHER_BOUNDED=.FALSE.
  IF(STORAGE_SIZE(X).NE.128.OR.DIGITS(X).NE.113.OR.RADIX(X).NE.2.OR.MAXEXPONENT(X).NE.16384) RETURN
  IF(ONE(HI).NE.ISHFT(16383_8,48).OR.ONE(3-HI).NE.0_8) RETURN
  ASSOCIATE(WORDS=>TRANSFER(X,ONE))
    GATHER_BOUNDED=IBITS(WORDS(HI),48,15).GE.15383_8.AND.IBITS(WORDS(HI),48,15).LE.17383_8
    IF(IBCLR(WORDS(HI),63).EQ.0_8.AND.WORDS(3-HI).EQ.0_8) GATHER_BOUNDED=.TRUE.
  END ASSOCIATE
END FUNCTION
"""
    else:
        helper = """ELEMENTAL LOGICAL FUNCTION GATHER_BOUNDED(X)
  REAL*8, INTENT(IN) :: X
  INTEGER*8, PARAMETER :: ONE(1)=TRANSFER(1.0_8,(/0_8/))
  GATHER_BOUNDED=.FALSE.
  IF(STORAGE_SIZE(X).NE.64.OR.DIGITS(X).NE.53.OR.RADIX(X).NE.2.OR.MAXEXPONENT(X).NE.1024) RETURN
  IF(ONE(1).NE.ISHFT(1023_8,52)) RETURN
  ASSOCIATE(WORDS=>TRANSFER(X,ONE))
    GATHER_BOUNDED=IBITS(WORDS(1),52,11).GE.923_8.AND.IBITS(WORDS(1),52,11).LE.1123_8
    IF(IBCLR(WORDS(1),63).EQ.0_8) GATHER_BOUNDED=.TRUE.
  END ASSOCIATE
END FUNCTION
"""
    last = result.rindex('\nEND')
    return result[:last]+'\n'+('CONTAINS\n' if kind==8 else '')+helper+result[last:]
