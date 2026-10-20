#!/usr/bin/env python3
"""Independent exact-product cache variant on the original scatter traversal."""
import json
from pathlib import Path

import gather

HERE=Path(__file__).resolve().parent


def groups():
    data=json.loads((HERE/'inventory.json').read_text())['qcd-3']
    shape=next(s for s in data['routines']['MP_FFV1L2_1']['shapes']
               if s['out_size']==4 and s['in_size']==4)
    return shape['identical_expression_same_k_groups']


def generate(generator,rank):
    assert generator.coef_format=='complex*32'
    groups_ = groups()
    ids={}
    for index,group in enumerate(groups_,1):
        assert len({key[2] for key in group})==1
        for key in group:
            assert tuple(key) not in ids
            ids[tuple(key)]=index
    mapping=','.join(str(ids.get((j,m,k),0)) for k in range(1,5)
                     for m in range(5) for j in range(1,5))
    ncoef=gather.get_number_of_coefs_for_rank(rank)
    name=generator.sub_prefix+f'UPDATE_WL_{rank}_1'
    source=gather.generate(generator,rank)
    source=source.replace('SUBROUTINE '+name+'_GATHER(',
                          'SUBROUTINE '+name+'_SHARED(',1)
    declarations=f"""  INTEGER GROUP_ID
  INTEGER, PARAMETER :: GROUP_MAP(4,0:4,4)=RESHAPE((/{mapping}/),(/4,5,4/))
  COMPLEX*32 PRODUCT_CACHE({len(groups_)},0:{ncoef-1},MAXLWFSIZE)
  LOGICAL PRODUCT_READY({len(groups_)},0:{ncoef-1},MAXLWFSIZE)
"""
    source=source.replace('  INTEGER I,J,K,L,M\n',
                          '  INTEGER I,J,K,L,M\n'+declarations,1)
    guard=f"""IF(IN_SIZE.NE.4.OR.OUT_SIZE.NE.4.OR.MAXLWFSIZE.NE.4) THEN
  CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
  RETURN
ENDIF
"""
    for group in groups_:
        reference=','.join(map(str,group[0]))
        for entry in group[1:]:
            other=','.join(map(str,entry))
            guard+=f"""IF(ANY(TRANSFER(B({reference}),(/0_8,0_8,0_8,0_8/)).NE.TRANSFER(B({other}),(/0_8,0_8,0_8,0_8/)))) THEN
  CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
  RETURN
ENDIF
"""
    original=generator.write_compact_wl_updater(rank,1).split('\n\nSUBROUTINE '+name+'_DENSE(',1)[0]
    begin=original.index('OUT(:,:,:)=',original.index('C Welcome'))
    body=original[begin:original.index('\nCONTAINS',begin)]
    body=body.replace('UPDATER_COEF = B(J,M,K)',
                      'UPDATER_COEF = B(J,M,K)\n          GROUP_ID=GROUP_MAP(J,M,K)',1)
    old='OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + A(K,L,I)*UPDATER_COEF'
    assert body.count(old)==1
    replacement="""IF(GROUP_ID.EQ.0) THEN
                OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + A(K,L,I)*UPDATER_COEF
              ELSE
                IF(.NOT.PRODUCT_READY(GROUP_ID,L,I)) THEN
                  PRODUCT_CACHE(GROUP_ID,L,I)=A(K,L,I)*UPDATER_COEF
                  PRODUCT_READY(GROUP_ID,L,I)=.TRUE.
                ENDIF
                OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + PRODUCT_CACHE(GROUP_ID,L,I)
              ENDIF"""
    body=body.replace(old,replacement)
    start=source.index('OUT(:,:,:)=',source.index('C Experimental'))
    end=source.index('\nCONTAINS',start)
    return source[:start]+guard+'PRODUCT_READY=.FALSE.\n'+body+source[end:]
