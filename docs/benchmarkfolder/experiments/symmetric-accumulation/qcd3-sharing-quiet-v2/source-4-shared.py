#!/usr/bin/env python3
"""Independent exact-product cache variant on the original scatter traversal."""
import json
from pathlib import Path

import gather

HERE=Path(__file__).resolve().parent


def groups(vertex='MP_FFV1L2_1', terminal=False, *, case='qcd-3',inventory_path=None):
    path=HERE/'inventory.json' if inventory_path is None else Path(inventory_path)
    data=json.loads(path.read_text())[case]
    shape=next(s for s in data['routines'][vertex]['shapes']
               if s['out_size']==4 and s['in_size']==4)
    # The terminal kernel has I=J. Different J therefore changes A even when K
    # is identical; its product groups must fix both J and K.
    return shape['identical_expression_same_jk_groups' if terminal
                 else 'identical_expression_same_k_groups']


def generate(generator,rank,vertex='MP_FFV1L2_1',terminal=False,suffix='_SHARED',
             *,case='qcd-3',inventory_path=None):
    assert generator.coef_format=='complex*32'
    groups_ = groups(vertex,terminal,case=case,inventory_path=inventory_path)
    assert groups_, 'No reusable products for this vertex/terminal shape'
    ids={}
    for index,group in enumerate(groups_,1):
        assert len({key[2] for key in group})==1
        if terminal:
            assert len({(key[0],key[2]) for key in group})==1
        for key in group:
            assert tuple(key) not in ids
            ids[tuple(key)]=index
    mapping=','.join(str(ids.get((j,m,k),0)) for k in range(1,5)
                     for m in range(5) for j in range(1,5))
    ncoef=gather.get_number_of_coefs_for_rank(rank)
    name=generator.sub_prefix+('TERMINAL' if terminal else 'UPDATE')+f'_WL_{rank}_1'
    extra=',WL_ENTRY_INEXACT' if terminal else ''
    source=gather.generate(generator,rank,terminal=terminal)
    source=source.replace('SUBROUTINE '+name+'_GATHER(',
                          'SUBROUTINE '+name+suffix+'(',1)
    assert len(name+suffix)<=63
    declarations=f"""  INTEGER GROUP_ID
  INTEGER, PARAMETER :: GROUP_MAP(4,0:4,4)=RESHAPE((/{mapping}/),(/4,5,4/))
  COMPLEX*32 PRODUCT_CACHE({len(groups_)},0:{ncoef-1},MAXLWFSIZE)
  LOGICAL PRODUCT_READY({len(groups_)},0:{ncoef-1},MAXLWFSIZE)
"""
    source=source.replace('  INTEGER I,J,K,L,M\n',
                          '  INTEGER I,J,K,L,M\n'+declarations,1)
    guard=f"""IF(IN_SIZE.NE.4.OR.OUT_SIZE.NE.4.OR.MAXLWFSIZE.NE.4) THEN
  CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT{extra})
  RETURN
ENDIF
"""
    for group in groups_:
        reference=','.join(map(str,group[0]))
        for entry in group[1:]:
            other=','.join(map(str,entry))
            guard+=f"""IF(ANY(TRANSFER(B({reference}),(/0_8,0_8,0_8,0_8/)).NE.TRANSFER(B({other}),(/0_8,0_8,0_8,0_8/)))) THEN
  CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT{extra})
  RETURN
ENDIF
"""
    original=generator.write_compact_wl_updater(rank,1,terminal=terminal)
    if not terminal:
        original=original.split('\n\nSUBROUTINE '+name+'_DENSE(',1)[0]
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
