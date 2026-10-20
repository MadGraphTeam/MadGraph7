#!/usr/bin/env python3
"""Replace only scale fixtures with exact admitted binary64 endpoints."""
import json


def specialize(source, vertex, driver):
    proof=json.loads(driver.PROOF.read_text())['vertices'][vertex]
    source=source.replace('saved_b(2*4*MAXB*4),one','saved_b(2*4*MAXB*4),one,endpoint_bits')
    source=source.replace('real*8 :: MASS','real*8 :: MASS,endpoint')
    start=source.index('  if(trial==4.or.trial==5) then')
    stop=source.index('  entry=trial/=8',start)
    rows=[]
    for m,entries in enumerate(proof['q_rows'],1):
        if m==1:
            continue
        for j,entry in enumerate(entries,1):
            for k in range(1,5):
                rows.append(f'    B({j},{m},{k})='+driver.rotate(f"B({entry['source_row']},1,{k})",entry['phase']))
    h=[]
    for j in range(1,5):
        for k in range(1,5):
            if f'{j},{k}' in proof['h_source_expressions']:
                h.append(f'    B({j},1,{k})=cmplx(endpoint*(-1.0_8)**{j+k},endpoint*(-1.0_8)**{k},8)')
    replacement=f"""  if(trial>=4.and.trial<=6) then
    n=merge(1223,823,trial==4)
    endpoint_bits=ior(ishft(int(n,8),52),int(z'000FFFFFFFFFFFFF',8))
    endpoint=transfer(endpoint_bits,endpoint)
    A(:,0:NC-1,:)=cmplx(endpoint,-endpoint,8)
    B(:,0:4,:)=cmplx(0.0_8,0.0_8,8)
    B(:,0,:)=cmplx(-endpoint,endpoint,8)
{chr(10).join(h)}
{chr(10).join(rows)}
    if(trial==6) then
      do l=0,NC-1
        n=merge(1223,823,mod(l,2)==0)
        endpoint_bits=ishft(int(n,8),52)
        endpoint=transfer(endpoint_bits,endpoint)
        A(:,l,:)=cmplx(endpoint,-endpoint,8)
      enddo
    endif
  endif
"""
    return source[:start]+replacement+source[stop:]
