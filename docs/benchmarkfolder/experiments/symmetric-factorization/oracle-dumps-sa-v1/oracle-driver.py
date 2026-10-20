#!/usr/bin/env python3
"""Generate a bit-dumping driver for the independent exact contraction oracle."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import re
import textwrap

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
PROOF = HERE.parent / "analysis.json"
PROOF_SHA256 = "f5bc8c65680a028e79103f788fefc2356962350a9a7cffd8ffacbfad49510f42"


def logical(path):
    rows = []
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw[0] in "cC*!":
            continue
        body = re.sub(r"\s+", "", raw[6:]).upper()
        if raw[5] not in " 0":
            rows[-1] += body
        else:
            rows.append(body)
    return rows


def wrap(statement):
    tokens = re.findall(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[QDE][+-]?\d+)?"
                        r"|[A-Za-z]\w*|\*\*|[^\s]", statement)
    return " &\n & ".join(textwrap.wrap(" ".join(tokens), width=100,
                                       break_long_words=False, break_on_hyphens=False))


def rotate(value, phase):
    if phase == [1, 0]:
        return value
    if phase == [-1, 0]:
        return f"(-{value})"
    if phase == [0, 1]:
        return f"CMPLX(-AIMAG({value}),REAL({value},16),16)"
    if phase == [0, -1]:
        return f"CMPLX(AIMAG({value}),-REAL({value},16),16)"
    raise ValueError(phase)


def driver(rank, vertex, terminal=False):
    assert 2 <= rank <= 4
    assert hashlib.sha256(PROOF.read_bytes()).hexdigest() == PROOF_SHA256
    proof = json.loads(PROOF.read_text())["cases"]["qcd-4"]["vertex_proofs"][vertex]
    assert proof["passed"]
    source = ROOT / proof["source"]
    # The separately retained source snapshot can replace disposable exports.
    candidates = list((HERE.parent / "proof-inputs").rglob(vertex+".f"))
    if not source.exists() and candidates:
        source = next(p for p in candidates
                      if hashlib.sha256(p.read_bytes()).hexdigest() == proof["source_sha256"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == proof["source_sha256"]
    assignments = []
    for line in logical(source):
        match = re.fullmatch(r"COEFF\((\d+),(\d+),(\d+)\)=(.*)", line)
        if not match:
            continue
        j, m, k = map(int, match.groups()[:3])
        rhs = match[4]
        rhs = re.sub(r"\w+%W\((\d+)\)", r"WV(\1)", rhs)
        rhs = re.sub(r"\bP[123]\((\d+)\)", r"PMOM(\1)", rhs)
        rhs = re.sub(r"\bM[123]\b", "MASS", rhs)
        assert "%" not in rhs
        assignments.append(wrap(f"B({j},{m},{k})={rhs}"))
    assert len(assignments) == 80
    transformed = []
    for m, rows in enumerate(proof["q_rows"], 1):
        if m == 1:
            continue
        for j, row in enumerate(rows, 1):
            for k in range(1, 5):
                transformed.append(wrap(f"B({j},{m},{k})="+
                    rotate(f"B({row['source_row']},1,{k})", row["phase"])))
    active_h = [(j, k) for j in range(1, 5) for k in range(1, 5)
                if f"{j},{k}" in proof["h_source_expressions"]]
    h_zero_fixtures = "\n".join(
        f"""if(trial==9) then
  B({j},1,{k})=cmplx(real({2*j+k},16)/8,0.0_16,16)
else
  B({j},1,{k})=cmplx(0.0_16,real({2*j-k},16)/8,16)
endif
call alter_zero_sign(B({j},1,{k}),mod({j+k},2)==0)"""
        for j, k in active_h)
    cancellation_ks = [k for j, k in active_h if j == 1]
    if len(cancellation_ks) >= 2:
        k1, k2 = cancellation_ks[:2]
        cancel = f"""A(:,0:NC-1,:)=cmplx(0.0_16,0.0_16,16)
do i=1,4
  do l=0,NC-1
    A({k1},l,i)=B(1,1,{k2})*scale(1.0_16,mod(l+i,7)-3)
    A({k2},l,i)=-B(1,1,{k1})*scale(1.0_16,mod(l+i,7)-3)
    A({k2},l,i)=A({k2},l,i)+cmplx(scale(real(mod(l,3)+1,16),-108),0.0_16,16)
  enddo
enddo"""
    else:
        # Scalar vertices have one term per H row, so cancellation is instead
        # exercised between shifted coefficients by alternating equal values.
        cancel = """do i=1,4
  do l=0,NC-1
    do k=1,4
      A(k,l,i)=cmplx((-1.0_16)**mod(l+k,2), &
        (-1.0_16)**mod(l+i,2),16)*(1.0_16+scale(1.0_16,-108))
    enddo
  enddo
enddo"""
    old = "TEST_"+("TERMINAL" if terminal else "UPDATE")+f"_WL_{rank}_1"
    old_args = ",entry" if terminal else ""
    text = f"""program exact_oracle_dump
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
integer, parameter :: NC={comb(rank+4,4)}, NO={comb(rank+5,4)}, MAXC=210, MAXB=15
complex*32 :: A(4,0:MAXC-1,4), B(4,0:MAXB-1,4)
complex*32 :: oldout(4,0:MAXC-1,4), newout(4,0:MAXC-1,4)
complex*32 :: WV(4), PMOM(0:3), COUP,COUP1,COUP2,CI
real*16 :: MASS
integer*8 :: saved_a(4*4*MAXC*4),saved_b(4*4*MAXB*4),one(2)
integer :: trial,i,j,k,l,m,n,hi,fast,fallback,diag
common/FACTOR_TRACE/fast,fallback
logical :: entry,oldflags(size(ieee_all)),newflags(size(ieee_all)),samea,sameb
if(storage_size(1.0_16)/=128.or.digits(1.0_16)/=113.or.maxexponent(1.0_16)/=16384) stop 1
one=transfer(1.0_16,one)
hi=merge(1,2,one(1)/=0_8)
if(one(hi)/=ishft(16383_8,48).or.one(3-hi)/=0_8) stop 2
call ieee_set_rounding_mode(ieee_nearest)
do i=1,size(ieee_all)
  if(.not.ieee_support_halting(ieee_all(i))) stop 3
  call ieee_set_halting_mode(ieee_all(i),.false.)
enddo
write(*,'(A)') 'FORMAT BINARY128_HEX_V1'
write(*,'(A,4(1X,I0))') 'LAYOUT',128,113,16384,hi
do trial=1,10
  CI=cmplx(0.0_16,1.0_16,16)
  MASS=1.5_16
  COUP=cmplx(1.125_16+scale(1.0_16,-108),-0.1875_16,16)
  COUP1=cmplx(0.875_16,-0.3125_16+scale(1.0_16,-109),16)
  COUP2=cmplx(-0.625_16,0.4375_16,16)
  WV(1)=cmplx(1.0_16+scale(1.0_16,-109),2.0_16,16)
  WV(2)=cmplx(-0.75_16,0.625_16,16)
  WV(3)=cmplx(0.4375_16,-0.28125_16,16)
  WV(4)=cmplx(1.375_16,0.8125_16,16)
  PMOM(0)=cmplx(1.625_16,0.0625_16,16)
  PMOM(1)=cmplx(-0.875_16,0.03125_16,16)
  PMOM(2)=cmplx(0.3125_16,-0.125_16,16)
  PMOM(3)=cmplx(0.5625_16,0.1875_16,16)
  A=cmplx(9.125_16,-0.25_16,16)
  B=cmplx(-17.5_16,0.25_16,16)
{chr(10).join(assignments)}
  if(trial==9.or.trial==10) then
{h_zero_fixtures}
  endif
  if(trial/=7) then
{chr(10).join(transformed)}
  endif
  if(trial==9.or.trial==10) then
    do j=1,4
      do m=1,4
        do k=1,4
          call alter_zero_sign(B(j,m,k),mod(j+m+k,2)==0)
        enddo
      enddo
    enddo
  endif
  do i=1,4
    do l=0,NC-1
      do k=1,4
        A(k,l,i)=cmplx(real(mod(17*k+11*l+7*i,31)-15,16)/16, &
                      real(mod(13*k+5*l+19*i,29)-14,16)/32,16)
        A(k,l,i)=A(k,l,i)*(1.0_16+scale(1.0_16,-109))
        if(trial==2.and.mod(l+k+i,3)/=0) A(k,l,i)=cmplx(0.0_16,-0.0_16,16)
      enddo
    enddo
  enddo
  if(trial==3) then
{cancel}
  endif
  if(trial==4.or.trial==5) then
    n=merge(900,-900,trial==4)
    A(:,0:NC-1,:)=A(:,0:NC-1,:)*scale(1.0_16,n)
    B(:,0:4,:)=B(:,0:4,:)*scale(1.0_16,n)
  endif
  if(trial==6) then
    do l=0,NC-1
      n=merge(900,-900,mod(l,2)==0)
      A(:,l,:)=A(:,l,:)*scale(1.0_16,n)
    enddo
  endif
  entry=trial/=8
  diag=merge(1,0,{'.true.' if terminal else '.false.'}.and.entry)
  write(*,'(A,7(1X,I0))') 'CASE',trial,{rank},{1 if terminal else 0},4,4,4,diag
  do i=1,4
    do l=0,NC-1
      do k=1,4
        call dump_complex('A',k,l,i,A(k,l,i))
      enddo
    enddo
  enddo
  do k=1,4
    do m=0,4
      do j=1,4
        call dump_complex('B',j,m,k,B(j,m,k))
      enddo
    enddo
  enddo
  saved_a=transfer(A,saved_a)
  saved_b=transfer(B,saved_b)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,entry)
  call {old}(A,4,B,4,4,oldout{old_args})
  call ieee_get_flag(ieee_all,oldflags)
  samea=all(saved_a==transfer(A,saved_a))
  sameb=all(saved_b==transfer(B,saved_b))
  write(*,'(A,2(1X,I0))') 'IMMUTABLE OLD',merge(1,0,samea),merge(1,0,sameb)
  if(.not.samea.or..not.sameb) stop 4
  fast=0
  fallback=0
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,entry)
  call {old}_FACTOR(A,4,B,4,4,newout,entry)
  call ieee_get_flag(ieee_all,newflags)
  samea=all(saved_a==transfer(A,saved_a))
  sameb=all(saved_b==transfer(B,saved_b))
  write(*,'(A,2(1X,I0))') 'IMMUTABLE NEW',merge(1,0,samea),merge(1,0,sameb)
  if(.not.samea.or..not.sameb) stop 5
  write(*,'(A,99(1X,I0))') 'FLAGS OLD',(merge(1,0,oldflags(i)),i=1,size(oldflags))
  write(*,'(A,99(1X,I0))') 'FLAGS NEW',(merge(1,0,newflags(i)),i=1,size(newflags))
  write(*,'(A,1X,I0)') 'TAIL_ZERO OLD',merge(1,0,all(oldout(:,NO:MAXC-1,:)==0))
  write(*,'(A,1X,I0)') 'TAIL_ZERO NEW',merge(1,0,all(newout(:,NO:MAXC-1,:)==0))
  write(*,'(A,2(1X,I0))') 'TRACE',fast,fallback
  do i=1,4
    do n=0,NO-1
      do j=1,4
        call dump_complex('OLD',j,n,i,oldout(j,n,i))
        call dump_complex('NEW',j,n,i,newout(j,n,i))
      enddo
    enddo
  enddo
  write(*,'(A)') 'ENDCASE'
enddo
write(*,'(A)') 'PASS_DUMP'
contains
subroutine dump_complex(tag,x,y,z,value)
  character(*),intent(in)::tag
  integer,intent(in)::x,y,z
  complex*32,intent(in)::value
  integer*8::re(2),im(2)
  re=transfer(real(value,16),re)
  im=transfer(aimag(value),im)
  write(*,'(A,3(1X,I0),4(1X,Z16.16))') tag,x,y,z,re(hi),re(3-hi),im(hi),im(3-hi)
end subroutine
subroutine alter_zero_sign(value,negative)
  complex*32,intent(inout)::value
  logical,intent(in)::negative
  integer*8::re(2),im(2)
  re=transfer(real(value,16),re)
  im=transfer(aimag(value),im)
  if(real(value,16)==0) then
    re=0_8
    if(negative) re(hi)=ibset(re(hi),63)
  endif
  if(aimag(value)==0) then
    im=0_8
    if(.not.negative) im(hi)=ibset(im(hi),63)
  endif
  value=cmplx(transfer(re,0.0_16),transfer(im,0.0_16),16)
end subroutine
end program
"""
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--vertex", required=True)
    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refuse to overwrite generated oracle source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(driver(args.rank, args.vertex, args.terminal))
