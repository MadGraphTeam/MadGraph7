"""Strict pure comparison of complete native integer output to frozen reference."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
MARK=0x4B45000000000000
def environment_ok(token):
 if token<0 or token>>32!=MARK>>32:return False
 csr=(token>>16)&0xffff;cw=token&0xffff
 return not(csr&0x003f or cw&0xe0c0 or csr&0xe040 or cw&0x0c00) and cw&0x300==0x300

def parse(raw,expected,tokens):
 lines=raw.decode('ascii').splitlines();cursor=0
 def take():
  nonlocal cursor
  if cursor>=len(lines):raise ValueError('Truncated native output')
  words=lines[cursor].split();cursor+=1;return words
 def equal(wanted):
  got=take()
  if got!=list(map(str,wanted)):raise ValueError(('Output/reference mismatch',cursor,got,wanted))
 first=take()
 if len(first)!=3 or first[0]!='ENV_BEGIN' or first[2]!='T':raise ValueError('Host environment not qualified')
 token=int(first[1],16)
 if not environment_ok(token):raise ValueError('Invalid actual environment token')
 flags=[]
 for e in expected:
  i=e['id'];equal(['C',i,int(e['route_ok']),int(e['producer_ok'])])
  for j,ch in enumerate(e['chords'],1):equal(['B',i,j,int(ch['have']),*ch['errors']])
  for p in e['pairs']:equal(['P',i,p['i'],p['j'],p['class'],int(p['have']),*p['errors']])
  got=take()
  if len(got)!=6 or got[:4]!=['E',str(i),'T','T'] or got[4]!=got[5] or len(got[4])!=5 or set(got[4])-set('FT'):raise ValueError(('IEEE flags/environment changed',got))
  flags.append(got[4])
 for i,t in enumerate(tokens,1):equal(['T',i,'T' if t['expected'] else 'F'])
 equal(['ENV_END',f'{token:016X}','T','T']);equal(['DONE',len(expected),len(tokens)])
 if cursor!=len(lines):raise ValueError('Extra native output')
 return {'passed':True,'cases':len(expected),'base_route_modes':104,'token_cases':len(tokens),'actual_environment_token':f'{token:016X}',
  'actual_environment_read_only_ready':True,'reference_mismatches':0,'observed_ieee_flag_changes':0,'observed_flag_patterns':sorted(set(flags)),
  'tensor_calls':0,'amplitude_qualified':False,'kernel_accuracy_requalified':False,'scope':'Compiled integer source-DAG bound factory and read-only current-host environment probe only.'}

def synthetic(expected,tokens):
 token=MARK|(0x1f80<<16)|0x033f;lines=[f'ENV_BEGIN {token:016X} T']
 for e in expected:
  i=e['id'];lines.append(f'C {i} {int(e["route_ok"])} {int(e["producer_ok"])}')
  for j,c in enumerate(e['chords'],1):lines.append(' '.join(map(str,['B',i,j,int(c['have']),*c['errors']])))
  for p in e['pairs']:lines.append(' '.join(map(str,['P',i,p['i'],p['j'],p['class'],int(p['have']),*p['errors']])))
  lines.append(f'E {i} T T FFFFT FFFFT')
 lines += [f'T {i} '+('T' if t['expected'] else 'F') for i,t in enumerate(tokens,1)]
 lines += [f'ENV_END {token:016X} T T',f'DONE {len(expected)} {len(tokens)}']
 return ('\n'.join(lines)+'\n').encode()
