#!/usr/bin/env python3
"""Generate an equivalent lower-overhead bit scan as a separate experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / 'docs/benchmarkfolder/experiments/dp-terminal'
DEST = BASE / 'packed-scan'
DEST.mkdir(exist_ok=True)
(ROOT / '.tirust-build/dp-terminal/packed-scan').mkdir(exist_ok=True)
source = (BASE / 'terminal.f90').read_text()
source = source.replace('integer :: LCUT_SIZE, IN_SIZE, OUT_SIZE, I, J, K, H', '''integer :: LCUT_SIZE, IN_SIZE, OUT_SIZE, I, J, K, H
integer(int64) :: WORDS_A(160), WORDS_B(160), WORD
integer(int64), parameter :: ONE_WORD=transfer(1.0_8,0_int64)
integer(int64), parameter :: MIN_WORD=shiftl(623_int64,52), MAX_WORD=shiftl(1424_int64,52)''')
begin = source.index('! Integer classification')
end = source.index('FAST=.true.', begin)
source = source[:begin] + '''! Transfer bits in bulk; no arithmetic on signaling or exceptional values.
if (storage_size(1.0_8)/=64.or.digits(1.0_8)/=53.or.radix(1.0_8)/=2.or.maxexponent(1.0_8)/=1024) goto 900
if (storage_size(A(1,0,1))/=128.or.ONE_WORD/=shiftl(1023_int64,52)) goto 900
WORDS_A=transfer(A(:,0:4,:),WORDS_A)
WORDS_B=transfer(B,WORDS_B)
do H=1,160
  WORD=ibclr(WORDS_A(H),63)
  if (WORD/=0_int64) then
    if (WORD<MIN_WORD.or.WORD>=MAX_WORD) goto 900
  endif
  WORD=ibclr(WORDS_B(H),63)
  if (WORD/=0_int64) then
    if (WORD<MIN_WORD.or.WORD>=MAX_WORD) goto 900
  endif
enddo
''' + source[end:]
source = source[:source.index('contains\n')] + 'end subroutine\n'
(DEST / 'terminal.f90').write_text(source)
for name in ['oracle.f90', 'guard-driver.f90', 'trap-driver.f90', 'microbenchmark.f90']:
    (DEST / name).write_bytes((BASE / name).read_bytes())
for name in ['run_guards.py', 'run_microbenchmark.py']:
    script = (BASE / name).read_text()
    script = script.replace("'docs/benchmarkfolder/experiments/dp-terminal'",
                            "'docs/benchmarkfolder/experiments/dp-terminal/packed-scan'")
    script = script.replace("'.tirust-build/dp-terminal'", "'.tirust-build/dp-terminal/packed-scan'")
    (DEST / name).write_text(script)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    'scope': 'Second isolated DP terminal prototype; integer bulk bit scan only',
    'base_source_sha256': {name: digest(BASE / name) for name in
                          ['terminal.f90', 'oracle.f90', 'guard-driver.f90',
                           'trap-driver.f90', 'microbenchmark.f90', 'run_guards.py',
                           'run_microbenchmark.py']},
    'generator_sha256': digest(Path(__file__)),
    'generated_sha256': {p.name: digest(p) for p in sorted(DEST.iterdir())
                         if p.suffix in ('.py', '.f90')},
    'guard_equivalence': 'Each real/imaginary bit word with its sign cleared is either zero, '
                         'or in [623 << 52, 1424 << 52), exactly the original exponent range.',
    'numerical_change': 'None to admission rules, initialization, consumed arithmetic or dense fallback.',
}
(DEST / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
