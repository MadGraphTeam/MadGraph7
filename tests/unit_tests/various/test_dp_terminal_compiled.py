"""Compiled DP terminal contracts against frozen fixtures and dense arithmetic.

The rank-one independent oracle and IEEE fixtures are immutable inputs. Higher
ranks use the unchanged public compact updater as the executable baseline,
preserving its zero-vertex skip and operation order, including exceptional
values. Only a test copy of the private routine receives an admission-output
argument; production keeps its ordinary ABI.
"""
from hashlib import sha256
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import unittest

from madgraph.iolibs.file_writers import FortranWriter
from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA as Exporter
from madgraph.various.q_polynomial import FortranPolynomialRoutines


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'docs/benchmarkfolder/experiments/dp-terminal'
COMPACT_FIXTURE = ROOT / 'docs/benchmarkfolder/experiments/dp-higher-rank/kernels/public.f'
COMPACT_SHA256 = '836116f037c543ca6acc5a40ff178459723f4a7fb5234cf4ee1ae6f0c17bacb2'
FROZEN_HASHES = {
    'oracle.f90': '9a0e0d96e493c69f51018cb1b24d3eedf8095e41c5be0f669dee1b930ec3a73e',
    'guard-driver.f90': '49d097b6035e27a693eb0cfa6fff14763ddd9d67eebdfa970e66ee588a092fb2',
    'trap-driver.f90': '791ad65a11649ffae18a364d3916215018d108e653e78459ef097339b3d09f36',
}
FLAGS = (
    ('o0-round', ['-O0', '-frounding-math']),
    ('o3', ['-O3']),
    ('o3-round', ['-O3', '-frounding-math']),
    ('o3-static', ['-O3', '-frounding-math', '-fno-automatic']),
)


def frozen(name):
    data = (FIXTURES / name).read_bytes()
    if sha256(data).hexdigest() != FROZEN_HASHES[name]:
        raise AssertionError('Independent frozen DP fixture changed: ' + name)
    return data.decode()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError('Expected exactly one source marker: ' + old)
    return text.replace(old, new, 1)


def instrument(private, left_rank=1):
    """Record the executed branch without extra floating-point operations."""
    private, count = re.subn(
        r'(?i)(SUBROUTINE TEST_DP_TERMINAL_WL_%d_1\([^\n]*)(\))' % left_rank,
        r'\1,TEST_FAST\2', private, count=1)
    if count != 1:
        raise AssertionError('Private terminal signature was not found')
    private = replace_once(private, '  LOGICAL WL_ENTRY_INEXACT\n',
        '  LOGICAL WL_ENTRY_INEXACT\n  LOGICAL, INTENT(OUT) :: TEST_FAST\n')
    private, count = re.subn(r'(?im)^(\s*IF\s*\(\.NOT\.WL_ENTRY_INEXACT\).*)$',
        r'TEST_FAST=.FALSE.\n\1', private, count=1)
    if count != 1:
        raise AssertionError('Private terminal entry guard was not found')
    private = replace_once(private, 'C DP_TERMINAL_FAST_PATH\n',
                           'C DP_TERMINAL_FAST_PATH\nTEST_FAST=.TRUE.\n')
    return private


def write_fixed(path, source, context=None):
    with FortranWriter(str(path)) as writer:
        writer.downcase = False
        writer.writelines(source, context={} if context is None else context)


def prepare(work, max_size=4, rank=2, vertex_rank=1, traced=True,
            reject_layout=False, left_rank=1):
    ncoef = (rank+1)*(rank+2)*(rank+3)*(rank+4)//24
    nvertex = (vertex_rank+1)*(vertex_rank+2)*(vertex_rank+3)*(vertex_rank+4)//24
    (work / 'coef_specs.inc').write_text(
        '      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
        '      PARAMETER(MAXLWFSIZE=%d,VERTEXMAXCOEFS=%d)\n' % (max_size, nvertex))
    (work / 'loop_max_coefs.inc').write_text(
        '      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=%d)\n' % ncoef)
    gen = FortranPolynomialRoutines(rank, updater_max_rank=vertex_rank,
        proc_prefix='TEST_', sub_prefix='TEST_', coef_format='complex*16')
    write_fixed(work / 'constants.f', gen.write_polynomial_constant_module())
    source = gen.write_dp_terminal_wl_updater(left_rank, 1)
    if reject_layout:
        # Force the actual representation-check branch to refuse, without
        # altering arithmetic, dimensions, inputs or the public fallback.
        source = replace_once(source, 'ONE_WORDS(1).NE.ISHFT(1023_8,52)',
                              'ONE_WORDS(1).NE.0_8')
    if traced:
        source = instrument(source, left_rank)
    public = (gen.write_expanded_wl_updater(1, 1) if left_rank == 1
              else gen.write_compact_wl_updater(left_rank, 1))
    write_fixed(work / 'generated.f', public + '\n' + source)
    # The frozen module is replaced by the actual generated module. The entire
    # independent dense arithmetic body, including its grouping, is retained.
    if left_rank == 1:
        (work/'reference_public.f').write_text('')
        oracle = frozen('oracle.f90').split('SUBROUTINE DP_DENSE', 1)[1]
        oracle = ('SUBROUTINE DP_DENSE' + oracle).replace(
            'USE ML5_0_POLYNOMIAL_CONSTANTS', 'USE TEST_POLYNOMIAL_CONSTANTS')
    else:
        data = COMPACT_FIXTURE.read_bytes()
        if sha256(data).hexdigest() != COMPACT_SHA256:
            raise AssertionError('Frozen compact baseline changed')
        reference = re.search(r'(?ms)^      SUBROUTINE TEST_UPDATE_WL_%d_1\(.*?^      END\n' %
                              left_rank, data.decode()).group(0)
        write_fixed(work/'public.f', public)
        if (work/'public.f').read_text().strip() != reference.strip():
            raise AssertionError('Public compact updater changed from frozen baseline')
        (work/'reference_public.f').write_text(reference.replace(
            'TEST_UPDATE_WL_%d_1' % left_rank, 'FROZEN_UPDATE_WL_%d_1' % left_rank))
        oracle = '''subroutine DP_DENSE(a,ls,b,ins,outs,out)
use TEST_POLYNOMIAL_CONSTANTS
implicit none
integer :: ls,ins,outs
complex(8) :: a(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
complex(8) :: b(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
complex(8) :: out(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
call FROZEN_UPDATE_WL_%d_1(a,ls,b,ins,outs,out)
end subroutine
''' % left_rank
    (work / 'oracle.f90').write_text(oracle)
    return gen


def guard_driver():
    driver = frozen('guard-driver.f90').replace(
        'ML5_0_POLYNOMIAL_CONSTANTS', 'TEST_POLYNOMIAL_CONSTANTS').replace(
        'call DP_TERMINAL(', 'call TEST_DP_TERMINAL_WL_1_1(')
    driver = replace_once(driver, 'logical :: entry,took_fast',
                          'logical :: entry,took_fast,want_fast')
    driver = replace_once(driver, 'if(took_fast) then', '''
want_fast=entry.and.rounding==1.and.ls==4.and.ins==4.and.outs==4.and. &
  any(trial==[1,2,4,5,6,13,14,17,18,19,21])
if(took_fast.neqv.want_fast) then
  write(*,*) 'ADMISSION',trial,rounding,flagstate,ls,ins,outs,took_fast
  stop 12
endif
if(took_fast) then
  do i=1,4
    do j=1,4
      if(i==j) cycle
      if(any(transfer(actual(j,:,i),new_bits,30)/=0_int64)) stop 13
    enddo
  enddo''')
    return driver


def higher_rank_driver(driver, left_rank, process_rank=6):
    """Resize immutable fixtures; arithmetic remains in compiled routines."""
    ncoef = (process_rank+1)*(process_rank+2)*(process_rank+3)*(process_rank+4)//24
    return (driver.replace('TEST_DP_TERMINAL_WL_1_1',
                           'TEST_DP_TERMINAL_WL_%d_1' % left_rank)
        .replace('0:14', '0:%d' % (ncoef-1))
        .replace('0,14', '0,%d' % (ncoef-1))
        .replace('4*4*15*2', '4*4*%d*2' % ncoef)
        .replace('15*2', '%d*2' % ncoef)
        .replace(',new_bits,30)', ',new_bits,%d)' % (2*ncoef)))


def higher_dimension_driver(left_rank):
    nleft = (left_rank+1)*(left_rank+2)*(left_rank+3)*(left_rank+4)//24
    nout = (left_rank+2)*(left_rank+3)*(left_rank+4)*(left_rank+5)//24
    source = _DIMENSION_DRIVER.replace('TEST_DP_TERMINAL_WL_1_1',
                                      'TEST_DP_TERMINAL_WL_%d_1' % left_rank)
    source = replace_once(source, 'logical :: entry,fast,want,',
                          'logical :: entry,fast,want,raised,')
    source = replace_once(source, 'do trial=1,2', 'do trial=1,5')
    source = replace_once(source, 'do j=0,4', 'do j=0,%d' % (nleft-1))
    source = replace_once(source,
        '        b(i,j,k)=cmplx(real(2*i+j+k,8)/4,real(i-2*j+k,8)/8,8)', '')
    source = replace_once(source, '  abits=transfer(a,abits)', '''  do k=1,MAXLWFSIZE
    do j=0,4
      do i=1,MAXLWFSIZE
        b(i,j,k)=cmplx(real(2*i+j+k,8)/4,real(i-2*j+k,8)/8,8)
      enddo
    enddo
  enddo
  ! The upper active endpoints must be guarded.
  if(trial==3) then
    a(4,%d,4)=cmplx(snan,snan,8)
    b=cmplx(-0.0_8,0.0_8,8)
  endif
  if(trial==4) b(4,4,4)=cmplx(snan,snan,8)
  if(trial==5) then
    a(:,0:%d,:)=cmplx(0.0_8,0.0_8,8)
    b=cmplx(0.0_8,0.0_8,8)
    a(1,%d,1)=cmplx(2.0_8,0.0_8,8)
    b(1,4,1)=cmplx(3.0_8,0.0_8,8)
  endif
  abits=transfer(a,abits)''' % (nleft-1, nleft-1, nleft-1))
    source = source.replace('trial==1)', 'trial/=2)')
    source = replace_once(source, 'want=MAXLWFSIZE==4.and.trial==1',
                          'want=MAXLWFSIZE==4.and.(trial==1.or.trial==5)')
    source = source.replace('0:14', '0:%d' % (nout-1)).replace(',30)', ',%d)' % (nout*2))
    source = replace_once(source, '        checks=checks+1', '''        if(trial==3) then
          if(any(transfer(new,newbits)/=0_int64)) stop 28
          call ieee_get_flag(ieee_invalid,raised)
          if(raised) stop 29
        endif
        if(trial==5) then
          if(new(1,%d,1)/=cmplx(6.0_8,0.0_8,8)) stop 30
          if(count(new/=cmplx(0.0_8,0.0_8,8))/=1) stop 31
        endif
        checks=checks+1''' % (nout-1))
    return replace_once(source, 'if(any(new(:,15:,:)/=sentinel)) stop 25',
        'if(any(transfer(new(:,%d:,:),newbits)/=0_int64)) stop 25' % nout)


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class DPTerminalCompiledTest(unittest.TestCase):
    def build(self, work, source, flags, output='check', extra=()):
        command = ['gfortran', *flags, '-fcheck=all', '-ffixed-line-length-132',
                   '-ffree-line-length-none', 'constants.f', 'oracle.f90',
                   'reference_public.f', 'generated.f', *extra, source, '-o', output]
        result = subprocess.run(command, cwd=work, capture_output=True,
                                text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def run_binary(self, work, name='check', arguments=(), traps=False):
        return subprocess.run([str(work/name), *map(str, arguments)], cwd=work,
            capture_output=True, text=True, timeout=90,
            env=dict(os.environ, GFORTRAN_ERROR_BACKTRACE='0'),
            preexec_fn=no_core if traps else None)

    def test_frozen_ieee_corpus_admission_bits_flags_and_traps(self):
        self.check_ieee_corpus(1)

    def test_higher_rank_ieee_corpus_admission_bits_flags_and_traps(self):
        for left_rank in (2, 3):
            with self.subTest(left_rank=left_rank):
                self.check_ieee_corpus(left_rank)

    def check_ieee_corpus(self, left_rank):
        with tempfile.TemporaryDirectory(prefix='madloop-dp-terminal-') as tmp:
            work = Path(tmp)
            prepare(work, rank=2 if left_rank == 1 else 6, left_rank=left_rank)
            guards = guard_driver()
            traps = frozen('trap-driver.f90').replace(
                'call DP_TERMINAL(', 'call TEST_DP_TERMINAL_WL_1_1(')
            if left_rank != 1:
                guards = higher_rank_driver(guards, left_rank)
                traps = higher_rank_driver(traps, left_rank)
                traps = replace_once(traps, 'case default', '''case(9)
  ! Compact traversal still compares B with LCUT_SIZE zero.
  selected=ieee_invalid
  b(1,0,1)=cmplx(ieee_value(0.0_8,ieee_signaling_nan),0.0_8,kind=8)
  cut=0
case default''')
            (work/'guards.f90').write_text(guards)
            (work/'traps.f90').write_text(traps)
            for name, flags in FLAGS:
                with self.subTest(compiler_mode=name):
                    self.build(work, 'guards.f90', flags)
                    result = self.run_binary(work)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    fields = result.stdout.split()
                    self.assertEqual(fields[:2], ['PASS', '384000'])
                    self.assertGreater(int(fields[2]), 0)
                    self.assertEqual(int(fields[2])+int(fields[3]), 384000)
                    self.build(work, 'traps.f90', flags, output='traps')
                    for mode in range(1, 9 if left_rank == 1 else 10):
                        with self.subTest(trap_mode=mode):
                            old = self.run_binary(work, 'traps', (mode, 0), traps=True)
                            new = self.run_binary(work, 'traps', (mode, 1), traps=True)
                            self.assertEqual(old.returncode, new.returncode,
                                old.stdout+old.stderr+new.stdout+new.stderr)
                            # INEXACT cases 4 and 7 arm a trap with its sticky
                            # flag set. Their paired outcome also covers trap
                            # setup; it does not alone locate the signal inside
                            # the updater. The clear-entry witness below tests
                            # omitted arithmetic's INEXACT independently.
                            if mode <= 4 or mode == 9:
                                self.assertEqual(old.returncode, -signal.SIGFPE)
                            elif mode in (5, 6, 8):
                                self.assertEqual(old.returncode, 0, old.stderr)
                            if old.returncode == 0:
                                self.assertEqual(old.stdout, new.stdout)

    def test_actual_dimensions_inactive_snan_tails_and_nonfour_fallback(self):
        for size in (4, 5):
            for flags in (['-O3'], ['-O3', '-frounding-math', '-fno-automatic']):
                with self.subTest(max_size=size, flags=flags), tempfile.TemporaryDirectory(
                        prefix='madloop-dp-dimensions-') as tmp:
                    work = Path(tmp)
                    prepare(work, max_size=size, rank=3, vertex_rank=2)
                    (work/'dimensions.f90').write_text(_DIMENSION_DRIVER)
                    self.build(work, 'dimensions.f90', flags)
                    result = self.run_binary(work)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn('PASS DIMENSIONS', result.stdout)

    def test_unknown_binary64_layout_refuses_optimization(self):
        with tempfile.TemporaryDirectory(prefix='madloop-dp-layout-') as tmp:
            work = Path(tmp)
            prepare(work, rank=3, vertex_rank=2, reject_layout=True)
            (work/'dimensions.f90').write_text(replace_once(_DIMENSION_DRIVER,
                'want=MAXLWFSIZE==4.and.trial==1', 'want=.false.'))
            self.build(work, 'dimensions.f90', ['-O3', '-frounding-math'])
            result = self.run_binary(work)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('PASS DIMENSIONS', result.stdout)

    def test_higher_rank_dimensions_inactive_tails_and_active_endpoints(self):
        for left_rank in (2, 3):
            for size, process_rank in ((4, left_rank+1), (4, 6), (5, 6)):
                with self.subTest(left_rank=left_rank, max_size=size,
                                  process_rank=process_rank), tempfile.TemporaryDirectory(
                        prefix='madloop-dp-higher-dimensions-') as tmp:
                    work = Path(tmp)
                    prepare(work, max_size=size, rank=process_rank, left_rank=left_rank)
                    (work/'dimensions.f90').write_text(higher_dimension_driver(left_rank))
                    self.build(work, 'dimensions.f90', ['-O3', '-frounding-math'])
                    result = self.run_binary(work)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn('PASS DIMENSIONS', result.stdout)

    def test_off_diagonal_inexact_requires_preexisting_flag(self):
        for left_rank in (1, 2, 3):
            with self.subTest(left_rank=left_rank), tempfile.TemporaryDirectory(
                    prefix='madloop-dp-inexact-') as tmp:
                work = Path(tmp)
                prepare(work, rank=2 if left_rank == 1 else 6, left_rank=left_rank)
                driver = _INEXACT_DRIVER
                if left_rank != 1:
                    driver = higher_rank_driver(driver, left_rank).replace('480', '6720')
                (work/'inexact.f90').write_text(driver)
                self.build(work, 'inexact.f90', ['-O3', '-frounding-math'])
                result = self.run_binary(work)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn('PASS OFF-DIAGONAL INEXACT', result.stdout)

    def test_actual_caller_capture_and_create_merge_preserve_coefficients(self):
        # The caller is transformed by the production exporter. Its final
        # vertex changes INEXACT in both directions, catching capture before
        # the vertex as well as omission of the query. CREATE and MERGE are
        # the actual production template and generator, not a trace sum.
        for left_rank in (1, 2, 3):
            with self.subTest(left_rank=left_rank):
                self.check_actual_caller(left_rank)

    def check_actual_caller(self, left_rank):
        for static in (False, True):
            with self.subTest(static=static), tempfile.TemporaryDirectory(
                    prefix='madloop-dp-consumer-') as tmp:
                work = Path(tmp)
                gen = prepare(work, rank=3 if left_rank == 1 else 6,
                    vertex_rank=2 if left_rank == 1 else 1,
                    traced=False, left_rank=left_rank)
                updater = 'TEST_UPDATE_WL_%d_1' % left_rank
                caller = _CALLER.replace('TEST_UPDATE_WL_1_1', updater)
                caller = replace_once(caller, '),2,4,2,3,-2,2,2)',
                                      '),%d,4,2,3,-2,2,2)' % (left_rank+1))
                write_fixed(work/'caller.f', caller)
                original = (work/'caller.f').read_text()
                expected = {2: dict(loop_mother_rank=left_rank, vertex_rank=1,
                    loop_mother_number=1, out=2, lcut_size=4, in_size=4, out_size=4)}
                candidate, found = Exporter.prepare_dp_terminal_batch(original, 'TEST_', expected)
                self.assertEqual(found, {2})
                self.assertIn('IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)', candidate)
                self.assertNotIn('TEST_FAST', candidate)
                (work/'caller.f').write_text(candidate)
                (work/'reference.f').write_text(original.replace(
                    'CHECK_CANDIDATE', 'CHECK_REFERENCE').replace(updater, 'DP_DENSE'))
                template = (ROOT/'madgraph/iolibs/template_files/loop_optimized/polynomial.inc').read_text()
                replacements = dict(mp_prefix='', proc_prefix='TEST_', nbornamps=2,
                    real_format='REAL*8', complex_format='COMPLEX*16', kind=8,
                    zero_def='0.0D0', one_def='1.0D0', nloopamps=2,
                    nloop_groups=2, ncomb=2, nSO=1, nSquaredSO=3, nAmpSO=2,
                    color_matrix_size='2')
                write_fixed(work/'consumer.f', template % replacements + '\n' + gen.write_wl_merger(),
                    context={'AmplitudeReduction': False, 'LoopInduced': False, 'ninja_available': False})
                driver = _CONSUMER_DRIVER
                if left_rank != 1:
                    nout = (left_rank+2)*(left_rank+3)*(left_rank+4)*(left_rank+5)//24
                    driver = driver.replace('0:14', '0:%d' % (nout-1)).replace(',30)', ',%d)' % (nout*2))
                    driver = replace_once(driver,
                        'if(any(wl(:,15:,:,2)/=sentinel)) stop 49',
                        'if(any(transfer(wl(:,%d:,:,2),wbits)/=0_int64)) stop 49' % nout)
                (work/'consumer_driver.f90').write_text(driver)
                options = ['-fno-automatic'] if static else []
                # Generated callers use ordinary flags; polynomial.f has its
                # own -O3 rule. Preserve that distinction in this integration.
                command = ['gfortran', '-O0', '-fcheck=all', '-ffixed-line-length-132',
                    *options, '-c', 'constants.f', 'caller.f', 'reference.f']
                built = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=90)
                self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
                self.build(work, 'consumer_driver.f90', ['-O3', '-frounding-math', *options],
                           extra=('consumer.f', 'caller.o', 'reference.o'))
                result = self.run_binary(work)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn('PASS CALLER CREATE MERGE', result.stdout)


_DIMENSION_DRIVER = r'''
program dimensions
use TEST_POLYNOMIAL_CONSTANTS
use, intrinsic :: iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(8) :: a(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
complex(8) :: b(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
complex(8) :: old(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE),new(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
complex(8) :: sentinel
integer(int64) :: oldbits(size(old)*2),newbits(size(new)*2),abits(size(a)*2),bbits(size(b)*2)
logical :: entry,fast,want,oldflags(size(ieee_all)),newflags(size(ieee_all))
integer :: i,j,k,trial,ls,ins,outs,checks
real(8) :: snan
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_rounding_mode(ieee_nearest)
sentinel=cmplx(-123.25_8,456.125_8,8)
snan=ieee_value(0.0_8,ieee_signaling_nan)
checks=0
do trial=1,2
  ! Both arrays have exceptional, unused high-rank coefficients. Admission
  ! must inspect only 0:4, while preserving the actual module array strides.
  a=cmplx(snan,snan,8)
  b=cmplx(snan,snan,8)
  do k=1,MAXLWFSIZE
    do j=0,4
      do i=1,MAXLWFSIZE
        a(i,j,k)=cmplx(real(3*i-2*j+k,8)/8,real(i+j-k,8)/16,8)
        b(i,j,k)=cmplx(real(2*i+j+k,8)/4,real(i-2*j+k,8)/8,8)
      enddo
    enddo
  enddo
  abits=transfer(a,abits)
  bbits=transfer(b,bbits)
  do ls=4,MAXLWFSIZE
    do ins=4,MAXLWFSIZE
      do outs=4,MAXLWFSIZE
        old=sentinel
        new=sentinel
        call ieee_set_flag(ieee_all,.false.)
        call ieee_set_flag(ieee_inexact,trial==1)
        call DP_DENSE(a,ls,b,ins,outs,old)
        call ieee_get_flag(ieee_all,oldflags)
        call ieee_set_flag(ieee_all,.false.)
        call ieee_set_flag(ieee_inexact,trial==1)
        call ieee_get_flag(ieee_inexact,entry)
        call TEST_DP_TERMINAL_WL_1_1(a,ls,b,ins,outs,new,entry,fast)
        call ieee_get_flag(ieee_all,newflags)
        want=MAXLWFSIZE==4.and.trial==1
        if(fast.neqv.want) stop 21
        oldbits=transfer(old,oldbits)
        newbits=transfer(new,newbits)
        if(fast) then
          do i=1,4
            if(any(transfer(old(i,0:14,i),oldbits,30)/=transfer(new(i,0:14,i),newbits,30))) stop 22
            do j=1,4
              if(i/=j.and.any(transfer(new(j,0:14,i),newbits,30)/=0_int64)) stop 23
            enddo
          enddo
        else
          if(any(oldbits/=newbits)) stop 24
        endif
        if(any(new(:,15:,:)/=sentinel)) stop 25
        if(any(oldflags.neqv.newflags)) stop 26
        if(any(transfer(a,abits)/=abits).or.any(transfer(b,bbits)/=bbits)) stop 27
        checks=checks+1
      enddo
    enddo
  enddo
enddo
write(*,*) 'PASS DIMENSIONS',MAXLWFSIZE,checks
end program
'''


_INEXACT_DRIVER = r'''
program off_diagonal_inexact
use, intrinsic :: iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(8) :: a(4,0:14,4),b(4,0:4,4),old(4,0:14,4),new(4,0:14,4)
integer(int64) :: oldbits(480),newbits(480)
logical :: entry,fast,oldflags(size(ieee_all)),newflags(size(ieee_all)),raised
integer :: trial,i
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_rounding_mode(ieee_nearest)
a=cmplx(0.0_8,0.0_8,8)
b=cmplx(0.0_8,0.0_8,8)
! The only nonzero product is omitted diagonal arithmetic's OUT(2,0,1).
! It executes in a separate compilation unit, so it cannot be folded here.
a(1,0,1)=cmplx(0.1_8,0.0_8,8)
b(2,0,1)=cmplx(0.1_8,0.0_8,8)
do trial=1,2
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,trial==2)
  call DP_DENSE(a,4,b,4,4,old)
  call ieee_get_flag(ieee_all,oldflags)
  call ieee_get_flag(ieee_inexact,raised)
  if(.not.raised) stop 31
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,trial==2)
  call ieee_get_flag(ieee_inexact,entry)
  call TEST_DP_TERMINAL_WL_1_1(a,4,b,4,4,new,entry,fast)
  call ieee_get_flag(ieee_all,newflags)
  if(fast.neqv.(trial==2)) stop 32
  if(any(oldflags.neqv.newflags)) stop 33
  oldbits=transfer(old,oldbits)
  newbits=transfer(new,newbits)
  if(.not.fast.and.any(oldbits/=newbits)) stop 34
  do i=1,4
    if(any(transfer(old(i,:,i),oldbits,30)/=transfer(new(i,:,i),newbits,30))) stop 35
  enddo
enddo
write(*,*) 'PASS OFF-DIAGONAL INEXACT'
end program
'''


_CALLER = '''SUBROUTINE CHECK_CANDIDATE()
USE TEST_POLYNOMIAL_CONSTANTS
IMPLICIT NONE
COMPLEX*16 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,2)
COMPLEX*16 COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
COMMON/CALLER_WL/WL,COEFS
CALL LAST_VERTEX()
CALL TEST_UPDATE_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,WL(1,0,1,2))
CALL RECORD_UPDATE_FLAGS()
CALL TEST_CREATE_LOOP_COEFS(WL(1,0,1,2),2,4,2,3,-2,2,2)
END
'''


_CONSUMER_DRIVER = r'''
program caller_consumers
use TEST_POLYNOMIAL_CONSTANTS
use, intrinsic :: iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(8) :: wl(4,0:LOOPMAXCOEFS-1,4,2),coefs(4,0:VERTEXMAXCOEFS-1,4)
common/CALLER_WL/wl,coefs
complex(8) :: loopcoefs(0:LOOPMAXCOEFS-1,3,2),amp(2)
common/TEST_LCOEFS/loopcoefs
common/TEST_AMPS/amp
integer :: cf_d(2,2),cf_n(2,2),goodhel(2),heloffset,helpicked
common/TEST_CF/cf_d,cf_n
logical :: checkphase,heldoublechecked,goodamp(3,2)
common/TEST_INIT/checkphase,heldoublechecked
common/TEST_Filters/goodamp,goodhel,heloffset
common/TEST_HELCHOICE/helpicked
integer :: vertex_mode
common/VERTEX_MODE_CONTROL/vertex_mode
logical :: update_flags(size(ieee_all))
common/UPDATER_FLAGS/update_flags
logical :: reference_update(size(ieee_all)),reference_final(size(ieee_all)),actual_final(size(ieee_all))
complex(8) :: saved_coefs(0:LOOPMAXCOEFS-1,3,2),saved_wl(4,0:LOOPMAXCOEFS-1,4),sentinel
integer(int64) :: cbits(size(loopcoefs)*2),wbits(size(saved_wl)*2)
integer(int64) :: input_bits(4*LOOPMAXCOEFS*4*2),vertex_bits(size(coefs)*2)
integer :: i,j,k,trial
type(ieee_round_type) :: after_rounding
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_rounding_mode(ieee_nearest)
sentinel=cmplx(-123.25_8,456.125_8,8)
cf_d=reshape([3,-5,-7,11],[2,2])
cf_n=reshape([2,-3,5,7],[2,2])
amp=[cmplx(0.375_8,-0.625_8,8),cmplx(1.125_8,0.25_8,8)]
goodhel=[2,3]
goodamp=.true.
heloffset=0
heldoublechecked=.true.
helpicked=-1
do trial=1,2
  checkphase=trial==2
  do k=1,4
    do j=0,LOOPMAXCOEFS-1
      do i=1,4
        wl(i,j,k,1)=cmplx(real(3*i-2*j+k,8)/8,real(i+j-k,8)/16,8)
      enddo
    enddo
    do j=0,VERTEXMAXCOEFS-1
      do i=1,4
        coefs(i,j,k)=cmplx(real(2*i+j+k,8)/4,real(i-2*j+k,8)/8,8)
      enddo
    enddo
  enddo
  input_bits=transfer(wl(:,:,:,1),input_bits)
  vertex_bits=transfer(coefs,vertex_bits)
  do vertex_mode=1,2
    wl(:,:,:,2)=sentinel
    loopcoefs=sentinel
    call ieee_set_flag(ieee_all,.false.)
    call ieee_set_flag(ieee_inexact,vertex_mode==2)
    call CHECK_REFERENCE()
    call ieee_get_flag(ieee_all,reference_final)
    reference_update=update_flags
    saved_coefs=loopcoefs
    saved_wl=wl(:,:,:,2)
    wl(:,:,:,2)=sentinel
    loopcoefs=sentinel
    call ieee_set_flag(ieee_all,.false.)
    call ieee_set_flag(ieee_inexact,vertex_mode==2)
    call CHECK_CANDIDATE()
    call ieee_get_flag(ieee_all,actual_final)
    if(any(reference_update.neqv.update_flags)) stop 41
    if(any(reference_final.neqv.actual_final)) stop 42
    if(any(transfer(saved_coefs,cbits)/=transfer(loopcoefs,cbits))) stop 43
    if(any(transfer(wl(:,:,:,1),input_bits)/=input_bits)) stop 44
    if(any(transfer(coefs,vertex_bits)/=vertex_bits)) stop 45
    if(vertex_mode==1) then
      ! Vertex-generated INEXACT must admit after a clear caller entry.
      do i=1,4
        do j=1,4
          if(i/=j.and.any(transfer(wl(j,0:14,i,2),wbits,30)/=0_int64)) stop 46
        enddo
        if(any(transfer(saved_wl(i,:,i),wbits,LOOPMAXCOEFS*2)/= &
           transfer(wl(i,:,i,2),wbits,LOOPMAXCOEFS*2))) stop 47
      enddo
    else
      ! A vertex that clears an incoming flag must force complete fallback.
      if(any(transfer(saved_wl,wbits)/=transfer(wl(:,:,:,2),wbits))) stop 48
    endif
    if(any(wl(:,15:,:,2)/=sentinel)) stop 49
    call ieee_get_rounding_mode(after_rounding)
    if(after_rounding/=ieee_nearest) stop 50
  enddo
enddo
write(*,*) 'PASS CALLER CREATE MERGE'
end program

subroutine LAST_VERTEX()
use, intrinsic :: ieee_exceptions
implicit none
integer :: vertex_mode
common/VERTEX_MODE_CONTROL/vertex_mode
real(8),volatile :: x,y,z
if(vertex_mode==1) then
  x=0.1_8
  y=0.1_8
  z=x*y
else
  call ieee_set_flag(ieee_inexact,.false.)
endif
end subroutine

subroutine RECORD_UPDATE_FLAGS()
use, intrinsic :: ieee_exceptions
implicit none
logical :: update_flags(size(ieee_all))
common/UPDATER_FLAGS/update_flags
call ieee_get_flag(ieee_all,update_flags)
end subroutine

integer function TEST_ML5SOINDEX_FOR_BORN_AMP(i)
integer :: i
TEST_ML5SOINDEX_FOR_BORN_AMP=i
end function
integer function TEST_ML5SOINDEX_FOR_LOOP_AMP(i)
integer :: i
TEST_ML5SOINDEX_FOR_LOOP_AMP=i
end function
integer function TEST_ML5SQSOINDEX(i,j)
integer :: i,j
TEST_ML5SQSOINDEX=i+j-1
end function
'''
