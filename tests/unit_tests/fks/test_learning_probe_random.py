"""Learning probes draw from their own random-number stream."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_born_spreading import mint_support, TEMPLATE


PROGRAM = r'''
program probe_stream
use mint_module
implicit none
double precision :: ordinary(8),mixed(8),probe(6),again(6),r
integer :: i
! The ordinary stream alone.
call rmarin(1802,9373)
do i=1,8
 call ranmar(ordinary(i))
enddo
! Two probes between ordinary draws: the ordinary draws are unchanged.
call rmarin(1802,9373)
call ranmar(mixed(1));call ranmar(mixed(2))
call swap_probe_random_numbers()
do i=1,3
 call ranmar(probe(i))
enddo
call swap_probe_random_numbers()
call ranmar(mixed(3));call ranmar(mixed(4))
call swap_probe_random_numbers()
do i=4,6
 call ranmar(probe(i))
enddo
call swap_probe_random_numbers()
do i=5,8
 call ranmar(mixed(i))
enddo
if(any(mixed.ne.ordinary))stop 1
do i=1,6
 if(any(probe(i).eq.ordinary))stop 2
enddo
! The probe stream continues across probes and is fixed by the seed.
call rmarin(1802,9373)
probe_rng_ready=.false.
call ranmar(r);call ranmar(r)
call swap_probe_random_numbers()
do i=1,6
 call ranmar(again(i))
enddo
call swap_probe_random_numbers()
if(any(again.ne.probe))stop 3
print *,'PASS probes draw from their own seeded stream'
end program
double precision function ran2()
call ranmar(ran2)
end function
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestLearningProbeRandom(unittest.TestCase):
    def test_ordinary_draws_do_not_depend_on_probes(self):
        with tempfile.TemporaryDirectory(prefix='mg7-probe-random-') as temporary:
            work = Path(temporary)
            support = mint_support(work, real_fks_params=True, real_ranmar=True)
            exposed = work / 'exposed_mint.f90'
            exposed.write_text((TEMPLATE / 'mint_module.f90').read_text().replace('private', 'public'))
            support = [str(exposed) if path == str(TEMPLATE / 'mint_module.f90') else path for path in support]
            program = work / 'check.f90'
            program.write_text(PROGRAM)
            result = subprocess.run(['gfortran', '-O0', '-fcheck=all', '-ffree-line-length-none',
                                     '-ffixed-line-length-132', '-I', str(work), *support,
                                     str(TEMPLATE.parent / 'Source/ranmar.f'), str(program),
                                     '-o', str(work / 'check')],
                                    cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
