"""The buffered FKS Born, its links and helicity sampling use the native provider."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph import MadGraph5Error
from madgraph.iolibs import born_support as support

ROOT = Path(support.__file__).resolve().parents[2]
TEMPLATES = Path(support.__file__).parent/'template_files'
SUBPROCESSES = ROOT/'Template/NLO/SubProcesses'


def template_sampler(nsqamps=1, ngraphs=2):
    """PickHelicityMC as export_fks writes it into born_hel.f."""
    text = (TEMPLATES/'born_hel_splitorders_fks.inc').read_text()
    text = re.search(r'(?ims)^subroutine PickHelicityMC\b.*?^\s*end\s*$', text).group()
    text = text % {'nSqAmpSplitOrders': nsqamps, 'ngraphs': ngraphs}
    return [line.strip() for line in text.splitlines()
            if line.strip() and not re.match(r'[cC](\s|$)|!', line)]


def write_includes(work, max_bhel=4, nsplitorders=2, amp_split_size=2):
    includes = {
        'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
        'genps.inc': 'integer ngraphs,ncolor\nparameter(ngraphs=2,ncolor=3)',
        'orders.inc': 'integer nsplitorders,amp_split_size,qcd_pos,qed_pos\n'
                      'parameter(nsplitorders=%d,amp_split_size=%d,qcd_pos=1,qed_pos=2)\n'
                      'double precision amp_split(amp_split_size)\n'
                      'double complex amp_split_cnt(amp_split_size,2,nsplitorders)\n'
                      'common/to_amp_split/amp_split,amp_split_cnt'
                      % (nsplitorders, amp_split_size),
        'born_nhel.inc': 'integer max_bhel\nparameter(max_bhel=%d)' % max_bhel,
    }
    for name, source in includes.items():
        (work/name).write_text(''.join('      '+line+'\n' for line in source.splitlines()))


def run(work, sources):
    result = subprocess.run([shutil.which('gfortran'), '-O0', '-fcheck=all',
                             '-ffpe-trap=invalid,zero,overflow',
                             '-ffixed-line-length-none', *sources, '-o', 'check'],
                            cwd=work, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    result = subprocess.run([str(work/'check')], cwd=work, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    return result.stdout


class TestProviderHelicitySampler(unittest.TestCase):
    record = {'nsqamps': 1, 'nhelicity': 4}

    def test_rewrite_reads_the_provider_common_and_keeps_other_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_includes(work)
            lines = support.provider_helicity_sampler(
                template_sampler()+['subroutine other(p,a,s,w)', 'call sborn_hel(p,a,s,w)', 'end'],
                self.record, work)
            text = '\n'.join(lines).lower()
            self.assertIn('call sborn_hel(p,born_wgt_recomp_direct)\n', text)
            self.assertEqual(text.count('common/c_born_hel_split/wgt_hel'), 1)
            self.assertIn('call sborn_hel_buffered(p,a,s,w)', text)
            self.assertNotIn('call sborn_hel(p,born_wgt_recomp_direct,', text)

    def test_rewrite_refuses_a_sampler_it_cannot_prove(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_includes(work)
            changed = [line.replace('born_wgt_recomp_direct,loc_saveamp',
                                    'born_wgt_recomp_direct, saved')
                       for line in template_sampler()]
            with self.assertRaises(MadGraph5Error):
                support.provider_helicity_sampler(changed, self.record, work)
            # COMMON/C_BORN_HEL_SPLIT/ is declared by sborn_hel with the
            # provider's extents; a different WGT_HEL shape would alias it.
            with self.assertRaises(MadGraph5Error):
                support.provider_helicity_sampler(template_sampler(nsqamps=2), self.record, work)
            with self.assertRaises(MadGraph5Error):
                support.provider_helicity_sampler(template_sampler(), {'nsqamps': 1, 'nhelicity': 8},
                                                  work)
            # Fixtures without a sampler have nothing to rewrite.
            self.assertEqual(support.provider_helicity_sampler(['subroutine x', 'end'],
                                                               self.record, work),
                             ['subroutine x', 'end'])

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_sampler_picks_from_provider_weights_not_saved_amplitudes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_includes(work)
            lines = ['subroutine sborn_hel(p,ans)', 'implicit none',
                     "include 'nexternal.inc'", 'double precision p(0:3,nexternal-1),ans',
                     'double precision helicity_orders(1,4)',
                     'common/c_born_hel_split/helicity_orders',
                     'helicity_orders(1,:)=[0d0,3d0,1d0,5d0]', 'ans=sum(helicity_orders)', 'end',
                     'double precision function ran2()', 'ran2=0.8d0', 'end']
            lines += support.provider_helicity_sampler(template_sampler(), self.record, work)
            support.write_fortran(work/'sampler.f', lines)
            (work/'check.f90').write_text('''program check
use, intrinsic :: ieee_arithmetic
implicit none
double precision p(0:3,4),vol
integer goodhel(4),hel(0:4),ihel
double complex saved(2,4)
p=1d0
! Only helicities 1..3 are good; saved amplitudes must not be read.
saved=cmplx(ieee_value(1d0,ieee_quiet_nan),0d0,kind=8)
goodhel=1
hel=[3,1,2,3,0]
call pickhelicitymc(p,goodhel,hel,ihel,vol,saved)
if(ihel/=3.or.abs(vol-0.25d0)>1d-15)error stop 'wrong helicity pick'
print*,'PASS'
end program
''')
            self.assertIn('PASS', run(work, ['sampler.f', 'check.f90']))


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestProviderBufferedBorn(unittest.TestCase):

    @unittest.skipUnless(shutil.which('make'), 'requires make')
    def test_standalone_link_drops_unused_native_frame_entry(self):
        # The native SBORN_SF_STORE entry shares sborn_sf.o with the scalar
        # provider facade. check_fks uses only the latter and must not pull in
        # SBORN_SF_FRAME and the rest of the native integration runtime.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root/'SubProcesses/P0_test'
            work.mkdir(parents=True)
            (root/'Source').mkdir()
            (root/'lib').mkdir()
            (root/'Source/make_opts').write_text(
                'FC = %s\nFFLAGS = -O0 -ffixed-line-length-none\n'
                % shutil.which('gfortran'))
            for path in ('SubProcesses/proc_characteristics',
                         'SubProcesses/analyse_opts',
                         'lib/mc_born_types.mod', 'lib/mc_born_support.mod'):
                (root/path).touch()
            (work/'born_support.f').touch()
            support.write_fortran(work/'sborn_sf.f', [
                'subroutine sborn_sf(p,m,n,wgt)', 'implicit none',
                'integer m,n', 'double precision p(0:3,4),wgt',
                'wgt=p(0,1)+m+n', 'end',
                'subroutine sborn_sf_store(p,m,n,wgt)', 'implicit none',
                'integer m,n', 'double precision p(0:3,4),wgt',
                'call sborn_sf_frame(p,m,n,wgt)', 'end'])
            support.write_fortran(work/'check_sa_fks.f', [
                'program check', 'implicit none',
                'double precision p(0:3,4),wgt', 'p=7d0',
                'call sborn_sf(p,1,2,wgt)',
                "if(wgt.ne.10d0)error stop 'wrong Born link'",
                "print*,'PASS'", 'end'])
            command = [shutil.which('make'), '-f',
                       str(SUBPROCESSES/'makefile_fks_dir'), 'check_fks',
                       'FKSSA=check_sa_fks.o sborn_sf.o', 'LINKLIBSSUD=',
                       'BORN_SUPPORT_LIB=', 'BORN_SUPPORT_LINK=', 'LDFLAGS=']
            built = subprocess.run(command, cwd=work, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            checked = subprocess.run([str(work/'check_fks')], cwd=work,
                                     capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout+checked.stderr)
            self.assertIn('PASS', checked.stdout)
            # Retaining the unused entry reproduces the real link failure;
            # this control also rejects a fixture that accidentally defines it.
            (work/'check_fks').unlink()
            retained = subprocess.run(command+['BORN_SUPPORT_GC='], cwd=work,
                                      capture_output=True, text=True)
            self.assertNotEqual(retained.returncode, 0)
            self.assertIn('sborn_sf_frame', retained.stdout+retained.stderr)

    def test_frame_born_returns_provider_buffers_without_saved_amplitudes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_includes(work)
            bridge = work/'boost_to_frame.f'
            shutil.copyfile(SUBPROCESSES/bridge.name, bridge)
            support.write_native_frame_bridge(bridge)
            wrapper = support.routine(bridge.read_text(), 'sborn_amp_frame')
            stub = ['subroutine sborn_frame(p,ans)', 'implicit none',
                    "include 'nexternal.inc'", "include 'genps.inc'", "include 'orders.inc'",
                    'double precision p(0:3,nexternal-1),ans',
                    'double precision amp2(ngraphs),jamp2(0:ncolor)',
                    'common/to_amps/amp2,jamp2',
                    'double complex ans_cnt(2,nsplitorders)', 'common/c_born_cnt/ans_cnt',
                    'ans=7d0', 'amp2=[1d0,2d0]', 'jamp2=[3d0,4d0,5d0,6d0]',
                    'amp_split=[8d0,9d0]', 'amp_split_cnt=(10d0,1d0)', 'ans_cnt=(11d0,2d0)',
                    'end']
            support.write_fortran(work/'wrapper.f', stub+wrapper)
            (work/'check.f90').write_text('''program check
implicit none
double precision p(0:3,4),ans,amp2(2),jamp2(0:3),split(2)
double complex split_cnt(2,2,2),ans_cnt(2,2),saved(2,4)
p=1d0
saved=(5d0,5d0)
call sborn_amp_frame(p,amp2,jamp2,split,split_cnt,ans,ans_cnt,saved)
if(ans/=7d0.or.any(amp2/=[1d0,2d0]).or.any(jamp2/=[3d0,4d0,5d0,6d0]))error stop 'Born buffers'
if(any(split/=[8d0,9d0]).or.any(split_cnt/=(10d0,1d0)).or.any(ans_cnt/=(11d0,2d0))) &
  error stop 'split buffers'
if(any(saved/=(0d0,0d0)))error stop 'stale saved amplitudes'
print*,'PASS'
end program
''')
            self.assertIn('PASS', run(work, ['wrapper.f', 'check.f90']))

    def test_ewsudakov_recomputes_the_saved_amplitudes_it_needs(self):
        # The provider Born returns no per-helicity amplitudes, so the
        # buffered EW Sudakov must fill them itself before using them.
        source = (SUBPROCESSES/'fks_buffered.f').read_text()
        body = '\n'.join(support.routine(source, 'compute_ewsudakov_buffered')).lower()
        refill = body.index('call sborn_amp_saved_frame(p_born,')
        self.assertLess(body.index('if (.not.has_ewsudakov) return'), refill)
        self.assertLess(refill, body.index('call sudakov_wrapper_frame('))
        saved = support.routine((SUBPROCESSES/'boost_to_frame.f').read_text(),
                                'sborn_amp_saved_frame')
        self.assertTrue(any(re.match(r'(?i)\s*call\s+sborn_amp\(p_f,', line) for line in saved))
        # The standalone Sudakov checks hand the same amplitudes to the kernel.
        for name in ('check_sudakov.f', 'check_sudakov_angle2.f'):
            calls = re.findall(r'(?i)call\s+(sborn_amp\w*_frame|sudakov_wrapper_frame)\b',
                               (SUBPROCESSES/name).read_text())
            for i, call in enumerate(calls):
                if call.lower() == 'sudakov_wrapper_frame':
                    self.assertEqual(calls[i-1].lower(), 'sborn_amp_saved_frame', name)


if __name__ == '__main__':
    unittest.main()
