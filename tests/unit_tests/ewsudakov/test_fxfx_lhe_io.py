"""Compile the LHE writer/readers and check preservation of FxFx matching data."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph import MG5DIR
from tests.unit_tests.fks.test_real_channels_phase_space import fortran_routine


SCALES = '<scales pt_clust_3="100.00000" pt_clust_4="99.99900"></scales>'
HISTORY = '<clustering type="born" iproc="0" step_1="1,3;80;mom=2"></clustering>'
REAL_HISTORY = '<clustering type="real" iproc="7" nFKS="7" step_1="2,4;90;mom=-2"></clustering>'
INFO = '#aMCatNLO 1 0 0 0 0 0.0 0.0 0 4 0 1.0 1.0 1.0 1.0 1.0'


def event(scales=SCALES, history=HISTORY, info=True, ordinary_scales=False):
    rows = [
        '<event>', '4 1 1.0 100.0 0.0 0.118',
        '2 -1 0 0 0 0 0.0 0.0 50.0 50.0 0.0 0.0 9.0',
        '-2 -1 0 0 0 0 0.0 0.0 -50.0 50.0 0.0 0.0 9.0',
        '21 1 1 2 0 0 50.0 0.0 0.0 50.0 0.0 0.0 9.0',
        '21 1 1 2 0 0 -50.0 0.0 0.0 50.0 0.0 0.0 9.0',
    ]
    if info:
        rows.append(INFO)
    if scales:
        rows.append(scales)
    if history:
        rows.append(history)
    if ordinary_scales:
        rows += ["<scales muf='-1' mur='-1' scalup_1_2='%14.8E'>" % 50., '</scales>']
    return '\n'.join(rows + ['</event>', ''])


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required for LHE I/O checks')
class TestFxFxLheIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='mg-fxfx-lhe-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.path = Path(cls.temporary.name)
        template = Path(MG5DIR) / 'Template/NLO'
        source = (template / 'SubProcesses/handling_lhe_events.f').read_text()
        # Compile the actual event routines; header generation needs process
        # cards and is independent of the event records exercised here.
        selected = source[source.index('      subroutine write_lhef_event('):
                          source.index('      subroutine copy_header(')]
        selected += fortran_routine(source, 'read_rwgt_line')
        (cls.path / 'event_io.f').write_text(selected)
        for name, directory in (('run.inc', 'Source'), ('extra_weights.f', 'Source'),
                                ('unlops.inc', 'SubProcesses'),
                                ('madfks_mcatnlo.inc', 'SubProcesses')):
            shutil.copyfile(template / directory / name, cls.path / name)
        (cls.path / 'driver.f').write_text(DRIVER)
        result = subprocess.run(
            ['gfortran', '-O0', '-g', '-fcheck=all', '-ffixed-line-length-none',
             '-o', 'check', 'extra_weights.f', 'event_io.f', 'driver.f'],
            cwd=cls.path, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def run_io(self, action, reader='normal', text='', metadata=SCALES,
               ickkw=3, delta=False, count=1, info_line=INFO):
        with tempfile.TemporaryDirectory(dir=self.path) as temporary:
            root = Path(temporary)
            (root / 'input.lhe').write_text(text + '</LesHouchesEvents>\n')
            (root / 'metadata.txt').write_text(metadata + '\n')
            (root / 'info.txt').write_text(info_line + '\n')
            result = subprocess.run(
                [str(self.path / 'check'), action, reader, str(ickkw),
                 'delta' if delta else 'plain', str(count)],
                cwd=root, capture_output=True, text=True, timeout=10)
            output = (root / 'output.lhe').read_text()
        return result, output

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_matching_scales_and_clustering_survive_round_trip(self):
        for reader in ('normal', 'catch'):
            with self.subTest(reader=reader):
                result, output = self.run_io('rewrite', reader, event())
                self.assert_success(result)
                self.assertIn(SCALES, output)
                self.assertIn('step_1="1,3;80;mom=2"', output)
                self.assertEqual(output.count('<clustering'), 1)
                again, _ = self.run_io('read', reader, output)
                self.assert_success(again)
                self.assertIn('PT:' + SCALES, again.stdout)
                self.assertIn('step_1="1,3;80;mom=2"', again.stdout)

    def test_valid_info_comment_allows_multiple_clustering_records(self):
        history = HISTORY + '\n' + REAL_HISTORY
        for reader in ('normal', 'catch'):
            with self.subTest(reader=reader):
                result, _ = self.run_io('read', reader, event(history=history))
                self.assert_success(result)
                self.assertIn('PT:' + SCALES, result.stdout.splitlines())
                self.assertIn('C0:' + HISTORY, result.stdout.splitlines())
                self.assertIn('C7:' + REAL_HISTORY, result.stdout.splitlines())

    def test_missing_info_comment_is_rejected_before_metadata(self):
        for reader in ('normal', 'catch'):
            for scales in ('', SCALES):
                for history in ('', HISTORY, HISTORY + '\n' + REAL_HISTORY):
                    with self.subTest(reader=reader, scales=bool(scales), history=bool(history)):
                        result, _ = self.run_io('read', reader,
                                                event(info=False, scales=scales, history=history))
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('missing the #aMCatNLO info line', result.stdout)
                        self.assertIn('AddInfoLHE', result.stdout)

    def test_missing_info_on_next_event_does_not_reuse_previous_event_type(self):
        for reader in ('normal', 'catch'):
            with self.subTest(reader=reader):
                result, output = self.run_io('rewrite', reader,
                                            event() + event(info=False), count=2)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('missing the #aMCatNLO info line', result.stdout)
                self.assertEqual(output.count('<event>'), 1)
                self.assertEqual(output.count('</event>'), 1)

    def test_writer_requires_fxfx_info_before_writing_event(self):
        for info in ('', '#other 1 0 0 0 0 0 0 0 4 0 1 1 1 1 1'):
            with self.subTest(info=info):
                result, output = self.run_io('write', info_line=info)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('missing the #aMCatNLO info line', result.stdout)
                self.assertEqual(output, '')

    def test_absent_matching_line_does_not_consume_history_or_event_end(self):
        for reader in ('normal', 'catch'):
            for history in ('', HISTORY):
                with self.subTest(reader=reader, history=bool(history)):
                    result, _ = self.run_io('read', reader, event(scales='', history=history))
                    self.assert_success(result)
                    lines = result.stdout.splitlines()
                    self.assertIn('PT:', lines)
                    self.assertIn('C0:' + history, lines)

    def test_missing_metadata_is_cleared_between_events(self):
        text = event() + event(scales='')
        for reader in ('normal', 'catch'):
            with self.subTest(reader=reader):
                result, _ = self.run_io('read', reader, text, count=2)
                self.assert_success(result)
                self.assertEqual([line for line in result.stdout.splitlines()
                                  if line.startswith('PT:')], ['PT:' + SCALES, 'PT:'])

    def test_ordinary_scales_block_is_preserved_with_or_without_matching_line(self):
        for reader in ('normal', 'catch'):
            for scales in ('', SCALES):
                for delta in (False, True):
                    with self.subTest(reader=reader, scales=bool(scales), delta=delta):
                        result, _ = self.run_io('read', reader,
                                                event(scales=scales, ordinary_scales=True), delta=delta)
                        self.assert_success(result)
                        self.assertIn('PT:' + scales, result.stdout.splitlines())
                        value = next(line[6:] for line in result.stdout.splitlines()
                                     if line.startswith('SCALE:'))
                        self.assertEqual(float(value), 50.)

    def test_writer_rejects_missing_or_invalid_metadata_before_writing_event(self):
        for metadata in ('', HISTORY, '<scales></scales>',
                         '<scales pt_clust_3="100">'):
            with self.subTest(metadata=metadata):
                result, output = self.run_io('write', metadata=metadata)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('FxFx matching scales', result.stdout)
                self.assertEqual(output, '')

    def test_reading_missing_metadata_cannot_silently_rewrite_fxfx_event(self):
        for reader in ('normal', 'catch'):
            with self.subTest(reader=reader):
                result, output = self.run_io('rewrite', reader, event() + event(scales=''), count=2)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('FxFx matching scales', result.stdout)
                self.assertEqual(output.count('<event>'), 1)
                self.assertEqual(output.count('</event>'), 1)

    def test_matching_record_cannot_be_silently_truncated(self):
        # XML whitespace preserves the same physical scales while exceeding
        # the Fortran matching buffer, without exceeding the reader buffer.
        oversized = SCALES.replace(' pt_clust_4', ' ' * 1000 + 'pt_clust_4')
        for reader in ('normal', 'catch'):
            with self.subTest(reader=reader):
                result, _ = self.run_io('read', reader, event(scales=oversized))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('FxFx matching scales exceed', result.stdout)
        result, output = self.run_io('write', metadata=oversized)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FxFx matching scales', result.stdout)
        self.assertEqual(output, '')

    def test_non_fxfx_writer_and_readers_accept_absent_matching_metadata(self):
        result, output = self.run_io('write', metadata='', ickkw=0)
        self.assert_success(result)
        self.assertNotIn('pt_clust_', output)
        for reader in ('normal', 'catch'):
            for info in (False, True):
                with self.subTest(reader=reader, info=info):
                    text = output if info else event(scales='', history='', info=False,
                                                     ordinary_scales=True)
                    result, _ = self.run_io('read', reader, text, ickkw=0)
                    self.assert_success(result)
                    self.assertIn('PT:', result.stdout.splitlines())

    def test_non_fxfx_writer_retains_optional_info_comment(self):
        result, output = self.run_io('write', info_line='', metadata='', ickkw=0)
        self.assert_success(result)
        self.assertNotIn('#aMCatNLO', output)
        for reader in ('normal', 'catch'):
            with self.subTest(reader=reader):
                result, _ = self.run_io('read', reader, output, ickkw=0)
                self.assert_success(result)


DRIVER = '''\
      program check_lhe_io
      use extra_weights
      implicit none
      include 'run.inc'
      include 'unlops.inc'
      integer nup,idprup,idup(MAXNUP),istup(MAXNUP)
      integer mothup(2,MAXNUP),icolup(2,MAXNUP)
      double precision xwgtup,scalup,aqedup,aqcdup
      double precision pup(5,MAXNUP),vtimup(MAXNUP),spinup(MAXNUP)
      double precision scalup_a(MAXNUP,MAXNUP)
      character*1000 buff,ptclusstring
      common /c_ptclusstring/ ptclusstring
      character*2000 clusterstrings(0:512)
      common /c_clusterstrings/ clusterstrings
      integer mg_rwgt_count,nattr,npNLO,npLO,nFKSprocess
      common /rwgt_count/ mg_rwgt_count
      common /event_attributes/ nattr,npNLO,npLO
      common /c_nFKSprocess/ nFKSprocess
      integer count,i,j
      character*16 action,reader,argument
      call get_command_argument(1,action)
      call get_command_argument(2,reader)
      call get_command_argument(3,argument)
      read(argument,*) ickkw
      call get_command_argument(4,argument)
      mcatnlo_delta=argument.eq.'delta'
      call get_command_argument(5,argument)
      read(argument,*) count
      do_rwgt_scale=.false.
      do_rwgt_pdf=.false.
      store_rwgt_info=.false.
      mg_rwgt_count=0
      nattr=0
      npNLO=-1
      npLO=-1
      nFKSprocess=0
      nup=4
      idprup=1
      xwgtup=1d0
      scalup=100d0
      aqedup=0d0
      aqcdup=.118d0
      idup(1:4)=(/2,-2,21,21/)
      istup(1:4)=(/-1,-1,1,1/)
      mothup=0
      icolup=0
      pup=0d0
      pup(4,1:4)=50d0
      pup(3,1)=50d0
      pup(3,2)=-50d0
      pup(1,3)=50d0
      pup(1,4)=-50d0
      vtimup=0d0
      spinup=9d0
      scalup_a=-999d0
      buff='#aMCatNLO 1 0 0 0 0 0.0 0.0 0 4 0 1 1 1 1 1'
      clusterstrings=' '
c Seed a valid old value to detect stale metadata when reading an event.
      ptclusstring='<scales pt_clust_3="123"></scales>'
      open(unit=12,file='input.lhe',status='old')
      open(unit=13,file='output.lhe',status='replace')
      if(action.eq.'write')then
         open(unit=14,file='metadata.txt',status='old')
         read(14,'(a)') ptclusstring
         close(14)
         open(unit=14,file='info.txt',status='old')
         read(14,'(a)') buff
         close(14)
         call write_lhef_event(13,nup,idprup,xwgtup,scalup,
     &        aqedup,aqcdup,idup,istup,mothup,icolup,pup,vtimup,
     &        spinup,buff,scalup_a)
      else
         do i=1,count
            if(reader.eq.'catch')then
               call read_lhef_event_catch(12,nup,idprup,xwgtup,
     &              scalup,aqedup,aqcdup,idup,istup,mothup,icolup,
     &              pup,vtimup,spinup,buff,scalup_a)
            else
               call read_lhef_event(12,nup,idprup,xwgtup,
     &              scalup,aqedup,aqcdup,idup,istup,mothup,icolup,
     &              pup,vtimup,spinup,buff,scalup_a)
            endif
            write(*,'(a)') 'PT:'//trim(adjustl(ptclusstring))
            write(*,'(a)') 'C0:'//trim(adjustl(clusterstrings(0)))
            do j=1,512
               if(len_trim(clusterstrings(j)).ne.0) then
                  write(*,'(a,i0,a)') 'C',j,
     &                 ':'//trim(adjustl(clusterstrings(j)))
               endif
            enddo
            write(*,'(a,es24.16)') 'SCALE:',scalup_a(1,2)
            if(action.eq.'rewrite')then
               call write_lhef_event(13,nup,idprup,xwgtup,
     &              scalup,aqedup,aqcdup,idup,istup,mothup,icolup,
     &              pup,vtimup,spinup,buff,scalup_a)
            endif
         enddo
         if(reader.eq.'catch')then
            call read_lhef_event_catch(12,nup,idprup,xwgtup,
     &           scalup,aqedup,aqcdup,idup,istup,mothup,icolup,
     &           pup,vtimup,spinup,buff,scalup_a)
            if(buff.ne.'endoffile') stop 10
         endif
      endif
      close(12)
      close(13)
      end
'''


if __name__ == '__main__':
    unittest.main()
