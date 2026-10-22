"""The MC@NLO event queue serves the records of the accepted lanes unchanged."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph import MG5DIR


class TestMCEventQueue(unittest.TestCase):

    def test_event_state_follows_its_lane(self):
        """Restore each lane's UNLOPS scale before writing its event."""
        compiler = shutil.which('gfortran')
        if not compiler:
            self.skipTest('gfortran required')
        source = (Path(MG5DIR)/'Template/NLO/SubProcesses/driver_mintMC.f'
                  ).read_text()

        def routine(name):
            start = source.index('      subroutine %s(' % name)
            return source[start:source.index('\n      end\n', start)+11]

        probe = '''      module driver_vec
      implicit none
      double precision lane_g(2),lane_muF12(2)
      double precision lane_scale1_lhe(2,2)
      integer lane_ipartner_lhe(2,2)
      end module

      program probe
      use driver_vec
      implicit none
      include 'nFKSconfigs.inc'
      include 'coupl.inc'
      integer iSorH_lhe,ifks_lhe(fks_configs),jfks_lhe(fks_configs)
     &     ,fksfather_lhe(fks_configs),ipartner_lhe(fks_configs)
      double precision scale1_lhe(fks_configs),scale2_lhe(fks_configs)
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe
      double precision muR2_current,muF12_current,
     #                 muF22_current,QES2_current
      common/cscales_current_values/muR2_current,muF12_current,
     #                              muF22_current,QES2_current
      ipartner_lhe=0
      scale1_lhe=0d0
      g=1d0
      muF12_current=100d0
      ipartner_lhe(1)=1
      scale1_lhe(1)=10d0
      call save_lane_event_state(1)
      g=2d0
      muF12_current=400d0
      ipartner_lhe(1)=3
      scale1_lhe(1)=30d0
      call save_lane_event_state(2)
      g=0d0
      muF12_current=0d0
      ipartner_lhe(1)=0
      scale1_lhe(1)=0d0
      call restore_lane_event_state(2)
      print *,g,muF12_current,ipartner_lhe(1),scale1_lhe(1)
      call restore_lane_event_state(1)
      print *,g,muF12_current,ipartner_lhe(1),scale1_lhe(1)
      end

''' + routine('save_lane_event_state') + '\n' + routine('restore_lane_event_state')
        with tempfile.TemporaryDirectory(prefix='mg7_mc_lane_state_') as tmp:
            tmp = Path(tmp)
            (tmp/'nexternal.inc').write_text(
                '      integer nexternal\n      parameter (nexternal=5)\n')
            (tmp/'nFKSconfigs.inc').write_text(
                '      integer fks_configs\n      parameter (fks_configs=2)\n')
            (tmp/'coupl.inc').write_text(
                '      double precision g\n      common /test_coupl/ g\n')
            (tmp/'probe.f').write_text(probe)
            compiled = subprocess.run(
                [compiler, '-ffixed-line-length-132', '-fcheck=all',
                 'probe.f', '-o', 'probe'], cwd=tmp,
                capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(tmp/'probe')], cwd=tmp,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        values = [[float(value) for value in line.split()]
                  for line in result.stdout.splitlines()]
        self.assertEqual(values, [[2., 400., 3., 30.],
                                  [1., 100., 1., 10.]])

    def test_queued_events_are_copied_unchanged(self):
        """Refill, rewind and serve as driver_mintMC.f does.

        Every accepted lane of a batch is written to a scratch queue and the
        queue is rewound; requests are served from it until it is empty, and
        the next batch overwrites it from the start. The served records must
        be the queued ones, byte for byte: lines longer than the copy buffer,
        exactly as long as it, or with trailing blanks, and no stale record of
        a longer earlier batch.
        """
        compiler = shutil.which('gfortran')
        if not compiler:
            self.skipTest('gfortran required')
        source = (Path(MG5DIR)/'Template/NLO/SubProcesses/driver_mintMC.f'
                  ).read_text()
        start = source.index('      subroutine copy_queued_event(')
        copy = source[start:source.index('\n      end\n', start)+11]

        def event(i):
            return ['  <event>',
                    ' %d 66 0.1E+01' % i,
                    'x' * 1024,
                    'y%d' % i * 1500,
                    '#aMCatNLO %d   ' % i,
                    '  </event>']

        # batches of accepted lanes, and the number of requests served from
        # each before the next batch is drawn (the last one keeps a lane)
        batches = [[1, 2, 3], [4], [5, 6]]
        served = [3, 1, 1]
        probe = '''      program probe
      implicit none
      integer lun,lout,ib,i,j,k,nbatch(3),nserve(3),length
      character*4000 line
      data nbatch/%s/
      data nserve/%s/
      open(newunit=lun,status='scratch',form='formatted')
      open(newunit=lout,file='served.lhe',status='replace')
      open(10,file='events.txt',status='old')
      do ib=1,3
         rewind(lun)
         do i=1,nbatch(ib)
            do j=1,6
               read(10,'(i5,a)') length,line
               write(lun,'(a)') line(1:length)
            enddo
         enddo
         rewind(lun)
         do k=1,nserve(ib)
            call copy_queued_event(lun,lout)
         enddo
      enddo
      close(lout)
      end

''' % (','.join(str(len(b)) for b in batches),
       ','.join(str(n) for n in served)) + copy
        expected = []
        with tempfile.TemporaryDirectory(prefix='mg7_mc_queue_') as tmp:
            tmp = Path(tmp)
            with open(tmp/'events.txt', 'w') as stream:
                for batch, nserve in zip(batches, served):
                    for i in batch:
                        for line in event(i):
                            stream.write('%5d%s\n' % (len(line), line))
                    for i in batch[:nserve]:
                        expected.extend(event(i))
            (tmp/'probe.f').write_text(probe)
            compiled = subprocess.run(
                [compiler, '-ffixed-line-length-132', '-fno-automatic',
                 '-fcheck=all', 'probe.f', '-o', 'probe'], cwd=tmp,
                capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(tmp/'probe')], cwd=tmp,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual((tmp/'served.lhe').read_text(),
                             '\n'.join(expected)+'\n')


if __name__ == '__main__':
    unittest.main()
