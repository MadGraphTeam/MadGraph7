"""Unused resonance/event-projection Born caches must not be prepared."""
import os
import shutil
import subprocess
import tempfile

import tests.unit_tests as unittest
from madgraph import MG5DIR


class TestMCMappingCache(unittest.TestCase):
    def test_static_born_caches_follow_their_consumers(self):
        """Execute the production amplitude block with counted Born calls.

        Event-projected sectors never consume collinear caches. Rotation is
        needed only for second-beam FKS emitters outside 2->1 Born kinematics.
        Ordinary Born calls at both coupling slots must remain in every case.
        """
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the Born-cache probe')
        with open(os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses',
                               'driver_vec.f')) as stream:
            source = stream.read()
        source = source[source.index('      recursive subroutine amplitudes_ivec('):]
        declarations = source[source.index('C Born variables'):source.index('C Arguments')]
        start = source.index('            ! if (passcuts_nbody) then')
        block = source[start:source.index('            ! endif', start)]
        caches = {
            'sev_amp2': ('double precision', '(1,3,2)'),
            'snorad_amp2': ('double precision', '(1,3,2)'),
        }
        for prefix in ('scb', 'sc1', 'snb', 'sn1', 'srot'):
            for suffix, kind, shape in (
                    ('amp_split', 'double precision', '(1,3,2)'),
                    ('amp_split_cnt', 'complex*16', '(1,2,2,3,2)'),
                    ('ans_cnt', 'complex*16', '(2,2,3,2)'),
                    ('saveamp', 'complex*16', '(1,1,3,2)'),
                    ('amp2', 'double precision', '(1,3,2)'),
                    ('jamp2', 'double precision', '(0:1,3,2)')):
                caches[prefix+'_'+suffix] = kind, shape
        header = '''      program probe
      implicit none
      integer ngraphs,ncolor,nsplitorders,amp_split_size,max_bhel
      parameter(ngraphs=1,ncolor=1,nsplitorders=2,
     $          amp_split_size=1,max_bhel=1)
      integer nexternal,FKS_J_D(3),BORN_FLAVOR_INDEX_D(3)
      integer iFKS,ivec,icoup,ev,col,legs,calls(5),slots(4)
      logical need_mapping_born(2,3,2)
      double precision p_born(0:3,4),p_born_coll(0:3,4)
      double precision p_born_rot(0:3,4),p_born_ev(0:3,4)
      double precision p_born_norad(0:3,4)
      common /counts/ calls,slots
'''
        header += declarations
        header += ''.join('      %s %s%s\n' % (kind, name, shape)
                          for name, (kind, shape) in caches.items())
        header += '''      FKS_J_D=(/3,1,2/)
      BORN_FLAVOR_INDEX_D=1
      icoup=1
      p_born=1d0
      p_born_coll=2d0
      p_born_rot=3d0
      p_born_ev=4d0
      p_born_norad=5d0
      do legs=4,5
       nexternal=legs
       do ev=0,1
        do col=0,1
         do ivec=1,2
          do iFKS=1,3
           need_mapping_born=.true.
           need_mapping_born(1,iFKS,ivec)=ev.eq.1
           need_mapping_born(2,iFKS,ivec)=col.eq.1
           calls=0
           slots=0
'''
        header += ''.join('           %s=0\n' % name for name in caches)
        tail = '''
           write(*,*) legs,ev,col,ivec,iFKS,calls,slots
          enddo
         enddo
        enddo
       enddo
      enddo
      end
      subroutine sborn_amp_vec_frame(p,a,j,s,c,w,ans,save,
     $                              slot,lane,sector,flavour)
      implicit none
      double precision p(0:3,4),a(1),j(0:1),s(1),w
      complex*16 c(1,2,2),ans(2,2),save(1,1)
      integer slot,lane,sector,flavour,calls(5),slots(4),tag
      common /counts/ calls,slots
      tag=nint(p(0,1))
      calls(tag)=calls(tag)+1
      slots(slot)=slots(slot)+1
      a=tag
      j=tag
      s=tag
      c=tag
      w=tag
      ans=tag
      save=tag
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_static_born_') as tmp:
            path = os.path.join(tmp, 'probe.f')
            binary = os.path.join(tmp, 'probe')
            with open(path, 'w') as stream:
                stream.write(header + block + tail)
            result = subprocess.run([compiler, '-O0', '-fcheck=all',
                '-ffixed-line-length-132', path, '-o', binary],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            output = subprocess.check_output([binary], text=True)
        rows = [list(map(int, line.split())) for line in output.splitlines()]
        self.assertEqual(len(rows), 48)
        for legs, ev, col, lane, sector, *counts in rows:
            rotation = int((3, 1, 2)[sector - 1] == 2 and legs != 4)
            with self.subTest(legs=legs, ev=ev, col=col, lane=lane, sector=sector):
                self.assertEqual(counts[:5], [2, 2*col, rotation, ev, col])
                self.assertEqual(counts[5:], [ev, col, 1+col, 1+col+rotation])

    def test_mapping_setup_keeps_resonances_and_nonstandard_beams(self):
        """Compile the actual serial prepass block and count prepared slots.

        The event mapping is consumed for resonances. The no-radiation
        mapping must remain available for beams without guaranteed event
        projection, including dressed leptons. Ordinary pp, photon, and
        fixed-energy beam pairs need no no-radiation cache.
        """
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the mapping-cache probe')
        with open(os.path.join(MG5DIR, 'Template', 'NLO', 'SubProcesses',
                               'driver_vec.f')) as stream:
            source = stream.read()
        start = source.index('c The resonance predicate', source.index(
            '      subroutine prepare_sector_cache('))
        stop = source.index('            call set_alphaS_and_vec(p1_cnt(0,1,0)', start)
        block = source[start:stop]
        probe = '''      program probe
      implicit none
      integer lpp(2),beams(2,6),i,r,iFKS,ivec,icoup,ncalls,slots
      logical granny_is_res,need_mapping_born(2,1,1)
      double precision p_born_ev(0:3,4),p_born_norad(0:3,4)
      double precision spb_ev(0:3,4,1,1),spb_norad(0:3,4,1,1)
      common /calls/ ncalls,slots
      beams=reshape((/1,1,2,2,0,0,1,-1,3,-3,4,4/),(/2,6/))
      iFKS=1
      ivec=1
      icoup=1
      p_born_ev=1d0
      p_born_norad=2d0
      do i=1,6
         lpp=beams(:,i)
         do r=0,1
            granny_is_res=r.eq.1
            need_mapping_born=.true.
            ncalls=0
            slots=0
''' + block + '''
            write(*,*) i,r,need_mapping_born(:,1,1),ncalls,slots
         enddo
      enddo
      end
      subroutine set_alphaS_and_vec(p,slot)
      double precision p(0:3,4)
      integer slot
      call set_alphaS(p)
      call set_alphaS_vec_from_scalar(slot)
      end
      subroutine set_alphaS(p)
      implicit none
      double precision p(0:3,4)
      integer ncalls,slots
      common /calls/ ncalls,slots
      ncalls=ncalls+1
      end
      subroutine set_alphaS_vec_from_scalar(slot)
      implicit none
      integer slot,ncalls,slots
      common /calls/ ncalls,slots
      slots=slots+2**(slot-1)
      end
'''
        with tempfile.TemporaryDirectory(prefix='mg7_mc_mapping_cache_') as tmp:
            path = os.path.join(tmp, 'probe.f')
            with open(path, 'w') as stream:
                stream.write(probe)
            binary = os.path.join(tmp, 'probe')
            compiled = subprocess.run([compiler, '-O0', '-fcheck=all',
                                       '-ffixed-line-length-132', path, '-o', binary],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            output = subprocess.check_output([binary], text=True)
        rows = [line.split() for line in output.splitlines()]
        self.assertEqual(len(rows), 12)
        for beam, resonance, ev, norad, calls, slots in rows:
            needs_ev = int(resonance) == 1
            needs_norad = int(beam) > 3
            with self.subTest(beam=beam, resonance=resonance):
                self.assertEqual(ev == 'T', needs_ev)
                self.assertEqual(norad == 'T', needs_norad)
                self.assertEqual(int(calls), int(needs_ev) + int(needs_norad))
                self.assertEqual(int(slots), int(needs_ev) + 2 * int(needs_norad))
