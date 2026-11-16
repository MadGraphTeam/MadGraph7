"""Fixed-order Born reuse preserves arrays and state across counterterms."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import tests.unit_tests as unittest
from madgraph import MG5DIR


class TestNLOFOBornReuse(unittest.TestCase):
    def test_actual_degenerate_term_preserves_ordinary_inputs(self):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the degenerate-term probe')
        root = Path(MG5DIR)
        native = (root/'Template/NLO/SubProcesses/fks_singular.f').read_text()
        buffered = (root/'Template/NLO/SubProcesses/fks_buffered.f').read_text()
        routines = []
        for name in ['sreal_deg_buffered', 'AP_reduced', 'AP_reduced_prime',
                     'xkplus', 'xkdelta', 'xklog', 'sborncol_isr_store_buffered',
                     'Qterms_reduced_spacelike']:
            source = buffered if name.endswith('_buffered') else native
            start = source.index('      subroutine '+name+'(')
            routines.append(source[start:source.index('\n      end\n', start)+11])
        includes = {
            'genps.inc': '      integer ngraphs,ncolor\n      parameter(ngraphs=1,ncolor=1)\n',
            'nexternal.inc': '      integer nexternal,nincoming\n      parameter(nexternal=4,nincoming=2)\n',
            'born_nhel.inc': '      integer max_bhel\n      parameter(max_bhel=2)\n',
            'nFKSconfigs.inc': '      integer fks_configs\n      parameter(fks_configs=1)\n',
            'coupl.inc': '      double precision g,nf\n      parameter(nf=5d0)\n      double complex gal(2)\n      common/probe_couplings/g,gal\n',
            'run.inc': '      double precision q2fact(2),xiAI(2),RAI(2)\n      common/probe_scales/q2fact,xiAI,RAI\n',
            'q_es.inc': '      double precision QES2\n      common/coupl_es/QES2\n',
            'orders.inc': '''      integer nsplitorders,amp_split_size,qcd_pos,qed_pos
      parameter(nsplitorders=2,amp_split_size=2,qcd_pos=1,qed_pos=2)
      double complex amp_split_cnt(amp_split_size,2,nsplitorders)
      common/probe_amplitudes/amp_split_cnt
'''}
        module = '''      module extra_weights
      logical doreweight
      double precision wgtdegrem_xi,wgtdegrem_lxi,wgtdegrem_muF
      end module
      module ALOHA_OBJECT
      type ALOHA
         double complex W(20)
      end type
      end module
      module fks_phase_space_data
      double precision p_born(0:3,3),p_born_coll(0:3,3)
      double precision shat,sqrtshat,ybst_til_tocm,ybst_til_tolab
      double precision p_i_fks_cnt(0:3,-2:2),p_i_fks_ev(0:3)
      double precision xi_i_fks_ev,xij_kperp(0:3)
      double complex xij_aor
      double precision y_ij_fks_ev
      end module
'''
        with tempfile.TemporaryDirectory(prefix='fo_degenerate_') as temp:
            work = Path(temp)/'SubProcesses/P0'
            work.mkdir(parents=True)
            pdf = Path(temp)/'Source/PDF/pdf.inc'
            pdf.parent.mkdir(parents=True)
            pdf.write_text('      integer PDFscheme\n      common/probe_pdf/PDFscheme\n')
            for name, text in includes.items():
                (work/name).write_text(text)
            (work/'blocks.f').write_text(module+'\n'.join(routines))
            shutil.copy2(root/'tests/input_files/nlo_fo_degenerate_probe.f', work/'probe.f')
            build = subprocess.run([compiler, '-O0', '-fcheck=all',
                                    '-ffixed-line-length-132', 'blocks.f', 'probe.f', '-o', 'probe'],
                                   cwd=work, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
            self.assertEqual(build.returncode, 0, build.stdout)
            result = subprocess.run(['./probe'], cwd=work, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn('DEGENERATE_PASS', result.stdout)

    def test_degenerate_term_sites_reuse_ordinary_born_and_refresh_hooks(self):
        compiler = shutil.which('gfortran')
        if compiler is None:
            self.skipTest('gfortran is required for the Born lifecycle probe')
        source = (Path(MG5DIR)/'Template/NLO/SubProcesses/driver_mintFO.f').read_text()
        sites = list(re.finditer(r'(?im)^ *call sreal_deg_buffered\(', source))
        self.assertEqual(len(sites), 4)
        blocks = []
        for site in sites:
            # Retain both the original preparation and the post-degenerate
            # refresh through the actual collinear consumer call.
            start = source.rfind('calculatedBorn=.false.', 0, site.start())
            start = source.rfind('calculatedBorn=.false.', 0, start)
            start = source.rfind('\n', 0, start)+1
            end = source.index('call sreal_store_buffered(', site.end())
            end = source.index('\n', end)+1
            while source[end:end+6].strip() == '$':
                end = source.index('\n', end)+1
            blocks.append(source[start:end])
        program = '''      module probe_state
      implicit none
      integer calls,iextra_cnt
      double precision cache,model
      logical calculatedBorn
      end module
      program probe
      use probe_state
      implicit none
      integer hook,site
      do hook=0,1
      do site=1,4
         calls=0
         iextra_cnt=hook
         cache=-999d0
         model=1d0
         calculatedBorn=.false.
         call exercise(site)
         if (calls.ne.2+2*hook) stop 1
         if (cache.ne.3d0*(1d0+hook)) stop 2
         if (.not.calculatedBorn) stop 3
      enddo
      enddo
      print *, 'PASS'
      end
      subroutine exercise(site)
      use probe_state
      implicit none
      integer site
      double precision p_born(0:3,3),p_born_coll(0:3,3)
      double precision p_born_used(0:3,3),p1_cnt(0:3,4,-2:2)
      double precision born_amp2(1),born_jamp2(0:1),wgt_born
      double precision coll_amp2(1),coll_jamp2(0:1),wgt_coll
      double precision born_amp_split(1),coll_amp_split(1)
      double complex born_amp_split_cnt(1,2,2),born_ans_cnt(2,2)
      double complex coll_amp_split_cnt(1,2,2),coll_ans_cnt(2,2)
      double complex born_saveamp(1,2),coll_saveamp(1,2)
      double precision ret_amp_split(1),real_amp_split(1)
      double precision fx_sc,fx_c,xi_i_fks_cnt(-2:2),zero,one
      double precision deg_xi_sc,deg_lxi_sc,deg_xi_c,deg_lxi_c
      double precision amp_split_wgtdegrem_xi(1)
      double precision amp_split_wgtdegrem_lxi(1)
      double precision amp_split_wgtdegrem_muF(1)
      double precision amp_split_wgtpsch_p(1),amp_split_wgtpsch_l(1)
      double precision amp_split_wgtpsch_d(1)
      zero=0d0
      one=1d0
      p_born=2d0
      p_born_coll=3d0
      p1_cnt=1d0
      xi_i_fks_cnt=.2d0
'''
        for index, block in enumerate(blocks, 1):
            program += '      if (site.eq.%d) then\n'%index+block
            if index in (2, 4):
                program += '''      if (any(born_ans_cnt.ne.coll_ans_cnt)) stop 18
      if (any(born_amp_split_cnt.ne.coll_amp_split_cnt)) stop 19
'''
            program += '      endif\n'
        program += '''      end
      subroutine sborn_amp_frame(p,a,j,s,c,w,b,h)
      use probe_state
      implicit none
      double precision p(0:3,3),a(1),j(0:1),s(1),w
      double complex c(1,2,2),b(2,2),h(1,2)
      if (calculatedBorn) stop 4
      calls=calls+1
      cache=p(0,1)*model
      a=cache
      j=cache
      s=cache
      c=dcmplx(cache,-cache)
      b=c(1,1,1)
      h=c(1,1,1)
      w=cache
      calculatedBorn=.true.
      end
      subroutine sreal_deg_buffered(p,x,y,dx,dl,wx,wl,wm,wp,wpl,wd,b,c,bc,cc)
      use probe_state
      implicit none
      double precision p(*),x,y,dx,dl,wx(*),wl(*),wm(*)
      double precision wp(*),wpl(*),wd(*)
      double complex b(2,2),c(1,2,2),bc(2,2),cc(1,2,2)
      if (any(b.ne.dcmplx(2d0,-2d0))) stop 5
      if (any(c.ne.dcmplx(2d0,-2d0))) stop 6
      if (any(bc.ne.dcmplx(3d0,-3d0))) stop 7
      if (any(cc.ne.dcmplx(3d0,-3d0))) stop 8
c A custom hook is allowed to change model and cache state.
      if (iextra_cnt.gt.0) then
         model=2d0
         cache=-777d0
         calculatedBorn=.false.
      endif
      end
      subroutine sreal_store_buffered(p,x,y,w,r,real,born,b,c,h,coll,bc,cc,hc)
      use probe_state
      implicit none
      double precision p(*),x,y,w,r(1),real(1),born(1),coll(1)
      double complex b(2,2),c(1,2,2),h(1,2)
      double complex bc(2,2),cc(1,2,2),hc(1,2)
      if (y.ne.1d0) stop 9
      if (any(born.ne.2d0*model)) stop 10
      if (any(coll.ne.3d0*model)) stop 11
      if (any(b.ne.dcmplx(2d0,-2d0)*model)) stop 12
      if (any(c.ne.dcmplx(2d0,-2d0)*model)) stop 13
      if (any(h.ne.dcmplx(2d0,-2d0)*model)) stop 14
      if (any(bc.ne.dcmplx(3d0,-3d0)*model)) stop 15
      if (any(cc.ne.dcmplx(3d0,-3d0)*model)) stop 16
      if (any(hc.ne.dcmplx(3d0,-3d0)*model)) stop 17
c Model the non-event-projected ISR consumer's array overwrite.
      if (x.gt.0d0) then
         b=bc
         c=cc
      endif
      end
'''
        with tempfile.TemporaryDirectory(prefix='fo_born_reuse_') as temp:
            directory = Path(temp)
            (directory/'probe.f').write_text(program)
            build = subprocess.run([compiler, '-O0', '-fcheck=all',
                                    '-ffixed-line-length-132', 'probe.f', '-o', 'probe'],
                                   cwd=temp, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
            self.assertEqual(build.returncode, 0, build.stdout)
            run = subprocess.run(['./probe'], cwd=temp, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(run.returncode, 0, run.stdout)
            self.assertIn('PASS', run.stdout)
