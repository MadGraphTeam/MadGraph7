"""Actual replay-regularized MINT boundaries and native10/SCL10 restarts."""
from pathlib import Path
import json
import os
import struct
import shutil
import subprocess
import tempfile
import unittest
from tests.unit_tests.fks.test_born_spreading import mint_support,TEMPLATE
from tests.input_files.tiled_s_event_reference import audit_archive,read_archive,reassemble,objective

PROGRAM = r'''

module online_fixture
implicit none
integer :: ordinary_nodes=0,probe_nodes=0,probe_ends=0,fold_count=0,last_iteration=0
integer :: ordinary_epoch(20)=0,probe_epoch(20)=0,grid_changes=0,map_changes=0,plot_points=0
integer,allocatable :: rng_a(:),rng_b(:)
double precision,volatile :: delay_sink=0d0
double precision :: old_grid(0:32,4),old_map(4,2),old_ratio(2)
integer :: slot_seen(2)=0,ordinary_slot_saved=0
logical :: zero_integrand=.false.
double precision,allocatable :: probe_vtot(:,:),probe_etot(:,:)
logical :: have_probe_snapshot=.false.
end module

program exercise_online
use mint_module
use online_fixture
implicit none
integer :: nseed,i,j
double precision :: replay_x(ndimmax),replay_a(ndimmax),replay_b(ndimmax),replay_ratio_a,replay_ratio_b
integer,allocatable :: seed(:)
character(32) :: action
logical :: exists
integer :: ids(2)=[1,2],providers(2)=[1,2],pdgs(2,5)
double precision,external :: integrand
call get_command_argument(1,action)
call DefaultFKSParam
if(index(trim(action),'card_').eq.1)then
 open(99,file='budget-card.dat',status='replace')
 if(trim(action).eq.'card_default')then
  write(99,'(a)')'#FKSExplicitSum','.true.'
 elseif(trim(action).eq.'card_legacy')then
  write(99,'(a)')'#FKSSectorLearning','29'
 elseif(trim(action).eq.'card_switch')then
  write(99,'(a)')'#FKSSectorLearning','.true.'
 else
  write(99,'(a)')'#FKSSectorLearningBudget'
  if(trim(action).eq.'card_value')write(99,*) .95d0
  if(trim(action).eq.'card_zero')write(99,*) 0d0
  if(trim(action).eq.'card_high')write(99,*) 1.01d0
 endif
 close(99)
 call FKSParamReader('budget-card.dat',.false.,.true.)
 if(trim(action).eq.'card_default'.and.(FKSSectorLearningBudget.ne..20d0.or.FKSSectorLearning))stop 229
 if(trim(action).eq.'card_switch'.and..not.FKSSectorLearning)stop 231
 if(trim(action).eq.'card_value'.and.FKSSectorLearningBudget.ne..95d0)stop 230
 print *,'PASS native CPU budget card',trim(action)
 stop 0
endif
FKSSectorLearningBudget=.35d0
use_poly_virtual=.false.
FKSSectorLearning=.true.
FKSExplicitSum=.true.;MCSubtractionAtFixedFlow=.true.
VirtTrickSurrogate=0
fixed_order=.false.;nlo_ps=.true.;born_spread_active=.false.
ndim=4;nchans=1;ichan=1;iconfigs(1)=1;iconfig=1
n_ord_virt=0;only_virt=.false.;wgt_mult=1d0
ifold=1;ifold_energy=2;ifold_yij=3;ifold_phi=4
min_virt_fraction_mint=.005d0;double_events=.false.
call random_seed(size=nseed);allocate(seed(nseed),rng_a(nseed),rng_b(nseed));seed=17291
call random_seed(put=seed)
allocate(probe_vtot(nintegrals,0:maxchannels),probe_etot(nintegrals,0:maxchannels))
call register_fixture_layouts
zero_integrand=index(trim(action),'zero').eq.1
if(trim(action).eq.'train'.or.trim(action).eq.'zero_train')then
 imode=0;ncalls0=4096;itmax=4;accuracy=0d0
 ! Automatic point counts (ncalls0<=0, as in MC@NLO runs) detect a zero integral.
 if(zero_integrand)ncalls0=-1
 call mint(integrand)
 if(online_ordinary_calls.ne.ordinary_nodes)stop 201
 if(online_probes.le.0.or.probe_nodes.ne.16*online_probes.or.probe_ends.ne.online_probes)stop 202
 if(online_reused_nodes.ne.16*online_probes.or.online_restore_checks.ne.online_probes)stop 203
 if(online_probes.gt.(online_ordinary_calls-48)/16)stop 204
 if(online_record_rejections.ne.0.or..not.sl_ready.or.sector_learning_collect)stop 205
 if(grid_changes.lt.1.or.last_iteration.ne.4)stop 206
 if(map_changes.lt.1)stop 207
 if(size(sl_entries).ne.3.or.size(unified_slots).ne.3)stop 231
 if(unified_slots(1)%gauge.ne.1.or.unified_slots(2)%gauge.ne.0.or.unified_slots(3)%gauge.ne.1)stop 232
 if(sl_entries(3)%nflavours.ne.0.or.online_slots(3)%total_samples.ne.0)stop 233
 if(any(sl_entries(3)%permutations.ne.spread([(j,j=1,16)],2,3)))stop 234
 if(any(online_slots(1:2)%total_samples.lt.16))stop 235
 if(any([(online_slots(j)%gx*online_slots(j)%gy.lt.4,j=1,2)]))stop 236
 if(abs(ans(2,0)).gt.2d0)stop 208
 if(plot_points.ne.ordinary_nodes)stop 209
 if(sum(online_ordinary_hits(:,1,:)).ne.ordinary_epoch(last_iteration))stop 210
 if(sum(online_proxy_hits(:,1,:)).ne.probe_epoch(last_iteration))stop 211
 if(online_extra_cpu.lt.online_probe_cpu+online_fit_cpu.or.online_ordinary_cpu.le.0d0)stop 212
 if(any(online_total_hits.ne.online_base_hits+online_ordinary_hits+online_proxy_hits))stop 213
 inquire(file='sector_learning_orbits.bin',exist=exists)
 if(exists)stop 214
 inquire(file='sector_learning_online.bin',exist=exists)
 if(.not.exists)stop 215
 call write_grids_to_file
 print *,'CHECK',ordinary_nodes,probe_nodes,online_probes,online_reused_nodes,online_restore_checks,grid_changes,map_changes
 print *,'CHECKCPU',online_ordinary_cpu,online_extra_cpu,online_probe_cpu,online_fit_cpu,online_write_cpu
 print *,'PASS native mint iteration/integral isolation'
else
 imode=1;ncalls0=64;itmax=1;accuracy=0d0;nint_used=0
 if(trim(action).eq.'production'.or.trim(action).eq.'zero_production')then
  imode=2
  call setup_basic_mint
 endif
 if(trim(action).eq.'wrongfold')ifold(2)=2
 if(trim(action).eq.'wrongmode')FKSSectorLearning=.false.
 if(trim(action).eq.'wrongbudget')FKSSectorLearningBudget=.95d0
 call read_grids_from_file
 call register_fixture_layouts
 if(size(sl_entries).ne.3.or.sl_entries(3)%nflavours.ne.0)stop 237
 if(unified_slots(1)%gauge.ne.1.or.unified_slots(2)%gauge.ne.0.or.unified_slots(3)%gauge.ne.1)stop 238
 if(trim(action).eq.'wronglayout')then
  call sector_learning_register_layout(3,[1,2,3],[1,2,3],reshape([(21,j=1,15)],[3,5]))
 endif
 call sector_learning_select_slot(1)
 if(.not.sl_ready)stop 216
 if(trim(action).eq.'production')then
  imode=2;replay_x=.371d0
  do i=1,3
   call sector_learning_select_slot(i)
   call random_seed(get=rng_a)
   call resample_sector_radiation(replay_x,replay_a,replay_ratio_a,sl_entries(i)%ids(1))
   call resample_sector_radiation(replay_x,replay_b,replay_ratio_b,sl_entries(i)%ids(1))
   call random_seed(get=rng_b)
   if(any(rng_a.ne.rng_b).or.any(replay_a.ne.replay_b).or.replay_ratio_a.ne.replay_ratio_b)stop 239
   if(replay_a(4).ne.replay_x(4))stop 240
  enddo
  print *,'PASS frozen GEN111 map replay including unsupported slot'
  stop 0
 endif
 call mint(integrand)
 if(probe_nodes.ne.0.or.sector_learning_collect)stop 217
 print *,'PASS fresh MINT1 restart without refit'
endif
end program

subroutine register_fixture_layouts
use mint_module
implicit none
integer :: pdg4(4,5),pdg3(3,5)
pdg4=21;pdg3=21
pdg4(:,3)=6;pdg3(:,3)=24
call sector_learning_register_layout(1,[1,2,3,4],[1,1,2,2],pdg4)
call sector_learning_register_layout(2,[1,3,5],[1,2,2],pdg3)
pdg3(:,3)=-6
call sector_learning_register_layout(3,[2,3,4],[1,2,3],pdg3)
end subroutine

double precision function integrand(x,vol,ifl,fout)
use mint_module
use online_fixture
implicit none
integer :: ifl,j,sector,first,slot,ns,ids(4),flavour
double precision :: x(ndimmax),vol,fout(nintegrals),mapped(ndimmax),ratio,value,total,anchor(ndimmax),pi
logical :: collect
if(ifl.eq.2)then
 if(sector_learning_collect)then
  if(fold_count.ne.16)stop 221
  probe_ends=probe_ends+1
 endif
 integrand=fout(2);return
endif
collect=sector_learning_collect;pi=acos(-1d0)
call register_fixture_layouts
if(collect.and.ifl.eq.0)call sector_learning_begin_point
slot=1+mod(ordinary_nodes,2)
if(collect)slot=1+mod(online_probes,2)
if(collect.and.nit.eq.2)then
 slot=1
 if(probe_epoch(nit).lt.16)slot=2
endif
if(collect.and.nit.eq.3)slot=1
if(.not.collect)ordinary_slot_saved=slot
ns=4;ids=[1,2,3,4]
if(slot.eq.2)then
 ns=3;ids=[1,3,5,0]
endif
call sector_learning_select_slot(slot)
if(.not.collect.and.imode.eq.0)then
 ordinary_nodes=ordinary_nodes+1;ordinary_epoch(nit)=ordinary_epoch(nit)+1
 anchor=.371d0
 call random_seed(get=rng_a)
 call resample_sector_radiation(anchor,mapped,ratio,ids(2))
 call random_seed(get=rng_b)
 if(any(rng_a.ne.rng_b))stop 222
 if(last_iteration.ne.nit)then
  if(last_iteration.gt.0)then
   if(any(xgrid(0:32,1:4,1).ne.old_grid))grid_changes=grid_changes+1
  endif
  old_grid=xgrid(0:32,1:4,1);last_iteration=nit
 endif
 if(any(xgrid(0:32,1:4,1).ne.old_grid))stop 223
 if(slot_seen(slot).eq.nit)then
  if(any(mapped(1:4).ne.old_map(:,slot)).or.ratio.ne.old_ratio(slot))stop 224
 else
  if(slot_seen(slot).gt.0.and.any(mapped(1:4).ne.old_map(:,slot)))map_changes=map_changes+1
  old_map(:,slot)=mapped(1:4);old_ratio(slot)=ratio;slot_seen(slot)=nit
 endif
elseif(collect)then
 if(any(xgrid(0:32,1:4,1).ne.old_grid))stop 225
 if(imode.ne.1)stop 226
 if(ifl.eq.0)then
  fold_count=0;call sector_learning_begin_point
  have_probe_snapshot=.true.
  probe_vtot=0d0;probe_etot=0d0
  probe_vtot(1:nintegrals,0:nchans)=vtot(1:nintegrals,0:nchans)
  probe_etot(1:nintegrals,0:nchans)=etot(1:nintegrals,0:nchans)
 else
  if(any(vtot(1:nintegrals,0:nchans).ne.probe_vtot(1:nintegrals,0:nchans)))stop 227
  if(any(etot(1:nintegrals,0:nchans).ne.probe_etot(1:nintegrals,0:nchans)))stop 228
 endif
 fold_count=fold_count+1;probe_nodes=probe_nodes+1;probe_epoch(nit)=probe_epoch(nit)+1
endif
! A real callback CPU cost makes the sparse probe token policy observable.
do j=1,20000
 delay_sink=delay_sink+sin(dble(j)+x(1))
enddo
total=.1d0*vol
if(collect)then
 first=(ns+1)*(fold_count-1)+1
 call sector_learning_add_contribution(first,fold_count,1,total)
 if(nit.ne.2)call sector_learning_add_contribution(first,fold_count,2,-.2d0*total)
 if(nit.eq.3)call sector_learning_add_contribution(first,fold_count,3,.15d0*total)
endif
do sector=1,ns
 call resample_sector_radiation(x,mapped,ratio,ids(sector))
 value=vol*ratio*(sin(2d0*pi*mapped(2))+.4d0*cos(2d0*pi*mapped(3)))
 if(mod(sector,2).eq.0)value=-.9d0*value
 value=value/dble(sector)
 total=total+value
 if(collect)then
  call sector_learning_mark_range(first+sector,first+sector,ids(sector))
  call sector_learning_add_contribution(first+sector,fold_count,1,value)
  if(nit.ne.2)call sector_learning_add_contribution(first+sector,fold_count,2,-.3d0*value)
  if(nit.eq.3)call sector_learning_add_contribution(first+sector,fold_count,3,-.1d0*value)
 endif
enddo
fout=0d0
if(collect)then
 ! Probe return values must never enter physical integral/plot accumulators.
 fout(1)=1d12*vol;fout(2)=-1d12*vol
elseif(.not.zero_integrand)then
 fout(1)=abs(total);fout(2)=total
endif
pass_cuts_check=.true.
integrand=fout(2)
end function

double precision function ran2()
call random_number(ran2)
end function
'''

@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestEpochReplayIteration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='mg7-epoch-replay-iteration-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work=Path(cls.temporary.name)
        support=mint_support(cls.work,real_fks_params=True)
        stubs=cls.work/'stubs.f90'
        text=stubs.read_text().replace('subroutine hwu_add_points\nend subroutine',
            'subroutine hwu_add_points\nuse online_fixture,only:plot_points\nplot_points=plot_points+1\nend subroutine')
        # Place the fixture module before production/stubs without changing MINT.
        module_text,program_text=PROGRAM.split('program exercise_online',1)
        fixture=cls.work/'fixture.f90';fixture.write_text(module_text)
        stubs.write_text(text)
        exposed=cls.work/'exposed_mint.f90'
        exposed.write_text((TEMPLATE/'mint_module.f90').read_text().replace('private','public').replace('module mint_module','module mint_module\nuse online_fixture,only:ordinary_slot_saved',1).replace('call sl_unified_maybe_probe(fun)','call sl_unified_maybe_probe(fun)\n          if(sl_current.ne.ordinary_slot_saved)error stop "Ordinary Born slot was not restored"'))
        support=[str(exposed)if p==str(TEMPLATE/'mint_module.f90')else p for p in support]
        check=cls.work/'check.f90';check.write_text('program exercise_online'+program_text)
        cls.executable=cls.work/'check'
        result=subprocess.run(['gfortran','-O0','-g','-fcheck=all','-ffree-line-length-none',
            '-ffixed-line-length-132','-I',str(cls.work),str(fixture),*support,str(check),'-o',str(cls.executable)],
            cwd=cls.work,capture_output=True,text=True)
        if result.returncode:raise AssertionError(result.stdout+result.stderr)

    def run_case(self,action,work,failure=None):
        """Run one fixture action; a failing action must stop at its own guard
        (failure is the expected message; 'ERROR STOP' excludes crashes)."""
        result=subprocess.run([str(self.executable),action],cwd=work,
            capture_output=True,text=True,timeout=90)
        output=result.stdout+result.stderr
        if failure is None:
            self.assertEqual(result.returncode,0,output)
            self.assertIn('PASS',output)
        else:
            self.assertNotEqual(result.returncode,0,output)
            self.assertNotIn('PASS',output)
            self.assertIn(failure,output)
        return output

    def test_actual_mint_iteration_grid_and_integral_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary)
            output=self.run_case('train',work)
            self.assertIn('mixed-map ABS summaries are TRAINING heuristics',output)
            self.assertIn('no separate calibration',output)
            self.assertTrue((work/'sector_learning_online.bin').is_file())
            self.assertFalse((work/'sector_learning_orbits.bin').exists())
            audit=audit_archive(work/'sector_learning_online.bin',work/'mint_grids',expected_budget=.35)
            self.assertTrue(audit['passed'],audit)
            parsed=read_archive(work/'sector_learning_online.bin')
            self.assertEqual(len(parsed['records']),4)
            self.assertEqual(parsed['records'][0]['entries'][0]['nf'],2)
            self.assertEqual(parsed['records'][1]['entries'][0]['nf'],1)
            self.assertEqual(parsed['records'][1]['entries'][1]['nt'],1)
            self.assertEqual(parsed['records'][1]['entries'][1]['fit_status'],1)
            self.assertEqual(parsed['records'][2]['entries'][1]['nt'],0)
            self.assertEqual(parsed['records'][2]['entries'][0]['nf'],3)
            self.assertGreater(parsed['records'][3]['entries'][1]['nt'],0)
            changed=False
            for epoch in parsed['records']:
                before,_=reassemble(epoch['entries'],epoch['acc']['proxy'].shape,map_key='before')
                delta=max(abs(x-y)for x,y in zip(before.data,epoch['acc']['proxy'].data))
                changed=changed or delta>1e-8*max(1e-300,max(abs(x)for x in before.data))
            self.assertTrue(changed,'Actual fitted fixture must distinguish AFTER from BEFORE proxy feed')
            for epoch in parsed['records']:
                self.assertEqual(len(epoch['entries']),3)
                unused=epoch['entries'][2]
                self.assertEqual((unused['nf'],unused['nt']),(0,0))
            destination=os.environ.get('FOLDING_EPOCH_REPLAY_ITERATION_EVIDENCE_DIR')
            if destination:
                target=Path(destination);target.mkdir(parents=True,exist_ok=True)
                for name in ['sector_learning_online.bin','mint_grids']:
                    shutil.copyfile(work/name,target/name)
                (target/'output.log').write_text(output)
                (target/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

    def test_fresh_process_restart_and_wrong_method_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary);self.run_case('train',work)
            before=(work/'sector_learning_online.bin').read_bytes()
            training_grid=(work/'mint_grids').read_bytes()
            self.run_case('restart',work)
            self.run_case('production',work)
            self.assertEqual((work/'sector_learning_online.bin').read_bytes(),before)
            (work/'mint_grids').write_bytes(training_grid)
            self.run_case('wrongmode',work,'FKSSectorLearning differs from the saved MINT training')
            self.run_case('wrongfold',work,'Learned sector production requires matched111')
            self.run_case('wrongbudget',work,'FKSSectorLearningBudget differs from the saved MINT training')
            self.run_case('wronglayout',work,'Learned sector layout changed')
            damaged=''.join(line for line in (work/'mint_grids').read_text().splitlines(True)
                            if not line.startswith(' SCL10')and not line.startswith('SCL10'))
            (work/'mint_grids').write_text(damaged)
            self.run_case('restart',work,'FKSSectorLearning differs from the saved MINT training')

    def test_zero_integral_ends_every_step_gracefully(self):
        # A zero-result job stops before training ends; its grid has no frozen
        # table, and later steps stop at zero before they would need one.
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary)
            for action in ('zero_train','zero_restart','zero_production'):
                with self.subTest(action=action):
                    result=subprocess.run([str(self.executable),action],cwd=work,
                        capture_output=True,text=True,timeout=90)
                    output=result.stdout+result.stderr
                    self.assertEqual(result.returncode,0,output)
                    self.assertNotIn('PASS',output)
                    if action=='zero_train':
                        self.assertIn('THE INTEGRAL APPEARS TO BE ZERO',output)
                    grids=(work/'mint_grids').read_text()
                    self.assertNotIn('SCL10',grids)

    def test_actual_saved_metadata_and_unsupported_slot_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary);self.run_case('train',work)
            original=(work/'mint_grids').read_text()
            for prefix,index,value in [('SCL10 R',-1,'4'),('SCL10 L',-1,'2'),
                                       ('SCL10 H',-1,'2'),('SCL10 G',6,'.9'),
                                       ('SCL10 P',4,'17')]:
                with self.subTest(prefix=prefix,index=index):
                    lines=original.splitlines();row=next(i for i,line in enumerate(lines)if line.strip().startswith(prefix))
                    words=lines[row].split();words[index]=value;lines[row]=' '.join(words)
                    (work/'mint_grids').write_text('\n'.join(lines)+'\n')
                    self.run_case('restart',work,'ERROR STOP')
            for addition in ('SCL7 C .65 0 0 0 0\n','SCL10 P 3 1 '+' '.join(map(str,range(1,17)))+'\n'):
                (work/'mint_grids').write_text(original+addition)
                self.run_case('restart',work,'Trailing unified learning metadata')
            for tag in ['H','C','L','R','P','G']:
                for mutation in ['long_tag','extra_column','long_marker']:
                    with self.subTest(tag=tag,mutation=mutation):
                        lines=original.splitlines()
                        index=next(i for i,line in enumerate(lines)if line.strip().startswith('SCL10 '+tag+' '))
                        words=lines[index].split()
                        if mutation=='long_tag':words[1]+='ACK'
                        if mutation=='long_marker':words[0]+='EXTRA'
                        if mutation=='extra_column':words.append('17')
                        lines[index]=' '.join(words)
                        (work/'mint_grids').write_text('\n'.join(lines)+'\n')
                        self.run_case('restart',work,'ERROR STOP Malformed SCL10' if tag!='H' or mutation!='long_marker'
                                      else 'Unexpected learning metadata before SCL10')

    def test_budget_default_range_and_nondefault_card(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary)
            self.run_case('card_default',work)
            self.run_case('card_value',work)
            self.run_case('card_zero',work,'FKSSectorLearningBudget must be in (0,1]')
            self.run_case('card_high',work,'FKSSectorLearningBudget must be in (0,1]')
            self.run_case('card_switch',work)
            self.run_case('card_legacy',work,'FKSSectorLearning must be .true. or .false.')

    def test_native10_diagnostic_only_loss_requires_immediate_previous_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary);self.run_case('train',work)
            archive=read_archive(work/'sector_learning_online.bin')
            entry=archive['records'][1]['entries'][1]
            self.assertEqual((entry['nt'],entry['fit_status']),(1,1))
            current_only=objective(entry['b'],entry['r'],entry['before'])
            self.assertGreater(abs(float(entry['diagnostics'][0])-current_only),1e-12)
            audit_archive(work/'sector_learning_online.bin',work/'mint_grids',expected_budget=.35)
            original=(work/'sector_learning_online.bin').read_bytes()
            needle=struct.pack('=4d',*entry['diagnostics'])
            self.assertEqual(original.count(needle),1,'Locate one exact diagnostic-only payload')
            changed=struct.pack('=4d',current_only,current_only,*entry['diagnostics'][2:])
            (work/'sector_learning_online.bin').write_bytes(original.replace(needle,changed,1))
            with self.assertRaises(ValueError):
                audit_archive(work/'sector_learning_online.bin',work/'mint_grids',expected_budget=.35)



class TestSectorLearningCards(unittest.TestCase):
    def test_learning_cards_are_checked_before_mint0(self):
        from madgraph.interface.amcatnlo_run_interface import sector_learning_card_error
        with tempfile.TemporaryDirectory() as temporary:
            card=Path(temporary)/'FKS_params.dat'
            good={'folding':[1,1,1],'born_spreading':False}
            card.write_text('! Learning\n#FKSSectorLearning\n.true.\n! Default :: .false.\n')
            self.assertIsNone(sector_learning_card_error(str(card),good))
            self.assertIn('folding = 1, 1, 1',sector_learning_card_error(str(card),dict(good,folding=[2,2,1])))
            self.assertIn('born_spreading',sector_learning_card_error(str(card),dict(good,born_spreading=True)))
            card.write_text('#FKSSectorLearning\nF\n')
            self.assertIsNone(sector_learning_card_error(str(card),dict(good,folding=[2,2,1])))
            self.assertIsNone(sector_learning_card_error(str(Path(temporary)/'missing.dat'),dict(good,folding=[2,2,1])))
