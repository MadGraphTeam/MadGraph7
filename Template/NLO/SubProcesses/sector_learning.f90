module sector_learning
  ! FKSSectorLearning (docs/tiled_s_event_integral.md): learned tile maps of
  ! the outer-sector radiation coordinates. This module holds the learning
  ! state, the probe records, the map fits, the epoch replay bank, the
  ! grid-split shares and the SCL10 table writer, and uses no MINT state.
  ! The hooks that need the integration grids or the integrand (probes,
  ! resampling, grid accumulation, epoch boundaries, table reading) are in
  ! mint_module, which uses this module.
  implicit none
  ! FKSSectorLearning: tile maps of the sector radiation coordinates,
  ! learned during MINT0 from 16-node (4,4,1) probes of a common Born point.
  ! B is the Born/nbody part at its original node, R the per-sector part.
  integer,parameter,public :: sl_nodes=16
  ! Fixed method tag stored in the native10 archive and the SCL10 table.
  integer,parameter :: sl_method_id=29
  ! Layout version of that archive and table; an older table needs a new MINT0.
  integer,parameter :: sl_format_version=12
  type :: sector_learning_entry
     integer :: slot=0,ntrain=0,nflavours=0
     integer,allocatable :: ids(:),providers(:),pdgs(:,:),groups(:)
     integer,allocatable :: permutations(:,:)
     double precision,allocatable :: b(:,:,:),r(:,:,:,:)
  end type
  type(sector_learning_entry),allocatable :: sl_entries(:)
  ! FKSSectorLearning as applied to this run: set by the MC@NLO driver for
  ! complete (abrv=all) runs; off at LO and at fixed order.
  logical,public :: sector_learning_active=.false.
  ! Write sector_learning_online.bin, with the epoch's complete grid
  ! accounting, for the archive audit of the tests.
  logical,public :: sector_learning_archive=.false.
  logical :: sl_ready=.false.,sl_point_rejected=.false.
  integer :: sl_current=0,sl_point_slot=0,sl_point_flavours=0
  double precision,allocatable :: sl_point_b(:,:),sl_point_r(:,:,:)
  double precision,allocatable :: sl_grid_snapshot(:,:,:)
  integer :: sl_grid_nint=0,sl_grid_ndim=0,sl_grid_nchans=0
  ! Ordinary MINT0 points with learned maps: each outer sector is evaluated
  ! in its own xi/y grid cells (sl_split_cell). The ABS weight of the point
  ! is split into a Born part (sl_split_share0, sampled cells) and sector
  ! parts (sl_split_share), each added to the grid where it was evaluated.
  logical :: sl_split_point=.false.,sl_split_valid=.false.
  integer :: sl_split_fallbacks=0
  integer,allocatable :: sl_split_cell(:,:)
  double precision :: sl_split_share0=0d0,sl_split_h0=0d0
  ! Sum of |contribution| over the point: the rounding scale of the split.
  double precision :: sl_split_abs=0d0
  double precision,allocatable :: sl_split_b(:),sl_split_r(:,:)
  double precision,allocatable :: sl_split_h(:),sl_split_share(:)

  ! Map fitting: static quarter-Sobol slot gauge plus a three-start
  ! (current/identity/gauge-zero) phase and tile-assignment search.
  type,public :: unified_fit_attempt
     integer :: kind=0,status=0,reuse=0
     double precision :: loss_start=0d0,loss_phase=0d0,loss_final=0d0
     integer,allocatable :: initial_map(:,:),phase_map(:,:),final_map(:,:)
     integer,allocatable :: phase_trace(:,:),tile_trace(:,:)
  end type
  type :: unified_slot_entry
     integer :: gauge=0,fit_status=0,selected_orientation=0
     integer :: primary_gx=0,primary_gy=0,primary_selected=0
     integer :: alternative_gx=0,alternative_gy=0,alternative_selected=0
     integer,allocatable :: qcodes(:,:)
     type(unified_fit_attempt) :: primary(3),alternative(3)
  end type
  type(unified_slot_entry),allocatable :: unified_slots(:)

  ! Epoch replay retains only the immediately preceding raw probe bank.
  ! Layouts cannot change inside MINT0 (sector_learning_register_layout).
  type :: epoch_replay_bank_entry
     integer :: nf=0,nt=0
     double precision,allocatable :: b(:,:,:),r(:,:,:,:)
  end type
  type(epoch_replay_bank_entry),allocatable :: epoch_replay_previous(:)

  ! Online maps: epoch-local raw441 probes; ordinary points alone enter the
  ! grid and the integral/error sums.
  type :: online_slot_entry
     integer :: gx=1,gy=1,oldgx=1,oldgy=1,total_samples=0
     integer,allocatable :: before(:,:)
     double precision :: loss_before=0d0,loss_after=0d0,alternative_x=0d0,alternative_y=0d0
  end type
  type(online_slot_entry),allocatable :: online_slots(:)
  integer :: online_ordinary_calls=0,online_probes=0
  integer :: online_epoch=0,online_nint_before=0,online_unit=0,online_record_rejections=0
  double precision :: online_ordinary_cpu=0d0,online_probe_cpu=0d0,online_fit_cpu=0d0
  double precision :: online_write_cpu=0d0,online_extra_cpu=0d0,online_clock=0d0
  double precision :: online_budget=.20d0
  double precision,allocatable :: online_grid_before(:,:,:),online_base_acc(:,:,:)
  double precision,allocatable :: online_ordinary_acc(:,:,:),online_total_acc(:,:,:)
  integer,allocatable :: online_base_hits(:,:,:),online_ordinary_hits(:,:,:),online_total_hits(:,:,:)
  ! A per-sector move of the coordinate descent (sl_coordinate_descent): the
  ! sector's map, or its translation code, is replaced only by a candidate
  ! whose loss with the other sectors summed in fixed is strictly lower.
  abstract interface
     subroutine sl_sector_proposer(fixed,r,sector,gx,gy,codes,map,loss)
       double precision,intent(in) :: fixed(:,:,:),r(:,:,:,:),loss
       integer,intent(in) :: sector,gx,gy
       integer,intent(inout) :: codes(:),map(:)
     end subroutine sl_sector_proposer
  end interface
  private :: sl_coordinate_descent,sl_phase_proposer,sl_tile_proposer

contains
  subroutine sector_learning_assignment(cost,permutation)
    ! Successive shortest augmenting paths for a square assignment. Rows
    ! are output tiles, columns the source tiles. Scale only the internal
    ! potentials; callers check any strict improvement in physical units.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    double precision,intent(in) :: cost(:,:)
    integer,intent(out) :: permutation(size(cost,1))
    double precision,allocatable :: u(:),v(:),minimum(:),scaled(:,:)
    integer,allocatable :: matched(:),previous(:)
    logical,allocatable :: visited(:)
    integer :: n,row,column,next_column,active_row,j
    double precision :: magnitude,delta,reduced
    n=size(cost,1)
    if(n.lt.1.or.size(cost,2).ne.n.or..not.all(ieee_is_finite(cost))) &
         error stop 'Invalid learned tile assignment costs'
    magnitude=maxval(abs(cost))
    if(magnitude.eq.0d0)then
       permutation=[(row,row=1,n)]
       return
    endif
    allocate(u(0:n),v(0:n),minimum(n),scaled(n,n),matched(0:n),previous(n),visited(0:n))
    scaled=cost/magnitude
    u=0d0
    v=0d0
    matched=0
    do row=1,n
       matched(0)=row
       column=0
       minimum=huge(1d0)
       previous=0
       visited=.false.
       do
          visited(column)=.true.
          active_row=matched(column)
          delta=huge(1d0)
          next_column=0
          do j=1,n
             if(visited(j))cycle
             reduced=scaled(active_row,j)-u(active_row)-v(j)
             if(reduced.lt.minimum(j))then
                minimum(j)=reduced
                previous(j)=column
             endif
             if(minimum(j).lt.delta)then
                delta=minimum(j)
                next_column=j
             endif
          enddo
          if(next_column.eq.0.or..not.ieee_is_finite(delta))error stop 'Unresolved learned tile assignment'
          do j=0,n
             if(visited(j))then
                u(matched(j))=u(matched(j))+delta
                v(j)=v(j)-delta
             elseif(j.gt.0)then
                minimum(j)=minimum(j)-delta
             endif
          enddo
          column=next_column
          if(matched(column).eq.0)exit
       enddo
       do
          next_column=previous(column)
          matched(column)=matched(next_column)
          column=next_column
          if(column.eq.0)exit
       enddo
    enddo
    do column=1,n
       permutation(matched(column))=column
    enddo
  end subroutine sector_learning_assignment

  subroutine sl_tile_losses_unchecked(b,r,permutations,loss)
    double precision,intent(in) :: b(:,:,:),r(:,:,:,:)
    integer,intent(in) :: permutations(:,:)
    double precision,intent(out) :: loss(size(b,3))
    integer :: orbit,node,flavour,sector
    double precision :: value
    loss=0d0
    do orbit=1,size(b,3)
       do node=1,sl_nodes
          do flavour=1,size(b,1)
             value=b(flavour,node,orbit)
             do sector=1,size(r,3)
                value=value+r(flavour,permutations(node,sector),sector,orbit)
             enddo
             loss(orbit)=loss(orbit)+max(0d0,-value)
          enddo
       enddo
    enddo
  end subroutine sl_tile_losses_unchecked

  logical function sl_valid_capacity(gx,gy)
    ! Coarse tile grids that the 4x4 node stencil supports.
    integer,intent(in) :: gx,gy
    sl_valid_capacity=(gx.eq.2.or.gx.eq.4).and.(gy.eq.2.or.gy.eq.4)
  end function sl_valid_capacity

  logical function sl_is_bijection(map)
    integer,intent(in) :: map(sl_nodes)
    integer :: node
    sl_is_bijection=.true.
    do node=1,sl_nodes
       if(count(map.eq.node).ne.1)sl_is_bijection=.false.
    enddo
  end function sl_is_bijection

  subroutine sl_tile_losses_one(fixed,r,sector,map,loss)
    ! Tile loss with every sector but one already summed into fixed.
    double precision,intent(in) :: fixed(:,:,:),r(:,:,:,:)
    integer,intent(in) :: sector,map(sl_nodes)
    double precision,intent(out) :: loss(size(fixed,3))
    integer :: orbit,node,flavour
    loss=0d0
    do orbit=1,size(fixed,3)
       do node=1,sl_nodes
          do flavour=1,size(fixed,1)
             loss(orbit)=loss(orbit)+max(0d0,-(fixed(flavour,node,orbit)+r(flavour,map(node),sector,orbit)))
          enddo
       enddo
    enddo
  end subroutine sl_tile_losses_one

  subroutine sl_split_ensure(nf)
    ! Grow the split accumulators to nf flavours and the current slot's sectors.
    integer,intent(in) :: nf
    integer :: ns,old_f,old_s
    integer,allocatable :: gc(:,:)
    double precision,allocatable :: gb(:),gr(:,:),gh(:)
    ns=size(sl_entries(sl_current)%ids)
    old_f=0
    old_s=0
    if(allocated(sl_split_b))then
       old_f=size(sl_split_b)
       old_s=size(sl_split_h)
    endif
    if(allocated(sl_split_b).and.nf.le.old_f.and.ns.le.old_s)return
    allocate(gb(max(nf,old_f)),gr(max(nf,old_f),max(ns,old_s)),gh(max(ns,old_s)),source=0d0)
    allocate(gc(2,max(ns,old_s)),source=0)
    if(allocated(sl_split_b))then
       gb(1:old_f)=sl_split_b
       gr(1:old_f,1:old_s)=sl_split_r
       gh(1:old_s)=sl_split_h
       gc(:,1:old_s)=sl_split_cell
    endif
    call move_alloc(gb,sl_split_b)
    call move_alloc(gr,sl_split_r)
    call move_alloc(gh,sl_split_h)
    call move_alloc(gc,sl_split_cell)
    if(allocated(sl_split_share))deallocate(sl_split_share)
    allocate(sl_split_share(size(sl_split_h)),source=0d0)
  end subroutine sl_split_ensure

  integer function sl_sector_index(sector)
    ! Position of an outer sector in the current Born slot, or 0.
    ! (A loop, not FINDLOC: NLO code must build with gfortran before 9.)
    integer,intent(in) :: sector
    integer :: i
    sl_sector_index=0
    do i=1,size(sl_entries(sl_current)%ids)
       if(sl_entries(sl_current)%ids(i).eq.sector)then
          sl_sector_index=i
          return
       endif
    enddo
  end function sl_sector_index

  integer function sl_split_index(owner)
    integer,intent(in) :: owner
    sl_split_index=sl_sector_index(owner)
    if(sl_split_index.eq.0)error stop 'Unknown learned sector owner in the grid split'
  end function sl_split_index

  subroutine sector_learning_split_add(owner,canonical_flavour,value)
    ! An S contribution as it is summed into the S event of its flavour;
    ! owner is its outer sector, 0 for the Born part (learning_owner).
    integer,intent(in) :: owner,canonical_flavour
    double precision,intent(in) :: value
    if(.not.sl_split_point)return
    if(sl_current.eq.0.or.owner.lt.0.or.canonical_flavour.lt.1)error stop 'Invalid grid split contribution'
    call sl_split_ensure(canonical_flavour)
    sl_split_abs=sl_split_abs+abs(value)
    if(owner.eq.0)then
       sl_split_b(canonical_flavour)=sl_split_b(canonical_flavour)+value
    else
       sl_split_r(canonical_flavour,sl_split_index(owner))= &
            sl_split_r(canonical_flavour,sl_split_index(owner))+value
    endif
  end subroutine sector_learning_split_add

  subroutine sector_learning_split_add_h(owner,abs_value)
    ! The ABS weight of an H event of the given outer sector.
    integer,intent(in) :: owner
    double precision,intent(in) :: abs_value
    if(.not.sl_split_point)return
    if(sl_current.eq.0.or.owner.lt.0)error stop 'Invalid grid split H event'
    call sl_split_ensure(0)
    sl_split_abs=sl_split_abs+abs_value
    if(owner.eq.0)then
       sl_split_h0=sl_split_h0+abs_value
    else
       sl_split_h(sl_split_index(owner))=sl_split_h(sl_split_index(owner))+abs_value
    endif
  end subroutine sector_learning_split_add_h

  subroutine sector_learning_split_close(f1)
    ! Split the ABS weight f1 of the point: each H event goes to its sector;
    ! each S event |B+sum_s R_s| in proportion to |B| and the |R_s|.
    double precision,intent(in) :: f1
    integer :: j,ns
    double precision :: d,s,total
    sl_split_valid=.false.
    if(.not.sl_split_point.or.sl_current.eq.0)return
    call sl_split_ensure(0)
    ns=size(sl_entries(sl_current)%ids)
    sl_split_share=0d0
    sl_split_share0=sl_split_h0
    sl_split_share(1:ns)=sl_split_h(1:ns)
    do j=1,size(sl_split_b)
       d=abs(sl_split_b(j))+sum(abs(sl_split_r(j,1:ns)))
       if(d.le.0d0)cycle
       s=abs(sl_split_b(j)+sum(sl_split_r(j,1:ns)))
       sl_split_share0=sl_split_share0+s*abs(sl_split_b(j))/d
       sl_split_share(1:ns)=sl_split_share(1:ns)+s*abs(sl_split_r(j,1:ns))/d
    enddo
    total=sl_split_share0+sum(sl_split_share(1:ns))
    if(f1.eq.0d0.and.total.eq.0d0)return
    ! B and the R_s are sums of contributions that can cancel, so the two
    ! summation orders differ by rounding of the size of those terms.
    if(abs(total-f1).le.1d-10*sl_split_abs.and.total.gt.0d0)then
       sl_split_share0=sl_split_share0*f1/total
       sl_split_share(1:ns)=sl_split_share(1:ns)*f1/total
       sl_split_valid=.true.
    else
       sl_split_fallbacks=sl_split_fallbacks+1
       if(sl_split_fallbacks.le.5.or.mod(sl_split_fallbacks,1000).eq.0) &
            write(*,*)'Grid split differs from the ABS weight:',total,f1,sl_split_fallbacks
    endif
  end subroutine sector_learning_split_close

  subroutine sector_learning_register_layout(slot,ids,providers,pdgs)
    integer,intent(in) :: slot,ids(:),providers(:),pdgs(:,:)
    type(sector_learning_entry),allocatable :: grown(:)
    integer :: i,index,previous
    ! Callers register layouts only when sector learning is active.
    if(slot.lt.1.or.size(ids).lt.1.or.size(ids).ne.size(providers).or.size(ids).ne.size(pdgs,1).or. &
         any(ids.lt.1).or.any(providers.lt.1))error stop 'Invalid learned sector layout'
    do i=1,size(ids)
       if(count(ids.eq.ids(i)).ne.1)error stop 'Duplicate learned sector layout identity'
    enddo
    index=0
    if(allocated(sl_entries))then
       do i=1,size(sl_entries)
          if(sl_entries(i)%slot.eq.slot)index=i
       enddo
    endif
    if(index.ne.0)then
       if(size(sl_entries(index)%ids).ne.size(ids).or. &
            size(sl_entries(index)%pdgs,2).ne.size(pdgs,2))error stop 'Learned sector layout dimensions changed'
       if(any(sl_entries(index)%ids.ne.ids).or.any(sl_entries(index)%providers.ne.providers).or. &
            any(sl_entries(index)%pdgs.ne.pdgs))error stop 'Learned sector layout changed'
       return
    endif
    if(sl_ready)error stop 'Production learned sector table is missing a Born layout'
    previous=0
    if(allocated(sl_entries))previous=size(sl_entries)
    allocate(grown(previous+1))
    if(previous.gt.0)grown(1:previous)=sl_entries
    call move_alloc(grown,sl_entries)
    index=previous+1
    sl_entries(index)%slot=slot
    sl_entries(index)%ids=ids
    sl_entries(index)%providers=providers
    sl_entries(index)%pdgs=pdgs
    allocate(sl_entries(index)%groups(size(ids)))
    call sl_canonical_groups(ids,providers,pdgs,sl_entries(index)%groups)
    call sl_unified_ensure()
  end subroutine sector_learning_register_layout

  subroutine sl_canonical_groups(ids,providers,pdgs,groups)
    ! Groups of identical real provider and ordered PDGs, numbered in the
    ! order (PDGs, provider, outer ID); stored with the learned table.
    integer,intent(in) :: ids(:),providers(:),pdgs(:,:)
    integer,intent(out) :: groups(size(ids))
    integer :: order(size(ids)),i,j,k,current,previous,ngroup
    logical :: less
    order=[(i,i=1,size(ids))]
    do i=2,size(ids)
       current=order(i)
       j=i-1
       do while(j.ge.1)
          less=.false.
          do k=1,size(pdgs,2)
             if(pdgs(current,k).eq.pdgs(order(j),k))cycle
             less=pdgs(current,k).lt.pdgs(order(j),k)
             exit
          enddo
          if(all(pdgs(current,:).eq.pdgs(order(j),:)))then
             if(providers(current).ne.providers(order(j)))then
                less=providers(current).lt.providers(order(j))
             else
                less=ids(current).lt.ids(order(j))
             endif
          endif
          if(.not.less)exit
          order(j+1)=order(j)
          j=j-1
       enddo
       order(j+1)=current
    enddo
    previous=0
    ngroup=0
    do i=1,size(ids)
       current=order(i)
       if(previous.eq.0)then
          ngroup=ngroup+1
       elseif(providers(current).ne.providers(previous).or.any(pdgs(current,:).ne.pdgs(previous,:)))then
          ngroup=ngroup+1
       endif
       groups(current)=ngroup
       previous=current
    enddo
  end subroutine sl_canonical_groups

  subroutine sl_append_orbit()
    ! Append the current (unrejected) probe orbit to its slot's bank.
    double precision,allocatable :: grown_b(:,:,:),grown_r(:,:,:,:)
    integer :: entry,nf,ns,old_nf,old_capacity,capacity,number
    entry=sl_point_slot
    if(entry.lt.1)error stop 'Learning orbit did not select a Born slot'
    nf=max(1,sl_point_flavours)
    ns=size(sl_entries(entry)%ids)
    number=sl_entries(entry)%ntrain+1
    old_nf=0
    old_capacity=0
    if(allocated(sl_entries(entry)%b))then
       old_nf=size(sl_entries(entry)%b,1)
       old_capacity=size(sl_entries(entry)%b,3)
    endif
    if(nf.gt.old_nf.or.number.gt.old_capacity)then
       capacity=old_capacity
       if(number.gt.old_capacity)capacity=max(number,max(64,2*old_capacity))
       nf=max(nf,old_nf)
       allocate(grown_b(nf,sl_nodes,capacity),source=0d0)
       allocate(grown_r(nf,sl_nodes,ns,capacity),source=0d0)
       if(old_capacity.gt.0)then
          grown_b(1:old_nf,:,1:number-1)=sl_entries(entry)%b(:,:,1:number-1)
          grown_r(1:old_nf,:,:,1:number-1)=sl_entries(entry)%r(:,:,:,1:number-1)
       endif
       call move_alloc(grown_b,sl_entries(entry)%b)
       call move_alloc(grown_r,sl_entries(entry)%r)
    endif
    if(sl_point_flavours.gt.0)then
       sl_entries(entry)%b(1:sl_point_flavours,:,number)=sl_point_b(1:sl_point_flavours,:)
       sl_entries(entry)%r(1:sl_point_flavours,:,:,number)=sl_point_r(1:sl_point_flavours,:,1:ns)
    endif
    sl_entries(entry)%nflavours=max(sl_entries(entry)%nflavours,max(1,sl_point_flavours))
    sl_entries(entry)%ntrain=number
  end subroutine sl_append_orbit

  subroutine sector_learning_unified_gauge(ids,gauge,qcodes)
    integer,intent(in) :: ids(:)
    integer,intent(out) :: gauge,qcodes(2,size(ids))
    integer :: s,word(2)
    if(size(ids).lt.1)error stop 'Empty unified sector layout'
    gauge=1;qcodes=0
    do s=1,size(ids)
       if(ids(s).lt.1.or.ids(s).gt.2**30)error stop 'Invalid unified outer sector ID'
       if(count(ids.eq.ids(s)).ne.1)error stop 'Duplicate unified outer sector ID'
       call sector_sobol_word(ids(s),word)
       if(any(mod(word,2**28).ne.0))gauge=0
       qcodes(:,s)=word/(2**28)
    enddo
    if(gauge.eq.0)qcodes=0
  end subroutine sector_learning_unified_gauge

  subroutine sector_sobol_word(sector_id,word)
    ! First two unscrambled Sobol coordinates of point sector_id-1, as
    ! 30-bit integers: the sector shift is word/2**30.
    integer,intent(in) :: sector_id
    integer,intent(out) :: word(2)
    integer :: gray,j,direction(2)
    gray=ieor(sector_id-1,ishft(sector_id-1,-1))
    word=0
    direction=2**29
    do j=1,30
       if(btest(gray,j-1))word=ieor(word,direction)
       direction(1)=ishft(direction(1),-1)
       direction(2)=ieor(direction(2),ishft(direction(2),-1))
    enddo
  end subroutine sector_sobol_word

  subroutine sl_unified_translation(codes,gx,gy,maps)
    integer,intent(in) :: codes(:),gx,gy
    integer,intent(out) :: maps(sl_nodes,size(codes))
    integer :: s,node,x,y,dx,dy
    if(.not.sl_valid_capacity(gx,gy))error stop 'Invalid unified phase geometry'
    do s=1,size(codes)
       if(codes(s).lt.0.or.codes(s).ge.gx*gy)error stop 'Invalid unified phase code'
       dx=(codes(s)/gy)*(4/gx);dy=mod(codes(s),gy)*(4/gy)
       do node=1,sl_nodes
          x=(node-1)/4;y=mod(node-1,4)
          maps(node,s)=1+4*mod(x+dx,4)+mod(y+dy,4)
       enddo
    enddo
  end subroutine sl_unified_translation

  subroutine sector_learning_unified_phase(b,r,gx,gy,start,maps,before,after,trace)
    ! Whole-map translations of each sector at capacity gx x gy.
    double precision,intent(in) :: b(:,:,:),r(:,:,:,:)
    integer,intent(in) :: gx,gy,start(:,:)
    integer,intent(out) :: maps(size(start,1),size(start,2))
    double precision,intent(out) :: before,after
    integer,allocatable,intent(out) :: trace(:,:)
    integer :: ns,nt,s,hx,hy
    integer,allocatable :: codes(:),check(:,:)
    double precision,allocatable :: losses(:)
    ns=size(start,2);nt=size(b,3)
    if(nt.lt.1.or.ns.lt.1.or.size(b,1).lt.1.or.size(start,1).ne.sl_nodes.or.size(b,2).ne.sl_nodes.or. &
         size(r,1).ne.size(b,1).or. &
         size(r,2).ne.sl_nodes.or.size(r,3).ne.ns.or.size(r,4).ne.nt) &
         error stop 'Invalid unified phase dimensions'
    if(.not.sl_valid_capacity(gx,gy))error stop 'Invalid unified phase capacity'
    hx=4/gx;hy=4/gy
    allocate(codes(ns),check(sl_nodes,ns),losses(nt))
    do s=1,ns
       if(start(1,s).lt.1.or.start(1,s).gt.sl_nodes)error stop 'Invalid unified phase start'
       if(mod((start(1,s)-1)/4,hx).ne.0.or.mod(mod(start(1,s)-1,4),hy).ne.0) &
            error stop 'Coarse-illegal unified phase start'
       codes(s)=((start(1,s)-1)/4/hx)*gy+mod(start(1,s)-1,4)/hy
    enddo
    call sl_unified_translation(codes,gx,gy,check)
    if(any(check.ne.start))error stop 'Unified phase start is not a legal translation'
    maps=start
    call sl_tile_losses_unchecked(b,r,maps,losses);before=sum(losses)
    call sl_coordinate_descent(b,r,gx,gy,sl_phase_proposer,codes,maps,before,after,trace)
  end subroutine sector_learning_unified_phase

  subroutine sl_phase_proposer(fixed,r,sector,gx,gy,codes,map,loss)
    ! The best translation code of the sector; ties keep the current one.
    double precision,intent(in) :: fixed(:,:,:),r(:,:,:,:),loss
    integer,intent(in) :: sector,gx,gy
    integer,intent(inout) :: codes(:),map(:)
    integer :: c,best,column(sl_nodes,1)
    double precision :: losses(size(fixed,3)),proposed,best_loss
    best=codes(sector);best_loss=loss
    do c=0,gx*gy-1
       if(c.eq.codes(sector))cycle
       call sl_unified_translation([c],gx,gy,column)
       call sl_tile_losses_one(fixed,r,sector,column(:,1),losses);proposed=sum(losses)
       if(proposed.lt.best_loss)then
          best=c;best_loss=proposed
       endif
    enddo
    if(best.ne.codes(sector))then
       codes(sector)=best
       call sl_unified_translation([best],gx,gy,column)
       map=column(:,1)
    endif
  end subroutine sl_phase_proposer

  subroutine sl_coordinate_descent(b,r,gx,gy,proposer,codes,maps,before,after,trace)
    ! Up to three passes over the sectors; each sector moves only to a
    ! strictly lower loss with the others fixed. before is the exact loss of
    ! maps; after is the exact loss of the result, compared bit for bit by
    ! the callers. trace records (pass, sector, map) after every move.
    double precision,intent(in) :: b(:,:,:),r(:,:,:,:),before
    integer,intent(in) :: gx,gy
    procedure(sl_sector_proposer) :: proposer
    integer,intent(inout) :: codes(:),maps(:,:)
    double precision,intent(out) :: after
    integer,allocatable,intent(out) :: trace(:,:)
    double precision,allocatable :: total(:,:,:),fixed(:,:,:),losses(:)
    integer,allocatable :: history(:,:)
    integer :: ns,nt,s,node,pass,steps,old(sl_nodes)
    logical :: changed
    ns=size(r,3);nt=size(b,3)
    allocate(total(size(b,1),sl_nodes,nt),fixed(size(b,1),sl_nodes,nt),losses(nt), &
         history(sl_nodes+2,3*ns))
    steps=0
    ! total = B plus every sector at its current tile; fixed removes s.
    total=b
    do s=1,ns
       do node=1,sl_nodes
          total(:,node,:)=total(:,node,:)+r(:,maps(node,s),s,:)
       enddo
    enddo
    do pass=1,3
       changed=.false.
       do s=1,ns
          do node=1,sl_nodes
             fixed(:,node,:)=total(:,node,:)-r(:,maps(node,s),s,:)
          enddo
          ! Current and trial losses use the same (rounded) fixed sum.
          call sl_tile_losses_one(fixed,r,s,maps(:,s),losses)
          old=maps(:,s)
          call proposer(fixed,r,s,gx,gy,codes,maps(:,s),sum(losses))
          if(any(maps(:,s).ne.old))changed=.true.
          steps=steps+1;history(:,steps)=[pass,s,maps(:,s)]
          do node=1,sl_nodes
             total(:,node,:)=fixed(:,node,:)+r(:,maps(node,s),s,:)
          enddo
       enddo
       if(.not.changed)exit
    enddo
    call sl_tile_losses_unchecked(b,r,maps,losses);after=sum(losses)
    trace=history(:,1:steps)
  end subroutine sl_coordinate_descent

  subroutine sector_learning_unified_fit(b,r,current,gx,gy,qcodes,maps,before,after,selected,attempts)
    double precision,intent(in) :: b(:,:,:),r(:,:,:,:)
    integer,intent(in) :: current(:,:),gx,gy,qcodes(:,:)
    integer,intent(out) :: maps(size(current,1),size(current,2)),selected
    double precision,intent(out) :: before,after
    type(unified_fit_attempt),intent(out) :: attempts(3)
    integer :: ns,nt,a,s,node,prior,ref
    double precision :: ignored
    double precision,allocatable :: losses(:)
    logical :: legal
    ns=size(current,2);nt=size(b,3)
    if(nt.lt.1.or.ns.lt.1.or.size(b,1).lt.1.or.size(current,1).ne.sl_nodes.or.size(b,2).ne.sl_nodes.or. &
         size(r,1).ne.size(b,1).or.size(r,2).ne.sl_nodes.or.size(r,3).ne.ns.or.size(r,4).ne.nt) &
         error stop 'Invalid unified fit dimensions'
    do s=1,ns
       if(.not.sl_is_bijection(current(:,s)))error stop 'Unified current map not bijective'
    enddo
    if(size(qcodes,1).ne.2.or.size(qcodes,2).ne.ns.or.any(qcodes.lt.0).or.any(qcodes.gt.3)) &
         error stop 'Invalid unified gauge codes'
    allocate(losses(nt))
    call sl_tile_losses_unchecked(b,r,current,losses)
    before=sum(losses);after=before;maps=current;selected=0
    do a=1,3
       attempts(a)%kind=a
       legal=.true.
       if(a.eq.3)legal=all(mod(gx*qcodes(1,:),4).eq.0).and.all(mod(gy*qcodes(2,:),4).eq.0)
       if(.not.legal)cycle
       allocate(attempts(a)%initial_map(sl_nodes,ns),attempts(a)%phase_map(sl_nodes,ns),attempts(a)%final_map(sl_nodes,ns))
       if(a.eq.1)then
          attempts(a)%initial_map=current
       elseif(a.eq.2)then
          call sl_unified_translation([(0,s=1,ns)],4,4,attempts(a)%initial_map)
       else
          ! The translation that cancels the static gauge.
          call sl_unified_translation(4*modulo(-qcodes(1,:),4)+modulo(-qcodes(2,:),4),4,4, &
               attempts(a)%initial_map)
       endif
       ref=0
       if(a.eq.3)then
          if(all(attempts(a)%initial_map.eq.attempts(2)%initial_map))ref=2
       endif
       if(ref.ne.0)then
          attempts(a)%status=3;attempts(a)%reuse=ref
          attempts(a)%phase_map=attempts(ref)%phase_map
          attempts(a)%final_map=attempts(ref)%final_map
          attempts(a)%loss_start=attempts(ref)%loss_start
          attempts(a)%loss_phase=attempts(ref)%loss_phase
          attempts(a)%loss_final=attempts(ref)%loss_final
          allocate(attempts(a)%phase_trace(sl_nodes+2,0),attempts(a)%tile_trace(sl_nodes+2,0))
       else
          attempts(a)%status=1
          if(a.eq.1)then
             attempts(a)%phase_map=current
             attempts(a)%loss_start=before;attempts(a)%loss_phase=before
             allocate(attempts(a)%phase_trace(sl_nodes+2,0))
          else
             call sector_learning_unified_phase(b,r,gx,gy,attempts(a)%initial_map, &
                  attempts(a)%phase_map,attempts(a)%loss_start,attempts(a)%loss_phase,attempts(a)%phase_trace)
          endif
          ref=0
          do prior=1,a-1
             if(attempts(prior)%status.eq.0)cycle
             if(all(attempts(a)%phase_map.eq.attempts(prior)%phase_map))then
                ref=prior;exit
             endif
          enddo
          if(ref.ne.0)then
             attempts(a)%status=2;attempts(a)%reuse=ref
             attempts(a)%final_map=attempts(ref)%final_map
             attempts(a)%loss_final=attempts(ref)%loss_final
             allocate(attempts(a)%tile_trace(sl_nodes+2,0))
          else
             call sector_learning_online_fit(b,r,attempts(a)%phase_map,gx,gy,attempts(a)%final_map, &
                  ignored,attempts(a)%loss_final,attempts(a)%tile_trace,attempts(a)%loss_phase)
          endif
       endif
       if(attempts(a)%loss_final.lt.after)then
          maps=attempts(a)%final_map;after=attempts(a)%loss_final;selected=a
       endif
    enddo
  end subroutine sector_learning_unified_fit

  subroutine sl_unified_ensure()
    ! Size the per-slot map, fit and gauge state to the registered slots;
    ! new slots start from identity maps.
    type(unified_slot_entry),allocatable :: grown(:)
    type(online_slot_entry),allocatable :: grown_online(:)
    integer :: n,i,previous,s,node
    if(.not.allocated(sl_entries))return
    n=size(sl_entries);previous=0
    if(allocated(online_slots))previous=size(online_slots)
    if(previous.lt.n)then
       allocate(grown_online(n))
       if(previous.gt.0)grown_online(1:previous)=online_slots
       call move_alloc(grown_online,online_slots)
    endif
    do i=1,n
       if(.not.allocated(sl_entries(i)%permutations))then
          allocate(sl_entries(i)%permutations(sl_nodes,size(sl_entries(i)%ids)))
          do s=1,size(sl_entries(i)%ids)
             sl_entries(i)%permutations(:,s)=[(node,node=1,sl_nodes)]
          enddo
       endif
       if(.not.allocated(online_slots(i)%before)) &
            online_slots(i)%before=sl_entries(i)%permutations
    enddo
    previous=0
    if(allocated(unified_slots))previous=size(unified_slots)
    if(previous.lt.n)then
       allocate(grown(n))
       if(previous.gt.0)grown(1:previous)=unified_slots
       call move_alloc(grown,unified_slots)
    endif
    do i=1,n
       if(allocated(unified_slots(i)%qcodes))cycle
       allocate(unified_slots(i)%qcodes(2,size(sl_entries(i)%ids)))
       call sector_learning_unified_gauge(sl_entries(i)%ids,unified_slots(i)%gauge,unified_slots(i)%qcodes)
    enddo
  end subroutine sl_unified_ensure

  subroutine sl_unified_write_attempt(unit,kind,a)
    integer,intent(in) :: unit,kind
    type(unified_fit_attempt),intent(in) :: a
    integer :: np,nt
    np=0;nt=0
    if(allocated(a%phase_trace))np=size(a%phase_trace,2)
    if(allocated(a%tile_trace))nt=size(a%tile_trace,2)
    write(unit)kind,a%status,a%reuse,np,nt
    write(unit)a%loss_start,a%loss_phase,a%loss_final
    if(a%status.gt.0)write(unit)a%initial_map,a%phase_map,a%final_map
    if(np.gt.0)write(unit)a%phase_trace
    if(nt.gt.0)write(unit)a%tile_trace
  end subroutine sl_unified_write_attempt

  subroutine sl_unified_write_extension(unit,i)
    integer,intent(in) :: unit,i
    integer :: a
    write(unit)unified_slots(i)%fit_status,unified_slots(i)%gauge, &
         unified_slots(i)%selected_orientation,unified_slots(i)%primary_gx,unified_slots(i)%primary_gy, &
         unified_slots(i)%primary_selected,unified_slots(i)%alternative_gx,unified_slots(i)%alternative_gy, &
         unified_slots(i)%alternative_selected
    write(unit)unified_slots(i)%qcodes
    do a=1,3
       call sl_unified_write_attempt(unit,a,unified_slots(i)%primary(a))
    enddo
    do a=1,3
       call sl_unified_write_attempt(unit,a,unified_slots(i)%alternative(a))
    enddo
  end subroutine sl_unified_write_extension

  subroutine sl_unified_check_groups(ids,providers,pdgs,groups)
    integer,intent(in) :: ids(:),providers(:),pdgs(:,:),groups(:)
    integer :: expected(size(ids))
    if(size(providers).ne.size(ids).or.size(pdgs,1).ne.size(ids).or.size(groups).ne.size(ids)) &
         error stop 'Invalid unified canonical group dimensions'
    call sl_canonical_groups(ids,providers,pdgs,expected)
    if(any(expected.ne.groups))error stop 'Saved unified canonical groups differ'
  end subroutine sl_unified_check_groups

  subroutine sector_learning_epoch_replay_bank(b,r,previous_b,previous_r,fit_b,fit_r)
    use,intrinsic :: ieee_arithmetic,only:ieee_is_finite
    double precision,intent(in) :: b(:,:,:),r(:,:,:,:),previous_b(:,:,:),previous_r(:,:,:,:)
    double precision,allocatable,intent(out) :: fit_b(:,:,:),fit_r(:,:,:,:)
    integer :: nt,nprevious,nf,ns
    double precision :: scale
    nt=size(b,3);nprevious=size(previous_b,3);ns=size(r,3)
    if(size(b,2).ne.sl_nodes.or.size(r,2).ne.sl_nodes.or.size(r,1).ne.size(b,1).or.size(r,4).ne.nt.or. &
         size(previous_b,2).ne.sl_nodes.or.size(previous_r,2).ne.sl_nodes.or. &
         size(previous_r,1).ne.size(previous_b,1).or.size(previous_r,3).ne.ns.or. &
         size(previous_r,4).ne.nprevious.or.ns.lt.1)error stop 'Invalid epoch replay bank shapes'
    if((nt.gt.0.and.size(b,1).lt.1).or.(nprevious.gt.0.and.size(previous_b,1).lt.1)) &
         error stop 'Missing canonical flavour support in epoch replay'
    if(any(.not.ieee_is_finite(b)).or.any(.not.ieee_is_finite(r)).or. &
         any(.not.ieee_is_finite(previous_b)).or.any(.not.ieee_is_finite(previous_r))) &
         error stop 'Nonfinite epoch replay bank'
    if(nt.eq.0.or.nprevious.eq.0)then
       ! Without history the fit uses the current bank's arrays unchanged.
       fit_b=b;fit_r=r
       return
    endif
    nf=max(size(b,1),size(previous_b,1))
    allocate(fit_b(nf,sl_nodes,nt+nprevious),fit_r(nf,sl_nodes,ns,nt+nprevious))
    fit_b=0d0;fit_r=0d0
    fit_b(1:size(b,1),:,1:nt)=b;fit_r(1:size(r,1),:,:,1:nt)=r
    scale=.5d0*dble(nt)/dble(nprevious)
    fit_b(1:size(previous_b,1),:,nt+1:)=scale*previous_b
    fit_r(1:size(previous_r,1),:,:,nt+1:)=scale*previous_r
    if(any(.not.ieee_is_finite(fit_b)).or.any(.not.ieee_is_finite(fit_r))) &
         error stop 'Nonfinite scaled epoch replay bank'
  end subroutine sector_learning_epoch_replay_bank

  subroutine sl_epoch_replay_build_bank(entry,b,r)
    use,intrinsic :: ieee_arithmetic,only:ieee_is_finite
    integer,intent(in) :: entry
    double precision,allocatable,intent(out) :: b(:,:,:),r(:,:,:,:)
    integer :: nf,nt,ns
    logical :: available
    nf=sl_entries(entry)%nflavours;nt=sl_entries(entry)%ntrain;ns=size(sl_entries(entry)%ids)
    available=.false.
    if(allocated(epoch_replay_previous))then
       if(entry.le.size(epoch_replay_previous))available=epoch_replay_previous(entry)%nt.gt.0
    endif
    if(.not.available)then
       b=sl_entries(entry)%b(1:nf,:,1:nt);r=sl_entries(entry)%r(1:nf,:,:,1:nt)
       ! The fits rely on this single check.
       if(.not.all(ieee_is_finite(b)).or..not.all(ieee_is_finite(r)))error stop 'Nonfinite epoch replay bank'
       return
    endif
    associate(previous=>epoch_replay_previous(entry))
      call sector_learning_epoch_replay_bank(sl_entries(entry)%b(1:nf,:,1:nt), &
           sl_entries(entry)%r(1:nf,:,:,1:nt),previous%b,previous%r,b,r)
    end associate
  end subroutine sl_epoch_replay_build_bank

  subroutine sl_epoch_replay_write_table(unit)
    integer,intent(in) :: unit
    integer :: i,s,c,d
    if(.not.sl_ready.or..not.allocated(sl_grid_snapshot))error stop 'Missing frozen online state'
    call sl_unified_ensure()
    ! Explicit formats keep one record per line (the reader counts tokens).
    write(unit,'(a,*(1x,i0))')'SCL10 H',sl_format_version,sl_method_id,size(sl_entries),sl_grid_nint,sl_grid_ndim,sl_grid_nchans, &
         online_epoch,online_ordinary_calls,online_probes,online_record_rejections, &
         64,16,1,3,3,1
    write(unit,'(a,*(1x,es24.16))')'SCL10 C',online_budget,online_ordinary_cpu,online_probe_cpu,online_fit_cpu,online_write_cpu
    do i=1,size(sl_entries)
       write(unit,'(a,*(1x,i0))')'SCL10 L',sl_entries(i)%slot,size(sl_entries(i)%ids),size(sl_entries(i)%pdgs,2), &
            sl_entries(i)%nflavours,online_slots(i)%gx,online_slots(i)%gy,online_slots(i)%total_samples,unified_slots(i)%gauge
       do s=1,size(sl_entries(i)%ids)
          write(unit,'(a,*(1x,i0))')'SCL10 R',sl_entries(i)%slot,s,sl_entries(i)%ids(s),sl_entries(i)%providers(s), &
               sl_entries(i)%groups(s),sl_entries(i)%pdgs(s,:),unified_slots(i)%qcodes(:,s)
          write(unit,'(a,*(1x,i0))')'SCL10 P',sl_entries(i)%slot,s,sl_entries(i)%permutations(:,s)
       enddo
    enddo
    do c=1,sl_grid_nchans
       do d=1,sl_grid_ndim
          write(unit,'(a,3(1x,i0),*(1x,es24.16))')'SCL10 G',c,d,sl_grid_nint,sl_grid_snapshot(:,d,c)
       enddo
    enddo
  end subroutine sl_epoch_replay_write_table

  subroutine sl_epoch_replay_validate_line(line,expected_tag,columns)
    ! Token count, marker and tag of one SCL10 record; callers check payloads.
    character(*),intent(in) :: line,expected_tag
    integer,intent(in) :: columns
    character(len=64) :: marker,tag
    integer :: i,words,ios
    logical :: active
    words=0;active=.false.
    do i=1,len_trim(line)
       if(line(i:i).eq.' '.or.line(i:i).eq.achar(9))then
          active=.false.
       else
          if(.not.active)words=words+1
          active=.true.
       endif
    enddo
    if(words.ne.columns.or.index(line,',').ne.0)error stop 'Malformed SCL10 token count'
    read(line,*,iostat=ios)marker,tag
    if(ios.ne.0.or.marker.ne.'SCL10'.or.tag.ne.expected_tag)error stop 'Malformed SCL10 marker/tag'
  end subroutine sl_epoch_replay_validate_line

  logical function sl_map_representable(map,gx,gy) result(ok)
    ! True if the 16-tile map moves whole (4/gx)x(4/gy) blocks rigidly.
    integer,intent(in) :: map(sl_nodes),gx,gy
    integer :: coarse(sl_nodes),fine(sl_nodes),k,node,t,hx,hy
    hx=4/gx;hy=4/gy
    do k=1,gx*gy
       node=1+4*((k-1)/gy)*hx+mod(k-1,gy)*hy
       t=map(node)-1
       coarse(k)=1+gy*((t/4)/hx)+mod(t,4)/hy
    enddo
    call sl_online_lift(coarse(1:gx*gy),gx,gy,fine)
    ok=all(fine.eq.map)
  end function sl_map_representable

  subroutine sl_online_lift(coarse,gx,gy,fine)
    integer,intent(in) :: gx,gy,coarse(gx*gy)
    integer,intent(out) :: fine(sl_nodes)
    integer :: node,x,y,c,target,tx,ty,hx,hy
    hx=4/gx;hy=4/gy
    do node=1,sl_nodes
       x=(node-1)/4;y=mod(node-1,4);c=1+gy*(x/hx)+y/hy
       target=coarse(c)-1;tx=target/gy;ty=mod(target,gy)
       fine(node)=1+4*(tx*hx+mod(x,hx))+ty*hy+mod(y,hy)
    enddo
  end subroutine sl_online_lift

  subroutine sector_learning_online_fit(b,r,current,gx,gy,maps,before,after,trace,known_before)
    ! Optimal coarse tile assignments of each sector at capacity gx x gy.
    ! known_before, if present, is the exact loss of current.
    double precision,intent(in) :: b(:,:,:),r(:,:,:,:)
    integer,intent(in) :: current(:,:),gx,gy
    integer,intent(out) :: maps(size(current,1),size(current,2))
    double precision,intent(out) :: before,after
    integer,allocatable,intent(out) :: trace(:,:)
    double precision,intent(in),optional :: known_before
    double precision,allocatable :: losses(:)
    integer :: s,codes(size(current,2))
    if(.not.sl_valid_capacity(gx,gy))error stop 'Invalid online tile capacity'
    if(size(current,1).ne.sl_nodes.or.size(b,2).ne.sl_nodes.or.size(r,2).ne.sl_nodes.or. &
         size(r,1).ne.size(b,1).or.size(r,4).ne.size(b,3).or.size(current,2).ne.size(r,3)) &
         error stop 'Invalid online tile dimensions'
    do s=1,size(current,2)
       if(.not.sl_is_bijection(current(:,s)))error stop 'Online map is not bijective'
       if(.not.sl_map_representable(current(:,s),gx,gy)) &
            error stop 'Current map not representable at requested capacity'
    enddo
    maps=current;codes=0
    if(present(known_before))then
       before=known_before
    else
       allocate(losses(size(b,3)))
       call sl_tile_losses_unchecked(b,r,maps,losses);before=sum(losses)
    endif
    call sl_coordinate_descent(b,r,gx,gy,sl_tile_proposer,codes,maps,before,after,trace)
  end subroutine sector_learning_online_fit

  subroutine sl_tile_proposer(fixed,r,sector,gx,gy,codes,map,loss)
    ! The optimal assignment of the sector's coarse tiles, lifted to nodes.
    double precision,intent(in) :: fixed(:,:,:),r(:,:,:,:),loss
    integer,intent(in) :: sector,gx,gy
    integer,intent(inout) :: codes(:),map(:)
    double precision :: cost(gx*gy,gx*gy),losses(size(fixed,3))
    integer :: coarse(gx*gy),fine(sl_nodes),k,hx,hy,dx,dy,i,j,orbit,flavour,source_node,dest_node
    k=gx*gy;hx=4/gx;hy=4/gy
    cost=0d0
    do dx=0,hx-1
       do dy=0,hy-1
          do i=1,k
             dest_node=1+4*((i-1)/gy*hx+dx)+mod(i-1,gy)*hy+dy
             do j=1,k
                source_node=1+4*((j-1)/gy*hx+dx)+mod(j-1,gy)*hy+dy
                do orbit=1,size(fixed,3)
                   do flavour=1,size(fixed,1)
                      cost(i,j)=cost(i,j)+max(0d0,-(fixed(flavour,dest_node,orbit)+ &
                           r(flavour,source_node,sector,orbit)))
                   enddo
                enddo
             enddo
          enddo
       enddo
    enddo
    call sector_learning_assignment(cost,coarse)
    call sl_online_lift(coarse,gx,gy,fine)
    call sl_tile_losses_one(fixed,r,sector,fine,losses)
    if(sum(losses).lt.loss)map=fine
  end subroutine sl_tile_proposer

end module sector_learning
