module qg_observer
  use, intrinsic :: iso_c_binding
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  implicit none
  integer, parameter :: nexternal=5, ncoef=70, ngroup=116, nborn=8, ncomb=48, model_words=254
  real(8) :: dg(2),ds(2),dm(6),dw(4),loop_scale
  complex(8) :: dc(34),dgweak(2)
  real(16) :: qg,qm(6),qw(4)
  complex(16) :: qs,qc(34),qweak(2),qamp(8),qco(0:69,1,116)
  common /strong/ dg
  common /rscale/ ds
  common /masses/ dm
  common /widths/ dw
  common /couplings/ dc
  common /weak/ dgweak
  common /mp_strong/ qg
  common /mp_rscale/ qs
  common /mp_masses/ qm
  common /mp_widths/ qw
  common /mp_couplings/ qc
  common /mp_weak/ qweak
  common /ml5_0_mp_amps/ qamp
  common /ml5_0_mp_lcoefs/ qco
  complex(8) :: loopres(3,1,116)
  logical :: stable_groups(1,116)
  common /ml5_0_loopres/ loopres,stable_groups
  integer :: group_id,so_id,loop_rank,ctmode,libindex
  common /ml5_0_loop/ group_id,so_id,loop_rank
  common /ml5_0_ct/ loop_scale,ctmode
  common /ml5_0_i_lib/ libindex
  interface
    function qg_setup(observed) bind(C) result(code)
      import
      integer(c_int), value :: observed
      integer(c_int) :: code
    end function
    subroutine qg_call(id) bind(C)
      import
      integer(c_int), value :: id
    end subroutine
    function qg_enabled() bind(C) result(take)
      import
      integer(c_int) :: take
    end function
    function qg_health() bind(C) result(code)
      import
      integer(c_int) :: code
    end function
    function qg_flush() bind(C) result(code)
      import
      integer(c_int) :: code
    end function
    subroutine qg_record(kind,part,p,n) bind(C)
      import
      integer(c_int), value :: kind,part
      type(c_ptr), value :: p
      integer(c_size_t), value :: n
    end subroutine
  end interface
contains
  subroutine state(words)
    integer(c_int64_t), intent(out) :: words(11)
    type(ieee_flag_type) :: flags(5)
    type(ieee_round_type) :: rounding
    logical :: value
    integer :: i
    flags=[ieee_invalid,ieee_divide_by_zero,ieee_overflow,ieee_underflow,ieee_inexact]
    do i=1,5
      call ieee_get_flag(flags(i),value); words(i)=merge(1_c_int64_t,0_c_int64_t,value)
      call ieee_get_halting_mode(flags(i),value); words(i+5)=merge(1_c_int64_t,0_c_int64_t,value)
    enddo
    call ieee_get_rounding_mode(rounding)
    words(11)=-1
    if(rounding==ieee_nearest) words(11)=0
    if(rounding==ieee_to_zero) words(11)=1
    if(rounding==ieee_up) words(11)=2
    if(rounding==ieee_down) words(11)=3
  end subroutine

  subroutine model(words)
    integer(c_int64_t), intent(out) :: words(model_words)
    words=[transfer(dm,[0_c_int64_t],6),transfer(dw,[0_c_int64_t],4), &
      transfer(dg(1),0_c_int64_t),transfer(ds(1),0_c_int64_t), &
      transfer(dgweak,[0_c_int64_t],4),transfer(dc,[0_c_int64_t],68), &
      transfer(qm,[0_c_int64_t],12),transfer(qw,[0_c_int64_t],8), &
      transfer(qg,[0_c_int64_t],2),transfer(qs,[0_c_int64_t],4), &
      transfer(qweak,[0_c_int64_t],8),transfer(qc,[0_c_int64_t],136)]
  end subroutine

  subroutine emit(kind,part,payload,always,lean)
    integer, intent(in) :: kind,part
    integer(c_int64_t), intent(in) :: payload(:)
    logical, optional, intent(in) :: always,lean
    integer(c_int64_t), allocatable, target :: words(:)
    integer(c_int64_t) :: mb(model_words),flags(11)
    logical :: take,short
    take=qg_enabled()/=0; short=.false.
    if(present(always)) take=take.or.always
    if(present(lean)) short=lean
    if(.not.take) return
    if(short) then
      words=payload
    else
      call state(flags); call model(mb)
      words=[payload,mb,flags]
    endif
    call qg_record(int(kind,c_int),int(part,c_int),c_loc(words),int(size(words),c_size_t)*8_c_size_t)
  end subroutine

  subroutine native_state() bind(C,name='qg_native_state')
    if(qg_enabled()==0) return
    if(group_id<1.or.group_id>ngroup.or.so_id/=1.or.loop_rank<0.or.loop_rank>4) then
      call emit(110,9,[int([group_id,so_id,loop_rank],c_int64_t)])
      return
    endif
    call emit(110,0,transfer(qco(:,1,group_id),[0_c_int64_t],4*ncoef))
  end subroutine

  subroutine native_ids(id,so,r,mode) bind(C,name='qg_state')
    integer(c_int), intent(out) :: id,so,r,mode
    id=group_id; so=so_id; r=loop_rank; mode=ctmode
  end subroutine

  subroutine wrap_born(p,hel,flavor,ans) bind(C,name='__wrap_ml5_0_smatrixhel_splitorders_')
    real(8) :: p(0:3,5),ans(0:1)
    integer(c_int) :: hel,flavor
    interface
      subroutine original(p,hel,flavor,ans) bind(C,name='__real_ml5_0_smatrixhel_splitorders_')
        import c_int
        real(8) :: p(0:3,5),ans(0:1)
        integer(c_int) :: hel,flavor
      end subroutine
    end interface
    call emit(10,0,[transfer(p,[0_c_int64_t],20),int([hel,flavor],c_int64_t)])
    call original(p,hel,flavor,ans)
    call emit(10,1,[transfer(p,[0_c_int64_t],20),int([hel,flavor],c_int64_t),transfer(ans,[0_c_int64_t],2)])
  end subroutine

  subroutine wrap_dp_repair(keep,p) bind(C,name='__wrap_ml5_0_improve_ps_point_precision_')
    integer(c_int) :: keep(5)
    real(8) :: p(0:3,5)
    interface
      subroutine original(keep,p) bind(C,name='__real_ml5_0_improve_ps_point_precision_')
        import c_int
        integer(c_int) :: keep(5)
        real(8) :: p(0:3,5)
      end subroutine
    end interface
    call emit(20,0,[transfer(p,[0_c_int64_t],20),int(keep,c_int64_t)])
    call original(keep,p)
    call emit(20,1,[transfer(p,[0_c_int64_t],20),int(keep,c_int64_t)])
  end subroutine

  subroutine wrap_qp_repair(keep,p) bind(C,name='__wrap_ml5_0_mp_improve_ps_point_precision_')
    integer(c_int) :: keep(5)
    real(16) :: p(0:3,5)
    interface
      subroutine original(keep,p) bind(C,name='__real_ml5_0_mp_improve_ps_point_precision_')
        import c_int
        integer(c_int) :: keep(5)
        real(16) :: p(0:3,5)
      end subroutine
    end interface
    call emit(21,0,[transfer(p,[0_c_int64_t],40),int(keep,c_int64_t)])
    call original(keep,p)
    call emit(21,1,[transfer(p,[0_c_int64_t],40),int(keep,c_int64_t)])
  end subroutine

  subroutine wrap_compute(p,ans) bind(C,name='__wrap_ml5_0_mp_compute_loop_coefs_')
    real(16) :: p(0:3,5)
    real(8) :: ans(3,0:1)
    interface
      subroutine original(p,ans) bind(C,name='__real_ml5_0_mp_compute_loop_coefs_')
        real(16) :: p(0:3,5)
        real(8) :: ans(3,0:1)
      end subroutine
    end interface
    call emit(30,0,[transfer(p,[0_c_int64_t],40),int([ctmode,libindex],c_int64_t)])
    call original(p,ans)
    call emit(30,1,[transfer(p,[0_c_int64_t],40),int([ctmode,libindex],c_int64_t),transfer(ans,[0_c_int64_t],6)])
    call emit(31,0,transfer(qco,[0_c_int64_t],4*ncoef*ngroup))
  end subroutine

  subroutine wrap_update() bind(C,name='__wrap_mp_update_as_param_')
    interface
      subroutine original() bind(C,name='__real_mp_update_as_param_')
      end subroutine
    end interface
    call emit(32,0,[integer(c_int64_t)::])
    call original()
    call emit(32,1,[integer(c_int64_t)::])
  end subroutine

  subroutine wrap_amp(p,nhel,h,ic) bind(C,name='__wrap_ml5_0_mp_helas_calls_ampb_1_')
    real(16) :: p(0:3,5)
    integer(c_int) :: nhel(5),h,ic(5)
    interface
      subroutine original(p,nhel,h,ic) bind(C,name='__real_ml5_0_mp_helas_calls_ampb_1_')
        import c_int
        real(16) :: p(0:3,5)
        integer(c_int) :: nhel(5),h,ic(5)
      end subroutine
    end interface
    call emit(34,0,[transfer(p,[0_c_int64_t],40),int(nhel,c_int64_t),int(h,c_int64_t),int(ic,c_int64_t)])
    call original(p,nhel,h,ic)
    call emit(34,1,[transfer(p,[0_c_int64_t],40),int(nhel,c_int64_t),int(h,c_int64_t),int(ic,c_int64_t), &
      transfer(qamp,[0_c_int64_t],4*nborn)])
  end subroutine

  subroutine wrap_coef(p,nhel,h,ic) bind(C,name='__wrap_ml5_0_mp_coef_construction_1_')
    real(16) :: p(0:3,5)
    integer(c_int) :: nhel(5),h,ic(5)
    interface
      subroutine original(p,nhel,h,ic) bind(C,name='__real_ml5_0_mp_coef_construction_1_')
        import c_int
        real(16) :: p(0:3,5)
        integer(c_int) :: nhel(5),h,ic(5)
      end subroutine
    end interface
    call emit(36,0,[transfer(p,[0_c_int64_t],40),int(nhel,c_int64_t),int(h,c_int64_t),int(ic,c_int64_t)])
    call original(p,nhel,h,ic)
    call emit(36,1,[transfer(p,[0_c_int64_t],40),int(nhel,c_int64_t),int(h,c_int64_t),int(ic,c_int64_t)])
  end subroutine

  subroutine wrap_mpnum(q,res) bind(C,name='__wrap_ml5_0_mploopnum_')
    complex(16) :: q(0:3),res
    interface
      subroutine original(q,res) bind(C,name='__real_ml5_0_mploopnum_')
        complex(16) :: q(0:3),res
      end subroutine
    end interface
    call emit(130,0,transfer(q,[0_c_int64_t],16),lean=.true.)
    call original(q,res)
    call emit(130,1,[transfer(q,[0_c_int64_t],16),transfer(res,[0_c_int64_t],4)],lean=.true.)
  end subroutine
  subroutine wrap_order(mode,n,pl,pden,mass) bind(C,name='__wrap_ml5_0_mp_switch_order_')
    integer(c_int) :: mode,n
    real(16) :: pl(0:3,n),pden(0:3,n-1)
    complex(16) :: mass(n)
    interface
      subroutine original(mode,n,pl,pden,mass) bind(C,name='__real_ml5_0_mp_switch_order_')
        import c_int
        integer(c_int) :: mode,n
        real(16) :: pl(0:3,n),pden(0:3,n-1)
        complex(16) :: mass(n)
      end subroutine
    end interface
    ! PDEN is a dummy in some callers: pass it through without reading it.
    call emit(140,0,[int([mode,n],c_int64_t),transfer(pl,[0_c_int64_t],8*n), &
      transfer(mass,[0_c_int64_t],4*n)],lean=.true.)
    call original(mode,n,pl,pden,mass)
    call emit(140,1,[int([mode,n],c_int64_t),transfer(pl,[0_c_int64_t],8*n), &
      transfer(mass,[0_c_int64_t],4*n)],lean=.true.)
  end subroutine

  subroutine wrap_reductions(p,nhel,h,ic) bind(C,name='__wrap_ml5_0_loop_ct_calls_1_')
    real(8) :: p(0:3,5)
    integer(c_int) :: nhel(5),h,ic(5)
    interface
      subroutine original(p,nhel,h,ic) bind(C,name='__real_ml5_0_loop_ct_calls_1_')
        import c_int
        real(8) :: p(0:3,5)
        integer(c_int) :: nhel(5),h,ic(5)
      end subroutine
    end interface
    call emit(40,0,[transfer(p,[0_c_int64_t],20),int(nhel,c_int64_t),int(h,c_int64_t),int(ic,c_int64_t)])
    call original(p,nhel,h,ic)
    call emit(40,1,[transfer(p,[0_c_int64_t],20),int(nhel,c_int64_t),int(h,c_int64_t),int(ic,c_int64_t), &
      transfer(loopres,[0_c_int64_t],6*ngroup), &
      reshape(merge(1_c_int64_t,0_c_int64_t,stable_groups),[ngroup])])
  end subroutine
end module
