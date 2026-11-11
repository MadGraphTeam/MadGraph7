! SOURCE ONLY. Exact existing binding/predicate; no process tables.
module ML5_0_INVARIANT_BOUNDS
  use,intrinsic::iso_c_binding,only:c_int64_t
  implicit none
  private
  public::read_kinematic_environment_token,environment_ok
  interface
    function read_kinematic_environment_token() result(token) &
        bind(C,name='ML5_0_kinematic_environment_token')
      import :: c_int64_t
      integer(c_int64_t) :: token
    end function
  end interface
contains
  logical function environment_ok(token) result(ok)
    integer(c_int64_t), intent(in) :: token
    integer(c_int64_t) :: csr,cw
    ok=.false.
    if(iand(token,int(z'FFFFFFFF00000000',c_int64_t))/= &
       int(z'4B45000000000000',c_int64_t)) return
    csr=iand(ishft(token,-16),int(z'FFFF',c_int64_t))
    cw=iand(token,int(z'FFFF',c_int64_t))
    if(iand(csr,int(z'003F',c_int64_t))/=0) return
    if(iand(cw,int(z'E0C0',c_int64_t))/=0) return
    if(iand(csr,int(z'E040',c_int64_t))/=0) return ! nearest, no FTZ/DAZ
    if(iand(cw,int(z'0C00',c_int64_t))/=0) return  ! x87 nearest
    if(iand(cw,int(z'0300',c_int64_t))/=int(z'0300',c_int64_t)) return
    ok=.true.
  end function
end module
