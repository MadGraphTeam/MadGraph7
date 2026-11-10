! Private integer/byte-only hooks. No native software or IEEE flag reset.
subroutine ml_tirust_capture_init(which)
  use iso_c_binding
  implicit none
  integer,intent(in)::which
  interface
    subroutine c_init(n) bind(C,name='ml_tcap_init')
      import c_int
      integer(c_int),value::n
    end subroutine
  end interface
  call c_init(int(which,c_int))
end subroutine

subroutine ml_tirust_capture_point(phase,call_id,point,method,mode)
  use iso_c_binding
  implicit none
  integer,intent(in)::phase,call_id,point,method,mode
  interface
    subroutine c_point(a,b,c,d,e) bind(C,name='ml_tcap_point')
      import c_int
      integer(c_int),value::a,b,c,d,e
    end subroutine
  end interface
  call c_point(int(phase,c_int),int(call_id,c_int),int(point,c_int),int(method,c_int),int(mode,c_int))
end subroutine

subroutine ml_tirust_capture_group(order,group,mode,n,rank,pl,masses)
  use iso_c_binding
  implicit none
  integer,intent(in)::order,group,mode,n,rank
  real(c_double),intent(in)::pl(0:3,n)
  complex(c_double_complex),intent(in)::masses(n)
  interface
    subroutine c_group(a,b,c,d,e,p,m) bind(C,name='ml_tcap_group')
      import c_int,c_double,c_double_complex
      integer(c_int),value::a,b,c,d,e
      real(c_double),intent(in)::p(*)
      complex(c_double_complex),intent(in)::m(*)
    end subroutine
  end interface
  call c_group(int(order,c_int),int(group,c_int),int(mode,c_int),int(n,c_int),int(rank,c_int),pl,masses)
end subroutine

subroutine ml_tirust_capture_stage(stage,detail)
  use iso_c_binding
  implicit none
  integer,intent(in)::stage,detail
  interface
    subroutine c_stage(a,b) bind(C,name='ml_tcap_stage')
      import c_int
      integer(c_int),value::a,b
    end subroutine
  end interface
  call c_stage(int(stage,c_int),int(detail,c_int))
end subroutine

subroutine ml_tirust_capture_status(stage,status)
  use iso_c_binding
  use mtirust,only:tirust_error_message
  implicit none
  integer,intent(in)::stage,status
  character(len=512)::message
  character(kind=c_char)::bytes(512)
  integer::i,n
  interface
    subroutine c_status(a,b,s,n) bind(C,name='ml_tcap_status')
      import c_int,c_char
      integer(c_int),value::a,b,n
      character(kind=c_char),intent(in)::s(*)
    end subroutine
  end interface
  message=''
  if(status/=0) call tirust_error_message(message)
  n=len_trim(message)
  do i=1,512
    bytes(i)=message(i:i)
  enddo
  call c_status(int(stage,c_int),int(status,c_int),bytes,int(n,c_int))
end subroutine

subroutine ml_tirust_capture_end(stable)
  use iso_c_binding
  implicit none
  logical,intent(in)::stable
  interface
    subroutine c_end(n) bind(C,name='ml_tcap_end')
      import c_int
      integer(c_int),value::n
    end subroutine
  end interface
  call c_end(merge(1_c_int,0_c_int,stable))
end subroutine
