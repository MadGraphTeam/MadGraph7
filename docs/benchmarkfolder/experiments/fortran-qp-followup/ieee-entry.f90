
program probe
use, intrinsic :: ieee_exceptions
implicit none
logical :: before, batch_value, helper_value, after, flags(4)
call ieee_set_halting_mode(ieee_inexact,.false.)
call ieee_set_flag(ieee_all,.false.)
call ieee_set_halting_mode(ieee_inexact,.true.)
call ieee_get_halting_mode(ieee_inexact,before)
call batch(batch_value,helper_value)
call ieee_get_halting_mode(ieee_inexact,after)
call ieee_set_halting_mode(ieee_inexact,.false.)
if (.not.(before.and.batch_value.and.helper_value.and.after)) stop 1
write(*,*) 'HALTING_PRESERVED',before,batch_value,helper_value,after
call ieee_set_flag(ieee_inexact,.true.)
call ieee_get_flag(ieee_inexact,flags(1))
call batch_flags(flags(2),flags(3))
call ieee_get_flag(ieee_inexact,flags(4))
if (.not.flags(1).or..not.flags(4)) stop 2
write(*,*) 'STICKY_FLAGS',flags
end program
subroutine batch(b,h)
use, intrinsic :: ieee_exceptions
implicit none
logical :: b,h
call ieee_get_halting_mode(ieee_inexact,b)
call helper(h)
end subroutine
subroutine helper(h)
use, intrinsic :: ieee_exceptions
implicit none
logical :: h
call ieee_get_halting_mode(ieee_inexact,h)
end subroutine
subroutine batch_flags(b,h)
use, intrinsic :: ieee_exceptions
implicit none
logical :: b,h
call ieee_get_flag(ieee_inexact,b)
call helper_flags(h)
end subroutine
subroutine helper_flags(h)
use, intrinsic :: ieee_exceptions
implicit none
logical :: h
call ieee_get_flag(ieee_inexact,h)
end subroutine
