function ninjars_abi_version() bind(c) result(version)
  use iso_c_binding
  integer(c_int) :: version
  version=2
end function
