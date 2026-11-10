! Private bounded classification evidence. No process-global observer state.
module cms_scalar_observer
  implicit none
  private
  integer, parameter, public :: CMS_OK=0, CMS_REFUSED=1, CMS_UNINITIALIZED=2
  integer, parameter, public :: CMS_DEFAULT_UNSUPPORTED=3, CMS_ZERO_TOPOLOGY_UNSUPPORTED=4
  integer, parameter, public :: CMS_EVENT_OVERFLOW=5, CMS_BAD_INPUT=6, CMS_OVERLOAD_UNSUPPORTED=7
  integer, parameter, public :: CMS_NONFINITE_RESULT=8
  integer, parameter, public :: CMS_MAX_EVENTS=32
  integer, parameter, public :: CMS_INPUT=1, CMS_MASS=2, CMS_DIFFERENCE=3
  integer, parameter, public :: CMS_ABS_REAL=1, CMS_ABS_COMPLEX=2, CMS_L1_COMPLEX=3
  type, public :: cms_scalar_event
    integer :: precision=0, topology=0, overload=0, source_line=0
    integer :: metric_kind=0, metric_norm=0, p_index=0, mass_index=0
    character(24) :: metric_name=''
    logical :: metric_underflowed=.false.
    real(16) :: native_metric=0, threshold=0, local_scale=0
    complex(16) :: original_value=(0,0), classified_value=(0,0)
    complex(16) :: original_invariant=(0,0), original_mass=(0,0)
    logical :: operands_recorded=.false.
  end type
  type, public :: cms_scalar_report
    integer :: status=CMS_OK, precision=0, topology=0, overload=0
    integer :: direction=0, subtopology=0, classification_case=0, soft_permutation=1
    integer :: count=0, dropped=0, p_sources(6)=0, mass_sources(4)=0
    logical :: result_valid=.false., native_called=.false.
    type(cms_scalar_event) :: events(CMS_MAX_EVENTS)
  end type
  public :: cms_reset, cms_record, cms_refuse_overload
  interface cms_record
    module procedure cms_record_dp, cms_record_qp
  end interface
contains
  subroutine cms_reset(report,precision,topology,overload)
    type(cms_scalar_report), intent(out) :: report
    integer, intent(in) :: precision,topology,overload
    report=cms_scalar_report()
    report%precision=precision
    report%topology=topology
    report%overload=overload
  end subroutine

  subroutine cms_refuse_overload(report,precision,topology)
    type(cms_scalar_report), intent(out) :: report
    integer, intent(in) :: precision,topology
    call cms_reset(report,precision,topology,0)
    report%status=CMS_OVERLOAD_UNSUPPORTED
  end subroutine

  subroutine cms_record_dp(report,source_line,metric_kind,metric_norm,p_index,mass_index, &
                           metric_name,native_metric,original_value,classified_value,threshold,local_scale, &
                           original_invariant,original_mass)
    type(cms_scalar_report), intent(inout) :: report
    integer, intent(in) :: source_line,metric_kind,metric_norm,p_index,mass_index
    character(*), intent(in) :: metric_name
    real(8), intent(in) :: native_metric,threshold,local_scale
    complex(8), intent(in) :: original_value,classified_value
    complex(8), intent(in), optional :: original_invariant,original_mass
    complex(16) :: p_operand,m_operand
    if (present(original_invariant).and.present(original_mass)) then
      p_operand=cmplx(original_invariant,kind=16)
      m_operand=cmplx(original_mass,kind=16)
      call cms_record_qp(report,source_line,metric_kind,metric_norm,p_index,mass_index,metric_name, &
         real(native_metric,16),cmplx(original_value,kind=16),cmplx(classified_value,kind=16), &
         real(threshold,16),real(local_scale,16),p_operand,m_operand)
      return
    endif
    ! Every binary64 value is exactly representable in binary128.
    call cms_record_qp(report,source_line,metric_kind,metric_norm,p_index,mass_index,metric_name, &
       real(native_metric,16),cmplx(original_value,kind=16),cmplx(classified_value,kind=16), &
       real(threshold,16),real(local_scale,16))
  end subroutine

  subroutine cms_record_qp(report,source_line,metric_kind,metric_norm,p_index,mass_index, &
                           metric_name,native_metric,original_value,classified_value,threshold,local_scale, &
                           original_invariant,original_mass)
    type(cms_scalar_report), intent(inout) :: report
    integer, intent(in) :: source_line,metric_kind,metric_norm,p_index,mass_index
    character(*), intent(in) :: metric_name
    real(16), intent(in) :: native_metric,threshold,local_scale
    complex(16), intent(in) :: original_value,classified_value
    complex(16), intent(in), optional :: original_invariant,original_mass
    integer :: i
    if (native_metric.eq.0.and.real(classified_value,16).eq.0.and.aimag(classified_value).eq.0) return
    if (report%count.ge.CMS_MAX_EVENTS) then
      report%dropped=report%dropped+1
      report%status=CMS_EVENT_OVERFLOW
      return
    endif
    i=report%count+1
    report%count=i
    if (report%status.eq.CMS_OK) report%status=CMS_REFUSED
    report%events(i)%precision=report%precision
    report%events(i)%topology=report%topology
    report%events(i)%overload=report%overload
    report%events(i)%source_line=source_line
    report%events(i)%metric_kind=metric_kind
    report%events(i)%metric_norm=metric_norm
    report%events(i)%p_index=p_index
    report%events(i)%mass_index=mass_index
    report%events(i)%metric_name=metric_name
    report%events(i)%metric_underflowed=(native_metric.eq.0)
    report%events(i)%native_metric=native_metric
    report%events(i)%original_value=original_value
    report%events(i)%classified_value=classified_value
    if (present(original_invariant).and.present(original_mass)) then
      report%events(i)%original_invariant=original_invariant
      report%events(i)%original_mass=original_mass
      report%events(i)%operands_recorded=.true.
    endif
    report%events(i)%threshold=threshold
    report%events(i)%local_scale=local_scale
  end subroutine
end module
