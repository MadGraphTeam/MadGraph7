"""Exercise the generated Fortran packing ABI at its real C call boundary."""
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from madgraph import MG5DIR
from madgraph.iolibs import export_fks, file_writers
from madmatrix.output import ProcessExporterMadMatrixNLOReal


@unittest.skipUnless(shutil.which('gfortran') and shutil.which('cc'), 'requires Fortran and C')
class TestProviderPacking(unittest.TestCase):
    def test_explicit_ew_masks_order_strides_and_physical_flavour(self):
        manifest = dict(real_matrix_elements=[dict(id=1, library_process_id='PMM_R',
            local_squared_orders=[[2], [3], [4]], backend_capabilities=dict(cuda=True, hip=True))],
            fks_rows=[dict(fks_row=7, real_me_id=1, fortran_flavour_index=2, local_to_global=[3,1,4])])
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with file_writers.FortranWriter(str(root/'real_me_chooser.f')) as chooser, \
                    file_writers.FortranWriter(str(root/'lum.f')) as lum:
                export_fks.ProcessExporterFortranFKS.write_real_wrappers(
                    None, chooser, lum, SimpleNamespace(real_processes=[]), None, None)
            exporter = object.__new__(ProcessExporterMadMatrixNLOReal)
            exporter._write_fortran_routing(manifest, name)
            (root/'nexternal.inc').write_text('integer, parameter :: nexternal=2\n')
            (root/'orders.inc').write_text('integer, parameter :: amp_split_size=4, nsplitorders=1, nlo_orders(1)=[3]\n')
            (root/'nlo_real_backend.inc').write_text('character(*), parameter :: default_backend="scalar"\n')
            (root/'provider.c').write_text("""
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
static int calls = 0;
int call_count(void) { return calls; }
int mg7_nlo_real_initialize(void **ctx, const char *p, const char *d, const char *b) {
    *ctx = (void *)&calls;
    return 0;
}
void mg7_nlo_real_set_trace(void *ctx, void *input, void *output) {}
const char *mg7_nlo_real_last_error(void *ctx) { return ""; }
int mg7_nlo_real_available(void *ctx, int id) { return ctx && id == 1; }
int mg7_nlo_real_evaluate_ew(void *ctx, int row, size_t n, const double *p,
    const double *g, double e, const int32_t *f, double *res) {
    if (row != 1 || !n) abort();
    ++calls;
    for (size_t i = 0; i < n; ++i) {
        int lane = (int)g[i];
        if (f[i] != 1 || lane <= 0) abort();
        for (int mu = 0; mu < 4; ++mu)
            for (int leg = 0; leg < 2; ++leg)
                if (p[n*(2*mu+leg)+i] != 100*lane+10*mu+leg) abort();
        res[i] = g[i] + e;
        res[n+i] = -g[i]/4;
        res[2*n+i] = 2*g[i];
    }
    return 0;
}
""")
            (root/'probe.f90').write_text("""
program probe
  use iso_c_binding
  implicit none
  integer :: i,j,k,n,stage
  real(8) :: p(0:3,2,9), g(9), ew, weights(9), amplitudes(4,9)
  logical :: mask(9),ok
  logical, external :: mc_provider_can_batch
  interface
    integer(c_int) function calls() bind(C,name='call_count')
      import c_int
    end function
  end interface
  if (mc_provider_can_batch(0,2).or.mc_provider_can_batch(7,1)) stop 1
  if (.not.mc_provider_can_batch(7,2)) stop 2
  do i=1,9
    g(i)=i
    do j=0,3
      do k=1,2
        p(j,k,i)=100*i+10*j+k-1
      enddo
    enddo
  enddo
  mask=.false.
  call mg7_nlo_real_try_batch(p,g,2d0,amplitudes,weights,mask,9,7,2,ok)
  if (.not.ok.or.calls()/=0.or.any(weights/=0).or.any(amplitudes/=0)) stop 3
  do stage=1,4
    n=9
    ew=5*stage
    select case(stage)
    case(1,3)
      mask=.false.
      mask(2:8:3)=.true.
    case(2)
      mask=.true.
    case(4)
      mask=.false.
      mask(1)=.true.
    end select
    call mg7_nlo_real_try_batch(p,g,ew,amplitudes,weights,mask,n,7,2,ok)
    if (.not.ok.or.calls()/=stage) stop 4
    do i=1,n
      if(mask(i))then
        if(any(amplitudes(:,i)/=[-g(i)/4,0d0,g(i)+ew,0d0])) stop 5
        if(weights(i)/=0.75d0*g(i)+ew) stop 6
      else
        if(any(amplitudes(:,i)/=0).or.weights(i)/=0) stop 7
      endif
    enddo
  enddo
  call mc_evaluate_provider_batch(p,g,0.125d0,7,2,5,weights,amplitudes,ok)
  if (.not.ok.or.calls()/=5.or.any(weights(:5)/=0.75d0*g(:5)+0.125d0)) stop 8
end program
subroutine amp_split_pos_to_orders(position, orders)
 integer position,orders(*)
 orders(1)=position
end subroutine
""")
            trace = Path(MG5DIR)/'Template/NLO/SubProcesses/mc_batch_trace.f90'
            for cmd in (['cc','-c','provider.c'],
                        ['gfortran','-fcheck=all','-ffree-line-length-none',trace,
                         'nlo_real_offload.f90','probe.f90','provider.o','-o','probe'],
                        ['./probe']):
                result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
