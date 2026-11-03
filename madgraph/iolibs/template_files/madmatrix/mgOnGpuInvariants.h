// Copyright (C) 2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
#ifndef MGONGPUINVARIANTS_H
#define MGONGPUINVARIANTS_H

#include "mgOnGpuVectors.h"
#include "Arithmetics/Invariant.h"

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  // Preserve the ordinary SIMD expression away from cancellation. Correct
  // individual binary64 lanes when three or more digits can be lost. This
  // does not change the stored wavefunctions, masses, widths or momenta.
  __host__ __device__ inline fptype_denom_sv
  fp_minkowski_sum2( const fptype_denom_sv p[4], const fptype_momenta_sv a[4],
                    const fptype_momenta_sv b[4], fptype_denom mass,
                    fptype_denom sign_a, fptype_denom sign_b )
  {
    fptype_denom_sv result = p[0]*p[0] - p[1]*p[1] - p[2]*p[2] - p[3]*p[3] - mass*mass;
#if defined(MGONGPU_FPTYPE_DOUBLE) && !defined(MGONGPU_SIMD_DENOM64)
    const fptype_denom_sv scale = p[0]*p[0] + p[1]*p[1] + p[2]*p[2] + p[3]*p[3] + mass*mass;
#ifdef MGONGPU_CPPSIMD
    for( int lane = 0; lane < neppV; ++lane )
    {
      if( ::fabs( result[lane] ) < 0x1p-10 * scale[lane] )
      {
        double left[4], right[4];
        for( int k = 0; k < 4; ++k )
        {
          left[k] = sign_a * a[k][lane];
          right[k] = sign_b * b[k][lane];
        }
        result[lane] = MG_ARITHM::minkowski_sum2( left, right, mass );
      }
    }
#else
    if( ::fabs( result ) < 0x1p-10 * scale )
    {
      double left[4], right[4];
      for( int k = 0; k < 4; ++k )
      {
        left[k] = sign_a * a[k];
        right[k] = sign_b * b[k];
      }
      result = MG_ARITHM::minkowski_sum2( left, right, mass );
    }
#endif
#endif
    return result;
  }
}

#endif
