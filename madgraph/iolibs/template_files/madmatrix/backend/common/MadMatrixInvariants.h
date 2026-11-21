// Copyright (C) 2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
#ifndef MadMatrixInvariants_H
#define MadMatrixInvariants_H 1

#include "MadMatrixConfig.h"
#ifdef MADMATRIX_HAS_VECTORS_H
#include "MadMatrixVectors.h"
#else
#include "MadMatrixCxtypes.h"
#endif
#include "Arithmetics/Invariant.h"

namespace madmatrix
{
  // The caller supplies the ordinary virtuality so unaffected lanes retain
  // precisely the backend's existing evaluation, precision and mass convention.
  inline __host__ __device__ fptype_denom_sv
  fp_minkowski_sum2( const fptype_denom_sv p[4], const fptype_momenta_sv a[4],
                    const fptype_momenta_sv b[4], fptype_denom mass,
                    fptype_denom sign_a, fptype_denom sign_b,
                    fptype_denom_sv result )
  {
#if defined(MADMATRIX_FPTYPE_DOUBLE) && !defined(MADMATRIX_SIMD_DENOM64)
    const fptype_denom_sv scale = p[0]*p[0] + p[1]*p[1] + p[2]*p[2] + p[3]*p[3] + mass*mass;
#ifdef MADMATRIX_HAS_VECTORS_H
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
