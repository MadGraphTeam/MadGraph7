// Copyright (C) 2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
#ifndef MG_ARITHM_INVARIANT_H
#define MG_ARITHM_INVARIANT_H

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define MG_INVARIANT_INLINE __host__ __device__ inline
#else
#define MG_INVARIANT_INLINE inline
#endif

namespace MG_ARITHM
{
  // Error-free transforms require individually rounded operations. In
  // particular, -ffast-math must not reassociate their residual expressions.
  MG_INVARIANT_INLINE double invariant_add( double a, double b )
  {
#ifdef __CUDA_ARCH__
    return __dadd_rn( a, b );
#else
    volatile double result = a + b;
    return result;
#endif
  }

  MG_INVARIANT_INLINE double invariant_multiply( double a, double b )
  {
#ifdef __CUDA_ARCH__
    return __dmul_rn( a, b );
#else
    volatile double result = a * b;
    return result;
#endif
  }

  MG_INVARIANT_INLINE double invariant_two_sum( double a, double b, double& error )
  {
    const double sum = invariant_add( a, b );
    const double bv = invariant_add( sum, -a );
    const double av = invariant_add( sum, -bv );
    error = invariant_add( invariant_add( a, -av ), invariant_add( b, -bv ) );
    return sum;
  }

  // Compute (a+b)^2-m^2 from the original binary64 parent momenta. Correcting
  // only the norm of an already rounded a+b cannot restore its lost low part.
  MG_INVARIANT_INLINE double minkowski_sum2( const double a[4], const double b[4], double mass )
  {
    const double mass2 = invariant_multiply( mass, mass );
    double total = -mass2;
    double correction = -::fma( mass, mass, -mass2 );
    for( int k = 0; k < 4; ++k )
    {
      double low;
      const double high = invariant_two_sum( a[k], b[k], low );
      const double square = invariant_multiply( high, high );
      const double residual = ::fma( high, high, -square ) + 2 * high * low + low * low;
      const double sign = k == 0 ? 1.0 : -1.0;
      double error;
      total = invariant_two_sum( total, sign * square, error );
      correction = invariant_add( correction, invariant_add( error, sign * residual ) );
    }
    return invariant_add( total, correction );
  }
}

#undef MG_INVARIANT_INLINE
#endif
