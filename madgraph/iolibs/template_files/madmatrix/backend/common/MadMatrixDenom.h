// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: F. Stloukal (Oct 2026) for MadGraph7.

#ifndef MadMatrixDenom_H
#define MadMatrixDenom_H 1

#include "MadMatrixInvariants.h"

// Propagator denominator of the helas routines; included by HelAmps after the backend's fpdenom_bcast.
namespace madmatrix
{
  // Squared propagator mass m2 = M^2 - i M W at denom precision: real mass and width, or complex mass (CMS).
  inline __host__ __device__
  cxtype_denom mass2( fptype M, fptype W )
  {
    const fptype_denom m = M;
    return cxtype_denom( m * m, -( m * static_cast<fptype_denom>( W ) ) );
  }

  inline __host__ __device__
  cxtype_denom mass2( const cxtype& M )
  {
    const cxtype_denom m( M.real(), M.imag() );
    return m * m;
  }

  // Virtuality P^2 - m2 (the propagator denominator, width included): only P^2 - Re(m2) at denom precision (it cancels near the pole).
  inline __host__ __device__
  cxtype_amp_sv virtuality( const fptype_denom_sv P[4], const cxtype_denom& m2 )
  {
    const fptype_denom_sv p2 = ( P[0] * P[0] ) - ( P[1] * P[1] ) - ( P[2] * P[2] ) - ( P[3] * P[3] );
    return cxmake( fpamp_of_mom( p2 - fpdenom_bcast( m2.real() ) ), static_cast<fptype_amp>( -m2.imag() ) );
  }

  // Standard real-mass VVV propagators can lose the low part of a parent sum
  // before its nearly cancelling norm is formed. Correct only those binary64
  // lanes, preserving the ordinary backend virtuality everywhere else.
  inline __host__ __device__
  cxtype_amp_sv virtuality_sum2( const fptype_denom_sv P[4],
                               const fptype_momenta_sv a[4], const fptype_momenta_sv b[4],
                               fptype M, fptype W, fptype_denom sign_a, fptype_denom sign_b )
  {
    const cxtype_amp_sv ordinary = virtuality( P, mass2( M, W ) );
#if defined(MGONGPU_FPTYPE_DOUBLE) && !defined(MGONGPU_SIMD_DENOM64)
    const fptype_denom_sv real = fp_minkowski_sum2( P, a, b, static_cast<fptype_denom>( M ),
                                                  sign_a, sign_b, cxreal( ordinary ) );
    return cxmake( fpamp_of_mom( real ), cximag( ordinary ) );
#else
    return ordinary;
#endif
  }

  template<typename C>
  inline __host__ __device__
  cxtype_amp_sv denom_sum2( fptype ccoeff, const C& coup, const fptype_denom_sv P[4],
                          const fptype_momenta_sv a[4], const fptype_momenta_sv b[4],
                          fptype M, fptype W, fptype_denom sign_a, fptype_denom sign_b )
  {
    return static_cast<fptype_amp>( ccoeff ) * coup / virtuality_sum2( P, a, b, M, W, sign_a, sign_b );
  }

  inline __host__ __device__
  cxtype_amp_sv denom_sum2( const fptype_denom_sv P[4],
                          const fptype_momenta_sv a[4], const fptype_momenta_sv b[4],
                          fptype M, fptype W, fptype_denom sign_a, fptype_denom sign_b )
  {
    return fptype_amp( 1 ) / virtuality_sum2( P, a, b, M, W, sign_a, sign_b );
  }

  // Propagator coupling Ccoeff * COUP / ( P^2 - m2 ), the division at vertex precision; mass as in mass2.
  template<typename C, typename... Mass> //look mom, I used variadic template for the 1st T in my life
  inline __host__ __device__
  cxtype_amp_sv denom( fptype ccoeff, const C& coup, const fptype_denom_sv P[4], const Mass&... mass )
  {
    return static_cast<fptype_amp>( ccoeff ) * coup / virtuality( P, mass2( mass... ) );
  }

  // Without coupling: 1 / ( P^2 - m2 ).
  template<typename... Mass>
  inline __host__ __device__
  cxtype_amp_sv denom( const fptype_denom_sv P[4], const Mass&... mass )
  {
    return fptype_amp( 1 ) / virtuality( P, mass2( mass... ) );
  }
}

#endif // MadMatrixDenom_H
