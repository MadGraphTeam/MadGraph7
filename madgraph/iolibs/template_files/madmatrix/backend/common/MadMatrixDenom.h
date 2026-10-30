// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: F. Stloukal (Oct 2026) for MadGraph7.

#ifndef MadMatrixDenom_H
#define MadMatrixDenom_H 1

// The propagator denominator of the helas routines, one copy for every backend.
// Included at the end of mgOnGpuCxtypes.h (cpu, gpu) or mgOnGpuVectors.h (simd),
// once the backend has defined the types, cxmake, fpamp_of_mom and fpdenom_bcast
// (the broadcast of a scalar into fptype_denom_sv, the one piece that differs).
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

  // Propagator coupling Ccoeff * COUP / ( P^2 - m2 ), the division at vertex precision; mass as in mass2.
  template<typename C, typename... Mass>
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
