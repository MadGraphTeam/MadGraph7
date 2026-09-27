// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUDW_H
#define MGONGPUDW_H 1

#include "mgOnGpuConfig.h"

#ifdef MGONGPU_DWTYPE

#include "CompleXDW/XDW.h"

#include <iostream>
#include <type_traits>

// The kernel helpers of mgOnGpuCxtypes.h and mgOnGpuFptypes.h for CompleXDW DW<T>/XDW<T>,
// T scalar (cpu, gpu) or a SIMD vector (simd): see MGONGPU_DWTYPE in mgOnGpuConfig.h
namespace madmatrix
{
  template<typename T>
  inline __host__ __device__ XDW<T>
  cxmake( const DW<T>& r, const std::type_identity_t<DW<T>>& i )
  {
    return XDW<T>( r, i );
  }

  template<typename R, typename T>
    requires std::is_arithmetic_v<R>
  inline __host__ __device__ XDW<T>
  cxmake( const R& r, const DW<T>& i )
  {
    return XDW<T>( DW<T>( r ), i );
  }

  template<typename T>
  inline __host__ __device__ DW<T>
  cxreal( const XDW<T>& c )
  {
    return real( c );
  }

  template<typename T>
  inline __host__ __device__ DW<T>
  cximag( const XDW<T>& c )
  {
    return imag( c );
  }

  template<typename T>
  inline __host__ __device__ XDW<T>
  cxconj( const XDW<T>& c )
  {
    return conj( c );
  }

  template<typename T>
  inline __host__ __device__ DW<T>
  cxabs2( const XDW<T>& c )
  {
    return norm( c );
  }

  // Not a template, so a plain number converts on either side
  inline __host__ __device__ fptype_amp
  fpternary( const bool& mask, const fptype_amp& a, const fptype_amp& b )
  {
    return mask ? a : b;
  }

  template<typename T>
  inline __host__ __device__ DW<T>
  fpsqrt( const DW<T>& x )
  {
    return sqrt( x );
  }

  template<typename T>
  inline __host__ __device__ DW<T>
  fpabs( const DW<T>& x )
  {
    return abs( x );
  }

  template<typename T>
  inline __host__ __device__ auto
  fpsignbit( const DW<T>& x )
  {
    return signbit( x );
  }

  template<typename T>
    requires std::floating_point<T>
  inline std::ostream&
  operator<<( std::ostream& out, const XDW<T>& c )
  {
    out << "(" << static_cast<double>( real( c ) ) << ", " << static_cast<double>( imag( c ) ) << ")";
    return out;
  }
}

#endif // MGONGPU_DWTYPE

#endif // MGONGPUDW_H
