// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUDW_H
#define MGONGPUDW_H 1

#include "mgOnGpuConfig.h"

#include <cstddef>

// Double words cannot be read through volatile, nor need it: it works around fast-math (#736)
#ifdef MGONGPU_DWTYPE
#define MGONGPU_VOLATILE
#else
#define MGONGPU_VOLATILE volatile
#endif

#ifndef MGONGPU_DWTYPE
namespace madmatrix
{
  // Entry i of a numerator or denominator buffer, read as in an ARRAY[nevt]
  template<typename Buffer>
  inline __host__ __device__ auto
  fpentry( const Buffer& buffer, const std::size_t i )
  {
    return buffer[i];
  }
}
#else

#include "CompleXDW/XDW.h"

#include <iostream>
#include <type_traits>

// std::abs in the helas templates
namespace std
{
  template<typename T>
  inline __host__ __device__ DW<T>
  abs( const DW<T>& x )
  {
    return ::abs( x );
  }
}

// The kernel helpers of mgOnGpuCxtypes.h, mgOnGpuFptypes.h and mgOnGpuVectors.h for CompleXDW DW<T>/XDW<T>,
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
  inline __host__ __device__ auto
  fpsignbit( const DW<T>& x )
  {
    return signbit( x );
  }

#ifdef MGONGPU_CPPSIMD
  // SIMD kernel vectors: DW/XDW of float or double lanes, fptype_v stays the FP64 buffer vector
  typedef fptype_amp::BaseType fptype_lane;
#ifdef __clang__
  typedef fptype_lane fptype_lane_v __attribute__( ( ext_vector_type( MGONGPU_CPPSIMD ) ) );
#else
  typedef fptype_lane fptype_lane_v __attribute__( ( vector_size( MGONGPU_CPPSIMD * sizeof( fptype_lane ) ), aligned( MGONGPU_CPPSIMD * sizeof( fptype_lane ) ) ) );
#endif
  typedef DW<fptype_lane_v> fptype_amp_v;
  typedef XDW<fptype_lane_v> cxtype_amp_v;

  template<typename Mask>
  inline fptype_amp_v
  fpternary( const Mask& mask, const fptype_amp_v& a, const fptype_amp_v& b )
  {
    return select( mask, a, b );
  }

  template<typename Mask>
  inline cxtype_amp_v
  cxternary( const Mask& mask, const cxtype_amp_v& a, const cxtype_amp_v& b )
  {
    return select( mask, a, b );
  }

  inline fptype_amp_v
  fpmax( const fptype_amp_v& a, const fptype_amp_v& b )
  {
    return max( a, b );
  }

  inline fptype_amp_v
  fpmin( const fptype_amp_v& a, const fptype_amp_v& b )
  {
    return min( a, b );
  }

  typedef fptype_amp_v fptype_amp_page;
#else
  typedef fptype_amp fptype_amp_page;
#endif

  // Numerator and denominator buffers of double words: pages of DW<T>, T scalar or the SIMD lanes
  struct KernelAccessDW
  {
    static __host__ __device__ fptype_amp*
    ieventAccessRecord( fptype_amp* buffer, const int ievt ) { return buffer + ievt; }

    static __host__ __device__ fptype_amp_page*
    kernelAccessP( fptype_amp* buffer ) { return reinterpret_cast<fptype_amp_page*>( buffer ); }

    static __host__ __device__ fptype_amp_page&
    kernelAccess( fptype_amp* buffer ) { return *kernelAccessP( buffer ); }
  };

  // Entry i of such a buffer: a DW<T> page holds all high words, then all low words
  template<typename Buffer>
  inline __host__ __device__ fptype
  fpentry( const Buffer& buffer, const std::size_t i )
  {
#ifdef MGONGPU_CPPSIMD
    return static_cast<fptype>( reinterpret_cast<const fptype_amp_v*>( &buffer[0] )[i / MGONGPU_CPPSIMD].lane( i % MGONGPU_CPPSIMD ) );
#else
    return static_cast<fptype>( buffer[i] );
#endif
  }

#ifdef MGONGPUCPP_GPUIMPL
  using ::atomicAdd;

  // DW<float> fits 64 bits: full double-word sum by CAS. DW<double> sums in the high word, already FP64.
  template<typename T>
  inline __device__ void
  atomicAdd( DW<T>* sum, const std::type_identity_t<DW<T>>& x )
  {
    if constexpr( std::is_same_v<T, float> )
    {
      unsigned long long* address = reinterpret_cast<unsigned long long*>( sum );
      unsigned long long old = *address, assumed;
      do
      {
        assumed = old;
        const DW<float> next = DW<float>( __uint_as_float( static_cast<unsigned int>( assumed ) ),
                                          __uint_as_float( static_cast<unsigned int>( assumed >> 32 ) ) ) + x;
        old = atomicCAS( address, assumed,
                         static_cast<unsigned long long>( __float_as_uint( next.lo() ) ) << 32 | __float_as_uint( next.hi() ) );
      } while( assumed != old );
    }
    else
      ::atomicAdd( &sum->hi(), static_cast<T>( x ) );
  }
#endif

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
