// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Nov 2020) for the MadGraph7 CUDACPP plugin.
// Further modified by: S. Roiser, A. Valassi, Z. Wettersten (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUVECTORS_H
#define MGONGPUVECTORS_H 1

#include "mgOnGpuCxtypes.h"
#include "mgOnGpuFptypes.h"

#include <cassert>
#include <iostream>
#include <type_traits>

//==========================================================================

//------------------------------
// Vector types - C++
//------------------------------

#ifdef __clang__
// If set: return a pair of (fptype&, fptype&) by non-const reference in cxtype_v::operator[]
// This is forbidden in clang ("non-const reference cannot bind to vector element")
// See also https://stackoverflow.com/questions/26554829
//#define MGONGPU_HAS_CPPCXTYPEV_BRK 1 // clang test (compilation fails also on clang 12.0, issue #182)
#undef MGONGPU_HAS_CPPCXTYPEV_BRK // clang default
#elif defined __INTEL_COMPILER
//#define MGONGPU_HAS_CPPCXTYPEV_BRK 1 // icc default?
#undef MGONGPU_HAS_CPPCXTYPEV_BRK // icc test
#else
#define MGONGPU_HAS_CPPCXTYPEV_BRK 1 // gcc default
//#undef MGONGPU_HAS_CPPCXTYPEV_BRK // gcc test (very slightly slower? issue #172)
#endif

// NB: the madgraph namespace: types are now split per backend file, not per namespace (see #318 and #725)
namespace madgraph
{

  const int neppV = 1;

}

//--------------------------------------------------------------------------

// DANGEROUS! this was mixing different cxtype definitions for CPU and GPU builds (see #318 and #725)
// DO NOT expose typedefs outside the namespace
//using mgOnGpu::neppV;
//#ifdef MGONGPU_CPPSIMD
//using mgOnGpu::fptype_v;
//using mgOnGpu::fptype2_v;
//using mgOnGpu::cxtype_v;
//using mgOnGpu::bool_v;
//#endif

//==========================================================================

// NB: the madgraph namespace: types are now split per backend file, not per namespace (see #318 and #725)
namespace madgraph
{

  //==========================================================================


  //------------------------------
  // Vector types - CUDA
  //------------------------------

  // Printout to std::cout for user defined types
  inline __host__ __device__ void
  print( const fptype& f )
  {
    printf( "%f\n", f );
  }
  inline __host__ __device__ void
  print( const cxtype& c )
  {
    printf( "[%f, %f]\n", cxreal( c ), cximag( c ) );
  }

  /*
  inline __host__ __device__ const cxtype&
  cxvmake( const cxtype& c )
  {
    return c;
  }
  */

  template<typename FP>
  inline __host__ __device__ FP
  fpternary( const bool& mask, const FP& a, const FP& b )
  {
    return ( mask ? a : b );
  }

  template<typename CX>
  inline __host__ __device__ CX
  cxternary( const bool& mask, const CX& a, const CX& b )
  {
    return ( mask ? a : b );
  }

  inline __host__ __device__ bool
  maskand( const bool& mask )
  {
    return mask;
  }


  //==========================================================================

  // Scalar-or-vector types: scalar in CUDA, vector or scalar in C++.
  // 3 mixed-precision stages: _amp (== fptype), _momenta (== _denom), _colour (== fptype2).
  typedef bool bool_sv;
  typedef fptype fptype_sv;
  typedef fptype2 fptype2_sv;
  typedef unsigned int uint_sv;
  typedef cxtype cxtype_sv;
  typedef cxtype_ref cxtype_sv_ref;
  typedef fptype_momenta fptype_momenta_sv;   typedef fptype_momenta fptype_momenta_v;
  typedef fptype_denom fptype_denom_sv;       typedef fptype_denom fptype_denom_v;
  typedef fptype_amp fptype_amp_sv;           typedef fptype_amp fptype_amp_v;
  typedef fptype_colour fptype_colour_sv;     typedef fptype_colour fptype_colour_v;
  typedef cxtype_momenta cxtype_momenta_sv;   typedef cxtype_momenta cxtype_momenta_v;
  typedef cxtype_denom cxtype_denom_sv;       typedef cxtype_denom cxtype_denom_v;
  typedef cxtype_amp cxtype_amp_sv;           typedef cxtype_amp cxtype_amp_v;
  typedef cxtype_colour cxtype_colour_sv;     typedef cxtype_colour cxtype_colour_v;

  // narrowing/casting operators for unification
#ifdef MGONGPU_SIMD_DENOM64
  inline fptype_amp_sv fpamp_of_mom( const fptype_momenta_sv& p ) { return fpdenom_narrow( p ); } 
#else
  inline __host__ __device__ fptype_amp_sv fpamp_of_mom( const fptype_momenta_sv& p ) { return static_cast<fptype_amp_sv>( p ); } 
#endif
  template<typename T>
  inline __host__ __device__ fptype_amp fpamp_scalar( const T& x ) { return static_cast<fptype_amp>( x ); }

  // Scalar-or-vector zeros: scalar in CUDA, vector or scalar in C++
  // Template version for multi-precision (explicit template parameter required)
  template<typename CX = cxtype>
  inline __host__ __device__ CX cxzero_sv(){ return CX{}; }

  //==========================================================================


  // Functions and operators for cxtype_sv (and multi-precision variant)
  template<typename CX>
  inline __host__ __device__ auto
  cxabs2( const CX& c ) -> decltype( cxreal( c ) * cxreal( c ) + cximag( c ) * cximag( c ) )
  {
    return cxreal( c ) * cxreal( c ) + cximag( c ) * cximag( c );
  }

  // ALOHA raises the denominator of a custom propagator to an integer power (the
  // squared Breit-Wigner of the SMEFTsim width corrections, for instance). There
  // is no std::pow overload for cxtype_v, so expand the power by repeated
  // multiplication: the exponent comes from the UFO propagator and is a small
  // positive integer.
  template<class T>
  inline __host__ __device__ T
  cxpow( const T& base, const fptype n )
  {
    const int k = (int)n;
    assert( (fptype)k == n && k >= 1 ); // only positive integer powers are supported
    T out = base;
    for( int i = 1; i < k; i++ ) out = out * base;
    return out;
  }

  //==========================================================================

} // end namespace madgraph

#endif // MGONGPUVECTORS_H
