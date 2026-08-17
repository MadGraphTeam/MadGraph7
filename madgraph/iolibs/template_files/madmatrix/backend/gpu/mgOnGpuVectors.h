// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Nov 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Roiser, A. Valassi, Z. Wettersten (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUVECTORS_H
#define MGONGPUVECTORS_H 1

#include "mgOnGpuCxtypes.h"
#include "mgOnGpuFptypes.h"

#include <iostream>

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

  inline __host__ __device__ fptype
  fpternary( const bool& mask, const fptype& a, const fptype& b )
  {
    return ( mask ? a : b );
  }

  inline __host__ __device__ cxtype
  cxternary( const bool& mask, const cxtype& a, const cxtype& b )
  {
    return ( mask ? a : b );
  }

  inline __host__ __device__ bool
  maskand( const bool& mask )
  {
    return mask;
  }


  //==========================================================================

  // Scalar-or-vector types: scalar in CUDA, vector or scalar in C++
  typedef bool bool_sv;
  typedef fptype fptype_sv;
  typedef fptype2 fptype2_sv;
  typedef unsigned int uint_sv;
  typedef cxtype cxtype_sv;
  typedef cxtype_ref cxtype_sv_ref;

  // Scalar-or-vector zeros: scalar in CUDA, vector or scalar in C++
  inline __host__ __device__ cxtype cxzero_sv(){ return cxtype( 0, 0 ); }

  //==========================================================================

  // Functions and operators for cxtype_sv
  inline __host__ __device__ fptype_sv
  cxabs2( const cxtype_sv& c )
  {
    return cxreal( c ) * cxreal( c ) + cximag( c ) * cximag( c );
  }

  //==========================================================================

} // end namespace madgraph

#endif // MGONGPUVECTORS_H
