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

  // Printout to stream for user defined types

#ifndef MGONGPU_CPPCXTYPE_CXSMPL // operator<< for cxsmpl has already been defined!
  inline std::ostream&
  operator<<( std::ostream& out, const cxtype& c )
  {
    out << "[" << cxreal( c ) << "," << cximag( c ) << "]";
    //out << cxreal(c) << "+i" << cximag(c);
    return out;
  }
#endif

  /*
#ifdef MGONGPU_CPPSIMD
  inline std::ostream&
  operator<<( std::ostream& out, const bool_v& v )
  {
    out << "{ " << v[0];
    for ( int i=1; i<neppV; i++ ) out << ", " << (bool)(v[i]);
    out << " }";
    return out;
  }
#endif
  */


  //--------------------------------------------------------------------------

  /*
  // Printout to std::cout for user defined types

  inline void print( const fptype& f ) { std::cout << f << std::endl; }

#ifdef MGONGPU_CPPSIMD
  inline void print( const fptype_v& v ) { std::cout << v << std::endl; }
#endif

  inline void print( const cxtype& c ) { std::cout << c << std::endl; }

#ifdef MGONGPU_CPPSIMD
  inline void print( const cxtype_v& v ) { std::cout << v << std::endl; }
#endif
  */

  //--------------------------------------------------------------------------

  // Functions and operators for fptype_v


  /*
#ifdef MGONGPU_CPPSIMD
  inline fptype_v
  fpvmake( const fptype v[neppV] )
  {
    fptype_v out = {}; // see #594
    for ( int i=0; i<neppV; i++ ) out[i] = v[i];
    return out;
  }
#endif
  */

  //--------------------------------------------------------------------------

  // Functions and operators for cxtype_v


  //--------------------------------------------------------------------------

  // Functions and operators for bool_v (ternary and masks)


  inline fptype
  fpternary( const bool& mask, const fptype& a, const fptype& b )
  {
    return ( mask ? a : b );
  }

  inline cxtype
  cxternary( const bool& mask, const cxtype& a, const cxtype& b )
  {
    return ( mask ? a : b );
  }

  /*
  inline bool
  maskor( const bool& mask )
  {
    return mask;
  }
  */

  inline bool
  maskand( const bool& mask )
  {
    return mask;
  }


  //--------------------------------------------------------------------------

  // Functions and operators for fptype_v (min/max)


  //--------------------------------------------------------------------------

  // Functions and operators for fptype2_v

  // Keeps living only on the SIMD /backend

  //==========================================================================


  //==========================================================================

  // Scalar-or-vector types: scalar in CUDA, vector or scalar in C++
  typedef bool bool_sv;
  typedef fptype fptype_sv;
  typedef fptype2 fptype2_sv;
  typedef unsigned int uint_sv;
  typedef cxtype cxtype_sv;
  typedef cxtype_ref cxtype_sv_ref;

  // Scalar-or-vector zeros: scalar in CUDA, vector or scalar in C++
  inline cxtype cxzero_sv() { return cxtype( 0, 0 ); }

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
