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

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
namespace mg5amcCpu
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

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
namespace mg5amcCpu
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


#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  inline std::ostream&
  operator<<( std::ostream& out, const fptype2_v& v )
  {
    out << "{ " << v[0];
    for( int i = 1; i < neppV2; i++ ) out << ", " << v[i];
    out << " }";
    return out;
  }
#endif



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

#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT

  inline fptype2_v
  fpvmerge( const fptype_v& v1, const fptype_v& v2 )
  {
    // This code is not very efficient! It makes mixed precision FFV/color not faster than double on C++ (#537).
    // I considered various alternatives, including
    // - in gcc12 and clang, __builtin_shufflevector (works with different vector lengths, BUT the same fptype...)
    // - casting vector(4)double to vector(4)float and then assigning via reinterpret_cast... but how to do the cast?
    // Probably the best solution is intrinsics?
    // - see https://stackoverflow.com/questions/5139363
    // - see https://stackoverflow.com/questions/54518744
    /*
    fptype2_v out;
    for( int ieppV = 0; ieppV < neppV; ieppV++ )
    {
      out[ieppV] = v1[ieppV];
      out[ieppV+neppV] = v2[ieppV];
    }
    return out;
    */
    return out;
  }

  inline fptype_v
  fpvsplit0( const fptype2_v& v )
  {
    /*
    fptype_v out = {}; // see #594
    for( int ieppV = 0; ieppV < neppV; ieppV++ )
    {
      out[ieppV] = v[ieppV];
    }
    */
    return out;
  }

  inline fptype_v
  fpvsplit1( const fptype2_v& v )
  {
    /*
    fptype_v out = {}; // see #594
    for( int ieppV = 0; ieppV < neppV; ieppV++ )
    {
      out[ieppV] = v[ieppV+neppV];
    }
    */
    return out;
  }

#endif // #if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT


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

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MGONGPUVECTORS_H
