#ifndef BASICOPS_H
#define BASICOPS_H

#pragma once

#include <limits>

#if defined( __CUDACC__ )
#include <cuda_runtime.h>
#endif




namespace MG_ARITHM{

#if defined( __CUDACC__ )
#define __tnl_inline__ __forceinline__
#elif defined( _MSC_VER )
#define __tnl_inline__ __forceinline
#elif defined( __GNUC__ ) || defined( __clang__ )
#define __tnl_inline__ __attribute__( ( always_inline ) ) inline
#else
#define __tnl_inline__ inline
#endif

#if defined( __CUDACC__ )
#define __cuda_callable__ \
__device__             \
__host__
#else
#define __cuda_callable__
#endif

#ifdef __CADNA__
   template <typename T>
     constexpr bool is_special_fp_v =
         std::is_same_v<T, double_st> || std::is_same_v<T, float_st>;
#endif

#ifdef __CADNA__
#define FLOAT_TEMPLATE_GUARD \
template< typename T, std::enable_if_t< is_special_fp_v< T >, int > = 0 >
#else
#define FLOAT_TEMPLATE_GUARD \
template< typename T, std::enable_if_t< std::is_floating_point_v< T >, int > = 0 >
#endif


FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __tnl_inline__ T
add_rn( const T x, const T y )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __dadd_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fadd_rn( x, y );
   }
#else
   return x + y;
#endif
}

FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __tnl_inline__ T
mul_rn( const T x, const T y )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __dmul_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fmul_rn( x, y );
   }
#else
   return x * y;
#endif
}

FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __tnl_inline__ T
div_rn( const T x, const T y )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __ddiv_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fdiv_rn( x, y );
   }
#else
   return x / y;
#endif
}

FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __tnl_inline__ T
fma_rn( const T x, const T y, const T z )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __fma_rn( x, y, z );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fmaf_rn( x, y, z );
   }
#else
   #ifdef FP_FAST_FMA
   return fma( x, y, z );
   #else
   printf( "There is no FMA. Do not enable FP_FAST_FMA e.g. with -mfma." );
   return std::numeric_limits< T >::quiet_NaN();
   #endif
#endif
}

}

#endif  //BASICOPS_H
