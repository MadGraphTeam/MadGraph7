
#ifndef ERRORFREEOPS_H
#define ERRORFREEOPS_H

#pragma once

#include "basicOPs.h"

namespace MG_ARITHM{

template< typename T >
struct rne
{
   T sum;
   T error;
};




namespace spliting_detail {

// double precision
static constexpr double DOUBLE_SPLITTER = 0x1p27 + 1.0;   // 2^27 + 1 = 134217729.0
static constexpr double DOUBLE_UNSCALE_FACTOR = 0x1p-28;  // 2^-28 = 3.7252902984619140625e-09
static constexpr double DOUBLE_RESCALE_FACTOR = 0x1p28;   // 2^28  = 268435456.0
static constexpr double DOUBLE_SPLIT_THRESH = 0x1p996;    // 2^996 ≈ 6.69692879491417e+299

// float precision
static constexpr float FLOAT_SPLITTER = 0x1p12f + 1.0f;  // 2^12 + 1 = 4097.0f
static constexpr float FLOAT_UNSCALE_FACTOR = 0x1p-13f;  // 2^-13 = 1.220703125e-04f
static constexpr float FLOAT_RESCALE_FACTOR = 0x1p13f;   // 2^13 = 8192.0f
static constexpr float FLOAT_SPLIT_THRESH = 0x1p115f;    // 2^115 ≈ 4.15383748682786e+34f

}

template< typename T, typename U, std::enable_if_t< std::is_floating_point_v< T >, int > = 0 >
__tnl_inline__ __cuda_callable__
constexpr rne< U >
split( T value )
{
   U high;
   U low;
   if constexpr( std::is_same_v< T, double > ) {
      U temp;
      if( value > spliting_detail::DOUBLE_SPLIT_THRESH || value < -spliting_detail::DOUBLE_SPLIT_THRESH ) {
         value *= spliting_detail::DOUBLE_UNSCALE_FACTOR;
         temp = spliting_detail::DOUBLE_SPLITTER * value;
         high = temp - ( temp - value );
         low = value - high;
         high *= spliting_detail::DOUBLE_RESCALE_FACTOR;
         low *= spliting_detail::DOUBLE_RESCALE_FACTOR;
      }
      else {
         temp = spliting_detail::DOUBLE_SPLITTER * value;
         high = temp - ( temp - value );
         low = value - high;
      }
      return { static_cast< U >( high ), static_cast< U >( low ) };
   }
   else if constexpr( std::is_same_v< T, float > ) {
      U temp;
      if( value > spliting_detail::FLOAT_SPLIT_THRESH || value < -spliting_detail::FLOAT_SPLIT_THRESH ) {
         value *= spliting_detail::FLOAT_UNSCALE_FACTOR;
         temp = spliting_detail::FLOAT_SPLITTER * value;
         high = temp - ( temp - value );
         low = value - high;
         high *= spliting_detail::FLOAT_RESCALE_FACTOR;
         low *= spliting_detail::FLOAT_RESCALE_FACTOR;
      }
      else {
         temp = spliting_detail::FLOAT_SPLITTER * value;
         high = temp - ( temp - value );
         low = value - high;
      }
      return { static_cast< U >( high ), static_cast< U >( low ) };
   }
}

FLOAT_TEMPLATE_GUARD
__cuda_callable__
constexpr __tnl_inline__ rne< T >
quick_two_sum( const T a, const T b )
{
  const T s = add_rn(a, b);
  const T z = add_rn(s, -a);
  const T err = add_rn(b, -z);
  return {s, err};
}

FLOAT_TEMPLATE_GUARD
__cuda_callable__
constexpr __tnl_inline__ rne< T >
quick_two_diff( const T a, const T b )
{
  const T s = add_rn(a, -b);
  const T z = add_rn(a, -s);
  const T err = add_rn(z, -b);
  return{s, err} ;
}


FLOAT_TEMPLATE_GUARD
__cuda_callable__
constexpr __tnl_inline__ rne< T >
two_sum( const T a, const T b )
{
  const T s = add_rn(a, b);
  const T aa = add_rn(s, -b);
  const T bb = add_rn(s, -aa);
  const T da = add_rn(a, -aa);
  const T db = add_rn(b, -bb);
  const T err = add_rn(da, db);
  return {s, err};
}


FLOAT_TEMPLATE_GUARD
__cuda_callable__
constexpr __tnl_inline__ rne< T >
two_diff( const T a, const T b )
{
   const T s  = add_rn(a, -b);
   const T bb = add_rn(s, -a);
   const T aa = add_rn(s, -bb);
   const T da = add_rn(a, -aa);
   const T db = add_rn(b, bb);
   const T err = add_rn(da, -db);
   return {s, err};
}

FLOAT_TEMPLATE_GUARD
__cuda_callable__
constexpr __tnl_inline__ rne< T >
two_prod( const T a, const T b )
{
#ifdef __CUDA_ARCH__
   const T p = mul_rn( a, b );
   const T err = fma_rn( a, b, -p );
   return { p, err };

#else
   #ifdef FP_FAST_FMA

   const T p = mul_rn( a, b );
   const T err = fma_rn( a, b, -p );
   return { p, err };

   #else

   const T p = mul_rn( a, b );
   const auto sp = split< T, T >( a );
   const T a_hi = sp.sum;
   const T a_lo = sp.error;
   const auto sp2 = split< T, T >( b );
   const T b_hi = sp2.sum;
   const T b_lo = sp2.error;
   const T ab_hh = mul_rn(a_hi, b_hi);           // a_hi * b_hi
   const T tmp1 = add_rn(ab_hh, -p);             // (a_hi * b_hi - p)
   const T ab_hl = mul_rn(a_hi, b_lo);           // a_hi * b_lo
   const T tmp2 = add_rn(tmp1, ab_hl);           // (a_hi * b_hi - p) + a_hi * b_lo
   const T ab_lh = mul_rn(a_lo, b_hi);           // a_lo * b_hi
   const T tmp3 = add_rn(tmp2, ab_lh);           // ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi)
   const T ab_ll = mul_rn(a_lo, b_lo);           // a_lo * b_lo
   const T err = add_rn(tmp3, ab_ll);            // ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
   return { p, err };
   #endif

#endif
}

/*template< typename T, std::enable_if_t< std::is_floating_point_v< T >, int > = 0 >
__cuda_callable__
constexpr rne< T >
two_sqr( const T a )
{
#ifdef __CUDA_ARCH__
   const T p = mul_rn( a, a );
   const T err = fma_rn( a, a, -p );
   return { p, err };

#else
   #ifdef FP_FAST_FMA

   const T p = mul_rn( a, a );
   const T err = fma_rn( a, a, -p );
   return { p, err };

   #else

   const T q = mul_rn( a, a );
   auto sp = split< T, T >( a );
   T temp = mul_rn( sp.sum, sp.sum );
   temp = add_rn( temp, -q );
   T err = mul_rn( static_cast<T>(2.0), mul_rn( sp.sum, sp.error ) );
   err = add_rn( err, temp );
   err = add_rn(mul_rn( sp.error, sp.error), err);
   // ( ( hi * hi - q ) + 2.0F * hi * lo ) + lo * lo;
   return { q, err };
   #endif
#endif
}*/

/*template< typename T, std::enable_if_t< std::is_floating_point_v< T >, int > = 0 >
__cuda_callable__
__tnl_inline__ constexpr T
nint( const T d )
{
   if( d == std::floor( d ) )
      return d;
   return std::floor( d + 0.5F );
}

template< typename T, std::enable_if_t< std::is_floating_point_v< T >, int > = 0 >
__cuda_callable__
__tnl_inline__ constexpr T
aint( const T d )
{
   return ( d >= 0.0F ) ? std::floor( d ) : std::ceil( d );
}*/


}

#endif  //ERRORFREEOPS.H
