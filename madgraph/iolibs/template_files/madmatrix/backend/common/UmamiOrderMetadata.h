#pragma once

#include "process_fingerprint.h"
#include "umami.h"

#include <algorithm>
#include <iterator>

inline UmamiStatus orderMetadata( UmamiMetaKey key, void* result )
{
  using namespace UmamiOrders;
  switch( key )
  {
    case UMAMI_META_COUPLING_NAMES:
      *static_cast<const char**>( result ) = names;
      break;
    case UMAMI_META_SQUARED_ORDER_POWERS:
      if( !resolved ) return UMAMI_ERROR_UNSUPPORTED_META;
      std::copy( std::begin( powers ), std::end( powers ), static_cast<int*>( result ) );
      break;
    case UMAMI_META_SQUARED_ORDER_SELECTED:
      std::copy( std::begin( selected ), std::end( selected ), static_cast<bool*>( result ) );
      break;
    case UMAMI_META_COLOR_REPRESENTATIONS:
      if( !physicalLegs ) return UMAMI_ERROR_UNSUPPORTED_META;
      std::copy( std::begin( colors ), std::end( colors ), static_cast<int*>( result ) );
      break;
    case UMAMI_META_CHARGES:
      if( !physicalLegs ) return UMAMI_ERROR_UNSUPPORTED_META;
      std::copy( std::begin( charges ), std::end( charges ), static_cast<double*>( result ) );
      break;
    case UMAMI_META_ALPHA_SCHEME:
      if( !alphaScheme[0] ) return UMAMI_ERROR_UNSUPPORTED_META;
      *static_cast<const char**>( result ) = alphaScheme;
      break;
    case UMAMI_META_TAGGED_PHOTONS:
      *static_cast<int*>( result ) = taggedPhotons;
      break;
    default:
      return UMAMI_ERROR_UNSUPPORTED_META;
  }
  return UMAMI_SUCCESS;
}
