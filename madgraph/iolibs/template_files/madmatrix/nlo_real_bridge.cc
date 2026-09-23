#include "nlo_real_bridge.h"

#include "nlo_real_bridge_config.h"
#include "umami.h"

#include <dlfcn.h>

#include <cstdint>
#include <exception>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  typedef UmamiStatus ( *GetMeta )( UmamiMetaKey, void* );
  typedef UmamiStatus ( *Initialize )( UmamiHandle*, const char* );
  typedef UmamiStatus ( *Evaluate )( UmamiHandle, size_t, size_t, size_t,
                                    size_t, const UmamiInputKey*,
                                    const void* const*, size_t,
                                    const UmamiOutputKey*, void* const* );
  typedef UmamiStatus ( *Free )( UmamiHandle );

  struct RealLibrary
  {
    void* library = nullptr;
    UmamiHandle handle = nullptr;
    GetMeta get_meta = nullptr;
    Evaluate evaluate = nullptr;
    Free free_handle = nullptr;
    int squared_order_count = 0;
    int particle_count = 0;
    int flavour_count = 0;
  };

  struct BridgeContext
  {
    std::vector<RealLibrary> reals;
    std::string error;
  };

  thread_local std::string g_last_error;

  template<typename Function>
  Function resolve( void* library, const char* name )
  {
    dlerror();
    void* symbol = dlsym( library, name );
    const char* error = dlerror();
    if( error || !symbol )
      throw std::runtime_error( std::string( "cannot resolve " ) + name +
                                ": " + ( error ? error : "missing symbol" ) );
    return reinterpret_cast<Function>( symbol );
  }

  std::string library_path( const char* directory, const char* process,
                            const char* backend )
  {
    std::string path = directory;
    if( !path.empty() && path.back() != '/' ) path += '/';
    return path + "libmadmatrix_" + process + "_" + backend + ".so";
  }

  void close_context( BridgeContext* context )
  {
    if( !context ) return;
    for( auto& real : context->reals )
    {
      if( real.handle && real.free_handle ) real.free_handle( real.handle );
      real.handle = nullptr;
      if( real.library ) dlclose( real.library );
      real.library = nullptr;
    }
  }

  int fail( BridgeContext* context, const std::string& message )
  {
    if( context ) context->error = message;
    g_last_error = message;
    return 1;
  }
}

extern "C"
{
  int mg7_nlo_real_initialize( void** opaque, const char* param_card,
                               const char* library_dir, const char* backend )
  {
    if( !opaque || *opaque || !param_card || !library_dir || !backend )
      return fail( nullptr, "invalid NLO real initialization arguments" );
    BridgeContext* context = nullptr;
    try
    {
      context = new BridgeContext();
      context->reals.resize( MG7_NLO_REAL_COUNT );
      const std::string selected = std::string( backend ) == "auto"
                                     ? "scalar" : backend;
      for( int index = 0; index < MG7_NLO_REAL_COUNT; ++index )
      {
        const MG7NLORealConfig& config = MG7_NLO_REAL_CONFIGS[index];
        RealLibrary& real = context->reals[index];
        const std::string path = library_path(
          library_dir, config.library_process_id, selected.c_str() );
        real.library = dlopen( path.c_str(), RTLD_NOW | RTLD_LOCAL );
        if( !real.library )
          throw std::runtime_error( "cannot load " + path + ": " + dlerror() );
        real.get_meta = resolve<GetMeta>( real.library, "umami_get_meta" );
        Initialize initialize = resolve<Initialize>(
          real.library, "umami_initialize" );
        real.evaluate = resolve<Evaluate>( real.library,
                                          "umami_matrix_element" );
        real.free_handle = resolve<Free>( real.library, "umami_free" );

        int major = 0;
        int minor = 0;
        const char* fingerprint = nullptr;
        if( real.get_meta( UMAMI_META_ABI_MAJOR_VERSION, &major ) != UMAMI_SUCCESS ||
            real.get_meta( UMAMI_META_ABI_MINOR_VERSION, &minor ) != UMAMI_SUCCESS ||
            real.get_meta( UMAMI_META_PROCESS_FINGERPRINT, &fingerprint ) != UMAMI_SUCCESS ||
            real.get_meta( UMAMI_META_SQUARED_ORDER_COUNT,
                           &real.squared_order_count ) != UMAMI_SUCCESS ||
            real.get_meta( UMAMI_META_PARTICLE_COUNT,
                           &real.particle_count ) != UMAMI_SUCCESS )
          throw std::runtime_error( "incomplete UMAMI metadata in " + path );
        if( major != UMAMI_MAJOR_VERSION || minor < 1 )
          throw std::runtime_error( "incompatible UMAMI ABI in " + path );
        if( !fingerprint || std::string( fingerprint ) != config.fingerprint )
          throw std::runtime_error( "process fingerprint mismatch in " + path );
        if( real.squared_order_count != config.squared_order_count ||
            real.particle_count != MG7_NLO_PARTICLE_COUNT )
          throw std::runtime_error( "process dimension mismatch in " + path );
        real.flavour_count = config.flavour_count;
        if( initialize( &real.handle, param_card ) != UMAMI_SUCCESS ||
            !real.handle )
          throw std::runtime_error( "UMAMI initialization failed for " + path );
      }
      *opaque = context;
      return 0;
    }
    catch( const std::exception& error )
    {
      const std::string message = error.what();
      close_context( context );
      delete context;
      return fail( nullptr, message );
    }
    catch( ... )
    {
      close_context( context );
      delete context;
      return fail( nullptr, "unknown exception while initializing NLO real bridge" );
    }
  }

  int mg7_nlo_real_evaluate( void* opaque, int real_me_id,
                             size_t event_count, const double* momenta,
                             const double* g_strong, const int32_t* flavour,
                             double* squared_orders )
  {
    BridgeContext* context = static_cast<BridgeContext*>( opaque );
    try
    {
      if( !context || real_me_id < 1 ||
          real_me_id > static_cast<int>( context->reals.size() ) )
        return fail( context, "invalid NLO real matrix-element id" );
      if( event_count == 0 ) return 0;
      if( !momenta || !g_strong || !flavour || !squared_orders )
        return fail( context, "null NLO real evaluation buffer" );
      RealLibrary& real = context->reals[real_me_id - 1];
      std::vector<unsigned int> local_flavour( event_count );
      for( size_t event = 0; event < event_count; ++event )
      {
        if( flavour[event] < 0 || flavour[event] >= real.flavour_count )
          return fail( context, "NLO real flavour index is out of range" );
        if( event && flavour[event] != flavour[0] )
          return fail( context, "one NLO real batch must have one local flavour" );
        local_flavour[event] = static_cast<unsigned int>( flavour[event] );
      }
      const UmamiInputKey input_keys[] = {
        UMAMI_IN_MOMENTA, UMAMI_IN_G_STRONG, UMAMI_IN_FLAVOR_INDEX };
      const void* inputs[] = { momenta, g_strong, local_flavour.data() };
      const UmamiOutputKey output_keys[] = { UMAMI_OUT_SQUARED_ORDERS };
      void* outputs[] = { squared_orders };
      const UmamiStatus status = real.evaluate(
        real.handle, event_count, event_count, 0, 3, input_keys, inputs,
        1, output_keys, outputs );
      if( status != UMAMI_SUCCESS )
      {
        std::ostringstream message;
        message << "UMAMI evaluation failed for real ME " << real_me_id
                << " with status " << static_cast<int>( status );
        return fail( context, message.str() );
      }
      context->error.clear();
      return 0;
    }
    catch( const std::exception& error )
    {
      return fail( context, error.what() );
    }
    catch( ... )
    {
      return fail( context, "unknown exception in NLO real evaluation" );
    }
  }

  int mg7_nlo_real_finalize( void** opaque )
  {
    if( !opaque ) return fail( nullptr, "null NLO real context pointer" );
    BridgeContext* context = static_cast<BridgeContext*>( *opaque );
    close_context( context );
    delete context;
    *opaque = nullptr;
    return 0;
  }

  const char* mg7_nlo_real_last_error( void* opaque )
  {
    BridgeContext* context = static_cast<BridgeContext*>( opaque );
    return context ? context->error.c_str() : g_last_error.c_str();
  }
}
