#include "nlo_real_bridge.h"

#include "nlo_real_bridge_config.h"
#include "nlo_real_build_identity.h"
#include "umami.h"

#include <dlfcn.h>

#include <cmath>
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
    std::vector<int> qcd_power, qed_power;
    std::string error;
  };

  struct BridgeContext
  {
    mg7_trace_input_fn trace_input = nullptr;
    mg7_trace_output_fn trace_output = nullptr;
    std::vector<RealLibrary> reals;
    std::vector<unsigned int> flavours;
    std::vector<double> units;
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
    return path + "libmadmatrix" + MG7_NLO_REAL_LIBTAG + "_" +
      MG7_NLO_LIBRARY_MODEL + "_" + process + "_" + backend + "_" +
      MG7_NLO_REAL_FPTYPE + ".so";
  }

  std::string relative_to_bridge( const char* relative )
  {
    Dl_info info{};
    if( dladdr( reinterpret_cast<const void*>( &mg7_nlo_real_initialize ),
                &info ) == 0 || !info.dli_fname )
      throw std::runtime_error( "cannot resolve NLO real bridge location" );
    std::string path = info.dli_fname;
    const std::string::size_type slash = path.find_last_of( '/' );
    path = slash == std::string::npos ? "." : path.substr( 0, slash );
    return path + "/" + relative;
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
      const std::string selected_param_card = std::string( param_card ) == "auto"
        ? relative_to_bridge( "../../Cards/param_card.dat" ) : param_card;
      const std::string selected_library_dir = std::string( library_dir ) == "auto"
        ? relative_to_bridge( "../../lib" ) : library_dir;
      int loaded = 0;
      for( int index = 0; index < MG7_NLO_REAL_COUNT; ++index )
      {
        const MG7NLORealConfig& config = MG7_NLO_REAL_CONFIGS[index];
        RealLibrary& real = context->reals[index];
        try
        {
          const std::string path = library_path(
            selected_library_dir.c_str(), config.library_process_id,
            selected.c_str() );
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
          if( major != UMAMI_MAJOR_VERSION || minor < 5 )
            throw std::runtime_error( "incompatible UMAMI ABI in " + path );
          if( !fingerprint || std::string( fingerprint ) != config.fingerprint )
            throw std::runtime_error( "process fingerprint mismatch in " + path );
          if( real.squared_order_count != config.squared_order_count ||
              real.particle_count != MG7_NLO_PARTICLE_COUNT )
            throw std::runtime_error( "process dimension mismatch in " + path );
          const char* names = nullptr;
          if( real.get_meta( UMAMI_META_COUPLING_NAMES, &names ) != UMAMI_SUCCESS ||
              !names || std::string( names ) != MG7_NLO_COUPLINGS )
            throw std::runtime_error( "coupling order names mismatch in " + path );
          std::vector<int> powers( real.squared_order_count * MG7_NLO_COUPLING_COUNT );
          if( powers.empty() ||
              real.get_meta( UMAMI_META_SQUARED_ORDER_POWERS, powers.data() ) != UMAMI_SUCCESS )
            throw std::runtime_error( "missing squared-order powers in " + path );
          real.qcd_power.assign( real.squared_order_count, 0 );
          real.qed_power.assign( real.squared_order_count, 0 );
          std::istringstream columns( names );
          for( int column = 0; column < MG7_NLO_COUPLING_COUNT; ++column )
          {
            std::string name;
            std::getline( columns, name, ',' );
            auto first = powers.begin() + column * real.squared_order_count;
            if( name == "QCD" ) real.qcd_power.assign( first, first + real.squared_order_count );
            if( name == "QED" ) real.qed_power.assign( first, first + real.squared_order_count );
          }
          real.flavour_count = config.flavour_count;
          if( initialize( &real.handle, selected_param_card.c_str() ) != UMAMI_SUCCESS ||
              !real.handle )
            throw std::runtime_error( "UMAMI initialization failed for " + path );
          ++loaded;
        }
        catch( const std::exception& error )
        {
          real.error = error.what();
          if( real.handle && real.free_handle ) real.free_handle( real.handle );
          real.handle = nullptr;
          if( real.library ) dlclose( real.library );
          real.library = nullptr;
          real.get_meta = nullptr;
          real.evaluate = nullptr;
          real.free_handle = nullptr;
          real.flavour_count = config.flavour_count;
        }
      }
      if( loaded == 0 )
        throw std::runtime_error(
          "no NLO real libraries are available for backend " + selected );
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

  int mg7_nlo_real_available( const void* opaque, int real_me_id )
  {
    const auto* context = static_cast<const BridgeContext*>( opaque );
    if( !context || real_me_id < 1 || static_cast<size_t>( real_me_id ) > context->reals.size() )
      return 0;
    const RealLibrary& real = context->reals[real_me_id - 1];
    return real.handle && real.evaluate;
  }

  static int evaluate_real( void* opaque, int real_me_id,
                             size_t event_count, const double* momenta,
                             const double* g_strong, const double* g_ew, const int32_t* flavour,
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
      if( !real.handle || !real.evaluate )
        return fail( context, "real ME " + std::to_string( real_me_id ) +
          " is unavailable: " + ( real.error.empty() ?
          std::string( "unknown loader error" ) : real.error ) );
      auto& local_flavour = context->flavours;
      local_flavour.resize( event_count );
      for( size_t event = 0; event < event_count; ++event )
      {
        if( flavour[event] < 0 || flavour[event] >= real.flavour_count )
          return fail( context, "NLO real flavour index is out of range" );
        if( event && flavour[event] != flavour[0] )
          return fail( context, "one NLO real batch must have one local flavour" );
        local_flavour[event] = static_cast<unsigned int>( flavour[event] );
      }
      // Legacy callers provide physical G and take e from the parameter card.
      // The NLO workflow supplies both couplings and requests unit blocks.
      auto& unit = context->units;
      if( g_ew ) unit.resize( event_count, 1. );
      const UmamiInputKey input_keys[] = {
        UMAMI_IN_MOMENTA, UMAMI_IN_G_STRONG, UMAMI_IN_FLAVOR_INDEX, UMAMI_IN_G_EW };
      const void* inputs[] = { momenta, g_ew ? unit.data() : g_strong, local_flavour.data(), unit.data() };
      const UmamiOutputKey output_keys[] = { UMAMI_OUT_SQUARED_ORDERS };
      void* outputs[] = { squared_orders };
      if( context->trace_input && g_ew )
        context->trace_input( event_count, real.particle_count, real_me_id, momenta, g_strong, *g_ew, flavour );
      const UmamiStatus status = real.evaluate(
        real.handle, event_count, event_count, 0, g_ew ? 4 : 3, input_keys, inputs,
        1, output_keys, outputs );
      if( status != UMAMI_SUCCESS )
      {
        std::ostringstream message;
        message << "UMAMI evaluation failed for real ME " << real_me_id
                << " with status " << static_cast<int>( status );
        return fail( context, message.str() );
      }
      if( g_ew )
      {
        for( int order = 0; order < real.squared_order_count; ++order )
        {
          const double ew_weight = std::pow( *g_ew, real.qed_power[order] );
          for( size_t event = 0; event < event_count; ++event )
            squared_orders[order * event_count + event] *=
              std::pow( g_strong[event], real.qcd_power[order] ) * ew_weight;
        }
      }
      if( context->trace_output && g_ew )
        context->trace_output( event_count, real.squared_order_count, squared_orders );
      // Non-finite components are returned unchanged. At degenerate
      // soft/collinear points the retained Fortran real matrix element is
      // itself non-finite, and the caller must see what the Fortran path
      // would give: rejecting the batch made the Fortran wrapper switch
      // backends in the middle of a run.
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

  int mg7_nlo_real_evaluate( void* context, int real_me_id, size_t count,
                             const double* momenta, const double* g_strong,
                             const int32_t* flavour, double* squared_orders )
  {
    return evaluate_real( context, real_me_id, count, momenta, g_strong,
                          nullptr, flavour, squared_orders );
  }

  int mg7_nlo_real_evaluate_ew( void* context, int real_me_id, size_t count,
                                const double* momenta, const double* g_strong,
                                double g_ew, const int32_t* flavour, double* squared_orders )
  {
    return evaluate_real( context, real_me_id, count, momenta, g_strong,
                          &g_ew, flavour, squared_orders );
  }

  void mg7_nlo_real_set_trace( void* opaque, mg7_trace_input_fn input, mg7_trace_output_fn output )
  {
    auto* context = static_cast<BridgeContext*>( opaque );
    if( context )
    {
      context->trace_input = input;
      context->trace_output = output;
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
