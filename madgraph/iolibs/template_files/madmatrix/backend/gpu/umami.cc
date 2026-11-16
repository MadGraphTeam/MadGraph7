// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: T. Heimel (Nov 2025) for the MadGraph7 CUDACPP plugin.
// Further modified by: D. Massaro (2026).
// Integrated with the MadGraph7 project in Feb 2026.

#include "umami.h"

#include "ProcessData.h"
#include "ProcessTables.h" // the crossing rows (choose_good_helicities)
#include "CPPProcess.h" // needed to construct/initProc the process object (umami_initialize)
#include "SigmaKin.h" // sigmaKin_getGoodHel/setGoodHel
#include "GpuRuntime.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "rambo.h" // rambo::has_phase_space (keep_every_helicity)

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <utility>

using namespace madmatrix;

namespace
{

  // The per-diagram channel weight handed to madspace as amp2. A subprocess whose
  // matrix element is identically zero -- an FCNC channel with every Wilson
  // coefficient at zero, say -- has all numerators at zero, so their sum, the
  // denominator, is zero too. Dividing would give 0/0 = nan for every diagram, and
  // that nan becomes the amp2 madspace builds its channel weights from, turning a
  // channel that should simply contribute nothing into a non-finite event weight
  // that aborts the whole integration.
  //
  // The denominator is a sum of |amp|^2 and so never negative: adding the smallest
  // normal double floors it away from zero (0/tiny is 0, no channel preferred) while
  // leaving every denominator a real amplitude produces bit-for-bit unchanged. This
  // must NOT be written as a test for zero -- this is compiled with -ffast-math, and
  // its -ffinite-math-only lets the compiler assume the quotient is finite and drop
  // such a guard as dead code (the trap behind #117 and #516).
  //
  // This is called from both the host and the device code paths, hence the
  // __host__ __device__ decoration in GPU builds. DBL_MIN is used rather than
  // std::numeric_limits<double>::min(), which is a host function and cannot be
  // called from device code.
#ifdef MGONGPUCPP_GPUIMPL
  __host__ __device__
#endif
    inline double
    channel_amp2( double numerator, double denominator )
  {
    return numerator / ( denominator + DBL_MIN );
  }


  __device__
    void
    transpose_momenta( const double* momenta_in, fptype_momenta* momenta_out, std::size_t i_event_in, std::size_t i_event_out, std::size_t stride )
  {
    std::size_t page_size = MemoryAccessMomentaBase::neppM;
    std::size_t i_page = i_event_out / page_size;
    std::size_t i_vector = i_event_out % page_size;

    for( std::size_t i_part = 0; i_part < ProcessData::npar; ++i_part )
    {
      for( std::size_t i_mom = 0; i_mom < 4; ++i_mom )
      {
        momenta_out[i_page * ProcessData::npar * 4 * page_size +
                    i_part * 4 * page_size + i_mom * page_size + i_vector] = momenta_in[stride * ( ProcessData::npar * i_mom + i_part ) + i_event_in];
      }
    }
  }


  __global__ void copy_inputs(
    const double* momenta_in,
    const double* helicity_random_in,
    const double* color_random_in,
    const double* diagram_random_in,
    const double* alpha_s_in,
    const unsigned int* flavor_indices_in,
    const int* helicity_in,
    fptype_momenta* momenta,
    fptype* helicity_random,
    fptype* color_random,
    fptype* diagram_random,
    fptype* g_s,
    unsigned int* flavor_indices,
    int* helicity_given,
    std::size_t count,
    std::size_t stride,
    std::size_t offset )
  {
    // One thread per event of the rounded (padded) batch: sigmaKin runs on all
    // n_blocks * n_threads events, not only on the first count. The padding events
    // are copies of the first event. Left uninitialized, their flavor index is
    // whatever the fresh allocation holds, and cFlavors[iflavor] in the matrix
    // element then reads far out of bounds (illegal memory access on AMD GPUs).
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    std::size_t i_in = i_event < count ? i_event : 0;

    transpose_momenta( &momenta_in[offset], momenta, i_in, i_event, stride );
    diagram_random[i_event] = diagram_random_in ? diagram_random_in[i_in + offset] : 0.5;
    helicity_random[i_event] = helicity_random_in ? helicity_random_in[i_in + offset] : 0.5;
    color_random[i_event] = color_random_in ? color_random_in[i_in + offset] : 0.5;
    g_s[i_event] = alpha_s_in ? sqrt( 4 * M_PI * alpha_s_in[i_in + offset] ) : 1.2177157847767195;
    // With crossing, an extended id K*nmaxflavor + flav (K a row of the crossing
    // table) is evaluated as that crossing by sigmaKin; one past the table gets a
    // zero |M|^2, as on cpu.
    flavor_indices[i_event] = flavor_indices_in ? flavor_indices_in[i_in + offset] : 0;
    if( helicity_in )
    {
      // an index out of [0,ncomb) cannot be reported from here (umami_matrix_element returns
      // before the kernels run): evaluate helicity 0 instead, never out of the helicity table,
      // and let copy_outputs flag the event (see umami_matrix_element)
      const int ihel = helicity_in[i_in + offset];
      helicity_given[i_event] = ( ihel >= 0 && ihel < ProcessData::ncomb ? ihel : 0 );
    }
  }

  __global__ void copy_outputs(
    fptype_amp* denominators,
    fptype_amp* numerators,
    fptype* matrix_elements,
    unsigned int* diagram_index,
    int* color_index,
    int* helicity_index,
    double* m2_out,
    double* amp2_out,
    int* diagram_out,
    int* color_out,
    int* helicity_out,
    const int* helicity_in,
    std::size_t count,
    std::size_t stride,
    std::size_t offset )
  {
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_event >= count ) return;
    if( helicity_in )
    {
      const int ihel = helicity_in[i_event + offset];
      if( ihel < 0 || ihel >= ProcessData::ncomb )
      {
        // an invalid given helicity: no |M|^2 and no helicity, see umami_matrix_element
        if( m2_out ) m2_out[i_event + offset] = 0;
        if( helicity_out ) helicity_out[i_event + offset] = -1;
        return;
      }
    }

    if( m2_out ) m2_out[i_event + offset] = matrix_elements[i_event];
    if( amp2_out )
    {
      double denominator = denominators[i_event];
      for( std::size_t i_diag = 0; i_diag < ProcessData::ndiagrams; ++i_diag )
      {
        amp2_out[stride * i_diag + i_event + offset] = channel_amp2( numerators[i_event * ProcessData::ndiagrams + i_diag], denominator );
      }
    }
    if( diagram_out ) diagram_out[i_event + offset] = diagram_index[i_event] - 1;
    if( color_out ) color_out[i_event + offset] = color_index[i_event] - 1;
    if( helicity_out ) helicity_out[i_event + offset] = helicity_index[i_event] - 1;
  }


  // What a UMAMI handle points to. Everything the matrix element of an instance depends
  // on is in here, and not shared with the other instances of the library: the
  // parameters (its CPPProcess), the good helicities and the device it lives on.
  struct InterfaceInstance
  {
    // The model parameters, and the parameters the matrix element is evaluated with
    CPPProcess process;
    // The latter on the device, when the kernels do not take them by value
    DeviceProcessParameters deviceParams;
    // The device the instance was created on (the current one in umami_initialize)
    int device = 0;
    // The good helicities of process.parameters(), chosen on the first events the
    // instance evaluates (nlearn) or set from the cards (madmatrix_set_helicity_choice):
    // how many of them it has looked at, and the helicities that contribute to one of them
    std::size_t learned = 0;
    bool good_hel[ProcessData::ncomb] = {};
    // where RAMBO has no phase space (a decay below threshold), every helicity is kept
    bool keep_every_helicity = false;
    // parameter interference_helicity_summed (run card [generation]
    // interference_helicity = "summed"): an interference |M|^2 stays the helicity
    // sum even when the caller draws the helicity (see umami_matrix_element)
    bool summed_helicity = false;
  };

  // Makes the device of an instance the current one for the duration of a call on it,
  // so that its kernels run there whatever the caller has as its current device
  class DeviceGuard
  {
  public:
    DeviceGuard( const int device )
    {
      checkGpu( gpuGetDevice( &m_previous ) );
      if( m_previous != device ) checkGpu( gpuSetDevice( device ) );
      m_device = device;
    }
    ~DeviceGuard()
    {
      if( m_previous != m_device ) gpuSetDevice( m_previous );
    }
    DeviceGuard( const DeviceGuard& ) = delete;
    DeviceGuard& operator=( const DeviceGuard& ) = delete;
  private:
    int m_previous;
    int m_device;
  };

  // The masses of the most recently initialized or modified instance, for
  // umami_get_meta(UMAMI_META_MASSES), which takes no handle
  std::vector<double> g_externalMasses;

  // The good helicities are chosen at run time, on the events the instance is
  // called with, as in the Fortran standalone output and in the cpu and simd
  // backends (see nlearn there): every helicity is evaluated on the first nlearn
  // events, and the helicities that contribute to one of them are kept for the rest
  // of the run, or until umami_set_parameter changes a parameter. So they are chosen
  // at the energies, and in the frame, that the matrix element is used at (chosen at
  // initialization on RAMBO points at 1.5 TeV, the all-plus helicities of g g > h g
  // in the heft model were dropped, 42% of |M|^2 at 130 GeV), and a first call of
  // special points (one forward u u~ > e+ e- event) is one of nlearn events only.
  // With crossing, the scan evaluates every row of the crossing table on these
  // events, into the one list they share. sigmaKin_getGoodHel samples the first
  // goodHelScanEvents (16) events of a call. Where there is no phase space (a decay
  // below threshold), every helicity is kept (InterfaceInstance::keep_every_helicity,
  // see the cpu backend). An mg7 run gives every instance the choice a CPU build
  // makes from the cards instead (madmatrix_set_helicity_choice).
  constexpr std::size_t nlearn = goodHelScanEvents;

  // Where the base process has no phase space at the masses of the instance
  bool no_phase_space( InterfaceInstance* instance )
  {
    const std::vector<fptype>& instanceMasses = instance->process.getMasses();
    const std::vector<double> masses( instanceMasses.begin(), instanceMasses.end() );
    return !rambo::has_phase_space( ProcessData::npari, masses );
  }

  // Keep every helicity of an instance (InterfaceInstance::keep_every_helicity)
  void keep_all_helicities( InterfaceInstance* instance )
  {
    for( bool& good : instance->good_hel ) good = true;
    instance->learned = nlearn;
    ProcessParameters& params = instance->process.parameters();
    sigmaKin_setGoodHel( params, instance->good_hel );
    instance->deviceParams.update( params );
  }

  // Add the helicities that contribute to the first events of a call to those of
  // the instance. Returns the number of events of the call it looked at.
  int learn_good_helicities(
    InterfaceInstance* instance,
    const fptype_momenta* momenta,
    const fptype* couplings,
    const unsigned int* flavor_indices,
    fptype* matrix_elements,
    fptype_amp* color_jamps,
    fptype_amp* numerators,
    fptype_amp* denominators,
    std::size_t count )
  {
    ProcessParameters& params = instance->process.parameters();
    bool is_good_hel[ProcessData::ncomb];
    const int scanned = sigmaKin_getGoodHel(
      instance->deviceParams.arg( params ), momenta, couplings, flavor_indices, matrix_elements, numerators, denominators,
      color_jamps,
      is_good_hel,
      count );
    for( int ihel = 0; ihel < ProcessData::ncomb; ihel++ )
      if( is_good_hel[ihel] ) instance->good_hel[ihel] = true;
    sigmaKin_setGoodHel( params, instance->good_hel );
    instance->deviceParams.update( params );
    return scanned;
  }

  // Forget the good helicities of an instance: the next events choose them again
  void reset_good_helicities( InterfaceInstance* instance )
  {
    instance->learned = 0;
    for( bool& good : instance->good_hel ) good = false;
    instance->keep_every_helicity = no_phase_space( instance );
  }

  // nevt RAMBO points of a 2 > n (n >= 2) process of external masses `masses` at a
  // collider of centre-of-mass energy sqrt_s, whose frame moves at `rapidity` along
  // the beams in the lab (asymmetric beams), in the layout of
  // rambo::helicity_scan_points (momenta[nevt * ( npar * mu + ipar ) + ievt]); none
  // if the process is closed at sqrt_s or RAMBO does not converge. The points come
  // in pairs, one in the partonic centre-of-mass frame and one in the lab frame,
  // since the matrix element is called in either (me_frame) and the helicity of a
  // massive leg is not boost invariant: a helicity can vanish in one frame only.
  // With `spread`, the partonic energy varies (PDFs, ISR, beamstrahlung): a pair at
  // each of nevt/2 energies, from just above the threshold of the process up to
  // sqrt_s in geometric steps, since a helicity whose |M|^2 is a power of
  // m/sqrt(s) below the others matters near threshold only (the all-plus
  // helicities of g g > h g in the heft model are (m_H^2/s)^4 of the rest: 1e-14 at
  // 13 TeV, 42% at 130 GeV). With massless external legs there is no threshold,
  // the helicities scale alike up to the internal masses, and the lowest energy is
  // sqrt_s/100. The partonic system of a lab-frame point moves by a random rapidity
  // of at most max_boost, or, if max_boost is negative, of at most
  // log( sqrt_s / energy ), the most that the beams boost a system of that energy.
  // Without `spread`, every point is at sqrt_s.
  std::vector<double> collider_scan_points( const std::vector<double>& masses, std::size_t nevt, double sqrt_s, double rapidity, bool spread, double max_boost )
  {
    constexpr int npar = ProcessData::npar;
    constexpr int npari = 2;
    double mfinal = 0, mtotal = 0;
    for( int ipar = 0; ipar < npar; ++ipar )
    {
      mtotal += masses[ipar];
      if( ipar >= npari ) mfinal += masses[ipar];
    }
    const double threshold = std::max( mtotal - mfinal, mfinal ); // incoming or outgoing masses
    // rambo exits on a closed phase space: check it first
    if( !( sqrt_s > threshold ) ) return {};
    const double ehigh = sqrt_s;
    double elow = ehigh;
    if( spread ) elow = threshold > 0 ? std::min( 1.1 * threshold, 0.5 * ( threshold + ehigh ) ) : ehigh / 100;
    const std::size_t nenergy = nevt / 2;
    // A private generator: the shared one (rambo::rn) makes the points of the check
    // drivers, which must not move.
    rambo::Random random;
    random.rmarin( 1802, 9373 );
    auto draw = [&random]( int )
    {
      double r = random.ranmar();
      while( r <= 1e-16 ) r = random.ranmar();
      return r;
    };
    std::vector<double> momenta( nevt * npar * 4 ); // momenta[nevt * ( npar * mu + ipar ) + ievt]
    for( std::size_t ievt = 0; ievt < nevt; ++ievt )
    {
      const double energy = elow * std::pow( ehigh / elow, double( ievt / 2 ) / ( nenergy - 1 ) );
      double weight = 0;
      std::vector<std::vector<double>> p = rambo::get_momenta( npari, energy, masses, weight, draw );
      double y = 0; // the partonic centre-of-mass frame, or the lab frame:
      if( ievt % 2 == 1 )
      {
        y = rapidity;
        if( max_boost != 0 ) y += ( max_boost < 0 ? std::log( ehigh / energy ) : max_boost ) * ( 2 * draw( 0 ) - 1 );
      }
      if( y != 0 )
      {
        const double ch = std::cosh( y ), sh = std::sinh( y );
        for( std::vector<double>& q : p )
        {
          const double e = q[0], pz = q[3];
          q[0] = e * ch + pz * sh;
          q[3] = pz * ch + e * sh;
        }
      }
      for( int ipar = 0; ipar < npar; ++ipar )
      {
        // on shell, or rambo did not converge: no points
        const double m2 = p[ipar][0] * p[ipar][0] - p[ipar][1] * p[ipar][1] - p[ipar][2] * p[ipar][2] - p[ipar][3] * p[ipar][3];
        if( std::abs( m2 - masses[ipar] * masses[ipar] ) > 1e-6 * p[ipar][0] * p[ipar][0] ) return {};
        for( int mu = 0; mu < 4; ++mu ) momenta[nevt * ( npar * mu + ipar ) + ievt] = p[ipar][mu];
      }
    }
    return momenta;
  }

  // The good-helicity scan points of choose_good_helicities: nlearn points per row
  // of the crossing table, block K (events K*nlearn on) at the masses of crossing K
  // -- its input slot k carries the mass of the base slot it is fed to -- and with
  // its extended id K*nmaxflavor, which is how sigmaKin_getGoodHel finds it, made
  // by points( masses ). The base points serve a crossing that moves no mass. A
  // crossing that does needs its own: a massive leg in a massless slot is off shell
  // on the base points, where a row it needs can vanish to rounding. A crossing
  // with no phase space at its masses (t > b e+ ve crossed into b~ > t~ e+ ve) is
  // never evaluated on shell and needs no row: getGoodHel skips it
  // (sigmaKin_setNoPhaseSpace), as the fortran warm-up does, and its block holds
  // the base points under the base id. One that `points` gives none for (closed at
  // the collider energy, RAMBO did not converge) gets them too, which getGoodHel
  // scans the conservative way. Returns the number of events, 0 when `points`
  // gives none for the base process. masses0 are the external masses of the
  // instance (in the base slot order).
  template<typename Points>
  std::size_t scan_points( const std::vector<double>& masses0, Points points, std::vector<double>& momenta, std::vector<unsigned int>& flavor_indices )
  {
    constexpr std::size_t nevt = nlearn;
    constexpr int npar = ProcessData::npar;
    constexpr int nblock = ProcessTables::use_crossing ? ProcessTables::ncross : 1;
    sigmaKin_setNoPhaseSpace( -1 ); // the rows of this instance only
    const std::vector<double> base = points( masses0 );
    if( base.empty() ) return 0;
    const std::size_t count = nblock * nevt;
    momenta.assign( count * npar * 4, 0. );
    flavor_indices.assign( count, 0 );
    for( int cross = 0; cross < nblock; ++cross )
    {
      std::vector<double> own;
      if( cross > 0 && !ProcessTables::xonshell_tab[cross] )
      {
        std::vector<double> masses( npar );
        for( int k = 0; k < npar; ++k ) masses[k] = masses0[ProcessTables::xmass_slot_tab[cross * npar + k]];
        if( rambo::has_phase_space( ProcessData::npari, masses ) )
          own = points( masses );
        else
          sigmaKin_setNoPhaseSpace( cross );
      }
      const bool use_base = own.empty();
      const std::vector<double>& block = use_base ? base : own;
      const bool own_id = cross == 0 || ProcessTables::xonshell_tab[cross] || !use_base;
      for( std::size_t ievt = 0; ievt < nevt; ++ievt )
      {
        const std::size_t jevt = cross * nevt + ievt;
        flavor_indices[jevt] = own_id ? cross * ProcessData::nmaxflavor : 0;
        for( int i = 0; i < npar * 4; ++i ) momenta[count * i + jevt] = block[nevt * i + ievt];
      }
    }
    return count;
  }

  // Choose the good helicities of an instance at once, on RAMBO points of a
  // collider (scan_points), as the cpu and simd backends do (see there): the points
  // are made on the host and copied to the device, and evaluated on the default
  // stream. Where the process has no phase space at all, every helicity is kept.
  // Returns false, and chooses nothing, where the points do not reach it: the first
  // events choose then.
  bool choose_good_helicities( InterfaceInstance* instance, double sqrt_s, double rapidity, bool spread, double max_boost )
  {
    DeviceGuard guard( instance->device );
    const std::vector<fptype>& instanceMasses = instance->process.getMasses();
    const std::vector<double> masses( instanceMasses.begin(), instanceMasses.end() );
    reset_good_helicities( instance );
    if( instance->keep_every_helicity )
    {
      keep_all_helicities( instance );
      checkGpu( gpuDeviceSynchronize() );
      return true;
    }
    std::vector<double> momenta;
    std::vector<unsigned int> flavor_indices;
    std::size_t count = 0;
    if( ProcessData::npari == 2 && ProcessData::npar > 3 )
      count = scan_points(
        masses, [&]( const std::vector<double>& m ) { return collider_scan_points( m, nlearn, sqrt_s, rapidity, spread, max_boost ); },
        momenta, flavor_indices );
    else
      count = scan_points(
        masses, []( const std::vector<double>& m ) { return rambo::helicity_scan_points( ProcessData::npari, m, nlearn ); },
        momenta, flavor_indices );
    if( count == 0 ) return false;
    double* device_momenta = nullptr;
    unsigned int* device_flavor_indices = nullptr;
    double* device_matrix_elements = nullptr;
    gpuMalloc( (void**)&device_momenta, momenta.size() * sizeof( double ) );
    gpuMalloc( (void**)&device_flavor_indices, count * sizeof( unsigned int ) );
    gpuMalloc( (void**)&device_matrix_elements, count * sizeof( double ) );
    gpuMemcpy( device_momenta, momenta.data(), momenta.size() * sizeof( double ), gpuMemcpyHostToDevice );
    gpuMemcpy( device_flavor_indices, flavor_indices.data(), count * sizeof( unsigned int ), gpuMemcpyHostToDevice );
    // no UMAMI_OUT_GPU_STREAM: the default stream, as the filter's own copies
    const UmamiInputKey input_keys[] = { UMAMI_IN_MOMENTA, UMAMI_IN_FLAVOR_INDEX };
    const void* inputs[] = { device_momenta, device_flavor_indices };
    const UmamiOutputKey output_keys[] = { UMAMI_OUT_MATRIX_ELEMENT };
    void* outputs[] = { device_matrix_elements };
    umami_matrix_element( instance, count, count, 0, 2, input_keys, inputs, 1, output_keys, outputs );
    // umami_matrix_element returns before its kernels end: wait before freeing
    checkGpu( gpuDeviceSynchronize() );
    gpuFree( device_matrix_elements );
    gpuFree( device_flavor_indices );
    gpuFree( device_momenta );
    instance->learned = nlearn;
    return true;
  }

}

extern "C"
{
  UmamiStatus umami_get_meta( UmamiMetaKey meta_key, void* result )
  {
    switch( meta_key )
    {
      case UMAMI_META_DEVICE:
      {
        UmamiDevice& device = *static_cast<UmamiDevice*>( result );
#ifdef __CUDACC__
        device = UMAMI_DEVICE_CUDA;
#elif defined( __HIPCC__ )
        device = UMAMI_DEVICE_HIP;
#endif
        break;
      }
      case UMAMI_META_PARTICLE_COUNT:
        *static_cast<int*>( result ) = ProcessData::npar;
        break;
      case UMAMI_META_DIAGRAM_COUNT:
        *static_cast<int*>( result ) = ProcessData::ndiagrams;
        break;
      case UMAMI_META_HELICITY_COUNT:
        *static_cast<int*>( result ) = ProcessData::ncomb;
        break;
      case UMAMI_META_COLOR_COUNT:
        return UMAMI_ERROR_UNSUPPORTED_META;
      case UMAMI_META_MASSES:
      {
        if( g_externalMasses.size() != (size_t)ProcessData::npar ) return UMAMI_ERROR_UNINITIALIZED_META;

        for( int ipar = 0; ipar < ProcessData::npar; ++ipar )
          static_cast<double*>( result )[ipar] = g_externalMasses[ipar];
        break;
      }
      default:
        return UMAMI_ERROR_UNSUPPORTED_META;
    }
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_supported_inputs( bool const** supported, int* count )
  {
    // MOMENTA, ALPHA_S, FLAVOR_INDEX, RANDOM_COLOR, RANDOM_HELICITY, RANDOM_DIAGRAM,
    // HELICITY_INDEX, DIAGRAM_INDEX=true, CHANNEL_INDEX=false
    static const bool data[UMAMI_INPUT_KEY_COUNT] = { true, true, true, true, true, true, true, true };
    *supported = data;
    *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_required_inputs( bool const** required, int* count )
  {
    static const bool data[UMAMI_INPUT_KEY_COUNT] = { true }; // MOMENTA only
    *required = data;
    *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_supported_outputs( bool const** supported, int* count )
  {
    // MATRIX_ELEMENT, DIAGRAM_AMP2, COLOR_INDEX, HELICITY_INDEX, DIAGRAM_INDEX, GPU_STREAM
    static const bool data[UMAMI_OUTPUT_KEY_COUNT] = { true, true, true, true, true, true };
    *supported = data;
    *count = UMAMI_OUTPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_initialize( UmamiHandle* handle, char const* param_card_path )
  {
    // The instance lives on the current device
    std::unique_ptr<InterfaceInstance> instance( new InterfaceInstance() );
    checkGpu( gpuGetDevice( &instance->device ) );
    instance->process.initProc( param_card_path );
    instance->deviceParams.update( instance->process.parameters() );

    const std::vector<fptype>& masses = instance->process.getMasses();
    g_externalMasses.assign( masses.begin(), masses.end() );
    reset_good_helicities( instance.get() ); // keep_every_helicity

    *handle = instance.release();
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_set_parameter(
    UmamiHandle handle,
    char const* name,
    double parameter_real,
    [[maybe_unused]] double parameter_imag )
  {
    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    DeviceGuard guard( instance->device );
    // The run card bw_cutoff (the $-excluded propagator window). Per instance, like
    // the model parameters read by umami_initialize.
    if( std::string( name ) == "bwcutoff" )
    {
      instance->process.parameters().bwcutoff = parameter_real;
      instance->deviceParams.update( instance->process.parameters() );
      return UMAMI_SUCCESS;
    }
    // Non-zero: the helicity choice of an interference |M|^2 keeps the helicity
    // sum as the |M|^2 (lower variance), and the chosen helicity is then not
    // meaningful (mg7 writes 9 in the LHE). Zero, the default: the madevent
    // convention, see umami_matrix_element. Per instance.
    if( std::string( name ) == "interference_helicity_summed" )
    {
      instance->summed_helicity = parameter_real != 0;
      return UMAMI_SUCCESS;
    }
    // The model parameters are the instance's own (those of its CPPProcess)
#ifdef MGONGPU_HARDCODE_PARAM
    // Hardcoded parameters are compile-time constants: there is nothing to set.
    return UMAMI_ERROR_NOT_IMPLEMENTED;
#else
    if( !instance->process.modelParameters().setParameterByName( name, parameter_real, parameter_imag ) )
      return UMAMI_ERROR_UNKNOWN_PARAMETER;
    // sigmaKin() and friends read copies of the independent parameters/couplings
    // (cIPD/cIPC/cIPF/bsmIndepParam) and of the masses, not the model parameters
    // themselves, for performance; refresh them so that the override takes effect.
    instance->process.refreshIndependentParams();
    instance->deviceParams.update( instance->process.parameters() );
    const std::vector<fptype>& masses = instance->process.getMasses();
    g_externalMasses.assign( masses.begin(), masses.end() );
    // A helicity may vanish with some parameters and not with others: choose again
    reset_good_helicities( instance );
    return UMAMI_SUCCESS;
#endif
  }

  UmamiStatus umami_get_parameter(
    [[maybe_unused]] UmamiHandle handle,
    [[maybe_unused]] char const* name,
    [[maybe_unused]] double* parameter_real,
    [[maybe_unused]] double* parameter_imag )
  {
#ifdef MGONGPU_HARDCODE_PARAM
    return UMAMI_ERROR_NOT_IMPLEMENTED;
#else
    if( !static_cast<InterfaceInstance*>( handle )->process.modelParameters().getParameterByName( name, parameter_real, parameter_imag ) )
      return UMAMI_ERROR_UNKNOWN_PARAMETER;
    return UMAMI_SUCCESS;
#endif
  }

  // Numbering of the outputs of a crossed call (crossing on, flavor index
  // K*nmaxflavor + flavor: row K of the crossing table in ProcessTables.h).
  // - UMAMI_OUT_HELICITY_INDEX is the crossed process's own helicity code,
  //   the one its --use_crossing=False output reports (selected_hel_code);
  //   UMAMI_IN_HELICITY_INDEX takes the same code (base_row_of_code).
  // - UMAMI_OUT_COLOR_INDEX, UMAMI_OUT_DIAGRAM_INDEX and UMAMI_OUT_DIAGRAM_AMP2
  //   number the colour flows and diagrams of the BASE subprocess (this P
  //   directory's). To read a flow on the crossed legs, input slot k is base
  //   leg xpinv_tab[K*npar+k], with colour and anticolour exchanged when
  //   xsgni_tab is -1 (the leg changed side). This is not the numbering of the
  //   --use_crossing=False output: two rows can serve the same flavours with
  //   different leg assignments, so one base flow is a different flow of the
  //   crossed process under each. `output mg7` decodes through the base this
  //   way (export_mg7.py, get_crossed_subprocess_info).
  UmamiStatus umami_matrix_element(
    UmamiHandle handle,
    size_t count,
    size_t stride,
    size_t offset,
    size_t input_count,
    UmamiInputKey const* input_keys,
    void const* const* inputs,
    size_t output_count,
    UmamiOutputKey const* output_keys,
    void* const* outputs )
  {
    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    DeviceGuard guard( instance->device );
    ProcessParameters& params = instance->process.parameters();

    const double* momenta_in = nullptr;
    const double* alpha_s_in = nullptr;
    const unsigned int* flavor_indices_in = nullptr;
    const double* random_color_in = nullptr;
    const double* random_helicity_in = nullptr;
    const double* random_diagram_in = nullptr;
    const int* helicity_in = nullptr;
    [[maybe_unused]] const int* diagram_in = nullptr; // TODO: unused

    for( std::size_t i = 0; i < input_count; ++i )
    {
      const void* input = inputs[i];
      switch( input_keys[i] )
      {
        case UMAMI_IN_MOMENTA:
          momenta_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_ALPHA_S:
          alpha_s_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_FLAVOR_INDEX:
          flavor_indices_in = static_cast<const unsigned int*>( input );
          break;
        case UMAMI_IN_RANDOM_COLOR:
          random_color_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_RANDOM_HELICITY:
          random_helicity_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_RANDOM_DIAGRAM:
          random_diagram_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_HELICITY_INDEX:
          helicity_in = static_cast<const int*>( input );
          break;
        case UMAMI_IN_DIAGRAM_INDEX:
          diagram_in = static_cast<const int*>( input );
          break;
        default:
          return UMAMI_ERROR_UNSUPPORTED_INPUT;
      }
    }
    if( !momenta_in ) return UMAMI_ERROR_MISSING_INPUT;
    // UMAMI_IN_HELICITY_INDEX: evaluate each event for the one helicity it is given, an index
    // into the full helicity table [0,ncomb) (the UMAMI_OUT_HELICITY_INDEX convention), instead
    // of summing over the helicities. The |M|^2 is then the contribution of that helicity to
    // the helicity-summed (and averaged) |M|^2, so that the sum over all ncomb helicities gives
    // back the |M|^2 without this input; UMAMI_OUT_HELICITY_INDEX returns the given helicity,
    // the color and diagram are chosen for it, and UMAMI_IN_RANDOM_HELICITY is not used. The
    // indices live on the device and are only read by the kernels, so that an index out of
    // [0,ncomb) cannot fail the call as on the CPU: that event gets |M|^2 = 0 and helicity -1.

    gpuStream_t gpu_stream = nullptr;
    double* m2_out = nullptr;
    double* amp2_out = nullptr;
    int* diagram_out = nullptr;
    int* color_out = nullptr;
    int* helicity_out = nullptr;
    for( std::size_t i = 0; i < output_count; ++i )
    {
      void* output = outputs[i];
      switch( output_keys[i] )
      {
        case UMAMI_OUT_MATRIX_ELEMENT:
          m2_out = static_cast<double*>( output );
          break;
        case UMAMI_OUT_DIAGRAM_AMP2:
          amp2_out = static_cast<double*>( output );
          break;
        case UMAMI_OUT_COLOR_INDEX:
          color_out = static_cast<int*>( output );
          break;
        case UMAMI_OUT_HELICITY_INDEX:
          helicity_out = static_cast<int*>( output );
          break;
        case UMAMI_OUT_DIAGRAM_INDEX:
          diagram_out = static_cast<int*>( output );
          break;
        case UMAMI_OUT_GPU_STREAM:
          gpu_stream = static_cast<gpuStream_t>( output );
          break;
        default:
          return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
      }
    }

    std::size_t n_threads = 256;
    std::size_t n_blocks = ( count + n_threads - 1 ) / n_threads;
    std::size_t rounded_count = n_blocks * n_threads;

    fptype_momenta* momenta;
    fptype_amp* numerators;
    fptype_amp* denominators;
    fptype *couplings, *g_s, *helicity_random, *color_random, *diagram_random;
    fptype *matrix_elements, *ghel_matrix_elements;
    fptype_amp *color_jamps, *ghel_jamps;
    int *helicity_index, *color_index, *helicity_given;
    unsigned int *flavor_indices, *diagram_index;

    std::size_t n_coup = madmatrix::Parameters_dependentCouplings::ndcoup;
    std::array<std::pair<void**, std::size_t>, 17> ptrs_and_sizes = {{
        {reinterpret_cast<void**>(&momenta), rounded_count * ProcessData::npar * 4 * sizeof( fptype_momenta )},
        {reinterpret_cast<void**>(&couplings), rounded_count * n_coup * 2 * sizeof( fptype )},
        {reinterpret_cast<void**>(&g_s), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&flavor_indices), rounded_count * sizeof( unsigned int )},
        {reinterpret_cast<void**>(&helicity_random), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&color_random), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&diagram_random), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&matrix_elements), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&diagram_index), rounded_count * sizeof( unsigned int )},
        // The color flow is picked among ncolor_flow structures, which is more than
        // ncolor when the color sum runs on the DDM basis
        {reinterpret_cast<void**>(&color_jamps), rounded_count * CPPProcess::ncolor_flow * sizeof( fptype_amp )},
        // The numerators are accumulated in place over all helicities via atomicAdd (no helicity dimension),
        // and the denominators are derived from them, so neither buffer carries the ncomb factor anymore.
        {reinterpret_cast<void**>(&numerators), rounded_count * ProcessData::ndiagrams * sizeof( fptype_amp )},
        {reinterpret_cast<void**>(&denominators), rounded_count * sizeof( fptype_amp )},
        {reinterpret_cast<void**>(&helicity_index), rounded_count * sizeof( int )},
        {reinterpret_cast<void**>(&color_index), rounded_count * sizeof( int )},
        {reinterpret_cast<void**>(&helicity_given), rounded_count * sizeof( int )},
        {reinterpret_cast<void**>(&ghel_matrix_elements), rounded_count * ProcessData::ncomb * sizeof( fptype )},
        // njampso = ncolor per amplitude split order (just ncolor without split orders)
        {reinterpret_cast<void**>(&ghel_jamps), rounded_count * ProcessData::ncomb * ProcessData::njampso * mgOnGpu::nx2 * sizeof( fptype_amp )},
    }};
    std::size_t total_size = 0;
    constexpr std::size_t MAX_SIZE = std::max( { sizeof( fptype ), sizeof( fptype_momenta ), sizeof( fptype ), sizeof( int ) } );
    for (auto [ptr, size] : ptrs_and_sizes) {
        std::size_t aligned_size = (size + MAX_SIZE - 1) / MAX_SIZE * MAX_SIZE;
        total_size += aligned_size;
    }
    uint8_t* buffer;
    // we can consider caching this between matrix element calls
    gpuMallocAsync( &buffer, total_size, gpu_stream );
    std::size_t buf_offset = 0;
    for (auto [ptr, size] : ptrs_and_sizes) {
        std::size_t aligned_size = (size + 7) / 8 * 8;
        *ptr = buffer + buf_offset;
        buf_offset += aligned_size;
    }

    copy_inputs<<<n_blocks, n_threads, 0, gpu_stream>>>(
      momenta_in,
      random_helicity_in,
      random_color_in,
      random_diagram_in,
      alpha_s_in,
      flavor_indices_in,
      helicity_in,
      momenta,
      helicity_random,
      color_random,
      diagram_random,
      g_s,
      flavor_indices,
      helicity_given,
      count,
      stride,
      offset );
    computeDependentCouplings<<<n_blocks, n_threads, 0, gpu_stream>>>( instance->deviceParams.arg( params ), g_s, couplings );
    checkGpu( gpuPeekAtLastError() );

    if( instance->learned < nlearn && count > 0 )
    {
      // the helicity filtering runs on the default stream: wait for the inputs copied on
      // gpu_stream, which need not synchronise with it (a non-blocking stream); its
      // synchronous copies back to the host finish it before the work below
      gpuStreamSynchronize( gpu_stream );
      // the jamp scratch of the helicity filtering (one helicity, njampso jamps, a few
      // events): ghel_jamps, whose size does not depend on ncolor_flow and nampso
      if( instance->keep_every_helicity )
        keep_all_helicities( instance );
      else
      {
        const std::size_t scanned = learn_good_helicities(
          instance, momenta, couplings, flavor_indices, matrix_elements, ghel_jamps, numerators, denominators, rounded_count );
        // the events among them, not the padding (copies of the first event, at the end)
        instance->learned += std::min( scanned, count );
      }
      // the good helicities reach dcGoodHel/dcNGoodHel through copies from pageable host
      // memory, which may return before the device holds them: wait for them before the
      // kernels of sigmaKin, on a stream that does not synchronise with the default one
      checkGpu( gpuDeviceSynchronize() );
    }

    sigmaKin(
      params,
      instance->deviceParams.arg( params ),
      momenta,
      couplings,
      flavor_indices,
      helicity_random,
      color_random,
      nullptr,
      diagram_random,
      matrix_elements,
      helicity_index,
      color_index,
      color_jamps,
      numerators,
      denominators,
      diagram_index,
      false,
      ghel_matrix_elements,
      ghel_jamps,
      nullptr,
      nullptr,
      &gpu_stream,
      true,
      n_blocks,
      n_threads,
      // a caller drawing the helicity (event generation) gets the madevent convention for
      // an interference |M|^2 (see add_and_select_hel), unless the instance was asked to
      // keep the helicity sum (interference_helicity_summed); without the random number
      // the |M|^2 stays the helicity sum (systematics re-evaluation, standalone checks)
      random_helicity_in != nullptr && !instance->summed_helicity,
      helicity_in ? helicity_given : nullptr );

    copy_outputs<<<n_blocks, n_threads, 0, gpu_stream>>>(
      denominators,
      numerators,
      matrix_elements,
      diagram_index,
      color_index,
      helicity_index,
      m2_out,
      amp2_out,
      diagram_out,
      color_out,
      helicity_out,
      helicity_in,
      count,
      stride,
      offset );
    checkGpu( gpuPeekAtLastError() );

    gpuFreeAsync( buffer, gpu_stream );
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_free( UmamiHandle handle )
  {
    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    DeviceGuard guard( instance->device ); // its device memory is freed there
    delete instance;
    return UMAMI_SUCCESS;
  }

  // A madmatrix extension, not part of UMAMI: the number of rows in the one
  // good-helicity list of an instance, shared by every flavor and crossing (see
  // nlearn): 0 before its first umami_matrix_element call, and while it is still
  // choosing, the rows that contribute to the events it has looked at so far.
  int madmatrix_good_helicity_count( UmamiHandle handle )
  {
    return static_cast<InterfaceInstance*>( handle )->process.parameters().nGoodHelRows;
  }

  // A madmatrix extension, not part of UMAMI: choose the good helicities of an
  // instance at once, on RAMBO points of a collider of centre-of-mass energy sqrt_s,
  // whose frame moves at `rapidity` along the beams, the partonic energy and frame
  // varying with `spread` (see the cpu backend). mg7 makes this choice on a CPU
  // build and gives it to the GPU instances (madmatrix_set_helicity_choice); this is
  // for a caller of the GPU build alone. UMAMI_ERROR where the points do not reach
  // the process; the first events choose then.
  UmamiStatus madmatrix_choose_good_helicities( UmamiHandle handle, double sqrt_s, double rapidity, int spread )
  {
    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    return choose_good_helicities( instance, sqrt_s, rapidity, spread != 0, spread != 0 ? -1. : 0. ) ? UMAMI_SUCCESS : UMAMI_ERROR;
  }

  // A madmatrix extension, not part of UMAMI: the good helicities an instance has
  // chosen, as madmatrix_set_helicity_choice takes them (see the cpu backend):
  // choice[0], the flags of the CPU builds (--hel_recycling, C-parity dedup: neither
  // here, 0), then the rows of the helicity table that are kept. Returns the size of the choice, written if it
  // is at most `capacity`, or -1 while the instance is still choosing.
  int madmatrix_get_helicity_choice( UmamiHandle handle, int* choice, int capacity )
  {
    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    if( instance->learned < nlearn ) return -1;
    int size = 1;
    for( bool good : instance->good_hel ) size += good ? 1 : 0;
    if( choice != nullptr && size <= capacity )
    {
      int k = 0;
      choice[k++] = 0;
      for( int ihel = 0; ihel < ProcessData::ncomb; ihel++ )
        if( instance->good_hel[ihel] ) choice[k++] = ihel;
    }
    return size;
  }

  // A madmatrix extension, not part of UMAMI: give an instance the good helicities
  // that madmatrix_get_helicity_choice wrote for an instance of this process with
  // the same parameters: mg7 chooses them from its cards on a CPU build
  // (madmatrix_choose_good_helicities), so that a run on a GPU evaluates the
  // helicities of a run on a CPU. They are kept until
  // umami_set_parameter changes a parameter. The flags of choice[0] are not used
  // here. UMAMI_ERROR, and nothing changes, if the choice does not fit the
  // helicities of this process.
  UmamiStatus madmatrix_set_helicity_choice( UmamiHandle handle, const int* choice, int size )
  {
    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    if( choice == nullptr || size < 1 ) return UMAMI_ERROR;
    for( int k = 1; k < size; ++k )
      if( choice[k] < 0 || choice[k] >= ProcessData::ncomb ) return UMAMI_ERROR;
    DeviceGuard guard( instance->device ); // deviceParams.update copies to it
    for( bool& good : instance->good_hel ) good = false;
    for( int k = 1; k < size; ++k ) instance->good_hel[choice[k]] = true;
    instance->learned = nlearn;
    ProcessParameters& params = instance->process.parameters();
    sigmaKin_setGoodHel( params, instance->good_hel );
    instance->deviceParams.update( params );
    // a copy from pageable host memory may return before the device holds it (see
    // umami_matrix_element): the kernels of a non-blocking stream must see it
    checkGpu( gpuDeviceSynchronize() );
    return UMAMI_SUCCESS;
  }
}
