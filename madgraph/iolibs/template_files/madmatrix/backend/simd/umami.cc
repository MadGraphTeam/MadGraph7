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
#include "UmamiInstance.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "rambo.h" // the RAMBO points of choose_good_helicities

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


  // The good helicities are chosen at run time, on the events the instance is
  // called with, as in the Fortran standalone output (matrix_standalone_v4.inc, on
  // its first 20 calls): every helicity is evaluated on the first nlearn events,
  // and the helicities that contribute to one of them are kept for the rest of the
  // run, or until umami_set_parameter changes a parameter. A helicity contributes
  // if its |M|^2 is above limhel/ncomb of the sum over helicities at one of them
  // (madevent's rule), or, with a massive external leg, if it is not exactly 0,
  // the list then closed over the legs whose helicity states mix under a boost
  // (sigmaKin_getGoodHel). So they are chosen at the energies, and in the frame,
  // that the matrix element is used at. Chosen at initialization instead, on RAMBO
  // points at sqrt(s) = 1.5 TeV, a helicity that is a power of m/sqrt(s) below the
  // others there was dropped for the whole run: the all-plus helicities of
  // g g > h g in the heft model, 1e-9 of |M|^2 at 1.5 TeV and 42% of it at 130 GeV.
  // Nor does a first call of special points fix the choice: one event in the exact
  // forward direction of u u~ > e+ e-, where the (1 - cos theta) helicities
  // vanish, is one of the nlearn events, and the next ones add them. With
  // crossing, the scan evaluates every row of the crossing table on these events,
  // into the one list they share (a crossing that moves a mass is then off shell,
  // scanned the conservative way, see sigmaKin_getGoodHel). nlearn is the number of
  // events sigmaKin_getGoodHel samples in a call (goodHelScanEvents, 16 at run
  // time), so that a first call of 16 events or more chooses at once; the events of
  // a call that it does not sample are evaluated with the helicities chosen so far.
  // Where there is no phase space (a decay below threshold: its momenta are off
  // shell, and in its rest frame a scalar to two off-shell vectors only has the
  // rows of equal helicities, the others are rounding residuals that are not 0 once
  // the decay is boosted), every helicity is kept instead
  // (UmamiInstance::keep_every_helicity).
  constexpr std::size_t nlearn = goodHelScanEvents;

  // Where the base process has no phase space at the masses of the instance
  bool no_phase_space( UmamiInstance* instance )
  {
    const std::vector<fptype>& instanceMasses = instance->process.getMasses();
    const std::vector<double> masses( instanceMasses.begin(), instanceMasses.end() );
    return !rambo::has_phase_space( ProcessData::npari, masses );
  }

  // Keep every helicity of an instance (UmamiInstance::keep_every_helicity)
  void keep_all_helicities( UmamiInstance* instance )
  {
    for( bool& good : instance->good_hel ) good = true;
    instance->learned = nlearn;
    instance->csym_bad = false;
    sigmaKin_setGoodHel( instance->process.parameters(), instance->good_hel ); // no scan: no C-parity dedup
  }

  // Add the helicities that contribute to the first events of a call to those of
  // the instance. Returns the number of lanes of the call it looked at.
  int learn_good_helicities(
    UmamiInstance* instance,
    const fptype_momenta* momenta,
    const fptype* couplings,
    const unsigned int* flavor_indices,
    fptype* matrix_elements,
    fptype_amp* numerators,
    fptype_amp* denominators,
    std::size_t count )
  {
    ProcessParameters& params = instance->process.parameters();
    bool is_good_hel[ProcessData::ncomb];
    const int scanned = sigmaKin_getGoodHel(
      params, momenta, couplings, flavor_indices, matrix_elements, numerators, denominators,
      is_good_hel,
      count );
    for( int ihel = 0; ihel < ProcessData::ncomb; ihel++ )
      if( is_good_hel[ihel] ) instance->good_hel[ihel] = true;
    // one C-parity verdict for every scan of the instance
    if( sigmaKin_csymScanBad() ) instance->csym_bad = true;
    sigmaKin_setCsymVerdict( instance->csym_bad );
    sigmaKin_setGoodHel( params, instance->good_hel );
    return scanned;
  }

  // Forget the good helicities of an instance: the next events choose them again
  void reset_good_helicities( UmamiInstance* instance )
  {
    instance->learned = 0;
    instance->csym_bad = false;
    for( bool& good : instance->good_hel ) good = false;
    instance->process.parameters().hrStale = false; // see hr_check_stale (SigmaKin.cc)
    instance->keep_every_helicity = no_phase_space( instance );
  }

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


  // The masses of the most recently initialized or modified instance, for
  // umami_get_meta(UMAMI_META_MASSES), which takes no handle
  std::vector<double> g_externalMasses;

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
  // collider (scan_points), rather than on the first nlearn events it is called
  // with: for a 2 > n process (n >= 2), collider_scan_points at centre-of-mass
  // energy sqrt_s, frame rapidity `rapidity`, `spread` and max_boost; for a decay
  // or a 2 > 1, rambo::helicity_scan_points (at rest or at the final mass, half of
  // them boosted in random directions: sqrt_s and rapidity are not used). Where the
  // process has no phase space at all (a decay below threshold), every helicity is
  // kept. Returns false, and chooses nothing, where it has some but the points do
  // not reach it (closed at sqrt_s, RAMBO did not converge): the first events
  // choose then.
  bool choose_good_helicities( UmamiInstance* instance, double sqrt_s, double rapidity, bool spread, double max_boost )
  {
    const std::vector<fptype>& instanceMasses = instance->process.getMasses();
    const std::vector<double> masses( instanceMasses.begin(), instanceMasses.end() );
    reset_good_helicities( instance );
    if( instance->keep_every_helicity )
    {
      keep_all_helicities( instance );
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
    // every helicity is evaluated on these points, for every flavor and crossing
    std::vector<double> matrix_elements( count );
    const UmamiInputKey input_keys[] = { UMAMI_IN_MOMENTA, UMAMI_IN_FLAVOR_INDEX };
    const void* inputs[] = { momenta.data(), flavor_indices.data() };
    const UmamiOutputKey output_keys[] = { UMAMI_OUT_MATRIX_ELEMENT };
    void* outputs[] = { matrix_elements.data() };
    umami_matrix_element( instance, count, count, 0, 2, input_keys, inputs, 1, output_keys, outputs );
    instance->learned = nlearn;
    return true;
  }

#ifdef MG_HR_PROBE
  // The --hel_recycling warm-up probe (make HRPROBE=1: built and run at generation
  // time, see madmatrix/hel_recycling_filter.py) bakes the helicity rows it finds
  // into the recycled code, and a row it misses is computed without recycling at
  // run time (hr_unbaked_row). So it chooses at initialization, on RAMBO points
  // that cover the energies and the frames of a run, and not on the events of
  // check_sa (one energy, the centre-of-mass frame): up to max( 1500 GeV, twice the
  // sum of the masses ), boosted by up to 2 units of rapidity when an incoming
  // parton comes from a PDF (or from the photon flux of a lepton). It chooses on
  // those points only: sigmaKin_setGoodHel prints the rows and the vanishing
  // amplitudes at every choice, the latter as seen so far, and the exporter takes
  // the union of what it prints.
  constexpr bool has_pdf_beams()
  {
    if( ProcessData::npari != 2 ) return false;
    for( int iflav = 0; iflav < ProcessData::nmaxflavor; ++iflav )
      for( int ipar = 0; ipar < ProcessData::npari; ++ipar )
      {
        const int pdg = ProcessData::flavorPDGs[iflav][ipar];
        const int apdg = pdg < 0 ? -pdg : pdg;
        if( ( apdg >= 1 && apdg <= 6 ) || apdg == 21 || apdg == 22 ) return true;
      }
    return false;
  }

  void choose_good_helicities_for_probe( UmamiInstance* instance )
  {
    double mtotal = 0;
    for( fptype mass : instance->process.getMasses() ) mtotal += mass;
    choose_good_helicities( instance, std::max( 1500., 2 * mtotal ), 0, true, has_pdf_beams() ? 2 : 0 );
  }
#endif

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
        device = UMAMI_DEVICE_CPU;
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
    static const bool data[UMAMI_OUTPUT_KEY_COUNT] = { true, true, true, true, true };
    *supported = data;
    *count = UMAMI_OUTPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_initialize( UmamiHandle* handle, char const* param_card_path )
  {
    std::unique_ptr<UmamiInstance> instance( new UmamiInstance() );
    instance->process.initProc( param_card_path );

    const std::vector<fptype>& masses = instance->process.getMasses();
    g_externalMasses.assign( masses.begin(), masses.end() );
    reset_good_helicities( instance.get() ); // keep_every_helicity

#ifdef MG_HR_PROBE
    choose_good_helicities_for_probe( instance.get() );
#endif
    *handle = instance.release();
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_set_parameter(
    UmamiHandle handle,
    char const* name,
    double parameter_real,
    [[maybe_unused]] double parameter_imag )
  {
    UmamiInstance* instance = static_cast<UmamiInstance*>( handle );
    // The run card bw_cutoff (the $-excluded propagator window). Per instance, like
    // the model parameters read by umami_initialize.
    if( std::string( name ) == "bwcutoff" )
    {
      instance->process.parameters().bwcutoff = parameter_real;
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
    const std::vector<fptype>& masses = instance->process.getMasses();
    g_externalMasses.assign( masses.begin(), masses.end() );
    // A helicity may vanish with some parameters and not with others: choose again
    reset_good_helicities( instance );
#ifdef MG_HR_PROBE
    choose_good_helicities_for_probe( instance );
#endif
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
    if( !static_cast<UmamiInstance*>( handle )->process.modelParameters().getParameterByName( name, parameter_real, parameter_imag ) )
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
    // the color and diagram are chosen for it, and UMAMI_IN_RANDOM_HELICITY is not used.
    if( helicity_in )
    {
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        const int ihel = helicity_in[i_event + offset];
        if( ihel < 0 || ihel >= ProcessData::ncomb ) return UMAMI_ERROR;
      }
    }

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
        default:
          return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
      }
    }

    constexpr std::size_t vector_size = MemoryAccessMomentaBase::neppM;
    // The kernels run on one SIMD page of events at a time (neppM == neppV events,
    // a single event for the scalar backend), and on two pages at a time in mixed
    // precision (double amplitudes, float colour algebra): the batch is rounded up
    // to that, and no further. Every extra event is computed for nothing: a call
    // with one event used to evaluate two of them on the scalar backend.
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    constexpr std::size_t page_size2 = 2 * vector_size;
#else
    constexpr std::size_t page_size2 = vector_size;
#endif
    std::vector<std::size_t> permutation;
    std::size_t rounded_count;

    // The SIMD grouping key is the REDUCED flavor (id % nmaxflavor), not the
    // full extended flavorID. The extended id encodes both a reduced flavor and
    // a crossing (id = cross*nmaxflavor + flavor). calculate_jamps only requires
    // the reduced flavor to be constant across a SIMD vector (the wavefunction
    // flavor is read once per vector); the crossing is applied per event by the
    // momentum gather, so one vector may legitimately mix crossings. Indexing
    // the grouping arrays by the full id (as an earlier version did) overflowed
    // them whenever a crossing was present (id >= nmaxflavor), corrupting the
    // stack and crashing (SIGABRT/SIGSEGV).
    constexpr std::size_t flavor_count = ProcessData::nmaxflavor;
    HostBufferBase<unsigned int, false> flavor_indices( ((count + page_size2 - 1) / page_size2 + flavor_count) * page_size2 );
    bool sort_flavors = vector_size > 1 && flavor_count > 1 && flavor_indices_in;
    if ( sort_flavors )
    {
      permutation.resize(count);
      std::size_t voffset = 0;
      std::size_t vector_indices[flavor_count] = {};
      std::size_t vector_counts[flavor_count] = {};
      // determine permutation of inputs such that all entries in a SIMD vector
      // share the same reduced flavor (they may still carry different crossings)
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        unsigned int flav = flavor_indices_in[i_event + offset];
        unsigned int rflav = flav % (unsigned int)ProcessData::nmaxflavor;
        auto& vcount = vector_counts[rflav];
        auto& vindex = vector_indices[rflav];
        if ( vcount == 0 )
        {
          vindex = voffset * page_size2;
          // no pre-fill: a padding lane copies the first event of its page,
          // extended id included (see the lane loop below)
          voffset += 1;
        }
        permutation[i_event] = vindex + vcount;
        vcount = (vcount + 1) % page_size2;
      }
      rounded_count = voffset * page_size2;
    } else {
      rounded_count = ( count + page_size2 - 1 ) / page_size2 * page_size2;
    }

    HostBufferBase<fptype_momenta, false> momenta( rounded_count * ProcessData::npar * 4 );
    HostBufferBase<fptype, false> couplings( rounded_count * madmatrix::Parameters_dependentCouplings::ndcoup * 2 );
    HostBufferBase<fptype, false> g_s( rounded_count );
    HostBufferBase<fptype, false> helicity_random( rounded_count );
    HostBufferBase<fptype, false> color_random( rounded_count );
    HostBufferBase<fptype, false> diagram_random( rounded_count );
    HostBufferBase<fptype, false> matrix_elements( rounded_count );
    HostBufferBase<unsigned int, false> diagram_index( rounded_count );
    HostBufferBase<fptype_amp, false> numerators( rounded_count * ProcessData::ndiagrams );
    HostBufferBase<fptype_amp, false> denominators( rounded_count );
    HostBufferBase<int, false> helicity_index( rounded_count );
    HostBufferBase<int, false> color_index( rounded_count );
    HostBufferBase<int, false> helicity_given( rounded_count ); // as the other inputs, the padding lanes copy their source event
    // The input event of each lane. The padding lanes, which complete the last page
    // of a flavor, are copies of the first event of their page, as on GPUs. Left at
    // zero (momenta and g_s), they evaluated to nan, and the good-helicity filter of
    // the first call (a helicity is kept if its |M|^2 is not 0 for one of its first
    // events) took that for a non-zero value whenever a padding lane was among
    // them, as in a first call with a single event: every helicity was then kept
    // for the rest of the run, the vanishing ones of a q q~ initial state included.
    std::vector<std::size_t> source( rounded_count, count );
    for( std::size_t i_event = 0; i_event < count; ++i_event )
    {
      source[sort_flavors ? permutation[i_event] : i_event] = i_event;
    }
    for( std::size_t lane = 0; lane < rounded_count; ++lane )
    {
      // a page is opened by an event, so its first lane is never padding
      if( source[lane] == count ) source[lane] = source[lane - lane % page_size2];
    }
    for( std::size_t lane = 0; lane < rounded_count; ++lane )
    {
      const std::size_t i_event = source[lane];
      transpose_momenta( &momenta_in[offset], momenta.data(), i_event, lane, stride );
      helicity_random[lane] = random_helicity_in ? random_helicity_in[i_event + offset] : 0.5;
      color_random[lane] = random_color_in ? random_color_in[i_event + offset] : 0.5;
      diagram_random[lane] = random_diagram_in ? random_diagram_in[i_event + offset] : 0.5;
      g_s[lane] = alpha_s_in ? sqrt( 4 * M_PI * alpha_s_in[i_event + offset] ) : 1.2177157847767195;
      flavor_indices[lane] = flavor_indices_in ? flavor_indices_in[i_event + offset] : 0;
      if( helicity_in ) helicity_given[lane] = helicity_in[i_event + offset];
    }
    UmamiInstance* instance = static_cast<UmamiInstance*>( handle );
    ProcessParameters& params = instance->process.parameters();
    computeDependentCouplings( params, g_s.data(), couplings.data(), rounded_count );

    if( instance->learned < nlearn && count > 0 )
    {
      if( instance->keep_every_helicity )
        keep_all_helicities( instance );
      else
      {
        const std::size_t scanned = learn_good_helicities(
          instance,
          momenta.data(),
          couplings.data(),
          flavor_indices.data(),
          matrix_elements.data(),
          numerators.data(),
          denominators.data(),
          rounded_count );
        // the events among them, not the padding lanes (copies of the event that opens their page)
        for( std::size_t lane = 0; lane < scanned; ++lane )
          if( lane % page_size2 == 0 || source[lane] != source[lane - lane % page_size2] ) ++instance->learned;
      }
    }

    sigmaKin(
      params,
      momenta.data(),
      couplings.data(),
      flavor_indices.data(),
      helicity_random.data(),
      color_random.data(),
      nullptr,
      diagram_random.data(),
      matrix_elements.data(),
      helicity_index.data(),
      color_index.data(),
      numerators.data(),
      denominators.data(),
      diagram_index.data(),
      false,
      rounded_count,
      // a caller drawing the helicity (event generation) gets the madevent convention for
      // an interference |M|^2, see select_helicity_signed, unless the instance was asked
      // to keep the helicity sum (interference_helicity_summed); without the random number
      // the |M|^2 stays the helicity sum (systematics re-evaluation, standalone checks)
      random_helicity_in != nullptr && !instance->summed_helicity,
      helicity_in ? helicity_given.data() : nullptr );

    if ( sort_flavors )
    {
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        std::size_t i_sorted = permutation[i_event];
        std::size_t page_size = MemoryAccessMomentaBase::neppM;
        std::size_t i_page = i_sorted / page_size;
        std::size_t i_vector = i_sorted % page_size; // vector lane

        double denominator = denominators[i_sorted];
        if( m2_out != nullptr )
        {
          m2_out[i_event + offset] = matrix_elements[i_sorted];
        }
        if( amp2_out != nullptr )
        {
          for( std::size_t i_diag = 0; i_diag < ProcessData::ndiagrams; ++i_diag )
          {
            amp2_out[stride * i_diag + i_event + offset] = channel_amp2( numerators[i_page * page_size * ProcessData::ndiagrams + i_diag * page_size + i_vector], denominator );
          }
        }
        if( diagram_out != nullptr )
        {
          diagram_out[i_event + offset] = diagram_index[i_sorted] - 1;
        }
        if( color_out != nullptr )
        {
          color_out[i_event + offset] = color_index[i_sorted] - 1;
        }
        if( helicity_out != nullptr )
        {
          helicity_out[i_event + offset] = helicity_index[i_sorted] - 1;
        }
      }
    } else {
      std::size_t page_size = MemoryAccessMomentaBase::neppM;
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        std::size_t i_page = i_event / page_size;
        std::size_t i_vector = i_event % page_size;

        double denominator = denominators[i_event];
        if( m2_out != nullptr )
        {
          m2_out[i_event + offset] = matrix_elements[i_event];
        }
        if( amp2_out != nullptr )
        {
          for( std::size_t i_diag = 0; i_diag < ProcessData::ndiagrams; ++i_diag )
          {
            amp2_out[stride * i_diag + i_event + offset] = channel_amp2( numerators[i_page * page_size * ProcessData::ndiagrams + i_diag * page_size + i_vector], denominator );
          }
        }
        if( diagram_out != nullptr )
        {
          diagram_out[i_event + offset] = diagram_index[i_event] - 1;
        }
        if( color_out != nullptr )
        {
          color_out[i_event + offset] = color_index[i_event] - 1;
        }
        if( helicity_out != nullptr )
        {
          helicity_out[i_event + offset] = helicity_index[i_event] - 1;
        }
      }
    }
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_free( UmamiHandle handle )
  {
    delete static_cast<UmamiInstance*>( handle );
    return UMAMI_SUCCESS;
  }

  // A madmatrix extension, not part of UMAMI: the number of rows in the one
  // good-helicity list of an instance, shared by every flavor and crossing (see
  // nlearn): 0 before its first umami_matrix_element call, and while it is still
  // choosing, the rows that contribute to the events it has looked at so far. It
  // counts both rows of a C-parity pair even when the helicity loop runs only one
  // of them, so every backend gives the same number.
  int madmatrix_good_helicity_count( UmamiHandle handle )
  {
    return static_cast<UmamiInstance*>( handle )->process.parameters().nGoodHelRows;
  }

  // A madmatrix extension, not part of UMAMI: choose the good helicities of an
  // instance at once, on nlearn RAMBO points of a collider of centre-of-mass energy
  // sqrt_s, whose frame moves at `rapidity` along the beams, the partonic energy and
  // frame varying with `spread` (beams with PDFs, ISR or beamstrahlung), see
  // choose_good_helicities. They are kept until umami_set_parameter changes a
  // parameter. mg7 chooses so, from its cards, on a CPU build, and gives the choice
  // to the libraries of the run (madmatrix_get_helicity_choice,
  // madmatrix_set_helicity_choice): the same for every thread, device and run.
  // With crossing, every row of the crossing table is scanned, each on points of
  // its own masses (scan_points). Where the process has no phase space at all (a
  // decay below threshold), every helicity is kept. UMAMI_ERROR where it has some
  // but the points do not reach it (the collider is below its threshold, RAMBO did
  // not converge); the first events choose then.
  UmamiStatus madmatrix_choose_good_helicities( UmamiHandle handle, double sqrt_s, double rapidity, int spread )
  {
    UmamiInstance* instance = static_cast<UmamiInstance*>( handle );
    return choose_good_helicities( instance, sqrt_s, rapidity, spread != 0, spread != 0 ? -1. : 0. ) ? UMAMI_SUCCESS : UMAMI_ERROR;
  }

  // A madmatrix extension, not part of UMAMI: the good helicities an instance has
  // chosen, as madmatrix_set_helicity_choice takes them: choice[0] is a set of
  // flags -- 1 if --hel_recycling has to use the plain code (hr_check_stale,
  // SigmaKin.cc), 2 if the C-parity dedup holds (params.csymOk: every good row is
  // paired with its mirror, of the same |M|^2 at every scan point) -- then the rows
  // of the helicity table that are kept. Returns the size of the
  // choice, written if it is at most `capacity`, or -1 while the instance is still
  // choosing.
  int madmatrix_get_helicity_choice( UmamiHandle handle, int* choice, int capacity )
  {
    UmamiInstance* instance = static_cast<UmamiInstance*>( handle );
    if( instance->learned < nlearn ) return -1;
    int size = 1;
    for( bool good : instance->good_hel ) size += good ? 1 : 0;
    if( choice != nullptr && size <= capacity )
    {
      int k = 0;
      const ProcessParameters& params = instance->process.parameters();
      choice[k++] = ( params.hrStale ? 1 : 0 ) | ( params.csymOk ? 2 : 0 );
      for( int ihel = 0; ihel < ProcessData::ncomb; ihel++ )
        if( instance->good_hel[ihel] ) choice[k++] = ihel;
    }
    return size;
  }

  // A madmatrix extension, not part of UMAMI: give an instance the good helicities
  // that madmatrix_get_helicity_choice wrote for an instance of this process with
  // the same parameters, of this build or of another (backend, precision). They are
  // kept until umami_set_parameter changes a parameter; the instance then chooses
  // on its first events again. The C-parity flag is the verdict of the scan that
  // made the choice, which this instance has not run (the dedup is off without
  // one, see sigmaKin_setGoodHel). UMAMI_ERROR, and nothing changes, if the choice
  // does not fit the helicities of this process.
  UmamiStatus madmatrix_set_helicity_choice( UmamiHandle handle, const int* choice, int size )
  {
    UmamiInstance* instance = static_cast<UmamiInstance*>( handle );
    if( choice == nullptr || size < 1 ) return UMAMI_ERROR;
    for( int k = 1; k < size; ++k )
      if( choice[k] < 0 || choice[k] >= ProcessData::ncomb ) return UMAMI_ERROR;
    for( bool& good : instance->good_hel ) good = false;
    for( int k = 1; k < size; ++k ) instance->good_hel[choice[k]] = true;
    instance->learned = nlearn;
    ProcessParameters& params = instance->process.parameters();
    params.hrStale = ( choice[0] & 1 ) != 0;
    instance->csym_bad = ( choice[0] & 2 ) == 0;
    sigmaKin_setCsymVerdict( instance->csym_bad );
    sigmaKin_setGoodHel( params, instance->good_hel );
    return UMAMI_SUCCESS;
  }
}
