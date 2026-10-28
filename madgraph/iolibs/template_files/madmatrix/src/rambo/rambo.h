// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
//==========================================================================
// RAMBO phase-space generator for the standalone (madmatrix) driver.
// Ported verbatim from the MadGraph7 standalone_cpp output (rambo.cc/rambo.h), so
// that check_sa generates the SAME phase-space point as the Fortran/C++ 'check'
// drivers, INCLUDING the particle masses. The only changes are:
//   - wrapped in namespace rambo and made header-only (inline);
//   - momenta returned as std::vector<std::vector<double>> (one [E,px,py,pz] per
//     external particle) instead of std::vector<double*> (no manual new[]/leaks).
//==========================================================================

#ifndef RAMBO_H
#define RAMBO_H 1

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace rambo
{
  // Universal (Marsaglia-Zaman) random number generator, FSU-SCRI-87-50.
  class Random
  {
  public:
    double ranmar(){
	/*     -----------------
	 * universal random number generator proposed by marsaglia and zaman
	 * in report fsu-scri-87-50
	 * in this version rvec is a double precision variable. */
	  double uni = ranu[iranmr] - ranu[jranmr];
	  if(uni < 0) uni = uni + 1;
	  ranu[iranmr] = uni;
	  iranmr = iranmr - 1;
	  jranmr = jranmr - 1;
	  if(iranmr == 0) iranmr = 97;
	  if(jranmr == 0) jranmr = 97;
	  ranc = ranc - rancd;
	  if(ranc < 0) ranc = ranc + rancm;
	  uni = uni - ranc;
	  if(uni < 0) uni = uni + 1;
	  return uni;
	}

    void rmarin(int ij, int kl){
	/*     -----------------
	 * initializing routine for ranmar, must be called before generating
	 * any pseudorandom numbers with ranmar. the input values should be in
	 * the ranges 0<=ij<=31328 ; 0<=kl<=30081 */
	/* this shows correspondence between the simplified input seeds ij, kl
	 * and the original marsaglia-zaman seeds i,j,k,l.
	 * to get the standard values in the marsaglia-zaman paper (i=12,j=34
	 * k=56,l=78) put ij=1802, kl=9373 */
	  int i = ij/177 % 177 + 2;
	  int j = ij % 177 + 2;
	  int k = (kl/169) % 178 + 1;
	  int l = kl % 169;
	  for (int ii = 1; ii < 98; ii++){
	    double s =  0;
	    double t = .5;
	    for (int jj = 1; jj < 25; jj++){
	      int m = ((i*j % 179)*k) % 179;
	      i = j;
	      j = k;
	      k = m;
	      l = (53*l+1) % 169;
	      if((l*m) % 64 >= 32) s = s + t;
	      t = .5*t;
	    }
	    ranu[ii] = s;
	  }
	  ranc  =   362436. / 16777216.;
	  rancd =  7654321. / 16777216.;
	  rancm = 16777213. / 16777216.;
	  iranmr = 97;
	  jranmr = 33;
	}

  private:
    double ranu[98];
    double ranc, rancd, rancm;
    int iranmr, jranmr;
  };

  inline double rn( [[maybe_unused]] int idummy )
  {
    static Random rand;
    double ran;
    static int init = 1;
    if( init == 1 )
    {
      init = 0;
      rand.rmarin( 1802, 9373 );
    }
    while( true )
    {
      ran = rand.ranmar();
      if( ran > 1e-16 ) break;
    }
    return ran;
  }

  // draw( i ) (i = 1 to 4, unused) returns the next random number in ]0,1[: the
  // shared generator rn for the 'check' drivers, a private one otherwise (so that
  // the points of the check drivers do not move).
  template<class Draw>
  inline std::vector<std::vector<double>> rambo( double et, std::vector<double>& xm, double& wt, Draw&& draw )
  {
	/**********************************************************************
	 *                       rambo                                         *
	 *    ra(ndom)  m(omenta)  b(eautifully)  o(rganized)                  *
	 *                                                                     *
	 *    a democratic multi-particle phase space generator                *
	 *    authors:  s.d. ellis,  r. kleiss,  w.j. stirling                 *
	 *    this is version 1.0 -  written by r. kleiss                      *
	 *    -- adjusted by hans kuijf, weights are logarithmic (20-08-90)    *
	 *                                                                     *
	 *    n  = number of particles                                         *
	 *    et = total centre-of-mass energy                                 *
	 *    xm = particle masses ( dim=nexternal-nincoming )                 *
	 *    p  = particle momenta ( dim=(4,nexternal-nincoming) )            *
	 *    wt = weight of the event                                         *
	 ***********************************************************************/
    int n = xm.size();
    std::vector<std::vector<double>> q( n, std::vector<double>( 4 ) ), p( n, std::vector<double>( 4 ) );
    std::vector<double> z( n ), r( 4 ), b( 3 ), p2( n ), xm2( n ), e( n ), v( n );
    static std::vector<int> iwarn( 5, 0 );
    static double acc = 1e-14;
    static int itmax = 6, ibegin = 0;
    static double twopi = 8. * atan( 1. );
    static double po2log = log( twopi / 4. );

    // initialization step: factorials for the phase space weight
    if( ibegin == 0 )
    {
      ibegin = 1;
      z[1] = po2log;
      for( int k = 2; k < n; k++ )
        z[k] = z[k - 1] + po2log - 2. * log( double( k - 1 ) );
      for( int k = 2; k < n; k++ )
        z[k] = ( z[k] - log( double( k ) ) );
    }
    // check on the number of particles
    if( n < 1 || n > 101 )
    {
      std::cout << "Too few or many particles: " << n << std::endl;
      exit( -1 );
    }
    // check whether total energy is sufficient; count nonzero masses
    double xmt = 0.;
    int nm = 0;
    for( int i = 0; i < n; i++ )
    {
      if( xm[i] != 0. ) nm = nm + 1;
      xmt = xmt + std::abs( xm[i] );
    }
    if( xmt > et )
    {
      std::cout << "Too low energy: " << et << " needed " << xmt << std::endl;
      exit( -1 );
    }

    // generate n massless momenta in infinite phase space
    for( int i = 0; i < n; i++ )
    {
      double r1 = draw( 1 );
      double c = 2. * r1 - 1.;
      double s = sqrt( 1. - c * c );
      double f = twopi * draw( 2 );
      r1 = draw( 3 );
      double r2 = draw( 4 );
      q[i][0] = -log( r1 * r2 );
      q[i][3] = q[i][0] * c;
      q[i][2] = q[i][0] * s * cos( f );
      q[i][1] = q[i][0] * s * sin( f );
    }
    // calculate the parameters of the conformal transformation
    for( int i = 0; i < 4; i++ )
      r[i] = 0.;
    for( int i = 0; i < n; i++ )
    {
      for( int k = 0; k < 4; k++ )
        r[k] = r[k] + q[i][k];
    }
    double rmas = sqrt( pow( r[0], 2 ) - pow( r[3], 2 ) - pow( r[2], 2 ) - pow( r[1], 2 ) );
    for( int k = 1; k < 4; k++ )
      b[k - 1] = -r[k] / rmas;
    double g = r[0] / rmas;
    double a = 1. / ( 1. + g );
    double x = et / rmas;

    // transform the q's conformally into the p's
    for( int i = 0; i < n; i++ )
    {
      double bq = b[0] * q[i][1] + b[1] * q[i][2] + b[2] * q[i][3];
      for( int k = 1; k < 4; k++ )
        p[i][k] = x * ( q[i][k] + b[k - 1] * ( q[i][0] + a * bq ) );
      p[i][0] = x * ( g * q[i][0] + bq );
    }

    // calculate weight and possible warnings
    wt = po2log;
    if( n != 2 ) wt = ( 2. * n - 4. ) * log( et ) + z[n - 1];
    if( wt < -180. )
    {
      if( iwarn[0] <= 5 ) std::cout << "Too small wt, risk for underflow: " << wt << std::endl;
      iwarn[0] = iwarn[0] + 1;
    }
    if( wt > 174. )
    {
      if( iwarn[1] <= 5 ) std::cout << "Too large wt, risk for overflow: " << wt << std::endl;
      iwarn[1] = iwarn[1] + 1;
    }

    // return for weighted massless momenta
    if( nm == 0 )
      return p;

    // massive particles: rescale the momenta by a factor x
    double xmax = sqrt( 1. - pow( xmt / et, 2 ) );
    for( int i = 0; i < n; i++ )
    {
      xm2[i] = pow( xm[i], 2 );
      p2[i] = pow( p[i][0], 2 );
    }
    int iter = 0;
    x = xmax;
    double accu = et * acc;
    while( true )
    {
      double f0 = -et;
      double g0 = 0.;
      double x2 = x * x;
      for( int i = 0; i < n; i++ )
      {
        e[i] = sqrt( xm2[i] + x2 * p2[i] );
        f0 = f0 + e[i];
        g0 = g0 + p2[i] / e[i];
      }
      if( std::abs( f0 ) <= accu ) break;
      iter = iter + 1;
      if( iter > itmax )
      {
        std::cout << "Too many iterations without desired accuracy: " << itmax << std::endl;
        break;
      }
      x = x - f0 / ( x * g0 );
    }
    for( int i = 0; i < n; i++ )
    {
      v[i] = x * p[i][0];
      for( int k = 1; k < 4; k++ )
        p[i][k] = x * p[i][k];
      p[i][0] = e[i];
    }

    // calculate the mass-effect weight factor
    double wt2 = 1.;
    double wt3 = 0.;
    for( int i = 0; i < n; i++ )
    {
      wt2 = wt2 * v[i] / e[i];
      wt3 = wt3 + pow( v[i], 2 ) / e[i];
    }
    double wtm = ( 2. * n - 3. ) * log( x ) + log( wt2 / wt3 * et );

    // return for weighted massive momenta
    wt = wt + wtm;
    if( wt < -180. )
    {
      if( iwarn[2] <= 5 ) std::cout << "Too small wt, risk for underflow: " << wt << std::endl;
      iwarn[2] = iwarn[2] + 1;
    }
    if( wt > 174. )
    {
      if( iwarn[3] <= 5 ) std::cout << "Too large wt, risk for overflow: " << wt << std::endl;
      iwarn[3] = iwarn[3] + 1;
    }
    return p;
  }

  inline std::vector<std::vector<double>> rambo( double et, std::vector<double>& xm, double& wt )
  {
    return rambo( et, xm, wt, rn );
  }

  // Build the full external momenta (initial + final), changing convention between
  // MadGraph7 and rambo. `masses` has one entry per external particle.
  template<class Draw>
  inline std::vector<std::vector<double>> get_momenta( int ninitial, double energy,
                                                       std::vector<double> masses, double& wgt, Draw&& draw )
  {
    int nexternal = masses.size();
    int nfinal = nexternal - ninitial;
    double e2 = pow( energy, 2 );
    double m1 = masses[0];

    if( ninitial == 1 )
    {
      // Momenta for the single incoming particle (at rest)
      std::vector<std::vector<double>> p;
      p.push_back( { m1, 0., 0., 0. } );
      std::vector<double> finalmasses( masses.begin() + 1, masses.end() );
      std::vector<std::vector<double>> p_rambo = rambo( m1, finalmasses, wgt, draw );
      for( auto& pf : p_rambo ) p.push_back( pf );
      return p;
    }
    else if( ninitial != 2 )
    {
      std::cout << "Rambo needs 1 or 2 incoming particles" << std::endl;
      exit( -1 );
    }

    if( nfinal == 1 )
      energy = m1;

    double m2 = masses[1];

    double mom = sqrt( ( pow( e2, 2 ) - 2 * e2 * pow( m1, 2 ) + pow( m1, 4 ) - 2 * e2 * pow( m2, 2 ) -
                         2 * pow( m1, 2 ) * pow( m2, 2 ) + pow( m2, 4 ) ) /
                       ( 4 * e2 ) );
    double energy1 = sqrt( pow( mom, 2 ) + pow( m1, 2 ) );
    double energy2 = sqrt( pow( mom, 2 ) + pow( m2, 2 ) );

    // Set momenta for the two incoming particles (back-to-back along z)
    std::vector<std::vector<double>> p;
    p.push_back( { energy1, 0., 0., mom } );
    p.push_back( { energy2, 0., 0., -mom } );

    if( nfinal == 1 )
    {
      p.push_back( { energy, 0., 0., 0. } );
      wgt = 1;
      return p;
    }
    std::vector<double> finalmasses( masses.begin() + 2, masses.end() );
    std::vector<std::vector<double>> p_rambo = rambo( energy, finalmasses, wgt, draw );
    for( auto& pf : p_rambo ) p.push_back( pf );
    return p;
  }

  inline std::vector<std::vector<double>> get_momenta( int ninitial, double energy,
                                                       std::vector<double> masses, double& wgt )
  {
    return get_momenta( ninitial, energy, masses, wgt, rn );
  }

  // A 2 > 1 point: the beams back to back along a random axis, at sqrt(s) = the
  // final mass, which is at rest.
  template<class Draw>
  inline std::vector<std::vector<double>> fusion_point( const std::vector<double>& masses, Draw&& draw )
  {
    const double m1 = masses[0], m2 = masses[1], energy = masses[2];
    const double e2 = energy * energy;
    const double mom = std::sqrt( std::max( 0., ( e2 - pow( m1 + m2, 2 ) ) * ( e2 - pow( m1 - m2, 2 ) ) ) ) / ( 2 * energy );
    const double cth = 2 * draw( 1 ) - 1, sth = std::sqrt( 1 - cth * cth );
    const double phi = 2 * M_PI * draw( 2 );
    const double n[3] = { sth * std::cos( phi ), sth * std::sin( phi ), cth };
    std::vector<std::vector<double>> p;
    p.push_back( { std::sqrt( mom * mom + m1 * m1 ), mom * n[0], mom * n[1], mom * n[2] } );
    p.push_back( { std::sqrt( mom * mom + m2 * m2 ), -mom * n[0], -mom * n[1], -mom * n[2] } );
    p.push_back( { energy, 0., 0., 0. } );
    return p;
  }

  // The points umami_initialize chooses the good helicities on, in the layout of
  // the UMAMI momenta input: momenta[nevt * ( npar * mu + ipar ) + ievt]. RAMBO
  // points in the partonic CM (a decay at rest). The helicity zeros of a massive
  // leg depend on the frame (a u g > u z row that is ~1e-33 in the CM is 0.6% of
  // |M|^2 in a boosted one), and the matrix element is called in the lab frame or
  // in the me_frame of the run card: the second half of the points is boosted to
  // a random frame, so that the good helicities are those of any frame. A 2 > 1
  // has a single point up to the frame, the beams back to back at sqrt(s) = the
  // final mass. Its helicity zeros are those of the beam axis (beams along z
  // make a W at rest in one J_z state: two of its three rows vanish), so the
  // axis of every point is drawn at random. Empty where there is no phase space
  // (a decay below threshold) or RAMBO does not converge. A private generator:
  // the shared one (rn) makes the points of the check drivers, which must not
  // move.
  // Whether a process of these masses has a physical point at all: not a decay
  // lighter than its products, nor a 2 > 1 lighter than its beams. Where it has
  // none it is never evaluated on shell (rambo would exit on it).
  inline bool has_phase_space( int ninitial, const std::vector<double>& masses )
  {
    const int npar = masses.size();
    double mfinal = 0;
    for( int ipar = ninitial; ipar < npar; ++ipar ) mfinal += masses[ipar];
    if( ninitial == 2 && npar == 3 ) return masses[2] > masses[0] + masses[1]; // 2 > 1
    return npar - ninitial >= 2 && ( ninitial != 1 || masses[0] > mfinal );
  }

  inline std::vector<double> helicity_scan_points( int ninitial, const std::vector<double>& masses, std::size_t nevt )
  {
    const int npar = masses.size();
    double mtotal = 0;
    for( int ipar = 0; ipar < npar; ++ipar ) mtotal += masses[ipar];
    const bool fusion = ninitial == 2 && npar == 3; // 2 > 1
    // rambo exits on a closed phase space: check it first
    if( !has_phase_space( ninitial, masses ) ) return {};
    // a decay is at rest instead, a 2 > 1 at sqrt(s) = the final mass
    const double energy = fusion ? masses[2] : std::max( 1500., 2 * mtotal );
    Random random;
    random.rmarin( 1802, 9373 );
    auto draw = [&random]( int )
    {
      double r = random.ranmar();
      while( r <= 1e-16 ) r = random.ranmar();
      return r;
    };
    std::vector<double> momenta( nevt * npar * 4 );
    for( std::size_t ievt = 0; ievt < nevt; ++ievt )
    {
      double weight = 0;
      const std::vector<std::vector<double>> p = fusion ? fusion_point( masses, draw ) : get_momenta( ninitial, energy, masses, weight, draw );
      for( int ipar = 0; ipar < npar; ++ipar )
      {
        // on shell, or rambo did not converge: no points
        const double m2 = p[ipar][0] * p[ipar][0] - p[ipar][1] * p[ipar][1] - p[ipar][2] * p[ipar][2] - p[ipar][3] * p[ipar][3];
        if( std::abs( m2 - masses[ipar] * masses[ipar] ) > 1e-6 * energy * energy ) return {};
        for( int mu = 0; mu < 4; ++mu ) momenta[nevt * ( npar * mu + ipar ) + ievt] = p[ipar][mu];
      }
    }
    // the boosts are drawn after the points: the first half are the CM points
    for( std::size_t ievt = nevt / 2; ievt < nevt; ++ievt )
    {
      const double beta = 0.3 + 0.6 * draw( 1 );
      const double cth = 2 * draw( 2 ) - 1, sth = std::sqrt( 1 - cth * cth );
      const double phi = 2 * M_PI * draw( 3 );
      const double b[3] = { beta * sth * std::cos( phi ), beta * sth * std::sin( phi ), beta * cth };
      const double gamma = 1 / std::sqrt( 1 - beta * beta );
      for( int ipar = 0; ipar < npar; ++ipar )
      {
        double* e = &momenta[nevt * ( npar * 0 + ipar ) + ievt];
        double* q[3] = { &momenta[nevt * ( npar * 1 + ipar ) + ievt],
                         &momenta[nevt * ( npar * 2 + ipar ) + ievt],
                         &momenta[nevt * ( npar * 3 + ipar ) + ievt] };
        const double bq = b[0] * *q[0] + b[1] * *q[1] + b[2] * *q[2];
        const double f = ( gamma - 1 ) * bq / ( beta * beta ) + gamma * *e;
        for( int i = 0; i < 3; ++i ) *q[i] += f * b[i];
        *e = gamma * ( *e + bq );
      }
    }
    return momenta;
  }

} // namespace rambo

#endif // RAMBO_H
