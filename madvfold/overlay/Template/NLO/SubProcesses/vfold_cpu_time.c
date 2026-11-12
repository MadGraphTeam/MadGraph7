// Copyright (C) 2026 CERN.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by A. Valassi (Oct 2026) for project MadVFold (arXiv:2609.23500).
//-------------------------------------------------------------------------

/* AV road B (step 11a): faster implementations of the Fortran cpu_time intrinsic.
 *
 * MG5aMC times its buckets with "call cpu_time(t)" (92 calls in SubProcesses and
 * Source), which gfortran compiles into calls to _gfortran_cpu_time_4 (all MG5
 * timers are real*4). libgfortran implements it with a getrusage() syscall: with
 * folding these timers are called ~9M times per MINT2 run of eebbbb at f841 k64,
 * and the syscalls alone cost ~19% of the C/avx2 run time (strace: 18.3M getrusage
 * calls = all of the 8.4 s of sys time).
 *
 * vfold_cpu_time(t) is a Fortran-callable replacement with three implementations,
 * chosen once at the first call by ENV_VFOLD_CPUTIME:
 *   0 (default) getrusage(RUSAGE_SELF) user+sys, exactly as libgfortran-11 does
 *   1           clock_gettime(CLOCK_MONOTONIC), via the vDSO (no syscall)
 *   2           __builtin_ia32_rdtsc() (x86_64 with constant_tsc only, otherwise 1),
 *               calibrated once against CLOCK_MONOTONIC at the first call
 *  -1 / -2 (step 11b) mode 0 in the reference modes F0 and F1, which are meant
 *               to behave like the original code, and mode 1 / 2 in F2, F and C
 *               (as ENV_VFOLD_BSF_FUSED=-1). The ME mode is read here directly from
 *               ENV_VFOLD_ME_MODE, as the first call (the start of the Total timer
 *               in driver_mintMC.f) comes before setup_av_me_mode: the Fortran
 *               check_av_cpu_time (vfold_mode.f90) then verifies that the mode chosen
 *               here agrees with vfold_is_reference_mode().
 * Modes 1 and 2 measure WALL time since the first call (not process CPU time):
 * the same for single-threaded CPU runs, different for multi-threaded or GPU runs.
 * The time since the first call (not an absolute time) is returned because t is
 * a real*4: an absolute monotonic time would have a resolution of ~1 s.
 *
 * _gfortran_cpu_time_4 below is defined in the executable, so the static linker
 * binds all "call cpu_time" of the executable (including the Source libraries)
 * to it rather than to libgfortran: no MG5 source file needs to be changed.
 * If this ever breaks, build with -DVFOLD_NO_CPU_TIME_INTERPOSE and replace
 * "call cpu_time(" by "call vfold_cpu_time(" in the Fortran sources instead.
 *
 * This file is compiled with $(CXX) by makefile_fks_dir (%.o: %.c rule).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

static int vfold_cputime_mode = -1;  /* -1: not yet initialised */
static int vfold_cputime_setting = 0; /* ENV_VFOLD_CPUTIME as parsed (-2..2) */
static struct timespec vfold_mono0;  /* modes 1 and 2: time of the first call */
static unsigned long long vfold_tsc0 = 0;
static double vfold_sec_per_tick = 0;

static double vfold_mono_since0( void )
{
  struct timespec t;
  clock_gettime( CLOCK_MONOTONIC, &t );
  return ( t.tv_sec - vfold_mono0.tv_sec ) + 1e-9 * ( t.tv_nsec - vfold_mono0.tv_nsec );
}

#if defined( __x86_64__ )
static int vfold_has_constant_tsc( void )
{
  int found = 0;
  char line[4096];
  FILE* f = fopen( "/proc/cpuinfo", "r" );
  if( !f ) return 0;
  while( fgets( line, sizeof( line ), f ) )
    if( strncmp( line, "flags", 5 ) == 0 ) { found = ( strstr( line, " constant_tsc" ) != NULL ); break; }
  fclose( f );
  return found;
}
#endif

static void vfold_cputime_init( void )
{
  const char* env = getenv( "ENV_VFOLD_CPUTIME" );
  int mode = ( env && env[0] ) ? atoi( env ) : 0;
  if( mode < -2 || mode > 2 ) mode = 0;
  vfold_cputime_setting = mode;
  if( mode < 0 )
  {
    /* step 11b: getrusage in the reference modes F0/F1 (see vfold_is_reference_mode) */
    const char* me = getenv( "ENV_VFOLD_ME_MODE" );
    int reference = me && ( strcmp( me, "F0" ) == 0 || strcmp( me, "F1" ) == 0 );
    mode = reference ? 0 : -mode;
  }
  clock_gettime( CLOCK_MONOTONIC, &vfold_mono0 );
  if( mode == 2 )
  {
#if defined( __x86_64__ )
    if( vfold_has_constant_tsc() )
    {
      /* calibrate the tsc frequency over ~10 ms (busy wait, once per job) */
      double dt;
      unsigned long long tsc1;
      vfold_tsc0 = __builtin_ia32_rdtsc();
      while( ( dt = vfold_mono_since0() ) < 0.01 ) {}
      tsc1 = __builtin_ia32_rdtsc();
      vfold_sec_per_tick = dt / (double)( tsc1 - vfold_tsc0 );
    }
    else
      mode = 1;
#else
    mode = 1;
#endif
  }
  vfold_cputime_mode = mode;
  if( mode == 2 )
    printf( " AV: vfold_cpu_time mode=2 (rdtsc, %.4f GHz) set by ENV_VFOLD_CPUTIME=%d\n", 1e-9 / vfold_sec_per_tick,
            vfold_cputime_setting );
  else if( env && env[0] )
    printf( " AV: vfold_cpu_time mode=%d (%s) set by ENV_VFOLD_CPUTIME=%d\n", mode,
            mode == 1 ? "clock_gettime monotonic" : "getrusage", vfold_cputime_setting );
  else
    printf( " AV: vfold_cpu_time mode=%d (getrusage)\n", mode );
  fflush( stdout );
}

/* Fortran: integer, external :: vfold_cpu_time_mode; mode = vfold_cpu_time_mode(setting)
   returns the implementation in use (0, 1, 2) and the ENV_VFOLD_CPUTIME setting (-2..2) */
int vfold_cpu_time_mode_( int* setting )
{
  if( vfold_cputime_mode < 0 ) vfold_cputime_init();
  *setting = vfold_cputime_setting;
  return vfold_cputime_mode;
}

/* Fortran: real*4 t; call vfold_cpu_time(t) */
void vfold_cpu_time_( float* t )
{
  if( vfold_cputime_mode < 0 ) vfold_cputime_init();
  if( vfold_cputime_mode == 0 )
  {
    /* same arithmetic as libgfortran-11 cpu_time_4 (intrinsics/cpu_time.c) */
    struct rusage ru;
    long sec, usec;
    if( getrusage( RUSAGE_SELF, &ru ) == 0 )
    {
      sec = ru.ru_utime.tv_sec + ru.ru_stime.tv_sec;
      usec = ru.ru_utime.tv_usec + ru.ru_stime.tv_usec; /* not normalised */
    }
    else
    {
      sec = -1;
      usec = 0;
    }
    *t = sec + usec * 1.e-6f;
  }
#if defined( __x86_64__ )
  else if( vfold_cputime_mode == 2 )
    *t = (float)( ( __builtin_ia32_rdtsc() - vfold_tsc0 ) * vfold_sec_per_tick );
#endif
  else
    *t = (float)vfold_mono_since0();
}

#ifndef VFOLD_NO_CPU_TIME_INTERPOSE
/* gfortran's entry point for "call cpu_time(t)" with a real*4 t */
void _gfortran_cpu_time_4( float* t )
{
  vfold_cpu_time_( t );
}
#endif

#ifdef __cplusplus
}
#endif
