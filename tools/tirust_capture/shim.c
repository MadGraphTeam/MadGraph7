// LD_PRELOAD capture shim for the MadLoop -> TIRust bridge (libmadloop_tirust.so).
//
// Adapted from the scratch shim written on 2026-11-13 by the Claude session
// "Speed up TIRust in MadLoop and its initialization" (TIRust PR #35 work); the
// record format is unchanged, so its recovered traces remain readable.
//
// Environment:
//   TIRUST_TRACE=<prefix>      write a binary trace to <prefix>.<pid>
//   TIRUST_TRACE_INPUTS=1      also record exact DP inputs and outputs (needed for replay)
//   TIRUST_TRACE_MP_INPUTS=1   record QP calls with their exact ASCII input/output records (kind 5)
//   TIRUST_SUMMARY=<prefix>    write a one-line call/time summary to <prefix>.<pid>
//
// Build with the MadLoop executables' GCC driver so the shim binds to their glibc
// (dlsym lives in libc since glibc 2.34): gfortran -O2 -fPIC -shared shim.c -o libtirust_capture.so
//
// Records (little endian, no file header), one byte kind followed by:
//   kind 1 (DP evaluation): i32 n, rank, poles, finite, check, cache, status; f64 seconds;
//          u8 with_inputs; if with_inputs: f64 shifts[4n], f64 invariants[n*n],
//          c64 masses[n], f64 mu2, u64 olen (0 when status != 0), c64 out[olen]
//   kind 2 (QP evaluation): i32 n, rank, poles, finite, check, cache, status; f64 seconds
//   kind 3 (plan registration): i32 n, rank, status; f64 seconds
//   kind 4 (cache clear, i.e. end of a phase-space point): f64 seconds
//   kind 5 (QP evaluation with inputs, replaces kind 2 when TIRUST_TRACE_MP_INPUTS is set):
//          i32 n, rank, poles, finite, check, cache, status; f64 seconds;
//          u64 shifts_bytes, invariants_bytes, masses_bytes, scale_bytes; the four byte arrays;
//          u64 output_bytes (0 when status != 0); u8 output[output_bytes]
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct { double re, im; } C64;
static FILE *trace;
static int with_inputs, with_mp_inputs;
static double t_eval_dp, t_eval_mp, t_reg, t_clear;
static uint64_t n_dp, n_mp, n_reg, n_clear, n_dp_fail, n_mp_fail;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}
static void *sym(const char *name) {
  void *f = dlsym(RTLD_NEXT, name);
  if (!f) { fprintf(stderr, "tirust_capture: missing %s\n", name); abort(); }
  return f;
}
static void open_trace(void) {
  if (trace) return;
  const char *p = getenv("TIRUST_TRACE");
  with_inputs = getenv("TIRUST_TRACE_INPUTS") != NULL;
  with_mp_inputs = getenv("TIRUST_TRACE_MP_INPUTS") != NULL;
  if (p) {
    char path[4096];
    snprintf(path, sizeof path, "%s.%d", p, (int)getpid());
    trace = fopen(path, "wb");
    if (!trace) { fprintf(stderr, "tirust_capture: cannot open %s\n", path); abort(); }
  }
}
static void report(void) {
  const char *sp = getenv("TIRUST_SUMMARY");
  if (sp && (n_dp || n_mp)) {
    char path[4096];
    snprintf(path, sizeof path, "%s.%d", sp, (int)getpid());
    FILE *f = fopen(path, "w");
    if (f) {
      fprintf(f, "dp %lu %.6f fail %lu mp %lu %.6f fail %lu register %lu %.6f clear %lu %.6f\n",
              (unsigned long)n_dp, t_eval_dp, (unsigned long)n_dp_fail, (unsigned long)n_mp, t_eval_mp,
              (unsigned long)n_mp_fail, (unsigned long)n_reg, t_reg, (unsigned long)n_clear, t_clear);
      fclose(f);
    }
  }
  if (trace) fclose(trace);
}
__attribute__((constructor)) static void init(void) { open_trace(); atexit(report); }

int32_t ml_tirust_register_plan(void *h, int32_t n, int32_t rank) {
  static int32_t (*f)(void *, int32_t, int32_t);
  if (!f) f = sym("ml_tirust_register_plan");
  double t0 = now();
  int32_t s = f(h, n, rank);
  double dt = now() - t0;
  t_reg += dt; n_reg++;
  if (trace) {
    uint8_t k = 3;
    fwrite(&k, 1, 1, trace); fwrite(&n, 4, 1, trace); fwrite(&rank, 4, 1, trace);
    fwrite(&s, 4, 1, trace); fwrite(&dt, 8, 1, trace);
  }
  return s;
}

int32_t ml_tirust_clear_cache(void *h) {
  static int32_t (*f)(void *);
  if (!f) f = sym("ml_tirust_clear_cache");
  double t0 = now();
  int32_t s = f(h);
  double dt = now() - t0;
  t_clear += dt; n_clear++;
  if (trace) { uint8_t k = 4; fwrite(&k, 1, 1, trace); fwrite(&dt, 8, 1, trace); }
  return s;
}

int32_t ml_tirust_evaluate_dp(void *h, int32_t n, int32_t rank, const double *p, size_t plen, const double *s,
                              size_t slen, const C64 *m, size_t mlen, double mu2, C64 *out, size_t olen,
                              int32_t poles, int32_t finite, int32_t check, int32_t cache) {
  static int32_t (*f)(void *, int32_t, int32_t, const double *, size_t, const double *, size_t, const C64 *, size_t,
                      double, C64 *, size_t, int32_t, int32_t, int32_t, int32_t);
  if (!f) f = sym("ml_tirust_evaluate_dp");
  double t0 = now();
  int32_t st = f(h, n, rank, p, plen, s, slen, m, mlen, mu2, out, olen, poles, finite, check, cache);
  double dt = now() - t0;
  t_eval_dp += dt; n_dp++;
  if (st) n_dp_fail++;
  if (trace) {
    uint8_t k = 1;
    int32_t hdr[6] = {n, rank, poles, finite, check, cache};
    fwrite(&k, 1, 1, trace); fwrite(hdr, 4, 6, trace); fwrite(&st, 4, 1, trace); fwrite(&dt, 8, 1, trace);
    uint8_t wi = with_inputs;
    fwrite(&wi, 1, 1, trace);
    if (wi) {
      fwrite(p, 8, plen, trace); fwrite(s, 8, slen, trace); fwrite(m, 16, mlen, trace); fwrite(&mu2, 8, 1, trace);
      uint64_t ol = st == 0 ? olen : 0;
      fwrite(&ol, 8, 1, trace);
      if (ol) fwrite(out, 16, ol, trace);
    }
  }
  return st;
}

int32_t ml_tirust_evaluate_mp(void *h, int32_t n, int32_t rank, const uint8_t *p, size_t plen, const uint8_t *s,
                              size_t slen, const uint8_t *m, size_t mlen, const uint8_t *sc, size_t sclen, uint8_t *out,
                              size_t olen, int32_t poles, int32_t finite, int32_t check, int32_t cache) {
  static int32_t (*f)(void *, int32_t, int32_t, const uint8_t *, size_t, const uint8_t *, size_t, const uint8_t *,
                      size_t, const uint8_t *, size_t, uint8_t *, size_t, int32_t, int32_t, int32_t, int32_t);
  if (!f) f = sym("ml_tirust_evaluate_mp");
  double t0 = now();
  int32_t st = f(h, n, rank, p, plen, s, slen, m, mlen, sc, sclen, out, olen, poles, finite, check, cache);
  double dt = now() - t0;
  t_eval_mp += dt; n_mp++;
  if (st) n_mp_fail++;
  if (trace) {
    uint8_t k = with_mp_inputs ? 5 : 2;
    int32_t hdr[6] = {n, rank, poles, finite, check, cache};
    fwrite(&k, 1, 1, trace); fwrite(hdr, 4, 6, trace); fwrite(&st, 4, 1, trace); fwrite(&dt, 8, 1, trace);
    if (with_mp_inputs) {
      uint64_t lens[4] = {plen, slen, mlen, sclen};
      fwrite(lens, 8, 4, trace);
      fwrite(p, 1, plen, trace); fwrite(s, 1, slen, trace); fwrite(m, 1, mlen, trace); fwrite(sc, 1, sclen, trace);
      uint64_t ol = st == 0 ? olen : 0;
      fwrite(&ol, 8, 1, trace);
      if (ol) fwrite(out, 1, ol, trace);
    }
  }
  return st;
}
