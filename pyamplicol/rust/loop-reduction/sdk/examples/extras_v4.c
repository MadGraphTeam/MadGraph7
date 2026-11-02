/* SPDX-License-Identifier: 0BSD */
/* Plan v4 SDK extras through the C ABI: ./extras_v4 artifact process input.txt
 * runs resolved, colour- and spin-correlated, double-double and
 * arbitrary-precision evaluations on the input's points
 * (tests/unit/test_loop_v2_sdk_extras.py compares them with the Python API).
 * input.txt: "points legs", per point 4*legs momenta and the scale (C99
 * hexadecimal), then the spin-leg count and per spin leg its label and 8 values
 * (re, im of four components). Per call it prints "call <name>", then
 * "born ..." and "virtual ..." (doubles as %a; complex as re im pairs) and
 * "decimal <json>" (the exact values of the extended carriers, else null). */
#include "rusticol_virtual.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *what) {
  printf("error %s %s\n", what, rusticol_virtual_last_error_message());
  exit(1);
}

static void print_json(int status, char *buffer) {
  if (status) fail("json");
  printf("decimal %s\n", buffer);
  free(buffer);
}

typedef struct {
  const char *name, *arithmetic;
  uint32_t digits;
  const char *poles;
  int kind; /* 0 ordinary, 1 correlated, 2 correlated with spin, 3 resolved */
  size_t points;
} call_t;

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  FILE *input = fopen(argv[3], "r");
  size_t points = 0, legs = 0, spin_legs = 0;
  if (!input || fscanf(input, "%zu %zu", &points, &legs) != 2) return 2;
  double *momenta = malloc(points * legs * 4 * sizeof(double));
  double *scales = malloc(points * sizeof(double));
  for (size_t p = 0; p < points; p++) {
    for (size_t i = 0; i < legs * 4; i++)
      if (fscanf(input, "%la", &momenta[p * legs * 4 + i]) != 1) return 2;
    if (fscanf(input, "%la", &scales[p]) != 1) return 2;
  }
  if (fscanf(input, "%zu", &spin_legs) != 1) return 2;
  size_t *leg_labels = malloc((spin_legs + 1) * sizeof(size_t));
  rusticol_virtual_complex64 *vectors = malloc((spin_legs * 4 + 1) * sizeof(rusticol_virtual_complex64));
  for (size_t l = 0; l < spin_legs; l++) {
    if (fscanf(input, "%zu", &leg_labels[l]) != 1) return 2;
    for (size_t c = 0; c < 4; c++)
      if (fscanf(input, "%la %la", &vectors[l * 4 + c].real, &vectors[l * 4 + c].imag) != 2) return 2;
  }
  fclose(input);
  rusticol_virtual_handle *runtime = NULL;
  if (rusticol_virtual_open(argv[1], argv[2], NULL, NULL, &runtime)) fail("open");
  call_t calls[] = {
      {"ordinary", "double", 0, "all", 0, 2},
      {"correlated", "double", 0, "all", 1, 2},
      {"correlated-spin", "double", 0, "all", 2, 2},
      {"correlated-uv", "double", 0, "uv", 1, 2},
      {"correlated-none", "double", 0, "none", 1, 2},
      {"resolved", "double", 0, "all", 3, 2},
      {"resolved-ir", "double", 0, "ir", 3, 2},
      {"ordinary-dd", "double-double", 0, "all", 0, 1},
      {"correlated-spin-dd", "double-double", 0, "all", 2, 1},
      {"resolved-dd", "double-double", 0, "all", 3, 1},
      {"ordinary-ap", "arbitrary", 32, "all", 0, 1},
      {"correlated-spin-ap", "arbitrary", 32, "all", 2, 1},
      {"resolved-ap", "arbitrary", 32, "all", 3, 1},
  };
  for (size_t k = 0; k < sizeof(calls) / sizeof(calls[0]); k++) {
    call_t call = calls[k];
    size_t n = call.points < points ? call.points : points;
    if (rusticol_virtual_set_arithmetic(runtime, call.arithmetic, call.digits)) fail("arithmetic");
    if (rusticol_virtual_set_poles(runtime, call.poles)) fail("poles");
    rusticol_virtual_batch_request request = {momenta, n * legs * 4, n, NULL, scales, n,
                                              call.kind == 3, NULL};
    printf("call %s\n", call.name);
    if (call.kind == 0 || call.kind == 3) {
      rusticol_virtual_result *result = NULL;
      rusticol_virtual_result_shape_info info;
      if (rusticol_virtual_evaluate(runtime, &request, &result)) fail(call.name);
      if (rusticol_virtual_result_shape(result, &info)) fail("shape");
      double *born = malloc((info.born_values + 1) * sizeof(double));
      double *values = malloc((info.virtual_values + 1) * sizeof(double));
      if (rusticol_virtual_result_copy_born(result, born, info.born_values) ||
          rusticol_virtual_result_copy_virtual(result, values, info.virtual_values))
        fail("copy");
      printf("born");
      for (size_t i = 0; i < info.born_values; i++) printf(" %a", born[i]);
      printf("\nvirtual");
      for (size_t i = 0; i < info.virtual_values; i++) printf(" %a", values[i]);
      printf("\n");
      size_t needed = 0;
      if (rusticol_virtual_result_decimal_json(result, NULL, 0, &needed)) fail("json size");
      char *text = malloc(needed);
      print_json(rusticol_virtual_result_decimal_json(result, text, needed, &needed), text);
      free(born);
      free(values);
      rusticol_virtual_result_free(result);
    } else {
      rusticol_virtual_correlated_result *result = NULL;
      rusticol_virtual_correlated_result_shape_info info;
      rusticol_virtual_spin_sources sources = {leg_labels, spin_legs, vectors, spin_legs * 4};
      int status = call.kind == 2
                       ? rusticol_virtual_evaluate_correlated_with_sources(runtime, &request, &sources, &result)
                       : rusticol_virtual_evaluate_correlated(runtime, &request, &result);
      if (status) fail(call.name);
      if (rusticol_virtual_correlated_result_shape(result, &info)) fail("shape");
      rusticol_virtual_complex64 *born = malloc((info.born_values + 1) * sizeof(*born));
      rusticol_virtual_complex64 *values = malloc((info.virtual_values + 1) * sizeof(*values));
      if (rusticol_virtual_correlated_result_copy_born(result, born, info.born_values) ||
          rusticol_virtual_correlated_result_copy_virtual(result, values, info.virtual_values))
        fail("copy");
      printf("born");
      for (size_t i = 0; i < info.born_values; i++) printf(" %a %a", born[i].real, born[i].imag);
      printf("\nvirtual");
      for (size_t i = 0; i < info.virtual_values; i++) printf(" %a %a", values[i].real, values[i].imag);
      printf("\n");
      size_t needed = 0;
      if (rusticol_virtual_correlated_result_decimal_json(result, NULL, 0, &needed)) fail("json size");
      char *text = malloc(needed);
      print_json(rusticol_virtual_correlated_result_decimal_json(result, text, needed, &needed), text);
      free(born);
      free(values);
      rusticol_virtual_correlated_result_free(result);
    }
  }
  rusticol_virtual_free(runtime);
  return 0;
}
