/* SPDX-License-Identifier: 0BSD */
/* Plan v4 SDK extras through the pyAmpliCol UMAMI virtual extension:
 * ./extras_v4_umami artifact process input.txt, with extras_v4.c's input and
 * output formats. The values travel in UMAMI's strided column-major layout and
 * are printed back in the SDK layout; the handle's arithmetic and pole
 * selection apply (rusticol_virtual_set_arithmetic/_set_poles), and the exact
 * decimals of the extended carriers come from the optional diagnostic result. */
#include "pyamplicol_umami_virtual_v1.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *what) {
  printf("error %s %s | %s\n", what, pyamplicol_umami_virtual_v1_last_error_message(),
         rusticol_virtual_last_error_message());
  exit(1);
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
  double *flat = malloc(points * legs * 4 * sizeof(double));
  double *scales = malloc(points * sizeof(double));
  for (size_t p = 0; p < points; p++) {
    for (size_t i = 0; i < legs * 4; i++)
      if (fscanf(input, "%la", &flat[p * legs * 4 + i]) != 1) return 2;
    if (fscanf(input, "%la", &scales[p]) != 1) return 2;
  }
  if (fscanf(input, "%zu", &spin_legs) != 1) return 2;
  size_t *labels = malloc((spin_legs + 1) * sizeof(size_t));
  rusticol_virtual_complex64 *vectors = malloc((spin_legs * 4 + 1) * sizeof(*vectors));
  for (size_t l = 0; l < spin_legs; l++) {
    if (fscanf(input, "%zu", &labels[l]) != 1) return 2;
    /* UMAMI broadcast layout: value[selected_leg + L*component]. */
    for (size_t c = 0; c < 4; c++)
      if (fscanf(input, "%la %la", &vectors[l + spin_legs * c].real, &vectors[l + spin_legs * c].imag) != 2)
        return 2;
  }
  fclose(input);
  rusticol_virtual_handle *runtime = NULL;
  if (rusticol_virtual_open(argv[1], argv[2], NULL, NULL, &runtime)) fail("open");
  rusticol_virtual_shape_info shape;
  rusticol_virtual_correlated_shape_info correlated;
  if (rusticol_virtual_shape(runtime, &shape) || rusticol_virtual_correlated_shape(runtime, &correlated))
    fail("shape");
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
    size_t n = call.points < points ? call.points : points, stride = n;
    if (rusticol_virtual_set_arithmetic(runtime, call.arithmetic, call.digits)) fail("arithmetic");
    if (rusticol_virtual_set_poles(runtime, call.poles)) fail("poles");
    double *momenta = malloc(stride * legs * 4 * sizeof(double));
    for (size_t p = 0; p < n; p++)
      for (size_t leg = 0; leg < legs; leg++)
        for (size_t c = 0; c < 4; c++) momenta[p + stride * (leg + legs * c)] = flat[p * legs * 4 + leg * 4 + c];
    uint32_t *statuses = malloc(stride * sizeof(uint32_t));
    int resolved = call.kind == 3;
    size_t H = resolved ? shape.helicities : 1, C = resolved ? shape.born_color_components : 1;
    size_t J = call.kind == 1 || call.kind == 2 ? correlated.histories : 1;
    size_t O = shape.born_orders, V = shape.virtual_orders, E = shape.epsilon_count;
    size_t born_width = O * H * C * J, virtual_width = V * H * C * E * J;
    int complex_values = call.kind == 1 || call.kind == 2;
    size_t element = complex_values ? sizeof(rusticol_virtual_complex64) : sizeof(double);
    void *born = malloc(stride * born_width * element), *virt = malloc(stride * virtual_width * element);
    pyamplicol_umami_virtual_v1_options options = {(uint32_t)resolved, NULL};
    printf("call %s\n", call.name);
    if (!complex_values) {
      const int input_keys[2] = {UMAMI_IN_MOMENTA, PYAMPLICOL_UMAMI_V1_IN_MU_R2};
      const void *inputs[2] = {momenta, scales};
      const size_t input_capacities[2] = {stride * legs * 4, stride};
      const int output_keys[3] = {PYAMPLICOL_UMAMI_V1_OUT_BORN, PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL,
                                  PYAMPLICOL_UMAMI_V1_OUT_STATUS};
      void *const outputs[3] = {born, virt, statuses};
      const size_t output_capacities[3] = {stride * born_width, stride * virtual_width, stride};
      rusticol_virtual_result *diagnostic = NULL;
      if (pyamplicol_umami_virtual_v1_evaluate(runtime, n, stride, 0, 2, input_keys, inputs, input_capacities, 3,
                                               output_keys, outputs, output_capacities, &options,
                                               &diagnostic) != UMAMI_SUCCESS)
        fail(call.name);
      const double *b = born, *v = virt;
      printf("born");
      for (size_t p = 0; p < n; p++)
        for (size_t o = 0; o < O; o++)
          for (size_t h = 0; h < H; h++)
            for (size_t c = 0; c < C; c++) printf(" %a", b[p + stride * (o + O * (h + H * c))]);
      printf("\nvirtual");
      for (size_t p = 0; p < n; p++)
        for (size_t o = 0; o < V; o++)
          for (size_t h = 0; h < H; h++)
            for (size_t c = 0; c < C; c++)
              for (size_t e = 0; e < E; e++) printf(" %a", v[p + stride * (o + V * (h + H * (c + C * e)))]);
      size_t needed = 0;
      if (rusticol_virtual_result_decimal_json(diagnostic, NULL, 0, &needed)) fail("json");
      char *text = malloc(needed);
      if (rusticol_virtual_result_decimal_json(diagnostic, text, needed, &needed)) fail("json");
      printf("\ndecimal %s\n", text);
      free(text);
      rusticol_virtual_result_free(diagnostic);
    } else {
      int input_keys[4] = {UMAMI_IN_MOMENTA, PYAMPLICOL_UMAMI_V1_IN_MU_R2, PYAMPLICOL_UMAMI_V1_IN_SPIN_LEGS,
                           PYAMPLICOL_UMAMI_V1_IN_SPIN_VECTORS_SCALAR};
      const void *inputs[4] = {momenta, scales, labels, vectors};
      const size_t input_capacities[4] = {stride * legs * 4, stride, spin_legs, spin_legs * 4};
      size_t input_count = call.kind == 2 ? 4 : 2;
      const int output_keys[3] = {PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_BORN,
                                  PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_VIRTUAL, PYAMPLICOL_UMAMI_V1_OUT_STATUS};
      void *const outputs[3] = {born, virt, statuses};
      const size_t output_capacities[3] = {stride * born_width, stride * virtual_width, stride};
      rusticol_virtual_correlated_result *diagnostic = NULL;
      if (pyamplicol_umami_virtual_v1_evaluate_correlated(runtime, n, stride, 0, input_count, input_keys, inputs,
                                                          input_capacities, 3, output_keys, outputs,
                                                          output_capacities, &options, &diagnostic) != UMAMI_SUCCESS)
        fail(call.name);
      const rusticol_virtual_complex64 *b = born, *v = virt;
      printf("born");
      for (size_t p = 0; p < n; p++)
        for (size_t j = 0; j < J; j++)
          for (size_t o = 0; o < O; o++) {
            rusticol_virtual_complex64 z = b[p + stride * (o + O * j)];
            printf(" %a %a", z.real, z.imag);
          }
      printf("\nvirtual");
      for (size_t p = 0; p < n; p++)
        for (size_t j = 0; j < J; j++)
          for (size_t o = 0; o < V; o++)
            for (size_t e = 0; e < E; e++) {
              rusticol_virtual_complex64 z = v[p + stride * (o + V * (j + J * e))];
              printf(" %a %a", z.real, z.imag);
            }
      size_t needed = 0;
      if (rusticol_virtual_correlated_result_decimal_json(diagnostic, NULL, 0, &needed)) fail("json");
      char *text = malloc(needed);
      if (rusticol_virtual_correlated_result_decimal_json(diagnostic, text, needed, &needed)) fail("json");
      printf("\ndecimal %s\n", text);
      free(text);
      rusticol_virtual_correlated_result_free(diagnostic);
    }
    free(momenta);
    free(statuses);
    free(born);
    free(virt);
  }
  rusticol_virtual_free(runtime);
  return 0;
}
