/* SPDX-License-Identifier: 0BSD */
/* Loop-induced selections through the pyAmpliCol UMAMI virtual extension:
 * ./selection_v4_umami artifact process input.txt anchors [binary64], with
 * selection_v4.c's input and output formats; "anchors" is the process's
 * amplitude-basis size (the capabilities' color_resolutions). The per-event
 * helicity row travels as UMAMI_IN_HELICITY_INDEX (the tree provider's key),
 * the colour resolution as PYAMPLICOL_UMAMI_V1_IN_COLOR_RESOLUTION; values
 * travel in UMAMI's strided column-major layout and are printed back in the
 * SDK layout; the exact decimals of the extended carriers come from the
 * optional diagnostic result. */
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
  int anchors; /* 1: resolved amplitude-basis anchors; 0: per-event helicity index */
  size_t points;
} call_t;

int main(int argc, char **argv) {
  if (argc < 5) return 2;
  int binary64_only = argc > 5 && strcmp(argv[5], "binary64") == 0;
  size_t basis = (size_t)strtoul(argv[4], NULL, 10);
  FILE *input = fopen(argv[3], "r");
  size_t points = 0, legs = 0, count = 0;
  if (!input || fscanf(input, "%zu %zu", &points, &legs) != 2) return 2;
  double *flat = malloc(points * legs * 4 * sizeof(double));
  double *scales = malloc(points * sizeof(double));
  for (size_t p = 0; p < points; p++) {
    for (size_t i = 0; i < legs * 4; i++)
      if (fscanf(input, "%la", &flat[p * legs * 4 + i]) != 1) return 2;
    if (fscanf(input, "%la", &scales[p]) != 1) return 2;
  }
  if (fscanf(input, "%zu", &count) != 1) return 2;
  for (size_t k = 0; k < count; k++) {
    char ignored[256];
    if (fscanf(input, "%255s", ignored) != 1) return 2;
  }
  int *rows = malloc((points + 1) * sizeof(int));
  for (size_t p = 0; p < points; p++)
    if (fscanf(input, "%d", &rows[p]) != 1) return 2;
  fclose(input);
  rusticol_virtual_handle *runtime = NULL;
  if (rusticol_virtual_open(argv[1], argv[2], NULL, NULL, &runtime)) fail("open");
  rusticol_virtual_shape_info shape;
  if (rusticol_virtual_shape(runtime, &shape)) fail("shape");
  call_t calls[] = {
      {"anchors", "double", 0, 1, 2},
      {"by-point", "double", 0, 0, 2},
      {"anchors-dd", "double-double", 0, 1, 1},
      {"by-point-dd", "double-double", 0, 0, 1},
      {"anchors-ap", "arbitrary", 32, 1, 1},
  };
  const char resolution[] = "amplitude-basis";
  for (size_t k = 0; k < sizeof(calls) / sizeof(calls[0]); k++) {
    call_t call = calls[k];
    if (binary64_only && strcmp(call.arithmetic, "double") != 0) continue;
    size_t n = call.points < points ? call.points : points;
    if (rusticol_virtual_set_arithmetic(runtime, call.arithmetic, call.digits)) fail("arithmetic");
    printf("call %s\n", call.name);
    /* UMAMI layout: momenta[event + stride*(leg + legs*component)]. */
    size_t stride = n;
    double *momenta = malloc(stride * legs * 4 * sizeof(double));
    for (size_t p = 0; p < n; p++)
      for (size_t leg = 0; leg < legs; leg++)
        for (size_t c = 0; c < 4; c++)
          momenta[p + stride * (leg + legs * c)] = flat[p * legs * 4 + 4 * leg + c];
    size_t h = call.anchors ? shape.helicities : 1, colours = call.anchors ? basis : 1;
    size_t width = shape.virtual_orders * h * colours * shape.epsilon_count;
    double *loop = calloc(stride * width + 1, sizeof(double));
    uint32_t *statuses = calloc(stride + 1, sizeof(uint32_t));
    int keys[3] = {UMAMI_IN_MOMENTA, PYAMPLICOL_UMAMI_V1_IN_MU_R2, 0};
    const void *in[3] = {momenta, scales, NULL};
    size_t capacities[3] = {stride * legs * 4, stride, 0};
    if (call.anchors) {
      keys[2] = PYAMPLICOL_UMAMI_V1_IN_COLOR_RESOLUTION;
      in[2] = resolution;
      capacities[2] = sizeof(resolution);
    } else {
      keys[2] = UMAMI_IN_HELICITY_INDEX;
      in[2] = rows;
      capacities[2] = stride;
    }
    int out_keys[2] = {PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL, PYAMPLICOL_UMAMI_V1_OUT_STATUS};
    void *out[2] = {loop, statuses};
    size_t out_capacities[2] = {stride * width, stride};
    pyamplicol_umami_virtual_v1_options options = {(uint32_t)call.anchors, NULL};
    rusticol_virtual_result *result = NULL;
    if (pyamplicol_umami_virtual_v1_evaluate(runtime, n, stride, 0, 3, keys, in, capacities, 2,
                                             out_keys, out, out_capacities, &options, &result)) {
      printf("refused %s\n", pyamplicol_umami_virtual_v1_last_error_message());
      free(momenta); free(loop); free(statuses);
      continue;
    }
    printf("statuses");
    for (size_t p = 0; p < n; p++) printf(" %u", statuses[p]);
    printf("\nvirtual");
    for (size_t p = 0; p < n; p++) {
      if (statuses[p] == RUSTICOL_VIRTUAL_FAILED) continue;
      for (size_t o = 0; o < shape.virtual_orders; o++)
        for (size_t hel = 0; hel < h; hel++)
          for (size_t c = 0; c < colours; c++)
            for (size_t e = 0; e < shape.epsilon_count; e++)
              printf(" %a", loop[p + stride * (o + shape.virtual_orders * (hel + h * (c + colours * e)))]);
    }
    printf("\n");
    size_t needed = 0;
    if (rusticol_virtual_result_decimal_json(result, NULL, 0, &needed)) fail("json size");
    char *text = malloc(needed);
    if (rusticol_virtual_result_decimal_json(result, text, needed, &needed)) fail("json");
    printf("decimal %s\n", text);
    free(text);
    if (rusticol_virtual_result_axes_json(result, NULL, 0, &needed)) fail("json size");
    text = malloc(needed);
    if (rusticol_virtual_result_axes_json(result, text, needed, &needed)) fail("json");
    printf("axes %s\n", text);
    free(text);
    rusticol_virtual_result_free(result);
    free(momenta); free(loop); free(statuses);
  }
  rusticol_virtual_free(runtime);
  return 0;
}
