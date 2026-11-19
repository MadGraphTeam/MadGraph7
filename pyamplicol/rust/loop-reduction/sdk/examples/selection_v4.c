/* SPDX-License-Identifier: 0BSD */
/* Loop-induced selections through the C ABI (rusticol_virtual_evaluate_selected):
 * ./selection_v4 artifact process input.txt [binary64]
 * input.txt: "points legs", per point 4*legs momenta and the scale (C99
 * hexadecimal), then "K id_1 ... id_K" (a helicity-ID subset, its first ID the
 * single-helicity selection) and per point its zero-based helicity row.
 * Per call it prints "call <name>", then "statuses ...", "virtual ..." (%a),
 * "decimal <json>", "axes <json>", or "refused <message>" for a refused call
 * (tests/unit/test_loop_v2_sdk_selection.py compares them with the Python API).
 * An optional fourth argument "binary64" runs the binary64 calls only. */
#include "rusticol_virtual.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *what) {
  printf("error %s %s\n", what, rusticol_virtual_last_error_message());
  exit(1);
}

static void print_json(const char *label, int (*query)(const rusticol_virtual_result *, char *, size_t, size_t *),
                       const rusticol_virtual_result *result) {
  size_t needed = 0;
  if (query(result, NULL, 0, &needed)) fail("json size");
  char *text = malloc(needed);
  if (query(result, text, needed, &needed)) fail("json");
  printf("%s %s\n", label, text);
  free(text);
}

typedef struct {
  const char *name, *arithmetic;
  uint32_t digits;
  int resolved, ids, by_point, colour;
  const char *resolution;
  size_t points;
} call_t;

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  int binary64_only = argc > 4 && strcmp(argv[4], "binary64") == 0;
  FILE *input = fopen(argv[3], "r");
  size_t points = 0, legs = 0, count = 0;
  if (!input || fscanf(input, "%zu %zu", &points, &legs) != 2) return 2;
  double *momenta = malloc(points * legs * 4 * sizeof(double));
  double *scales = malloc(points * sizeof(double));
  for (size_t p = 0; p < points; p++) {
    for (size_t i = 0; i < legs * 4; i++)
      if (fscanf(input, "%la", &momenta[p * legs * 4 + i]) != 1) return 2;
    if (fscanf(input, "%la", &scales[p]) != 1) return 2;
  }
  if (fscanf(input, "%zu", &count) != 1) return 2;
  char **ids = malloc((count + 1) * sizeof(char *));
  for (size_t k = 0; k < count; k++) {
    ids[k] = malloc(256);
    if (fscanf(input, "%255s", ids[k]) != 1) return 2;
  }
  uint32_t *rows = malloc((points + 1) * sizeof(uint32_t));
  for (size_t p = 0; p < points; p++)
    if (fscanf(input, "%u", &rows[p]) != 1) return 2;
  fclose(input);
  rusticol_virtual_handle *runtime = NULL;
  if (rusticol_virtual_open(argv[1], argv[2], NULL, NULL, &runtime)) fail("open");
  /* ids: 0 none, 1 the first ID, 2 every ID. */
  call_t calls[] = {
      {"anchors", "double", 0, 1, 0, 0, 0, "amplitude-basis", 2},
      {"one", "double", 0, 0, 1, 0, 0, NULL, 2},
      {"subset", "double", 0, 0, 2, 0, 0, NULL, 2},
      {"resolved-subset", "double", 0, 1, 2, 0, 0, "contracted", 2},
      {"anchors-subset", "double", 0, 1, 2, 0, 0, "amplitude-basis", 2},
      {"by-point", "double", 0, 0, 0, 1, 0, NULL, 2},
      {"colour", "double", 0, 0, 0, 0, 1, NULL, 2},
      {"anchors-dd", "double-double", 0, 1, 0, 0, 0, "amplitude-basis", 1},
      {"by-point-dd", "double-double", 0, 0, 0, 1, 0, NULL, 1},
      {"anchors-ap", "arbitrary", 32, 1, 0, 0, 0, "amplitude-basis", 1},
      {"subset-ap", "arbitrary", 32, 0, 2, 0, 0, NULL, 1},
  };
  const char *flows[] = {"1"};
  for (size_t k = 0; k < sizeof(calls) / sizeof(calls[0]); k++) {
    call_t call = calls[k];
    if (binary64_only && strcmp(call.arithmetic, "double") != 0) continue;
    size_t n = call.points < points ? call.points : points;
    if (rusticol_virtual_set_arithmetic(runtime, call.arithmetic, call.digits)) fail("arithmetic");
    printf("call %s\n", call.name);
    rusticol_virtual_batch_request request = {momenta, n * legs * 4, n, NULL, scales, n,
                                              (uint32_t)call.resolved, NULL};
    rusticol_virtual_selection selection = {
        (const char *const *)ids, call.ids == 0 ? 0 : call.ids == 1 ? 1 : count,
        flows, call.colour ? 1 : 0,
        rows, call.by_point ? n : 0,
        NULL, 0,
        call.resolution};
    rusticol_virtual_result *result = NULL;
    if (rusticol_virtual_evaluate_selected(runtime, &request, &selection, &result)) {
      if (result != NULL) fail("a refused call wrote its output");
      printf("refused %s\n", rusticol_virtual_last_error_message());
      continue;
    }
    rusticol_virtual_result_shape_info info;
    if (rusticol_virtual_result_shape(result, &info)) fail("shape");
    double *values = malloc((info.virtual_values + 1) * sizeof(double));
    uint32_t *statuses = malloc((info.requested_points + 1) * sizeof(uint32_t));
    if (rusticol_virtual_result_copy_virtual(result, values, info.virtual_values) ||
        rusticol_virtual_result_copy_statuses(result, statuses, info.requested_points))
      fail("copy");
    printf("statuses");
    for (size_t i = 0; i < info.requested_points; i++) printf(" %u", statuses[i]);
    printf("\nvirtual");
    for (size_t i = 0; i < info.virtual_values; i++) printf(" %a", values[i]);
    printf("\n");
    print_json("decimal", rusticol_virtual_result_decimal_json, result);
    print_json("axes", rusticol_virtual_result_axes_json, result);
    free(values);
    free(statuses);
    rusticol_virtual_result_free(result);
  }
  rusticol_virtual_free(runtime);
  return 0;
}
