/* SPDX-License-Identifier: 0BSD */
/* One-loop artifacts through the native loop SDK (rusticol_virtual.h): the
 * calls of the Python walkthrough (python/loop_walkthrough.py) without Python
 * or a Symbolica licence. Build with `make -C native loop`; run
 *
 *   ./loop_client ARTIFACT PROCESS MOMENTA.txt [HELICITY_ID]
 *
 * MOMENTA.txt: "points legs" on the first line, then per point its mu_R^2
 * and its legs' (E, px, py, pz) (the walkthrough writes it). The client opens
 * the process, prints its squared-order axes, evaluates with the per-type
 * default backend ("auto") and the other backend explicitly, in the checked
 * default and the unchecked timing mode, evaluates the declared colour and
 * spin correlations and, for a loop-induced process, one helicity
 * (HELICITY_ID) against the summed call. */
#define _POSIX_C_SOURCE 200809L
#include "rusticol_virtual.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void check(int status, const char *what) {
  if (status != RUSTICOL_VIRTUAL_OK) {
    fprintf(stderr, "%s: %s\n", what, rusticol_virtual_last_error_message());
    exit(1);
  }
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* The JSON value of "key" in text (an array or object, by bracket depth). */
static void print_json_field(const char *text, const char *key) {
  char pattern[64];
  snprintf(pattern, sizeof pattern, "\"%s\":", key);
  const char *start = strstr(text, pattern);
  if (start == NULL) {
    printf("  %s: absent\n", key);
    return;
  }
  start += strlen(pattern);
  int depth = 0;
  const char *end = start;
  do {
    if (*end == '[' || *end == '{') depth++;
    if (*end == ']' || *end == '}') depth--;
    end++;
  } while (*end && depth > 0);
  printf("  %s: %.*s\n", key, (int)(end - start), start);
}

static char *handle_json(const rusticol_virtual_handle *handle,
                         int (*get)(const rusticol_virtual_handle *, char *, size_t, size_t *)) {
  size_t size = 0;
  check(get(handle, NULL, 0, &size), "json size");
  char *text = malloc(size);
  check(get(handle, text, size, &size), "json");
  return text;
}

static char *result_json(const rusticol_virtual_result *result,
                         int (*get)(const rusticol_virtual_result *, char *, size_t, size_t *)) {
  size_t size = 0;
  check(get(result, NULL, 0, &size), "json size");
  char *text = malloc(size);
  check(get(result, text, size, &size), "json");
  return text;
}

/* One ordinary evaluation: its wall time, values per point and sector, and
 * the reduction record of its diagnostics. */
static double evaluate(rusticol_virtual_handle *handle, const rusticol_virtual_batch_request *request,
                       const rusticol_virtual_selection *selection, const char *label) {
  rusticol_virtual_result *result = NULL;
  double begin = now();
  if (selection == NULL)
    check(rusticol_virtual_evaluate(handle, request, &result), label);
  else
    check(rusticol_virtual_evaluate_selected(handle, request, selection, &result), label);
  double seconds = now() - begin;
  rusticol_virtual_result_shape_info info;
  check(rusticol_virtual_result_shape(result, &info), "result shape");
  double *values = malloc((info.virtual_values ? info.virtual_values : 1) * sizeof(double));
  check(rusticol_virtual_result_copy_virtual(result, values, info.virtual_values), "copy");
  size_t orders = info.axes.virtual_orders;
  printf("%s: %zu of %zu points, %.3f ms per point\n", label, info.successful_points,
         info.requested_points, 1e3 * seconds / (double)info.requested_points);
  for (size_t point = 0; point < info.successful_points && point < 2; point++)
    for (size_t order = 0; order < orders; order++) {
      const double *v = values + (point * orders + order) * 3;
      printf("  point %zu, order %zu: finite %.12e, 1/eps %.6e, 1/eps^2 %.6e\n", point,
             order, v[0], v[1], v[2]);
    }
  char *diagnostics = result_json(result, rusticol_virtual_result_diagnostics_json);
  print_json_field(diagnostics, "reduction");
  free(diagnostics);
  free(values);
  rusticol_virtual_result_free(result);
  return seconds;
}

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    fprintf(stderr, "usage: %s ARTIFACT PROCESS MOMENTA.txt [HELICITY_ID]\n", argv[0]);
    return 2;
  }
  FILE *input = fopen(argv[3], "r");
  size_t points = 0, legs = 0;
  if (input == NULL || fscanf(input, "%zu %zu", &points, &legs) != 2) {
    fprintf(stderr, "cannot read %s\n", argv[3]);
    return 2;
  }
  double *momenta = malloc(points * legs * 4 * sizeof(double));
  double *scales = malloc(points * sizeof(double));
  for (size_t p = 0; p < points; p++) {
    if (fscanf(input, "%lf", &scales[p]) != 1) return 2;
    for (size_t k = 0; k < legs * 4; k++)
      if (fscanf(input, "%lf", &momenta[p * legs * 4 + k]) != 1) return 2;
  }
  fclose(input);

  rusticol_virtual_handle *handle = NULL;
  check(rusticol_virtual_open(argv[1], argv[2], NULL, NULL, &handle), "open");
  rusticol_virtual_shape_info shape;
  check(rusticol_virtual_shape(handle, &shape), "shape");
  char *metadata = handle_json(handle, rusticol_virtual_metadata_json);
  int induced = strstr(metadata, "\"contribution_kind\":\"loop-induced\"") != NULL;
  printf("process %s: %s, %zu legs, %zu squared orders, %zu helicities\n", argv[2],
         induced ? "loop-induced" : "Born-interfered", shape.external_count,
         shape.virtual_orders, shape.helicities);
  print_json_field(metadata, "sectors");
  free(metadata);

  rusticol_virtual_batch_request request = {momenta, points * legs * 4, points, NULL,
                                            scales, points, 0, NULL};
  /* The per-type default backend, then the other one explicitly. */
  check(rusticol_virtual_set_reduction(handle, "auto"), "reduction auto");
  evaluate(handle, &request, NULL, "auto (checked)");
  const char *other = induced ? "ninjars" : "tirust";
  if (rusticol_virtual_set_reduction(handle, other) == RUSTICOL_VIRTUAL_OK)
    evaluate(handle, &request, NULL, induced ? "ninjars (checked)" : "tirust (checked)");
  else /* a library built without the backend refuses it, typed */
    printf("%s refused: %s\n", other, rusticol_virtual_last_error_message());
  /* The unchecked timing mode (no accuracy certificate). */
  check(rusticol_virtual_set_reduction(handle, "auto"), "reduction auto");
  check(rusticol_virtual_set_reduction_unchecked(handle, 1), "unchecked");
  evaluate(handle, &request, NULL, "auto (unchecked)");
  check(rusticol_virtual_set_reduction_unchecked(handle, 0), "checked");

  /* Colour- and spin-correlated evaluation of the declared histories. */
  char *histories = handle_json(handle, rusticol_virtual_correlation_histories_json);
  printf("colour histories: %s\n", histories);
  free(histories);
  char *classes = handle_json(handle, rusticol_virtual_spin_correlation_classes_json);
  printf("spin classes: %s\n", classes);
  free(classes);
  rusticol_virtual_correlated_result *correlated = NULL;
  check(rusticol_virtual_evaluate_correlated(handle, &request, &correlated), "correlated");
  rusticol_virtual_correlated_result_shape_info cinfo;
  check(rusticol_virtual_correlated_result_shape(correlated, &cinfo), "correlated shape");
  rusticol_virtual_complex64 *cvalues =
      malloc((cinfo.virtual_values ? cinfo.virtual_values : 1) * sizeof(rusticol_virtual_complex64));
  check(rusticol_virtual_correlated_result_copy_virtual(correlated, cvalues, cinfo.virtual_values),
        "correlated copy");
  size_t per_history = cinfo.axes.virtual_orders * cinfo.axes.epsilon_count;
  for (size_t history = 0; history < cinfo.axes.histories; history++)
    printf("  point 0, history %zu, order 0: finite (%.12e, %.12e)\n", history,
           cvalues[history * per_history].real, cvalues[history * per_history].imag);
  free(cvalues);
  rusticol_virtual_correlated_result_free(correlated);

  /* One helicity of a loop-induced process: only its rows are formed. */
  if (induced && argc == 5) {
    const char *ids[1] = {argv[4]};
    rusticol_virtual_selection selection = {ids, 1, NULL, 0, NULL, 0, NULL, 0, NULL};
    double summed = evaluate(handle, &request, NULL, "summed (auto)");
    double one = evaluate(handle, &request, &selection, "one helicity (auto)");
    printf("one helicity / summed call: %.3f\n", one / summed);
  } else if (!induced) {
    printf("helicity selection: a Born-interfered selection is served at leading "
           "colour by the Python API only\n");
  }
  rusticol_virtual_free(handle);
  free(momenta);
  free(scales);
  return 0;
}
