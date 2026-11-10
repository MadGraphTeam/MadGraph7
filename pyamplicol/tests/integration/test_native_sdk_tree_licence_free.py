# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501 -- embedded native client sources have their own formatting.
"""Binary64 tree evaluation in the native SDKs needs no Symbolica licensing.

USER 2026-11-08 ~20:55 (F5): the native SDKs never require any Symbolica
licensing in double precision -- neither a user licence nor pyAmpliCol's
package key -- for every tree observable they offer. This test runs one
client per SDK language (C ABI, C++, Fortran 2008, the packaged Rust wrapper
and the UMAMI provider) over every binary64 tree observable: ordinary values,
helicity- and colour-resolved values, global helicity / colour-flow
selections and per-point helicity / colour-flow indices, squared-order
partials, colour-correlated and spin-correlated values (explicit and default
spin vectors), the on-the-fly warm-up, a model-parameter update, and the
UMAMI matrix element with a fixed and a sampled helicity and its colour index.
The C client covers every tree execution mode (recurrence at leading and full
colour, compiled, on-the-fly, eager) and a correlator artifact; each wrapper
covers each of its calls on the leading-colour recurrence, on-the-fly and
correlator artifacts.

All clients run at once, as concurrent processes, with ``SYMBOLICA_LICENSE``
removed and the package key disabled (``PYAMPLICOL_NATIVE_LIBRARY_KEY=off``)
while another process holds the user's restricted Symbolica permit (a private
lock directory, ``tools/loop_artifact_portability.RestrictedPermit``): any
Symbolica state access would abort the client with Symbolica's restricted-mode
message. Every output (the bits of every double) equals the licensed run's.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pyamplicol import (
    ColorCorrelator,
    CorrelatorConfig,
    EmitGluon,
    Generator,
    ModelSource,
    ProcessSet,
    Runtime,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    JITConfig,
    RunConfig,
)
from tests.integration.test_runtime_selector_api_parity import (
    _compile_probe,
    _NativeSdk,
    _source_environment,
    _unavailable,
)
from tests.integration.test_runtime_selector_api_parity import (
    native_sdk as _native_sdk_fixture,
)
from tests.integration.test_umami_api import _config_executable

native_sdk = _native_sdk_fixture

_ROOT = Path(__file__).resolve().parents[2]
_PROCESS = "d d~ > d d~ g"
_NAME = "dd_ddg"
_SPIN_LEG = 5
_POINTS = 6
# A built-in SM parameter every QCD value depends on.
_PARAMETER = ("normalization.alpha_s_me_check", 0.125)
# (artifact, colour accuracy, execution mode, declares correlators). The first
# also carries the API bundle whose UMAMI provider is built and driven;
# declaring correlators generates the compiled full-colour lane.
_ARTIFACTS = (
    ("recurrence-lc", "lc", "recurrence", False),
    ("recurrence-full", "full", "recurrence", False),
    ("compiled-lc", "lc", "compiled", False),
    ("on-the-fly-lc", "lc", "on-the-fly", False),
    ("eager-lc", "lc", "eager", False),
    ("correlated", "full", "auto", True),
)
_WRAPPED = ("recurrence-lc", "on-the-fly-lc", "correlated")
_RESTRICTED = (
    "restricted Symbolica",
    "Cannot start another restricted Symbolica thread",
)

_C_CLIENT = r"""
#include <rusticol.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One line per call: its name, its status and the bits of every double. */
static void emit(const char *name, int status, const double *values, size_t count) {
    printf("%s %d", name, status);
    if (status == RUSTICOL_STATUS_OK)
        for (size_t i = 0; i < count; ++i) {
            uint64_t bits;
            memcpy(&bits, values + i, sizeof bits);
            printf(" %016llx", (unsigned long long)bits);
        }
    putchar('\n');
}

typedef int (*indexed)(const RusticolRuntimeHandle *, size_t, char *, size_t, size_t *);

static char *indexed_text(const RusticolRuntimeHandle *runtime, indexed getter, size_t index) {
    size_t required = 0;
    if (getter(runtime, index, NULL, 0, &required) != RUSTICOL_STATUS_OK || required == 0)
        return NULL;
    char *text = malloc(required);
    if (text != NULL && getter(runtime, index, text, required, &required) != RUSTICOL_STATUS_OK) {
        free(text);
        return NULL;
    }
    return text;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    FILE *input = fopen(argv[3], "r");
    size_t points, legs, spin_leg;
    char parameter[256];
    double parameter_value;
    if (input == NULL || fscanf(input, "%zu %zu", &points, &legs) != 2) return 2;
    const size_t length = 4 * points * legs;
    double *momenta = calloc(length, sizeof *momenta);
    double *spin = calloc(8 * points, sizeof *spin);
    for (size_t i = 0; i < length; ++i) if (fscanf(input, "%lf", momenta + i) != 1) return 2;
    if (fscanf(input, "%zu", &spin_leg) != 1) return 2;
    for (size_t i = 0; i < 8 * points; ++i) if (fscanf(input, "%lf", spin + i) != 1) return 2;
    if (fscanf(input, "%255s %lf", parameter, &parameter_value) != 2) return 2;
    fclose(input);

    RusticolRuntimeHandle *runtime = NULL;
    int status = rusticol_runtime_load(argv[1], argv[2], NULL, &runtime);
    printf("load %d\n", status);
    if (status != RUSTICOL_STATUS_OK) return 3;
    size_t helicities = 0, colors = 0, orders = 0, correlations = 0;
    if (rusticol_runtime_helicity_count(runtime, &helicities) != RUSTICOL_STATUS_OK) return 4;
    if (rusticol_runtime_color_count(runtime, &colors) != RUSTICOL_STATUS_OK) return 4;
    const int order_status = rusticol_runtime_squared_order_count(runtime, &orders);
    const int correlation_status = rusticol_runtime_color_correlation_count(runtime, &correlations);
    printf("shape %zu %zu %d %zu %d %zu\n", helicities, colors, order_status, orders,
           correlation_status, correlations);
    char *first_helicity = indexed_text(runtime, rusticol_runtime_helicity_id, 0);
    char *last_helicity = indexed_text(runtime, rusticol_runtime_helicity_id, helicities - 1);
    char *first_color = indexed_text(runtime, rusticol_runtime_color_id, 0);
    if (first_helicity == NULL || last_helicity == NULL || first_color == NULL) return 5;
    const char *helicity_ids[] = {first_helicity, last_helicity};
    const char *color_ids[] = {first_color};
    const size_t capacity = points * (helicities * colors + orders + 2 * (correlations + 4) + 1);
    double *out = calloc(capacity, sizeof *out);
    double *partials = calloc(points * (orders + 1), sizeof *partials);
    uint32_t *helicity_by_point = calloc(points, sizeof *helicity_by_point);
    uint32_t *color_by_point = calloc(points, sizeof *color_by_point);
    for (size_t p = 0; p < points; ++p) {
        helicity_by_point[p] = (uint32_t)((3 * p + 1) % helicities);
        color_by_point[p] = (uint32_t)((p + 1) % colors);
    }
    size_t returned_helicities = 0, returned_colors = 0;

    status = rusticol_runtime_evaluate_f64(runtime, momenta, length, points, out, points);
    emit("values", status, out, points);
    status = rusticol_runtime_evaluate_resolved_f64(runtime, momenta, length, points, NULL, 0,
        NULL, 0, out, capacity, &returned_helicities, &returned_colors);
    emit("resolved", status, out, points * returned_helicities * returned_colors);
    returned_helicities = returned_colors = 0;
    status = rusticol_runtime_evaluate_resolved_f64(runtime, momenta, length, points,
        helicity_ids, 2, color_ids, 1, out, capacity, &returned_helicities, &returned_colors);
    emit("resolved-selected", status, out, points * returned_helicities * returned_colors);

#define SELECTED(name, h, hn, c, cn, hp, hpn, cp, cpn) \
    status = rusticol_runtime_evaluate_selected_f64(runtime, momenta, length, points, \
        h, hn, c, cn, hp, hpn, cp, cpn, out, points); \
    emit(name, status, out, points)
    SELECTED("selected-none", NULL, 0, NULL, 0, NULL, 0, NULL, 0);
    SELECTED("selected-helicity", helicity_ids, 1, NULL, 0, NULL, 0, NULL, 0);
    SELECTED("selected-helicities", helicity_ids, 2, NULL, 0, NULL, 0, NULL, 0);
    SELECTED("selected-color", NULL, 0, color_ids, 1, NULL, 0, NULL, 0);
    SELECTED("selected-both", helicity_ids, 1, color_ids, 1, NULL, 0, NULL, 0);
    SELECTED("selected-by-point-helicity", NULL, 0, NULL, 0, helicity_by_point, points, NULL, 0);
    SELECTED("selected-by-point-color", NULL, 0, NULL, 0, NULL, 0, color_by_point, points);
    SELECTED("selected-by-point-both", NULL, 0, NULL, 0, helicity_by_point, points,
             color_by_point, points);

    if (order_status == RUSTICOL_STATUS_OK) {
        status = rusticol_runtime_evaluate_orders_f64(runtime, momenta, length, points, NULL, 0,
            NULL, 0, out, points, partials, points * orders);
        emit("orders-totals", status, out, points);
        emit("orders-partials", status, partials, points * orders);
        status = rusticol_runtime_evaluate_orders_f64(runtime, momenta, length, points,
            helicity_ids, 1, NULL, 0, out, points, partials, points * orders);
        emit("orders-helicity", status, partials, points * orders);
    }

    if (correlation_status == RUSTICOL_STATUS_OK && correlations > 0) {
        size_t required = 0;
        if (rusticol_runtime_color_correlation_id(runtime, 0, NULL, 0, &required) != RUSTICOL_STATUS_OK)
            return 6;
        char *first_correlation = malloc(required);
        if (first_correlation == NULL || rusticol_runtime_color_correlation_id(
                runtime, 0, first_correlation, required, &required) != RUSTICOL_STATUS_OK)
            return 6;
        RusticolSpinCorrelationVector vector = {spin_leg, points, spin, 8 * points};
        RusticolCorrelatedRequest requests[] = {
            {"born", NULL, 0, 0},
            {first_correlation, NULL, 0, 0},
            {first_correlation, &vector, 1, 0},
            {"born", &vector, 1, 0},
        };
        status = rusticol_runtime_evaluate_correlated_many_f64(runtime, momenta, length, points,
            requests, 4, NULL, 0, out, 8 * points);
        emit("correlated", status, out, 8 * points);
        status = rusticol_runtime_set_spin_correlation_vectors_f64(runtime, &vector, 1);
        printf("spin-default %d\n", status);
        RusticolCorrelatedRequest defaults[] = {{first_correlation, NULL, 0, 1}};
        status = rusticol_runtime_evaluate_correlated_many_f64(runtime, momenta, length, points,
            defaults, 1, NULL, 0, out, 2 * points);
        emit("correlated-default-spin", status, out, 2 * points);
        free(first_correlation);
    }

    /* The binary64 kinematics improvement (m-lit lit-psimprove): the switch on,
     * MadLoop's default variant; one refused setting; then off again. */
    status = rusticol_runtime_set_kinematics_improvement(runtime, "ORIG");
    printf("kinematics-refused %d\n", status);
    status = rusticol_runtime_set_kinematics_improvement(runtime, "z-shift");
    printf("kinematics-setting %d\n", status);
    status = rusticol_runtime_set_kinematics_improvement_binary64(runtime, 1);
    printf("kinematics-binary64 %d\n", status);
    status = rusticol_runtime_evaluate_f64(runtime, momenta, length, points, out, points);
    emit("values-kinematics", status, out, points);
    SELECTED("selected-kinematics", NULL, 0, NULL, 0, helicity_by_point, points, NULL, 0);
    returned_helicities = returned_colors = 0;
    status = rusticol_runtime_evaluate_resolved_f64(runtime, momenta, length, points, NULL, 0,
        NULL, 0, out, capacity, &returned_helicities, &returned_colors);
    emit("resolved-kinematics", status, out, points * returned_helicities * returned_colors);
    if (order_status == RUSTICOL_STATUS_OK) {
        status = rusticol_runtime_evaluate_orders_f64(runtime, momenta, length, points, NULL, 0,
            NULL, 0, out, points, partials, points * orders);
        emit("orders-kinematics", status, partials, points * orders);
    }
    if (correlation_status == RUSTICOL_STATUS_OK && correlations > 0) {
        RusticolCorrelatedRequest born[] = {{"born", NULL, 0, 0}};
        status = rusticol_runtime_evaluate_correlated_many_f64(runtime, momenta, length, points,
            born, 1, NULL, 0, out, 2 * points);
        emit("correlated-kinematics", status, out, 2 * points);
    }
    {
        size_t required = 0;
        status = rusticol_runtime_take_warnings_json(runtime, NULL, 0, &required);
        char *notes = malloc(required > 0 ? required : 1);
        if (notes == NULL) return 8;
        notes[0] = '\0';
        if (status == RUSTICOL_STATUS_OK)
            status = rusticol_runtime_take_warnings_json(runtime, notes, required, &required);
        printf("kinematics-notes %d %s\n", status, notes);
        free(notes);
    }
    status = rusticol_runtime_set_kinematics_improvement_binary64(runtime, 0);
    printf("kinematics-off %d\n", status);

    RusticolWarmUpResult warm;
    memset(&warm, 0, sizeof warm);
    status = rusticol_runtime_warm_up_f64(runtime, momenta, 4 * legs, NULL, 0, NULL, 0, NULL,
        NULL, &warm);
    printf("warm-up %d %llu %llu %u %u\n", status, (unsigned long long)warm.query_count,
           (unsigned long long)warm.warmed_query_count, warm.already_warm,
           warm.first_evaluation_completed);
    status = rusticol_runtime_evaluate_f64(runtime, momenta, length, points, out, points);
    emit("values-after-warm-up", status, out, points);

    status = rusticol_runtime_set_model_parameter(runtime, parameter, parameter_value, 0.0);
    printf("set-parameter %d\n", status);
    double real = 0.0, imaginary = 0.0;
    status = rusticol_runtime_get_model_parameter(runtime, parameter, &real, &imaginary);
    double parameter_values[] = {real, imaginary};
    emit("get-parameter", status, parameter_values, 2);
    status = rusticol_runtime_evaluate_f64(runtime, momenta, length, points, out, points);
    emit("values-after-parameter", status, out, points);
    SELECTED("selected-after-parameter", NULL, 0, NULL, 0, helicity_by_point, points, NULL, 0);

    free(first_helicity);
    free(last_helicity);
    free(first_color);
    free(out);
    free(partials);
    free(helicity_by_point);
    free(color_by_point);
    free(momenta);
    free(spin);
    return rusticol_runtime_free(runtime) == RUSTICOL_STATUS_OK ? 0 : 7;
}
"""

_CPP_CLIENT = r"""
#include <rusticol.hpp>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

static void emit(const std::string &name, const std::vector<double> &values) {
    std::printf("%s", name.c_str());
    for (double value : values) {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof bits);
        std::printf(" %016llx", static_cast<unsigned long long>(bits));
    }
    std::printf("\n");
}

// Each call prints its doubles' bits, or "error" when the artifact refuses it.
static void attempt(const std::string &name, const std::function<std::vector<double>()> &call) {
    try {
        emit(name, call());
    } catch (const std::exception &) {
        std::printf("%s error\n", name.c_str());
    }
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    std::ifstream input(argv[3]);
    std::size_t points = 0, legs = 0, spin_leg = 0;
    input >> points >> legs;
    std::vector<double> momenta(4 * points * legs);
    for (double &value : momenta) input >> value;
    input >> spin_leg;
    rusticol::SpinCorrelationVector vector;
    vector.leg = spin_leg;
    vector.components.resize(points);
    for (auto &point : vector.components)
        for (auto &value : point) {
            double re = 0.0, im = 0.0;
            input >> re >> im;
            value = {re, im};
        }
    std::string parameter;
    double parameter_value = 0.0;
    input >> parameter >> parameter_value;
    if (!input) return 2;
    rusticol::Runtime runtime(argv[1], argv[2]);
    const auto helicities = runtime.helicities();
    const auto colors = runtime.colors();
    std::vector<std::uint32_t> helicity_by_point(points), color_by_point(points);
    for (std::size_t p = 0; p < points; ++p) {
        helicity_by_point[p] = static_cast<std::uint32_t>((3 * p + 1) % helicities.size());
        color_by_point[p] = static_cast<std::uint32_t>((p + 1) % colors.size());
    }
    attempt("values", [&] { return runtime.evaluate(momenta, points); });
    attempt("resolved", [&] { return runtime.evaluate_resolved(momenta, points).values; });
    attempt("selected-helicity", [&] {
        return runtime.evaluate_selected(momenta, points, {helicities[0].id}); });
    attempt("selected-color", [&] {
        return runtime.evaluate_selected(momenta, points, {}, {colors[0].id}); });
    attempt("selected-by-point-helicity", [&] {
        return runtime.evaluate_selected(momenta, points, {}, {}, helicity_by_point); });
    attempt("selected-by-point-color", [&] {
        return runtime.evaluate_selected(momenta, points, {}, {}, {}, color_by_point); });
    attempt("orders-partials", [&] { return runtime.evaluate_orders(momenta, points).partials; });
    attempt("correlated", [&] {
        const auto ids = runtime.color_correlation_ids();
        std::vector<rusticol::CorrelatedRequest> requests{
            {"born", std::vector<rusticol::SpinCorrelationVector>{}},
            {ids.at(0), std::vector<rusticol::SpinCorrelationVector>{}},
            {ids.at(0), std::vector<rusticol::SpinCorrelationVector>{vector}},
        };
        std::vector<double> flat;
        for (const auto &value : runtime.evaluate_correlated_many(momenta, points, requests).values) {
            flat.push_back(value.real());
            flat.push_back(value.imag());
        }
        return flat;
    });
    attempt("correlated-default-spin", [&] {
        runtime.set_spin_correlation_vectors({vector});
        std::vector<double> flat;
        for (const auto &value : runtime.evaluate_correlated(momenta, points, runtime.color_correlation_ids().at(0))) {
            flat.push_back(value.real());
            flat.push_back(value.imag());
        }
        return flat;
    });
    attempt("values-kinematics", [&] {
        runtime.set_kinematics_improvement("z-shift");
        runtime.set_kinematics_improvement_binary64(true);
        auto values = runtime.evaluate(momenta, points);
        runtime.set_kinematics_improvement_binary64(false);
        return values;
    });
    attempt("warm-up", [&] {
        const std::vector<double> point(momenta.begin(), momenta.begin() + 4 * legs);
        const auto warm = runtime.warm_up(point);
        return std::vector<double>{static_cast<double>(warm.query_count),
                                   static_cast<double>(warm.warmed_query_count)};
    });
    attempt("values-after-parameter", [&] {
        runtime.set_model_parameter(parameter, parameter_value);
        return runtime.evaluate(momenta, points);
    });
    return 0;
}
"""

_FORTRAN_CLIENT = r"""
program tree_licence_probe
  use, intrinsic :: iso_c_binding
  use rusticol
  implicit none
  type(rusticol_runtime) :: runtime
  type(rusticol_helicity_configuration), allocatable :: helicities(:)
  type(rusticol_color_component), allocatable :: colors(:)
  type(rusticol_correlated_request) :: requests(2)
  type(rusticol_warm_up_result) :: warm
  type(rusticol_string), allocatable :: correlation_ids(:)
  character(len=4096) :: artifact, key, filename
  character(len=256) :: parameter
  integer :: unit, points, legs, spin_leg, p, mu
  integer(c_int) :: ierr
  integer(c_int32_t), allocatable :: helicity_by_point(:), color_by_point(:)
  real(c_double), allocatable :: momenta(:), values(:), resolved(:, :, :), totals(:), partials(:, :)
  complex(c_double_complex), allocatable :: correlated(:, :)
  real(c_double) :: re, im, parameter_value

  call get_command_argument(1, artifact)
  call get_command_argument(2, key)
  call get_command_argument(3, filename)
  open(newunit=unit, file=trim(filename), status="old", action="read")
  read(unit, *) points, legs
  allocate(momenta(4 * points * legs))
  read(unit, *) momenta
  read(unit, *) spin_leg
  requests(1)%color_correlation = "born"
  allocate(requests(1)%spin_vectors(0))
  allocate(requests(2)%spin_vectors(1))
  requests(2)%spin_vectors(1)%leg = int(spin_leg, c_size_t)
  allocate(requests(2)%spin_vectors(1)%components(4, points))
  do p = 1, points
    do mu = 1, 4
      read(unit, *) re, im
      requests(2)%spin_vectors(1)%components(mu, p) = cmplx(re, im, kind=c_double)
    end do
  end do
  read(unit, *) parameter, parameter_value
  close(unit)

  call runtime%load(trim(artifact), process_key=trim(key))
  helicities = runtime%helicities()
  colors = runtime%colors()
  allocate(helicity_by_point(points), color_by_point(points))
  do p = 0, points - 1
    helicity_by_point(p + 1) = int(mod(3 * p + 1, size(helicities)), c_int32_t)
    color_by_point(p + 1) = int(mod(p + 1, size(colors)), c_int32_t)
  end do
  call runtime%evaluate(momenta, int(points, c_size_t), values, ierr=ierr)
  call emit("values", ierr, values)
  call runtime%evaluate_resolved(momenta, int(points, c_size_t), resolved, ierr=ierr)
  if (ierr == 0) then
    call emit("resolved", ierr, reshape(resolved, [size(resolved)]))
  else
    call emit("resolved", ierr, [real(c_double) ::])
  end if
  call runtime%evaluate_selected(momenta, int(points, c_size_t), values, &
      helicity_ids=[helicities(1)%id], ierr=ierr)
  call emit("selected-helicity", ierr, values)
  call runtime%evaluate_selected(momenta, int(points, c_size_t), values, &
      color_ids=[colors(1)%id], ierr=ierr)
  call emit("selected-color", ierr, values)
  call runtime%evaluate_selected(momenta, int(points, c_size_t), values, &
      helicity_by_point=helicity_by_point, ierr=ierr)
  call emit("selected-by-point-helicity", ierr, values)
  call runtime%evaluate_selected(momenta, int(points, c_size_t), values, &
      color_flow_by_point=color_by_point, ierr=ierr)
  call emit("selected-by-point-color", ierr, values)
  call runtime%evaluate_orders(momenta, int(points, c_size_t), totals, partials, ierr=ierr)
  if (ierr == 0) then
    call emit("orders-partials", ierr, reshape(partials, [size(partials)]))
  else
    call emit("orders-partials", ierr, [real(c_double) ::])
  end if
  correlation_ids = runtime%color_correlation_ids(ierr=ierr)
  if (ierr == 0) then
    requests(2)%color_correlation = correlation_ids(1)%value
    call runtime%evaluate_correlated_many(momenta, int(points, c_size_t), requests, correlated, &
        ierr=ierr)
  end if
  if (ierr == 0) then
    call emit("correlated", ierr, [(real(correlated(p, 1), c_double), aimag(correlated(p, 1)), &
        p = 1, points), (real(correlated(p, 2), c_double), aimag(correlated(p, 2)), p = 1, points)])
  else
    call emit("correlated", ierr, [real(c_double) ::])
  end if
  call runtime%set_kinematics_improvement("z-shift", ierr=ierr)
  if (ierr == 0) call runtime%set_kinematics_improvement_binary64(.true., ierr=ierr)
  if (ierr == 0) call runtime%evaluate(momenta, int(points, c_size_t), values, ierr=ierr)
  call emit("values-kinematics", ierr, values)
  call runtime%set_kinematics_improvement_binary64(.false., ierr=ierr)
  call runtime%warm_up(momenta(1:4 * legs), warm, ierr=ierr)
  call emit("warm-up", ierr, [real(warm%query_count, c_double), real(warm%warmed_query_count, c_double)])
  call runtime%set_model_parameter(trim(parameter), parameter_value, ierr=ierr)
  if (ierr == 0) call runtime%evaluate(momenta, int(points, c_size_t), values, ierr=ierr)
  call emit("values-after-parameter", ierr, values)
  call runtime%close()

contains

  subroutine emit(name, status, items)
    character(len=*), intent(in) :: name
    integer(c_int), intent(in) :: status
    real(c_double), intent(in) :: items(:)
    integer :: i
    if (status /= 0) then
      write(*, '(A,1X,A)') name, "error"
      return
    end if
    write(*, '(A)', advance='no') name
    do i = 1, size(items)
      write(*, '(1X,Z16.16)', advance='no') transfer(items(i), 0_c_int64_t)
    end do
    write(*, '(A)') ""
  end subroutine emit
end program tree_licence_probe
"""

_RUST_CLIENT = r"""
#[allow(dead_code)]
mod rusticol { include!(env!("RUSTICOL_RUST_SOURCE")); }
use rusticol::{Complex64, CorrelatedRequest, Runtime, Selectors, SpinCorrelationVector};

// Each call prints its doubles' bits, or "error" when the artifact refuses it.
fn emit(name: &str, values: Result<Vec<f64>, rusticol::Error>) {
    match values {
        Ok(values) => {
            let bits: Vec<String> = values.iter().map(|v| format!("{:016x}", v.to_bits())).collect();
            println!("{} {}", name, bits.join(" "));
        }
        Err(_) => println!("{} error", name),
    }
}

fn flat(values: &[Complex64]) -> Vec<f64> {
    values.iter().flat_map(|v| [v.re, v.im]).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read_to_string(&args[3])?;
    let mut tokens = data.split_whitespace();
    let mut next = || tokens.next().expect("input token").to_owned();
    let points: usize = next().parse()?;
    let legs: usize = next().parse()?;
    let mut momenta = Vec::new();
    for _ in 0..4 * points * legs {
        momenta.push(next().parse::<f64>()?);
    }
    let leg: usize = next().parse()?;
    let mut components = vec![[Complex64::default(); 4]; points];
    for point in &mut components {
        for value in point {
            let re: f64 = next().parse()?;
            let im: f64 = next().parse()?;
            *value = Complex64::new(re, im);
        }
    }
    let vector = SpinCorrelationVector { leg, components };
    let parameter = next();
    let parameter_value: f64 = next().parse()?;
    let mut runtime = Runtime::load(&args[1], Some(&args[2]), None)?;
    let helicities = runtime.helicities()?;
    let colors = runtime.colors()?;
    let helicity_by_point: Vec<u32> = (0..points).map(|p| ((3 * p + 1) % helicities.len()) as u32).collect();
    let color_by_point: Vec<u32> = (0..points).map(|p| ((p + 1) % colors.len()) as u32).collect();
    emit("values", runtime.evaluate_f64(&momenta, points));
    emit("resolved", runtime.evaluate_resolved_f64(&momenta, points, &Selectors::all()).map(|r| r.values().to_vec()));
    let one_helicity = Selectors::all().with_helicities([helicities[0].id.as_str()]);
    emit("selected-helicity", runtime.evaluate_selected_f64(&momenta, points, &one_helicity, None, None));
    let one_color = Selectors::all().with_colors([colors[0].id.as_str()]);
    emit("selected-color", runtime.evaluate_selected_f64(&momenta, points, &one_color, None, None));
    emit(
        "selected-by-point-helicity",
        runtime.evaluate_selected_f64(&momenta, points, &Selectors::all(), Some(&helicity_by_point), None),
    );
    emit(
        "selected-by-point-color",
        runtime.evaluate_selected_f64(&momenta, points, &Selectors::all(), None, Some(&color_by_point)),
    );
    emit(
        "orders-partials",
        runtime.evaluate_orders_f64(&momenta, points, &Selectors::all()).map(|o| o.partials().to_vec()),
    );
    let correlated = runtime.color_correlation_ids().and_then(|ids| {
        let requests = vec![
            CorrelatedRequest { color_correlation: "born".into(), spin_vectors: Some(vec![]) },
            CorrelatedRequest { color_correlation: ids[0].clone(), spin_vectors: Some(vec![]) },
            CorrelatedRequest { color_correlation: ids[0].clone(), spin_vectors: Some(vec![vector.clone()]) },
        ];
        runtime.evaluate_correlated_many_f64(&momenta, points, &requests, &[]).map(|c| flat(&c.values))
    });
    emit("correlated", correlated);
    let defaults = runtime.color_correlation_ids().and_then(|ids| {
        runtime.set_spin_correlation_vectors(&[vector.clone()])?;
        let request = CorrelatedRequest { color_correlation: ids[0].clone(), spin_vectors: None };
        runtime.evaluate_correlated_many_f64(&momenta, points, &[request], &[]).map(|c| flat(&c.values))
    });
    emit("correlated-default-spin", defaults);
    let kinematics = runtime
        .set_kinematics_improvement("z-shift")
        .and_then(|()| runtime.set_kinematics_improvement_binary64(true))
        .and_then(|()| runtime.evaluate_f64(&momenta, points));
    runtime.set_kinematics_improvement_binary64(false)?;
    emit("values-kinematics", kinematics);
    emit(
        "warm-up",
        runtime
            .warm_up(&momenta[..4 * legs], &Selectors::all(), None, None)
            .map(|w| vec![w.query_count as f64, w.warmed_query_count as f64]),
    );
    let after = runtime
        .set_model_parameter(&parameter, parameter_value, 0.0)
        .and_then(|()| runtime.evaluate_f64(&momenta, points));
    emit("values-after-parameter", after);
    Ok(())
}
"""

# The UMAMI provider is a shared library; this driver (Python's ctypes, no
# pyAmpliCol import) evaluates its matrix element, every fixed helicity and a
# sampled one, with the colour index on leading-colour providers.
_UMAMI_DRIVER = r"""
import ctypes as ct, json, struct, sys

library, artifact, input_path = sys.argv[1:4]
provider = ct.CDLL(library)
provider.umami_initialize.argtypes = (ct.POINTER(ct.c_void_p), ct.c_char_p)
provider.umami_free.argtypes = (ct.c_void_p,)
provider.umami_matrix_element.argtypes = (
    ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.POINTER(ct.c_int),
    ct.POINTER(ct.c_void_p), ct.c_size_t, ct.POINTER(ct.c_int), ct.POINTER(ct.c_void_p),
)
spec = json.loads(open(input_path).read())
points, helicities, lc = spec["points"], spec["helicities"], spec["lc"]
stride, legs = len(points), len(points[0])
handle = ct.c_void_p()
assert provider.umami_initialize(ct.byref(handle), artifact.encode()) == 0


def call(name, helicity=None, random_helicity=0.5):
    momenta = (ct.c_double * (4 * legs * stride))(*(
        float(points[event][leg][component])
        for component in range(4) for leg in range(legs) for event in range(stride)
    ))
    # UMAMI_IN_MOMENTA 0, UMAMI_IN_FLAVOR_INDEX 2, UMAMI_IN_RANDOM_HELICITY 4,
    # UMAMI_IN_HELICITY_INDEX 6, UMAMI_IN_CHANNEL_INDEX 8.
    inputs = {
        0: momenta,
        2: (ct.c_int * stride)(*[0] * stride),
        4: (ct.c_double * stride)(*[random_helicity] * stride),
        8: (ct.c_uint * stride)(*[0] * stride),
    }
    if helicity is not None:
        inputs[6] = (ct.c_int * stride)(*[helicity] * stride)
    outputs = {0: (ct.c_double * stride)(), 3: (ct.c_int * stride)()}
    if lc:
        outputs[2] = (ct.c_int * stride)()
    status = provider.umami_matrix_element(
        handle, stride, stride, 0, len(inputs), (ct.c_int * len(inputs))(*inputs),
        (ct.c_void_p * len(inputs))(*(ct.cast(v, ct.c_void_p) for v in inputs.values())),
        len(outputs), (ct.c_int * len(outputs))(*outputs),
        (ct.c_void_p * len(outputs))(*(ct.cast(v, ct.c_void_p) for v in outputs.values())),
    )
    bits = " ".join(struct.pack("<d", v).hex() for v in outputs[0])
    labels = " ".join(str(v) for key in sorted(outputs) if key != 0 for v in outputs[key])
    print(name, status, bits, "|", labels, flush=True)


call("matrix-element")
for index in helicities:
    call(f"helicity-{index}", helicity=index)
call("sampled-helicity", random_helicity=0.75)
assert provider.umami_free(handle) == 0
"""


def _rambo(rng: random.Random, energy: float) -> list[list[float]]:
    """A massless 2 -> 3 point in the centre-of-mass frame (RAMBO)."""

    raw = []
    for _ in range(3):
        cos = 2.0 * rng.random() - 1.0
        phi = 2.0 * math.pi * rng.random()
        e = -math.log(rng.random() * rng.random())
        sin = math.sqrt(1.0 - cos * cos)
        raw.append([e, e * sin * math.cos(phi), e * sin * math.sin(phi), e * cos])
    total = [sum(p[i] for p in raw) for i in range(4)]
    mass = math.sqrt(total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2)
    boost = [-total[i] / mass for i in (1, 2, 3)]
    gamma, scale = total[0] / mass, energy / mass
    a = 1.0 / (1.0 + gamma)
    final = []
    for p in raw:
        bq = sum(boost[i] * p[i + 1] for i in range(3))
        final.append(
            [scale * (gamma * p[0] + bq)]
            + [
                scale * (p[i + 1] + boost[i] * p[0] + a * bq * boost[i])
                for i in range(3)
            ]
        )
    half = energy / 2.0
    return [[half, 0.0, 0.0, half], [half, 0.0, 0.0, -half], *final]


@pytest.fixture(scope="module")
def tree_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    root = tmp_path_factory.mktemp("tree-licence-free")
    declarations = CorrelatorConfig(
        color_correlations=(
            ColorCorrelator(
                "cascade",
                (EmitGluon(2, -3), EmitGluon(-3, -1), EmitGluon(-3, -2)),
                (EmitGluon(1, -1), EmitGluon(3, -2), EmitGluon(4, -3)),
            ),
        ),
        spin_correlations=((_SPIN_LEG,),),
    )
    artifacts = {}
    for name, accuracy, mode, correlated in _ARTIFACTS:
        config = RunConfig(
            action="generate",
            color=ColorConfig(accuracy=accuracy),
            generation=GenerationConfig(
                workers=1, emit_api_bundle=name == _ARTIFACTS[0][0]
            ),
            evaluator=EvaluatorConfig(
                execution_mode=mode,
                optimization=EvaluatorOptimizationConfig(cores=1),
                jit=JITConfig(optimization_level=1),
            ),
        )
        Generator(config).generate(
            ProcessSet.from_expressions((_PROCESS,), names=(_NAME,)),
            root / name,
            model=ModelSource.built_in_sm(),
            correlators=declarations if correlated else None,
        )
        artifacts[name] = root / name
    return artifacts


def _umami_provider(artifact: Path, build: Path) -> dict:
    if shutil.which("make") is None:
        _unavailable("the UMAMI provider build requires make")
    completed = subprocess.run(
        [
            "make",
            "-C",
            str(artifact / "API/umami"),
            f"BUILD_DIR={build}",
            f"RUSTICOL_CONFIG={_config_executable()}",
        ],
        env=_source_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    metadata = json.loads((artifact / "API/umami/metadata.json").read_text())
    runtime = metadata["channels"][0]["processes"][0]["runtime"]
    return {
        "library": str(build / metadata["provider"]["library"]),
        "helicities": list(runtime["helicity_export"]),
        "lc": metadata["provider"]["color_accuracy"] == "lc",
    }


def _lines(output: str) -> dict[str, list[str]]:
    calls = {}
    for line in output.splitlines():
        # Fortran writes the bits in upper-case hexadecimal.
        name, *fields = line.lower().split()
        assert name not in calls, line
        calls[name] = fields
    return calls


def test_binary64_tree_observables_need_no_symbolica_licensing(
    tree_artifacts: dict[str, Path], native_sdk: _NativeSdk, tmp_path: Path
) -> None:
    spec = importlib.util.spec_from_file_location(
        "loop_artifact_portability", _ROOT / "tools" / "loop_artifact_portability.py"
    )
    assert spec is not None and spec.loader is not None
    portability = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portability)

    rng = random.Random(20261108)
    points = [_rambo(rng, 1000.0) for _ in range(_POINTS)]
    # A complex spin source per point on the gluon (exact binary64 values).
    spin = [
        [(0.0, 0.0), (rng.uniform(-1, 1), 0.25), (rng.uniform(-1, 1), -0.5), (0.0, 1.0)]
        for _ in points
    ]
    lines = [f"{_POINTS} {len(points[0])}"]
    lines.extend(repr(c) for point in points for leg in point for c in leg)
    lines.append(str(_SPIN_LEG))
    lines.extend(repr(x) for vector in spin for pair in vector for x in pair)
    lines.append(f"{_PARAMETER[0]} {_PARAMETER[1]!r}")
    inputs = tmp_path / "input.txt"
    inputs.write_text("\n".join(lines) + "\n")

    clients = {}
    for language, source in (
        ("c", _C_CLIENT),
        ("cpp", _CPP_CLIENT),
        ("fortran", _FORTRAN_CLIENT),
        ("rust", _RUST_CLIENT),
    ):
        directory = tmp_path / f"client-{language}"
        directory.mkdir()
        clients[language] = _compile_probe(language, source, native_sdk, directory)
    first = tree_artifacts[_ARTIFACTS[0][0]]
    umami = _umami_provider(first, tmp_path / "umami build")
    (tmp_path / "umami.json").write_text(json.dumps({"points": points, **umami}))
    (tmp_path / "umami_driver.py").write_text(textwrap.dedent(_UMAMI_DRIVER).lstrip())
    commands = {
        f"c {name}": [str(clients["c"]), str(artifact), _NAME, str(inputs)]
        for name, artifact in tree_artifacts.items()
    }
    for language in ("cpp", "fortran", "rust"):
        for name in _WRAPPED:
            commands[f"{language} {name}"] = [
                str(clients[language]),
                str(tree_artifacts[name]),
                _NAME,
                str(inputs),
            ]
    commands["umami"] = [
        sys.executable,
        "-I",
        str(tmp_path / "umami_driver.py"),
        umami["library"],
        str(first),
        str(tmp_path / "umami.json"),
    ]

    def run_all(env: dict[str, str]) -> dict[str, str]:
        # Every client at once: concurrent processes.
        running = {
            name: subprocess.Popen(
                command,
                env=env,
                cwd=tmp_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for name, command in commands.items()
        }
        outputs = {}
        for name, process in running.items():
            stdout, stderr = process.communicate(timeout=600)
            text = stdout[-3000:] + stderr[-3000:]
            assert process.returncode == 0, f"{name}:\n{text}"
            assert not any(marker in text for marker in _RESTRICTED), f"{name}:\n{text}"
            outputs[name] = stdout
        return outputs

    lock = tmp_path / "symbolica-locks"
    lock.mkdir()
    lock.chmod(0o700)
    base = {
        key: value
        for key, value in _source_environment().items()
        if key != "SYMBOLICA_LICENSE"
    }
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        assert permit.record["verified"], permit.record
        unlicensed = run_all(
            permit.environment({**base, "PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"})
        )
    licensed = run_all(_source_environment())
    assert unlicensed == licensed

    # Coverage: every observable was served somewhere (a refusal is typed and
    # compared as well, but it would not exercise the evaluation).
    calls = {name: _lines(output) for name, output in licensed.items()}
    served = {
        call
        for name, record in calls.items()
        if name.startswith("c ")
        for call, fields in record.items()
        if fields[:1] == ["0"]
    }
    assert served >= {
        "values",
        "resolved",
        "resolved-selected",
        "selected-none",
        "selected-helicity",
        "selected-helicities",
        "selected-color",
        "selected-both",
        "selected-by-point-helicity",
        "selected-by-point-color",
        "selected-by-point-both",
        "orders-totals",
        "orders-partials",
        "orders-helicity",
        "correlated",
        "correlated-default-spin",
        "warm-up",
        "values-after-parameter",
        "get-parameter",
        # m-lit lit-psimprove: the binary64 kinematics improvement switch, on
        # every f64 observable, with no licensing either.
        "values-kinematics",
        "selected-kinematics",
        "resolved-kinematics",
        "orders-kinematics",
        "correlated-kinematics",
    }, served
    # The parameter update changes the values (it is evaluated, not ignored).
    reference = calls[f"c {_ARTIFACTS[0][0]}"]
    assert reference["values-after-parameter"] != reference["values"]
    # The kinematics improvement (m-lit lit-psimprove): the native policy's
    # one refusal (status 1), the switch's records in the warnings channel
    # (MadLoop's default variant; these beam-frame points are produced by it),
    # and the Python tree runtime's binary64 switch gives the C ABI's values
    # bitwise (one native implementation, the same binary64 masses).
    for name in tree_artifacts:
        record = calls[f"c {name}"]
        assert record["kinematics-refused"] == ["1"], (name, record)
        assert record["kinematics-setting"] == record["kinematics-binary64"] == ["0"]
        assert record["kinematics-off"] == ["0"]
        notes = " ".join(record["kinematics-notes"][1:])
        assert record["kinematics-notes"][0] == "0", (name, notes)
        assert (
            f"kinematics-improvement: double at precision 16 (z-shift): z-shift {_POINTS}"
            in notes
        ), (name, notes)
        assert "kinematics-not-improved" not in notes, (name, notes)
    python = Runtime.load(tree_artifacts[_ARTIFACTS[0][0]], process=_NAME)
    improved = python.evaluate(points, kinematics_improvement_binary64=True)
    assert [struct.pack(">d", float(v)).hex() for v in improved] == reference[
        "values-kinematics"
    ][1:]
    assert any("(z-shift): z-shift" in text for text in python.take_warnings())
    plain = python.evaluate(points)
    assert [struct.pack(">d", float(v)).hex() for v in plain] == reference["values"][1:]
    for language in ("cpp", "fortran", "rust"):
        wrapped = {}
        for name in _WRAPPED:
            for call, fields in calls[f"{language} {name}"].items():
                if fields != ["error"]:
                    wrapped.setdefault(call, (name, fields))
        assert set(wrapped) == {
            "values",
            "resolved",
            "selected-helicity",
            "selected-color",
            "selected-by-point-helicity",
            "selected-by-point-color",
            "orders-partials",
            "correlated",
            "warm-up",
            "values-after-parameter",
            "values-kinematics",
        } | ({"correlated-default-spin"} if language != "fortran" else set()), (
            language,
            sorted(wrapped),
        )
        # The wrappers return the C ABI's values bitwise.
        for call in (
            "values",
            "resolved",
            "values-after-parameter",
            "values-kinematics",
        ):
            name, fields = wrapped[call]
            assert fields == calls[f"c {name}"][call][1:], (language, call)
    umami_lines = licensed["umami"].splitlines()
    assert len(umami_lines) == 2 + len(umami["helicities"])
    assert all(line.split()[1] == "0" for line in umami_lines), umami_lines
