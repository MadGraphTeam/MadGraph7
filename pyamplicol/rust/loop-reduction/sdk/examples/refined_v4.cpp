// SPDX-License-Identifier: 0BSD
// Caller-supplied higher-accuracy kinematics through the C++ wrapper
// (VirtualRuntime::evaluate_refined): ./refined_v4_cpp artifact process
// input.txt. Input: "points legs", then per point its 4*legs binary64 words,
// its 4*legs correction words and its mu_R^2, every double in C99
// hexadecimal. Output per call: "call <name>" and "virtual ..." (hexadecimal
// doubles), or "refused <status> <message>": the plain evaluation of the
// binary64 words, the refined one with zero corrections, the refined point,
// and a refined point whose binary64 words are not its rounding.
#include "rusticol_virtual.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {
double hex(std::istream &in) {
  std::string text;
  if (!(in >> text)) throw std::runtime_error("short input");
  return std::strtod(text.c_str(), nullptr);
}
void print(const char *name, const rusticol::VirtualEvaluation &result) {
  std::printf("call %s\nvirtual", name);
  for (double v : result.virtual_values()) std::printf(" %a", v);
  std::printf("\n");
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  std::ifstream in(argv[3]);
  std::size_t points = 0, legs = 0;
  in >> points >> legs;
  const std::size_t width = legs * 4;
  std::vector<double> hi(points * width), lo(points * width), scales(points);
  for (std::size_t p = 0; p < points; ++p) {
    for (std::size_t i = 0; i < width; ++i) hi[p * width + i] = hex(in);
    for (std::size_t i = 0; i < width; ++i) lo[p * width + i] = hex(in);
    scales[p] = hex(in);
  }
  try {
    rusticol::VirtualRuntime runtime(argv[1], std::string(argv[2]));
    runtime.set_poles("none");
    runtime.set_reduction("ninjars");
    print("plain", runtime.evaluate(hi, points, scales));
    print("zero", runtime.evaluate_refined(hi, std::vector<double>(hi.size(), 0.0), points, scales));
    print("refined", runtime.evaluate_refined(hi, lo, points, scales));
    // Point 1, leg 1, py: a correction far beyond half an ulp of its word.
    std::vector<double> far = lo;
    far[width + 6] = 1e-3 * (std::fabs(hi[width + 6]) + 1.0);
    std::printf("call inconsistent\n");
    try {
      runtime.evaluate_refined(hi, far, points, scales);
      std::printf("accepted\n");
    } catch (const rusticol::VirtualError &refused) {
      std::printf("refused %d %s\n", refused.status(), refused.what());
    }
  } catch (const rusticol::VirtualError &error) {
    std::printf("error %d %s\n", error.status(), error.what());
    return 1;
  }
  return 0;
}
