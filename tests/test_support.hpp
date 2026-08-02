#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "core/snapshot.hpp"

namespace cps_test {

// Execution tests need a real device. On a machine without one they must skip
// with the reason stated rather than fail, so a continuous integration run on
// a GPU free runner stays honest instead of merely green.
inline bool cuda_device_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess && count > 0;
}

#define SKIP_WITHOUT_GPU()                                                            \
    do {                                                                              \
        if (!::cps_test::cuda_device_available()) {                                   \
            GTEST_SKIP() << "hardware required: no CUDA device is visible, so this "  \
                            "execution test cannot run here";                         \
        }                                                                             \
    } while (false)

// Independent double precision reference for the gravitational acceleration.
//
// Deliberately the most literal possible transcription of the force law: no
// tiling, no reciprocal square root, no single precision anywhere. Its only
// job is to be obviously correct, so that when it disagrees with a kernel the
// kernel is what is wrong.
inline void reference_accelerations(const cps::HostParticleState& state, double softening,
                                    double gravitational_constant, std::vector<double>& ax,
                                    std::vector<double>& ay, std::vector<double>& az) {
    const std::size_t n = state.size();
    ax.assign(n, 0.0);
    ay.assign(n, 0.0);
    az.assign(n, 0.0);
    const double softening_sq = softening * softening;

    for (std::size_t i = 0; i < n; ++i) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            const double dx = static_cast<double>(state.x[j]) - static_cast<double>(state.x[i]);
            const double dy = static_cast<double>(state.y[j]) - static_cast<double>(state.y[i]);
            const double dz = static_cast<double>(state.z[j]) - static_cast<double>(state.z[i]);
            const double dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;
            const double inv = 1.0 / std::sqrt(dist_sq);
            const double scale = static_cast<double>(state.mass[j]) * inv * inv * inv;
            sum_x += dx * scale;
            sum_y += dy * scale;
            sum_z += dz * scale;
        }
        ax[i] = gravitational_constant * sum_x;
        ay[i] = gravitational_constant * sum_y;
        az[i] = gravitational_constant * sum_z;
    }
}

inline double reference_potential(const cps::HostParticleState& state, double softening,
                                  double gravitational_constant) {
    const std::size_t n = state.size();
    const double softening_sq = softening * softening;
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double dx = static_cast<double>(state.x[j]) - static_cast<double>(state.x[i]);
            const double dy = static_cast<double>(state.y[j]) - static_cast<double>(state.y[i]);
            const double dz = static_cast<double>(state.z[j]) - static_cast<double>(state.z[i]);
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + softening_sq);
            total -= static_cast<double>(state.mass[i]) * static_cast<double>(state.mass[j]) / dist;
        }
    }
    return gravitational_constant * total;
}

// A reproducible cloud with no two particles on top of each other.
inline cps::HostParticleState random_cloud(int count, uint32_t seed, float extent = 1.0f) {
    cps::HostParticleState state;
    state.resize(static_cast<std::size_t>(count));
    uint32_t rng = seed == 0u ? 1u : seed;
    auto next = [&rng]() {
        // xorshift32, written out so the sequence does not depend on the
        // standard library version.
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return (static_cast<double>(rng) + 0.5) / 4294967296.0;
    };
    for (int i = 0; i < count; ++i) {
        state.x[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * extent);
        state.y[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * extent);
        state.z[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * extent);
        state.vx[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * 0.1);
        state.vy[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * 0.1);
        state.vz[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * 0.1);
        state.mass[static_cast<std::size_t>(i)] = static_cast<float>(0.5 + next());
    }
    return state;
}

inline double relative_error(double measured, double expected) {
    const double scale = std::abs(expected);
    return scale > 0.0 ? std::abs(measured - expected) / scale : std::abs(measured);
}

// Upper tail of the chi squared distribution, needed by the Plummer radial
// density gate. Implemented here rather than pulled in from a statistics
// library because it is forty lines and adding a dependency to evaluate one
// special function would be a poor trade.
//
// Q(a, x) is the regularised upper incomplete gamma function. The series
// converges quickly below x = a + 1 and the continued fraction above it, which
// is the standard split.
inline double incomplete_gamma_series(double a, double x) {
    const int max_iterations = 1000;
    const double epsilon = 3.0e-14;
    double ap = a;
    double sum = 1.0 / a;
    double term = sum;
    for (int n = 0; n < max_iterations; ++n) {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if (std::abs(term) < std::abs(sum) * epsilon) {
            break;
        }
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

inline double incomplete_gamma_continued_fraction(double a, double x) {
    const int max_iterations = 1000;
    const double epsilon = 3.0e-14;
    const double tiny = 1.0e-300;

    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= max_iterations; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny) {
            d = tiny;
        }
        c = b + an / c;
        if (std::abs(c) < tiny) {
            c = tiny;
        }
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < epsilon) {
            break;
        }
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

inline double regularised_upper_gamma(double a, double x) {
    if (x < 0.0 || a <= 0.0) {
        return std::nan("");
    }
    if (x == 0.0) {
        return 1.0;
    }
    if (x < a + 1.0) {
        return 1.0 - incomplete_gamma_series(a, x);
    }
    return incomplete_gamma_continued_fraction(a, x);
}

// Probability of seeing a chi squared statistic at least this large by chance,
// given the number of degrees of freedom.
inline double chi_square_p_value(double chi_square, int degrees_of_freedom) {
    if (degrees_of_freedom <= 0) {
        return std::nan("");
    }
    return regularised_upper_gamma(0.5 * degrees_of_freedom, 0.5 * chi_square);
}

// Records the measured value of a physics gate alongside its limit.
//
// The report quotes gate results from this file rather than from anything typed
// by hand, so a gate that moves cannot silently disagree with the prose that
// describes it. The first call in a process truncates, so a rerun replaces the
// previous results instead of appending to them.
inline void record_gate(const std::string& name, double measured, double limit,
                        const std::string& comparison, bool passed) {
#if defined(CPS_PROJECT_ROOT)
    static bool started = false;
    const std::string path =
        std::string(CPS_PROJECT_ROOT) + "/experiments/results/validation_gates.csv";
    std::FILE* file = std::fopen(path.c_str(), started ? "a" : "w");
    if (file == nullptr) {
        return;
    }
    if (!started) {
        std::fprintf(file, "gate,measured,limit,comparison,passed\n");
        started = true;
    }
    std::fprintf(file, "%s,%.17g,%.17g,%s,%d\n", name.c_str(), measured, limit, comparison.c_str(),
                 passed ? 1 : 0);
    std::fclose(file);
#else
    (void)name;
    (void)measured;
    (void)limit;
    (void)comparison;
    (void)passed;
#endif
}

// Writes a two column artifact next to the repository so the report can plot a
// gate rather than merely assert it.
inline void write_series_csv(const std::string& path, const std::string& header,
                             const std::vector<double>& first,
                             const std::vector<double>& second) {
    std::FILE* file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        return;
    }
    std::fprintf(file, "%s\n", header.c_str());
    const std::size_t n = std::min(first.size(), second.size());
    for (std::size_t i = 0; i < n; ++i) {
        std::fprintf(file, "%.17g,%.17g\n", first[i], second[i]);
    }
    std::fclose(file);
}

}  // namespace cps_test
