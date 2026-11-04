#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "cps/core/config.hpp"

namespace cps::bench {

// The arithmetic behind the derived columns of a benchmark row. It lives here,
// away from the driver, so the published numbers can be tested on a machine
// with no GPU. Every time is milliseconds per step, and a time that is not
// positive gives zero rather than a division by zero.

// The conventional operation count for one softened gravitational interaction,
// as used by Nyland, Harris and Prins in GPU Gems 3. Counting it any other way
// would make the numbers here incomparable with every other N-body result in
// the literature, so the convention is followed and stated rather than
// recounted.
inline constexpr double FLOPS_PER_INTERACTION = 20.0;

inline double steps_per_second(double ms_per_step) {
    return ms_per_step > 0.0 ? 1000.0 / ms_per_step : 0.0;
}

// N-body counts the N squared pair interactions of a step; the fluid has no such
// count, so its column holds particles updated per second.
inline double interactions_per_second(SolverType solver, int num_particles, double ms_per_step) {
    const double n = static_cast<double>(num_particles);
    const double interactions = solver == SolverType::NBody ? n * n : n;
    return ms_per_step > 0.0 ? interactions / (ms_per_step * 1.0e-3) : 0.0;
}

// Zero for the fluid, whose particle update is not an interaction count.
inline double gflops(SolverType solver, int num_particles, double ms_per_step) {
    if (solver != SolverType::NBody) {
        return 0.0;
    }
    return interactions_per_second(solver, num_particles, ms_per_step) * FLOPS_PER_INTERACTION /
           1.0e9;
}

// Compulsory global memory traffic for one step, counted from the algorithm
// rather than measured.
//
// This is the traffic the step cannot avoid: what it must read from and write
// to global memory at least once. It deliberately excludes the tile reads,
// which are the whole point of the tiled kernel and which the caches and the
// shared memory staging are there to absorb. Quoted as a model, and the
// profiler numbers in the debug report are what check it.
inline double model_bytes_per_step(SolverType solver, NBodyVariant variant, int num_particles) {
    const double n = static_cast<double>(num_particles);
    if (solver == SolverType::Sph) {
        // cell, sort, bounds, gather, density and force passes, counted as the
        // arrays each one touches once.
        return n * 4.0 * 44.0;
    }
    if (variant == NBodyVariant::Fused) {
        // Reads x, y, z, mass, and the three velocities; writes three
        // positions, three velocities and three accelerations.
        return n * 4.0 * (7.0 + 9.0);
    }
    // Each array counts once per kernel. The force sum's repeated source reads
    // are left out: both paths share them and they dominate, so this prices the
    // difference between the paths and not the traffic of a step.
    // Split path: the force kernel reads four arrays and writes three, the
    // drift kernel reads nine and writes six, the closing kick reads six and
    // writes three.
    return n * 4.0 * ((4.0 + 3.0) + (9.0 + 6.0) + (6.0 + 3.0));
}

inline double model_gb_s(double bytes_per_step, double ms_per_step) {
    return ms_per_step > 0.0 ? bytes_per_step / (ms_per_step * 1.0e-3) / 1.0e9 : 0.0;
}

inline double model_bandwidth_fraction(double gb_s, double peak_bandwidth_gb_s) {
    return peak_bandwidth_gb_s > 0.0 ? gb_s / peak_bandwidth_gb_s : 0.0;
}

// The sample statistics of a run of trials. Each throws std::invalid_argument on
// an empty sample, which only a bug in the driver could produce.

// The value at a fraction in [0, 1] of the way through an already sorted sample,
// interpolating linearly between neighbours. gen_report_assets.py uses the same
// definition, so a statistic means the same thing in the CSV and in the report.
inline double quantile_of_sorted(const std::vector<double>& sorted, double fraction) {
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t below = static_cast<std::size_t>(std::floor(position));
    const std::size_t above = std::min(below + 1, sorted.size() - 1);
    return sorted[below] + (position - static_cast<double>(below)) * (sorted[above] - sorted[below]);
}

inline std::vector<double> sorted_sample(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("statistic of an empty sample");
    }
    std::sort(values.begin(), values.end());
    return values;
}

inline double median(const std::vector<double>& values) {
    return quantile_of_sorted(sorted_sample(values), 0.5);
}

inline double interquartile_range(const std::vector<double>& values) {
    const std::vector<double> sorted = sorted_sample(values);
    return quantile_of_sorted(sorted, 0.75) - quantile_of_sorted(sorted, 0.25);
}

inline double mean(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("mean of an empty sample");
    }
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

// The sample standard deviation, dividing by n - 1. A single value has none, and
// reports zero.
inline double standard_deviation(const std::vector<double>& values) {
    const double centre = mean(values);
    if (values.size() < 2) {
        return 0.0;
    }
    double sum = 0.0;
    for (double value : values) {
        sum += (value - centre) * (value - centre);
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

}  // namespace cps::bench
