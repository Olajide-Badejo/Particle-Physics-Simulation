#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "cps/core/bench_rows.hpp"

namespace {

using cps::NBodyVariant;
using cps::SolverType;
namespace bench = cps::bench;

// The worked examples use 1000 particles at 2 ms per step, so a step is 0.002 s
// and the working stays short.

TEST(BenchRow, StepsPerSecond) {
    // 1000 ms / 2 ms = 500 steps per second.
    EXPECT_DOUBLE_EQ(bench::steps_per_second(2.0), 500.0);
    EXPECT_DOUBLE_EQ(bench::steps_per_second(0.0), 0.0);
}

TEST(BenchRow, InteractionsPerSecond) {
    // N-body: 1000^2 = 1e6 interactions / 0.002 s = 5e8 per second.
    EXPECT_DOUBLE_EQ(bench::interactions_per_second(SolverType::NBody, 1000, 2.0), 5.0e8);
    // Fluid: 1000 particles / 0.002 s = 5e5 per second.
    EXPECT_DOUBLE_EQ(bench::interactions_per_second(SolverType::Sph, 1000, 2.0), 5.0e5);
    EXPECT_DOUBLE_EQ(bench::interactions_per_second(SolverType::NBody, 1000, 0.0), 0.0);
}

TEST(BenchRow, Gflops) {
    // 5e8 interactions per second * 20 flops = 1e10 flops per second = 10 GFLOP/s.
    EXPECT_DOUBLE_EQ(bench::gflops(SolverType::NBody, 1000, 2.0), 10.0);
    // The fluid has no interaction count to convert.
    EXPECT_DOUBLE_EQ(bench::gflops(SolverType::Sph, 1000, 2.0), 0.0);
}

TEST(BenchRow, ModelBytesPerStep) {
    // Split path: 1000 * 4 bytes * ((4 + 3) + (9 + 6) + (6 + 3) = 31 arrays) = 124000.
    EXPECT_DOUBLE_EQ(bench::model_bytes_per_step(SolverType::NBody, NBodyVariant::Naive, 1000),
                     124000.0);
    EXPECT_DOUBLE_EQ(bench::model_bytes_per_step(SolverType::NBody, NBodyVariant::Tiled, 1000),
                     124000.0);
    // Fused: 1000 * 4 bytes * (7 + 9 = 16 arrays) = 64000.
    EXPECT_DOUBLE_EQ(bench::model_bytes_per_step(SolverType::NBody, NBodyVariant::Fused, 1000),
                     64000.0);
    // Fluid: 1000 * 4 bytes * 54 arrays = 216000, whatever the variant field holds.
    EXPECT_DOUBLE_EQ(bench::model_bytes_per_step(SolverType::Sph, NBodyVariant::Fused, 1000),
                     216000.0);
}

TEST(BenchRow, ModelGbPerSecond) {
    // 124000 bytes / 0.002 s = 6.2e7 bytes per second = 0.062 GB/s.
    EXPECT_DOUBLE_EQ(bench::model_gb_s(124000.0, 2.0), 0.062);
    EXPECT_DOUBLE_EQ(bench::model_gb_s(124000.0, 0.0), 0.0);
}

TEST(BenchRow, ModelBandwidthFraction) {
    // 0.062 GB/s of a 100 GB/s peak is 6.2e-4.
    EXPECT_DOUBLE_EQ(bench::model_bandwidth_fraction(0.062, 100.0), 6.2e-4);
    EXPECT_DOUBLE_EQ(bench::model_bandwidth_fraction(0.062, 0.0), 0.0);
}

// Quartiles interpolate linearly on the sorted sample: for n values the quantile
// q sits at position q (n - 1), counting from zero.

TEST(BenchRowStatistics, OddLengthMedianIsTheMiddleValue) {
    EXPECT_DOUBLE_EQ(bench::median({3.0, 1.0, 2.0}), 2.0);
}

TEST(BenchRowStatistics, EvenLengthMedianAveragesTheMiddleTwo) {
    // Sorted 1 2 3 4: position 1.5, halfway between 2 and 3.
    EXPECT_DOUBLE_EQ(bench::median({4.0, 1.0, 3.0, 2.0}), 2.5);
}

TEST(BenchRowStatistics, InterquartileRange) {
    // Sorted 1 2 3 4: Q1 at position 0.75 is 1.75, Q3 at position 2.25 is 3.25,
    // so the range is 1.5.
    EXPECT_DOUBLE_EQ(bench::interquartile_range({4.0, 1.0, 3.0, 2.0}), 1.5);
    // Sorted 1 2 3 4 5: Q1 at position 1 is 2, Q3 at position 3 is 4.
    EXPECT_DOUBLE_EQ(bench::interquartile_range({5.0, 1.0, 4.0, 2.0, 3.0}), 2.0);
}

TEST(BenchRowStatistics, MeanAndSampleStandardDeviation) {
    const std::vector<double> sample = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    // The sum is 40 over 8 values.
    EXPECT_DOUBLE_EQ(bench::mean(sample), 5.0);
    // Squared deviations 9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32, and sqrt(32 / 7) = 2.13808994.
    EXPECT_NEAR(bench::standard_deviation(sample), 2.13808994, 1.0e-8);
}

TEST(BenchRowStatistics, SingleValueHasNoSpread) {
    const std::vector<double> sample = {7.0};
    EXPECT_DOUBLE_EQ(bench::median(sample), 7.0);
    EXPECT_DOUBLE_EQ(bench::interquartile_range(sample), 0.0);
    EXPECT_DOUBLE_EQ(bench::standard_deviation(sample), 0.0);
}

TEST(BenchRowStatistics, EmptySampleThrows) {
    const std::vector<double> empty;
    EXPECT_THROW(bench::median(empty), std::invalid_argument);
    EXPECT_THROW(bench::mean(empty), std::invalid_argument);
    EXPECT_THROW(bench::standard_deviation(empty), std::invalid_argument);
}

// Cross-checks against rows the v1.0 driver wrote to experiments/results. The
// formulas moved out of the driver and have to give the same answers. The CSV
// holds six significant figures, so the comparison is to that precision. The
// peak bandwidth is the 672.048 GB/s in device.json.
constexpr double PEAK_GB_S = 672.048;
constexpr double SIX_FIGURES = 5.0e-6;

void expect_figures(double actual, double recorded) {
    EXPECT_NEAR(actual / recorded, 1.0, SIX_FIGURES) << actual << " against " << recorded;
}

TEST(BenchRow, ReproducesACommittedNaiveRow) {
    // benchmarks.csv, nbody naive at 10000 particles.
    const double ms = 0.482291;
    expect_figures(bench::steps_per_second(ms), 2073.44);
    expect_figures(bench::interactions_per_second(SolverType::NBody, 10000, ms), 2.07344e+11);
    expect_figures(bench::gflops(SolverType::NBody, 10000, ms), 4146.88);
    const double bytes = bench::model_bytes_per_step(SolverType::NBody, NBodyVariant::Naive, 10000);
    expect_figures(bytes, 1.24e+06);
    const double gb_s = bench::model_gb_s(bytes, ms);
    expect_figures(gb_s, 2.57106);
    expect_figures(bench::model_bandwidth_fraction(gb_s, PEAK_GB_S), 0.00382571);
}

TEST(BenchRow, ReproducesACommittedFusedRow) {
    // tile_sweep.csv, fused at 100000 particles and tile width 32.
    const double ms = 13.0958;
    expect_figures(bench::steps_per_second(ms), 76.3603);
    expect_figures(bench::gflops(SolverType::NBody, 100000, ms), 15272.1);
    const double bytes = bench::model_bytes_per_step(SolverType::NBody, NBodyVariant::Fused, 100000);
    expect_figures(bytes, 6.4e+06);
    const double gb_s = bench::model_gb_s(bytes, ms);
    expect_figures(gb_s, 0.488706);
    expect_figures(bench::model_bandwidth_fraction(gb_s, PEAK_GB_S), 0.000727189);
}

}  // namespace
